// Original FP32 BERT inference kernels. Shapes are checked by gpu.rs, capped at 256 tokens.
struct Params {
    n: u32, input_dim: u32, output_dim: u32, weight: u32,
    bias: u32, extra: u32, mode: u32, unused: u32,
}
@group(0) @binding(0) var<storage, read> weights: array<f32>;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> dst: array<f32>;
@group(0) @binding(4) var<uniform> p: Params;

@compute @workgroup_size(64)
fn embed(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= p.n * 512u { return; }
    let row = i / 512u;
    let col = i % 512u;
    dst[i] = weights[p.weight + u32(a[row]) * 512u + col]
        + weights[p.bias + row * 512u + col]
        + weights[p.extra + u32(b[row]) * 512u + col];
}

var<workgroup> reduction: array<f32, 64>;
@compute @workgroup_size(64)
fn norm(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    let row = group.x;
    var sum = 0.0;
    for (var col = lane; col < 512u; col += 64u) {
        let i = row * 512u + col;
        sum += a[i] + select(0.0, b[i], p.mode != 0u);
    }
    reduction[lane] = sum;
    workgroupBarrier();
    for (var stride = 32u; stride > 0u; stride /= 2u) {
        if lane < stride { reduction[lane] += reduction[lane + stride]; }
        workgroupBarrier();
    }
    let mean = reduction[0] / 512.0;
    // Every lane must finish reading the mean before the shared storage is reused.
    workgroupBarrier();
    var variance = 0.0;
    for (var col = lane; col < 512u; col += 64u) {
        let i = row * 512u + col;
        let value = a[i] + select(0.0, b[i], p.mode != 0u) - mean;
        variance += value * value;
    }
    reduction[lane] = variance;
    workgroupBarrier();
    for (var stride = 32u; stride > 0u; stride /= 2u) {
        if lane < stride { reduction[lane] += reduction[lane + stride]; }
        workgroupBarrier();
    }
    let scale = inverseSqrt(reduction[0] / 512.0 + 1e-12);
    for (var col = lane; col < 512u; col += 64u) {
        let i = row * 512u + col;
        let value = a[i] + select(0.0, b[i], p.mode != 0u);
        dst[i] = (value - mean) * scale * weights[p.weight + col] + weights[p.bias + col];
    }
}

// Abramowitz-Stegun 7.1.26: absolute erf error < 1.5e-7; GELU matches the erf BERT activation.
fn gelu(value: f32) -> f32 {
    let x = value * 0.7071067811865476;
    let t = 1.0 / (1.0 + 0.3275911 * abs(x));
    let poly = (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t;
    let erf = sign(x) * (1.0 - poly * exp(-x * x));
    return 0.5 * value * (1.0 + erf);
}

var<workgroup> tile_a: array<f32, 256>;
var<workgroup> tile_w: array<f32, 256>;
@compute @workgroup_size(16, 16)
fn dense(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let col = gid.x;
    let row = gid.y;
    let local = lid.y * 16u + lid.x;
    var value = 0.0;
    for (var base = 0u; base < p.input_dim; base += 16u) {
        tile_a[local] = 0.0;
        tile_w[local] = 0.0;
        if row < p.n && base + lid.x < p.input_dim {
            tile_a[local] = a[row * p.input_dim + base + lid.x];
        }
        if col < p.output_dim && base + lid.y < p.input_dim {
            tile_w[local] = weights[p.weight + col * p.input_dim + base + lid.y];
        }
        workgroupBarrier();
        for (var k = 0u; k < 16u; k += 1u) {
            value += tile_a[lid.y * 16u + k] * tile_w[k * 16u + lid.x];
        }
        workgroupBarrier();
    }
    if row < p.n && col < p.output_dim {
        value += weights[p.bias + col];
        if p.mode != 0u { value = gelu(value); }
        dst[row * p.output_dim + col] = value;
    }
}

@compute @workgroup_size(64)
fn scores(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= 8u * p.n * p.n { return; }
    let head = i / (p.n * p.n);
    let row = (i / p.n) % p.n;
    let col = i % p.n;
    var value = 0.0;
    for (var k = 0u; k < 64u; k += 1u) {
        value += a[row * 512u + head * 64u + k] * b[col * 512u + head * 64u + k];
    }
    dst[i] = value * 0.125;
}

@compute @workgroup_size(64)
fn softmax(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    let base = group.x * p.n;
    var largest = -3.402823e38;
    for (var i = lane; i < p.n; i += 64u) { largest = max(largest, a[base + i]); }
    reduction[lane] = largest;
    workgroupBarrier();
    for (var stride = 32u; stride > 0u; stride /= 2u) {
        if lane < stride { reduction[lane] = max(reduction[lane], reduction[lane + stride]); }
        workgroupBarrier();
    }
    let maximum = reduction[0];
    workgroupBarrier();
    var sum = 0.0;
    for (var i = lane; i < p.n; i += 64u) { sum += exp(a[base + i] - maximum); }
    reduction[lane] = sum;
    workgroupBarrier();
    for (var stride = 32u; stride > 0u; stride /= 2u) {
        if lane < stride { reduction[lane] += reduction[lane + stride]; }
        workgroupBarrier();
    }
    for (var i = lane; i < p.n; i += 64u) { dst[base + i] = exp(a[base + i] - maximum) / reduction[0]; }
}

@compute @workgroup_size(64)
fn context(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= p.n * 512u { return; }
    let row = i / 512u;
    let col = i % 512u;
    let head = col / 64u;
    var value = 0.0;
    for (var k = 0u; k < p.n; k += 1u) {
        value += a[(head * p.n + row) * p.n + k] * b[k * 512u + col];
    }
    dst[i] = value;
}
