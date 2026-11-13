//! BGE's complete transformer on a hardware GPU, using the app's pure Rust wgpu backend.
//! Candle is used only to deserialize checked FP32 weights. No CPU inference fallback.
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Duration;

use candle_nn::VarBuilder;
use wgpu::util::DeviceExt;

use super::model::{DIM, MAX_TOKENS};

const TIMEOUT: Duration = Duration::from_secs(15);
const KERNELS: [&str; 6] = ["embed", "norm", "dense", "scores", "softmax", "context"];

#[derive(Clone, Copy)]
struct Affine {
    weight: u32,
    bias: u32,
    input: u32,
    output: u32,
}

struct Layer {
    query: Affine,
    key: Affine,
    value: Affine,
    output: Affine,
    attention_norm: Affine,
    intermediate: Affine,
    feed_forward: Affine,
    output_norm: Affine,
}

#[derive(Default)]
struct Weights {
    values: Vec<f32>,
}

impl Weights {
    fn tensor(&mut self, vb: &VarBuilder<'_>, name: &str, shape: &[usize]) -> Result<u32, String> {
        let offset = u32::try_from(self.values.len()).map_err(|e| e.to_string())?;
        let values = vb.get(shape, name).and_then(|t| t.flatten_all()?.to_vec1::<f32>()).map_err(|e| e.to_string())?;
        if values.iter().any(|v| !v.is_finite()) || self.values.len().saturating_add(values.len()) > 24_000_000 {
            return Err("Invalid text model weights".into());
        }
        self.values.extend(values);
        Ok(offset)
    }

    fn affine(&mut self, vb: VarBuilder<'_>, input: u32, output: u32, norm: bool) -> Result<Affine, String> {
        let shape = if norm { vec![output as usize] } else { vec![output as usize, input as usize] };
        Ok(Affine { weight: self.tensor(&vb, "weight", &shape)?, bias: self.tensor(&vb, "bias", &[output as usize])?, input, output })
    }

    fn layer(&mut self, vb: VarBuilder<'_>) -> Result<Layer, String> {
        Ok(Layer {
            query: self.affine(vb.pp("attention.self.query"), 512, 512, false)?,
            key: self.affine(vb.pp("attention.self.key"), 512, 512, false)?,
            value: self.affine(vb.pp("attention.self.value"), 512, 512, false)?,
            output: self.affine(vb.pp("attention.output.dense"), 512, 512, false)?,
            attention_norm: self.affine(vb.pp("attention.output.LayerNorm"), 512, 512, true)?,
            intermediate: self.affine(vb.pp("intermediate.dense"), 512, 2048, false)?,
            feed_forward: self.affine(vb.pp("output.dense"), 2048, 512, false)?,
            output_norm: self.affine(vb.pp("output.LayerNorm"), 512, 512, true)?,
        })
    }
}

pub(super) struct Encoder {
    device: wgpu::Device,
    queue: wgpu::Queue,
    layout: wgpu::BindGroupLayout,
    kernels: BTreeMap<&'static str, wgpu::ComputePipeline>,
    weights: wgpu::Buffer,
    embeddings: [u32; 3],
    norm: Affine,
    layers: Vec<Layer>,
    adapter: String,
    bytes: u64,
    failure: Arc<Mutex<Option<String>>>,
    serial: Mutex<()>,
}

struct Call<'a> {
    kernel: &'static str,
    inputs: [&'a wgpu::Buffer; 2],
    output: &'a wgpu::Buffer,
    params: [u32; 8],
    groups: [u32; 3],
}

impl Encoder {
    pub(super) fn load(vb: VarBuilder<'_>) -> Result<Self, String> {
        let mut weights = Weights::default();
        let embeddings = [
            weights.tensor(&vb, "embeddings.word_embeddings.weight", &[21128, DIM])?,
            weights.tensor(&vb, "embeddings.position_embeddings.weight", &[512, DIM])?,
            weights.tensor(&vb, "embeddings.token_type_embeddings.weight", &[2, DIM])?,
        ];
        let norm = weights.affine(vb.pp("embeddings.LayerNorm"), 512, 512, true)?;
        let layers = (0..4).map(|i| weights.layer(vb.pp(format!("encoder.layer.{i}")))).collect::<Result<Vec<_>, _>>()?;
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| Self::create(weights, embeddings, norm, layers)))
            .unwrap_or_else(|_| Err("Text GPU initialization failed safely. Check the graphics driver.".into()))
    }

    fn create(weights: Weights, embeddings: [u32; 3], norm: Affine, layers: Vec<Layer>) -> Result<Self, String> {
        let backends = lightcraft_gpu::backend::compute_backends().ok_or("Text embeddings require a GPU; GPU compute is disabled.")?;
        let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
        desc.backends = backends;
        desc.backend_options = lightcraft_gpu::backend::backend_options();
        desc.flags = lightcraft_gpu::backend::instance_flags_from_env(desc.flags);
        let instance = wgpu::Instance::new(desc);
        let adapter = lightcraft_gpu::backend::pick_adapter(&instance, backends)?;
        let info = adapter.get_info();
        if !matches!(info.device_type, wgpu::DeviceType::DiscreteGpu | wgpu::DeviceType::IntegratedGpu) {
            return Err(format!("Text embeddings require a hardware GPU; {} is {:?}.", info.name, info.device_type));
        }
        let limits = adapter.limits();
        let bytes = weights.values.len() as u64 * 4;
        if bytes > limits.max_storage_buffer_binding_size
            || bytes > limits.max_buffer_size
            || limits.max_storage_buffers_per_shader_stage < 4
            || limits.max_compute_invocations_per_workgroup < 256
            || limits.max_compute_workgroup_size_x < 64
            || limits.max_compute_workgroup_size_y < 16
            || limits.max_compute_workgroup_storage_size < 2304
        {
            return Err(format!("{} cannot run the text embedding kernels within its GPU limits.", info.name));
        }
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("lightcraft-text-embeddings"),
            required_limits: limits,
            ..Default::default()
        }))
        .map_err(|e| e.to_string())?;
        let failure = Arc::new(Mutex::new(None));
        let errors = failure.clone();
        device.on_uncaptured_error(Arc::new(move |e| {
            *errors.lock().unwrap_or_else(PoisonError::into_inner) = Some(e.to_string());
        }));
        let lost = failure.clone();
        device.set_device_lost_callback(move |reason, msg| {
            *lost.lock().unwrap_or_else(PoisonError::into_inner) = Some(format!("GPU device lost ({reason:?}): {msg}"));
        });
        let scopes = [
            device.push_error_scope(wgpu::ErrorFilter::OutOfMemory),
            device.push_error_scope(wgpu::ErrorFilter::Validation),
            device.push_error_scope(wgpu::ErrorFilter::Internal),
        ];
        let mut entries = Vec::new();
        for binding in 0..5 {
            let ty = if binding == 4 { wgpu::BufferBindingType::Uniform } else { wgpu::BufferBindingType::Storage { read_only: binding != 3 } };
            entries.push(wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer { ty, has_dynamic_offset: false, min_binding_size: None },
                count: None,
            });
        }
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("BGE"), entries: &entries });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("BGE"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("BGE"),
            source: wgpu::ShaderSource::Wgsl(include_str!("bert.wgsl").into()),
        });
        let mut kernels = BTreeMap::new();
        for entry in KERNELS {
            let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some(entry),
                compilation_options: wgpu::PipelineCompilationOptions { zero_initialize_workgroup_memory: false, ..Default::default() },
                cache: None,
            });
            kernels.insert(entry, pipeline);
        }
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("BGE weights"),
            contents: bytemuck::cast_slice(&weights.values),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let mut initialization_error = None;
        for scope in scopes.into_iter().rev() {
            if let Some(error) = pollster::block_on(scope.pop()) {
                initialization_error = Some(format!("Text GPU initialization failed: {error}"));
            }
        }
        if let Some(error) = initialization_error {
            return Err(error);
        }
        Ok(Self {
            device,
            queue,
            layout,
            kernels,
            weights: buffer,
            embeddings,
            norm,
            layers,
            adapter: format!("{} ({:?})", info.name, info.backend),
            bytes,
            failure,
            serial: Mutex::new(()),
        })
    }

    pub(super) fn adapter(&self) -> &str {
        &self.adapter
    }
    pub(super) fn weight_bytes(&self) -> u64 {
        self.bytes
    }

    fn buffer(&self, label: &str, floats: u32) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: u64::from(floats) * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        })
    }

    fn dispatch(&self, encoder: &mut wgpu::CommandEncoder, call: Call<'_>) -> Result<(), String> {
        let params = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&call.params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let buffers = [&self.weights, call.inputs[0], call.inputs[1], call.output, &params];
        let entries: Vec<_> = buffers
            .iter()
            .enumerate()
            .map(|(binding, b)| wgpu::BindGroupEntry { binding: binding as u32, resource: b.as_entire_binding() })
            .collect();
        let bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &self.layout, entries: &entries });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some(call.kernel), timestamp_writes: None });
        pass.set_pipeline(self.kernels.get(call.kernel).ok_or("Missing BGE GPU kernel")?);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups(call.groups[0], call.groups[1], call.groups[2]);
        Ok(())
    }

    fn affine(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        inputs: [&wgpu::Buffer; 2],
        output: &wgpu::Buffer,
        affine: Affine,
        shape: (u32, bool, bool),
    ) -> Result<(), String> {
        let (n, norm, mode) = shape;
        self.dispatch(
            encoder,
            Call {
                kernel: if norm { "norm" } else { "dense" },
                inputs,
                output,
                params: [n, affine.input, affine.output, affine.weight, affine.bias, 0, u32::from(mode), 0],
                groups: if norm { [n, 1, 1] } else { [affine.output.div_ceil(16), n.div_ceil(16), 1] },
            },
        )
    }

    pub(super) fn encode(&self, ids: &[u32], types: &[u32], cancel: &AtomicBool) -> Result<Vec<f32>, String> {
        if ids.is_empty() || ids.len() > MAX_TOKENS || ids.len() != types.len() || ids.iter().any(|id| *id >= 21128) || types.iter().any(|t| *t > 1) {
            return Err("Invalid text model token input".into());
        }
        let _serial = self.serial.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(error) = self.failure.lock().unwrap_or_else(PoisonError::into_inner).clone() {
            return Err(error);
        }
        let scopes = [
            self.device.push_error_scope(wgpu::ErrorFilter::OutOfMemory),
            self.device.push_error_scope(wgpu::ErrorFilter::Validation),
            self.device.push_error_scope(wgpu::ErrorFilter::Internal),
        ];
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| self.forward(ids, types, cancel)))
            .unwrap_or_else(|_| Err("Text GPU inference failed safely. Check the graphics driver.".into()));
        let mut failure = None;
        for scope in scopes.into_iter().rev() {
            if let Some(error) = pollster::block_on(scope.pop()) {
                failure = Some(format!("Text GPU inference failed: {error}"));
            }
        }
        match failure {
            Some(error) => Err(error),
            None => result,
        }
    }

    fn forward(&self, ids: &[u32], types: &[u32], cancel: &AtomicBool) -> Result<Vec<f32>, String> {
        let n = u32::try_from(ids.len()).map_err(|e| e.to_string())?;
        let tokens = |values: &[u32]| {
            self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("BGE tokens"),
                contents: bytemuck::cast_slice(&values.iter().map(|v| *v as f32).collect::<Vec<_>>()),
                usage: wgpu::BufferUsages::STORAGE,
            })
        };
        let ids = tokens(ids);
        let types = tokens(types);
        let hidden = self.buffer("hidden", n * 512);
        let temporary = self.buffer("temporary", n * 512);
        let query = self.buffer("query", n * 512);
        let key = self.buffer("key", n * 512);
        let value = self.buffer("value", n * 512);
        let scores = self.buffer("attention scores", 8 * n * n);
        let probabilities = self.buffer("attention probabilities", 8 * n * n);
        let context = self.buffer("context", n * 512);
        let residual = self.buffer("residual", n * 512);
        let intermediate = self.buffer("intermediate", n * 2048);
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("BGE encode") });
        let [words, positions, token_types] = self.embeddings;
        self.dispatch(
            &mut encoder,
            Call {
                kernel: "embed",
                inputs: [&ids, &types],
                output: &temporary,
                params: [n, 0, 0, words, positions, token_types, 0, 0],
                groups: [(n * 512).div_ceil(64), 1, 1],
            },
        )?;
        self.affine(&mut encoder, [&temporary, &temporary], &hidden, self.norm, (n, true, false))?;
        for layer in &self.layers {
            if cancel.load(Ordering::Relaxed) {
                return Err("Cancelled".into());
            }
            for (affine, output) in [(layer.query, &query), (layer.key, &key), (layer.value, &value)] {
                self.affine(&mut encoder, [&hidden, &hidden], output, affine, (n, false, false))?;
            }
            self.dispatch(
                &mut encoder,
                Call {
                    kernel: "scores",
                    inputs: [&query, &key],
                    output: &scores,
                    params: [n, 0, 0, 0, 0, 0, 0, 0],
                    groups: [(8 * n * n).div_ceil(64), 1, 1],
                },
            )?;
            self.dispatch(
                &mut encoder,
                Call {
                    kernel: "softmax",
                    inputs: [&scores, &scores],
                    output: &probabilities,
                    params: [n, 0, 0, 0, 0, 0, 0, 0],
                    groups: [8 * n, 1, 1],
                },
            )?;
            self.dispatch(
                &mut encoder,
                Call {
                    kernel: "context",
                    inputs: [&probabilities, &value],
                    output: &context,
                    params: [n, 0, 0, 0, 0, 0, 0, 0],
                    groups: [(n * 512).div_ceil(64), 1, 1],
                },
            )?;
            self.affine(&mut encoder, [&context, &context], &temporary, layer.output, (n, false, false))?;
            self.affine(&mut encoder, [&temporary, &hidden], &residual, layer.attention_norm, (n, true, true))?;
            self.affine(&mut encoder, [&residual, &residual], &intermediate, layer.intermediate, (n, false, true))?;
            self.affine(&mut encoder, [&intermediate, &intermediate], &temporary, layer.feed_forward, (n, false, false))?;
            self.affine(&mut encoder, [&temporary, &residual], &hidden, layer.output_norm, (n, true, true))?;
        }
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("CLS readback"),
            size: DIM as u64 * 4,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(&hidden, 0, &staging, 0, DIM as u64 * 4);
        let submission = self.queue.submit([encoder.finish()]);
        let (tx, rx) = std::sync::mpsc::channel();
        staging.slice(..).map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        // wgpu can panic while polling a lost device; turn that dependency failure into an error.
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.device.poll(wgpu::PollType::Wait { submission_index: Some(submission), timeout: Some(TIMEOUT) })
        }))
        .map_err(|_| "Text GPU device failed while waiting".to_string())?
        .map_err(|e| e.to_string())?;
        rx.recv_timeout(TIMEOUT).map_err(|e| e.to_string())?.map_err(|e| e.to_string())?;
        let mapped = staging.slice(..).get_mapped_range().map_err(|e| e.to_string())?;
        let mut vector = bytemuck::try_cast_slice::<u8, f32>(&mapped).map_err(|e| e.to_string())?.to_vec();
        drop(mapped);
        staging.unmap();
        if cancel.load(Ordering::Relaxed) {
            return Err("Cancelled".into());
        }
        super::model::normalize(&mut vector)?;
        Ok(vector)
    }
}
