//! Opt-in GPU benchmark; writes no media or user metadata.
#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable)]

use std::sync::atomic::AtomicBool;
use std::time::Instant;

use lightcraft_extension::semantic_search::model::{self, Encoder};

fn main() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let mode = args.next().unwrap_or_default();
    let dir = args.next().map(std::path::PathBuf::from).unwrap_or_else(model::default_dir);
    let cancel = AtomicBool::new(false);
    if mode == "--download" {
        model::download(&dir, &cancel, &mut |_| {})?;
        println!("{}", dir.display());
        return Ok(());
    }
    let pool = rayon::ThreadPoolBuilder::new().num_threads(1).build().map_err(|e| e.to_string())?;
    pool.install(|| bench(&dir, &cancel, mode != "--model-only"))
}

fn bench(dir: &std::path::Path, cancel: &AtomicBool, scan: bool) -> Result<(), String> {
    let started = Instant::now();
    let model = Encoder::load(dir)?;
    let load_ms = started.elapsed().as_secs_f64() * 1000.;
    let descriptions = [
        "一只白色小狗在草地上奔跑，嘴里叼着红色飞盘，背景是树木。",
        "蓝色海水拍打沙滩，远处停着帆船，天空中有白云。",
        "一个穿红色外套的人站在石桥上，桥下有河水。",
        "一只猫趴在室内窗台上，旁边摆着绿色盆栽。",
        "夜晚的城市街道上有汽车，建筑的窗户亮着灯。",
        "木桌上放着一碗面条，旁边有筷子和一杯水。",
    ];
    let documents: Vec<_> = descriptions.iter().map(|text| model.encode(text, false, cancel)).collect::<Result<_, _>>()?;
    let cases = [("在户外跑动的宠物", 0), ("海边的船", 1), ("红衣人物站在桥上", 2), ("窗边的猫咪", 3), ("城市夜景", 4), ("桌上的食物", 5)];
    let mut times = Vec::new();
    let mut rankings = Vec::new();
    let mut correct = 0;
    for _ in 0..5 {
        for (query, expected) in cases {
            let at = Instant::now();
            let encoded = model.encode(query, true, cancel)?;
            times.push(at.elapsed().as_secs_f64() * 1000.);
            let mut scores: Vec<_> =
                documents.iter().enumerate().map(|(id, v)| (id, v.iter().zip(&encoded).map(|(a, b)| a * b).sum::<f32>())).collect();
            scores.sort_by(|a, b| b.1.total_cmp(&a.1));
            if scores.first().is_some_and(|s| s.0 == expected) {
                correct += 1;
            }
            if rankings.len() < cases.len() {
                rankings.push(serde_json::json!({"query":query,"expected":expected,"ranked":scores}));
            }
        }
    }
    times.sort_by(f64::total_cmp);
    let p50 = times.get(times.len() / 2).copied().unwrap_or_default();
    let p95 = times.get(times.len().saturating_mul(95) / 100).copied().unwrap_or_default();
    let query = model.encode("宠物在户外运动", true, cancel)?;
    let mut scans = Vec::new();
    for size in [10_000usize, 100_000] {
        if !scan {
            break;
        }
        // Repeated real embeddings measure scanning only, not retrieval quality at this scale.
        let values: Vec<_> = documents.iter().cycle().take(size).flat_map(|v| v.iter().copied()).collect();
        let mut elapsed = Vec::new();
        for _ in 0..5 {
            let at = Instant::now();
            let mut top = Vec::<f32>::new();
            for vector in values.chunks_exact(model::DIM) {
                let score: f32 = vector.iter().zip(&query).map(|(a, b)| a * b).sum();
                let at = top.partition_point(|v| *v > score);
                if at < 100 {
                    top.insert(at, score);
                    top.truncate(100);
                }
            }
            std::hint::black_box(top);
            elapsed.push(at.elapsed().as_secs_f64() * 1000.);
        }
        scans.push(serde_json::json!({"photos":size,"payload_bytes":values.len()*4,"scan_ms":elapsed}));
    }
    println!(
        "{}",
        serde_json::json!({"device":model.device(),"gpu_weight_bytes":model.weight_bytes(),"load_ms":load_ms,"query_p50_ms":p50,"query_p95_ms":p95,"top1_correct":correct,"queries":times.len(),"rankings":rankings,"scan":scans})
    );
    Ok(())
}
