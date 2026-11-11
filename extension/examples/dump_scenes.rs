//! Dump the demo library's scenes as JPEG photos, for the local-model benchmark
//! (`extension/server/bench_model.py`): photo-like content the model can actually
//! read keywords off, instead of the synthetic gradients the benchmark made before.
//!
//! ```text
//! cargo run -p lightcraft-extension --example dump_scenes -- <dir> [long edge]
//! ```
//!
//! The images land in a gitignored target folder; nothing here is product code.

use std::path::PathBuf;

use lightcraft_codecs::{ChromaSubsampling, EncodeImage, EncodeMeta, encode_jpeg};

fn main() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let dir = args.next().map(PathBuf::from).unwrap_or_else(|| PathBuf::from("target/bench-photos"));
    let edge: usize = args.next().and_then(|a| a.parse().ok()).unwrap_or(4096);
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    for sc in lightcraft_scenes::demo_library() {
        let img = sc.render_fit_serial(edge).to_srgb8();
        let jpeg = encode_jpeg(&EncodeImage::rgba8(&img), 92, ChromaSubsampling::S420, &EncodeMeta::default()).map_err(|e| e.to_string())?;
        let name = format!("{:02}-{}.jpg", sc.id, sc.name.to_lowercase().replace(' ', "-"));
        let path = dir.join(name);
        std::fs::write(&path, &jpeg).map_err(|e| e.to_string())?;
        println!("{} {}×{}", path.display(), img.width, img.height);
    }
    Ok(())
}
