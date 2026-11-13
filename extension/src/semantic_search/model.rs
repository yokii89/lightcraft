//! The pinned four-layer BGE encoder on a hardware GPU. Inputs are bounded by MAX_TOKENS.
//! Only verified official weights load; no CPU inference fallback.
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};

use lightcraft_fetch::{FileSpec, Options, Progress};
#[cfg(not(target_arch = "wasm32"))]
use sha2::{Digest, Sha256};
use tokenizers::Tokenizer;

pub const REVISION: &str = "7999e1d3359715c523056ef9478215996d62a620";
pub const DIM: usize = 512;
pub const MAX_TOKENS: usize = 256;
pub const ENCODING: &str = "bge-small-zh-v1.5@7999e1d3/cls-l2/256/query-instruction/v1";
const WEIGHTS: FileSpec<'static> = FileSpec {
    name: "model.safetensors",
    size: Some(95_827_648),
    max: 95_827_648,
    sha256: Some("354763b9b1357bc9c44f62c6be2276321081ed2567773608c0d0785b61d5a026"),
};
const TOKENIZER: FileSpec<'static> = FileSpec {
    name: "tokenizer.json",
    size: Some(439_125),
    max: 439_125,
    sha256: Some("48cea5d44424912a6fd1ea647bf4fe50b55ab8b1e5879c3275f80e339e8fae26"),
};

pub fn default_dir() -> PathBuf {
    std::env::var_os("LIGHTCRAFT_TEXT_MODEL").map(PathBuf::from).unwrap_or_else(|| {
        lightcraft_engine::config::config_dir().unwrap_or_else(std::env::temp_dir).join("extension-models").join("bge-small-zh-v1.5").join(REVISION)
    })
}

pub fn download(dir: &Path, cancel: &AtomicBool, progress: &mut dyn FnMut(&Progress)) -> Result<(), String> {
    let base = format!("https://huggingface.co/BAAI/bge-small-zh-v1.5/resolve/{REVISION}");
    lightcraft_fetch::download(&[WEIGHTS, TOKENIZER], &[base], dir, &Options::default(), cancel, progress).map_err(|e| e.to_string())?;
    lightcraft_catalog::safe_file::write_atomic(&dir.join("LICENSE-MIT.txt"), include_bytes!("../../licenses/BGE-MIT.txt")).map_err(|e| e.to_string())
}

#[cfg(not(target_arch = "wasm32"))]
fn verified(dir: &Path, spec: &FileSpec<'_>) -> Result<Vec<u8>, String> {
    use std::io::Read;
    let path = dir.join(spec.name);
    let file = std::fs::File::open(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    if !file.metadata().is_ok_and(|m| m.is_file() && Some(m.len()) == spec.size) {
        return Err(format!("{}: wrong file size; download the model again", path.display()));
    }
    let mut bytes = Vec::new();
    file.take(spec.max.saturating_add(1)).read_to_end(&mut bytes).map_err(|e| e.to_string())?;
    if Some(bytes.len() as u64) != spec.size || spec.sha256 != Some(format!("{:x}", Sha256::digest(&bytes)).as_str()) {
        return Err(format!("{}: SHA-256 mismatch; download the model again", path.display()));
    }
    Ok(bytes)
}

pub struct Encoder {
    tokenizer: Tokenizer,
    #[cfg(not(target_arch = "wasm32"))]
    gpu: super::gpu::Encoder,
}

impl Encoder {
    #[cfg(not(target_arch = "wasm32"))]
    pub fn load(dir: &Path) -> Result<Self, String> {
        let mut tokenizer = Tokenizer::from_bytes(verified(dir, &TOKENIZER)?).map_err(|e| e.to_string())?;
        tokenizer.with_padding(None);
        tokenizer.with_truncation(None).map_err(|e| e.to_string())?;
        let bytes = verified(dir, &WEIGHTS)?;
        // CPU tensors only deserialize the checked checkpoint; all inference runs in WGSL.
        let vb =
            candle_nn::VarBuilder::from_slice_safetensors(&bytes, candle_core::DType::F32, &candle_core::Device::Cpu).map_err(|e| e.to_string())?;
        let gpu = super::gpu::Encoder::load(vb)?;
        Ok(Self { tokenizer, gpu })
    }

    #[cfg(target_arch = "wasm32")]
    pub fn load(_dir: &Path) -> Result<Self, String> {
        Err("GPU text embeddings are not available in the browser build.".into())
    }

    pub fn device(&self) -> &str {
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.gpu.adapter()
        }
        #[cfg(target_arch = "wasm32")]
        {
            "unavailable"
        }
    }

    pub fn weight_bytes(&self) -> u64 {
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.gpu.weight_bytes()
        }
        #[cfg(target_arch = "wasm32")]
        {
            0
        }
    }

    pub fn encode(&self, text: &str, query: bool, cancel: &AtomicBool) -> Result<Vec<f32>, String> {
        if text.trim().is_empty() || text.len() > 16_384 {
            return Err("Enter a shorter, non-empty search text.".into());
        }
        let text = if query { format!("为这个句子生成表示以用于检索相关文章：{text}") } else { text.to_string() };
        let encoding = self.tokenizer.encode(text.as_str(), true).map_err(|e| e.to_string())?;
        if encoding.len() > MAX_TOKENS {
            return Err("The text exceeds the 256-token limit. Please shorten it.".into());
        }
        if cancel.load(Ordering::Relaxed) {
            return Err("Cancelled".into());
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.gpu.encode(encoding.get_ids(), encoding.get_type_ids(), cancel)
        }
        #[cfg(target_arch = "wasm32")]
        {
            Err("GPU text embeddings are not available in the browser build.".into())
        }
    }
}

pub(crate) fn normalize(vector: &mut [f32]) -> Result<(), String> {
    if vector.len() != DIM || vector.iter().any(|v| !v.is_finite()) {
        return Err("Invalid embedding".into());
    }
    let norm = vector.iter().map(|v| f64::from(*v).powi(2)).sum::<f64>().sqrt();
    if !norm.is_finite() || norm < 1e-12 {
        return Err("Invalid embedding norm".into());
    }
    for v in vector {
        *v = (f64::from(*v) / norm) as f32;
    }
    Ok(())
}
