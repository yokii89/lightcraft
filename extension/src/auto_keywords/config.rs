//! Settings of the Auto Keywords page: what the page shows and what is saved as
//! `<library>/auto-keywords.json` between sessions.
//!
//! The file is user data read back from disk, so it is held to [`MAX_FILE_BYTES`], every field
//! carries a serde default, and anything unreadable falls back to the defaults: a damaged or
//! truncated file must never stop the page from opening. Version 1 files (endpoint/protocol of
//! the retired Ollama backend) read fine: their unknown fields are ignored, the new fields take
//! their defaults.

use std::path::Path;

use serde::{Deserialize, Serialize};

use lightcraft_catalog::safe_file::write_atomic;

/// What happens with the generated keywords: shown for confirmation first, or written at once.
/// Both keep the whole batch a single undo step.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum ApplyMode {
    #[default]
    Review,
    Direct,
}

/// The vision model the self-hosted service loads (§6 of the design): a Hugging Face
/// repository id. Apache-2.0, conversational image-text-to-text. The plain Instruct VLM
/// follows the JSON keyword prompt; a grounding/segmentation model (WeVisOne-VL) answers
/// in `<ref>`/`<box>` markup instead, which is why this is not one.
pub const MODEL_DEFAULT: &str = "Qwen/Qwen3-VL-2B-Instruct";

/// The seconds the model service waits for a new call before it shuts itself down (§6.6):
/// the "quick release" half of lazy start, so a forgotten server never keeps the machine busy.
pub const IDLE_SECS: u64 = 300;

/// The interpreter the service runs with, when the settings name one; empty finds `python`,
/// `python3` or `py` on the PATH (§6.2).
pub const PYTHON_DEFAULT: &str = "";

/// The most keywords asked for per photo (the model is asked for 2 up to this many).
pub const MAX_KEYWORDS_DEFAULT: u32 = 8;
pub const MAX_KEYWORDS_MIN: u32 = 1;
pub const MAX_KEYWORDS_MAX: u32 = 16;

/// The seconds a request may stall without receiving a byte before it is given up on.
pub const TIMEOUT_DEFAULT: u32 = 120;
pub const TIMEOUT_MIN: u32 = 5;
pub const TIMEOUT_MAX: u32 = 600;

/// The largest settings file read: a file of this size is far past anything these few fields make.
const MAX_FILE_BYTES: u64 = 64 << 10; // 64 KiB

/// The page's settings.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct Settings {
    /// Of this file format.
    pub version: u32,
    /// The Hugging Face repository the service loads.
    pub model_repo: String,
    /// A Python interpreter to run the service with; empty finds one on the PATH.
    pub python_path: String,
    /// The most keywords per photo.
    pub max_keywords: u32,
    /// Every generated keyword is saved as `{parent}|{category}|{keyword}`; empty writes
    /// `{category}|{keyword}`.
    pub parent: String,
    /// Leave out photos that already carry keywords.
    pub skip_tagged: bool,
    pub apply_mode: ApplyMode,
    /// The per-request read-stall timeout in seconds.
    pub timeout_secs: u32,
}

impl Default for Settings {
    fn default() -> Self {
        Settings {
            version: 2,
            model_repo: MODEL_DEFAULT.to_string(),
            python_path: PYTHON_DEFAULT.to_string(),
            max_keywords: MAX_KEYWORDS_DEFAULT,
            parent: PARENT_DEFAULT.to_string(),
            skip_tagged: false,
            apply_mode: ApplyMode::Review,
            timeout_secs: TIMEOUT_DEFAULT,
        }
    }
}

/// The default parent keyword the generated keywords go under (`{parent}|{category}|{keyword}`).
pub const PARENT_DEFAULT: &str = "AI";

impl Settings {
    /// The keyword count asked for per photo, held to the page's range.
    pub fn max_keywords(&self) -> usize {
        self.max_keywords.clamp(MAX_KEYWORDS_MIN, MAX_KEYWORDS_MAX) as usize
    }

    /// The read-stall timeout, held to the page's range.
    pub fn timeout(&self) -> u32 {
        self.timeout_secs.clamp(TIMEOUT_MIN, TIMEOUT_MAX)
    }

    /// The model repository, trimmed; falls back to [`MODEL_DEFAULT`] when the file held nothing.
    pub fn model_repo(&self) -> &str {
        match self.model_repo.trim() {
            "" => MODEL_DEFAULT,
            repo => repo,
        }
    }
}

/// The settings file's place next to the library. `None`: no library on disk — the settings then
/// live in memory for this session only.
pub fn settings_path(library_dir: Option<&Path>) -> Option<std::path::PathBuf> {
    library_dir.map(|dir| dir.join("auto-keywords.json"))
}

/// Read the settings file. Anything unreadable — missing, oversized, damaged JSON, wrong types —
/// gives the defaults: the page must open and run whatever the file holds.
pub fn read(path: &Path) -> Settings {
    let Ok(bytes) = std::fs::File::open(path).and_then(|f| {
        use std::io::Read;
        let mut bytes = Vec::new();
        f.take(MAX_FILE_BYTES + 1).read_to_end(&mut bytes).map(|_| bytes)
    }) else {
        return Settings::default();
    };
    if bytes.len() as u64 > MAX_FILE_BYTES {
        return Settings::default();
    }
    serde_json::from_slice(&bytes).unwrap_or_default()
}

/// Write the settings file atomically. A failure is only reported: the page keeps working, and
/// the next change or run tries again.
pub fn write(path: &Path, settings: &Settings) -> Result<(), String> {
    let mut bytes = serde_json::to_vec_pretty(settings).map_err(|e| e.to_string())?;
    bytes.push(b'\n');
    write_atomic(path, &bytes).map_err(|e| format!("{}: {e}", path.display()))
}
