//! Library-local descriptions and a rebuildable, bounded FP32 vector cache.
use std::collections::BTreeMap;
use std::io::Read;
use std::path::Path;

use lightcraft_catalog::{Photo, safe_file::write_atomic, safe_file::write_atomic_with};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::model::{DIM, ENCODING};

pub const MAX_PHOTOS: usize = 100_000;
const MAX_TEXT_BYTES: u64 = 64 << 20;
const MAGIC: &[u8; 8] = b"LCSTV001";
const RECORD_BYTES: usize = 8 + 32 + DIM * 4;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Description {
    pub id: u64,
    pub basis: [u8; 32],
    pub description: String,
    /// Accepted full keyword paths. Undoing/removing them deactivates this cached description.
    pub guard_keywords: Vec<String>,
}

impl Description {
    fn valid(&self) -> bool {
        self.basis != [0; 32]
            && !self.description.trim().is_empty()
            && self.description.chars().count() <= 256
            && !self.description.chars().any(char::is_control)
            && !self.guard_keywords.is_empty()
            && self.guard_keywords.len() <= 16
            && self.guard_keywords.iter().all(|k| !k.is_empty() && k.len() <= 1024)
    }
}

#[derive(Default, Serialize, Deserialize)]
pub struct Descriptions {
    version: u32,
    pub photos: BTreeMap<u64, Description>,
}

pub(crate) fn bounded_read(path: &Path, max: u64) -> Result<Vec<u8>, String> {
    let file = std::fs::File::open(path).map_err(|e| e.to_string())?;
    if !file.metadata().is_ok_and(|m| m.is_file() && m.len() <= max) {
        return Err("File exceeds the size limit".into());
    }
    let mut data = Vec::new();
    file.take(max.saturating_add(1)).read_to_end(&mut data).map_err(|e| e.to_string())?;
    if data.len() as u64 > max {
        return Err("File exceeds the size limit".into());
    }
    Ok(data)
}

impl Descriptions {
    pub fn load(dir: Option<&Path>) -> Result<Self, String> {
        let Some(path) = dir.map(|d| d.join("semantic-descriptions.json")) else { return Ok(Self::default()) };
        if !path.exists() {
            return Ok(Self::default());
        }
        let result: Self = serde_json::from_slice(&bounded_read(&path, MAX_TEXT_BYTES)?).map_err(|e| e.to_string())?;
        if result.version != 1 || result.photos.len() > MAX_PHOTOS || result.photos.iter().any(|(id, d)| *id != d.id || !d.valid()) {
            return Err("Invalid photo description cache; the existing file was preserved.".into());
        }
        Ok(result)
    }

    pub fn save(&self, dir: Option<&Path>) -> Result<(), String> {
        let Some(dir) = dir else { return Ok(()) };
        let data = serde_json::to_vec(self).map_err(|e| e.to_string())?;
        if data.len() as u64 > MAX_TEXT_BYTES {
            return Err("The description cache is full.".into());
        }
        write_atomic(&dir.join("semantic-descriptions.json"), &data).map_err(|e| e.to_string())
    }

    pub fn accept(&mut self, entries: Vec<Description>) -> Result<(), String> {
        if entries.len() > MAX_PHOTOS || entries.iter().any(|d| !d.valid()) {
            return Err("Invalid photo description".into());
        }
        let new =
            entries.iter().map(|d| d.id).collect::<std::collections::BTreeSet<_>>().into_iter().filter(|id| !self.photos.contains_key(id)).count();
        if self.photos.len().saturating_add(new) > MAX_PHOTOS {
            return Err("The description cache is full.".into());
        }
        self.version = 1;
        for entry in entries {
            self.photos.insert(entry.id, entry);
        }
        Ok(())
    }
}

#[derive(Clone)]
pub struct PhotoText {
    pub id: u64,
    pub basis: [u8; 32],
    pub name: String,
    pub metadata: String,
    pub keywords: Vec<String>,
}

pub fn basis(photo: &Photo) -> [u8; 32] {
    struct HashWriter(Sha256);
    impl std::io::Write for HashWriter {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            self.0.update(bytes);
            Ok(bytes.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    // Stream large edit settings into the hash, without allocating another serialized copy.
    // The all-zero sentinel is never accepted as a description basis.
    let mut writer = HashWriter(Sha256::new());
    if serde_json::to_writer(&mut writer, &(&photo.source, &photo.content_hash, photo.file_size, photo.width, photo.height, &photo.develop)).is_err()
    {
        return [0; 32];
    }
    writer.0.finalize().into()
}

impl PhotoText {
    pub fn from_photo(photo: &Photo) -> Self {
        let fields = [&photo.meta.title, &photo.meta.caption, &photo.meta.alt_text, &photo.meta.extended_description];
        let metadata = if fields.iter().fold(0usize, |sum, s| sum.saturating_add(s.len())) > 16_000 {
            String::new()
        } else {
            fields.into_iter().map(String::as_str).filter(|s| !s.trim().is_empty()).collect::<Vec<_>>().join("；")
        };
        let keywords = if photo.meta.keywords.len() > 1024 || photo.meta.keywords.iter().fold(0usize, |sum, k| sum.saturating_add(k.len())) > 16_000 {
            Vec::new()
        } else {
            photo.meta.keywords.clone()
        };
        Self { id: photo.id.0, basis: basis(photo), name: photo.file_name.chars().take(512).collect(), metadata, keywords }
    }

    pub fn text(&self, descriptions: &Descriptions) -> String {
        let caption = descriptions
            .photos
            .get(&self.id)
            .filter(|entry| {
                self.basis != [0; 32]
                    && entry.basis == self.basis
                    && entry.guard_keywords.iter().all(|k| self.keywords.iter().any(|v| lightcraft_catalog::keywords::same(k, v)))
            })
            .map_or(self.metadata.as_str(), |entry| entry.description.as_str());
        if caption.len() > 16_384 || self.keywords.len() > 1024 {
            return String::new();
        }
        let keywords = self.keywords.join("，");
        if keywords.len() > 16_384 {
            return String::new();
        }
        format!("{caption} {keywords}").trim().to_string()
    }

    pub fn fingerprint(&self, text: &str) -> [u8; 32] {
        let mut hash = Sha256::new();
        hash.update(self.basis);
        hash.update(text.as_bytes());
        hash.finalize().into()
    }
}

pub struct Vector {
    pub fingerprint: [u8; 32],
    pub values: Vec<f32>,
}

#[derive(Default)]
pub struct Index {
    pub records: BTreeMap<u64, Vector>,
}

impl Index {
    pub fn load(dir: Option<&Path>) -> Result<Self, String> {
        let Some(path) = dir.map(|d| d.join("semantic-vectors.bin")) else { return Ok(Self::default()) };
        if !path.exists() {
            return Ok(Self::default());
        }
        // Stream records: loading a 100k-photo index must not briefly keep a second 200 MB
        // copy of the file next to the decoded vectors.
        let file = std::fs::File::open(&path).map_err(|e| e.to_string())?;
        let metadata = file.metadata().map_err(|e| e.to_string())?;
        let length = metadata.len();
        if !metadata.is_file() || length > (48 + MAX_PHOTOS * RECORD_BYTES) as u64 {
            return Err("File exceeds the size limit".into());
        }
        let mut reader = std::io::BufReader::new(file);
        let mut magic = [0; 8];
        let mut version = [0; 32];
        let mut count = [0; 8];
        reader.read_exact(&mut magic).map_err(|e| e.to_string())?;
        reader.read_exact(&mut version).map_err(|e| e.to_string())?;
        reader.read_exact(&mut count).map_err(|e| e.to_string())?;
        let count = u64::from_le_bytes(count);
        if &magic != MAGIC
            || version != <[u8; 32]>::from(Sha256::digest(ENCODING.as_bytes()))
            || count > MAX_PHOTOS as u64
            || length.saturating_sub(48) != count.saturating_mul(RECORD_BYTES as u64)
        {
            return Err("The vector cache is damaged or belongs to a different model. Rebuild it.".into());
        }
        let mut records = BTreeMap::new();
        for _ in 0..count {
            let mut id = [0; 8];
            let mut fingerprint = [0; 32];
            reader.read_exact(&mut id).map_err(|e| e.to_string())?;
            reader.read_exact(&mut fingerprint).map_err(|e| e.to_string())?;
            let mut values = Vec::with_capacity(DIM);
            for _ in 0..DIM {
                let mut value = [0; 4];
                reader.read_exact(&mut value).map_err(|e| e.to_string())?;
                values.push(f32::from_le_bytes(value));
            }
            validate(&values)?;
            if records.insert(u64::from_le_bytes(id), Vector { fingerprint, values }).is_some() {
                return Err("Duplicate photo in vector cache".into());
            }
        }
        let mut extra = [0];
        if reader.read(&mut extra).map_err(|e| e.to_string())? != 0 {
            return Err("The vector cache changed during loading.".into());
        }
        Ok(Self { records })
    }

    pub fn save(&self, dir: Option<&Path>) -> Result<(), String> {
        if self.records.len() > MAX_PHOTOS {
            return Err("The vector cache is full.".into());
        }
        for record in self.records.values() {
            validate(&record.values)?;
        }
        let Some(dir) = dir else { return Ok(()) };
        write_atomic_with(&dir.join("semantic-vectors.bin"), &mut |writer| {
            writer.write_all(MAGIC)?;
            writer.write_all(&Sha256::digest(ENCODING.as_bytes()))?;
            writer.write_all(&(self.records.len() as u64).to_le_bytes())?;
            for (id, record) in &self.records {
                writer.write_all(&id.to_le_bytes())?;
                writer.write_all(&record.fingerprint)?;
                for value in &record.values {
                    writer.write_all(&value.to_le_bytes())?;
                }
            }
            Ok(())
        })
        .map_err(|e| e.to_string())?;
        Ok(())
    }
}

fn validate(values: &[f32]) -> Result<(), String> {
    let norm: f64 = values.iter().map(|v| f64::from(*v).powi(2)).sum();
    if values.len() != DIM || values.iter().any(|v| !v.is_finite()) || (norm - 1.0).abs() > 0.01 {
        return Err("Invalid embedding in vector cache".into());
    }
    Ok(())
}
