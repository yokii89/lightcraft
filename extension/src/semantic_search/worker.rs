//! One lazily started GPU inference worker. Its encoder and in-memory index expire after 60 idle seconds.
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, SyncSender, sync_channel};
use std::sync::{Arc, Mutex, OnceLock, PoisonError};
use std::time::{Duration, Instant};

use lightcraft_engine::activity::TaskGuard;

use super::lifecycle::View;
use super::model::{self, Encoder};
use super::store::{Description, Descriptions, Index, MAX_PHOTOS, PhotoText, Vector};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Library {
    pub generation: u64,
    pub dir: Option<PathBuf>,
}

pub enum Operation {
    Save(Vec<Description>),
    Reconcile,
    Status,
    Download,
    Index(Vec<Arc<lightcraft_catalog::Photo>>),
    Search { photos: Vec<Arc<lightcraft_catalog::Photo>>, query: String, minimum: f32 },
}

pub struct Request {
    pub library: Library,
    pub view: View,
    pub operation: Operation,
    pub cancel: Arc<AtomicBool>,
    pub guard: TaskGuard,
    pub reply: SyncSender<Result<Reply, String>>,
}

#[derive(Default)]
pub struct Reply {
    pub descriptions: usize,
    pub indexed: usize,
    pub built: usize,
    pub skipped: usize,
    pub hits: Vec<Hit>,
    pub warning: Option<String>,
    pub elapsed_ms: u128,
}

#[derive(Clone, Debug)]
pub struct Hit {
    pub id: u64,
    pub text: String,
    pub semantic: Option<f32>,
}

static WORKER: OnceLock<Mutex<Option<SyncSender<Request>>>> = OnceLock::new();

pub fn submit(view: View, operation: Operation, cancel: Arc<AtomicBool>, guard: TaskGuard) -> Result<Receiver<Result<Reply, String>>, String> {
    let mut service = WORKER.get_or_init(|| Mutex::new(None)).lock().unwrap_or_else(PoisonError::into_inner);
    let (tx, rx) = sync_channel(1);
    let request = Request { library: view.library.clone(), view, operation, cancel, guard, reply: tx };
    if service.is_none() {
        let (send, receive) = sync_channel(4);
        std::thread::Builder::new().name("semantic-search".into()).spawn(move || run(receive)).map_err(|e| e.to_string())?;
        *service = Some(send);
    }
    let Some(service) = service.as_ref() else { return Err("The search worker is unavailable.".into()) };
    service.try_send(request).map_err(|_| "The search worker is busy. Please try again.".to_string())?;
    Ok(rx)
}

fn run(receiver: Receiver<Request>) {
    let mut cache = Cache::default();
    let pool = rayon::ThreadPoolBuilder::new().num_threads(1).build();
    loop {
        let request = match receiver.recv_timeout(Duration::from_secs(60)) {
            Ok(request) => request,
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                cache.encoder = None;
                if !cache.index_dirty && cache.library.as_ref().is_some_and(|l| l.dir.is_some()) {
                    cache.index = None;
                }
                continue;
            }
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => return,
        };
        let started = Instant::now();
        // Last-resort guard for an escaped dependency panic. Report it, release the model, and
        // leave the atomic files intact so a later request can recover.
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| match &pool {
            Ok(pool) => pool.install(|| cache.execute(&request)),
            Err(error) => Err(error.to_string()),
        }))
        .unwrap_or_else(|_| {
            cache.encoder = None;
            Err("The search worker failed safely. Please try again.".into())
        });
        let result = result.map(|mut reply| {
            reply.elapsed_ms = started.elapsed().as_millis();
            reply
        });
        let _ = request.reply.send(result);
    }
}

#[derive(Default)]
struct Cache {
    library: Option<Library>,
    descriptions: Descriptions,
    descriptions_loaded: bool,
    descriptions_dirty: bool,
    index: Option<Index>,
    index_dirty: bool,
    encoder: Option<Encoder>,
}

impl Cache {
    fn encoder(&mut self) -> Result<&Encoder, String> {
        if self.encoder.is_none() {
            self.encoder = Some(Encoder::load(&model::default_dir())?);
        }
        self.encoder.as_ref().ok_or_else(|| "The text model is unavailable.".into())
    }

    fn switch(&mut self, library: &Library) -> Result<(), String> {
        if self.library.as_ref() == Some(library) {
            return Ok(());
        }
        self.descriptions = Descriptions::default();
        self.descriptions_loaded = false;
        self.descriptions_dirty = false;
        self.index = None;
        self.index_dirty = false;
        self.library = Some(library.clone());
        Ok(())
    }

    fn load_descriptions(&mut self, dir: Option<&std::path::Path>) -> Result<(), String> {
        if !self.descriptions_loaded {
            self.descriptions = Descriptions::load(dir)?;
            self.descriptions_loaded = true;
        }
        Ok(())
    }

    fn flush_descriptions(&mut self, view: &View) -> Result<(), String> {
        let dir = view.library.dir.as_deref();
        self.load_descriptions(dir)?;
        let retained = view.retained_ids()?;
        let before = self.descriptions.photos.len();
        self.descriptions.photos.retain(|id, _| retained.contains(id));
        self.descriptions_dirty |= before != self.descriptions.photos.len();
        if self.descriptions_dirty {
            self.descriptions.save(dir)?;
            self.descriptions_dirty = false;
        }
        Ok(())
    }

    fn flush_index(&mut self, view: &View) -> Result<(), String> {
        let dir = view.library.dir.as_deref();
        if self.index.is_none() {
            self.index = Some(Index::load(dir)?);
        }
        let retained = view.retained_ids()?;
        if let Some(index) = &mut self.index {
            let before = index.records.len();
            index.records.retain(|id, _| retained.contains(id));
            self.index_dirty |= before != index.records.len();
            if self.index_dirty {
                index.save(dir)?;
                self.index_dirty = false;
            }
        }
        Ok(())
    }

    fn reconcile(&mut self, view: &View) -> Result<(), String> {
        // These are independent atomic replacements: one failed/corrupt file must not block
        // reclamation of the other. Dirty flags survive failed writes for the next retry.
        let descriptions = self.flush_descriptions(view);
        let vectors = self.flush_index(view);
        let errors: Vec<_> = [descriptions.err(), vectors.err()].into_iter().flatten().collect();
        if errors.is_empty() { Ok(()) } else { Err(errors.join("; ")) }
    }

    fn execute(&mut self, request: &Request) -> Result<Reply, String> {
        if request.cancel.load(Ordering::Relaxed) {
            return Err("Cancelled".into());
        }
        request.view.ensure_active()?;
        self.switch(&request.library)?;
        let dir = request.library.dir.as_deref();
        if !matches!(request.operation, Operation::Reconcile | Operation::Download) {
            self.load_descriptions(dir)?;
        }
        let mut result = Reply { descriptions: self.descriptions.photos.len(), ..Reply::default() };
        match &request.operation {
            Operation::Save(entries) => {
                let entries: Vec<_> = entries.iter().filter(|entry| request.view.accepts(entry)).cloned().collect();
                if !entries.is_empty() {
                    self.descriptions.accept(entries)?;
                    self.descriptions_dirty = true;
                }
                self.flush_descriptions(&request.view)?;
                result.descriptions = self.descriptions.photos.len();
            }
            Operation::Reconcile => {
                self.reconcile(&request.view)?;
                result.descriptions = self.descriptions.photos.len();
            }
            Operation::Download => {
                model::download(&model::default_dir(), &request.cancel, &mut |p| request.guard.handle().progress(p.done, p.total))?;
            }
            Operation::Status => {
                if self.index.is_none() {
                    match Index::load(dir) {
                        Ok(index) => self.index = Some(index),
                        Err(error) => result.warning = Some(error),
                    }
                }
            }
            Operation::Index(photos) => {
                if photos.len() > MAX_PHOTOS {
                    return Err("This search index supports at most 100,000 photos.".into());
                }
                if self.index.is_none() {
                    self.index = Some(Index::load(dir).unwrap_or_default());
                }
                let photos = prepare(&request.view.active_photos(photos), &request.cancel)?;
                let mut index = self.index.take().unwrap_or_default();
                let built = self.build(&photos, &mut index, request, &mut result);
                // Keep completed rows on cancellation or one failed photo. Never replace the
                // previous file with a partial write, including on a full disk.
                result.indexed = index.records.len();
                self.index = Some(index);
                self.index_dirty = true;
                self.flush_index(&request.view)?;
                built?;
            }
            Operation::Search { photos, query, minimum } => {
                if photos.len() > MAX_PHOTOS || query.len() > 4096 || query.trim().is_empty() || !minimum.is_finite() {
                    return Err("Enter a shorter, non-empty search text.".into());
                }
                if self.index.is_none() {
                    match Index::load(dir) {
                        Ok(index) => self.index = Some(index),
                        Err(error) => result.warning = Some(error),
                    }
                }
                let has_vectors = self.index.as_ref().is_some_and(|i| !i.records.is_empty());
                let vector = if has_vectors {
                    match self.encoder().and_then(|m| m.encode(query, true, &request.cancel)) {
                        Ok(vector) => Some(vector),
                        Err(error) => {
                            result.warning = Some(error);
                            None
                        }
                    }
                } else {
                    result.warning = Some("Build the search index to enable semantic matching.".into());
                    None
                };
                request.view.ensure_active()?;
                let photos = prepare(&request.view.active_photos(photos), &request.cancel)?;
                result.hits = rank(&photos, &self.descriptions, self.index.as_ref(), vector.as_deref(), query, *minimum, &request.cancel)?;
                result.hits.retain(|hit| request.view.current(hit.id).is_some_and(|p| p.in_library()));
            }
        }
        result.indexed = self.index.as_ref().map_or(0, |index| index.records.len());
        Ok(result)
    }

    fn build(&mut self, photos: &[PhotoText], index: &mut Index, request: &Request, result: &mut Reply) -> Result<(), String> {
        for (number, photo) in photos.iter().enumerate() {
            request.view.ensure_active()?;
            if request.cancel.load(Ordering::Relaxed) {
                result.warning = Some("Indexing stopped. Completed photos were kept.".into());
                break;
            }
            let text = photo.text(&self.descriptions);
            if !request.view.valid_vector(photo, &text, &self.descriptions) {
                result.skipped += 1;
                continue;
            }
            let hash = photo.fingerprint(&text);
            request.guard.handle().progress(number as u64, photos.len() as u64);
            if text.is_empty() {
                index.records.remove(&photo.id);
                result.skipped += 1;
                continue;
            }
            if index.records.get(&photo.id).is_some_and(|v| v.fingerprint == hash) {
                continue;
            }
            let encoded = self.encoder()?.encode(&text, false, &request.cancel);
            match encoded {
                Ok(values) => {
                    // Encoding can outlive a delete, replacement or metadata edit. Revalidate
                    // against the observed catalog after inference, before committing a row.
                    if self.record_vector(&request.view, photo, &text, values, index) {
                        result.built += 1;
                    } else {
                        result.skipped += 1;
                    }
                }
                Err(error) => {
                    result.skipped += 1;
                    result.warning = Some(error);
                    index.records.remove(&photo.id);
                }
            }
            // Yield between GPU jobs so background indexing leaves time for interactive renders.
            std::thread::sleep(Duration::from_millis(10));
        }
        request.guard.handle().progress(photos.len() as u64, photos.len() as u64);
        Ok(())
    }

    fn record_vector(&self, view: &View, photo: &PhotoText, text: &str, values: Vec<f32>, index: &mut Index) -> bool {
        if !view.valid_vector(photo, text, &self.descriptions) {
            return false;
        }
        index.records.insert(photo.id, Vector { fingerprint: photo.fingerprint(text), values });
        true
    }
}

#[cfg(test)]
#[path = "worker_tests.rs"]
mod tests;

const TOP: usize = 100;

fn prepare(photos: &[Arc<lightcraft_catalog::Photo>], cancel: &AtomicBool) -> Result<Vec<PhotoText>, String> {
    let mut prepared = Vec::with_capacity(photos.len());
    let mut text_bytes = 0usize;
    for photo in photos {
        if cancel.load(Ordering::Relaxed) {
            return Err("Cancelled".into());
        }
        let photo = PhotoText::from_photo(photo);
        text_bytes = text_bytes.saturating_add(photo.metadata.len()).saturating_add(photo.name.len());
        for keyword in &photo.keywords {
            text_bytes = text_bytes.saturating_add(keyword.len());
        }
        if text_bytes > 64 << 20 {
            return Err("The search text exceeds the 64 MB memory limit.".into());
        }
        prepared.push(photo);
    }
    Ok(prepared)
}

fn push_top(list: &mut Vec<(u64, f32)>, item: (u64, f32)) {
    if !item.1.is_finite() {
        return;
    }
    let at = list.partition_point(|v| v.1 > item.1 || (v.1 == item.1 && v.0 < item.0));
    if at < TOP {
        list.insert(at, item);
        list.truncate(TOP);
    }
}

fn for_each_term(text: &str, mut visit: impl FnMut(&str)) -> usize {
    let mut start = None;
    let mut count = 0usize;
    for (offset, ch) in text.char_indices() {
        if ch.is_alphanumeric() {
            start.get_or_insert(offset);
        } else if let Some(begin) = start.take()
            && let Some(word) = text.get(begin..offset)
        {
            count = count.saturating_add(visit_word(word, &mut visit));
        }
    }
    if let Some(begin) = start
        && let Some(word) = text.get(begin..)
    {
        count = count.saturating_add(visit_word(word, &mut visit));
    }
    count
}

fn visit_word(word: &str, visit: &mut impl FnMut(&str)) -> usize {
    let word = word.to_lowercase();
    if word.is_ascii() {
        visit(&word);
        return 1;
    }
    let mut chars = word.char_indices();
    let Some((mut start, _)) = chars.next() else { return 0 };
    let Some((next, ch)) = chars.next() else {
        visit(&word);
        return 1;
    };
    let mut count = 1usize;
    if let Some(pair) = word.get(start..next.saturating_add(ch.len_utf8())) {
        visit(pair);
    }
    start = next;
    for (next, ch) in chars {
        if let Some(pair) = word.get(start..next.saturating_add(ch.len_utf8())) {
            visit(pair);
        }
        start = next;
        count = count.saturating_add(1);
    }
    count
}

fn bm25_score(tf: u32, dl: usize, df: usize, documents: usize, average_dl: f64) -> f64 {
    const K1: f64 = 1.2;
    const B: f64 = 0.75;
    if tf == 0 || df == 0 || documents == 0 || average_dl <= 0.0 {
        return 0.0;
    }
    let idf = (1.0 + (documents as f64 - df as f64 + 0.5) / (df as f64 + 0.5)).ln();
    let tf = f64::from(tf);
    let length_norm = K1 * (1.0 - B + B * dl as f64 / average_dl);
    idf * tf * (K1 + 1.0) / (tf + length_norm)
}

fn lexical_rank(photos: &[PhotoText], descriptions: &Descriptions, query: &str, cancel: &AtomicBool) -> Result<Vec<(u64, f32)>, String> {
    #[derive(Default)]
    struct Term {
        documents: usize,
        seen_document: Option<usize>,
        frequency: u32,
    }
    let mut query_terms = BTreeMap::new();
    for_each_term(query, |term| {
        query_terms.entry(term.to_string()).or_insert_with(Term::default);
    });
    if query_terms.is_empty() {
        return Ok(Vec::new());
    }
    let mut lengths = Vec::with_capacity(photos.len());
    let mut total_length = 0u64;
    let mut document_count = 0usize;
    for (doc_id, photo) in photos.iter().enumerate() {
        if cancel.load(Ordering::Relaxed) {
            return Err("Cancelled".into());
        }
        let text = photo.text(descriptions);
        let mut length = 0usize;
        let mut observe = |term: &str| {
            if let Some(stats) = query_terms.get_mut(term)
                && stats.seen_document != Some(doc_id)
            {
                stats.seen_document = Some(doc_id);
                stats.documents = stats.documents.saturating_add(1);
            }
        };
        length = length.saturating_add(for_each_term(&photo.name, &mut observe));
        length = length.saturating_add(for_each_term(&text, &mut observe));
        if length > 0 {
            total_length = total_length.saturating_add(length as u64);
            document_count = document_count.saturating_add(1);
        }
        lengths.push(length);
    }
    if document_count == 0 {
        return Ok(Vec::new());
    }
    let average_length = total_length as f64 / document_count as f64;
    let mut ranked = Vec::new();
    for (photo, length) in photos.iter().zip(lengths) {
        if cancel.load(Ordering::Relaxed) {
            return Err("Cancelled".into());
        }
        let text = photo.text(descriptions);
        let mut count = |term: &str| {
            if let Some(stats) = query_terms.get_mut(term) {
                stats.frequency = stats.frequency.saturating_add(1);
            }
        };
        for_each_term(&photo.name, &mut count);
        for_each_term(&text, &mut count);
        // Ordered query terms keep floating-point accumulation and ranking deterministic.
        let score: f64 = query_terms
            .values_mut()
            .map(|stats| {
                let tf = std::mem::take(&mut stats.frequency);
                bm25_score(tf, length, stats.documents, document_count, average_length)
            })
            .sum();
        if score.is_finite() && score > 0.0 {
            push_top(&mut ranked, (photo.id, score as f32));
        }
    }
    Ok(ranked)
}

pub fn rank(
    photos: &[PhotoText],
    descriptions: &Descriptions,
    index: Option<&Index>,
    vector: Option<&[f32]>,
    query: &str,
    minimum: f32,
    cancel: &AtomicBool,
) -> Result<Vec<Hit>, String> {
    let lexical = lexical_rank(photos, descriptions, query, cancel)?;
    let mut semantic = Vec::new();
    for photo in photos {
        if cancel.load(Ordering::Relaxed) {
            return Err("Cancelled".into());
        }
        let text = photo.text(descriptions);
        if let (Some(query), Some(record)) = (vector, index.and_then(|i| i.records.get(&photo.id)))
            && record.fingerprint == photo.fingerprint(&text)
            && query.len() == record.values.len()
        {
            let similarity: f32 = query.iter().zip(&record.values).map(|(a, b)| a * b).sum();
            if similarity >= minimum {
                push_top(&mut semantic, (photo.id, similarity));
            }
        }
    }
    let mut fused = std::collections::BTreeMap::<u64, f32>::new();
    for list in [&lexical, &semantic] {
        for (rank, (id, _)) in list.iter().enumerate() {
            *fused.entry(*id).or_default() += 1.0 / (60.0 + rank as f32 + 1.0);
        }
    }
    let mut ranked = Vec::new();
    for entry in fused {
        push_top(&mut ranked, entry);
    }
    let positions: std::collections::BTreeMap<_, _> = photos.iter().map(|p| (p.id, p)).collect();
    Ok(ranked
        .into_iter()
        .filter_map(|(id, _)| {
            positions.get(&id).map(|p| Hit {
                id,
                text: p.text(descriptions),
                semantic: semantic.iter().find(|(p, _)| *p == id).map(|(_, value)| *value),
            })
        })
        .collect())
}
