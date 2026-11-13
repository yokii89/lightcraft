//! Live catalog observations shared with the serial worker; never infer deletion from the disk.
use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::AtomicBool;
use std::sync::mpsc::{Receiver, TryRecvError};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::{Duration, Instant};

use lightcraft_catalog::{Catalog, Photo};
use lightcraft_engine::activity::Cancel;
use lightcraft_ui_egui::LightcraftApp;

use super::store::{self, Description, MAX_PHOTOS};
use super::worker::{self, Library, Operation, Reply};

#[derive(Clone)]
pub(super) struct View {
    pub library: Library,
    observed: Arc<Mutex<Observed>>,
}

struct Observed {
    active: bool,
    revision: u64,
    photos: BTreeMap<u64, Arc<Photo>>,
}

fn retained(photo: &Photo) -> bool {
    // Recently Deleted is recoverable, including photos deleted from the Local browser.
    (!photo.local || photo.deleted) && !photo.is_video()
}

impl View {
    pub fn new(library: Library, catalog: &Catalog) -> Result<Self, String> {
        let mut photos = BTreeMap::new();
        for photo in catalog.photos().filter(|p| retained(p)) {
            if photos.len() >= MAX_PHOTOS {
                return Err("This search index supports at most 100,000 photos.".into());
            }
            photos.insert(photo.id.0, photo.clone());
        }
        Ok(Self { library, observed: Arc::new(Mutex::new(Observed { active: true, revision: catalog.revision, photos })) })
    }

    pub fn refresh(&self, catalog: &Catalog) -> Result<bool, String> {
        let revision = self.observed.lock().unwrap_or_else(PoisonError::into_inner).revision;
        if revision == catalog.revision {
            return Ok(false);
        }
        let Some(ids) = catalog.touched_since(revision) else {
            let replacement = Self::new(self.library.clone(), catalog)?;
            let mut current = self.observed.lock().unwrap_or_else(PoisonError::into_inner);
            current.photos = std::mem::take(&mut replacement.observed.lock().unwrap_or_else(PoisonError::into_inner).photos);
            current.revision = catalog.revision;
            return Ok(true);
        };
        let mut current = self.observed.lock().unwrap_or_else(PoisonError::into_inner);
        // Validate the final size first: an oversized catalog must never publish a partial view.
        let additions = ids.iter().filter(|id| !current.photos.contains_key(&id.0) && catalog.photo(**id).is_some_and(|p| retained(p))).count();
        let removals = ids.iter().filter(|id| current.photos.contains_key(&id.0) && !catalog.photo(**id).is_some_and(|p| retained(p))).count();
        if current.photos.len().saturating_sub(removals).saturating_add(additions) > MAX_PHOTOS {
            return Err("This search index supports at most 100,000 photos.".into());
        }
        for id in ids {
            if let Some(photo) = catalog.photo(id).filter(|p| retained(p)) {
                current.photos.insert(id.0, photo.clone());
            } else {
                current.photos.remove(&id.0);
            }
        }
        current.revision = catalog.revision;
        Ok(true)
    }

    pub fn invalidate(&self) {
        self.observed.lock().unwrap_or_else(PoisonError::into_inner).active = false;
    }

    pub fn ensure_active(&self) -> Result<(), String> {
        if self.observed.lock().unwrap_or_else(PoisonError::into_inner).active { Ok(()) } else { Err("Cancelled".into()) }
    }

    pub fn current(&self, id: u64) -> Option<Arc<Photo>> {
        let current = self.observed.lock().unwrap_or_else(PoisonError::into_inner);
        current.active.then(|| current.photos.get(&id).cloned()).flatten()
    }

    pub fn retained_ids(&self) -> Result<BTreeSet<u64>, String> {
        let current = self.observed.lock().unwrap_or_else(PoisonError::into_inner);
        if !current.active {
            return Err("Cancelled".into());
        }
        Ok(current.photos.keys().copied().collect())
    }

    pub fn accepts(&self, entry: &Description) -> bool {
        self.current(entry.id).is_some_and(|p| {
            p.in_library()
                && store::basis(&p) == entry.basis
                && entry.guard_keywords.iter().all(|k| p.meta.keywords.iter().any(|v| lightcraft_catalog::keywords::same(k, v)))
        })
    }

    pub fn active_photos(&self, photos: &[Arc<Photo>]) -> Vec<Arc<Photo>> {
        photos.iter().filter_map(|p| self.current(p.id.0).filter(|p| p.in_library())).collect()
    }

    pub fn valid_vector(&self, photo: &store::PhotoText, text: &str, descriptions: &store::Descriptions) -> bool {
        self.current(photo.id).filter(|p| p.in_library()).is_some_and(|p| {
            let current = store::PhotoText::from_photo(&p);
            current.fingerprint(&current.text(descriptions)) == photo.fingerprint(text)
        })
    }
}

#[derive(Default)]
pub(super) struct Tracker {
    view: Option<View>,
    pending: Option<(Library, Receiver<Result<Reply, String>>)>,
    dirty: bool,
    due: Option<Instant>,
    error: Option<String>,
}

impl Tracker {
    pub fn observe(&mut self, library: Library, catalog: &Catalog) -> Result<View, String> {
        if self.view.as_ref().is_none_or(|v| v.library != library) {
            if let Some(view) = &self.view {
                view.invalidate();
            }
            self.view = Some(View::new(library, catalog)?);
            self.dirty = true;
            self.due = None;
        } else if let Some(view) = &self.view {
            match view.refresh(catalog) {
                Ok(true) => {
                    self.dirty = true;
                    self.due.get_or_insert_with(|| Instant::now() + Duration::from_millis(500));
                }
                Ok(false) => {}
                Err(error) => {
                    // Fail closed: never clean or write against a partial/oversized catalog.
                    view.invalidate();
                    self.view = None;
                    return Err(error);
                }
            }
        }
        let view = self.view.as_ref().ok_or("The search worker is unavailable.")?;
        Ok(view.clone())
    }

    pub fn poll(&mut self, app: &LightcraftApp, view: View) -> (bool, Option<String>) {
        let mut notice = None;
        if let Some((library, rx)) = &self.pending {
            let finished = match rx.try_recv() {
                Ok(result) => Some(result.map(|_| ())),
                Err(TryRecvError::Disconnected) => Some(Err("The search worker is unavailable.".into())),
                Err(TryRecvError::Empty) => None,
            };
            if let Some(result) = finished {
                if *library == view.library {
                    match result {
                        Ok(()) => self.error = None,
                        Err(error) => notice = self.failed(error),
                    }
                }
                self.pending = None;
            }
        }
        if self.dirty && self.pending.is_none() && self.due.is_none_or(|due| Instant::now() >= due) {
            let guard = app.session.activity.start("semanticMaintenance", "Saving photo descriptions", Cancel::No);
            match worker::submit(view.clone(), Operation::Reconcile, Arc::new(AtomicBool::new(false)), guard) {
                Ok(rx) => {
                    self.pending = Some((view.library, rx));
                    self.dirty = false;
                    self.due = None;
                }
                Err(error) => notice = self.failed(error),
            }
        }
        (self.dirty || self.pending.is_some(), notice)
    }

    pub fn failed(&mut self, error: String) -> Option<String> {
        self.dirty = true;
        self.due = Some(Instant::now() + Duration::from_secs(5));
        let changed = self.error.as_ref() != Some(&error);
        self.error = Some(error.clone());
        changed.then_some(error)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lightcraft_catalog::Op;
    use lightcraft_engine::Session;

    #[test]
    fn opening_observes_once_changes_coalesce_and_retry_needs_no_new_event() {
        let mut session = Session::with_demo();
        let key = Library { generation: session.library_generation(), dir: None };
        let mut tracker = Tracker::default();
        let view = tracker.observe(key.clone(), &session.catalog).unwrap();
        assert!(tracker.dirty && tracker.due.is_none(), "opening schedules an immediate reconciliation");
        tracker.dirty = false;
        tracker.observe(key.clone(), &session.catalog).unwrap();
        assert!(!tracker.dirty, "an unchanged catalog never schedules another disk write");
        let id = session.catalog.photos().next().unwrap().id;
        session.catalog.apply(Op::SetDeleted { id, deleted: true }).unwrap();
        tracker.observe(key.clone(), &session.catalog).unwrap();
        let due = tracker.due;
        assert!(tracker.dirty && due.is_some());
        assert!(!view.current(id.0).unwrap().in_library());
        session.catalog.apply(Op::SetDeleted { id, deleted: false }).unwrap();
        tracker.observe(key.clone(), &session.catalog).unwrap();
        assert_eq!(tracker.due, due, "a burst of changes shares one bounded debounce window");
        assert!(view.current(id.0).unwrap().in_library());
        tracker.dirty = false;
        assert_eq!(tracker.failed("write failed".into()), Some("write failed".into()));
        assert!(tracker.dirty && tracker.due.is_some_and(|due| due > Instant::now()));
        assert!(tracker.failed("write failed".into()).is_none(), "do not repeat the same failure toast");
        let other = Library { generation: key.generation + 1, dir: None };
        tracker.observe(other, &session.catalog).unwrap();
        assert!(view.ensure_active().is_err(), "old-library requests become unusable on a switch");
        assert!(tracker.dirty && tracker.due.is_none());
    }
}
