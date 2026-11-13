//! The Geographic classification feature (level 2 of the extensions hub, design:
//! `extension/docs/01-扩展两级导航与地理位置分类设计.md`, persistence and auto-sync:
//! `extension/docs/05-地图相册持久化与自动增量设计.md`): the library's photos are grouped on the
//! map album by where they were taken — country and city, looked up offline from each photo's GPS
//! with the GeoNames place names — and the place is written as a hierarchical keyword
//! `{parent}|{country}|{city}`, one undo step for the whole batch.
//!
//! The album is persistent and bound to the library (`<library>/geo-classify.json`, see
//! [`store`]): photos with GPS join it automatically and incrementally — the sync diffs the
//! catalog each frame within a small budget, so no user action adds a photo, and reopening the
//! page shows the album as it was. Writing the keywords stays a reviewed action: the pending
//! places (the album entries whose photo still lacks the current path keyword) are confirmed on
//! the review page, where groups can be unticked, before one `Op::Batch` commits.
//!
//! The lookup uses [`lightcraft_geocode`] directly, the same public crate the native
//! `metadata.reverseGeocode` command is built on, for three reasons: that command's enabled check
//! wants a selection (the whole-library album has no selection), it writes Location *metadata* —
//! this feature writes keywords — and its per-place details (Sublocation, nearest city) serve the
//! metadata fields, not the two-level keyword. The dataset itself (GeoNames cities15000,
//! CC-BY 4.0, ~3 MB) is downloaded by the native `metadata.downloadPlaceNames` command and shared
//! with it; the page shows the attribution wherever it shows the dataset.

use std::collections::{BTreeMap, HashSet};
use std::path::PathBuf;
use std::sync::mpsc::{Receiver, TryRecvError, channel};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock, PoisonError};
use std::time::{Duration, Instant, SystemTime};

use egui::{Align, RichText, Sense, Ui, vec2};
use lightcraft_catalog::{Aspects, Meta, Op, PhotoId};
use lightcraft_engine::Session;
use lightcraft_engine::catalog::keywords;
use lightcraft_engine::cmd::keywords as keywords_cmd;
use lightcraft_geocode::{GEONAMES_FILES, GeoNames, LatLon};
use lightcraft_meta::{Tracklog, parse_gpx};
use lightcraft_ui_egui::button::Button;
use lightcraft_ui_egui::icons::{self, Icon};
use lightcraft_ui_egui::pick::PickRequest;
use lightcraft_ui_egui::progress::spinner;
use lightcraft_ui_egui::text_field::TextField;
use lightcraft_ui_egui::theme::Tokens;
use lightcraft_ui_egui::{LightcraftApp, i18n};
use serde_json::Value;

use crate::ui::{
    Pill, action_bar, action_note, card, card_head, field, hint, path_example, primary_action, split_page, status_pill, success_card, trn, warning,
    with_alpha,
};

mod map;
mod store;

/// The feature's id in the hub.
pub(crate) const PAGE_ID: &str = "geo-classify";

/// The label of the write's single undo step (§8 of the Auto Keywords design; English, translated
/// from this source string).
const UNDO_LABEL: &str = "Geo Classify";

/// The parent keyword the place paths go under (`{parent}|{country}|{city}`).
const PARENT_DEFAULT: &str = "地点";

/// How often the place-names status is fetched again while a download runs (§: the download is a
/// background task; the page follows it).
const STATUS_POLL: Duration = Duration::from_millis(500);

/// How long the store save waits after the album changed, so a burst of syncs writes one file.
const SAVE_DEBOUNCE: Duration = Duration::from_millis(500);

/// How often a failed store save tries again (the failure is logged; the album stays in memory).
const SAVE_RETRY: Duration = Duration::from_secs(5);

/// Photos classified per frame at most: the album fills without holding a frame.
const SYNC_BUDGET: usize = 400;

/// The wall-clock part of the per-frame sync budget (a pathological reverse lookup cannot eat a
/// frame).
const SYNC_BUDGET_TIME: Duration = Duration::from_millis(4);

/// After a failed gazetteer load, the next try waits this long (a present-but-broken dataset must
/// not be re-parsed every frame).
const SYNC_RETRY: Duration = Duration::from_secs(2);

/// Selected GPX files are user input; keep both disk reads and route rendering bounded.
const MAX_TRACK_BYTES: u64 = 64 << 20;
const MAX_TRACKS: usize = 4;
const MAX_TRACK_POINTS: usize = 1_000_000;
const MAX_DRAWN_TRACK_POINTS: usize = 20_000;

/// The map's fixed height in the album card (the review's map fills the panel instead).
const ALBUM_MAP_H: f32 = 340.0;

/// What the album snapshot must rebuild on: photos added or removed, or their metadata (GPS,
/// keywords, name) or file changed. Develop edits and ratings do not rebuild it.
const SNAPSHOT_ASPECTS: Aspects = Aspects::PRESENCE.with(Aspects::META).with(Aspects::FILE);

// ---------------------------------------------------------------- state

/// Where the page is. `Applying` passes through in one frame, like on the Auto Keywords page.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum Phase {
    #[default]
    Idle,
    Review,
    Applying,
    Done,
}

/// One photo's looked-up place.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct PhotoPlace {
    id: PhotoId,
    name: String,
    country: String,
    city: String,
    /// ISO 3166-1 alpha-2 of the country ("CN" keeps the group on the map).
    iso: String,
    /// The photo's own GPS — the map's thumbnail position and its group's centroid.
    lat: f64,
    lon: f64,
    /// Metres from the photo's position to the place found.
    distance: f64,
}

impl PhotoPlace {
    /// The keyword path the photo's place writes: `{parent}|{country}|{city}`, or without the
    /// city when none was found (a photo far out of town classifies by country alone).
    fn path(&self, parent: &str) -> String {
        let mut path = if self.city.is_empty() { format!("{parent}|{}", self.country) } else { format!("{parent}|{}|{}", self.country, self.city) };
        path = keywords::clean(&path);
        path
    }
}

/// One group of the review: the photos a single keyword path would classify.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Group {
    /// The exact keyword path the group writes.
    pub path: String,
    pub country: String,
    pub city: String,
    /// ISO 3166-1 alpha-2 of the group's country (groups abroad stay off the map).
    pub iso: String,
    pub places: Vec<PhotoPlace>,
}

/// The groups of `places`, by their keyword path, in stable (path) order. A photo without a
/// found place is in no group.
pub(crate) fn groups(parent: &str, places: &[PhotoPlace]) -> Vec<Group> {
    let mut by_path: BTreeMap<String, Group> = BTreeMap::new();
    for place in places {
        let path = place.path(parent);
        if path.is_empty() {
            continue;
        }
        let country = keywords::clean(&place.country);
        let city = if place.city.is_empty() { String::new() } else { keywords::clean(&place.city) };
        let group = by_path.entry(path.clone()).or_insert_with(|| Group { path, country, city, iso: place.iso.clone(), places: Vec::new() });
        group.places.push(place.clone());
    }
    by_path.into_values().collect()
}

/// What the write of a batch came to.
struct Applied {
    photos: usize,
    keywords: usize,
    undone: bool,
}

#[derive(Clone)]
struct TrackOverlay {
    path: String,
    name: String,
    log: Arc<Tracklog>,
}

/// The incremental sync of the album: what still needs a lookup, and where the last pass stopped.
#[derive(Default)]
struct Sync {
    /// Photo ids to (re)classify, from the last diff. Rebuilt only once drained, so a change
    /// mid-processing waits for the current queue instead of dropping it.
    queue: Vec<PhotoId>,
    /// How far the budgeted processing has got.
    cursor: usize,
    /// The catalog revision the album was last diffed at; `None` wants a full pass.
    baseline: Option<u64>,
    /// When the gazetteer may be tried again after a failed load.
    retry_at: Option<Instant>,
}

#[derive(Default)]
struct State {
    parent: String,
    phase: Phase,
    /// The last `metadata.placeNamesStatus` answer.
    status: Option<Value>,
    /// When to fetch the status again (the first render, a 重新检查, or the next poll while a
    /// download runs).
    status_at: Option<Instant>,
    /// The parsed gazetteer, kept while its file stands (the native `PlaceNames` cache's shape).
    gazetteer: Option<(PathBuf, u64, Option<SystemTime>, Arc<GeoNames>)>,
    /// GPX overlays belong to the current library; paths are persisted, parsed data is rebuilt.
    tracks: Vec<TrackOverlay>,
    track_pick: Option<Receiver<Vec<String>>>,
    track_loads: Vec<Receiver<(String, Result<Tracklog, String>)>>,
    // the album (design 05): the persisted classification, bound to the library
    store: store::Album,
    /// The library generation the store was loaded for.
    loaded_gen: Option<u64>,
    /// Bumped on every album mutation: part of the snapshot's key.
    store_gen: u64,
    /// Whether the album changed since its file was written.
    store_dirty: bool,
    /// When the debounced store save runs.
    save_at: Option<Instant>,
    /// Photos the sync added to the album since the page opened (the card's 新增).
    added: usize,
    /// In-library photos without a usable GPS, from the last full pass.
    no_gps: usize,
    // the album snapshot: the places derived from the store and the catalog as they are now
    places: Vec<PhotoPlace>,
    /// The pending places: album photos still lacking the current path keyword (the review).
    pending: Vec<PhotoPlace>,
    /// The snapshot's key: (store generation, catalog revision, parent).
    snapshot: Option<(u64, u64, String)>,
    /// Photos the lookup found no place for, from the snapshot.
    unmatched: usize,
    // the sync
    sync: Sync,
    // the review
    groups_off: HashSet<String>,
    /// The map's view and what is selected on it.
    map: map::MapView,
    map_sel: Option<String>,
    /// The photo shown in the detail view from a thumbnail click (for the ring).
    photo_sel: Option<PhotoId>,
    /// The online AMap basemap is switched off (the offline schematic map shows instead).
    tiles_off: bool,
    // the outcome
    applied: Option<Applied>,
    apply_error: Option<String>,
    error: Option<String>,
}

impl State {
    /// The included places: their group is ticked in the map's place card.
    fn included<'a>(&'a self, all: &'a [Group]) -> impl Iterator<Item = &'a PhotoPlace> {
        all.iter().filter(|g| !self.groups_off.contains(&g.path)).flat_map(|g| g.places.iter())
    }

    /// How many photos and how many groups the review would write now.
    fn review_counts(&self, all: &[Group]) -> (usize, usize) {
        let mut photos = 0;
        let mut groups = 0;
        for g in all {
            if self.groups_off.contains(&g.path) {
                continue;
            }
            if !g.places.is_empty() {
                photos += g.places.len();
                groups += 1;
            }
        }
        (photos, groups)
    }
}

static STATE: OnceLock<Mutex<State>> = OnceLock::new();

fn lock() -> MutexGuard<'static, State> {
    STATE
        .get_or_init(|| {
            Mutex::new(State {
                parent: PARENT_DEFAULT.to_string(),
                // the first render answers the dataset question (downloaded? how big?)
                status_at: Some(Instant::now()),
                ..State::default()
            })
        })
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
}

/// Tests of the page share the process-wide [`STATE`]; this guard serialises every test that
/// renders the page so a parallel test cannot swap the phase under another's feet.
#[cfg(test)]
pub(crate) fn page_test_lock() -> MutexGuard<'static, ()> {
    static TEST_MU: OnceLock<Mutex<()>> = OnceLock::new();
    let guard = TEST_MU.get_or_init(|| Mutex::new(())).lock().unwrap_or_else(PoisonError::into_inner);
    // the shared tile store holds egui textures of whichever context made them
    crate::tiles::reset_for_test();
    guard
}

// ---------------------------------------------------------------- entry points

/// The Geographic classification page (hub level 2).
pub(crate) fn page(app: &mut LightcraftApp, ui: &mut Ui) {
    pump(app, ui.ctx(), &mut lock());
    widgets_page(ui);
    draw(app, ui, &mut lock());
}

fn widgets_page(ui: &mut Ui) {
    lightcraft_ui_egui::widgets::register(ui, format!("page:{PAGE_ID}"), ui.max_rect());
}

/// The per-frame work: follow a running place-names download by re-reading the status, then run
/// the album's sync (§05), then the debounced store save.
fn pump(app: &mut LightcraftApp, ctx: &egui::Context, st: &mut State) {
    let due = st.status_at.is_some_and(|at| Instant::now() >= at);
    if due {
        st.status_at = None;
        if let Ok(v) = app.run("metadata.placeNamesStatus", Value::Null) {
            let downloading = v.get("downloading").and_then(Value::as_bool).unwrap_or(false);
            if downloading {
                st.status_at = Some(Instant::now() + STATUS_POLL);
                ctx.request_repaint_after(STATUS_POLL);
            }
            st.status = Some(v);
        }
    }
    pump_album(app, ctx, st);
    pump_tracks(app, ctx, st);
    if st.save_at.is_some_and(|at| Instant::now() >= at) {
        save_store(app, st);
    }
}

// ---------------------------------------------------------------- the album sync (design 05)

/// The library's directory, when it is on disk: the store lives there, otherwise the album is
/// memory-only for this session.
fn library_dir(app: &LightcraftApp) -> Option<PathBuf> {
    app.session.library.as_ref().filter(|l| l.on_disk).map(|l| l.dir.clone())
}

/// The album's per-frame work: reload the store when the library changed, diff the catalog for
/// new, moved and removed photos once the last queue has drained, and process the queue within a
/// per-frame budget once the gazetteer is available.
fn pump_album(app: &mut LightcraftApp, ctx: &egui::Context, st: &mut State) {
    // the album is bound to the library: another library loads its own store (or starts empty)
    let library_gen = app.session.library_generation();
    if st.loaded_gen != Some(library_gen) {
        st.loaded_gen = Some(library_gen);
        let dir = library_dir(app);
        match store::path(dir.as_deref()).and_then(|p| store::read(&p)) {
            Some(album) => {
                st.parent = album.parent.clone();
                st.store = album;
            }
            None => {
                st.parent = PARENT_DEFAULT.to_string();
                st.store = store::Album::default();
            }
        }
        st.store_gen += 1;
        st.store_dirty = false;
        st.save_at = None;
        st.added = 0;
        let tracks_before = st.store.tracks.len();
        st.store.tracks.truncate(MAX_TRACKS);
        if st.store.tracks.len() != tracks_before {
            st.store_dirty = true;
            save_soon(st);
        }
        st.sync = Sync::default(); // discard work and ids belonging to the previous library
        st.snapshot = None;
        st.places.clear();
        st.pending.clear();
        st.unmatched = 0;
        st.no_gps = 0;
        st.groups_off.clear();
        st.tracks.clear();
        st.track_pick = None;
        st.track_loads.clear();
        let track_paths: Vec<String> = st.store.tracks.iter().take(MAX_TRACKS).cloned().collect();
        for path in track_paths {
            load_track(app, ctx, st, path);
        }
        st.phase = Phase::Idle;
        st.map = map::MapView::default();
        st.map_sel = None;
        st.photo_sel = None;
        st.applied = None;
        st.apply_error = None;
        st.error = None;
    }
    // the diff runs only once the last queue has drained: a change mid-processing waits, so the
    // queue is never dropped (the next diff re-derives whatever it held)
    let revision = app.session.catalog.revision;
    if st.sync.cursor >= st.sync.queue.len() && st.sync.baseline != Some(revision) {
        sync_diff(app, st);
        st.sync.baseline = Some(revision);
        if st.store_dirty && st.sync.queue.is_empty() {
            save_soon(st);
        }
    }
    // the queue: lookups within the budget, once the place names are on disk (the dataset card
    // offers the download while they are not)
    let present = place_names_present();
    if st.sync.cursor < st.sync.queue.len() && present && st.sync.retry_at.is_none_or(|at| Instant::now() >= at) {
        match gazetteer(st) {
            Ok(names) => {
                st.sync.retry_at = None;
                sync_process(app, st, &names);
            }
            Err(e) => {
                if st.sync.retry_at.is_none_or(|at| Instant::now() >= at) {
                    st.sync.retry_at = Some(Instant::now() + SYNC_RETRY);
                    st.error = Some(e);
                }
            }
        }
    }
    // keep the frames coming while the sync works or a save is due
    if st.sync.cursor < st.sync.queue.len() && present {
        if let Some(at) = st.sync.retry_at {
            ctx.request_repaint_after(at.saturating_duration_since(Instant::now()));
        } else {
            ctx.request_repaint();
        }
    } else if let Some(at) = st.save_at {
        ctx.request_repaint_after(at.saturating_duration_since(Instant::now()));
    }
}

/// Ask the host for a GPX file without blocking the map's UI thread.
fn pick_track(app: &mut LightcraftApp, ctx: &egui::Context, st: &mut State) {
    if st.track_pick.is_some() || st.store.tracks.len() >= MAX_TRACKS {
        return;
    }
    let request = PickRequest::file(i18n::tr("Add GPX track"), i18n::tr("GPS Track Log"), &["gpx"]);
    if let Some(picker) = app.services.picker.as_mut() {
        match picker(request) {
            Ok(rx) => st.track_pick = Some(rx),
            Err(e) => st.error = Some(e),
        }
    } else if let Some(paths) = app.services.pick_tracklog.as_mut().map(|pick| pick())
        && let Some(path) = paths.into_iter().next()
    {
        add_track(app, ctx, st, path);
    }
}

fn add_track(app: &mut LightcraftApp, ctx: &egui::Context, st: &mut State, path: String) {
    if st.store.tracks.len() >= MAX_TRACKS || st.store.tracks.iter().any(|known| known == &path) {
        return;
    }
    st.store.tracks.push(path.clone());
    st.store_dirty = true;
    save_soon(st);
    load_track(app, ctx, st, path);
}

fn load_track(app: &mut LightcraftApp, ctx: &egui::Context, st: &mut State, path: String) {
    let (tx, rx) = channel();
    let repaint = ctx.clone();
    let guard = app.session.activity.start("geoTracklog", "Loading GPX track", lightcraft_engine::activity::Cancel::No);
    let worker_path = path.clone();
    let result_path = path.clone();
    let worker = std::thread::Builder::new().name("lightcraft-gpx".to_string()).spawn(move || {
        let result = (|| {
            let file = std::fs::File::open(&worker_path).map_err(|e| format!("{}: {e}", worker_path))?;
            let meta = file.metadata().map_err(|e| format!("{}: {e}", worker_path))?;
            if !meta.is_file() || meta.len() > MAX_TRACK_BYTES {
                return Err(format!("{}: GPX file exceeds the 64 MiB limit or is not a regular file", worker_path));
            }
            use std::io::Read;
            let mut bytes = Vec::new();
            file.take(MAX_TRACK_BYTES + 1).read_to_end(&mut bytes).map_err(|e| format!("{}: {e}", worker_path))?;
            if bytes.len() as u64 > MAX_TRACK_BYTES {
                return Err(format!("{}: GPX file exceeds the 64 MiB limit", worker_path));
            }
            let text = String::from_utf8(bytes).map_err(|e| format!("{}: GPX must be UTF-8: {e}", worker_path))?;
            let log = parse_gpx(&text).map_err(|e| format!("{}: {e}", worker_path))?;
            if log.points.is_empty() {
                return Err(format!("{}: the track has no timed points", worker_path));
            }
            if log.points.len() > MAX_TRACK_POINTS {
                return Err(format!("{}: the track exceeds the 1,000,000 point limit", worker_path));
            }
            Ok(log)
        })();
        let _ = tx.send((result_path, result));
        drop(guard);
        repaint.request_repaint();
    });
    match worker {
        Ok(_) => st.track_loads.push(rx),
        Err(e) => {
            st.store.tracks.retain(|stored| stored != &path);
            st.store_dirty = true;
            save_soon(st);
            st.error = Some(format!("GPX loader could not start: {e}"));
        }
    }
}

fn pump_tracks(app: &mut LightcraftApp, ctx: &egui::Context, st: &mut State) {
    if let Some(rx) = st.track_pick.as_ref() {
        match rx.try_recv() {
            Ok(paths) => {
                st.track_pick = None;
                if let Some(path) = paths.into_iter().next() {
                    add_track(app, ctx, st, path);
                }
            }
            Err(TryRecvError::Disconnected) => st.track_pick = None,
            Err(TryRecvError::Empty) => ctx.request_repaint_after(Duration::from_millis(250)),
        }
    }
    let mut waiting = Vec::new();
    for rx in std::mem::take(&mut st.track_loads) {
        match rx.try_recv() {
            Ok((path, Ok(log))) => {
                if !st.tracks.iter().any(|t| t.path == path) {
                    let name = std::path::Path::new(&path).file_name().and_then(|n| n.to_str()).unwrap_or("GPX track").to_string();
                    st.tracks.push(TrackOverlay { path, name, log: Arc::new(log) });
                }
            }
            Ok((path, Err(e))) => {
                st.store.tracks.retain(|stored| stored != &path);
                st.store_dirty = true;
                save_soon(st);
                st.error = Some(e);
            }
            Err(TryRecvError::Empty) => waiting.push(rx),
            Err(TryRecvError::Disconnected) => {}
        }
    }
    st.track_loads = waiting;
    if !st.track_loads.is_empty() {
        ctx.request_repaint_after(Duration::from_millis(250));
    }
}

/// Rebuild the queue: prune the album of photos that left the library or lost their GPS, and queue
/// every photo that is new to the album or moved since its classification. The first pass (no
/// baseline) walks the whole library; later passes only the photos the catalog reports touched
/// since it — falling back to the whole library when that history is no longer remembered. The
/// caller records the catalog revision the pass ran at.
fn sync_diff(app: &LightcraftApp, st: &mut State) {
    st.sync.queue.clear();
    st.sync.cursor = 0;
    let touched = st.sync.baseline.and_then(|rev| app.session.catalog.touched_since(rev));
    match touched {
        Some(ids) => {
            for id in ids {
                sync_photo(app, st, id);
            }
        }
        None => {
            // the whole library: prune the album first, then queue what is missing or moved
            let mut gone = Vec::new();
            for (&id, _) in st.store.photos.iter() {
                let keep = match app.session.catalog.photo(PhotoId(id)) {
                    None => false,
                    Some(p) => p.in_library() && !p.is_video() && p.meta.gps.is_some_and(|(lat, lon)| LatLon::new(lat, lon).is_valid()),
                };
                if !keep {
                    gone.push(id);
                }
            }
            for id in gone {
                st.store_dirty |= st.store.remove(PhotoId(id));
            }
            let mut no_gps = 0usize;
            for p in app.session.catalog.photos() {
                if !p.in_library() || p.is_video() {
                    continue;
                }
                let Some((lat, lon)) = p.meta.gps else {
                    no_gps += 1;
                    continue;
                };
                if !LatLon::new(lat, lon).is_valid() {
                    no_gps += 1;
                    continue;
                }
                let id = p.id;
                if !st.store.entry(id).is_some_and(|e| e.at_basis(lat, lon)) {
                    st.sync.queue.push(id);
                }
            }
            st.no_gps = no_gps;
        }
    }
}

/// One touched photo: prune its album entry when it left the library or lost its GPS, queue it
/// when it is new to the album or moved since its classification, and leave an entry that still
/// classifies the photo's current GPS alone.
fn sync_photo(app: &LightcraftApp, st: &mut State, id: PhotoId) {
    let Some(p) = app.session.catalog.photo(id) else {
        st.store_dirty |= st.store.remove(id);
        return;
    };
    if !p.in_library() || p.is_video() {
        st.store_dirty |= st.store.remove(id);
        return;
    }
    let Some((lat, lon)) = p.meta.gps else {
        st.store_dirty |= st.store.remove(id);
        return;
    };
    if !LatLon::new(lat, lon).is_valid() {
        // an unusable position: not stored, so nothing re-queues for it
        st.store_dirty |= st.store.remove(id);
        return;
    }
    if !st.store.entry(id).is_some_and(|e| e.at_basis(lat, lon)) {
        st.sync.queue.push(id);
    }
}

/// Classify up to [`SYNC_BUDGET`] queued photos against `names`, writing the album. An unmatched
/// photo records a bare entry so later passes skip it instead of re-looking it up.
fn sync_process(app: &LightcraftApp, st: &mut State, names: &GeoNames) {
    let deadline = Instant::now() + SYNC_BUDGET_TIME;
    let mut done = 0usize;
    while st.sync.cursor < st.sync.queue.len() && done < SYNC_BUDGET && Instant::now() < deadline {
        let id = st.sync.queue[st.sync.cursor];
        st.sync.cursor += 1;
        done += 1;
        let Some(p) = app.session.catalog.photo(id) else { continue };
        let Some((lat, lon)) = p.meta.gps else { continue }; // gone mid-queue: the next diff prunes
        let at = LatLon::new(lat, lon);
        if !at.is_valid() {
            continue;
        }
        let entry = match names.reverse(at) {
            Some(place) => store::Entry { id: id.0, country: place.country, city: place.city, iso: place.iso, distance: place.distance, lat, lon },
            None => store::Entry { id: id.0, lat, lon, ..store::Entry::default() },
        };
        let fresh = st.store.entry(id).is_none();
        if st.store.entry(id) != Some(&entry) {
            st.store.insert(entry);
            st.store_dirty = true;
            st.store_gen += 1;
            if fresh {
                st.added += 1;
            }
        }
    }
    if st.sync.cursor >= st.sync.queue.len() {
        st.sync.queue.clear();
        st.sync.cursor = 0;
        if st.store_dirty {
            save_soon(st);
        }
    }
}

/// The album snapshot ([`State::places`] and [`State::pending`], and the unmatched count): rebuilt
/// when the store, the parent, or the catalog's photo-relevant state changed — develop edits and
/// ratings do not rebuild it. Entries record the GPS the classification was computed from; the
/// places draw the photo's current GPS, so a moved photo shows at its new spot until the sync
/// re-classifies it.
fn sync_snapshot(app: &LightcraftApp, st: &mut State) {
    let key_gen = st.store_gen;
    let revision = app.session.catalog.revision;
    if let Some((s_gen, s_rev, s_parent)) = st.snapshot.clone()
        && s_gen == key_gen
        && s_parent == st.parent
    {
        if s_rev == revision {
            return;
        }
        if let Some(a) = app.session.catalog.aspects_since(s_rev)
            && !a.any_of(SNAPSHOT_ASPECTS)
        {
            st.snapshot = Some((key_gen, revision, st.parent.clone()));
            return;
        }
    }
    let mut places = Vec::new();
    let mut pending = Vec::new();
    let mut unmatched = 0usize;
    for entry in st.store.photos.values() {
        if entry.unmatched() {
            unmatched += 1;
            continue;
        }
        let id = PhotoId(entry.id);
        let Some(p) = app.session.catalog.photo(id) else { continue };
        if !p.in_library() || p.is_video() {
            continue;
        }
        let Some((lat, lon)) = p.meta.gps else { continue };
        let place = PhotoPlace {
            id,
            name: p.file_name.clone(),
            country: entry.country.clone(),
            city: entry.city.clone(),
            iso: entry.iso.clone(),
            lat,
            lon,
            distance: entry.distance,
        };
        let path = place.path(&st.parent);
        let written = !path.is_empty() && p.meta.keywords.iter().any(|k| keywords::same(k, &path));
        if !written {
            pending.push(place.clone());
        }
        places.push(place);
    }
    st.places = places;
    st.pending = pending;
    st.unmatched = unmatched;
    st.snapshot = Some((key_gen, revision, st.parent.clone()));
}

/// The store file's save: debounced; a failure is logged and retried, never a toast (the sync
/// saves by itself, so an unreadable library directory must not nag every few seconds). Without a
/// library on disk there is nothing to write — the album lives in memory.
fn save_store(app: &LightcraftApp, st: &mut State) {
    st.save_at = None;
    if !st.store_dirty {
        return;
    }
    let dir = library_dir(app);
    let Some(path) = store::path(dir.as_deref()) else { return };
    match store::write(&path, &st.store) {
        Ok(()) => st.store_dirty = false,
        Err(e) => {
            log::warn!("geo classify: the album was not saved: {e}");
            st.save_at = Some(Instant::now() + SAVE_RETRY);
        }
    }
}

/// Save the store file on the next frame.
fn save_soon(st: &mut State) {
    st.save_at = Some(Instant::now() + SAVE_DEBOUNCE);
}

/// Whether the place-names dataset is on disk (the sync waits for it silently; the dataset card
/// offers the download).
fn place_names_present() -> bool {
    lightcraft_engine::config::config_dir().is_some_and(|d| std::fs::metadata(d.join("geonames").join(GEONAMES_FILES[0])).is_ok())
}

// ---------------------------------------------------------------- the dataset

/// Fetch the dataset status on the next frame.
fn status_soon(st: &mut State) {
    st.status_at = Some(Instant::now());
}

/// Whether the dataset is downloaded and what it says, from the last status answer.
fn dataset_ready(status: Option<&Value>) -> bool {
    status.and_then(|v| v.get("downloaded")).and_then(Value::as_bool).unwrap_or(false)
}

/// Start the GeoNames download through the native command; the status follows it.
fn start_download(app: &mut LightcraftApp, st: &mut State) {
    match app.run("metadata.downloadPlaceNames", Value::Null) {
        Ok(_) => status_soon(st),
        Err(e) => st.error = Some(e),
    }
}

/// The gazetteer from the settings folder, parsed once and re-read only when the file stands
/// changed (the native [`lightcraft_engine::cmd::geocode`] cache's rule).
fn gazetteer(st: &mut State) -> Result<Arc<GeoNames>, String> {
    let Some(dir) = lightcraft_engine::config::config_dir().map(|d| d.join("geonames")) else {
        return Err(i18n::tr("The settings folder could not be found.").to_string());
    };
    let file = dir.join(GEONAMES_FILES[0]);
    let Ok(meta) = std::fs::metadata(&file) else {
        return Err(i18n::tr("The place names are not downloaded yet.").to_string());
    };
    let (size, changed) = (meta.len(), meta.modified().ok());
    if let Some((cached_dir, cached_size, cached_changed, names)) = &st.gazetteer
        && *cached_dir == dir
        && *cached_size == size
        && *cached_changed == changed
    {
        return Ok(names.clone());
    }
    GeoNames::load(&dir)
        .map(|names| {
            let names = Arc::new(names);
            st.gazetteer = Some((dir, size, changed, names.clone()));
            names
        })
        .map_err(|e| format!("{}: {e}", i18n::tr("The place names could not be read.")))
}

// ---------------------------------------------------------------- the write

/// Write the kept groups as one `Op::Batch` of `SetMeta`s — one undo step. Only photos with at
/// least one new keyword change; nothing is committed when none do.
fn apply(app: &mut LightcraftApp, st: &State, all: &[Group]) -> Result<(usize, usize), String> {
    let parent = keywords::clean(&st.parent);
    let mut ops: Vec<Op> = Vec::new();
    let mut added: Vec<String> = Vec::new();
    let mut photos = 0usize;
    let mut count = 0usize;
    for place in st.included(all) {
        let path = place.path(&parent);
        if path.is_empty() {
            continue;
        }
        let Some(p) = app.session.catalog.photo(place.id) else { continue };
        if p.meta.keywords.iter().any(|k| keywords::same(k, &path)) {
            continue;
        }
        let mut kws = p.meta.keywords.clone();
        kws.push(path.clone());
        added.push(path);
        photos += 1;
        count += 1;
        ops.push(Op::SetMeta { id: place.id, meta: Box::new(Meta { keywords: kws, ..p.meta.clone() }) });
    }
    if ops.is_empty() {
        return Ok((0, 0));
    }
    let op = Op::Batch { ops };
    let added = added.clone();
    app.session
        .execute_fn("extension.geoClassify.apply", move |s: &mut Session| {
            s.commit(UNDO_LABEL, op)?;
            keywords_cmd::note_recent(s, &added);
            Ok(Value::Null)
        })
        .map_err(|e| e.to_string())?;
    Ok((photos, count))
}

/// Open the review of what the album would write now (the pending places).
fn enter_review(st: &mut State) {
    st.phase = Phase::Review;
    st.map = map::MapView::default();
    st.map_sel = None;
    st.photo_sel = None;
    st.applied = None;
    st.apply_error = None;
    // groups unticked on the album map stay unticked in the review
}

/// Write what the review keeps (Review → Done).
fn write_review(app: &mut LightcraftApp, st: &mut State) {
    let all = groups(&st.parent, &st.pending);
    st.phase = Phase::Applying;
    match apply(app, st, &all) {
        Ok((photos, count)) => {
            st.applied = Some(Applied { photos, keywords: count, undone: false });
            st.apply_error = None;
            st.phase = Phase::Done;
        }
        Err(e) => {
            st.phase = Phase::Review;
            st.apply_error = Some(e);
        }
    }
}

/// Undo the batch, when it is still the newest step (like the Auto Keywords page's: only while
/// nothing was edited after it).
fn undo_batch(app: &mut LightcraftApp, st: &mut State) {
    let is_last = app.session.undo.last().is_some_and(|e| e.label == UNDO_LABEL);
    if is_last
        && st.applied.as_ref().is_some_and(|a| !a.undone)
        && app.run("edit.undo", Value::Null).is_ok()
        && let Some(a) = &mut st.applied
    {
        a.undone = true;
    }
}

/// Back to the album (Discard all / 返回相册). The album and its store stay; only the review's
/// run state clears.
fn back_to_idle(st: &mut State) {
    st.phase = Phase::Idle;
    st.groups_off.clear();
    st.map = map::MapView::default();
    st.map_sel = None;
    st.photo_sel = None;
    st.applied = None;
    st.apply_error = None;
    st.error = None;
}

// ---------------------------------------------------------------- drawing

fn draw(app: &mut LightcraftApp, ui: &mut Ui, st: &mut State) {
    sync_snapshot(app, st);
    let footer_h = match st.phase {
        Phase::Idle => 66.0,
        Phase::Review => 62.0,
        _ => 0.0,
    };
    let (body, footer) = split_page(ui, footer_h);
    let mut body_ui = ui.new_child(egui::UiBuilder::new().max_rect(body));
    if st.phase == Phase::Review {
        // the map fills the panel, so the review runs without the scroll area
        draw_review(app, &mut body_ui, st);
    } else {
        egui::ScrollArea::vertical().id_salt("geo-scroll").auto_shrink([false, false]).show(&mut body_ui, |ui| {
            egui::Frame::NONE.inner_margin(egui::Margin { left: 12, right: 12, top: 10, bottom: 12 }).show(ui, |ui| {
                ui.set_width(ui.available_width());
                match st.phase {
                    Phase::Review => {} // handled above
                    Phase::Applying => {
                        let t = Tokens::get(ui.ctx());
                        ui.add_space(6.0);
                        ui.horizontal(|ui| {
                            spinner(ui, 14.0);
                            ui.label(RichText::new(i18n::tr("Writing the keywords…")).font(t.font(12.0)).color(t.text_label));
                        });
                    }
                    Phase::Done => draw_done(app, ui, st),
                    Phase::Idle => draw_idle(app, ui, st),
                }
            });
        });
    }
    let Some(bar) = footer else { return };
    let mut bar_ui = ui.new_child(egui::UiBuilder::new().max_rect(bar));
    action_bar(&mut bar_ui, |ui| match st.phase {
        Phase::Idle => {
            let label = trn("Write keywords ({n} photos)", &[("n", st.pending.len().to_string())]);
            if primary_action(ui, &label, !st.pending.is_empty(), "geo:start") {
                enter_review(st);
            }
            let note = if dataset_ready(st.status.as_ref()) {
                i18n::tr("New photos join the album automatically; confirm each place before writing.")
            } else {
                i18n::tr("Download the place names once.")
            };
            action_note(ui, note);
        }
        Phase::Review => {
            let all = groups(&st.parent, &st.pending);
            let (photos, _) = st.review_counts(&all);
            let label = trn("Write to {n} photos", &[("n", photos.to_string())]);
            if primary_action(ui, &label, photos > 0, "geo:write") {
                write_review(app, st);
            }
            let out = st.no_gps.saturating_add(st.unmatched);
            let note = if out > 0 {
                trn("Not written: {n} photos", &[("n", out.to_string())])
            } else {
                i18n::tr("The write is one undo step; revert it from Edit ▸ Undo.").to_string()
            };
            action_note(ui, &note);
        }
        _ => {}
    });
}

/// The idle page is the album's home: the dataset card, the album card (the map of everything
/// already in the album, with the sync's progress), the parent keyword.
fn draw_idle(app: &mut LightcraftApp, ui: &mut Ui, st: &mut State) {
    let status = st.status.clone();
    let downloading = status.as_ref().and_then(|v| v.get("downloading")).and_then(Value::as_bool).unwrap_or(false);
    let ready = dataset_ready(status.as_ref());
    card(ui, |ui| {
        let (pill, kind) = if downloading {
            ("Downloading…", Pill::Busy)
        } else if ready {
            ("Ready", Pill::Ready)
        } else {
            ("Not downloaded", Pill::Idle)
        };
        card_head(ui, Icon::Info, "Place names", |ui| status_pill(ui, pill, kind));
        let t = Tokens::get(ui.ctx());
        let title = if downloading {
            i18n::tr("Downloading…").to_string()
        } else if ready {
            status
                .as_ref()
                .and_then(|v| v.get("bytes"))
                .and_then(Value::as_u64)
                .map(|bytes| trn("Ready · {mb} MB", &[("mb", format!("{:.1}", bytes as f64 / (1 << 20) as f64))]))
                .unwrap_or_else(|| i18n::tr("Ready").to_string())
        } else {
            i18n::tr("Offline place names").to_string()
        };
        let line = if ready {
            i18n::tr("The place names list towns of 15 000 people or more.")
        } else {
            i18n::tr("Download once, then lookups are fully offline. About 3 MB.")
        };
        ui.horizontal(|ui| {
            let (tile, _) = ui.allocate_exact_size(vec2(28.0, 28.0), Sense::hover());
            let p = ui.painter();
            p.rect_filled(tile, 7.0, with_alpha(t.accent, 20));
            icons::paint(p, tile.shrink(7.0), Icon::Info, t.accent);
            ui.vertical(|ui| {
                ui.label(RichText::new(title).font(t.font(11.5)).color(t.text_label));
                ui.label(RichText::new(line).font(t.font(10.0)).color(t.text_dim));
            });
        });
        ui.add_space(6.0);
        ui.horizontal(|ui| {
            ui.with_layout(egui::Layout::right_to_left(Align::Center), |ui| {
                if ready {
                    if Button::new(i18n::tr("Re-check")).ghost().small().widget("geo:recheck").show(ui).clicked() {
                        status_soon(st);
                    }
                } else if Button::new(i18n::tr("Download")).secondary().small().widget("geo:download").show(ui).clicked() {
                    start_download(app, st);
                }
            });
        });
        if let Some(e) = &st.error {
            ui.add_space(4.0);
            warning(ui, e);
        }
        ui.add_space(7.0);
        ui.separator();
        ui.add_space(4.0);
        let attribution = status.as_ref().and_then(|v| v.get("attribution")).and_then(Value::as_str).unwrap_or("GeoNames · CC-BY 4.0");
        hint(ui, attribution);
    });
    draw_album_card(app, ui, st);
    card(ui, |ui| {
        card_head(ui, Icon::Tag, "Keywords", |_| {});
        field(ui, "Parent keyword", |ui| {
            let resp = TextField::singleline("geo:parent", &mut st.parent).hint(PARENT_DEFAULT).width(120.0).show(ui);
            if resp.response.changed() || resp.committed() {
                st.parent = keywords::clean(&st.parent);
                // the album remembers the parent between sessions (§05)
                st.store.parent = st.parent.clone();
                st.store_dirty = true;
                st.store_gen += 1;
                st.groups_off.clear();
                save_soon(st);
            }
        });
        let parent = if st.parent.trim().is_empty() { i18n::tr("Country").to_string() } else { st.parent.trim().to_string() };
        path_example(ui, &[parent.as_str(), "中国", "杭州"]);
        hint(ui, i18n::tr("A photo taken in a smaller town classifies by its country."));
    });
}

/// The album card: the map of everything the album holds (the photos the sync has classified),
/// with the sync's progress and the counts the write would cover.
fn draw_album_card(app: &mut LightcraftApp, ui: &mut Ui, st: &mut State) {
    let syncing = st.sync.cursor < st.sync.queue.len();
    let queued = st.sync.queue.len() - st.sync.cursor.min(st.sync.queue.len());
    let pill = if syncing {
        i18n::tr("Syncing…").to_string()
    } else if st.places.is_empty() {
        i18n::tr("Empty").to_string()
    } else {
        trn("{n} photos", &[("n", st.places.len().to_string())])
    };
    let kind = if syncing {
        Pill::Busy
    } else if st.places.is_empty() {
        Pill::Idle
    } else {
        Pill::Ready
    };
    let tracks = st.tracks.clone();
    card(ui, |ui| {
        card_head(ui, Icon::Album, "Map album", |ui| status_pill(ui, &pill, kind));
        draw_tracks_row(app, ui, st);
        if st.places.is_empty() && tracks.is_empty() {
            let line = if place_names_present() {
                i18n::tr("Photos with a GPS position join the album automatically.")
            } else {
                i18n::tr("Download the place names, and photos with a GPS position join the album automatically.")
            };
            ui.add_space(2.0);
            hint(ui, line);
        } else {
            let all = groups(&st.parent, &st.places);
            map::draw_map(app, ui, st, &all, &tracks, Some(ALBUM_MAP_H));
        }
        ui.add_space(6.0);
        let mut line = trn("{n} in the album", &[("n", st.places.len().to_string())]);
        if !st.pending.is_empty() {
            line.push_str(" · ");
            line.push_str(&trn("{n} to write", &[("n", st.pending.len().to_string())]));
        }
        if st.unmatched > 0 {
            line.push_str(" · ");
            line.push_str(&trn("{n} without a place", &[("n", st.unmatched.to_string())]));
        }
        if st.no_gps > 0 {
            line.push_str(" · ");
            line.push_str(&trn("{n} without GPS", &[("n", st.no_gps.to_string())]));
        }
        hint(ui, &line);
        if syncing {
            ui.add_space(2.0);
            hint(ui, &trn("Adding {n} photos…", &[("n", queued.to_string())]));
        } else if st.added > 0 {
            ui.add_space(2.0);
            hint(ui, &trn("Added {n} photos automatically this session.", &[("n", st.added.to_string())]));
        }
    });
}

fn draw_tracks_row(app: &mut LightcraftApp, ui: &mut Ui, st: &mut State) {
    let ctx = ui.ctx().clone();
    let selected = app.session.selection.ids.len();
    ui.horizontal(|ui| {
        if Button::new(i18n::tr("Add GPX track"))
            .secondary()
            .small()
            .enabled(app.services.picker.is_some() || app.services.pick_tracklog.is_some())
            .widget("geo:track:add")
            .show(ui)
            .clicked()
        {
            pick_track(app, &ctx, st);
        }
        ui.label(trn("{n} selected", &[("n", selected.to_string())]));
    });
    for (index, track) in st.tracks.clone().into_iter().enumerate() {
        ui.horizontal(|ui| {
            ui.label(trn("{name} · {n} points", &[("name", track.name.clone()), ("n", track.log.points.len().to_string())]));
            if Button::icon_only(Icon::Target, "Focus on GPX track").ghost().small().widget(format!("geo:track:focus:{index}")).show(ui).clicked()
                && let Some(view) = map::fit_track(&track.log)
            {
                st.map = view;
            }
            if Button::new(i18n::tr("Match selected photos"))
                .secondary()
                .small()
                .enabled(selected > 0)
                .widget(format!("geo:track:match:{index}"))
                .show(ui)
                .clicked()
            {
                let ids: Vec<u64> = app.session.selection.ids.iter().map(|id| id.0).collect();
                if let Err(e) = app.run("photo.tagFromTracklog", serde_json::json!({ "path": track.path, "ids": ids })) {
                    st.error = Some(e);
                }
            }
            if Button::icon_only(Icon::Close, "Remove GPX track").ghost().small().widget(format!("geo:track:remove:{index}")).show(ui).clicked() {
                st.tracks.retain(|loaded| loaded.path != track.path);
                st.store.tracks.retain(|path| path != &track.path);
                st.store_dirty = true;
                st.store_gen += 1;
                st.snapshot = None;
                save_soon(st);
            }
        });
    }
}

/// The review: the summary card, then the map of the pending places. The write sits in the sticky
/// footer.
fn draw_review(app: &mut LightcraftApp, ui: &mut Ui, st: &mut State) {
    let t = Tokens::get(ui.ctx());
    if let Some(e) = st.apply_error.clone() {
        warning(ui, &trn("The write failed: {e}", &[("e", e)]));
        ui.add_space(4.0);
    }
    let all = groups(&st.parent, &st.pending);
    let tracks = st.tracks.clone();
    let (photos, count) = st.review_counts(&all);
    let abroad: usize = all.iter().filter(|g| !g.iso.eq_ignore_ascii_case("CN")).map(|g| g.places.len()).sum();
    card(ui, |ui| {
        card_head(ui, Icon::Check, "Ready to write", |_| {});
        ui.horizontal(|ui| {
            ui.label(RichText::new(photos.to_string()).font(t.semibold(15.0)).color(t.text));
            ui.label(RichText::new(i18n::tr("Photos")).font(t.font(10.5)).color(t.text_dim));
            ui.label(RichText::new("·").font(t.font(10.5)).color(t.text_disabled));
            ui.label(RichText::new(count.to_string()).font(t.semibold(15.0)).color(t.text));
            ui.label(RichText::new(i18n::tr("Places")).font(t.font(10.5)).color(t.text_dim));
        });
        ui.add_space(6.0);
        ui.horizontal(|ui| {
            ui.with_layout(egui::Layout::right_to_left(Align::Center), |ui| {
                if Button::new(i18n::tr("Discard all")).secondary().small().widget("geo:discard").show(ui).clicked() {
                    back_to_idle(st);
                }
                if Button::new(i18n::tr("Accept all")).secondary().small().widget("geo:accept").show(ui).clicked() {
                    st.groups_off.clear();
                }
            });
        });
        ui.add_space(5.0);
        hint(
            ui,
            &trn(
                "Outside China: {n} photos · no match: {m} · no GPS: {k} — not shown, not written",
                &[("n", abroad.to_string()), ("m", st.unmatched.to_string()), ("k", st.no_gps.to_string())],
            ),
        );
    });
    map::draw_map(app, ui, st, &all, &tracks, None);
    ui.add_space(6.0);
    hint(ui, i18n::tr("Bubble size = photo count · scroll to zoom, drag to pan · zoom in for photo thumbnails"));
}

/// The done page: what was written, the undo, back to the album.
fn draw_done(app: &mut LightcraftApp, ui: &mut Ui, st: &mut State) {
    let Some(a) = st.applied.as_ref().map(|a| (a.photos, a.keywords, a.undone)) else { return };
    let (photos, count, undone) = a;
    let title = trn("Wrote {n} photos", &[("n", photos.to_string())]);
    let body = if count == 0 {
        i18n::tr("Everything the album held was already on the photos.").to_string()
    } else if undone {
        i18n::tr("The batch was undone.").to_string()
    } else {
        String::new()
    };
    let out = st.no_gps.saturating_add(st.unmatched);
    let stats = [(i18n::tr("Photos"), photos.to_string()), (i18n::tr("Keywords"), count.to_string()), (i18n::tr("Not written"), out.to_string())];
    success_card(ui, &title, &body, &stats);
    card(ui, |ui| {
        ui.horizontal(|ui| {
            let undo_ok = !undone && app.session.undo.last().is_some_and(|e| e.label == UNDO_LABEL);
            if Button::new(i18n::tr("Undo this batch")).secondary().small().enabled(undo_ok).widget("geo:undo").show(ui).clicked() {
                undo_batch(app, st);
            }
            if Button::new(i18n::tr("Back to the album")).secondary().small().widget("geo:more").show(ui).clicked() {
                back_to_idle(st);
            }
        });
        ui.add_space(5.0);
        hint(ui, i18n::tr("All changes are one undo step: Edit ▸ Undo reverts the whole batch."));
    });
}

#[cfg(test)]
mod tests;
