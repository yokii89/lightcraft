//! The Geographic classification feature (level 2 of the extensions hub, design:
//! `extension/docs/01-扩展两级导航与地理位置分类设计.md`): the photos of a scope are grouped by
//! where they were taken — country and city, looked up offline from each photo's GPS with the
//! GeoNames place names — and the place is written as a hierarchical keyword
//! `{parent}|{country}|{city}`, one undo step for the whole batch.
//!
//! The lookup uses [`lightcraft_geocode`] directly, the same public crate the native
//! `metadata.reverseGeocode` command is built on, for three reasons: that command's enabled check
//! wants a selection (the Entire-library scope has none), it writes Location *metadata* — this
//! feature writes keywords — and its per-place details (Sublocation, nearest city) serve the
//! metadata fields, not the two-level keyword. The dataset itself (GeoNames cities15000,
//! CC-BY 4.0, ~3 MB) is downloaded by the native `metadata.downloadPlaceNames` command and shared
//! with it; the page shows the attribution wherever it shows the dataset.

use std::collections::{BTreeMap, HashSet};
use std::path::PathBuf;
use std::sync::{Arc, Mutex, MutexGuard, OnceLock, PoisonError};
use std::time::{Instant, SystemTime};

use egui::{Align, RichText, Sense, Ui, vec2};
use lightcraft_catalog::{Meta, Op, PhotoId};
use lightcraft_engine::Session;
use lightcraft_engine::catalog::keywords;
use lightcraft_engine::cmd::keywords as keywords_cmd;
use lightcraft_geocode::{GEONAMES_FILES, GeoNames, LatLon};
use lightcraft_ui_egui::button::Button;
use lightcraft_ui_egui::icons::{self, Icon};
use lightcraft_ui_egui::progress::spinner;
use lightcraft_ui_egui::text_field::TextField;
use lightcraft_ui_egui::theme::Tokens;
use lightcraft_ui_egui::{LightcraftApp, i18n};
use serde_json::Value;

use crate::scope::{self, Scope};
use crate::ui::{
    Pill, action_bar, action_note, card, card_head, field, hint, path_example, primary_action, split_page, status_pill, success_card, trn, warning,
    with_alpha,
};

mod map;

/// The feature's id in the hub.
pub(crate) const PAGE_ID: &str = "geo-classify";

/// The label of the write's single undo step (§8 of the Auto Keywords design; English, translated
/// from this source string).
const UNDO_LABEL: &str = "Geo Classify";

/// The parent keyword the place paths go under (`{parent}|{country}|{city}`).
const PARENT_DEFAULT: &str = "地点";

/// How often the place-names status is fetched again while a download runs (§: the download is a
/// background task; the page follows it).
const STATUS_POLL: std::time::Duration = std::time::Duration::from_millis(500);

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

/// The groups of the report, by their keyword path, in stable (path) order. A photo without a
/// found place is in no group (the review says separately how many had no GPS or no match).
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

#[derive(Default)]
struct State {
    scope: Scope,
    /// The library count cache for the scope counts (see [`scope::counts`]).
    lib_count: Option<(u64, usize)>,
    parent: String,
    phase: Phase,
    /// The last `metadata.placeNamesStatus` answer.
    status: Option<Value>,
    /// When to fetch the status again (the first render, a 重新检查, or the next poll while a
    /// download runs).
    status_at: Option<Instant>,
    /// The parsed gazetteer, kept while its file stands (the native `PlaceNames` cache's shape).
    gazetteer: Option<(PathBuf, u64, Option<SystemTime>, Arc<GeoNames>)>,
    // the report
    entries: Vec<(PhotoId, String)>,
    places: Vec<PhotoPlace>,
    without_gps: usize,
    unmatched: usize,
    error: Option<String>,
    // the review
    groups_off: HashSet<String>,
    /// The map's view and what is selected on it.
    map: map::MapView,
    map_sel: Option<String>,
    /// The online AMap basemap is switched off (the offline schematic map shows instead).
    tiles_off: bool,
    /// The photo shown in the detail view from a thumbnail click (for the ring).
    photo_sel: Option<PhotoId>,
    // the outcome
    applied: Option<Applied>,
    apply_error: Option<String>,
}

impl State {
    /// The included photos: their group is ticked in the map's place card.
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

/// The per-frame work: follow a running place-names download by re-reading the status.
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

// ---------------------------------------------------------------- the classification

/// Look up every photo of the scope that carries a GPS position and build the review (§: the
/// lookup is offline, from the shared dataset; nothing is written here).
fn run_review(app: &mut LightcraftApp, st: &mut State) {
    st.error = None;
    st.apply_error = None;
    let (entries, _skipped) = scope::enumerate(app, st.scope, false);
    let names = match gazetteer(st) {
        Ok(names) => names,
        Err(e) => {
            st.error = Some(e);
            return;
        }
    };
    let mut places = Vec::new();
    let mut without_gps = 0usize;
    let mut unmatched = 0usize;
    for (id, name) in &entries {
        let Some(p) = app.session.catalog.photo(*id) else { continue };
        let Some((lat, lon)) = p.meta.gps else {
            without_gps += 1;
            continue;
        };
        let at = LatLon::new(lat, lon);
        if !at.is_valid() {
            unmatched += 1;
            continue;
        }
        match names.reverse(at) {
            Some(place) => places.push(PhotoPlace {
                id: *id,
                name: name.clone(),
                country: place.country,
                city: place.city,
                iso: place.iso,
                lat,
                lon,
                distance: place.distance,
            }),
            None => unmatched += 1,
        }
    }
    st.entries = entries;
    st.places = places;
    st.without_gps = without_gps;
    st.unmatched = unmatched;
    st.groups_off.clear();
    st.map = map::MapView::default();
    st.map_sel = None;
    st.photo_sel = None;
    st.applied = None;
    st.apply_error = None;
    st.phase = Phase::Review;
}

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

/// Write what the review keeps (Review → Done).
fn write_review(app: &mut LightcraftApp, st: &mut State) {
    let all = groups(&st.parent, &st.places);
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

/// Back to the settings (Discard all / Classify again), keeping the dataset cache.
fn back_to_idle(st: &mut State) {
    st.phase = Phase::Idle;
    st.places.clear();
    st.entries.clear();
    st.without_gps = 0;
    st.unmatched = 0;
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
            let (selected, view, library) = scope::counts(app, &mut st.lib_count);
            let in_scope = scope::count_of(st.scope, (selected, view, library));
            let ready = dataset_ready(st.status.as_ref());
            let label = trn("Classify ({n} photos)", &[("n", in_scope.to_string())]);
            if primary_action(ui, &label, ready && in_scope > 0, "geo:start") {
                run_review(app, st);
            }
            let note = if !ready { i18n::tr("Download the place names once.") } else { i18n::tr("One undo step reverts the whole batch.") };
            action_note(ui, note);
        }
        Phase::Review => {
            let all = groups(&st.parent, &st.places);
            let (photos, _) = st.review_counts(&all);
            let label = trn("Write to {n} photos", &[("n", photos.to_string())]);
            if primary_action(ui, &label, photos > 0, "geo:write") {
                write_review(app, st);
            }
            let out = st.without_gps.saturating_add(st.unmatched);
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

/// The idle page: the dataset card, the scope, the parent keyword. The one main action lives in
/// the sticky footer.
fn draw_idle(app: &mut LightcraftApp, ui: &mut Ui, st: &mut State) {
    let (selected, view, library) = scope::counts(app, &mut st.lib_count);
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
    card(ui, |ui| {
        card_head(ui, Icon::GridSquare, "Scope", |_| {});
        if let Some(scope) = scope::selector(ui, st.scope, (selected, view, library)) {
            st.scope = scope;
        }
        ui.add_space(6.0);
        hint(ui, i18n::tr("Only photos with a GPS position are classified."));
    });
    card(ui, |ui| {
        card_head(ui, Icon::Tag, "Keywords", |_| {});
        field(ui, "Parent keyword", |ui| {
            let resp = TextField::singleline("geo:parent", &mut st.parent).hint(PARENT_DEFAULT).width(120.0).show(ui);
            if resp.response.changed() || resp.committed() {
                st.parent = keywords::clean(&st.parent);
            }
        });
        let parent = if st.parent.trim().is_empty() { i18n::tr("Country").to_string() } else { st.parent.trim().to_string() };
        path_example(ui, &[parent.as_str(), "中国", "杭州"]);
        hint(ui, i18n::tr("A photo taken in a smaller town classifies by its country."));
    });
}

/// The review: the summary card, then the map as the main view. The write sits in the sticky
/// footer.
fn draw_review(app: &mut LightcraftApp, ui: &mut Ui, st: &mut State) {
    let t = Tokens::get(ui.ctx());
    if let Some(e) = st.apply_error.clone() {
        warning(ui, &trn("The write failed: {e}", &[("e", e)]));
        ui.add_space(4.0);
    }
    let all = groups(&st.parent, &st.places);
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
                &[("n", abroad.to_string()), ("m", st.unmatched.to_string()), ("k", st.without_gps.to_string())],
            ),
        );
    });
    map::draw_map(app, ui, st, &all);
    ui.add_space(6.0);
    hint(ui, i18n::tr("Bubble size = photo count · scroll to zoom, drag to pan · zoom in for photo thumbnails"));
}

/// The done page: what was written, the undo, another round.
fn draw_done(app: &mut LightcraftApp, ui: &mut Ui, st: &mut State) {
    let Some(a) = st.applied.as_ref().map(|a| (a.photos, a.keywords, a.undone)) else { return };
    let (photos, count, undone) = a;
    let title = trn("Wrote {n} photos", &[("n", photos.to_string())]);
    let body = if count == 0 {
        i18n::tr("Everything the model found was already on the photos.").to_string()
    } else if undone {
        i18n::tr("The batch was undone.").to_string()
    } else {
        String::new()
    };
    let out = st.without_gps.saturating_add(st.unmatched);
    let stats = [(i18n::tr("Photos"), photos.to_string()), (i18n::tr("Keywords"), count.to_string()), (i18n::tr("Not written"), out.to_string())];
    success_card(ui, &title, &body, &stats);
    card(ui, |ui| {
        ui.horizontal(|ui| {
            let undo_ok = !undone && app.session.undo.last().is_some_and(|e| e.label == UNDO_LABEL);
            if Button::new(i18n::tr("Undo this batch")).secondary().small().enabled(undo_ok).widget("geo:undo").show(ui).clicked() {
                undo_batch(app, st);
            }
            if Button::new(i18n::tr("Classify more")).secondary().small().widget("geo:more").show(ui).clicked() {
                back_to_idle(st);
            }
        });
        ui.add_space(5.0);
        hint(ui, i18n::tr("All changes are one undo step: Edit ▸ Undo reverts the whole batch."));
    });
}

#[cfg(test)]
mod tests;
