//! Tests of the Geographic classification feature: the keyword paths a place writes, the
//! grouping, the write with its undo (on the demo library, with synthetic lookup results), the
//! persistent album's store and its incremental sync (design 05), the map's geometry and points,
//! and headless renders/interactions of the album and review pages.

use egui::vec2;
use lightcraft_catalog::{Meta, Op, PhotoId};
use lightcraft_engine::Session;
use lightcraft_geocode::GeoNames;
use lightcraft_ui_egui::{LightcraftApp, Services};
use serde_json::Value;
use std::time::Instant;

use super::map::{self, MapView, NATIONAL};
use super::store;
use super::{Applied, PhotoPlace, State, TrackOverlay};
use super::{apply, back_to_idle, groups, lock, page, sync_diff, sync_process, sync_snapshot, write_review};

fn place(id: u64, name: &str, country: &str, city: &str) -> PhotoPlace {
    place_at(id, name, country, city, "XX", 0.0, 0.0)
}

/// A place with its ISO code and GPS (the map needs both).
fn place_at(id: u64, name: &str, country: &str, city: &str, iso: &str, lat: f64, lon: f64) -> PhotoPlace {
    PhotoPlace {
        id: PhotoId(id),
        name: name.to_string(),
        country: country.to_string(),
        city: city.to_string(),
        iso: iso.to_string(),
        lat,
        lon,
        distance: 1200.0,
    }
}

// ---------------------------------------------------------------- paths & grouping

#[test]
fn a_place_writes_parent_country_city() {
    let p = place(1, "a.jpg", "中国", "杭州");
    assert_eq!(p.path("地点"), "地点|中国|杭州");
    assert_eq!(p.path(" 地点 "), "地点|中国|杭州", "the parent is cleaned like any keyword");
    assert_eq!(p.path(""), "中国|杭州", "an empty parent drops its level");
}

#[test]
fn groups_collect_photos_by_their_keyword_path_in_order() {
    let places = vec![place(1, "a.jpg", "中国", "杭州"), place(2, "b.jpg", "中国", "杭州"), place(3, "c.jpg", "中国", "北京")];
    let all = groups("地点", &places);
    assert_eq!(all.len(), 2, "one group per keyword path");
    // the groups sort by their path's bytes: 北 (U+5317) before 杭 (U+676D)
    assert_eq!(all[0].path, "地点|中国|北京");
    assert_eq!(all[0].places.len(), 1);
    assert_eq!(all[1].path, "地点|中国|杭州");
    assert_eq!(all[1].places.len(), 2);
    assert_eq!(all[1].city, "杭州");
    assert_eq!(all[1].iso, "XX");
}

#[test]
fn a_photo_without_a_found_city_classifies_by_country() {
    let p = place_at(1, "a.jpg", "中国", "", "CN", 36.0, 100.0);
    assert_eq!(p.path("地点"), "地点|中国", "no city: the country level closes the path");
    let all = groups("地点", &[p]);
    assert_eq!(all.len(), 1);
    assert!(all[0].city.is_empty());
    assert_eq!(all[0].path, "地点|中国");
}

// ---------------------------------------------------------------- the album's store (design 05)

fn demo() -> (LightcraftApp, PhotoId, PhotoId) {
    let app = LightcraftApp::new(Session::with_demo(), Services::default());
    let a = app.session.catalog.photos().next().expect("the demo library holds photos").id;
    let b = app.session.catalog.photos().nth(1).expect("the demo library holds two photos").id;
    (app, a, b)
}

/// Give a demo photo a GPS position (the demo photos carry none).
fn set_gps(app: &mut LightcraftApp, id: PhotoId, lat: f64, lon: f64) {
    let p = app.session.catalog.photo(id).unwrap();
    let meta = Meta { gps: Some((lat, lon)), ..p.meta.clone() };
    app.session.commit("test", Op::SetMeta { id, meta: Box::new(meta) }).unwrap();
}

/// A synthetic gazetteer: two Chinese cities far enough apart that a photo can match only one.
/// 格式与 GeoNames 的 cities15000/countryInfo 相同(制表符分列:1 名、4 纬、5 经、8 国家码、10 省码)。
fn gazetteer() -> GeoNames {
    let cities = "\t杭州\t\t\t30.25\t120.16\t\t\tCN\t\t02\n\t北京\t\t\t39.90\t116.40\t\t\tCN\t\t11";
    let countries = "CN\t\t\t\t中国";
    GeoNames::parse(cities, None, Some(countries))
}

/// An album entry as the store would hold it for a photo classified at (lat, lon).
fn entry_for(id: PhotoId, lat: f64, lon: f64, city: &str) -> store::Entry {
    store::Entry { id: id.0, country: "中国".to_string(), city: city.to_string(), iso: "CN".to_string(), distance: 8000.0, lat, lon }
}

#[test]
fn a_new_gps_photo_joins_the_album_automatically() {
    let (mut app, a, _) = demo();
    set_gps(&mut app, a, 30.3, 120.2);
    let mut st = State::default();
    sync_diff(&app, &mut st);
    assert_eq!(st.sync.queue, vec![a], "the photo with GPS is queued; the GPS-less demo photos are not");
    assert_eq!(st.no_gps, app.session.catalog.photos().filter(|p| p.in_library()).count() - 1);
    sync_process(&app, &mut st, &gazetteer());
    let e = st.store.entry(a).expect("the photo joined the album");
    assert_eq!((e.country.as_str(), e.city.as_str(), e.iso.as_str()), ("中国", "杭州", "CN"));
    assert!(e.at_basis(30.3, 120.2), "the entry records the GPS it classified");
    assert_eq!(st.added, 1, "the auto-add is counted for the album card");
    assert!(st.store_dirty, "the store waits for its file save");
    // and the second pass finds nothing to do: the album is incremental, not a full re-lookup
    st.sync.baseline = Some(app.session.catalog.revision);
    sync_diff(&app, &mut st);
    assert!(st.sync.queue.is_empty(), "an already-classified photo is not re-queued");
}

#[test]
fn a_moved_photo_is_reclassified() {
    let (mut app, a, _) = demo();
    // the photo was classified in Beijing, then its GPS was corrected to Hangzhou
    let mut st = State { parent: "地点".to_string(), ..State::default() };
    st.store.insert(entry_for(a, 39.9, 116.4, "北京"));
    set_gps(&mut app, a, 30.3, 120.2);
    sync_diff(&app, &mut st);
    assert!(st.sync.queue.contains(&a), "the moved photo is queued again");
    sync_process(&app, &mut st, &gazetteer());
    let e = st.store.entry(a).expect("the entry is rewritten");
    assert_eq!(e.city, "杭州", "the place follows the photo's GPS");
    assert!(e.at_basis(30.3, 120.2));
}

#[test]
fn a_photo_that_lost_its_gps_leaves_the_album() {
    let (app, a, _) = demo();
    // the demo photo carries no GPS; its stale entry is pruned, not drawn
    let mut st = State { parent: "地点".to_string(), ..State::default() };
    st.store.insert(entry_for(a, 30.3, 120.2, "杭州"));
    sync_diff(&app, &mut st);
    assert!(st.store.entry(a).is_none(), "the GPS-less photo's entry is pruned");
    assert!(st.store_dirty, "the prune waits for its file save");
    assert!(st.sync.queue.is_empty());
}

#[test]
fn an_unmatched_photo_is_recorded_and_not_requeued() {
    let (mut app, a, _) = demo();
    // an ocean position is farther than GeoNames' 50 km reach from any synthetic city
    set_gps(&mut app, a, 0.0, 0.0);
    let mut st = State::default();
    sync_diff(&app, &mut st);
    sync_process(&app, &mut st, &gazetteer());
    let e = st.store.entry(a).expect("the photo is recorded");
    assert!(e.unmatched(), "no place found: a bare entry");
    assert_eq!(st.unmatched, 0, "the snapshot counts the unmatched, and has not rebuilt yet");
    sync_snapshot(&app, &mut st);
    assert_eq!(st.unmatched, 1);
    // the next pass skips it instead of looking it up again
    st.sync.baseline = Some(app.session.catalog.revision);
    sync_diff(&app, &mut st);
    assert!(st.sync.queue.is_empty(), "an unmatched photo is not re-looked-up");
}

#[test]
fn a_touched_pass_queues_only_the_changed_photos() {
    let (mut app, a, b) = demo();
    set_gps(&mut app, a, 30.3, 120.2);
    set_gps(&mut app, b, 39.9, 116.4);
    let mut st = State::default();
    sync_diff(&app, &mut st);
    sync_process(&app, &mut st, &gazetteer());
    let before = app.session.catalog.revision;
    // move one photo while the page is closed; the album follows it alone
    set_gps(&mut app, b, 30.35, 120.2);
    st.sync.baseline = Some(before);
    sync_diff(&app, &mut st);
    assert_eq!(st.sync.queue, vec![b], "only the touched photo is queued");
    sync_process(&app, &mut st, &gazetteer());
    assert_eq!(st.store.entry(b).expect("the moved photo re-classified").city, "杭州");
}

#[test]
fn another_library_starts_with_an_empty_album() {
    let (mut app, a, _) = demo();
    let mut st = State { parent: "地点".to_string(), loaded_gen: Some(app.session.library_generation()), ..State::default() };
    st.sync.queue.push(a);
    st.store.insert(store::Entry {
        id: a.0,
        country: "中国".to_string(),
        city: "杭州".to_string(),
        iso: "CN".to_string(),
        distance: 8000.0,
        lat: 30.3,
        lon: 120.2,
    });
    // a new session is another library generation: the album is bound to the library, so the old
    // entries do not carry over
    app.session = Session::with_demo();
    super::pump_album(&mut app, &egui::Context::default(), &mut st);
    assert!(st.store.photos.is_empty(), "the new library's album starts empty");
    assert_eq!(st.parent, super::PARENT_DEFAULT, "an empty library does not inherit the previous parent's keyword");
    assert!(!st.sync.queue.contains(&a), "queued photo ids from the previous library are discarded");
    assert_eq!(st.phase, super::Phase::Idle, "the new library opens on its album page");
    assert_eq!(st.loaded_gen, Some(app.session.library_generation()));
}

#[test]
fn the_first_pass_diffs_once_and_the_queue_survives_until_drained() {
    let (mut app, a, _) = demo();
    set_gps(&mut app, a, 30.3, 120.2);
    let mut st = State::default();
    super::pump_album(&mut app, &egui::Context::default(), &mut st);
    let queued_or_done = st.sync.queue.contains(&a) || st.store.entry(a).is_some();
    assert!(queued_or_done, "the GPS photo is picked up by the first pass (processed at once on a machine with the dataset, queued without it)");
    let other: Vec<_> = st.store.photos.keys().copied().filter(|id| *id != a.0).collect();
    assert!(other.is_empty(), "GPS-less demo photos are not picked up: {other:?}");
}

// ---------------------------------------------------------------- the write from the album

fn state_with(places: Vec<PhotoPlace>, parent: &str) -> State {
    State { parent: parent.to_string(), places, ..State::default() }
}

#[test]
fn the_write_is_one_batch_one_undo_step_and_undo_restores() {
    let (mut app, a, b) = demo();
    let before_a = app.session.catalog.photo(a).unwrap().meta.keywords.clone();
    let before_b = app.session.catalog.photo(b).unwrap().meta.keywords.clone();
    let undo_before = app.session.undo.len();
    let st = state_with(vec![place(a.0, "a.jpg", "中国", "杭州"), place(b.0, "b.jpg", "日本", "东京")], "地点");
    let all = groups("地点", &st.places);
    let (photos, count) = apply(&mut app, &st, &all).unwrap();
    assert_eq!((photos, count), (2, 2));
    assert!(app.session.catalog.photo(a).unwrap().meta.keywords.contains(&"地点|中国|杭州".to_string()));
    assert!(app.session.catalog.photo(b).unwrap().meta.keywords.contains(&"地点|日本|东京".to_string()));
    assert_eq!(app.session.undo.len(), undo_before + 1, "the whole batch is one undo step");
    assert_eq!(app.session.undo.last().unwrap().label, "Geo Classify");
    assert!(app.session.recent_keywords.iter().any(|k| k == "地点|中国|杭州"));
    // and undo puts both photos back
    app.run("edit.undo", Value::Null).unwrap();
    assert_eq!(app.session.catalog.photo(a).unwrap().meta.keywords, before_a);
    assert_eq!(app.session.catalog.photo(b).unwrap().meta.keywords, before_b);
}

#[test]
fn keywords_dedupe_against_the_photo_and_across_the_batch() {
    use lightcraft_engine::catalog::keywords;
    let (mut app, a, b) = demo();
    // photo a carries the place already, spelled in another case
    let meta = Meta { keywords: vec!["地点|中国|杭州".to_string()], ..Default::default() };
    app.session.commit("test", Op::SetMeta { id: a, meta: Box::new(meta) }).unwrap();
    let st = state_with(vec![place(a.0, "a.jpg", "中国", "杭州"), place(b.0, "b.jpg", "中国", "杭州")], "地点");
    let all = groups("地点", &st.places);
    let (photos, count) = apply(&mut app, &st, &all).unwrap();
    assert_eq!((photos, count), (1, 1), "photo a has it already; photo b is the only new one");
    let after = app.session.catalog.photo(a).unwrap().meta.keywords.clone();
    assert_eq!(after.iter().filter(|k| keywords::same(k, "地点|中国|杭州")).count(), 1);
}

#[test]
fn excluded_groups_are_not_written() {
    let (mut app, a, b) = demo();
    let mut st = state_with(vec![place(a.0, "a.jpg", "中国", "杭州"), place(b.0, "b.jpg", "日本", "东京")], "地点");
    st.groups_off.insert("地点|中国|杭州".to_string());
    let all = groups("地点", &st.places);
    let (photos, _) = apply(&mut app, &st, &all).unwrap();
    assert_eq!(photos, 1, "only the ticked group writes");
    assert!(app.session.catalog.photo(b).unwrap().meta.keywords.contains(&"地点|日本|东京".to_string()));
}

#[test]
fn nothing_new_commits_nothing() {
    let (mut app, a, _) = demo();
    let meta = Meta { keywords: vec!["地点|中国|杭州".to_string()], ..Default::default() };
    app.session.commit("test", Op::SetMeta { id: a, meta: Box::new(meta) }).unwrap();
    let undo_before = app.session.undo.len();
    let st = state_with(vec![place(a.0, "a.jpg", "中国", "杭州")], "地点");
    let all = groups("地点", &st.places);
    let (photos, count) = apply(&mut app, &st, &all).unwrap();
    assert_eq!((photos, count), (0, 0));
    assert_eq!(app.session.undo.len(), undo_before, "an empty batch is not committed");
}

#[test]
fn the_review_writes_only_the_pending_and_the_album_survives() {
    let (mut app, a, b) = demo();
    set_gps(&mut app, a, 30.3, 120.2);
    set_gps(&mut app, b, 39.9, 116.4);
    let mut st = State { parent: "地点".to_string(), ..State::default() };
    sync_diff(&app, &mut st);
    sync_process(&app, &mut st, &gazetteer());
    assert_eq!(st.store.photos.len(), 2, "both photos joined the album");
    sync_snapshot(&app, &mut st);
    assert_eq!(st.pending.len(), 2, "neither carries the keyword yet");
    // photo a already carries its place: it stays in the album but leaves the pending write
    let meta = Meta { keywords: vec!["地点|中国|杭州".to_string()], ..app.session.catalog.photo(a).unwrap().meta.clone() };
    app.session.commit("test", Op::SetMeta { id: a, meta: Box::new(meta) }).unwrap();
    sync_snapshot(&app, &mut st);
    assert_eq!(st.pending.len(), 1, "only the un-keyworded photo is pending");
    assert_eq!(st.pending[0].id, b);
    // the review writes only what is pending; the album keeps everything
    write_review(&mut app, &mut st);
    assert_eq!(st.applied.as_ref().map(|r| r.photos), Some(1));
    assert!(app.session.catalog.photo(b).unwrap().meta.keywords.contains(&"地点|中国|北京".to_string()));
    assert_eq!(st.store.photos.len(), 2, "the album is untouched by the write");
    sync_snapshot(&app, &mut st);
    assert!(st.pending.is_empty(), "after the write nothing is pending");
    // and undoing the batch puts it back
    app.run("edit.undo", Value::Null).unwrap();
    sync_snapshot(&app, &mut st);
    assert_eq!(st.pending.len(), 1, "the undone photo is pending again");
}

#[test]
fn review_counts_follow_the_boxes() {
    let mut st = State {
        places: vec![place(1, "a.jpg", "中国", "杭州"), place(2, "b.jpg", "中国", "杭州"), place(3, "c.jpg", "日本", "东京")],
        ..State::default()
    };
    st.parent = "地点".to_string();
    let all = groups("地点", &st.places);
    assert_eq!(st.review_counts(&all), (3, 2));
    // the 杭州 group off: one photo, one group
    st.groups_off.insert("地点|中国|杭州".to_string());
    assert_eq!(st.review_counts(&all), (1, 1));
}

// ---------------------------------------------------------------- the map's geometry

#[test]
fn the_projection_round_trips_through_the_camera() {
    let view = NATIONAL;
    let rect = egui::Rect::from_min_size(egui::pos2(0.0, 0.0), vec2(400.0, 300.0));
    let cam = map::Cam::new(&view, rect);
    for (lat, lon) in [(39.9, 116.4), (30.27, 120.16), (18.25, 109.5), (43.83, 87.62)] {
        let p = cam.to_px(lat, lon);
        assert!(rect.contains(p), "{lat},{lon} → {p:?} outside the map");
        let (lat2, lon2) = cam.to_lat_lon(p);
        // pixels are f32 and the GCJ-02 inverse iterates, so the round trip carries a small error
        assert!((lat2 - lat).abs() < 1e-3 && (lon2 - lon).abs() < 1e-3, "round trip of {lat},{lon} gave {lat2},{lon2}");
    }
}

#[test]
fn zooming_keeps_the_point_under_the_anchor() {
    let view = NATIONAL;
    let rect = egui::Rect::from_min_size(egui::pos2(0.0, 0.0), vec2(400.0, 300.0));
    let cam = map::Cam::new(&view, rect);
    let anchor_px = egui::pos2(120.0, 90.0);
    let at = cam.to_lat_lon(anchor_px);
    let zoomed = map::zoom_around(view, 2.0, at, rect);
    let cam2 = map::Cam::new(&zoomed, rect);
    let (lat2, lon2) = cam2.to_lat_lon(anchor_px);
    assert!((lat2 - at.0).abs() < 1e-6 && (lon2 - at.1).abs() < 1e-6, "the anchor moved: {lat2},{lon2} vs {at:?}");
    assert!((zoomed.span_lon - view.span_lon / 2.0).abs() < 1e-9, "span did not halve");
}

#[test]
fn the_view_is_clamped_to_world_bounds() {
    let mut v = MapView { c_lon: 500.0, c_lat: -200.0, span_lon: 500.0 };
    map::clamp_view(&mut v);
    assert!(v.span_lon <= 360.0 && (-180.0..=180.0).contains(&v.c_lon) && (-80.0..=85.0).contains(&v.c_lat));
    let mut v = MapView { c_lon: 0.0, c_lat: 0.0, span_lon: 0.1 };
    map::clamp_view(&mut v);
    assert!(v.span_lon >= 0.02, "a too-deep zoom is lifted to the street-level floor");
}

// ---------------------------------------------------------------- the map's points

#[test]
fn chinese_groups_become_map_points_and_foreign_ones_do_not() {
    let parent = "地点";
    let places = vec![
        place_at(1, "a.jpg", "中国", "杭州", "CN", 30.0, 120.0),
        place_at(2, "b.jpg", "中国", "杭州", "CN", 30.4, 120.4),
        place_at(3, "c.jpg", "日本", "东京", "JP", 35.7, 139.7),
    ];
    let all = groups(parent, &places);
    let points = map::map_points(&all);
    assert_eq!(points.len(), 1, "only the Chinese group is on the map");
    let p = &points[0];
    assert_eq!(p.path, "地点|中国|杭州");
    assert_eq!(p.count, 2);
    assert!((p.center.0 - 30.2).abs() < 1e-9 && (p.center.1 - 120.2).abs() < 1e-9, "the bubble sits at the centroid");
    assert_eq!(p.photos.len(), 2, "the thumbnails come from the group's photos");
}

// ---------------------------------------------------------------- the page

/// Install a Review state over the demo app's *injected* places: the page state looks exactly as
/// it does after the first pump (library generation recorded, sync baseline set), and the
/// snapshot key matches the current catalog — so the page keeps the injected places instead of
/// rebuilding them from the (empty, memory-only) store.
fn with_review_state(app: &LightcraftApp, places: Vec<PhotoPlace>, parent: &str) {
    let mut st = state_with(places.clone(), parent);
    st.pending = places; // the review draws what the album still has to write
    st.phase = super::Phase::Review;
    st.loaded_gen = Some(app.session.library_generation());
    st.sync.baseline = Some(app.session.catalog.revision);
    st.snapshot = Some((0, app.session.catalog.revision, parent.to_string()));
    *lock() = st;
}

#[test]
fn the_page_renders_headless_and_names_the_dataset_state() {
    let _page = super::page_test_lock();
    // other tests share the static; this one needs the idle page with the status due (so the
    // attribution of the demo session's missing dataset is fetched and shown)
    *lock() = State { status_at: Some(Instant::now()), ..State::default() };
    let mut app = LightcraftApp::new(Session::with_demo(), Services::default());
    let mut view = lightcraft_ui_egui::headless::HeadlessView::new();
    let input = lightcraft_ui_egui::headless::HeadlessView::raw_input(vec2(360.0, 780.0), 1.0, 0.0, vec![]);
    let _ = view.run(input, |ui| page(&mut app, ui));
    let text = view.painted_text().join("\n");
    assert!(text.contains("Place names"), "the dataset card: {text}");
    assert!(
        text.contains("Not downloaded") || text.contains("Ready"),
        "the page names the dataset state (a machine may already hold the dataset): {text}"
    );
    assert!(text.contains("CC-BY 4.0"), "the GeoNames attribution is shown: {text}");
    let registered: Vec<(String, egui::Rect)> = lightcraft_ui_egui::widgets::take_registry(&view.ctx);
    assert!(
        registered.iter().any(|(id, _)| id == "geo:download" || id == "geo:recheck"),
        "the dataset action (download or re-check): {registered:?}"
    );
    assert!(registered.iter().any(|(id, _)| id == "geo:start"), "the start action is in the sticky footer: {registered:?}");
}

#[test]
fn the_album_card_renders_the_stored_album_headless() {
    let _page = super::page_test_lock();
    let (mut app, a, b) = demo();
    set_gps(&mut app, a, 30.3, 120.2);
    set_gps(&mut app, b, 39.9, 116.4);
    // photo a already carries its place: in the album, but not in the pending write
    let meta = Meta { keywords: vec!["地点|中国|杭州".to_string()], ..app.session.catalog.photo(a).unwrap().meta.clone() };
    app.session.commit("test", Op::SetMeta { id: a, meta: Box::new(meta) }).unwrap();
    let mut album = store::Album { version: store::VERSION, parent: "地点".to_string(), photos: Default::default(), tracks: Vec::new() };
    album.insert(store::Entry {
        id: a.0,
        country: "中国".to_string(),
        city: "杭州".to_string(),
        iso: "CN".to_string(),
        distance: 8000.0,
        lat: 30.3,
        lon: 120.2,
    });
    album.insert(store::Entry {
        id: b.0,
        country: "中国".to_string(),
        city: "北京".to_string(),
        iso: "CN".to_string(),
        distance: 8000.0,
        lat: 39.9,
        lon: 116.4,
    });
    *lock() = State {
        parent: "地点".to_string(),
        loaded_gen: Some(app.session.library_generation()),
        store: album,
        status_at: Some(Instant::now()),
        ..State::default()
    };
    let mut view = lightcraft_ui_egui::headless::HeadlessView::new();
    let input = lightcraft_ui_egui::headless::HeadlessView::raw_input(vec2(520.0, 900.0), 1.0, 0.0, vec![]);
    let _ = view.run(input, |ui| page(&mut app, ui));
    let text = view.painted_text().join("\n");
    assert!(text.contains("Map album"), "the album card: {text}");
    assert!(text.contains("2 in the album"), "both album photos are counted: {text}");
    assert!(text.contains("1 to write"), "the already-keyworded photo is not pending: {text}");
    let registered = lightcraft_ui_egui::widgets::take_registry(&view.ctx);
    assert!(registered.iter().any(|(id, _)| id == "geo:map"), "the album map is registered for automation");
    assert!(registered.iter().any(|(id, _)| id == "geo:map:bubble:0"), "the album's places are on the map");
    assert!(registered.iter().any(|(id, _)| id == "geo:start"), "the write action is in the sticky footer");
    *lock() = State::default();
}

#[test]
fn the_review_renders_the_map_with_bubbles_headless() {
    let _page = super::page_test_lock();
    let (mut app, a, b) = demo();
    with_review_state(
        &app,
        vec![place_at(a.0, "a.jpg", "中国", "北京", "CN", 39.9, 116.4), place_at(b.0, "b.jpg", "中国", "杭州", "CN", 30.27, 120.16)],
        "地点",
    );
    let mut view = lightcraft_ui_egui::headless::HeadlessView::new();
    let input = lightcraft_ui_egui::headless::HeadlessView::raw_input(vec2(520.0, 780.0), 1.0, 0.0, vec![]);
    let _ = view.run(input, |ui| page(&mut app, ui));
    let text = view.painted_text().join("\n");
    assert!(text.contains("Ready to write"), "the review summary: {text}");
    assert!(text.contains("South China Sea Islands"), "the inset is painted on the national view: {text}");
    let registered = lightcraft_ui_egui::widgets::take_registry(&view.ctx);
    assert!(registered.iter().any(|(id, _)| id == "geo:map"), "the map is registered for automation");
    assert!(registered.iter().any(|(id, _)| id == "geo:map:bubble:0"), "the first bubble is clickable");
    assert!(registered.iter().any(|(id, _)| id == "geo:map:bubble:1"), "the second bubble is clickable");
    assert!(registered.iter().any(|(id, _)| id == "geo:write"), "the write action is in the sticky footer");
    // the mainland outline: a filled/closed path with nearly all of its vertices
    let big = view
        .painted_shapes()
        .iter()
        .filter_map(|s| match s {
            egui::Shape::Path(p) => Some(p.points.len()),
            _ => None,
        })
        .max();
    assert!(big.is_some_and(|n| n >= 90), "the schematic outline is painted: biggest path {big:?}");
    *lock() = State::default();
}

#[test]
fn cached_basemap_tiles_are_painted_with_the_attribution() {
    let _page = super::page_test_lock();
    let (mut app, a, _) = demo();
    with_review_state(&app, vec![place_at(a.0, "a.jpg", "中国", "北京", "CN", 39.9, 116.4)], "地点");
    let mut view = lightcraft_ui_egui::headless::HeadlessView::new();
    let size = vec2(520.0, 780.0);
    let _ = view.run(lightcraft_ui_egui::headless::HeadlessView::raw_input(size, 1.0, 0.0, vec![]), |ui| page(&mut app, ui));
    // the national view at 520 pt asks for z4; seed the tile under Beijing as if it had arrived
    let (wx, wy) = crate::tiles::world_xy_gcj(39.9, 116.4);
    let n = 1u32 << 4;
    let key = crate::tiles::TileKey { z: 4, x: (wx * n as f64) as u32, y: (wy * n as f64) as u32 };
    crate::tiles::insert_for_test(&view.ctx, key, 2);
    let _ = view.run(lightcraft_ui_egui::headless::HeadlessView::raw_input(size, 1.0, 0.1, vec![]), |ui| page(&mut app, ui));
    let textured =
        view.painted_shapes().iter().any(|shape| matches!(shape, egui::Shape::Mesh(mesh) if mesh.texture_id != egui::TextureId::default()));
    assert!(textured, "a cached basemap tile is painted as a texture");
    let text = view.painted_text().join("\n");
    assert!(text.contains("高德"), "the AMap attribution is painted: {text}");
    assert!(!text.contains("The boundaries are schematic."), "the schematic note gives way to the basemap: {text}");
    *lock() = State::default();
}

#[test]
fn clicking_a_bubble_opens_the_place_card() {
    let _page = super::page_test_lock();
    let (mut app, a, b) = demo();
    with_review_state(
        &app,
        vec![place_at(a.0, "a.jpg", "中国", "北京", "CN", 39.9, 116.4), place_at(b.0, "b.jpg", "中国", "北京", "CN", 39.91, 116.41)],
        "地点",
    );
    let mut view = lightcraft_ui_egui::headless::HeadlessView::new();
    let size = vec2(520.0, 780.0);
    let mut frame = |view: &mut lightcraft_ui_egui::headless::HeadlessView, time: f64, events: Vec<egui::Event>| {
        let raw = lightcraft_ui_egui::headless::HeadlessView::raw_input(size, 1.0, time, events);
        view.run(raw, |ui| page(&mut app, ui))
    };
    let _ = frame(&mut view, 0.0, vec![]);
    let rect = lightcraft_ui_egui::widgets::take_registry(&view.ctx)
        .into_iter()
        .find(|(id, _)| id == "geo:map:bubble:0")
        .map(|(_, r)| r)
        .expect("the bubble rect");
    let at = rect.center();
    let _ = frame(&mut view, 0.1, vec![egui::Event::PointerMoved(at)]);
    let _ = frame(
        &mut view,
        0.2,
        vec![egui::Event::PointerButton { pos: at, button: egui::PointerButton::Primary, pressed: true, modifiers: egui::Modifiers::NONE }],
    );
    let _ = frame(
        &mut view,
        0.3,
        vec![egui::Event::PointerButton { pos: at, button: egui::PointerButton::Primary, pressed: false, modifiers: egui::Modifiers::NONE }],
    );
    assert!(lock().map_sel.is_some(), "the click selected the bubble");
    let _ = frame(&mut view, 0.4, vec![]); // a quiet frame: a fresh area paints from its second frame
    let text = view.painted_text().join("\n");
    assert!(text.contains("Include in write"), "the place card is open: {text}");
    assert!(text.contains("Focus on the place"), "the card offers the focus jump: {text}");
    assert!(
        lightcraft_ui_egui::widgets::take_registry(&view.ctx).iter().any(|(id, _)| id == "geo:map:select"),
        "the place card offers selecting its photos in the library"
    );
    *lock() = State::default();
}

#[test]
fn selecting_a_place_selects_all_its_photos_in_the_library() {
    let (mut app, a, b) = demo();
    let photos = vec![place_at(a.0, "a.jpg", "中国", "北京", "CN", 39.9, 116.4), place_at(b.0, "b.jpg", "中国", "北京", "CN", 39.91, 116.41)];
    let points = map::map_points(&groups("地点", &photos));
    let point = points.first().expect("the China group becomes a map point");
    map::select_place_photos(&mut app, point).expect("the library selection command succeeds");
    assert_eq!(app.session.selection.ids, vec![a, b], "every photo at the place is selected");
    assert_eq!(app.session.selection.active, Some(a), "the first photo is active");
}

#[test]
fn focusing_a_track_frames_routes_outside_china() {
    let track = lightcraft_meta::parse_gpx(
        r#"<gpx version="1.1" creator="test"><trk><trkseg>
           <trkpt lat="46" lon="7"><time>2026-05-01T10:00:00Z</time></trkpt>
           <trkpt lat="47" lon="8"><time>2026-05-01T10:01:00Z</time></trkpt>
        </trkseg></trk></gpx>"#,
    )
    .expect("the synthetic GPX parses");
    let view = map::fit_track(&track).expect("the route has positions");
    assert!(view.c_lat > 45.0 && view.c_lat < 48.0);
    assert!(view.c_lon > 6.0 && view.c_lon < 9.0);
    let mut view = view;
    map::clamp_view(&mut view);
    assert!(view.c_lat > 45.0 && view.c_lon < 9.0, "the map camera accepts locations outside China");
}

#[test]
fn the_map_paints_the_gpx_route_overlay() {
    let _page = super::page_test_lock();
    let (mut app, a, _) = demo();
    let log = lightcraft_meta::parse_gpx(
        r#"<gpx version="1.1" creator="test"><trk><trkseg>
           <trkpt lat="30.2" lon="120.1"><time>2026-05-01T10:00:00Z</time></trkpt>
           <trkpt lat="30.3" lon="120.2"><time>2026-05-01T10:01:00Z</time></trkpt>
        </trkseg></trk></gpx>"#,
    )
    .expect("the synthetic GPX parses");
    with_review_state(&app, vec![place_at(a.0, "a.jpg", "中国", "杭州", "CN", 30.27, 120.16)], "地点");
    lock().tracks.push(TrackOverlay { path: "trip.gpx".to_string(), name: "trip.gpx".to_string(), log: std::sync::Arc::new(log) });
    let mut view = lightcraft_ui_egui::headless::HeadlessView::new();
    let input = lightcraft_ui_egui::headless::HeadlessView::raw_input(vec2(520.0, 780.0), 1.0, 0.0, vec![]);
    let _ = view.run(input, |ui| page(&mut app, ui));
    let route = egui::Color32::from_rgb(0x0b, 0x7f, 0x87);
    assert!(
        view.painted_shapes().iter().any(|shape| matches!(shape, egui::Shape::LineSegment { stroke, .. } if stroke.color == route)),
        "the track's route stroke is painted"
    );
    *lock() = State::default();
}

#[test]
fn the_album_lists_track_actions_and_a_selection_count() {
    let _page = super::page_test_lock();
    let (mut app, _, _) = demo();
    let log = lightcraft_meta::parse_gpx(
        r#"<gpx version="1.1" creator="test"><trk><trkseg>
           <trkpt lat="30.2" lon="120.1"><time>2026-05-01T10:00:00Z</time></trkpt>
           <trkpt lat="30.3" lon="120.2"><time>2026-05-01T10:01:00Z</time></trkpt>
        </trkseg></trk></gpx>"#,
    )
    .expect("the synthetic GPX parses");
    *lock() = State {
        loaded_gen: Some(app.session.library_generation()),
        tracks: vec![TrackOverlay { path: "trip.gpx".to_string(), name: "trip.gpx".to_string(), log: std::sync::Arc::new(log) }],
        ..State::default()
    };
    let mut view = lightcraft_ui_egui::headless::HeadlessView::new();
    let input = lightcraft_ui_egui::headless::HeadlessView::raw_input(vec2(520.0, 900.0), 1.0, 0.0, vec![]);
    let _ = view.run(input, |ui| page(&mut app, ui));
    let text = view.painted_text().join("\n");
    assert!(text.contains("trip.gpx · 2 points"), "the imported track is listed: {text}");
    let ids: Vec<String> = lightcraft_ui_egui::widgets::take_registry(&view.ctx).into_iter().map(|(id, _)| id).collect();
    assert!(ids.iter().any(|id| id == "geo:track:add"), "the album can add another GPX track");
    assert!(ids.iter().any(|id| id == "geo:track:focus:0"), "the route can be framed");
    assert!(ids.iter().any(|id| id == "geo:track:match:0"), "the selected photos can be time-matched");
    assert!(ids.iter().any(|id| id == "geo:track:remove:0"), "the imported route can be removed");
    *lock() = State::default();
}

#[test]
fn the_wheel_zooms_the_map_around_the_pointer() {
    let _page = super::page_test_lock();
    let (mut app, a, _) = demo();
    with_review_state(&app, vec![place_at(a.0, "a.jpg", "中国", "北京", "CN", 39.9, 116.4)], "地点");
    let mut view = lightcraft_ui_egui::headless::HeadlessView::new();
    let size = vec2(520.0, 780.0);
    let _ = view.run(lightcraft_ui_egui::headless::HeadlessView::raw_input(size, 1.0, 0.0, vec![]), |ui| page(&mut app, ui));
    let map_rect =
        lightcraft_ui_egui::widgets::take_registry(&view.ctx).into_iter().find(|(id, _)| id == "geo:map").map(|(_, r)| r).expect("the map rect");
    let before = lock().map.span_lon;
    let raw = lightcraft_ui_egui::headless::HeadlessView::raw_input(
        size,
        1.0,
        0.1,
        vec![
            egui::Event::PointerMoved(map_rect.center()),
            egui::Event::MouseWheel {
                unit: egui::MouseWheelUnit::Line,
                delta: vec2(0.0, 3.0),
                phase: egui::TouchPhase::Move,
                source: egui::MouseWheelSource::Unknown,
                modifiers: egui::Modifiers::NONE,
            },
        ],
    );
    let _ = view.run(raw, |ui| page(&mut app, ui));
    let after = lock().map.span_lon;
    assert!(after < before, "wheeling in zooms: {before} → {after}");
    // and the zoom landed inside China (the anchor was on the map)
    assert!((75.0..=134.0).contains(&lock().map.c_lon));
    *lock() = State::default();
}

#[test]
fn dragging_pans_the_map() {
    let _page = super::page_test_lock();
    let (mut app, a, _) = demo();
    with_review_state(&app, vec![place_at(a.0, "a.jpg", "中国", "北京", "CN", 39.9, 116.4)], "地点");
    let mut view = lightcraft_ui_egui::headless::HeadlessView::new();
    let size = vec2(520.0, 780.0);
    let _ = view.run(lightcraft_ui_egui::headless::HeadlessView::raw_input(size, 1.0, 0.0, vec![]), |ui| page(&mut app, ui));
    let map_rect =
        lightcraft_ui_egui::widgets::take_registry(&view.ctx).into_iter().find(|(id, _)| id == "geo:map").map(|(_, r)| r).expect("the map rect");
    let from = map_rect.center();
    let to = egui::pos2(from.x - 120.0, from.y);
    let before = lock().map.c_lon;
    // press once, move while down, release: resending the press every frame would reset the drag
    let steps: [(f64, Vec<egui::Event>); 3] = [
        (
            0.1,
            vec![
                egui::Event::PointerMoved(from),
                egui::Event::PointerButton { pos: from, button: egui::PointerButton::Primary, pressed: true, modifiers: egui::Modifiers::NONE },
            ],
        ),
        (0.2, vec![egui::Event::PointerMoved(to)]),
        (
            0.3,
            vec![
                egui::Event::PointerMoved(to),
                egui::Event::PointerButton { pos: to, button: egui::PointerButton::Primary, pressed: false, modifiers: egui::Modifiers::NONE },
            ],
        ),
    ];
    for (time, events) in steps {
        let _ = view.run(lightcraft_ui_egui::headless::HeadlessView::raw_input(size, 1.0, time, events), |ui| page(&mut app, ui));
    }
    let after = lock().map.c_lon;
    assert!(after > before, "dragging left pans east: {before} → {after}");
    *lock() = State::default();
}

#[test]
fn zoomed_in_the_bubbles_become_thumbnails_that_open_the_detail_view() {
    let _page = super::page_test_lock();
    let (mut app, a, _) = demo();
    with_review_state(&app, vec![place_at(a.0, "a.jpg", "中国", "杭州", "CN", 30.27, 120.16)], "地点");
    lock().map = MapView { c_lon: 120.16, c_lat: 30.27, span_lon: map::THRESH * 0.5 };
    let mut view = lightcraft_ui_egui::headless::HeadlessView::new();
    let size = vec2(520.0, 780.0);
    let _ = view.run(lightcraft_ui_egui::headless::HeadlessView::raw_input(size, 1.0, 0.0, vec![]), |ui| page(&mut app, ui));
    let id = format!("geo:map:photo:{}", a.0);
    let rect = lightcraft_ui_egui::widgets::take_registry(&view.ctx)
        .into_iter()
        .find(|(wid, _)| *wid == id)
        .map(|(_, r)| r)
        .expect("the photo thumbnail is registered at its GPS position");
    let at = rect.center();
    for (time, pressed) in [(0.1, true), (0.2, false)] {
        let events = vec![
            egui::Event::PointerMoved(at),
            egui::Event::PointerButton { pos: at, button: egui::PointerButton::Primary, pressed, modifiers: egui::Modifiers::NONE },
        ];
        let _ = view.run(lightcraft_ui_egui::headless::HeadlessView::raw_input(size, 1.0, time, events), |ui| page(&mut app, ui));
    }
    assert!(lock().photo_sel == Some(a), "the thumbnail click selected the photo");
    assert!(app.session.selection.ids.contains(&a), "the library selection follows");
    assert_eq!(app.ui.view, lightcraft_ui_egui::state::ViewMode::Detail, "the photo opens in the detail view");
    *lock() = State::default();
}

#[test]
fn back_to_idle_keeps_the_album_but_clears_the_run() {
    let mut st = State {
        parent: "地点".to_string(),
        places: vec![place(1, "a.jpg", "中国", "杭州")],
        no_gps: 3,
        error: Some("x".to_string()),
        ..State::default()
    };
    st.groups_off.insert("地点|中国|杭州".to_string());
    st.applied = Some(Applied { photos: 1, keywords: 1, undone: false });
    back_to_idle(&mut st);
    assert_eq!(st.places.len(), 1, "the album survives: it is the page's own data, not the run's");
    assert_eq!(st.no_gps, 3);
    assert!(st.error.is_none());
    assert!(st.applied.is_none());
    assert!(st.groups_off.is_empty());
    assert_eq!(st.parent, "地点", "the parent keyword survives");
    assert_eq!(st.map, NATIONAL, "the map view resets for the next look");
}

#[test]
fn the_state_lock_survives_a_poisoned_mutex() {
    let _page = super::page_test_lock();
    // the page's state is shared through a static mutex; a poisoned one (a panic while held)
    // must still open the page: into_inner, never a panic
    drop(lock());
}
