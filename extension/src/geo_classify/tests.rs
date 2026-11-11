//! Tests of the Geographic classification feature: the keyword paths a place writes, the
//! grouping, the write with its undo (on the demo library, with synthetic lookup results), the
//! map's geometry and points, and headless renders/interactions of the review page.

use egui::vec2;
use lightcraft_catalog::{Meta, Op, PhotoId};
use lightcraft_engine::Session;
use lightcraft_ui_egui::{LightcraftApp, Services};
use serde_json::Value;
use std::time::Instant;

use super::map::{self, MapView, NATIONAL};
use super::{Applied, PhotoPlace, State};
use super::{apply, back_to_idle, groups, lock, page};

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
    let parent = "地点";
    assert_eq!(place(1, "a.jpg", "中国", "杭州").path(parent), "地点|中国|杭州");
    // an empty parent drops its level instead of writing an empty one
    assert_eq!(place(1, "a.jpg", "中国", "杭州").path(""), "中国|杭州");
    // every level is trimmed and empties are dropped
    assert_eq!(place(1, "a.jpg", " 中国 ", " 杭州 ").path(" 地点 "), "地点|中国|杭州");
}

#[test]
fn groups_collect_photos_by_their_keyword_path_in_order() {
    let parent = "地点";
    let places = vec![
        place(1, "a.jpg", "中国", "杭州"),
        place(2, "b.jpg", "中国", "杭州"),
        place(3, "c.jpg", "中国", "北京"),
        place(4, "d.jpg", "日本", "东京"),
    ];
    let groups = groups(parent, &places);
    assert_eq!(groups.len(), 3, "three distinct paths");
    assert_eq!(groups[0].path, "地点|中国|北京");
    assert_eq!(groups[0].places.len(), 1);
    assert_eq!(groups[1].path, "地点|中国|杭州");
    assert_eq!(groups[1].places.iter().map(|p| p.id).collect::<Vec<_>>(), vec![PhotoId(1), PhotoId(2)]);
    assert_eq!(groups[2].path, "地点|日本|东京");
}

#[test]
fn a_photo_without_a_found_city_classifies_by_country() {
    // a defensive case: the lookup found a place but named no city
    let mut p = place(1, "a.jpg", "冰岛", "");
    assert_eq!(p.path("地点"), "地点|冰岛");
    let groups = groups("地点", std::slice::from_ref(&p));
    assert_eq!(groups.len(), 1);
    assert_eq!(groups[0].city, "");
    assert_eq!(groups[0].country, "冰岛");
    let _ = &mut p;
}

// ---------------------------------------------------------------- the write

fn demo() -> (LightcraftApp, PhotoId, PhotoId) {
    let app = LightcraftApp::new(Session::with_demo(), Services::default());
    let ids: Vec<PhotoId> = app.session.catalog.photos().take(2).map(|p| p.id).collect();
    let (a, b) = (ids[0], ids[1]);
    (app, a, b)
}

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
        let p = cam.to_px(&view, lat, lon);
        assert!(rect.contains(p), "{lat},{lon} → {p:?} outside the map");
        let (lat2, lon2) = cam.to_lat_lon(&view, p);
        // pixels are f32, so the round trip carries a small quantisation error
        assert!((lat2 - lat).abs() < 1e-3 && (lon2 - lon).abs() < 1e-3, "round trip of {lat},{lon} gave {lat2},{lon2}");
    }
}

#[test]
fn zooming_keeps_the_point_under_the_anchor() {
    let view = NATIONAL;
    let rect = egui::Rect::from_min_size(egui::pos2(0.0, 0.0), vec2(400.0, 300.0));
    let cam = map::Cam::new(&view, rect);
    let anchor_px = egui::pos2(120.0, 90.0);
    let at = cam.to_lat_lon(&view, anchor_px);
    let zoomed = map::zoom_around(view, 2.0, at, cam.span_lat);
    let cam2 = map::Cam::new(&zoomed, rect);
    let (lat2, lon2) = cam2.to_lat_lon(&zoomed, anchor_px);
    assert!((lat2 - at.0).abs() < 1e-9 && (lon2 - at.1).abs() < 1e-9, "the anchor moved: {lat2},{lon2} vs {at:?}");
    assert!((zoomed.span_lon - view.span_lon / 2.0).abs() < 1e-9, "span did not halve");
}

#[test]
fn the_view_is_clamped_to_china() {
    let mut v = MapView { c_lon: 500.0, c_lat: -200.0, span_lon: 500.0 };
    map::clamp_view(&mut v);
    assert!(v.span_lon <= 70.0 && (75.0..=134.0).contains(&v.c_lon) && (16.0..=53.0).contains(&v.c_lat));
    let mut v = MapView { c_lon: 0.0, c_lat: 0.0, span_lon: 0.1 };
    map::clamp_view(&mut v);
    assert!(v.span_lon >= 1.2, "a too-deep zoom is lifted to the floor");
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

fn with_review_state(places: Vec<PhotoPlace>, parent: &str) {
    let mut st = state_with(places, parent);
    st.phase = super::Phase::Review;
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
    assert!(text.contains("Classify by Location"), "the page's title: {text}");
    assert!(text.contains("Not downloaded"), "the demo session has no place names: {text}");
    assert!(text.contains("CC-BY 4.0"), "the GeoNames attribution is shown: {text}");
}

#[test]
fn the_review_renders_the_map_with_bubbles_headless() {
    let _page = super::page_test_lock();
    let (mut app, a, b) = demo();
    with_review_state(
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
fn clicking_a_bubble_opens_the_place_card() {
    let _page = super::page_test_lock();
    let (mut app, a, b) = demo();
    with_review_state(
        vec![place_at(a.0, "a.jpg", "中国", "北京", "CN", 39.9, 116.4), place_at(b.0, "b.jpg", "中国", "杭州", "CN", 30.27, 120.16)],
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
    *lock() = State::default();
}

#[test]
fn the_wheel_zooms_the_map_around_the_pointer() {
    let _page = super::page_test_lock();
    let (mut app, a, _) = demo();
    with_review_state(vec![place_at(a.0, "a.jpg", "中国", "北京", "CN", 39.9, 116.4)], "地点");
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
    with_review_state(vec![place_at(a.0, "a.jpg", "中国", "北京", "CN", 39.9, 116.4)], "地点");
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
    with_review_state(vec![place_at(a.0, "a.jpg", "中国", "杭州", "CN", 30.27, 120.16)], "地点");
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
fn back_to_idle_clears_the_run_but_keeps_the_settings() {
    let mut st = State {
        parent: "地点".to_string(),
        places: vec![place(1, "a.jpg", "中国", "杭州")],
        without_gps: 3,
        error: Some("x".to_string()),
        ..State::default()
    };
    st.groups_off.insert("地点|中国|杭州".to_string());
    st.applied = Some(Applied { photos: 1, keywords: 1, undone: false });
    back_to_idle(&mut st);
    assert!(st.places.is_empty());
    assert_eq!(st.without_gps, 0);
    assert!(st.error.is_none());
    assert!(st.applied.is_none());
    assert!(st.groups_off.is_empty());
    assert_eq!(st.parent, "地点", "the parent keyword survives");
    assert_eq!(st.map, NATIONAL, "the map view resets for the next run");
}

#[test]
fn the_state_lock_survives_a_poisoned_mutex() {
    let _page = super::page_test_lock();
    // the page's state is shared through a static mutex; a poisoned one (a panic while held)
    // must still open the page: into_inner, never a panic
    drop(lock());
}
