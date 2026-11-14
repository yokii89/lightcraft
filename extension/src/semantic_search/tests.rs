use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use lightcraft_engine::{Session, activity::Cancel};
use lightcraft_ui_egui::{LightcraftApp, Services};

use super::{model, store, worker};

pub(super) fn temp(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("lightcraft-semantic-{name}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn photo(id: u64, text: &str) -> store::PhotoText {
    store::PhotoText { id, basis: [id as u8; 32], name: format!("{id}.jpg"), metadata: text.into(), keywords: vec![] }
}

pub(super) fn vector(dimension: usize) -> Vec<f32> {
    let mut values = vec![0.; model::DIM];
    values[dimension] = 1.;
    values
}

#[test]
fn descriptions_roundtrip_and_undo_or_replacement_invalidates_them() {
    let dir = temp("descriptions");
    let mut descriptions = store::Descriptions::default();
    let mut photo = photo(1, "用户原有说明");
    photo.keywords = vec!["AI|动物|小狗".into()];
    descriptions
        .accept(vec![store::Description {
            id: 1,
            basis: photo.basis,
            description: "白色小狗在草地上奔跑".into(),
            guard_keywords: photo.keywords.clone(),
        }])
        .unwrap();
    descriptions.save(Some(&dir)).unwrap();
    let descriptions = store::Descriptions::load(Some(&dir)).unwrap();
    assert!(photo.text(&descriptions).contains("白色小狗"));
    photo.keywords.clear();
    assert_eq!(photo.text(&descriptions), "用户原有说明");
    photo.keywords.push("AI|动物|小狗".into());
    photo.basis = [9; 32];
    assert!(photo.text(&descriptions).contains("用户原有说明"));
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn a_corrupt_description_file_is_preserved_and_reported() {
    let dir = temp("corrupt-text");
    std::fs::write(dir.join("semantic-descriptions.json"), b"{").unwrap();
    assert!(store::Descriptions::load(Some(&dir)).is_err());
    assert_eq!(std::fs::read(dir.join("semantic-descriptions.json")).unwrap(), b"{");
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn binary_vectors_roundtrip_and_reject_torn_unknown_and_nonfinite_data() {
    let dir = temp("vectors");
    let mut index = store::Index::default();
    index.records.insert(42, store::Vector { fingerprint: [7; 32], values: vector(3) });
    index.save(Some(&dir)).unwrap();
    let path = dir.join("semantic-vectors.bin");
    let bytes = std::fs::read(&path).unwrap();
    assert_eq!(bytes.len(), 48 + 8 + 32 + 512 * 4);
    let loaded = store::Index::load(Some(&dir)).unwrap();
    assert_eq!(loaded.records[&42].values, vector(3));
    assert_eq!(loaded.records[&42].fingerprint, [7; 32]);
    for bad in [
        bytes[..bytes.len() - 1].to_vec(),
        vec![0; 48],
        {
            let mut bad = bytes.clone();
            bad[88..92].copy_from_slice(&f32::NAN.to_le_bytes());
            bad
        },
        {
            let mut bad = bytes.clone();
            bad[40..48].copy_from_slice(&u64::MAX.to_le_bytes());
            bad
        },
    ] {
        std::fs::write(&path, bad).unwrap();
        assert!(store::Index::load(Some(&dir)).is_err());
    }
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn hybrid_search_recalls_semantic_matches_without_keyword_prefiltering() {
    let photos = vec![photo(1, "白色小狗在草地上奔跑"), photo(2, "阳光照着海滩"), photo(3, "雪山")];
    let descriptions = store::Descriptions::default();
    let mut index = store::Index::default();
    for p in &photos {
        index.records.insert(p.id, store::Vector { fingerprint: p.fingerprint(&p.text(&descriptions)), values: vector((p.id - 1) as usize) });
    }
    let cancel = AtomicBool::new(false);
    let hits = worker::rank(&photos, &descriptions, Some(&index), Some(&vector(0)), "宠物在户外运动", 0.6, &cancel).unwrap();
    assert_eq!(hits[0].id, 1, "synonyms can be found with no literal term in common");
    assert!(hits.iter().all(|h| h.id != 2));
    // The metadata changed since indexing: stale vectors must no longer be used.
    let changed = vec![photo(1, "新的内容")];
    assert!(worker::rank(&changed, &descriptions, Some(&index), Some(&vector(0)), "宠物", 0.6, &cancel).unwrap().is_empty());
    assert!(worker::rank(&photos, &descriptions, None, None, "海滩", 0.6, &cancel).unwrap().iter().any(|h| h.id == 2));
    assert!(worker::rank(&photos, &descriptions, None, None, "不存在的东西", 0.6, &cancel).unwrap().is_empty());
}

#[test]
fn cancelled_search_stops_and_the_top_results_are_bounded() {
    let photos: Vec<_> = (1..250).map(|id| photo(id, "草地上的小狗")).collect();
    assert!(worker::rank(&photos, &store::Descriptions::default(), None, None, "小狗", 0.6, &AtomicBool::new(true)).is_err());
    let hits = worker::rank(&photos, &store::Descriptions::default(), None, None, "小狗", 0.6, &AtomicBool::new(false)).unwrap();
    assert_eq!(hits.len(), 100);
    assert_eq!(hits[0].id, 1, "ties have deterministic id ordering");
}

fn lexical_ids(photos: &[store::PhotoText], query: &str) -> Vec<u64> {
    worker::rank(photos, &store::Descriptions::default(), None, None, query, 0.6, &AtomicBool::new(false))
        .unwrap()
        .into_iter()
        .map(|hit| hit.id)
        .collect()
}

#[test]
fn bm25_rewards_rare_terms_instead_of_tying_on_term_coverage() {
    let photos = vec![photo(1, "common"), photo(2, "rare"), photo(3, "common"), photo(4, "common"), photo(5, "common")];
    assert_eq!(lexical_ids(&photos, "common rare").first(), Some(&2));
}

#[test]
fn bm25_rewards_term_frequency_at_equal_document_lengths() {
    let photos = vec![photo(1, "dog tree tree tree"), photo(2, "dog dog dog tree")];
    assert_eq!(lexical_ids(&photos, "dog"), vec![2, 1]);
    assert_eq!(lexical_ids(&photos, "dog dog dog"), vec![2, 1], "query repetitions must not change ranking");
}

#[test]
fn bm25_normalizes_document_length() {
    let photos = vec![photo(1, &format!("dog {}", "tree ".repeat(200))), photo(2, "dog")];
    assert_eq!(lexical_ids(&photos, "dog"), vec![2, 1]);
}

#[test]
fn bm25_tokenization_handles_chinese_unicode_and_whole_english_words() {
    let photos = vec![photo(1, "concatenate"), photo(2, "CAT"), photo(3, "红色小猫，坐在窗边。𠮷野")];
    assert_eq!(lexical_ids(&photos, "cat"), vec![2]);
    assert_eq!(lexical_ids(&photos, "小猫"), vec![3]);
    assert_eq!(lexical_ids(&photos, "𠮷野"), vec![3]);
    assert!(lexical_ids(&photos, "?! 🐈").is_empty());
}

#[test]
fn bm25_recomputes_corpus_statistics_after_photo_changes() {
    let photos = vec![photo(1, "common"), photo(2, "rare"), photo(3, "common"), photo(4, "common")];
    assert_eq!(lexical_ids(&photos, "common rare").first(), Some(&2));
    let changed = vec![photo(1, "common"), photo(2, "rare"), photo(3, "rare"), photo(4, "rare")];
    assert_eq!(lexical_ids(&changed, "common rare").first(), Some(&1));
}

#[test]
fn save_worker_holds_activity_and_isolates_libraries() {
    let dir = temp("worker");
    let mut app = LightcraftApp::new(Session::with_demo(), Services::default());
    let id = app.session.catalog.photos().next().unwrap().id;
    app.run("photo.setMeta", serde_json::json!({"ids": [id.0], "keywords": ["小狗"]})).unwrap();
    let guard = app.session.activity.start("semanticDescriptions", "Saving photo descriptions", Cancel::No);
    assert!(app.session.activity.list().iter().any(|task| task.kind == "semanticDescriptions"));
    let key = worker::Library { generation: 987654, dir: Some(dir.clone()) };
    let rx = worker::submit(
        super::lifecycle::View::new(key.clone(), &app.session.catalog).unwrap(),
        worker::Operation::Save(vec![store::Description {
            id: id.0,
            basis: store::basis(app.session.catalog.photo(id).unwrap()),
            description: "草地上的小狗".into(),
            guard_keywords: vec!["小狗".into()],
        }]),
        Arc::new(AtomicBool::new(false)),
        guard,
    )
    .unwrap();
    assert_eq!(rx.recv_timeout(std::time::Duration::from_secs(10)).unwrap().unwrap().descriptions, 1);
    assert_eq!(store::Descriptions::load(Some(&dir)).unwrap().photos.len(), 1);
    let other = worker::Library { generation: key.generation + 1, dir: None };
    let guard = app.session.activity.start("semanticLoad", "Loading the photo search index", Cancel::No);
    let rx = worker::submit(
        super::lifecycle::View::new(other, &app.session.catalog).unwrap(),
        worker::Operation::Status,
        Arc::new(AtomicBool::new(false)),
        guard,
    )
    .unwrap();
    assert_eq!(rx.recv_timeout(std::time::Duration::from_secs(10)).unwrap().unwrap().descriptions, 0);
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn search_page_registers_actions_and_preserves_user_captions() {
    let mut app = LightcraftApp::new(Session::with_demo(), Services::default());
    let before = app.session.catalog.photos().map(|p| (p.id, p.meta.caption.clone())).collect::<Vec<_>>();
    let mut view = lightcraft_ui_egui::headless::HeadlessView::new();
    let input = lightcraft_ui_egui::headless::HeadlessView::raw_input(egui::vec2(440., 900.), 1., 0., vec![]);
    let mut state = super::State { minimum: 0.6, ..super::State::default() };
    let _ = view.run(input, |ui| super::draw(&mut app, ui, &mut state));
    let registered: Vec<_> = lightcraft_ui_egui::widgets::take_registry(&view.ctx).into_iter().map(|(id, _)| id).collect();
    assert!(registered.iter().any(|id| id == "semantic:query"));
    assert!(registered.iter().any(|id| id == "semantic:advanced"));
    assert!(!registered.iter().any(|id| id == "semantic:index"));
    assert!(registered.iter().any(|id| id == "semantic:search"));
    for (id, caption) in before {
        assert_eq!(app.session.catalog.photo(id).unwrap().meta.caption, caption);
    }
    assert!(crate::hub::features().iter().any(|f| f.id == super::PAGE_ID));
}

fn draw_page(
    app: &mut LightcraftApp,
    view: &mut lightcraft_ui_egui::headless::HeadlessView,
    state: &mut super::State,
    size: egui::Vec2,
    events: Vec<egui::Event>,
) -> std::collections::HashMap<String, egui::Rect> {
    let input = lightcraft_ui_egui::headless::HeadlessView::raw_input(size, 1., 0., events);
    let _ = view.run(input, |ui| super::draw(app, ui, state));
    lightcraft_ui_egui::widgets::take_registry(&view.ctx).into_iter().collect()
}

fn click_page(
    app: &mut LightcraftApp,
    view: &mut lightcraft_ui_egui::headless::HeadlessView,
    state: &mut super::State,
    size: egui::Vec2,
    pos: egui::Pos2,
) -> std::collections::HashMap<String, egui::Rect> {
    let event = |pressed| egui::Event::PointerButton { pos, button: egui::PointerButton::Primary, pressed, modifiers: egui::Modifiers::NONE };
    draw_page(app, view, state, size, vec![egui::Event::PointerMoved(pos), event(true)]);
    draw_page(app, view, state, size, vec![event(false)])
}

#[test]
fn search_footer_stays_visible_and_advanced_controls_toggle_at_narrow_widths() {
    let mut app = LightcraftApp::new(Session::with_demo(), Services::default());
    for width in [270., 440.] {
        let size = egui::vec2(width, 780.);
        let mut view = lightcraft_ui_egui::headless::HeadlessView::new();
        let mut state = super::State { minimum: 0.4, ..Default::default() };
        let widgets = draw_page(&mut app, &mut view, &mut state, size, vec![]);
        let footer = widgets["semantic:search"];
        assert!(footer.top() >= size.y - 66. && footer.bottom() <= size.y);
        assert!(widgets["semantic:query"].bottom() < footer.top());
        for id in ["semantic:index", "semantic:download", "semantic:minimum", "semantic:show"] {
            assert!(!widgets.contains_key(id), "{id} should not clutter the initial page");
        }
        let text = view.painted_text().join("\n");
        assert!(!text.contains("Semantic Photo Search"), "the host already supplies the title");
        assert!(!text.contains("0 search results") && !text.contains("0 updated"));
        click_page(&mut app, &mut view, &mut state, size, widgets["semantic:advanced"].center());
        assert!(state.advanced);
        let widgets = draw_page(&mut app, &mut view, &mut state, size, vec![]);
        assert_eq!(widgets["semantic:search"], footer, "expanding settings must not move the primary action");
        for id in ["semantic:index", "semantic:download", "semantic:minimum"] {
            let rect = widgets[id];
            assert!(rect.left() >= 0. && rect.right() <= width && rect.bottom() < footer.top(), "{id}: {rect:?}");
        }
        click_page(&mut app, &mut view, &mut state, size, widgets["semantic:advanced"].center());
        assert!(!state.advanced);
        let widgets = draw_page(&mut app, &mut view, &mut state, size, vec![]);
        assert!(!widgets.contains_key("semantic:index"));
    }
}

#[test]
fn compact_search_results_keep_selection_and_library_filter_actions() {
    let mut app = LightcraftApp::new(Session::with_demo(), Services::default());
    let ids: Vec<_> = app.session.catalog.photos().take(2).map(|p| p.id.0).collect();
    let size = egui::vec2(270., 780.);
    let mut view = lightcraft_ui_egui::headless::HeadlessView::new();
    let mut state = super::State { searched: true, ..Default::default() };
    state.reply.hits = ids.iter().map(|&id| worker::Hit { id, text: "A dog running through a field. ".repeat(40), semantic: Some(0.789) }).collect();
    let widgets = draw_page(&mut app, &mut view, &mut state, size, vec![]);
    let footer = widgets["semantic:search"];
    assert!(widgets["semantic:show"].bottom() < footer.top());
    assert!(!view.painted_text().join("\n").contains("0.789"), "scores belong in the tooltip");
    for id in &ids {
        let rect = widgets[&format!("semantic:photo:{id}")];
        assert!(rect.right() <= size.x && rect.bottom() < footer.top(), "a long description must not push the next result away");
    }
    click_page(&mut app, &mut view, &mut state, size, widgets[&format!("semantic:photo:{}", ids[1])].center());
    assert_eq!(app.session.selection.active.map(|id| id.0), Some(ids[1]));
    click_page(&mut app, &mut view, &mut state, size, widgets["semantic:show"].center());
    let visible: Vec<_> = app.session.visible().iter().map(|id| id.0).collect();
    assert_eq!(visible.len(), ids.len());
    assert!(ids.iter().all(|id| visible.contains(id)));
}

#[test]
fn normalization_rejects_hostile_vectors() {
    assert!(model::normalize(&mut vec![0.; model::DIM]).is_err());
    assert!(model::normalize(&mut vec![f32::NAN; model::DIM]).is_err());
    assert!(model::normalize(&mut [1.; 3]).is_err());
}

#[test]
fn real_text_model_distinguishes_subjects_and_checks_reference_vectors() {
    let dir = model::default_dir();
    if !dir.join("model.safetensors").exists() {
        lightcraft_testkit::skip!(Model, "set LIGHTCRAFT_TEXT_MODEL to the pinned BGE model directory");
        return;
    }
    let model = model::Encoder::load(&dir).unwrap();
    assert!(!model.device().to_ascii_lowercase().contains("cpu"));
    assert!(model.weight_bytes() > 90_000_000);
    let cancel = AtomicBool::new(false);
    let dog = model.encode("一只白色小狗在草地上奔跑。", false, &cancel).unwrap();
    let beach = model.encode("蓝色海水拍打沙滩，远处停着帆船。", false, &cancel).unwrap();
    let query = model.encode("在户外跑动的宠物", true, &cancel).unwrap();
    let dot = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(a, b)| a * b).sum::<f32>();
    assert!(dot(&query, &dog) > dot(&query, &beach) + 0.05);
    assert!(model.encode(&"很长的输入".repeat(300), true, &cancel).is_err());
    assert!(model.encode("", false, &cancel).is_err());
    assert_eq!(model.encode("猫", false, &AtomicBool::new(true)).unwrap_err(), "Cancelled");
    let reference = dir.join("reference.json");
    if reference.exists() {
        let reference: serde_json::Value = serde_json::from_slice(&std::fs::read(reference).unwrap()).unwrap();
        for case in reference.as_array().unwrap() {
            let encoded = model.encode(case["text"].as_str().unwrap(), case["query"].as_bool().unwrap(), &cancel).unwrap();
            let expected: Vec<f32> = case["vector"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap() as f32).collect();
            assert_eq!(encoded.len(), expected.len());
            let max_error = encoded.iter().zip(&expected).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
            eprintln!("GPU {} official reference max error {max_error}", model.device());
            assert!(max_error < 0.0001, "official Transformers vs Rust max error {max_error}");
        }
    }
}

#[test]
fn real_worker_builds_incrementally_searches_and_ignores_deleted_photos() {
    if !model::default_dir().join("model.safetensors").exists() {
        lightcraft_testkit::skip!(Model, "set LIGHTCRAFT_TEXT_MODEL to the pinned BGE model directory");
        return;
    }
    let dir = temp("real-worker");
    let mut app = LightcraftApp::new(Session::with_demo(), Services::default());
    let ids: Vec<_> = app.session.catalog.photos().take(2).map(|p| p.id).collect();
    let descriptions = ["一只白色小狗在草地上奔跑。", "蓝色海水拍打沙滩，远处停着帆船。"];
    for (id, caption) in ids.iter().zip(descriptions) {
        app.run("photo.setMeta", serde_json::json!({"ids":[id.0], "caption":caption})).unwrap();
    }
    let photos: Vec<_> = ids.iter().map(|id| app.session.catalog.photo(*id).unwrap().clone()).collect();
    let library = worker::Library { generation: 87654321, dir: Some(dir.clone()) };
    let view = super::lifecycle::View::new(library, &app.session.catalog).unwrap();
    let submit = |operation| {
        let guard = app.session.activity.start("semanticTest", "Indexing photo descriptions", Cancel::No);
        worker::submit(view.clone(), operation, Arc::new(AtomicBool::new(false)), guard)
            .unwrap()
            .recv_timeout(std::time::Duration::from_secs(30))
            .unwrap()
            .unwrap()
    };
    let first = submit(worker::Operation::Index(photos.clone()));
    assert_eq!((first.built, first.indexed), (2, 2));
    let second = submit(worker::Operation::Index(photos.clone()));
    assert_eq!((second.built, second.indexed), (0, 2));
    let found = submit(worker::Operation::Search { photos: photos.clone(), query: "在户外跑动的宠物".into(), minimum: 0.4 });
    assert_eq!(found.hits.first().unwrap().id, ids[0].0);
    assert!(found.hits.first().unwrap().semantic.is_some());
    let remaining = vec![photos[1].clone()];
    let removed = submit(worker::Operation::Search { photos: remaining, query: "在户外跑动的宠物".into(), minimum: 0.4 });
    assert!(removed.hits.iter().all(|h| h.id != ids[0].0));
    let saved = store::Index::load(Some(&dir)).unwrap();
    assert_eq!(saved.records.len(), 2);
    std::fs::remove_dir_all(dir).unwrap();
}
