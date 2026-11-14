//! Lightweight, library-local description search. New features stay inside the extension.
#[cfg(not(target_arch = "wasm32"))]
mod gpu;
mod lifecycle;
pub mod model;
pub(crate) mod store;
mod worker;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, TryRecvError};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock, PoisonError};
use std::time::Duration;

use egui::{RichText, Ui, vec2};
use lightcraft_catalog::PhotoId;
use lightcraft_engine::activity::Cancel;
use lightcraft_ui_egui::button::Button;
use lightcraft_ui_egui::icons::Icon;
use lightcraft_ui_egui::text_field::{Ending, TextField};
use lightcraft_ui_egui::theme::Tokens;
use lightcraft_ui_egui::{LightcraftApp, i18n, widgets};

use crate::ui::{
    Pill, action_bar, action_note, card, card_head, hint, photo_thumb, primary_action, secondary_action, split_page, status_pill, trn, warning,
};
use store::Description;
use worker::{Library, Operation, Reply};

pub(crate) const PAGE_ID: &str = "semantic-search";
type Answer = Receiver<Result<Reply, String>>;

#[derive(Default)]
struct State {
    library: Option<Library>,
    query: String,
    minimum: f32,
    pending: Option<Answer>,
    cancel: Arc<AtomicBool>,
    reply: Reply,
    error: Option<String>,
    search_revision: Option<u64>,
    searching: bool,
    searched: bool,
    advanced: bool,
}

static STATE: OnceLock<Mutex<State>> = OnceLock::new();
static SAVES: OnceLock<Mutex<Vec<(Library, Answer)>>> = OnceLock::new();
static TRACKER: OnceLock<Mutex<lifecycle::Tracker>> = OnceLock::new();

fn tracker() -> MutexGuard<'static, lifecycle::Tracker> {
    TRACKER.get_or_init(|| Mutex::new(lifecycle::Tracker::default())).lock().unwrap_or_else(PoisonError::into_inner)
}

fn observe_view(app: &LightcraftApp) -> Result<lifecycle::View, String> {
    tracker().observe(library(app), &app.session.catalog)
}

pub(crate) fn observe(app: &LightcraftApp) {
    let _ = observe_view(app);
}
fn state() -> MutexGuard<'static, State> {
    STATE.get_or_init(|| Mutex::new(State { minimum: 0.4, ..State::default() })).lock().unwrap_or_else(PoisonError::into_inner)
}

fn library(app: &LightcraftApp) -> Library {
    Library { generation: app.session.library_generation(), dir: app.session.library.as_ref().filter(|l| l.on_disk).map(|l| l.dir.clone()) }
}

/// Accepted AI descriptions are derived cache data. Their accepted keyword guards ensure
/// undo/removal invalidates them without touching the user's own caption or catalog schema.
pub(crate) fn save_descriptions(app: &LightcraftApp, descriptions: Vec<Description>) -> Result<(), String> {
    if descriptions.is_empty() {
        return Ok(());
    }
    let key = library(app);
    let view = observe_view(app)?;
    let cancel = Arc::new(AtomicBool::new(false));
    let guard = app.session.activity.start("semanticDescriptions", "Saving photo descriptions", Cancel::No);
    let rx = worker::submit(view, Operation::Save(descriptions), cancel, guard)?;
    SAVES.get_or_init(|| Mutex::new(Vec::new())).lock().unwrap_or_else(PoisonError::into_inner).push((key, rx));
    Ok(())
}

/// Runs even with Extensions closed: observe catalog changes, reconcile caches and retry failures.
pub(crate) fn poll(app: &mut LightcraftApp, ctx: &egui::Context) {
    let (busy, notice) = {
        let mut tracker = tracker();
        match tracker.observe(library(app), &app.session.catalog) {
            Ok(view) => tracker.poll(app, view),
            Err(error) => (true, tracker.failed(error)),
        }
    };
    if let Some(error) = notice {
        app.toast(ctx, format!("{}: {error}", i18n::tr("Photo descriptions could not be saved")));
    }
    if busy {
        ctx.request_repaint_after(Duration::from_millis(100));
    }
    let mut saves = SAVES.get_or_init(|| Mutex::new(Vec::new())).lock().unwrap_or_else(PoisonError::into_inner);
    saves.retain(|(key, rx)| match rx.try_recv() {
        Ok(Err(error)) => {
            if *key == library(app) {
                tracker().failed(error.clone());
                app.toast(ctx, format!("{}: {error}", i18n::tr("Photo descriptions could not be saved")));
            }
            false
        }
        Ok(Ok(_)) => false,
        Err(TryRecvError::Empty) => true,
        Err(TryRecvError::Disconnected) => {
            app.toast(ctx, i18n::tr("Photo descriptions could not be saved"));
            false
        }
    });
    if !saves.is_empty() {
        ctx.request_repaint_after(Duration::from_millis(100));
    }
}

fn snapshot(app: &LightcraftApp) -> Result<Vec<Arc<lightcraft_catalog::Photo>>, String> {
    let mut photos = Vec::new();
    for photo in app.session.catalog.photos().filter(|p| p.in_library() && !p.is_video()) {
        if photos.len() >= store::MAX_PHOTOS {
            return Err("This search index supports at most 100,000 photos.".into());
        }
        photos.push(photo.clone());
    }
    Ok(photos)
}

fn start(app: &LightcraftApp, st: &mut State, operation: Operation) {
    st.error = None;
    let view = match observe_view(app) {
        Ok(view) => view,
        Err(error) => {
            st.error = Some(error);
            return;
        }
    };
    st.searching = matches!(operation, Operation::Search { .. });
    st.searched = false;
    st.search_revision = st.searching.then_some(app.session.catalog.revision);
    st.cancel.store(true, Ordering::Relaxed);
    st.cancel = Arc::new(AtomicBool::new(false));
    let (kind, label) = match &operation {
        Operation::Download => ("semanticModel", "Downloading the text search model"),
        Operation::Index(_) => ("semanticIndex", "Indexing photo descriptions"),
        Operation::Search { .. } => ("semanticSearch", "Searching photo descriptions"),
        _ => ("semanticLoad", "Loading the photo search index"),
    };
    let guard = app.session.activity.start(kind, label, Cancel::Flag(st.cancel.clone()));
    match worker::submit(view, operation, st.cancel.clone(), guard) {
        Ok(rx) => st.pending = Some(rx),
        Err(error) => st.error = Some(error),
    }
}

pub(crate) fn page(app: &mut LightcraftApp, ui: &mut Ui) {
    let mut st = state();
    let key = library(app);
    if st.library.as_ref() != Some(&key) {
        st.cancel.store(true, Ordering::Relaxed);
        *st = State { library: Some(key), minimum: 0.4, ..State::default() };
        start(app, &mut st, Operation::Status);
    }
    if let Some(rx) = &st.pending {
        match rx.try_recv() {
            Ok(Ok(reply)) => {
                st.reply = reply;
                st.searched = st.searching;
                st.pending = None;
            }
            Ok(Err(error)) => {
                if error != "Cancelled" {
                    st.error = Some(error);
                }
                st.pending = None;
            }
            Err(TryRecvError::Disconnected) => {
                st.error = Some("The search worker is unavailable.".into());
                st.pending = None;
            }
            Err(TryRecvError::Empty) => {}
        }
    }
    if st.pending.is_some() {
        ui.ctx().request_repaint_after(Duration::from_millis(100));
    }
    // Restores and removals change both candidates and BM25 statistics. A response to an old
    // revision cannot pin the page to its previous catalog snapshot.
    if st.search_revision.is_some_and(|revision| revision != app.session.catalog.revision) && (st.pending.is_none() || st.searching) {
        match snapshot(app) {
            Ok(photos) => {
                let operation = Operation::Search { photos, query: st.query.trim().to_string(), minimum: st.minimum };
                start(app, &mut st, operation);
            }
            Err(error) => st.error = Some(error),
        }
    }
    widgets::register(ui, format!("page:{PAGE_ID}"), ui.max_rect());
    draw(app, ui, &mut st);
}

fn draw(app: &mut LightcraftApp, ui: &mut Ui, st: &mut State) {
    let (body, footer) = split_page(ui, 66.0);
    let mut body_ui = ui.new_child(egui::UiBuilder::new().max_rect(body));
    egui::ScrollArea::vertical().id_salt("semantic-scroll").auto_shrink([false, false]).show(&mut body_ui, |ui| {
        egui::Frame::NONE.inner_margin(egui::Margin { left: 12, right: 12, top: 10, bottom: 12 }).show(ui, |ui| {
            ui.set_width(ui.available_width());
            draw_query(app, ui, st);
            if let Some(error) = &st.error {
                warning(ui, &format!("{}: {}", i18n::tr("Search error"), i18n::tr(error)));
            }
            if let Some(note) = &st.reply.warning {
                warning(ui, i18n::tr(note));
            }
            if st.searched {
                draw_results(app, ui, st);
            }
            // A very short dock cannot reserve a footer; keep the action reachable by scrolling.
            if footer.is_none() {
                draw_action(app, ui, st);
            }
        });
    });
    if let Some(bar) = footer {
        let mut bar_ui = ui.new_child(egui::UiBuilder::new().max_rect(bar));
        action_bar(&mut bar_ui, |ui| draw_action(app, ui, st));
    }
}

fn submit_search(app: &LightcraftApp, st: &mut State) {
    if st.query.trim().is_empty() || st.pending.is_some() {
        return;
    }
    match snapshot(app) {
        Ok(photos) => start(app, st, Operation::Search { photos, query: st.query.trim().to_string(), minimum: st.minimum }),
        Err(error) => st.error = Some(error),
    }
}

fn draw_action(app: &LightcraftApp, ui: &mut Ui, st: &mut State) {
    if st.pending.is_some() {
        if secondary_action(ui, i18n::tr("Stop"), true, "semantic:stop") {
            st.cancel.store(true, Ordering::Relaxed);
        }
        action_note(ui, i18n::tr("The task continues if the panel closes."));
    } else {
        if primary_action(ui, i18n::tr("Search"), !st.query.trim().is_empty(), "semantic:search") {
            submit_search(app, st);
        }
        action_note(ui, i18n::tr("Search the entire library."));
    }
}

fn draw_query(app: &LightcraftApp, ui: &mut Ui, st: &mut State) {
    card(ui, |ui| {
        card_head(ui, Icon::Search, "Search photos", |ui| {
            if st.pending.is_some() {
                status_pill(ui, "Working…", Pill::Busy);
            }
        });
        let query = TextField::singleline("semantic:query", &mut st.query)
            .hint(i18n::tr("For example: a dog running on the grass"))
            .width(ui.available_width())
            .show(ui);
        if query.response.changed() {
            st.search_revision = None;
            st.cancel.store(true, Ordering::Relaxed);
            st.pending = None;
            st.reply.hits.clear();
            st.reply.warning = None;
            st.error = None;
            st.searched = false;
        }
        if query.ending == Some(Ending::Return) {
            submit_search(app, st);
        }
        ui.add_space(6.0);
        let chevron = if st.advanced { Icon::ChevronDown } else { Icon::ChevronRight };
        if Button::new(i18n::tr("Advanced")).icon(chevron).ghost().small().widget("semantic:advanced").show(ui).clicked() {
            st.advanced = !st.advanced;
        }
        if st.advanced {
            ui.add_space(6.0);
            ui.separator();
            ui.add_space(4.0);
            draw_advanced(app, ui, st);
        }
    });
}

fn draw_advanced(app: &LightcraftApp, ui: &mut Ui, st: &mut State) {
    let t = Tokens::get(ui.ctx());
    let busy = st.pending.is_some();
    ui.label(RichText::new(i18n::tr("Search index")).font(t.semibold(11.0)).color(t.text_label));
    hint(
        ui,
        &trn(
            "{descriptions} descriptions · {indexed} indexed photos",
            &[("descriptions", st.reply.descriptions.to_string()), ("indexed", st.reply.indexed.to_string())],
        ),
    );
    if secondary_action(ui, i18n::tr("Update search index"), !busy, "semantic:index") {
        match snapshot(app) {
            Ok(photos) => start(app, st, Operation::Index(photos)),
            Err(error) => st.error = Some(error),
        }
    }
    hint(ui, i18n::tr("Accept descriptions in Auto Keywords, then update the index."));
    ui.add_space(8.0);
    ui.label(RichText::new(i18n::tr("Model and matching")).font(t.semibold(11.0)).color(t.text_label));
    if secondary_action(ui, i18n::tr("Download text model (96 MB)"), !busy, "semantic:download") {
        start(app, st, Operation::Download);
    }
    hint(ui, i18n::tr("GPU required. Download once, then use offline."));
    ui.add_space(6.0);
    ui.label(RichText::new(i18n::tr("Minimum semantic similarity")).font(t.font(11.0)).color(t.text_label));
    let slider = ui.add(egui::Slider::new(&mut st.minimum, 0.3..=0.95).fixed_decimals(2));
    widgets::register(ui, "semantic:minimum", slider.rect);
    hint(ui, i18n::tr("Higher values return fewer semantic matches."));
    if st.reply.built > 0 || st.reply.skipped > 0 || st.reply.elapsed_ms > 0 {
        ui.add_space(8.0);
        hint(
            ui,
            &trn(
                "{built} updated · {skipped} skipped · {ms} ms",
                &[("built", st.reply.built.to_string()), ("skipped", st.reply.skipped.to_string()), ("ms", st.reply.elapsed_ms.to_string())],
            ),
        );
    }
}

fn draw_results(app: &mut LightcraftApp, ui: &mut Ui, st: &mut State) {
    let t = Tokens::get(ui.ctx());
    let hits: Vec<_> = st.reply.hits.iter().filter(|h| app.session.catalog.photo(PhotoId(h.id)).is_some_and(|p| p.in_library())).cloned().collect();
    card(ui, |ui| {
        card_head(ui, Icon::GridSquare, "Search results", |ui| {
            hint(ui, &hits.len().to_string());
        });
        if hits.is_empty() {
            hint(ui, i18n::tr("No matching photos. Try a different description."));
            return;
        }
        if secondary_action(ui, i18n::tr("Show results in library"), true, "semantic:show") {
            let ids: Vec<_> = hits.iter().map(|h| h.id).collect();
            if let Err(error) = app.run("library.filter", serde_json::json!({"only": ids, "onlyNone": false})) {
                st.error = Some(error);
            }
        }
        ui.add_space(8.0);
        let mut budget = 3;
        for (index, hit) in hits.iter().enumerate() {
            if index > 0 {
                ui.separator();
                ui.add_space(4.0);
            }
            let Some(photo) = app.session.catalog.photo(PhotoId(hit.id)) else { continue };
            let name = photo.file_name.clone();
            ui.horizontal(|ui| {
                let (rect, _) = ui.allocate_exact_size(vec2(56.0, 40.0), egui::Sense::hover());
                photo_thumb(app, ui, rect, hit.id, &mut budget);
                ui.vertical(|ui| {
                    let response = ui.add(
                        egui::Label::new(RichText::new(&name).font(t.semibold(11.0)).color(t.text_label)).truncate().sense(egui::Sense::click()),
                    );
                    widgets::register(ui, format!("semantic:photo:{}", hit.id), response.rect);
                    let selected = response.clicked()
                        || (response.has_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter) || i.key_pressed(egui::Key::Space)));
                    response.on_hover_text(&name);
                    if selected && let Err(error) = app.run("library.select", serde_json::json!({"ids": [hit.id], "active": hit.id})) {
                        st.error = Some(error);
                    }
                    let response = ui.add(egui::Label::new(RichText::new(&hit.text).font(t.font(10.5)).color(t.text_dim)).truncate());
                    response.on_hover_ui(|ui| {
                        ui.label(&hit.text);
                        if let Some(similarity) = hit.semantic {
                            hint(ui, &format!("{}: {similarity:.3}", i18n::tr("Semantic similarity")));
                        }
                    });
                });
            });
            ui.add_space(4.0);
        }
    });
}

#[cfg(test)]
mod tests;
