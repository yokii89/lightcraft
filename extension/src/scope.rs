//! What the extension pages share: the photo scope (selected / current view / entire library),
//! its enumeration and its counts, and the segmented scope control both pages draw.

use egui::Ui;
use lightcraft_catalog::PhotoId;
use lightcraft_ui_egui::theme::Tokens;
use lightcraft_ui_egui::widgets;
use lightcraft_ui_egui::{LightcraftApp, i18n};

use crate::ui::trn;

/// Which photos a run or a classification covers (§9 of the Auto Keywords design).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Scope {
    #[default]
    Selected,
    View,
    Library,
}

impl Scope {
    /// Each scope with its widget-id fragment and its translated name, in display order.
    pub const ALL: [(Scope, &'static str, &'static str); 3] =
        [(Scope::Selected, "selected", "Selected photos"), (Scope::View, "view", "Current view"), (Scope::Library, "library", "Entire library")];
}

/// The photos of `scope` in order, videos left out. `skip_tagged` also leaves out photos that
/// already carry keywords; the caller counts those separately (Auto Keywords shows them as
/// 跳过). Returns the queue with display names and how many were skipped.
pub fn enumerate(app: &mut LightcraftApp, scope: Scope, skip_tagged: bool) -> (Vec<(PhotoId, String)>, usize) {
    let ids: Vec<PhotoId> = match scope {
        Scope::Selected => app.session.selected(),
        Scope::View => app.session.visible().to_vec(),
        Scope::Library => app.session.catalog.photos().filter(|p| p.in_library()).map(|p| p.id).collect(),
    };
    let mut entries = Vec::new();
    let mut skipped = 0usize;
    for id in ids {
        let Some(p) = app.session.catalog.photo(id) else { continue };
        if p.is_video() {
            continue;
        }
        if skip_tagged && !p.meta.keywords.is_empty() {
            skipped += 1;
            continue;
        }
        entries.push((id, p.file_name.clone()));
    }
    (entries, skipped)
}

/// The photos in each scope right now: selected, current view, entire library (videos included —
/// the numbers are what the user sees in the sidebar). The library count is cached per catalog
/// revision behind `cache`, which belongs to the calling page's state: a big library is not
/// walked every frame.
pub fn counts(app: &mut LightcraftApp, cache: &mut Option<(u64, usize)>) -> (usize, usize, usize) {
    let selected = app.session.selected().len();
    let view = app.session.visible().len();
    let revision = app.session.catalog.revision;
    let library = match cache {
        Some((rev, n)) if *rev == revision => *n,
        _ => {
            let n = app.session.catalog.photos().filter(|p| p.in_library()).count();
            *cache = Some((revision, n));
            n
        }
    };
    (selected, view, library)
}

/// The number the scope's segment shows for `scope`, from [`counts`]'s three numbers.
pub fn count_of(scope: Scope, counts: (usize, usize, usize)) -> usize {
    match scope {
        Scope::Selected => counts.0,
        Scope::View => counts.1,
        Scope::Library => counts.2,
    }
}

/// The segmented scope control (§10 范围): three equal segments with their live counts, the name
/// on top and the count below. Returns the scope that was clicked, if any.
pub fn selector(ui: &mut Ui, scope: Scope, counts: (usize, usize, usize)) -> Option<Scope> {
    let t = Tokens::get(ui.ctx());
    let width = ui.available_width();
    let height = 40.0;
    let (rect, _) = ui.allocate_exact_size(egui::vec2(width, height), egui::Sense::hover());
    let p = ui.painter();
    p.rect_filled(rect, 5.0, t.field);
    p.rect_stroke(rect, 5.0, egui::Stroke::new(1.0, t.field_border), egui::StrokeKind::Inside);
    let seg_w = (width / 3.0).max(1.0);
    let mut clicked = None;
    for (index, (option, name, label)) in Scope::ALL.into_iter().enumerate() {
        let left = rect.left() + seg_w * index as f32;
        let seg = egui::Rect::from_min_size(egui::pos2(left, rect.top()), egui::vec2(seg_w, height));
        if index > 0 {
            p.line_segment(
                [egui::pos2(seg.left(), rect.top() + 6.0), egui::pos2(seg.left(), rect.bottom() - 6.0)],
                egui::Stroke::new(1.0, t.field_border),
            );
        }
        let response = ui.interact(seg, ui.scope_id().with(("ext-scope", name)), egui::Sense::click());
        let on = scope == option;
        if on {
            p.rect_filled(seg.shrink(1.0), 4.0, t.pressed);
            p.rect_stroke(seg.shrink(1.0), 4.0, egui::Stroke::new(1.0, t.button_border), egui::StrokeKind::Inside);
        } else if response.hovered() {
            p.rect_filled(seg.shrink(1.0), 4.0, t.hover.gamma_multiply(0.5));
        }
        // A narrow panel splits three segments into ~80 pt: the full names do not fit there, so
        // the segments carry their short form and the full name stays in the screen reader.
        let short = match option {
            Scope::Selected => "Selected",
            Scope::View => "View",
            Scope::Library => "Library",
        };
        p.text(
            egui::pos2(seg.center().x, seg.top() + 6.0),
            egui::Align2::CENTER_TOP,
            i18n::tr(short),
            t.font(10.5),
            if on { t.text } else { t.text_dim },
        );
        p.text(
            egui::pos2(seg.center().x, seg.top() + 21.0),
            egui::Align2::CENTER_TOP,
            trn("{n} photos", &[("n", count_of(option, counts).to_string())]),
            t.font(9.5),
            if on { t.text_label } else { t.text_dim },
        );
        widgets::register(ui, format!("ext-scope:{name}"), seg);
        response.widget_info(|| egui::WidgetInfo::selected(egui::Role::RadioButton, true, on, i18n::tr(label)));
        let activated =
            response.clicked() || (response.has_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter) || i.key_pressed(egui::Key::Space)));
        if activated {
            clicked = Some(option);
        }
    }
    clicked
}
