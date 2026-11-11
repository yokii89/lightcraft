//! What the extension pages share: the photo scope (selected / current view / entire library),
//! its enumeration and its counts, and the segmented scope control both pages draw.

use egui::{RichText, Ui};
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

/// The segmented scope control (§10 范围): three segments with their live counts. Returns the
/// scope that was clicked, if any.
pub fn selector(ui: &mut Ui, scope: Scope, counts: (usize, usize, usize)) -> Option<Scope> {
    let t = Tokens::get(ui.ctx());
    let mut clicked = None;
    ui.horizontal(|ui| {
        for (option, name, label) in Scope::ALL {
            let on = scope == option;
            let text = RichText::new(format!("{}\n{}", i18n::tr(label), trn("{n} photos", &[("n", count_of(option, counts).to_string())])))
                .font(t.font(10.5))
                .color(if on { t.text } else { t.text_dim });
            let response = ui.selectable_label(on, text);
            widgets::register(ui, format!("ext-scope:{name}"), response.rect);
            if response.clicked() {
                clicked = Some(option);
            }
        }
    });
    clicked
}
