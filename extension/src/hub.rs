//! The Extensions panel's one page, in two levels (design: `extension/docs/01-扩展两级导航与地理位置分类设计.md`):
//!
//! * Level 1 — the feature list: what the host's extension crate adds, one card each. Clicking a
//!   card opens the feature.
//! * Level 2 — the feature itself (Auto Keywords, Geographic classification), which keeps its own
//!   state and draws its own title. A breadcrumb row above it leads back to the list.
//!
//! The navigation (which feature is open) lives in a static mutex, so it survives the panel being
//! closed and reopened, exactly like the feature states it points into. The native
//! [`ExtensionPage`] machinery only ever sees this one page: features come and go here, and no
//! native edit follows them.
//!
//! [`ExtensionPage`]: lightcraft_ui_egui::ext::ExtensionPage

use std::sync::{Mutex, MutexGuard, PoisonError};

use egui::{Align2, RichText, Ui, pos2, vec2};
use lightcraft_ui_egui::button::Button;
use lightcraft_ui_egui::icons::{self, Icon};
use lightcraft_ui_egui::theme::Tokens;
use lightcraft_ui_egui::widgets;
use lightcraft_ui_egui::{LightcraftApp, i18n};

use crate::auto_keywords;
use crate::geo_classify;
use crate::ui::card;

/// The hub's page id (the one [`ExtensionPage`] the host renders).
pub(crate) const PAGE_ID: &str = "extensions";

/// One extension feature: a level-2 page of the hub.
pub(crate) struct Feature {
    /// Stable id, unique among the features (also the automation id suffix).
    pub id: &'static str,
    /// The translated title (card and breadcrumb).
    pub title: &'static str,
    /// The translated one-liner the list shows under the title.
    pub description: &'static str,
    pub icon: Icon,
    pub render: fn(&mut LightcraftApp, &mut Ui),
}

/// The features of this extension crate, in display order.
pub(crate) fn features() -> Vec<Feature> {
    vec![
        Feature {
            id: auto_keywords::PAGE_ID,
            title: "Auto Keywords",
            description: "Generate keywords with a local vision model.",
            icon: Icon::Tag,
            render: auto_keywords::page,
        },
        Feature {
            id: geo_classify::PAGE_ID,
            title: "Classify by Location",
            description: "Group photos by where they were taken and write the place as a keyword.",
            icon: Icon::GridPhoto,
            render: geo_classify::page,
        },
    ]
}

/// The feature currently open; `None` is the level-1 list.
static OPEN: Mutex<Option<&'static str>> = Mutex::new(None);

fn open() -> MutexGuard<'static, Option<&'static str>> {
    OPEN.lock().unwrap_or_else(PoisonError::into_inner)
}

pub(crate) fn open_feature() -> Option<&'static str> {
    *open()
}

/// Open a feature (level 2). An unknown id keeps the list.
pub(crate) fn set_open(id: &'static str) {
    if features().iter().any(|f| f.id == id) {
        *open() = Some(id);
    }
}

/// Back to the list (level 1).
pub(crate) fn close() {
    *open() = None;
}

/// The hub page the host renders.
pub(crate) fn page(app: &mut LightcraftApp, ui: &mut Ui) {
    match open_feature() {
        Some(id) => draw_feature(app, ui, id),
        None => draw_list(ui),
    }
}

/// Level 1: the feature list.
fn draw_list(ui: &mut Ui) {
    let t = Tokens::get(ui.ctx());
    widgets::register(ui, format!("page:{PAGE_ID}"), ui.max_rect());
    egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
        ui.add_space(4.0);
        ui.label(RichText::new(i18n::tr("Extensions")).font(t.semibold(14.0)).color(t.text));
        ui.add_space(2.0);
        ui.label(RichText::new(i18n::tr("The library's extension tools. Pick one to open it.")).font(t.font(11.0)).color(t.text_dim));
        ui.add_space(10.0);
        for feature in features() {
            feature_card(ui, &feature);
        }
    });
}

/// The height of one feature card.
const CARD_H: f32 = 48.0;

/// One feature's card in the list: icon, title, one-liner, chevron — the whole row is the click.
fn feature_card(ui: &mut Ui, feature: &Feature) {
    let t = Tokens::get(ui.ctx());
    card(ui, |ui| {
        let (rect, response) = ui.allocate_exact_size(vec2(ui.available_width(), CARD_H), egui::Sense::click());
        if response.hovered() {
            ui.painter().rect_filled(rect, 4.0, t.hover.gamma_multiply(0.5));
        }
        let p = ui.painter();
        let icon_rect = egui::Rect::from_center_size(pos2(rect.left() + 24.0, rect.center().y), vec2(28.0, 28.0));
        p.circle_filled(icon_rect.center(), 14.0, t.accent.gamma_multiply(0.16));
        icons::paint(p, icon_rect.shrink(7.0), feature.icon, t.accent);
        p.text(pos2(rect.left() + 46.0, rect.top() + 9.0), Align2::LEFT_TOP, i18n::tr(feature.title), t.semibold(12.5), t.text);
        p.text(pos2(rect.left() + 46.0, rect.top() + 26.0), Align2::LEFT_TOP, i18n::tr(feature.description), t.font(10.5), t.text_dim);
        let chevron = egui::Rect::from_center_size(pos2(rect.right() - 16.0, rect.center().y), vec2(16.0, 16.0));
        icons::paint(p, chevron, Icon::ChevronRight, t.text_dim);
        widgets::register(ui, format!("ext-feature:{}", feature.id), rect);
        if response.clicked() {
            set_open(feature.id);
        }
    });
}

/// Level 2: the breadcrumb row (back to the list), then the feature itself.
fn draw_feature(app: &mut LightcraftApp, ui: &mut Ui, id: &'static str) {
    let Some(feature) = features().into_iter().find(|f| f.id == id) else {
        close();
        return;
    };
    let t = Tokens::get(ui.ctx());
    ui.horizontal(|ui| {
        if Button::icon_only(Icon::ChevronLeft, "Back to the extension list").ghost().small().widget("ext:back").show(ui).clicked() {
            close();
        }
        ui.label(RichText::new(i18n::tr("Extensions")).font(t.font(11.5)).color(t.text_dim));
        ui.label(RichText::new("›").font(t.font(11.5)).color(t.text_disabled));
        ui.label(RichText::new(i18n::tr(feature.title)).font(t.semibold(12.0)).color(t.text_label));
    });
    ui.separator();
    (feature.render)(app, ui);
}
#[cfg(test)]
mod tests {
    use egui::vec2;
    use lightcraft_engine::Session;
    use lightcraft_ui_egui::{LightcraftApp, Services};

    /// The hub's two levels, in one test: the navigation state is a static shared by every
    /// render, so the levels are exercised serially here (the geo feature renders, since the
    /// Auto Keywords page's state has tests of its own that hold its lock).
    #[test]
    fn the_hub_renders_the_list_and_the_open_feature() {
        // the geo page reads the feature's shared static; keep parallel page tests out
        let _page = crate::geo_classify::page_test_lock();
        let mut app = LightcraftApp::new(Session::with_demo(), Services::default());
        let mut view = lightcraft_ui_egui::headless::HeadlessView::new();
        let input = lightcraft_ui_egui::headless::HeadlessView::raw_input(vec2(360.0, 780.0), 1.0, 0.0, vec![]);

        // level 1: both feature cards
        super::close();
        let _ = view.run(input.clone(), |ui| super::page(&mut app, ui));
        let text = view.painted_text().join(
            "
",
        );
        assert!(text.contains("Auto Keywords"), "the Auto Keywords card: {text}");
        assert!(text.contains("Classify by Location"), "the Geographic classification card: {text}");
        let registered: Vec<String> = lightcraft_ui_egui::widgets::take_registry(&view.ctx).into_iter().map(|(id, _)| id).collect();
        assert!(registered.iter().any(|id| id == "ext-feature:auto-keywords"), "the cards are registered: {registered:?}");
        assert!(registered.iter().any(|id| id == "ext-feature:geo-classify"));

        // level 2: the breadcrumb and the feature under it
        super::set_open("geo-classify");
        let _ = view.run(input, |ui| super::page(&mut app, ui));
        let text = view.painted_text().join(
            "
",
        );
        assert!(text.contains("Classify by Location"), "the open feature's title: {text}");
        assert!(text.contains("Not downloaded"), "the feature page drew under the breadcrumb: {text}");
        let registered: Vec<String> = lightcraft_ui_egui::widgets::take_registry(&view.ctx).into_iter().map(|(id, _)| id).collect();
        assert!(registered.iter().any(|id| id == "ext:back"), "the breadcrumb's back button: {registered:?}");
        assert!(!registered.iter().any(|id| id == "ext-feature:geo-classify"), "the list is not drawn on level 2");
        super::close();
    }
}
