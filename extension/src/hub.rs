//! The Extensions panel's one page, in two levels (design: `extension/docs/04-扩展页面视觉设计.md`):
//!
//! * Level 1 — a short header and the feature list, one illustrated card each. Clicking a card
//!   opens the feature.
//! * Level 2 — the feature itself (Auto Keywords, Geographic classification), under a fixed
//!   breadcrumb header: the feature scrolls, the header stays.
//!
//! The navigation (which feature is open) lives in a static mutex, so it survives the panel being
//! closed and reopened, exactly like the feature states it points into. The native
//! [`ExtensionPage`] machinery only ever sees this one page: features come and go here, and no
//! native edit follows them.
//!
//! [`ExtensionPage`]: lightcraft_ui_egui::ext::ExtensionPage

use std::sync::{Mutex, MutexGuard, PoisonError};

use egui::{Align2, Color32, Rect, RichText, Sense, Shape, Stroke, StrokeKind, Ui, pos2, vec2};
use lightcraft_ui_egui::button::Button;
use lightcraft_ui_egui::icons::{self, Icon};
use lightcraft_ui_egui::theme::{Tokens, current};
use lightcraft_ui_egui::widgets;
use lightcraft_ui_egui::{LightcraftApp, i18n};

use crate::auto_keywords;
use crate::geo_classify;
use crate::ui::with_alpha;

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
    pub render: fn(&mut LightcraftApp, &mut Ui),
}

/// The features of this extension crate, in display order.
pub(crate) fn features() -> Vec<Feature> {
    vec![
        Feature {
            id: auto_keywords::PAGE_ID,
            title: "Auto Keywords",
            description: "Generate keywords with a local vision model.",
            render: auto_keywords::page,
        },
        Feature {
            id: geo_classify::PAGE_ID,
            title: "Classify by Location",
            description: "Group photos by where they were taken and write the place as a keyword.",
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

/// Level 1: the header and the feature cards.
fn draw_list(ui: &mut Ui) {
    widgets::register(ui, format!("page:{PAGE_ID}"), ui.max_rect());
    egui::ScrollArea::vertical().id_salt("ext-hub-scroll").auto_shrink([false, false]).show(ui, |ui| {
        egui::Frame::NONE.inner_margin(egui::Margin { left: 12, right: 12, top: 10, bottom: 14 }).show(ui, |ui| {
            ui.set_width(ui.available_width());
            hub_hero(ui);
            ui.add_space(12.0);
            section_line(ui, "Features", features().len());
            ui.add_space(8.0);
            for feature in features() {
                feature_card(ui, &feature);
            }
            ui.add_space(2.0);
            hub_foot(ui);
        });
    });
}

/// The header: the extension mark, the promise and the privacy line, over the theme's inset with a
/// soft accent glow and the prototype's fine diagonal hatch.
fn hub_hero(ui: &mut Ui) {
    let t = Tokens::get(ui.ctx());
    let dark = current(ui.ctx()).unwrap_or_default().is_dark();
    let width = ui.available_width();
    let subtitle = i18n::tr("Smart tools for your library, all running on this computer.");
    let subtitle_galley = ui.painter().layout(subtitle.to_string(), t.font(10.5), t.text_dim, (width - 63.0).max(40.0));
    let height = (31.0 + subtitle_galley.size().y + 10.0 + 14.0 + 10.0).max(86.0);
    let (rect, _) = ui.allocate_exact_size(vec2(width, height), Sense::hover());
    let p = ui.painter().with_clip_rect(rect);
    p.rect_filled(rect, 8.0, t.inset);
    let height = rect.height();
    let mut x = rect.left() - height;
    while x < rect.right() {
        p.line_segment([pos2(x, rect.bottom()), pos2(x + height, rect.top())], Stroke::new(1.0, with_alpha(t.text, 6)));
        x += 7.0;
    }
    p.circle_filled(pos2(rect.right() - 18.0, rect.top() - 6.0), 58.0, with_alpha(t.accent, 30));
    p.rect_stroke(rect, 8.0, Stroke::new(1.0, t.field_border.gamma_multiply(0.9)), StrokeKind::Inside);
    let mark = Rect::from_min_size(pos2(rect.left() + 12.0, rect.top() + 12.0), vec2(30.0, 30.0));
    p.rect_filled(mark, 8.0, t.accent);
    p.circle_filled(pos2(mark.left() + 7.0, mark.top() + 7.0), 12.0, with_alpha(Color32::WHITE, 26));
    icons::paint(&p, mark.shrink(7.0), Icon::Extension, Color32::WHITE);
    let ink = accent_ink(&t, dark);
    let title = p.layout_no_wrap(i18n::tr("Extensions").to_string(), t.semibold(15.0), t.text);
    p.galley(pos2(rect.left() + 51.0, rect.top() + 11.0), title.clone(), t.text);
    let tag = p.layout_no_wrap(i18n::tr("On this device").to_string(), t.font(9.0), ink);
    let tag_rect = Rect::from_min_size(pos2(rect.left() + 57.0 + title.size().x, rect.top() + 13.0), vec2(tag.size().x + 10.0, 17.0));
    p.rect_filled(tag_rect, 3.0, with_alpha(t.accent, 30));
    p.rect_stroke(tag_rect, 3.0, Stroke::new(1.0, with_alpha(t.accent, 85)), StrokeKind::Inside);
    p.galley(pos2(tag_rect.left() + 5.0, tag_rect.center().y - tag.size().y / 2.0), tag, ink);
    p.galley(pos2(rect.left() + 51.0, rect.top() + 31.0), subtitle_galley.clone(), t.text_dim);
    let privacy_y = rect.top() + 31.0 + subtitle_galley.size().y + 9.0;
    let shield = Rect::from_min_size(pos2(rect.left() + 13.0, privacy_y), vec2(11.0, 12.0));
    paint_shield(&p, shield, t.accent);
    p.text(pos2(rect.left() + 29.0, privacy_y - 1.0), Align2::LEFT_TOP, i18n::tr("Your photos never leave this computer."), t.font(10.0), t.text_dim);
}

/// The "功能 — 2" rule between the header and the cards.
fn section_line(ui: &mut Ui, title: &str, count: usize) {
    let t = Tokens::get(ui.ctx());
    let (rect, _) = ui.allocate_exact_size(vec2(ui.available_width(), 16.0), Sense::hover());
    let p = ui.painter();
    let label = p.layout_no_wrap(i18n::tr(title).to_string(), t.semibold(10.0), t.text_dim);
    p.galley(pos2(rect.left(), rect.center().y - label.size().y / 2.0), label.clone(), t.text_dim);
    let line_start = rect.left() + label.size().x + 8.0;
    p.line_segment(
        [pos2(line_start, rect.center().y), pos2(rect.right() - 18.0, rect.center().y)],
        Stroke::new(1.0, t.field_border.gamma_multiply(0.7)),
    );
    p.text(pos2(rect.right(), rect.center().y), Align2::RIGHT_CENTER, count.to_string(), t.font(9.5), t.text_dim);
}

/// The two short attribute chips under a feature's description.
fn feature_chips(id: &str) -> [&'static str; 2] {
    if id == auto_keywords::PAGE_ID { ["On this device", "Review first"] } else { ["Offline names", "One undo"] }
}

/// One feature's card in the list: the illustration, title, description, chips and chevron — the
/// whole row is the click.
fn feature_card(ui: &mut Ui, feature: &Feature) {
    let t = Tokens::get(ui.ctx());
    let dark = current(ui.ctx()).unwrap_or_default().is_dark();
    let width = ui.available_width();
    let title = i18n::tr(feature.title).to_string();
    let description = i18n::tr(feature.description).to_string();
    let art = 56.0;
    let pad = 10.0;
    let text_x = pad + art + 10.0;
    let text_w = (width - text_x - pad - 30.0).max(56.0);
    let galley = ui.painter().layout(description, t.font(10.5), t.text_dim, text_w);
    let height = (pad + 16.0 + 3.0 + galley.size().y + 6.0 + 18.0 + pad).max(art + 2.0 * pad);
    let (rect, _) = ui.allocate_exact_size(vec2(width, height), Sense::hover());
    let response = ui.interact(rect, ui.scope_id().with(("ext-feature-card", feature.id)), Sense::click());
    widgets::register(ui, format!("ext-feature:{}", feature.id), rect);
    let activated = response.clicked() || (response.has_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter) || i.key_pressed(egui::Key::Space)));
    let hovered = response.hovered();
    let p = ui.painter();
    let fill = if hovered { t.inset.lerp_to_gamma(t.hover, 0.35) } else { t.inset };
    p.rect_filled(rect, 7.0, fill);
    p.rect_stroke(rect, 7.0, Stroke::new(1.0, if hovered { t.button_border } else { t.field_border }), StrokeKind::Inside);
    if response.has_focus() {
        p.rect_stroke(rect.expand(2.0), 9.0, Stroke::new(2.0, t.accent), StrokeKind::Outside);
    }
    let art_rect = Rect::from_center_size(pos2(rect.left() + pad + art / 2.0, rect.center().y), vec2(art, art));
    paint_art(p, art_rect, &t, feature.id, dark);
    p.text(pos2(rect.left() + text_x, rect.top() + pad), Align2::LEFT_TOP, title.clone(), t.semibold(12.5), t.text);
    let desc_y = rect.top() + pad + 19.0;
    p.galley(pos2(rect.left() + text_x, desc_y), galley.clone(), t.text_dim);
    let chips_y = desc_y + galley.size().y + 6.0;
    let mut chip_x = rect.left() + text_x;
    for (index, chip) in feature_chips(feature.id).into_iter().enumerate() {
        if index > 0 {
            chip_x += 4.0;
        }
        chip_x += paint_chip(p, pos2(chip_x, chips_y), chip, &t);
    }
    let centre = pos2(rect.right() - pad - 10.0, rect.center().y);
    let circle = Rect::from_center_size(centre, vec2(20.0, 20.0));
    p.circle_filled(centre, 10.0, if hovered { t.accent } else { t.field });
    p.circle_stroke(centre, 10.0, Stroke::new(1.0, if hovered { t.accent } else { t.field_border }));
    icons::paint(p, circle.shrink(5.0), Icon::ChevronRight, if hovered { Color32::WHITE } else { t.text_dim });
    response.widget_info(|| egui::WidgetInfo::labeled(egui::Role::Button, ui.is_enabled(), title.clone()));
    if activated {
        set_open(feature.id);
    }
    ui.add_space(9.0);
}

/// A small outlined attribute chip; returns its width.
fn paint_chip(p: &egui::Painter, at: egui::Pos2, text: &str, t: &Tokens) -> f32 {
    let galley = p.layout_no_wrap(i18n::tr(text).to_string(), t.font(9.5), t.text_dim);
    let width = galley.size().x + 10.0;
    let rect = Rect::from_min_size(at, vec2(width, 18.0));
    p.rect_filled(rect, 3.0, t.field);
    p.rect_stroke(rect, 3.0, Stroke::new(1.0, t.field_border), StrokeKind::Inside);
    p.galley(pos2(rect.left() + 5.0, rect.center().y - galley.size().y / 2.0), galley, t.text_dim);
    width
}

/// The level-1 footer promise.
fn hub_foot(ui: &mut Ui) {
    let t = Tokens::get(ui.ctx());
    ui.horizontal_wrapped(|ui| {
        let (rect, _) = ui.allocate_exact_size(vec2(11.0, 11.0), Sense::hover());
        icons::paint(ui.painter(), rect, Icon::Info, t.text_dim);
        ui.label(RichText::new(i18n::tr("Both tools only add keywords. One undo step reverts a whole batch.")).font(t.font(10.0)).color(t.text_dim));
    });
}

/// A shield with a check, the privacy line's mark.
fn paint_shield(p: &egui::Painter, rect: Rect, color: Color32) {
    let (l, r, top, bottom) = (rect.left(), rect.right(), rect.top(), rect.bottom());
    let mid = rect.center().x;
    let points = vec![
        pos2(mid, top),
        pos2(r, top + rect.height() * 0.2),
        pos2(r, top + rect.height() * 0.55),
        pos2(mid, bottom),
        pos2(l, top + rect.height() * 0.55),
        pos2(l, top + rect.height() * 0.2),
    ];
    p.add(Shape::closed_line(points, Stroke::new(1.0, color)));
    p.line_segment([pos2(l + rect.width() * 0.25, top + rect.height() * 0.45), pos2(mid - 0.5, top + rect.height() * 0.62)], Stroke::new(1.0, color));
    p.line_segment([pos2(mid - 0.5, top + rect.height() * 0.62), pos2(r - rect.width() * 0.2, top + rect.height() * 0.32)], Stroke::new(1.0, color));
}

/// The light version of the accent for small tinted text (the badge's ink).
fn accent_ink(t: &Tokens, dark: bool) -> Color32 {
    if dark { t.accent.lerp_to_gamma(Color32::WHITE, 0.55) } else { t.accent.gamma_multiply(0.85) }
}

// ---------------------------------------------------------------- the card illustrations

fn paint_art(p: &egui::Painter, rect: Rect, t: &Tokens, id: &str, dark: bool) {
    let ak = id == auto_keywords::PAGE_ID;
    let bg = if ak { t.inset.lerp_to_gamma(t.accent, 0.2) } else { t.inset.lerp_to_gamma(t.caution, 0.12) };
    p.rect_filled(rect, 7.0, bg);
    let clip = p.with_clip_rect(rect);
    clip.circle_filled(pos2(rect.right() - 8.0, rect.top() + 4.0), 17.0, with_alpha(if ak { t.accent } else { t.caution }, 38));
    if ak {
        paint_tags(&clip, rect, t, bg);
    } else {
        paint_folded_map(&clip, rect, t, dark);
    }
    p.rect_stroke(rect, 7.0, Stroke::new(1.0, t.field_border), StrokeKind::Inside);
}

/// Two stacked keyword tags and a sparkle.
fn paint_tags(p: &egui::Painter, rect: Rect, t: &Tokens, bg: Color32) {
    let (l, top) = (rect.left(), rect.top());
    let ink = t.inset.lerp_to_gamma(t.accent, 0.55);
    p.add(Shape::convex_polygon(
        vec![pos2(l + 9.0, top + 22.0), pos2(l + 22.0, top + 16.0), pos2(l + 34.0, top + 20.0), pos2(l + 21.0, top + 26.0)],
        ink,
        Stroke::new(1.0, t.accent.gamma_multiply(0.9)),
    ));
    p.add(Shape::convex_polygon(
        vec![pos2(l + 17.0, top + 34.0), pos2(l + 30.0, top + 28.0), pos2(l + 42.0, top + 32.0), pos2(l + 29.0, top + 38.0)],
        t.inset.lerp_to_gamma(t.accent, 0.35),
        Stroke::new(1.0, t.accent.gamma_multiply(0.9)),
    ));
    p.circle_filled(pos2(l + 14.5, top + 22.5), 1.3, bg);
    p.circle_filled(pos2(l + 23.0, top + 32.0), 1.3, bg);
    let (sx, sy) = (l + 39.0, top + 40.0);
    p.add(Shape::convex_polygon(
        vec![
            pos2(sx, sy - 5.0),
            pos2(sx + 1.4, sy - 1.4),
            pos2(sx + 5.0, sy),
            pos2(sx + 1.4, sy + 1.4),
            pos2(sx, sy + 5.0),
            pos2(sx - 1.4, sy + 1.4),
            pos2(sx - 5.0, sy),
            pos2(sx - 1.4, sy - 1.4),
        ],
        t.text,
        Stroke::NONE,
    ));
}

/// A folded map with a dashed route and an amber pin.
fn paint_folded_map(p: &egui::Painter, rect: Rect, t: &Tokens, dark: bool) {
    let (l, top) = (rect.left(), rect.top());
    let panel = t.inset.lerp_to_gamma(t.text_dim, if dark { 0.18 } else { 0.1 });
    let edge = Stroke::new(1.0, t.text_dim.gamma_multiply(0.8));
    p.add(Shape::convex_polygon(
        vec![pos2(l + 7.0, top + 16.0), pos2(l + 22.0, top + 13.0), pos2(l + 22.0, top + 43.0), pos2(l + 7.0, top + 46.0)],
        panel,
        edge,
    ));
    p.add(Shape::convex_polygon(
        vec![pos2(l + 22.0, top + 13.0), pos2(l + 35.0, top + 17.0), pos2(l + 35.0, top + 47.0), pos2(l + 22.0, top + 43.0)],
        panel.lerp_to_gamma(t.text_dim, 0.08),
        edge,
    ));
    p.add(Shape::convex_polygon(
        vec![pos2(l + 35.0, top + 17.0), pos2(l + 49.0, top + 14.0), pos2(l + 49.0, top + 44.0), pos2(l + 35.0, top + 47.0)],
        panel,
        edge,
    ));
    for step in 0..7 {
        let f = step as f32;
        let from = pos2(l + 12.0 + f * 3.4, top + 38.0 - f * 1.6);
        let to = pos2(l + 12.0 + (f + 0.55) * 3.4, top + 38.0 - (f + 0.55) * 1.6);
        p.line_segment([from, to], Stroke::new(1.2, with_alpha(t.caution, 190)));
    }
    let pin = pos2(l + 39.0, top + 23.0);
    p.circle_filled(pin, 4.8, t.caution);
    p.add(Shape::convex_polygon(
        vec![pos2(pin.x - 3.4, pin.y + 2.8), pos2(pin.x + 3.4, pin.y + 2.8), pos2(pin.x, pin.y + 8.0)],
        t.caution,
        Stroke::NONE,
    ));
    p.circle_filled(pin, 1.5, t.inset);
}

/// Level 2: the fixed breadcrumb header, then the feature (which scrolls and holds its own action
/// bar).
fn draw_feature(app: &mut LightcraftApp, ui: &mut Ui, id: &'static str) {
    let Some(feature) = features().into_iter().find(|f| f.id == id) else {
        close();
        return;
    };
    let t = Tokens::get(ui.ctx());
    let width = ui.available_width();
    let subtitle = i18n::tr(feature.description).to_string();
    let wrap = (width - 38.0 - 12.0).max(40.0);
    let subtitle_galley = ui.painter().layout(subtitle, t.font(10.5), t.text_dim, wrap);
    let height = 34.0 + subtitle_galley.size().y + 8.0;
    let (rect, _) = ui.allocate_exact_size(vec2(width, height), Sense::hover());
    ui.painter().line_segment([pos2(rect.left(), rect.bottom() - 0.5), pos2(rect.right(), rect.bottom() - 0.5)], Stroke::new(1.0, t.divider));
    let back = Rect::from_center_size(pos2(rect.left() + 22.0, rect.top() + 19.0), vec2(24.0, 24.0));
    if Button::icon_only(Icon::ChevronLeft, "Back to the extension list").ghost().small().widget("ext:back").show_at(ui, back).clicked() {
        close();
        return;
    }
    let p = ui.painter();
    let x = back.right() + 4.0;
    let crumb = p.layout_no_wrap(i18n::tr("Extensions").to_string(), t.font(11.5), t.text_dim);
    p.galley(pos2(x, back.center().y - crumb.size().y / 2.0), crumb.clone(), t.text_dim);
    let arrow_x = x + crumb.size().x + 6.0;
    p.text(pos2(arrow_x, back.center().y), Align2::LEFT_CENTER, "›", t.font(11.5), t.text_disabled);
    p.text(pos2(arrow_x + 10.0, back.center().y), Align2::LEFT_CENTER, i18n::tr(feature.title), t.semibold(11.5), t.text_label);
    p.galley(pos2(x, rect.top() + 32.0), subtitle_galley, t.text_dim);
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
        assert!(text.contains("Place names"), "the feature page drew under the breadcrumb: {text}");
        let registered: Vec<String> = lightcraft_ui_egui::widgets::take_registry(&view.ctx).into_iter().map(|(id, _)| id).collect();
        assert!(registered.iter().any(|id| id == "ext:back"), "the breadcrumb's back button: {registered:?}");
        assert!(!registered.iter().any(|id| id == "ext-feature:geo-classify"), "the list is not drawn on level 2");
        super::close();
    }
}
