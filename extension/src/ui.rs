//! Small drawing helpers shared by the extension pages: the card and its head, the status pill,
//! the scope segments, the option cards, the path example, the stepper, the sticky action bar and
//! the success card. Everything draws in the native look ([`Tokens`], so every theme works) and
//! registers its own widget ids where a caller can press it.

use egui::{Align2, Color32, Key, Rect, RichText, Sense, Stroke, StrokeKind, Ui, pos2, vec2};
use lightcraft_catalog::PhotoId;
use lightcraft_ui_egui::button::Button;
use lightcraft_ui_egui::icons::{self, Icon};
use lightcraft_ui_egui::panels::grid::request_thumb;
use lightcraft_ui_egui::theme::{Tokens, current};
use lightcraft_ui_egui::{LightcraftApp, i18n, widgets};

/// Translate a source string with `{name}` placeholders filled in (the placeholders travel in the
/// translation, so the word order is each language's own).
pub fn trn(source: &str, args: &[(&str, String)]) -> String {
    let mut text = i18n::tr(source).to_string();
    for (key, value) in args {
        text = text.replace(&format!("{{{key}}}"), value);
    }
    text
}

/// `2 min 30 s` / `5 s`, translated.
pub fn fmt_seconds(secs: f64) -> String {
    let s = secs.round() as u64;
    if s >= 60 { trn("{m} min {s} s", &[("m", (s / 60).to_string()), ("s", (s % 60).to_string())]) } else { trn("{s} s", &[("s", s.to_string())]) }
}

/// A colour with its own alpha, for the soft fills and glows the design uses.
pub fn with_alpha(color: Color32, alpha: u8) -> Color32 {
    Color32::from_rgba_unmultiplied(color.r(), color.g(), color.b(), alpha)
}

/// The ready/done green, dark enough to read as text on a light theme and light enough on a dark one.
pub fn ok_color(ctx: &egui::Context) -> Color32 {
    if current(ctx).unwrap_or_default().is_dark() { Color32::from_rgb(0x8f, 0xcf, 0x94) } else { Color32::from_rgb(0x2e, 0x7d, 0x32) }
}

/// A card like the mock's: an inset panel with a border, drawn above a little gap.
pub fn card(ui: &mut Ui, add: impl FnOnce(&mut Ui)) {
    let t = Tokens::get(ui.ctx());
    egui::Frame::NONE
        .fill(t.inset)
        .stroke(egui::Stroke::new(1.0, t.field_border.gamma_multiply(0.82)))
        .corner_radius(6)
        .inner_margin(egui::Margin::same(11))
        .outer_margin(egui::Margin { left: 0, right: 0, top: 0, bottom: 10 })
        .show(ui, |ui| {
            ui.set_width(ui.available_width());
            add(ui);
        });
}

/// A card's header: a small icon, the section name and, at the right, whatever the caller puts
/// there (usually a [`status_pill`]).
pub fn card_head(ui: &mut Ui, icon: Icon, title: &str, right: impl FnOnce(&mut Ui)) {
    let t = Tokens::get(ui.ctx());
    ui.horizontal(|ui| {
        let (r, _) = ui.allocate_exact_size(vec2(13.0, 13.0), Sense::hover());
        icons::paint(ui.painter(), r.shrink(1.0), icon, t.text_dim);
        ui.add_space(2.0);
        ui.label(RichText::new(i18n::tr(title)).font(t.semibold(11.0)).color(t.text_label));
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), right);
    });
    ui.add_space(7.0);
}

/// What a status pill says: idle (grey), busy (amber), ready (green) or a neutral accent.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pill {
    Idle,
    Busy,
    Ready,
}

/// A small capsule in a card's head: a dot and a short state, never a sentence.
pub fn status_pill(ui: &mut Ui, text: &str, kind: Pill) {
    let t = Tokens::get(ui.ctx());
    let text = i18n::tr(text).to_string();
    let (color, fill, border) = match kind {
        Pill::Idle => (t.text_dim, t.field, t.field_border),
        Pill::Busy => (t.caution, with_alpha(t.caution, 26), with_alpha(t.caution, 90)),
        Pill::Ready => {
            let ok = ok_color(ui.ctx());
            (ok, with_alpha(ok, 26), with_alpha(ok, 90))
        }
    };
    let galley = ui.painter().layout_no_wrap(text, t.font(9.5), color);
    let width = 24.0 + galley.size().x;
    let (rect, _) = ui.allocate_exact_size(vec2(width, 18.0), Sense::hover());
    let p = ui.painter();
    p.rect_filled(rect, 9.0, fill);
    p.rect_stroke(rect, 9.0, Stroke::new(1.0, border), StrokeKind::Inside);
    p.circle_filled(pos2(rect.left() + 10.0, rect.center().y), 2.5, color);
    p.galley(pos2(rect.left() + 17.0, rect.center().y - galley.size().y / 2.0), galley, color);
}

/// A field row: the caption on the left (the mock's label column), the control taking the rest.
pub fn field(ui: &mut Ui, caption: &str, add: impl FnOnce(&mut Ui)) {
    let t = Tokens::get(ui.ctx());
    ui.horizontal(|ui| {
        ui.add_sized([90.0, 18.0], egui::Label::new(RichText::new(i18n::tr(caption)).font(t.font(11.5)).color(t.text_dim)));
        add(ui);
    });
    ui.add_space(4.0);
}

pub fn hint(ui: &mut Ui, text: &str) {
    let t = Tokens::get(ui.ctx());
    ui.label(RichText::new(text).font(t.font(10.5)).color(t.text_dim));
}

pub fn warning(ui: &mut Ui, text: &str) {
    let t = Tokens::get(ui.ctx());
    ui.label(RichText::new(text).font(t.font(10.5)).color(t.caution));
}

/// The body and the sticky action bar of a page: consumes all of `ui`'s remaining height, leaving
/// `footer_h` at the bottom when there is room for both. `None` means the body has it all.
pub fn split_page(ui: &mut Ui, footer_h: f32) -> (Rect, Option<Rect>) {
    let full = ui.available_rect_before_wrap();
    ui.advance_cursor_after_rect(full);
    if footer_h <= 0.0 || full.height() < footer_h + 48.0 {
        return (full, None);
    }
    let footer = Rect::from_min_max(pos2(full.left(), full.bottom() - footer_h), full.max);
    let body = Rect::from_min_max(full.min, pos2(full.right(), footer.top()));
    (body, Some(footer))
}

/// Draw the sticky action bar in the child `ui` that [`split_page`] returned: the panel chrome,
/// a hairline on top and the caller's button and note inside the padding.
pub fn action_bar(ui: &mut Ui, add: impl FnOnce(&mut Ui)) {
    let rect = ui.max_rect();
    let t = Tokens::get(ui.ctx());
    let p = ui.painter();
    p.rect_filled(rect, 0.0, t.chrome);
    p.line_segment([rect.left_top(), rect.right_top()], Stroke::new(1.0, t.divider));
    let mut inner = ui.new_child(egui::UiBuilder::new().max_rect(rect.shrink2(vec2(12.0, 8.0))));
    add(&mut inner);
}

/// A full-width primary action: the page's one blue button.
pub fn primary_action(ui: &mut Ui, label: &str, enabled: bool, widget: &str) -> bool {
    full_button(ui, label, enabled, widget, true)
}

/// A full-width quiet action (the run's Stop).
pub fn secondary_action(ui: &mut Ui, label: &str, enabled: bool, widget: &str) -> bool {
    full_button(ui, label, enabled, widget, false)
}

fn full_button(ui: &mut Ui, label: &str, enabled: bool, widget: &str, primary: bool) -> bool {
    let width = ui.available_width();
    let (rect, _) = ui.allocate_exact_size(vec2(width, 30.0), Sense::hover());
    let button = if primary { Button::new(label).primary() } else { Button::new(label).secondary() };
    button.enabled(enabled).widget(widget).show_at(ui, rect).clicked()
}

/// The small line under the action button.
pub fn action_note(ui: &mut Ui, text: &str) {
    let t = Tokens::get(ui.ctx());
    ui.add_space(4.0);
    ui.vertical_centered(|ui| {
        ui.label(RichText::new(text).font(t.font(9.5)).color(t.text_dim));
    });
}

/// One radio option card: the whole card is the click; a radio, a title and a description.
pub fn option_card(ui: &mut Ui, selected: bool, title: &str, description: &str, widget: &str) -> bool {
    let t = Tokens::get(ui.ctx());
    let title = i18n::tr(title).to_string();
    let description = i18n::tr(description).to_string();
    let width = ui.available_width();
    let text_w = (width - 42.0).max(24.0);
    let galley = ui.painter().layout(description, t.font(10.5), t.text_dim, text_w);
    let height = (9.0 + 15.0 + 2.0 + galley.size().y + 9.0).max(44.0);
    let (rect, _) = ui.allocate_exact_size(vec2(width, height), Sense::hover());
    let response = ui.interact(rect, ui.scope_id().with(("ext-option", widget)), Sense::click());
    widgets::register(ui, widget, rect);
    let activated = response.clicked() || (response.has_focus() && ui.input(|i| i.key_pressed(Key::Enter) || i.key_pressed(Key::Space)));
    let fill = if selected {
        t.canvas.lerp_to_gamma(t.accent, 0.12)
    } else if response.hovered() {
        t.canvas.lerp_to_gamma(t.hover, 0.35)
    } else {
        t.canvas
    };
    let border = if selected {
        t.accent
    } else if response.hovered() {
        t.button_border
    } else {
        t.field_border
    };
    let p = ui.painter();
    p.rect_filled(rect, 5.0, fill);
    p.rect_stroke(rect, 5.0, Stroke::new(1.0, border), StrokeKind::Inside);
    if response.has_focus() {
        p.rect_stroke(rect.expand(2.0), 7.0, Stroke::new(2.0, t.accent), StrokeKind::Outside);
    }
    let centre = pos2(rect.left() + 16.5, rect.top() + 15.5);
    p.circle_stroke(centre, 6.5, Stroke::new(1.0, if selected { t.accent } else { t.field_border }));
    if selected {
        p.circle_filled(centre, 3.0, t.accent);
    }
    p.text(
        pos2(rect.left() + 29.0, rect.top() + 9.0),
        Align2::LEFT_TOP,
        title.clone(),
        t.semibold(11.5),
        if selected { t.text } else { t.text_label },
    );
    p.galley(pos2(rect.left() + 29.0, rect.top() + 26.0), galley, t.text_dim);
    response.widget_info(|| egui::WidgetInfo::selected(egui::Role::RadioButton, ui.is_enabled(), selected, title.clone()));
    ui.add_space(6.0);
    activated
}

/// A compact number stepper: minus, the value, plus.
pub fn stepper(ui: &mut Ui, value: &mut u32, min: u32, max: u32, prefix: &str) -> bool {
    let t = Tokens::get(ui.ctx());
    let mut changed = false;
    ui.horizontal(|ui| {
        if Button::icon_only(Icon::Minus, "One fewer").ghost().small().widget(format!("{prefix}:minus")).show(ui).clicked() {
            let next = value.saturating_sub(1).max(min);
            if next != *value {
                *value = next;
                changed = true;
            }
        }
        let (rect, _) = ui.allocate_exact_size(vec2(26.0, 24.0), Sense::hover());
        ui.painter().text(rect.center(), Align2::CENTER_CENTER, value.to_string(), t.font(11.5), t.text);
        if Button::icon_only(Icon::Plus, "One more").ghost().small().widget(format!("{prefix}:plus")).show(ui).clicked() {
            let next = value.saturating_add(1).min(max);
            if next != *value {
                *value = next;
                changed = true;
            }
        }
    });
    changed
}

/// The `parent › country › city` example under a keyword field.
pub fn path_example(ui: &mut Ui, parts: &[&str]) {
    let t = Tokens::get(ui.ctx());
    ui.add_space(3.0);
    egui::Frame::NONE
        .fill(t.canvas)
        .stroke(Stroke::new(1.0, t.field_border.gamma_multiply(0.9)))
        .corner_radius(4)
        .inner_margin(egui::Margin::same(7))
        .show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.horizontal_wrapped(|ui| {
                let dim = t.font(9.5);
                ui.label(RichText::new(i18n::tr("Writing example")).font(dim.clone()).color(t.text_dim));
                for (index, part) in parts.iter().enumerate() {
                    if index > 0 {
                        ui.label(RichText::new("›").font(dim.clone()).color(t.text_disabled));
                    }
                    ui.label(RichText::new(*part).font(dim.clone()).color(t.accent).background_color(with_alpha(t.accent, 28)));
                }
            });
        });
    ui.add_space(4.0);
}

/// A photo thumbnail: a skeleton until its texture arrives, then the picture. `budget` keeps one
/// review frame from asking for hundreds of renders at once.
pub fn photo_thumb(app: &mut LightcraftApp, ui: &Ui, rect: Rect, id: u64, budget: &mut u32) {
    let t = Tokens::get(ui.ctx());
    let p = ui.painter();
    p.rect_filled(rect, 3.0, t.skeleton);
    if *budget > 0 {
        request_thumb(app, PhotoId(id), 96, 20);
        *budget -= 1;
    }
    if let Some(texture) = app.renderer.thumb(PhotoId(id)) {
        p.image(texture.id(), rect, Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0)), Color32::WHITE);
    } else {
        icons::paint(p, rect.shrink(8.0), Icon::Photos, t.text_disabled);
    }
}

/// A generic tile for a photo we do not know the texture of (the one being analysed).
pub fn photo_placeholder(ui: &Ui, rect: Rect) {
    let t = Tokens::get(ui.ctx());
    let p = ui.painter();
    p.rect_filled(rect, 3.0, t.skeleton);
    icons::paint(p, rect.shrink(8.0), Icon::Photos, t.text_disabled);
}

/// The done card: a check ring, the outcome and the numbers the user came for.
pub fn success_card(ui: &mut Ui, title: &str, body: &str, stats: &[(&str, String)]) {
    let t = Tokens::get(ui.ctx());
    let ok = ok_color(ui.ctx());
    egui::Frame::NONE
        .fill(t.inset)
        .stroke(Stroke::new(1.0, t.field_border.gamma_multiply(0.82)))
        .corner_radius(7)
        .inner_margin(egui::Margin { left: 12, right: 12, top: 18, bottom: 14 })
        .outer_margin(egui::Margin { left: 0, right: 0, top: 0, bottom: 10 })
        .show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.vertical_centered(|ui| {
                let (ring, _) = ui.allocate_exact_size(vec2(38.0, 38.0), Sense::hover());
                let p = ui.painter();
                p.circle_filled(ring.center(), 19.0, with_alpha(ok, 28));
                p.circle_stroke(ring.center(), 19.0, Stroke::new(1.0, with_alpha(ok, 90)));
                icons::paint(p, ring.shrink(10.0), Icon::Check, ok);
                ui.add_space(7.0);
                ui.label(RichText::new(title).font(t.semibold(13.0)).color(t.text));
                if !body.is_empty() {
                    ui.add_space(4.0);
                    ui.label(RichText::new(body).font(t.font(10.5)).color(t.text_dim));
                }
                if !stats.is_empty() {
                    ui.add_space(10.0);
                    ui.separator();
                    ui.add_space(8.0);
                    ui.horizontal(|ui| {
                        for (index, (label, value)) in stats.iter().enumerate() {
                            if index > 0 {
                                ui.add_space(16.0);
                            }
                            ui.vertical(|ui| {
                                ui.label(RichText::new(value).font(t.semibold(14.0)).color(t.text));
                                ui.label(RichText::new(*label).font(t.font(9.5)).color(t.text_dim));
                            });
                        }
                    });
                }
            });
        });
}
