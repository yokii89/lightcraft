//! Small drawing helpers shared by the extension pages: the mock's card, the section and field
//! captions, the hint and warning lines, the `{name}` placeholder translation and the seconds
//! format. Everything here draws in the native look ([`Tokens`]) and registers nothing by
//! itself — controls register their own ids.

use egui::{RichText, Ui};
use lightcraft_ui_egui::i18n;
use lightcraft_ui_egui::theme::Tokens;

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

/// A card like the mock's: an inset panel with a border, drawn above a little gap.
pub fn card(ui: &mut Ui, add: impl FnOnce(&mut Ui)) {
    let t = Tokens::get(ui.ctx());
    egui::Frame::NONE
        .fill(t.inset)
        .stroke(egui::Stroke::new(1.0, t.divider))
        .corner_radius(5)
        .inner_margin(egui::Margin::same(10))
        .outer_margin(egui::Margin { left: 0, right: 0, top: 0, bottom: 10 })
        .show(ui, |ui| {
            ui.set_width(ui.available_width());
            add(ui);
        });
}

/// A section's small header (§10: 模型服务 / 范围 / 关键字 / 写入方式).
pub fn section(ui: &mut Ui, title: &str) {
    let t = Tokens::get(ui.ctx());
    ui.add_space(2.0);
    ui.label(RichText::new(i18n::tr(title)).font(t.semibold(11.0)).color(t.text_label));
    ui.add_space(4.0);
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

pub fn status_dot(ui: &mut Ui, color: egui::Color32) {
    let (rect, _) = ui.allocate_exact_size(egui::vec2(10.0, 10.0), egui::Sense::hover());
    ui.painter().circle_filled(rect.center(), 3.5, color);
}
