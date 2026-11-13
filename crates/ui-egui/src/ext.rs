//! Extension pages: the host registers them before the first frame, and this crate only renders
//! them.
//!
//! The desktop host (`apps/lightcraft`) owns the extension crate and hands its pages to
//! [`LightcraftApp::register_extension_page`]. `lightcraft-ui-egui` never depends on it, so the
//! dependency direction stays host → extension → ui-egui.

use crate::LightcraftApp;

/// Host-owned background work, independent of whether an extension page is visible.
#[derive(Clone, Copy)]
pub struct ExtensionBackground {
    /// Observe catalog/library changes after commands and UI passes; must not perform I/O.
    pub observe: fn(&LightcraftApp),
    /// Drain results and schedule background work, including during hidden logic ticks.
    pub poll: fn(&mut LightcraftApp, &egui::Context),
}

/// One page of the Extensions panel. `render` draws into the panel's [`egui::Ui`], exactly like a
/// built-in panel body.
#[derive(Clone, Copy, Debug)]
pub struct ExtensionPage {
    /// Stable id, unique among the registered pages (persisted in the layout).
    pub id: &'static str,
    /// The tab label, translated through [`crate::i18n`].
    pub title: &'static str,
    pub render: fn(&mut LightcraftApp, &mut egui::Ui),
}

impl LightcraftApp {
    pub fn register_extension_background(&mut self, background: ExtensionBackground) {
        self.extension_background.push(background);
    }

    pub(crate) fn observe_extensions(&self) {
        for background in &self.extension_background {
            (background.observe)(self);
        }
    }

    pub(crate) fn poll_extensions(&mut self, ctx: &egui::Context) {
        for background in self.extension_background.clone() {
            (background.poll)(self, ctx);
        }
    }

    /// Add a page to the Extensions panel. Pages keep their registration order; the first one is
    /// shown until the user picks another tab.
    pub fn register_extension_page(&mut self, page: ExtensionPage) {
        self.extension_pages.push(page);
    }
}

/// The Extensions panel's body: a tab strip when there is more than one page, then the active
/// page.
pub(crate) fn show(app: &mut LightcraftApp, ui: &mut egui::Ui) {
    if app.extension_pages.is_empty() {
        ui.label(crate::i18n::tr("No extension pages are installed."));
        return;
    }
    let active = app.extension_page.or_else(|| app.extension_pages.first().map(|page| page.id));
    if app.extension_pages.len() > 1 {
        let tabs: Vec<(&'static str, &'static str)> = app.extension_pages.iter().map(|page| (page.id, page.title)).collect();
        ui.horizontal_wrapped(|ui| {
            for (id, title) in tabs {
                if ui.selectable_label(Some(id) == active, crate::i18n::tr(title)).clicked() {
                    app.extension_page = Some(id);
                }
            }
        });
        ui.separator();
    }
    let page = app.extension_pages.iter().copied().find(|page| Some(page.id) == active).or_else(|| app.extension_pages.first().copied());
    if let Some(page) = page {
        (page.render)(app, ui);
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;

    fn app() -> LightcraftApp {
        LightcraftApp::new(lightcraft_engine::Session::new(), crate::Services::default())
    }

    fn noop(_: &mut LightcraftApp, _: &mut egui::Ui) {}

    #[test]
    fn pages_keep_their_registration_order() {
        let mut app = app();
        app.register_extension_page(ExtensionPage { id: "one", title: "One", render: noop });
        app.register_extension_page(ExtensionPage { id: "two", title: "Two", render: noop });
        assert_eq!(app.extension_pages.iter().map(|page| page.id).collect::<Vec<_>>(), ["one", "two"]);
        assert_eq!(app.extension_page, None, "the first page is shown until the user picks one");
    }

    #[test]
    fn background_callbacks_observe_commands_and_run_with_no_extension_panel() {
        static OBSERVED: AtomicUsize = AtomicUsize::new(0);
        static POLLED: AtomicUsize = AtomicUsize::new(0);
        fn observe(app: &LightcraftApp) {
            OBSERVED.store(app.session.catalog.photos().filter(|p| p.deleted).count(), Ordering::Relaxed);
        }
        fn poll(_: &mut LightcraftApp, _: &egui::Context) {
            POLLED.fetch_add(1, Ordering::Relaxed);
        }
        let mut app = LightcraftApp::new(lightcraft_engine::Session::with_demo(), crate::Services::default());
        app.register_extension_background(ExtensionBackground { observe, poll });
        let id = app.session.catalog.photos().next().unwrap().id;
        app.run("photo.delete", serde_json::json!({"ids": [id.0]})).unwrap();
        assert_eq!(OBSERVED.load(Ordering::Relaxed), 1, "commands publish catalog state before returning");
        let ctx = egui::Context::default();
        let mut input = egui::RawInput::default();
        input.viewports.entry(egui::ViewportId::ROOT).or_default().minimized = Some(true);
        let _ = ctx.run_logic(&input, |ctx| app.logic(ctx));
        assert!(app.extension_pages.is_empty());
        assert!(POLLED.load(Ordering::Relaxed) > 0, "hidden logic ticks drive background work");
    }

    #[test]
    fn show_renders_the_first_page_until_another_is_chosen() {
        static RENDERS: AtomicUsize = AtomicUsize::new(0);
        fn counted(_: &mut LightcraftApp, _: &mut egui::Ui) {
            RENDERS.fetch_add(1, Ordering::Relaxed);
        }
        let mut app = app();
        app.register_extension_page(ExtensionPage { id: "counted", title: "Counted", render: counted });
        let mut view = crate::headless::HeadlessView::new();
        let input = crate::headless::HeadlessView::raw_input(egui::vec2(400.0, 300.0), 1.0, 0.0, vec![]);
        let _ = view.run(input, |ui| show(&mut app, ui));
        assert!(RENDERS.load(Ordering::Relaxed) >= 1, "the registered page was not rendered");
    }
}
