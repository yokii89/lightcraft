//! LightCraft extensions: native pages that live outside the app's own crates.
//!
//! The desktop host calls [`install`] once, before the first frame. Everything a page shows and
//! does stays in this crate: `lightcraft-ui-egui` only knows the generic [`ExtensionPage`] handle
//! and never depends on it.
//!
//! The panel carries ONE page, the [`hub`] (1级): a list of the features this crate adds, each of
//! which opens as a level-2 page of its own (see `hub.rs`).
#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable)]

use lightcraft_ui_egui::{LightcraftApp, ext::ExtensionPage};

pub(crate) mod auto_keywords;
pub(crate) mod geo_classify;
pub(crate) mod hub;
pub(crate) mod scope;
pub mod semantic_search;
pub(crate) mod tiles;
pub(crate) mod ui;

/// The pages the host renders: the hub, the extensions' level-1 page.
pub fn pages() -> Vec<ExtensionPage> {
    vec![ExtensionPage { id: hub::PAGE_ID, title: "Extensions", render: hub::page }]
}

/// Register every page with the running app.
pub fn install(app: &mut LightcraftApp) {
    app.register_extension_background(lightcraft_ui_egui::ext::ExtensionBackground {
        observe: semantic_search::observe,
        poll: semantic_search::poll,
    });
    for page in pages() {
        app.register_extension_page(page);
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn install_registers_the_hub_page_only() {
        let mut app = lightcraft_ui_egui::LightcraftApp::new(lightcraft_engine::Session::with_demo(), lightcraft_ui_egui::Services::default());
        super::install(&mut app);
        assert_eq!(app.extension_pages.iter().map(|page| page.id).collect::<Vec<_>>(), [super::hub::PAGE_ID]);
    }

    #[test]
    fn the_hub_lists_every_feature_once_with_the_expected_entries() {
        let features = super::hub::features();
        for feature in &features {
            assert!(features.iter().filter(|f| f.id == feature.id).count() == 1, "feature ids are unique: {}", feature.id);
        }
        assert!(features.iter().any(|f| f.id == super::auto_keywords::PAGE_ID), "Auto Keywords is a feature");
        assert!(features.iter().any(|f| f.id == super::geo_classify::PAGE_ID), "Geographic classification is a feature");
    }
}
