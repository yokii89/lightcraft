# LightCraft extensions — agent rules

These rules apply to everything under `extension/`. The repository's root `AGENTS.md` still applies
in full; nothing here relaxes it. Where the two overlap, the stricter rule wins.

## The boundary

1. **All new feature code lives here.** UI, state, actions, tests, assets and docs for an extension
   live under `extension/`. Do not add feature code to `crates/`, `apps/`, `xtask/` or `docs/`.
2. **Everything outside `extension/` is read-only.** Use its public APIs; do not edit it.
3. **The only exception is page registration into the native app.** A registration edit may only
   touch the seams listed below, and must be:
   - **append-only** — a new module, a new enum variant, one added line or match arm; never a
     rewrite of an existing function;
   - **feature-agnostic** — it may know `ExtensionPage`, never a specific extension's state,
     strings or behaviour;
   - **recorded** in "Native registration seams" below, in the same change.

If a feature seems to need any other native edit, stop and extend the generic seam instead; do not
patch native logic inline.

## Dependency direction

`lightcraft-extension` → `lightcraft-ui-egui` / `lightcraft-engine`. Never the other way: native
crates must not depend on this crate. Registration is a function pointer the host hands to
`LightcraftApp`, so there is no cycle.

## The extension API (use only)

- `LightcraftApp::run(id, params)` — the single entry point for every native command (UI and engine),
  the same one menus, shortcuts, the CLI, MCP and the control channel use.
- `app.session` (library, catalog, selection, develop settings) and `app.ui` (UI state).
- `lightcraft_ui_egui::{theme::Tokens, widgets, icons, i18n, ext::ExtensionPage}` for native look
  and behaviour.
- Add dependencies to `extension/Cargo.toml` only, from `[workspace.dependencies]` where possible.

## Documentation

All extension documentation lives under `extension/docs/`:

- design and decision records: `extension/docs/NN-<topic>.md`;
- HTML UI design mocks: `extension/docs/ui-design/<page>.html`, standalone files that open in a
  browser without a build step.

Do not put extension notes in the product's `docs/` (that is reference documentation for
LightCraft itself) or in the optional `plan/` notebook. When a change adds or updates an extension
feature, update its design record in `extension/docs/` in the same change.

## Code rules

The root house rules apply in full, including:

- the crate root keeps `#![forbid(unsafe_code)]` and the six `clippy` denies, and `Cargo.toml`
  keeps `[lints] workspace = true`;
- no `unwrap` / `expect` / `panic` / `todo` / `unimplemented` in non-test code; return errors;
- input from files, users and agents is hostile: checked math, no indexing at an untrusted offset;
- every page lands with a test that fails without it;
- user-facing strings go through `lightcraft_ui_egui::i18n::tr`; add Simplified Chinese to
  `crates/ui-egui/locales/zh-hans.json` (a translation data edit, not feature code);
- new assets need an `assets/ATTRIBUTION.md` entry — the assets check scans the whole repository.

## Native registration seams (the complete list)

- `Cargo.toml` — workspace `members`
- `xtask/src/layers.rs` — `TABLE` entry (every workspace crate must be registered there)
- `apps/lightcraft/Cargo.toml`, `apps/lightcraft/src/main.rs` — the host dependency and
  `lightcraft_extension::install(&mut app)`
- `crates/ui-egui/src/ext.rs` — `ExtensionPage`, `register_extension_page`, the panel body
- `crates/ui-egui/src/docking.rs`, `crates/ui-egui/src/state.rs` — `Panel::Extension`,
  `extension_open`, and the Extensions tab sharing the Tools tab group (`extension_tabbed` is a
  session-only flag for the one-time merge of an older two-column layout)
- `crates/ui-egui/src/menus/*` — `view.extensionPage`
- `crates/ui-egui/src/panels/left.rs`, `crates/ui-egui/src/icons.rs` — the sidebar block and its icon
- `crates/ui-egui/locales/zh-hans.json` — translations for new strings

Adding a seam means adding it to this list. Keep each one a small, append-only hunk so merging
`upstream/main` stays painless.

## Adding a page

1. Write `fn page(app: &mut LightcraftApp, ui: &mut egui::Ui)` in `extension/src/lib.rs` (or a
   module under `extension/src/`).
2. Add `ExtensionPage { id: "my-page", title: "My Page", render: page }` to `pages()`.
3. Run `cargo test -p lightcraft-extension`, then open the page from the sidebar's Extensions block.

## Before committing

```bash
cargo fmt --all -- --check
cargo test -p lightcraft-extension
cargo clippy -p lightcraft-extension --all-targets -- -D warnings
```

extension\参考 是两个参考项目，用于展示如何实现一个简单的图片地图功能，我们准备参考其中的一些功能实现。
我们计划使用 extension\参考\map-main 作为我们的地图参考，因为它的地图更加高清
计划使用 extension\参考\picmap-main 作为我们的图片地图功能。

Then check the diff: nothing outside the seams above.
