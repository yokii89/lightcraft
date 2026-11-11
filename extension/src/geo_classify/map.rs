//! The review map (design: `extension/docs/02-地理分类地图视图设计.md`): a schematic China drawn
//! with the egui painter. The review's Chinese groups become bubbles at the centroid of their
//! photos' GPS; the wheel zooms around the pointer, dragging pans, and past a zoom threshold the
//! bubbles split into the photos' thumbnails at their own positions — a thumbnail click opens the
//! photo in the library's detail view, exactly like double-clicking a grid cell. One bubble's
//! popup card (include in write / focus) replaces the review list's per-group rows.
//!
//! The boundaries are original simplified vectors (our own work, recorded in
//! `assets/ATTRIBUTION.md`); the on-map note stays until standard boundary data replaces them.

use std::sync::{Arc, LazyLock};

use egui::{
    Align2, Color32, CursorIcon, Id, Mesh, MouseWheelUnit, Painter, PointerButton, Pos2, Rect, RichText, Sense, Shape, Stroke, StrokeKind, Ui, Vec2,
    pos2, vec2,
};
use lightcraft_ui_egui::button::Button;
use lightcraft_ui_egui::panels::grid::request_thumb;
use lightcraft_ui_egui::theme::Tokens;
use lightcraft_ui_egui::{i18n, icons, widgets};

use super::{Group, State};
use crate::ui::trn;

// ---------------------------------------------------------------- the boundaries
// (lat, lon) pairs, clockwise from the north tip. Original simplified vectors — a schematic
// outline, not cartographic data (see the note the map paints and the design doc).

/// The mainland (the rooster: northeast tip, east coast, the southwest Himalaya arc, Xinjiang).
const MAINLAND: &[(f64, f64)] = &[
    (53.5, 122.4),
    (53.2, 124.8),
    (52.3, 126.6),
    (50.7, 127.5),
    (49.8, 129.8),
    (48.9, 131.3),
    (48.1, 133.2),
    (48.4, 134.4),
    (47.0, 133.4),
    (45.6, 132.2),
    (44.7, 131.2),
    (43.3, 131.1),
    (42.4, 130.6),
    (41.6, 129.0),
    (41.5, 127.2),
    (40.9, 125.7),
    (40.1, 124.3),
    (39.6, 122.6),
    (38.8, 121.3),
    (39.5, 121.8),
    (40.6, 122.0),
    (40.5, 121.0),
    (39.9, 119.5),
    (39.1, 118.0),
    (38.2, 118.6),
    (37.5, 119.9),
    (37.5, 121.5),
    (37.4, 122.7),
    (36.6, 121.0),
    (35.9, 119.9),
    (34.7, 119.3),
    (32.8, 120.9),
    (31.4, 121.9),
    (30.6, 121.0),
    (29.9, 121.6),
    (28.0, 120.7),
    (26.8, 119.9),
    (25.9, 119.6),
    (24.9, 118.6),
    (24.4, 118.1),
    (23.4, 116.9),
    (22.6, 114.7),
    (22.0, 113.4),
    (21.6, 112.2),
    (21.2, 110.6),
    (20.2, 110.3),
    (21.4, 109.8),
    (21.5, 109.0),
    (21.6, 108.0),
    (22.8, 106.7),
    (23.3, 105.3),
    (22.5, 103.9),
    (22.4, 102.1),
    (21.7, 101.1),
    (21.2, 100.9),
    (21.6, 100.1),
    (22.5, 99.9),
    (23.0, 99.2),
    (24.1, 98.3),
    (24.8, 97.6),
    (25.5, 97.9),
    (26.1, 98.7),
    (27.0, 98.5),
    (28.2, 97.6),
    (28.6, 95.5),
    (27.7, 94.0),
    (27.0, 92.2),
    (27.3, 90.4),
    (27.4, 89.1),
    (27.9, 88.1),
    (28.1, 86.6),
    (28.6, 84.5),
    (29.3, 82.5),
    (30.1, 81.0),
    (31.1, 79.7),
    (32.6, 78.6),
    (34.6, 78.0),
    (35.6, 76.8),
    (36.7, 75.9),
    (37.2, 74.9),
    (38.0, 74.6),
    (39.6, 75.2),
    (40.7, 74.8),
    (41.0, 76.0),
    (41.4, 78.0),
    (42.2, 80.2),
    (43.2, 80.6),
    (43.2, 82.6),
    (44.2, 83.5),
    (44.8, 85.0),
    (46.0, 85.7),
    (46.6, 86.5),
    (47.9, 88.0),
    (47.7, 90.6),
    (45.0, 93.5),
    (44.3, 95.5),
    (43.0, 96.8),
    (42.5, 99.5),
    (42.5, 101.9),
    (41.8, 104.5),
    (42.4, 107.3),
    (42.4, 109.5),
    (43.3, 111.0),
    (44.7, 113.5),
    (45.5, 115.5),
    (46.5, 117.4),
    (46.7, 119.9),
    (48.0, 119.3),
    (48.1, 117.8),
    (49.8, 116.7),
    (50.3, 119.3),
    (51.6, 120.7),
    (53.0, 122.0),
];
/// Hainan, off the south coast.
const HAINAN: &[(f64, f64)] = &[(20.1, 110.0), (19.6, 110.6), (18.8, 110.5), (18.2, 109.7), (18.5, 108.9), (19.4, 108.6), (20.0, 109.2)];
/// Taiwan, off the southeast coast.
const TAIWAN: &[(f64, f64)] = &[(25.3, 121.8), (24.5, 121.9), (23.5, 121.6), (21.9, 120.9), (22.6, 120.2), (23.8, 120.1), (24.8, 120.9)];

/// Ear-clipping triangulation of a boundary polygon, computed once: topology never changes with
/// pan/zoom, only the projection of the triangle corners does. Degenerate input yields no
/// triangles instead of a panic — the polygons are consts, but the tessellation stays guarded.
fn triangulate(poly: &[(f64, f64)]) -> Vec<[usize; 3]> {
    let n = poly.len();
    if n < 3 {
        return Vec::new();
    }
    let area2: f64 = (0..n)
        .map(|i| {
            let (a1, a2) = poly[i];
            let (b1, b2) = poly[(i + 1) % n];
            a1 * b2 - b1 * a2
        })
        .sum();
    if !area2.is_finite() || area2.abs() < 1e-9 {
        return Vec::new();
    }
    // work counterclockwise so an ear's cross product is positive
    let mut ring: Vec<usize> = if area2 > 0.0 { (0..n).collect() } else { (0..n).rev().collect() };
    let cross = |a: usize, b: usize, c: usize| {
        let (ax, ay) = poly[a];
        let (bx, by) = poly[b];
        let (cx, cy) = poly[c];
        (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
    };
    let mut tris = Vec::new();
    // a hostile/degenerate ring can never loop longer than one clip attempt per vertex pair
    let mut guard = 0usize;
    while ring.len() > 3 && guard <= n * n + 16 {
        guard += 1;
        let mut ear: Option<usize> = None;
        for k in 0..ring.len() {
            let prev = ring[(k + ring.len() - 1) % ring.len()];
            let mid = ring[k];
            let next = ring[(k + 1) % ring.len()];
            if cross(prev, mid, next) <= 1e-9 {
                continue; // reflex or degenerate corner: not an ear
            }
            let covered = ring.iter().all(|&m| m == prev || m == mid || m == next || !point_in_tri(poly[m], poly[prev], poly[mid], poly[next]));
            if covered {
                ear = Some(k);
                break;
            }
        }
        let Some(k) = ear else { break }; // numeric trouble: ship the triangles we have
        let prev = ring[(k + ring.len() - 1) % ring.len()];
        let mid = ring[k];
        let next = ring[(k + 1) % ring.len()];
        tris.push([prev, mid, next]);
        ring.remove(k);
    }
    if ring.len() == 3 {
        tris.push([ring[0], ring[1], ring[2]]);
    }
    tris
}

/// Barycentric inside test (on the edge counts as inside, which is the conservative direction).
fn point_in_tri(p: (f64, f64), a: (f64, f64), b: (f64, f64), c: (f64, f64)) -> bool {
    let s = |a: (f64, f64), b: (f64, f64), q: (f64, f64)| (b.0 - a.0) * (q.1 - a.1) - (b.1 - a.1) * (q.0 - a.0);
    let (d1, d2, d3) = (s(a, b, p), s(b, c, p), s(c, a, p));
    d1 >= 0.0 && d2 >= 0.0 && d3 >= 0.0
}

static MAINLAND_TRIS: LazyLock<Vec<[usize; 3]>> = LazyLock::new(|| triangulate(MAINLAND));
static HAINAN_TRIS: LazyLock<Vec<[usize; 3]>> = LazyLock::new(|| triangulate(HAINAN));
static TAIWAN_TRIS: LazyLock<Vec<[usize; 3]>> = LazyLock::new(|| triangulate(TAIWAN));

// ---------------------------------------------------------------- the view (geometry)

/// The national view.
pub(crate) const NATIONAL: MapView = MapView { c_lon: 104.5, c_lat: 36.2, span_lon: 62.0 };
/// World units per degree of latitude (the map's fixed scale).
const K_Y: f64 = 16.0;
/// World units per degree of longitude (compressed by cos 35°, like the prototype).
const K_X: f64 = K_Y * 0.819;
/// Zoom bounds, in degrees of longitude across the view.
const MIN_SPAN: f64 = 1.2;
const MAX_SPAN: f64 = 70.0;
/// Below this span (degrees of longitude across the view) bubbles split into photo thumbnails.
pub(crate) const THRESH: f64 = 9.0;
/// Zoom factor per wheel line-notch.
const WHEEL_NOTCH: f64 = 1.35;
/// Vertical room the map leaves for the legend line and the write button below it.
const BELOW_RESERVE: f32 = 84.0;
const MIN_MAP_H: f32 = 260.0;
/// Thumbnails requested per frame at most (the cache fills over the following frames).
const THUMB_BUDGET: usize = 80;
/// South China Sea islands inset: hidden once the view is regional or closer.
const INSET_SPAN: f64 = 20.0;

/// The map's view: a centre in degrees and how much longitude the view spans.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct MapView {
    pub c_lon: f64,
    pub c_lat: f64,
    pub span_lon: f64,
}

impl Default for MapView {
    fn default() -> Self {
        NATIONAL
    }
}

impl MapView {
    /// Degrees of latitude the view spans for a widget of this size (the aspect correction).
    fn span_lat(&self, size: Vec2) -> f64 {
        let w = (size.x as f64).max(1.0);
        let h = (size.y as f64).max(1.0);
        self.span_lon * (h / w) * (K_X / K_Y)
    }
}

/// Zoom by `factor` keeping the position under `at` (lat, lon) where it is.
pub(crate) fn zoom_around(view: MapView, factor: f64, at: (f64, f64), span_lat: f64) -> MapView {
    let factor = factor.clamp(0.2, 5.0);
    let span_lat = span_lat.max(1e-6);
    let span2 = (view.span_lon / factor).clamp(MIN_SPAN, MAX_SPAN);
    let span_lat2 = span_lat * span2 / view.span_lon;
    let (at_lat, at_lon) = at;
    MapView {
        c_lon: at_lon - (at_lon - view.c_lon) * span2 / view.span_lon,
        c_lat: at_lat + (view.c_lat - at_lat) * span_lat2 / span_lat,
        span_lon: span2,
    }
}

/// Keep the view inside China's neighbourhood.
pub(crate) fn clamp_view(view: &mut MapView) {
    view.span_lon = view.span_lon.clamp(MIN_SPAN, MAX_SPAN);
    view.c_lon = view.c_lon.clamp(75.0, 134.0);
    view.c_lat = view.c_lat.clamp(16.0, 53.0);
}

/// The lat/lon → pixels transform for one frame.
pub(crate) struct Cam {
    rect: Rect,
    /// Degrees of latitude the view spans at this size (tests read it for the zoom invariant).
    pub(crate) span_lat: f64,
}

impl Cam {
    pub(crate) fn new(view: &MapView, rect: Rect) -> Self {
        Cam { rect, span_lat: view.span_lat(rect.size()) }
    }

    pub(crate) fn to_px(&self, view: &MapView, lat: f64, lon: f64) -> Pos2 {
        let w = self.rect.width();
        let h = self.rect.height();
        let fx = (lon - (view.c_lon - view.span_lon * 0.5)) / view.span_lon;
        let fy = ((view.c_lat + self.span_lat * 0.5) - lat) / self.span_lat;
        pos2(self.rect.left() + (fx * w as f64) as f32, self.rect.top() + (fy * h as f64) as f32)
    }

    pub(crate) fn to_lat_lon(&self, view: &MapView, p: Pos2) -> (f64, f64) {
        let w = (self.rect.width() as f64).max(1.0);
        let h = (self.rect.height() as f64).max(1.0);
        let lon = view.c_lon - view.span_lon * 0.5 + ((p.x - self.rect.left()) as f64 / w) * view.span_lon;
        let lat = view.c_lat + self.span_lat * 0.5 - ((p.y - self.rect.top()) as f64 / h) * self.span_lat;
        (lat, lon)
    }
}

/// Quick presets, shown as chips at the map's top left.
const PRESETS: [(&str, MapView); 4] = [
    ("All of China", NATIONAL),
    ("East China", MapView { c_lon: 119.8, c_lat: 31.0, span_lon: 15.0 }),
    ("Pearl River Delta", MapView { c_lon: 113.7, c_lat: 22.6, span_lon: 6.0 }),
    ("Western Sichuan", MapView { c_lon: 101.2, c_lat: 29.6, span_lon: 8.0 }),
];

// ---------------------------------------------------------------- the points

/// One review group on the map: a bubble at the centroid of its photos' GPS.
pub(crate) struct MapPoint {
    /// The exact keyword path (also the popup's and the write's key).
    pub path: String,
    /// The city, or the country when no city was found.
    pub label: String,
    pub count: usize,
    /// (lat, lon) centroid of the photos' own GPS.
    pub center: (f64, f64),
    /// (id, file name, lat, lon, distance to the place in metres).
    pub photos: Vec<(lightcraft_catalog::PhotoId, String, f64, f64, f64)>,
    /// Unticked in the popup: not part of the write, drawn dimmed.
    pub off: bool,
}

/// The Chinese groups of the review as map points; groups abroad stay off the map.
pub(crate) fn map_points(all: &[Group]) -> Vec<MapPoint> {
    let mut points = Vec::new();
    for g in all {
        if !g.iso.eq_ignore_ascii_case("CN") || g.places.is_empty() {
            continue;
        }
        let (mut lat_sum, mut lon_sum) = (0.0, 0.0);
        for p in &g.places {
            lat_sum += p.lat;
            lon_sum += p.lon;
        }
        let n = g.places.len() as f64;
        points.push(MapPoint {
            path: g.path.clone(),
            label: if g.city.is_empty() { g.country.clone() } else { g.city.clone() },
            count: g.places.len(),
            center: (lat_sum / n, lon_sum / n),
            photos: g.places.iter().map(|p| (p.id, p.name.clone(), p.lat, p.lon, p.distance)).collect(),
            off: false,
        });
    }
    points
}

fn bubble_radius(count: usize) -> f32 {
    (2.35 * (count as f64).sqrt()).clamp(8.0, 30.0) as f32
}

// ---------------------------------------------------------------- the colours

struct MapColors {
    sea: Color32,
    land: Color32,
    edge: Color32,
    graticule: Color32,
    land_text: Color32,
}

fn map_colors(dark: bool, t: Tokens) -> MapColors {
    if dark {
        MapColors {
            sea: Color32::from_rgb(0x14, 0x14, 0x14),
            land: Color32::from_rgb(0x24, 0x24, 0x24),
            edge: Color32::from_rgb(0x4a, 0x4a, 0x4a),
            graticule: Color32::from_white_alpha(8),
            land_text: t.text_dim,
        }
    } else {
        MapColors {
            sea: Color32::from_rgb(0xea, 0xea, 0xe8),
            land: Color32::from_rgb(0xd9, 0xd9, 0xd6),
            edge: Color32::from_rgb(0xa5, 0xa5, 0xa0),
            graticule: Color32::from_black_alpha(10),
            land_text: t.text_dim,
        }
    }
}

/// Everything the item painters need (one bundle keeps the signatures small).
struct Ctx<'a> {
    ui: &'a mut Ui,
    painter: Painter,
    cam: Cam,
    view: MapView,
    c: MapColors,
    t: Tokens,
}

// ---------------------------------------------------------------- the widget

/// The map fills the review: allocate, read input, paint the base, the items, the popup and the
/// floating controls. The map is the whole interaction surface; `st` carries view and selection.
pub(crate) fn draw_map(app: &mut lightcraft_ui_egui::LightcraftApp, ui: &mut Ui, st: &mut State, all: &[Group]) {
    let t = Tokens::get(ui.ctx());
    let dark = lightcraft_ui_egui::theme::current(ui.ctx()).unwrap_or_default().is_dark();
    let mut points = map_points(all);
    if points.is_empty() {
        ui.add_space(6.0);
        ui.label(RichText::new(i18n::tr("No places in China to show on the map.")).font(t.font(11.0)).color(t.text_dim));
        return;
    }
    for p in &mut points {
        p.off = st.groups_off.contains(&p.path);
    }
    let height = (ui.available_height() - BELOW_RESERVE).clamp(MIN_MAP_H, 760.0);
    let (rect, base) = ui.allocate_exact_size(vec2(ui.available_width(), height), Sense::click_and_drag());
    widgets::register(ui, "geo:map", rect);
    let c = map_colors(dark, t);
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 4.0, c.sea);

    handle_input(ui, st, &base, rect);
    // the view may have moved with the input: build the camera fresh
    let t = Tokens::get(ui.ctx());
    paint_base(&painter, rect, &st.map, &c);
    let mut cx = Ctx { ui, painter, cam: Cam::new(&st.map, rect), view: st.map, c, t };
    let (item_drag, item_hit) = if cx.view.span_lon <= THRESH { draw_thumbs(app, st, &points, &mut cx) } else { draw_bubbles(st, &points, &mut cx) };
    let base_drag = if base.dragged_by(PointerButton::Primary) { base.drag_delta() } else { Vec2::ZERO };
    apply_pan(&mut *cx.ui, st, &base, item_drag + base_drag, rect, item_hit);
    if st.map.span_lon > INSET_SPAN {
        paint_inset(&cx.t, &cx.painter, rect, &cx.c);
    }
    paint_note(&cx.t, &cx.painter, rect);
    draw_popup(st, &points, &mut cx, rect);
    draw_controls(st, &mut *cx.ui, rect);
}

/// Wheel zoom around the pointer, double-click zoom, and the grab cursors.
fn handle_input(ui: &mut Ui, st: &mut State, base: &egui::Response, rect: Rect) {
    let Some(p) = ui.input(|i| i.pointer.hover_pos()) else { return };
    if !rect.contains(p) {
        return;
    }
    if base.hovered() {
        ui.ctx().set_cursor_icon(CursorIcon::Grab);
    }
    if base.dragged_by(PointerButton::Primary) {
        ui.ctx().set_cursor_icon(CursorIcon::Grabbing);
    }
    let mut notches = 0.0f32;
    ui.input(|i| {
        for ev in &i.events {
            if let egui::Event::MouseWheel { unit, delta, modifiers, .. } = ev {
                if modifiers.ctrl || modifiers.alt {
                    continue; // ctrl+wheel arrives as zoom_delta too: zoom once, not twice
                }
                notches += match unit {
                    MouseWheelUnit::Point => delta.y / 24.0,
                    MouseWheelUnit::Line => delta.y,
                    MouseWheelUnit::Page => delta.y * 8.0,
                };
            }
        }
    });
    let zoom_delta = ui.input(|i| i.zoom_delta());
    let mut factor = 1.0f64;
    if zoom_delta != 1.0 {
        factor = zoom_delta as f64;
    }
    if notches.abs() > 1e-6 {
        factor *= WHEEL_NOTCH.powf(notches as f64);
    }
    if (factor - 1.0).abs() > 1e-9 {
        let cam = Cam::new(&st.map, rect);
        let at = cam.to_lat_lon(&st.map, p);
        st.map = zoom_around(st.map, factor, at, cam.span_lat);
        clamp_view(&mut st.map);
        st.map_sel = None;
        // the map owns the wheel while hovered; nothing behind it scrolls
        ui.input_mut(|i| i.smooth_scroll_delta = Vec2::ZERO);
        ui.ctx().request_repaint();
    }
    if base.double_clicked()
        && let Some(p2) = base.interact_pointer_pos()
        && rect.contains(p2)
    {
        let cam = Cam::new(&st.map, rect);
        let at = cam.to_lat_lon(&st.map, p2);
        st.map = zoom_around(st.map, 1.6, at, cam.span_lat);
        clamp_view(&mut st.map);
        st.map_sel = None;
    }
}

/// Graticule, then the three land polygons (ear-clip fill + outline).
fn paint_base(painter: &Painter, rect: Rect, view: &MapView, c: &MapColors) {
    let cam = Cam::new(view, rect);
    let mut lon = ((view.c_lon - view.span_lon * 0.5) / 10.0).ceil() * 10.0;
    while lon <= view.c_lon + view.span_lon * 0.5 {
        let x = cam.to_px(view, view.c_lat, lon).x;
        painter.line_segment([pos2(x, rect.top()), pos2(x, rect.bottom())], Stroke::new(1.0, c.graticule));
        lon += 10.0;
    }
    let mut lat = ((view.c_lat - cam.span_lat * 0.5) / 10.0).ceil() * 10.0;
    while lat <= view.c_lat + cam.span_lat * 0.5 {
        let y = cam.to_px(view, lat, view.c_lon).y;
        painter.line_segment([pos2(rect.left(), y), pos2(rect.right(), y)], Stroke::new(1.0, c.graticule));
        lat += 10.0;
    }
    for (poly, tris) in [(MAINLAND, &MAINLAND_TRIS), (HAINAN, &HAINAN_TRIS), (TAIWAN, &TAIWAN_TRIS)] {
        let mut mesh = Mesh::default();
        for tri in tris.iter() {
            for &i in tri {
                let (la, lo) = poly[i];
                mesh.colored_vertex(cam.to_px(view, la, lo), c.land);
            }
        }
        mesh.indices.extend(0..(tris.len() * 3) as u32);
        painter.add(Shape::Mesh(Arc::new(mesh)));
        let pts: Vec<Pos2> = poly.iter().map(|(la, lo)| cam.to_px(view, *la, *lo)).collect();
        painter.add(Shape::closed_line(pts, Stroke::new(1.2, c.edge)));
    }
}

/// The South China Sea islands inset, bottom right, while the view is national or provincial.
fn paint_inset(t: &Tokens, painter: &Painter, rect: Rect, c: &MapColors) {
    let (w, h) = (84.0, 104.0);
    let r = Rect::from_min_size(pos2(rect.right() - w - 10.0, rect.bottom() - h - 10.0), vec2(w, h));
    painter.rect_filled(r, 3.0, c.sea);
    painter.rect_stroke(r, 3.0, Stroke::new(1.0, c.edge), StrokeKind::Inside);
    painter.text(pos2(r.center().x, r.top() + 9.0), Align2::CENTER_CENTER, i18n::tr("South China Sea Islands"), t.font(9.0), t.text_dim);
    // a schematic dashed U with a few island dots — the inset's standard treatment
    let u = [
        pos2(r.left() + 16.0, r.top() + 24.0),
        pos2(r.left() + 10.0, r.top() + 44.0),
        pos2(r.left() + 11.0, r.top() + 62.0),
        pos2(r.left() + 20.0, r.top() + 78.0),
        pos2(r.left() + 36.0, r.top() + 88.0),
        pos2(r.left() + 55.0, r.top() + 91.0),
        pos2(r.left() + 70.0, r.top() + 85.0),
        pos2(r.left() + 78.0, r.top() + 72.0),
        pos2(r.left() + 79.0, r.top() + 58.0),
    ];
    for seg in Shape::dashed_line(&u, Stroke::new(1.0, c.edge), 5.0, 4.0) {
        painter.add(seg);
    }
    for (fx, fy) in [(0.44, 0.62), (0.55, 0.72), (0.66, 0.79), (0.32, 0.52), (0.78, 0.42)] {
        painter.circle_filled(pos2(r.left() + w * fx, r.top() + h * fy), 1.5, c.land_text);
    }
}

/// The schematic-boundary note, bottom left of the map.
fn paint_note(t: &Tokens, painter: &Painter, rect: Rect) {
    painter.text(
        pos2(rect.left() + 10.0, rect.bottom() - 8.0),
        Align2::LEFT_BOTTOM,
        i18n::tr("The boundaries are schematic."),
        t.font(9.5),
        t.text_disabled,
    );
}

fn in_view(rect: Rect, p: Pos2, margin: f32) -> bool {
    rect.expand(margin).contains(p)
}

/// The bubble layer: one bubble per Chinese group, big ones first so small ones paint on top.
/// Returns the drag started on an item and whether any item was clicked this frame.
fn draw_bubbles(st: &mut State, points: &[MapPoint], cx: &mut Ctx) -> (Vec2, bool) {
    let mut drag = Vec2::ZERO;
    let mut hit = false;
    let show_labels = cx.view.span_lon <= 18.0;
    let mut order: Vec<usize> = (0..points.len()).collect();
    order.sort_by(|&a, &b| points[b].count.cmp(&points[a].count));
    for &pi in &order {
        let p = &points[pi];
        let pos = cx.cam.to_px(&cx.view, p.center.0, p.center.1);
        if !in_view(cx.cam.rect, pos, 40.0) {
            continue;
        }
        let r = bubble_radius(p.count);
        let fill = if p.off { cx.t.text_dim.gamma_multiply(0.45) } else { cx.t.accent };
        if r >= 12.0 {
            cx.painter.circle_filled(pos, r + 5.0, cx.t.accent.gamma_multiply(0.15));
        }
        cx.painter.circle_filled(pos, r, fill);
        cx.painter.circle_stroke(pos, r, Stroke::new(1.0, cx.t.inset));
        if r >= 24.0 {
            cx.painter.text(pos, Align2::CENTER_CENTER, p.count.to_string(), cx.t.font(10.0), Color32::WHITE);
        }
        if show_labels && !p.off {
            cx.painter.text(pos2(pos.x, pos.y + r + 9.0), Align2::CENTER_TOP, &p.label, cx.t.font(10.5), cx.c.land_text);
        }
        if st.map_sel.as_deref() == Some(p.path.as_str()) {
            cx.painter.circle_stroke(pos, r + 4.0, Stroke::new(1.8, cx.t.ring_selected));
        }
        let item_rect = Rect::from_center_size(pos, vec2(r * 2.0, r * 2.0));
        let resp = cx.ui.interact(item_rect, Id::unique(("geo-map-bubble", pi)), Sense::click_and_drag());
        widgets::register(cx.ui, format!("geo:map:bubble:{pi}"), item_rect);
        let mut tip = trn("{place} · {n} photos", &[("place", p.label.clone()), ("n", p.count.to_string())]);
        if p.off {
            tip.push_str(" · ");
            tip.push_str(i18n::tr("Not included in the write"));
        }
        let resp = resp.on_hover_text(tip);
        if resp.clicked() {
            st.map_sel = Some(p.path.clone());
            hit = true;
        }
        if resp.dragged_by(PointerButton::Primary) {
            drag += resp.drag_delta();
        }
    }
    (drag, hit)
}

/// The thumbnail layer: every photo at its own GPS position, opened in the detail view on click.
/// Returns the drag started on an item and whether any item was clicked this frame.
fn draw_thumbs(app: &mut lightcraft_ui_egui::LightcraftApp, st: &mut State, points: &[MapPoint], cx: &mut Ctx) -> (Vec2, bool) {
    let mut drag = Vec2::ZERO;
    let mut hit = false;
    let size = (24.0f32 + 46.0f32 / cx.view.span_lon.max(1.0).sqrt() as f32).clamp(26.0, 54.0);
    let h = size * 0.72;
    let mut visible = vec![0usize; points.len()];
    for (pi, p) in points.iter().enumerate() {
        for (_, _, lat, lon, _) in &p.photos {
            if in_view(cx.cam.rect, cx.cam.to_px(&cx.view, *lat, *lon), 30.0) {
                visible[pi] += 1;
            }
        }
    }
    let mut requested = 0usize;
    for (pi, p) in points.iter().enumerate() {
        if visible[pi] == 0 {
            continue;
        }
        // the thumbnails sit at their own GPS positions; the labels hug their bounding box
        let mut bounds: Option<Rect> = None;
        let mut painted = 0usize;
        for (id, file, lat, lon, dist_m) in &p.photos {
            let pos = cx.cam.to_px(&cx.view, *lat, *lon);
            if !in_view(cx.cam.rect, pos, 30.0) {
                continue;
            }
            let rect = Rect::from_center_size(pos, vec2(size, h));
            bounds = Some(match bounds {
                Some(b) => b.union(rect),
                None => rect,
            });
            if requested < THUMB_BUDGET {
                request_thumb(app, *id, 128, 10);
                requested += 1;
            }
            painted += 1;
            let tex = app.renderer.thumb(*id).map(|tx| tx.id());
            match tex {
                Some(tex_id) => {
                    cx.painter.image(tex_id, rect, Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0)), Color32::WHITE);
                }
                None => {
                    cx.painter.rect_filled(rect, 2.0, cx.t.skeleton);
                }
            }
            let sel = st.photo_sel == Some(*id);
            cx.painter.rect_stroke(
                rect,
                2.0,
                Stroke::new(if sel { 2.0 } else { 1.0 }, if sel { cx.t.ring_selected } else { cx.c.edge }),
                StrokeKind::Outside,
            );
            let resp = cx.ui.interact(rect, Id::unique(("geo-map-photo", id.0)), Sense::click_and_drag());
            widgets::register(cx.ui, format!("geo:map:photo:{}", id.0), rect);
            let resp = resp.on_hover_text(trn("{file} · {km} km", &[("file", file.clone()), ("km", format!("{:.0}", dist_m / 1000.0))]));
            if resp.clicked() {
                // what a grid cell's double-click does: select, then the detail (loupe) view
                let _ = app.run("library.select", serde_json::json!({ "ids": [id.0] }));
                let _ = app.run("view.detail", serde_json::json!({}));
                st.photo_sel = Some(*id);
                hit = true;
            }
            if resp.dragged_by(PointerButton::Primary) {
                drag += resp.drag_delta();
            }
        }
        // the place's name under its thumbnails, and how many stayed off-screen
        let Some(b) = bounds else { continue };
        cx.painter.text(
            pos2(b.center().x, b.bottom() + 12.0),
            Align2::CENTER_TOP,
            trn("{place} · {n}", &[("place", p.label.clone()), ("n", p.count.to_string())]),
            cx.t.font(10.5),
            cx.c.land_text,
        );
        if p.count > painted {
            cx.painter.text(
                pos2(b.right() + 8.0, b.center().y),
                Align2::LEFT_CENTER,
                trn("+{n} more", &[("n", (p.count - painted).to_string())]),
                cx.t.font(10.0),
                cx.t.text_dim,
            );
        }
    }
    (drag, hit)
}

/// Drag (from the base or started on an item) pans the view; a click on the sea closes the
/// popup, a click on an item keeps what it selected.
fn apply_pan(ui: &mut Ui, st: &mut State, base: &egui::Response, drag: Vec2, rect: Rect, item_hit: bool) {
    if drag != Vec2::ZERO {
        let w = (rect.width() as f64).max(1.0);
        let h = (rect.height() as f64).max(1.0);
        let span_lat = st.map.span_lat(rect.size());
        st.map.c_lon -= drag.x as f64 * st.map.span_lon / w;
        st.map.c_lat += drag.y as f64 * span_lat / h;
        clamp_view(&mut st.map);
        st.map_sel = None;
        ui.ctx().request_repaint();
    }
    if base.clicked() && !item_hit {
        st.map_sel = None;
    }
}

/// The selected bubble's card: include in write / focus / close.
fn draw_popup(st: &mut State, points: &[MapPoint], cx: &mut Ctx, rect: Rect) {
    let Some(sel) = st.map_sel.clone() else { return };
    let Some(p) = points.iter().find(|p| p.path == sel) else { return };
    let pos = cx.cam.to_px(&cx.view, p.center.0, p.center.1);
    let size = vec2(190.0, 118.0);
    let mut origin = pos2(pos.x + bubble_radius(p.count) + 10.0, pos.y - 12.0);
    origin.x = origin.x.min(rect.right() - size.x - 6.0).max(rect.left() + 6.0);
    origin.y = origin.y.min(rect.bottom() - size.y - 6.0).max(rect.top() + 6.0);
    egui::Area::new(Id::unique("geo-map-popup")).fixed_pos(origin).order(egui::Order::Foreground).show(cx.ui.ctx(), |ui| {
        egui::Frame::NONE.inner_margin(egui::Margin::same(8)).fill(cx.t.inset).stroke(Stroke::new(1.0, cx.t.divider)).corner_radius(5.0).show(
            ui,
            |ui| {
                ui.set_width(size.x);
                ui.horizontal(|ui| {
                    ui.label(RichText::new(&p.label).font(cx.t.semibold(11.5)).color(cx.t.text));
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if Button::icon_only(icons::Icon::Close, "Close the place card")
                            .ghost()
                            .small()
                            .widget("geo:map:popup:close")
                            .show(ui)
                            .clicked()
                        {
                            st.map_sel = None;
                        }
                    });
                });
                ui.label(
                    RichText::new(trn("{path} · {n}", &[("path", p.path.clone()), ("n", p.count.to_string())]))
                        .font(cx.t.font(10.0))
                        .color(cx.t.text_dim),
                );
                ui.add_space(4.0);
                let mut on = !st.groups_off.contains(&p.path);
                let resp = ui.checkbox(&mut on, i18n::tr("Include in write"));
                widgets::register(ui, "geo:map:include", resp.rect);
                if resp.changed() {
                    if on {
                        st.groups_off.remove(&p.path);
                    } else {
                        st.groups_off.insert(p.path.clone());
                    }
                }
                ui.add_space(4.0);
                if Button::new(i18n::tr("Focus on the place")).secondary().small().widget("geo:map:focus").show(ui).clicked() {
                    st.map = MapView { c_lon: p.center.1, c_lat: p.center.0, span_lon: 3.2 };
                    st.map_sel = None;
                }
            },
        );
    });
}

/// The preset chips (top left) and the zoom buttons (top right), as foreground areas.
fn draw_controls(st: &mut State, ui: &mut Ui, rect: Rect) {
    egui::Area::new(Id::unique("geo-map-chips")).fixed_pos(rect.left_top() + vec2(8.0, 8.0)).order(egui::Order::Foreground).show(ui.ctx(), |ui| {
        ui.horizontal(|ui| {
            for (i, (label, view)) in PRESETS.iter().enumerate() {
                let on = (st.map.c_lon - view.c_lon).abs() < 1e-9
                    && (st.map.c_lat - view.c_lat).abs() < 1e-9
                    && (st.map.span_lon - view.span_lon).abs() < 1e-9;
                let mut btn = Button::new(i18n::tr(label)).small().widget(format!("geo:map:preset:{i}"));
                if on {
                    btn = btn.primary();
                }
                if btn.show(ui).clicked() {
                    st.map = *view;
                    st.map_sel = None;
                }
            }
        });
    });
    egui::Area::new(Id::unique("geo-map-zoom")).fixed_pos(pos2(rect.right() - 104.0, rect.top() + 8.0)).order(egui::Order::Foreground).show(
        ui.ctx(),
        |ui| {
            ui.horizontal(|ui| {
                let zoom = |st: &mut State, factor: f64, rect: Rect| {
                    let at = (st.map.c_lat, st.map.c_lon);
                    let span_lat = st.map.span_lat(rect.size());
                    st.map = zoom_around(st.map, factor, at, span_lat);
                    clamp_view(&mut st.map);
                };
                if Button::icon_only(icons::Icon::Plus, "Zoom in").small().widget("geo:map:zoomin").show(ui).clicked() {
                    zoom(st, 1.5, rect);
                }
                if Button::icon_only(icons::Icon::Minus, "Zoom out").small().widget("geo:map:zoomout").show(ui).clicked() {
                    zoom(st, 1.0 / 1.5, rect);
                }
                if Button::icon_only(icons::Icon::Target, "Reset view").small().widget("geo:map:home").show(ui).clicked() {
                    st.map = NATIONAL;
                    st.map_sel = None;
                }
            });
        },
    );
}
