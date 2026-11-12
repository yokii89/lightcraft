//! The online raster basemap of the Geographic classification map: AMap (高德地图) 256 px tiles,
//! fetched in the background, decoded off the UI thread and kept as GPU textures for the session.
//!
//! # Why AMap, and the licence line
//!
//! The reference project's map (`extension/参考/map-main`) is AMap's default raster basemap. AMap's
//! open-platform terms forbid fetching its tiles outside their own SDK, and this app is native
//! Rust with no webview. The maintainer decided on 2026-11-12 to use the raster endpoints anyway
//! (design record: `extension/docs/02-地理分类地图视图设计.md` §3); the map therefore shows the
//! "© 高德地图" attribution, offers a switch to the offline schematic basemap, and keeps tiles in
//! memory only — nothing is written to disk.
//!
//! # Coordinates
//!
//! EXIF GPS is WGS-84; AMap's tiles are GCJ-02 ("Mars coordinates"). Every position is shifted by
//! [`wgs_to_gcj`] before projection, and the inverse [`gcj_to_wgs`] turns a screen position back
//! into WGS-84, so a photo's marker lands on the street the photo was taken on.
//!
//! # Never blocking the UI
//!
//! A small pool of workers fetches and decodes tiles (bounded response size), and the UI thread
//! only uploads finished RGBA images and paints textures. A failed tile is retried after
//! [`RETRY_AFTER`]; while a basemap tile is missing the map paints its schematic base underneath.

use std::collections::{HashMap, HashSet};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock, PoisonError};
use std::time::{Duration, Instant};

use egui::{ColorImage, Context, TextureHandle, TextureId, TextureOptions};

/// The attribution the map paints while the AMap basemap is showing.
pub(crate) const ATTRIBUTION: &str = "© 高德地图 (AutoNavi)";

/// The tile edge in pixels (AMap serves 256 px tiles).
const TILE_PX: f64 = 256.0;
/// Zoom bounds of the raster pyramid.
const MIN_ZOOM: u8 = 3;
const MAX_ZOOM: u8 = 18;
/// A hostile or broken tile must not allocate more than this.
const MAX_TILE_BYTES: usize = 4 << 20;
const MAX_TILE_DIM: u32 = 512;
const MAX_TILE_ALLOC: u64 = 16 << 20;
/// Fetch workers; each holds one request at a time.
const WORKERS: usize = 4;
/// Requests queued or running at once. The map asks for more; the rest wait for the next frame.
const MAX_IN_FLIGHT: usize = 12;
/// Finished tiles uploaded to the GPU per frame (the rest arrive in the following frames).
const MAX_UPLOADS_PER_FRAME: usize = 8;
/// Resident textures; the least recently painted is dropped first.
const MAX_TEXTURES: usize = 192;
/// A failed tile is tried again after this long, so a short network drop recovers.
const RETRY_AFTER: Duration = Duration::from_secs(30);
/// Network fetching is off under `cargo test`: no test may depend on a third-party server.
const NETWORK: bool = !cfg!(test);

// ---------------------------------------------------------------- coordinates

/// Whether a position is outside the area the GCJ-02 shift covers.
fn out_of_china(lat: f64, lon: f64) -> bool {
    !(73.66..=135.05).contains(&lon) || !(3.86..=53.55).contains(&lat)
}

fn transform_lat(x: f64, y: f64) -> f64 {
    let mut ret = -100.0 + 2.0 * x + 3.0 * y + 0.2 * y * y + 0.1 * x * y + 0.2 * x.abs().sqrt();
    ret += (20.0 * (6.0 * x * std::f64::consts::PI).sin() + 20.0 * (2.0 * x * std::f64::consts::PI).sin()) * 2.0 / 3.0;
    ret += (20.0 * (y * std::f64::consts::PI).sin() + 40.0 * (y / 3.0 * std::f64::consts::PI).sin()) * 2.0 / 3.0;
    ret += (160.0 * (y / 12.0 * std::f64::consts::PI).sin() + 320.0 * (y * std::f64::consts::PI / 30.0).sin()) * 2.0 / 3.0;
    ret
}

fn transform_lon(x: f64, y: f64) -> f64 {
    let mut ret = 300.0 + x + 2.0 * y + 0.1 * x * x + 0.1 * x * y + 0.1 * x.abs().sqrt();
    ret += (20.0 * (6.0 * x * std::f64::consts::PI).sin() + 20.0 * (2.0 * x * std::f64::consts::PI).sin()) * 2.0 / 3.0;
    ret += (20.0 * (x * std::f64::consts::PI).sin() + 40.0 * (x / 3.0 * std::f64::consts::PI).sin()) * 2.0 / 3.0;
    ret += (150.0 * (x / 12.0 * std::f64::consts::PI).sin() + 300.0 * (x / 30.0 * std::f64::consts::PI).sin()) * 2.0 / 3.0;
    ret
}

/// WGS-84 → GCJ-02 (the datum AMap's tiles are drawn in). Outside China the input is returned.
pub(crate) fn wgs_to_gcj(lat: f64, lon: f64) -> (f64, f64) {
    if !lat.is_finite() || !lon.is_finite() || out_of_china(lat, lon) {
        return (lat, lon);
    }
    let a = 6_378_245.0;
    let ee = 0.006_693_421_622_965_943;
    let dlat = transform_lat(lon - 105.0, lat - 35.0);
    let dlon = transform_lon(lon - 105.0, lat - 35.0);
    let rad = lat.to_radians();
    let magic = 1.0 - ee * rad.sin() * rad.sin();
    let sqrt_magic = magic.sqrt();
    let dlat = dlat * 180.0 / ((a * (1.0 - ee)) / (magic * sqrt_magic) * std::f64::consts::PI);
    let dlon = dlon * 180.0 / (a / sqrt_magic * rad.cos() * std::f64::consts::PI);
    let out = (lat + dlat, lon + dlon);
    if out.0.is_finite() && out.1.is_finite() { out } else { (lat, lon) }
}

/// GCJ-02 → WGS-84, by fixed-point iteration on [`wgs_to_gcj`] (converges in a few rounds).
pub(crate) fn gcj_to_wgs(lat: f64, lon: f64) -> (f64, f64) {
    if !lat.is_finite() || !lon.is_finite() || out_of_china(lat, lon) {
        return (lat, lon);
    }
    let (mut wlat, mut wlon) = (lat, lon);
    for _ in 0..6 {
        let (glat, glon) = wgs_to_gcj(wlat, wlon);
        let (dlat, dlon) = (glat - lat, glon - lon);
        if !dlat.is_finite() || !dlon.is_finite() {
            return (lat, lon);
        }
        wlat -= dlat;
        wlon -= dlon;
        if dlat.abs() < 1e-10 && dlon.abs() < 1e-10 {
            break;
        }
    }
    (wlat, wlon)
}

/// A position on the world square, both axes in `0..=1` (Web Mercator).
pub(crate) fn world_xy(lat: f64, lon: f64) -> (f64, f64) {
    let lat = lat.clamp(-85.051_128_78, 85.051_128_78);
    let x = (lon + 180.0) / 360.0;
    let s = lat.to_radians().sin();
    let y = 0.5 - ((1.0 + s) / (1.0 - s)).ln() / (4.0 * std::f64::consts::PI);
    if !x.is_finite() || !y.is_finite() {
        return (0.5, 0.5);
    }
    (x.clamp(0.0, 1.0), y.clamp(0.0, 1.0))
}

/// The inverse of [`world_xy`].
pub(crate) fn lat_lon(x: f64, y: f64) -> (f64, f64) {
    let x = x.clamp(0.0, 1.0);
    let y = y.clamp(0.0, 1.0);
    let lon = x * 360.0 - 180.0;
    let lat = (std::f64::consts::PI * (1.0 - 2.0 * y)).sinh().atan().to_degrees();
    (lat, lon)
}

/// A WGS-84 position projected on AMap's datum: GCJ-02, then Web Mercator.
pub(crate) fn world_xy_gcj(lat: f64, lon: f64) -> (f64, f64) {
    let (lat, lon) = wgs_to_gcj(lat, lon);
    world_xy(lat, lon)
}

/// A screen/world position on AMap's datum back to WGS-84.
pub(crate) fn lat_lon_gcj(x: f64, y: f64) -> (f64, f64) {
    let (lat, lon) = lat_lon(x, y);
    gcj_to_wgs(lat, lon)
}

// ---------------------------------------------------------------- the pyramid

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct TileKey {
    pub(crate) z: u8,
    pub(crate) x: u32,
    pub(crate) y: u32,
}

/// The AMap raster URL of a tile (the road basemap with Chinese labels).
pub(crate) fn tile_url(key: TileKey) -> String {
    // four sibling hosts, picked by the tile so a burst spreads over them
    let server = 1 + key.x.wrapping_add(key.y) % 4;
    format!("https://webrd0{server}.is.autonavi.com/appmaptile?lang=zh_cn&size=1&scale=1&style=7&x={}&y={}&z={}", key.x, key.y, key.z)
}

/// The zoom level whose tiles are at least as fine as the screen (`pixels_per_point` makes the
/// choice on HiDPI windows, where a point is more than one physical pixel).
pub(crate) fn zoom_for(width_pt: f32, span_lon: f64, pixels_per_point: f32) -> u8 {
    let width = (width_pt as f64).max(1.0);
    let span = span_lon.max(1e-9);
    let px_per_world = width * 360.0 / span;
    let wanted = px_per_world * (pixels_per_point as f64).clamp(0.5, 4.0) / TILE_PX;
    let zoom = wanted.max(1.0).log2().ceil();
    if !zoom.is_finite() {
        return MIN_ZOOM;
    }
    (zoom as i64).clamp(MIN_ZOOM as i64, MAX_ZOOM as i64) as u8
}

// ---------------------------------------------------------------- the store

struct TexEntry {
    texture: TextureHandle,
    /// The store frame the tile was last painted at (eviction order).
    used: u64,
}

struct DecodedTile {
    width: usize,
    height: usize,
    rgba: Vec<u8>,
}

type TileResult = (TileKey, Result<DecodedTile, ()>);

#[derive(Default)]
struct Store {
    textures: HashMap<TileKey, TexEntry>,
    failed: HashMap<TileKey, Instant>,
    in_flight: HashSet<TileKey>,
    job_tx: Option<Sender<TileKey>>,
    results: Option<Receiver<TileResult>>,
    started: bool,
    frame: u64,
}

static STORE: OnceLock<Mutex<Store>> = OnceLock::new();

fn lock() -> MutexGuard<'static, Store> {
    STORE.get_or_init(|| Mutex::new(Store::default())).lock().unwrap_or_else(PoisonError::into_inner)
}

/// Upload finished tiles and age the resident ones. Called once per frame before painting.
pub(crate) fn pump(ctx: &Context) {
    let mut store = lock();
    store.frame = store.frame.wrapping_add(1);
    let frame = store.frame;
    let mut finished = Vec::new();
    if let Some(rx) = &store.results {
        for _ in 0..MAX_UPLOADS_PER_FRAME {
            match rx.try_recv() {
                Ok(result) => finished.push(result),
                Err(_) => break,
            }
        }
    }
    for (key, result) in finished {
        store.in_flight.remove(&key);
        match result {
            Ok(tile) => {
                let image = ColorImage::from_rgba_unmultiplied([tile.width, tile.height], &tile.rgba);
                let texture = ctx.load_texture(format!("map-tile-{}-{}-{}", key.z, key.x, key.y), image, TextureOptions::LINEAR);
                store.textures.insert(key, TexEntry { texture, used: frame });
                store.failed.remove(&key);
            }
            Err(()) => {
                store.failed.insert(key, Instant::now());
            }
        }
    }
    if !store.in_flight.is_empty() {
        // the workers wake us when they finish; this only keeps the upload drain moving
        ctx.request_repaint_after(Duration::from_millis(100));
    }
    while store.textures.len() > MAX_TEXTURES {
        let Some(oldest) = store.textures.iter().min_by_key(|(_, entry)| entry.used).map(|(key, _)| *key) else { break };
        store.textures.remove(&oldest);
    }
}

/// Ask for a tile unless it is resident, running, or recently failed.
pub(crate) fn request(ctx: &Context, key: TileKey) {
    if !NETWORK {
        return;
    }
    let mut store = lock();
    if store.textures.contains_key(&key) || store.in_flight.contains(&key) {
        return;
    }
    if let Some(at) = store.failed.get(&key) {
        if at.elapsed() < RETRY_AFTER {
            return;
        }
        store.failed.remove(&key);
    }
    if store.in_flight.len() >= MAX_IN_FLIGHT {
        return;
    }
    if !store.started && !ensure_started(&mut store, ctx) {
        return;
    }
    let Some(tx) = store.job_tx.clone() else { return };
    if tx.send(key).is_ok() {
        store.in_flight.insert(key);
    }
}

/// The resident texture of a tile, marking it as freshly used.
pub(crate) fn texture(key: TileKey) -> Option<TextureId> {
    let mut store = lock();
    let frame = store.frame;
    let entry = store.textures.get_mut(&key)?;
    entry.used = frame;
    Some(entry.texture.id())
}

/// Start the fetch workers on the first real frame (never under `cargo test`).
fn ensure_started(store: &mut Store, ctx: &Context) -> bool {
    let (job_tx, job_rx) = mpsc::channel::<TileKey>();
    let (result_tx, result_rx) = mpsc::channel::<TileResult>();
    let job_rx = Arc::new(Mutex::new(job_rx));
    store.job_tx = Some(job_tx);
    store.results = Some(result_rx);
    store.started = true;
    let mut workers = 0usize;
    for index in 0..WORKERS {
        let jobs = job_rx.clone();
        let results = result_tx.clone();
        let ctx = ctx.clone();
        let spawned = std::thread::Builder::new().name(format!("map-tile-{index}")).spawn(move || {
            loop {
                let key = {
                    let jobs = jobs.lock().unwrap_or_else(PoisonError::into_inner);
                    jobs.recv()
                };
                let Ok(key) = key else { break };
                if results.send((key, load(key))).is_err() {
                    break;
                }
                ctx.request_repaint();
            }
        });
        match spawned {
            Ok(_) => workers += 1,
            Err(why) => log::warn!("map tile worker could not start: {why}"),
        }
    }
    workers > 0
}

fn load(key: TileKey) -> Result<DecodedTile, ()> {
    let url = tile_url(key);
    let mut response = agent().get(&url).call().map_err(|why| {
        log::debug!("map tile {url}: {why}");
    })?;
    let bytes = response.body_mut().with_config().limit(MAX_TILE_BYTES as u64).read_to_vec().map_err(|why| log::debug!("map tile {url}: {why}"))?;
    decode(&bytes)
}

/// Decode a tile image, rejecting empty, oversized or malformed input without panicking. The
/// reader's limits apply before the pixels are allocated, so a compressed bomb can't exhaust memory.
fn decode(bytes: &[u8]) -> Result<DecodedTile, ()> {
    if bytes.is_empty() || bytes.len() > MAX_TILE_BYTES {
        return Err(());
    }
    let mut reader = image::ImageReader::new(std::io::Cursor::new(bytes)).with_guessed_format().map_err(|_| ())?;
    let mut limits = image::Limits::default();
    limits.max_image_width = Some(MAX_TILE_DIM);
    limits.max_image_height = Some(MAX_TILE_DIM);
    limits.max_alloc = Some(MAX_TILE_ALLOC);
    reader.limits(limits);
    let image = reader.decode().map_err(|_| ())?;
    let (width, height) = (image.width(), image.height());
    if width == 0 || height == 0 || width > MAX_TILE_DIM || height > MAX_TILE_DIM {
        return Err(());
    }
    let rgba = image.to_rgba8();
    Ok(DecodedTile { width: rgba.width() as usize, height: rgba.height() as usize, rgba: rgba.into_raw() })
}

fn agent() -> &'static ureq::Agent {
    static AGENT: OnceLock<ureq::Agent> = OnceLock::new();
    AGENT
        .get_or_init(|| ureq::Agent::config_builder().timeout_global(Some(Duration::from_secs(10))).user_agent("LightCraft map tiles").build().into())
}

// ---------------------------------------------------------------- tests

#[cfg(test)]
pub(crate) fn reset_for_test() {
    let mut store = lock();
    store.textures.clear();
    store.failed.clear();
    store.in_flight.clear();
    store.job_tx = None;
    store.results = None;
    store.started = false;
    store.frame = 0;
}

/// Put a solid tile into the store, standing in for a downloaded one.
#[cfg(test)]
pub(crate) fn insert_for_test(ctx: &Context, key: TileKey, size: usize) {
    let size = size.max(1);
    let pixels = vec![200u8; size * size * 4];
    let image = ColorImage::from_rgba_unmultiplied([size, size], &pixels);
    let texture = ctx.load_texture("test-map-tile", image, TextureOptions::LINEAR);
    let mut store = lock();
    store.textures.insert(key, TexEntry { texture, used: 0 });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gcj_shifts_china_and_leaves_the_rest_alone() {
        let (lat, lon) = wgs_to_gcj(39.9087, 116.3975);
        let moved = ((lat - 39.9087).powi(2) + (lon - 116.3975).powi(2)).sqrt();
        assert!((0.0005..0.01).contains(&moved), "Beijing shifts by a few hundred metres, got {moved}");
        assert_eq!(wgs_to_gcj(35.68, 139.69), (35.68, 139.69), "Tokyo is outside the GCJ area");
        let (nan_lat, nan_lon) = wgs_to_gcj(f64::NAN, 116.0);
        assert!(nan_lat.is_nan() && nan_lon == 116.0, "hostile input passes through");
    }

    #[test]
    fn gcj_round_trips_through_its_inverse() {
        for (lat, lon) in [(39.9087, 116.3975), (30.2741, 120.1551), (22.5431, 114.0579), (43.8256, 87.6168)] {
            let (glat, glon) = wgs_to_gcj(lat, lon);
            let (wlat, wlon) = gcj_to_wgs(glat, glon);
            assert!((wlat - lat).abs() < 1e-7 && (wlon - lon).abs() < 1e-7, "{lat},{lon} → {wlat},{wlon}");
        }
    }

    #[test]
    fn mercator_round_trips() {
        for (lat, lon) in [(0.0, 0.0), (36.2, 104.5), (-33.9, 151.2), (60.0, -120.0)] {
            let (x, y) = world_xy(lat, lon);
            let (lat2, lon2) = lat_lon(x, y);
            assert!((lat2 - lat).abs() < 1e-9 && (lon2 - lon).abs() < 1e-9, "{lat},{lon} → {lat2},{lon2}");
        }
        let (x, y) = world_xy(0.0, 0.0);
        assert!((x - 0.5).abs() < 1e-12 && (y - 0.5).abs() < 1e-12, "the origin is the centre of the world square");
    }

    #[test]
    fn zoom_follows_the_span_and_screen_density() {
        // 500 pt of screen across 62°: z4 (a China-wide tile set)
        assert_eq!(zoom_for(500.0, 62.0, 1.0), 4);
        // the same width across 1.2° needs roughly 16× the tiles in each direction
        let deep = zoom_for(500.0, 1.2, 1.0);
        assert!(deep >= 9, "a city view uses a deep level, got {deep}");
        // HiDPI asks for one more level
        assert!(zoom_for(500.0, 62.0, 2.0) > zoom_for(500.0, 62.0, 1.0));
        // the bounds always hold, even for hostile spans
        assert!((MIN_ZOOM..=MAX_ZOOM).contains(&zoom_for(500.0, 1e-12, 1.0)));
        assert!((MIN_ZOOM..=MAX_ZOOM).contains(&zoom_for(500.0, 1e12, 1.0)));
    }

    #[test]
    fn the_tile_url_names_the_amap_raster_service() {
        let url = tile_url(TileKey { z: 4, x: 13, y: 6 });
        assert!(url.starts_with("https://webrd0"), "{url}");
        assert!(url.contains(".is.autonavi.com/appmaptile?"), "{url}");
        assert!(url.contains("style=7"), "the road basemap: {url}");
        assert!(url.contains("x=13&y=6&z=4"), "{url}");
    }

    #[test]
    fn a_hostile_tile_is_rejected_without_panicking() {
        assert!(decode(&[]).is_err(), "empty input");
        assert!(decode(b"not an image").is_err(), "garbage input");
        assert!(decode(&vec![0u8; MAX_TILE_BYTES + 1]).is_err(), "oversized input");
        // a valid tiny PNG still decodes
        let image = image::DynamicImage::ImageRgba8(image::RgbaImage::new(2, 2));
        let mut bytes = Vec::new();
        image.write_to(&mut std::io::Cursor::new(&mut bytes), image::ImageFormat::Png).unwrap();
        let tile = decode(&bytes).unwrap();
        assert_eq!((tile.width, tile.height, tile.rgba.len()), (2, 2, 16));
        // a small file claiming huge dimensions is refused by the reader's limits
        let wide = image::DynamicImage::ImageRgba8(image::RgbaImage::new(MAX_TILE_DIM + 1, 1));
        let mut bytes = Vec::new();
        wide.write_to(&mut std::io::Cursor::new(&mut bytes), image::ImageFormat::Png).unwrap();
        assert!(decode(&bytes).is_err(), "an oversized image is refused");
    }
}
