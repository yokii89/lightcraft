//! The map album's persistent store (design: `extension/docs/05-地图相册持久化与自动增量设计.md`):
//! the classification of every photo the feature has placed on the map, saved as
//! `<library>/geo-classify.json` — bound to the library, like the Auto Keywords settings file.
//!
//! The album is *derived* data: every entry is the offline reverse lookup of the photo's GPS, so a
//! damaged, oversized or unknown-version file reads as empty and the next sync rebuilds it — the
//! page never blocks on it and nothing warns. Entries record the GPS the lookup was computed from
//! (the *basis*): the map draws the photo's current GPS live from the catalog, and the sync
//! re-classifies an entry when the two diverge. The display name is never stored; it comes from
//! the catalog at draw time, so renames need no sync.
//!
//! The file is user data read back from disk: held to [`MAX_FILE_BYTES`], every field carries a
//! serde default, and nothing unreadable is an error. When the library is not on disk (the demo
//! and scratch libraries) there is no file and the album lives in memory for the session.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use lightcraft_catalog::PhotoId;
use lightcraft_catalog::safe_file::write_atomic;

/// The store file's name inside the library directory.
pub const FILE_NAME: &str = "geo-classify.json";

/// Of this file format. A file of another version reads as empty (the sync rebuilds it).
pub const VERSION: u32 = 1;

/// The largest store file read: about 700 000 photos' worth of entries, far past any library.
/// Anything bigger is junk, not an album.
const MAX_FILE_BYTES: u64 = 64 << 20; // 64 MiB

/// One photo's place in the album.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct Entry {
    /// The catalog photo id (stable within the library).
    pub id: u64,
    /// The looked-up country name; empty when the lookup found no place (recorded so later syncs
    /// skip the photo instead of re-looking it up every time).
    pub country: String,
    /// The looked-up city, empty when the photo classifies by country alone or matched nothing.
    pub city: String,
    /// ISO 3166-1 alpha-2 of the country ("CN" keeps the group on the map).
    pub iso: String,
    /// Metres from the photo's position to the place found.
    pub distance: f64,
    /// The GPS the classification was computed from. The map draws the photo's current GPS; a
    /// diverging basis makes the sync re-classify.
    pub lat: f64,
    pub lon: f64,
}

impl Default for Entry {
    fn default() -> Self {
        Entry { id: 0, country: String::new(), city: String::new(), iso: String::new(), distance: 0.0, lat: 0.0, lon: 0.0 }
    }
}

impl Entry {
    /// Whether the lookup found no place for the photo.
    pub fn unmatched(&self) -> bool {
        self.country.is_empty()
    }

    /// Whether this entry classifies the same GPS position as `(lat, lon)`.
    pub fn at_basis(&self, lat: f64, lon: f64) -> bool {
        self.lat == lat && self.lon == lon
    }
}

/// The album: every classified photo of the library, by photo id (order-stable file, cheap
/// lookups during the sync), plus the parent keyword the paths are written under.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct Album {
    /// Of this file format.
    pub version: u32,
    /// The parent keyword the place paths go under (`{parent}|{country}|{city}`); empty classifies
    /// by country alone.
    pub parent: String,
    pub photos: BTreeMap<u64, Entry>,
    /// GPX files overlaid on the map, stored as user-selected paths and loaded on demand.
    pub tracks: Vec<String>,
}

impl Album {
    /// The entry of `id`, if the album holds one.
    pub fn entry(&self, id: PhotoId) -> Option<&Entry> {
        self.photos.get(&id.0)
    }

    /// Put `entry` in the album, replacing any entry of the same photo.
    pub fn insert(&mut self, entry: Entry) {
        self.photos.insert(entry.id, entry);
    }

    /// Drop the photo's entry, when there is one. Whether something was dropped.
    pub fn remove(&mut self, id: PhotoId) -> bool {
        self.photos.remove(&id.0).is_some()
    }
}

/// The store file's place next to the library. `None`: no library on disk — the album then lives
/// in memory for this session only.
pub fn path(library_dir: Option<&Path>) -> Option<PathBuf> {
    library_dir.map(|dir| dir.join(FILE_NAME))
}

/// Read the store file. `None` when it is missing, oversized, damaged, of another version, or of
/// the wrong shape: the album then starts empty and the sync rebuilds it.
pub fn read(path: &Path) -> Option<Album> {
    let Ok(bytes) = std::fs::File::open(path).and_then(|f| {
        use std::io::Read;
        let mut bytes = Vec::new();
        f.take(MAX_FILE_BYTES + 1).read_to_end(&mut bytes).map(|_| bytes)
    }) else {
        return None;
    };
    if bytes.len() as u64 > MAX_FILE_BYTES {
        return None;
    }
    let album: Album = serde_json::from_slice(&bytes).ok()?;
    (album.version == VERSION).then_some(album)
}

/// Write the store file atomically, compact (a big library's album is tens of megabytes). A
/// failure is reported to the caller: the page keeps working and the next save tries again.
pub fn write(path: &Path, album: &Album) -> Result<(), String> {
    let mut album = album.clone();
    album.version = VERSION;
    let mut bytes = serde_json::to_vec(&album).map_err(|e| e.to_string())?;
    bytes.push(b'\n');
    write_atomic(path, &bytes).map_err(|e| format!("{}: {e}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(id: u64, country: &str, city: &str, iso: &str) -> Entry {
        Entry { id, country: country.to_string(), city: city.to_string(), iso: iso.to_string(), distance: 8342.5, lat: 30.25, lon: 120.16 }
    }

    #[test]
    fn round_trips_through_the_file() {
        let dir = std::env::temp_dir().join(format!("lightcraft-geo-store-{}", std::process::id()));
        let _ = std::fs::create_dir(&dir);
        let path = path(Some(&dir)).expect("the library dir gives a path");
        let mut album = Album { version: VERSION, parent: "地点".to_string(), photos: BTreeMap::new(), tracks: Vec::new() };
        album.tracks.push("/photos/trip.gpx".to_string());
        album.insert(entry(42, "中国", "杭州", "CN"));
        album.insert(Entry { id: 43, country: String::new(), ..Entry::default() });
        write(&path, &album).expect("the write lands");
        let read = read(&path).expect("the file reads back");
        assert_eq!(read, album);
        assert_eq!(read.parent, "地点");
        assert_eq!(read.tracks, vec!["/photos/trip.gpx"]);
        let e = read.entry(PhotoId(42)).expect("the photo is in the album");
        assert_eq!((e.country.as_str(), e.city.as_str(), e.iso.as_str()), ("中国", "杭州", "CN"));
        assert!(read.entry(PhotoId(43)).expect("the unmatched photo is in the album").unmatched());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn damaged_oversized_and_foreign_files_read_as_empty() {
        let dir = std::env::temp_dir().join(format!("lightcraft-geo-store-bad-{}", std::process::id()));
        let _ = std::fs::create_dir(&dir);
        let path = path(Some(&dir)).expect("the library dir gives a path");
        // a damaged file
        std::fs::write(&path, "{\"version\":1,\"photos\":").expect("the damage lands");
        assert_eq!(read(&path), None, "damaged JSON reads as nothing");
        // an unknown version: the sync rebuilds instead of guessing
        std::fs::write(&path, "{\"version\":99,\"parent\":\"x\",\"photos\":{}}").expect("the damage lands");
        assert_eq!(read(&path), None, "a foreign version reads as nothing");
        // an oversized file is junk, not an album
        let junk = vec![b'x'; (MAX_FILE_BYTES + 1) as usize];
        std::fs::write(&path, &junk).expect("the damage lands");
        assert_eq!(read(&path), None, "an oversized file reads as nothing");
        // a missing file
        assert_eq!(read(&dir.join("absent.json")), None);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn albums_from_before_track_overlays_default_to_no_tracks() {
        let json = r#"{"version":1,"parent":"地点","photos":{}}"#;
        let album: Album = serde_json::from_str(json).expect("the v1 album remains readable");
        assert!(album.tracks.is_empty());
    }

    #[test]
    fn entries_replace_and_remove_by_photo_id() {
        let mut album = Album::default();
        album.insert(entry(7, "中国", "杭州", "CN"));
        album.insert(entry(3, "中国", "北京", "CN"));
        assert_eq!(album.photos.len(), 2);
        // replacing keeps one entry per photo
        album.insert(entry(7, "中国", "上海", "CN"));
        assert_eq!(album.photos.len(), 2);
        assert_eq!(album.entry(PhotoId(7)).expect("the photo is in the album").city, "上海");
        assert!(album.remove(PhotoId(7)), "removing an existing photo reports it");
        assert!(!album.remove(PhotoId(7)), "removing it again reports nothing");
        assert_eq!(album.photos.len(), 1);
        assert_eq!(album.entry(PhotoId(3)).expect("the other photo stays"), &entry(3, "中国", "北京", "CN"));
        assert!(Album::default().photos.is_empty());
        assert!(!album.photos.is_empty());
    }

    #[test]
    fn the_basis_compare_drives_the_recheck() {
        let e = entry(1, "中国", "杭州", "CN");
        assert!(e.at_basis(30.25, 120.16), "the same position needs no re-lookup");
        assert!(!e.at_basis(31.0, 120.16), "a moved photo is re-classified");
        assert!(!e.at_basis(f64::NAN, 120.16), "an unusable position is re-classified");
    }
}
