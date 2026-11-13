use super::*;
use lightcraft_catalog::{Meta, Op, PhotoId, Source};
use lightcraft_engine::{Session, activity::Cancel};

struct Fixture {
    session: Session,
    view: View,
    dir: PathBuf,
    ids: Vec<PhotoId>,
}

impl Fixture {
    fn new(name: &str) -> Self {
        let mut session = Session::with_demo();
        let dir = super::super::tests::temp(name);
        let ids: Vec<_> = session.catalog.photos().take(3).map(|p| p.id).collect();
        for id in &ids {
            session.catalog.apply(Op::SetMeta { id: *id, meta: Box::new(Meta { keywords: vec!["dog".into()], ..Meta::default() }) }).unwrap();
        }
        let view = View::new(Library { generation: session.library_generation(), dir: Some(dir.clone()) }, &session.catalog).unwrap();
        Self { session, view, dir, ids }
    }

    fn description(&self, id: PhotoId) -> Description {
        Description {
            id: id.0,
            basis: super::super::store::basis(self.session.catalog.photo(id).unwrap()),
            description: "A dog running through grass".into(),
            guard_keywords: vec!["dog".into()],
        }
    }

    fn seed(&self) {
        let mut descriptions = Descriptions::default();
        descriptions.accept(self.ids.iter().map(|id| self.description(*id)).collect()).unwrap();
        descriptions.save(Some(&self.dir)).unwrap();
        let mut index = Index::default();
        for id in &self.ids {
            let photo = PhotoText::from_photo(self.session.catalog.photo(*id).unwrap());
            index.records.insert(id.0, Vector { fingerprint: photo.fingerprint(&photo.text(&descriptions)), values: super::super::tests::vector(0) });
        }
        index.save(Some(&self.dir)).unwrap();
    }

    fn request(&self, operation: Operation) -> Request {
        let (reply, _) = sync_channel(1);
        Request {
            library: self.view.library.clone(),
            view: self.view.clone(),
            operation,
            cancel: Arc::new(AtomicBool::new(false)),
            guard: self.session.activity.start("semanticMaintenance", "Saving photo descriptions", Cancel::No),
            reply,
        }
    }

    fn apply(&mut self, op: Op) {
        self.session.catalog.apply(op).unwrap();
        self.view.refresh(&self.session.catalog).unwrap();
    }

    fn assert_cached(&self, ids: &[PhotoId]) {
        let descriptions = Descriptions::load(Some(&self.dir)).unwrap();
        let index = Index::load(Some(&self.dir)).unwrap();
        let expected: Vec<_> = ids.iter().map(|id| id.0).collect();
        assert_eq!(descriptions.photos.keys().copied().collect::<Vec<_>>(), expected);
        assert_eq!(index.records.keys().copied().collect::<Vec<_>>(), expected);
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.dir).unwrap();
    }
}

#[test]
fn trash_and_restore_keep_caches_while_stale_search_requests_follow_the_catalog() {
    let mut f = Fixture::new("trash-restore");
    f.seed();
    let photos = f.session.catalog.photos().cloned().collect::<Vec<_>>();
    let search = f.request(Operation::Search { photos: photos.clone(), query: "dog".into(), minimum: 0.4 });
    let id = f.ids[0];
    f.apply(Op::SetDeleted { id, deleted: true });
    let mut cache = Cache::default();
    cache.execute(&f.request(Operation::Reconcile)).unwrap();
    f.assert_cached(&f.ids);
    // Updating an index while everything is trashed must also preserve recoverable vectors.
    for id in f.ids.clone() {
        f.apply(Op::SetDeleted { id, deleted: true });
    }
    cache.execute(&f.request(Operation::Index(vec![]))).unwrap();
    f.assert_cached(&f.ids);
    // No model is needed to prove visibility/BM25 behaviour.
    cache.index = Some(Index::default());
    assert!(cache.execute(&search).unwrap().hits.is_empty());
    f.apply(Op::SetDeleted { id, deleted: false });
    assert_eq!(cache.execute(&search).unwrap().hits.iter().map(|h| h.id).collect::<Vec<_>>(), vec![id.0]);
}

#[test]
fn reopen_reconciles_permanent_deletion_and_library_removal_but_keeps_missing_sources() {
    let mut f = Fixture::new("reopen");
    let mut missing = (**f.session.catalog.photo(f.ids[2]).unwrap()).clone();
    missing.source = Source::File { path: f.dir.join("offline-drive/photo.jpg").to_string_lossy().into_owned() };
    f.apply(Op::RemovePhoto { id: missing.id });
    f.apply(Op::AddPhoto { photo: Box::new(missing) });
    f.seed();
    f.apply(Op::RemovePhoto { id: f.ids[0] });
    f.apply(Op::SetLocal { id: f.ids[1], local: true });
    let mut cache = Cache::default();
    let request = f.request(Operation::Reconcile);
    assert!(f.session.activity.list().iter().any(|t| t.kind == "semanticMaintenance"));
    cache.execute(&request).unwrap();
    f.assert_cached(&f.ids[2..]);
    assert!(f.view.current(f.ids[2].0).unwrap().in_library());
    assert!(cache.encoder.is_none(), "reconciliation never loads the GPU model");
}

#[test]
fn delayed_saves_reject_deleted_replaced_and_unaccepted_results() {
    let mut f = Fixture::new("stale-save");
    let entries = f.ids.iter().map(|id| f.description(*id)).collect();
    let request = f.request(Operation::Save(entries));
    f.apply(Op::RemovePhoto { id: f.ids[0] });
    f.apply(Op::SetDeleted { id: f.ids[1], deleted: true });
    f.apply(Op::SetMeta { id: f.ids[2], meta: Box::new(Meta::default()) });
    let mut cache = Cache::default();
    assert_eq!(cache.execute(&request).unwrap().descriptions, 0);
    assert!(!f.dir.join("semantic-descriptions.json").exists());
    // Same ID, different content/version, while inference was in flight.
    let id = f.ids[1];
    f.apply(Op::SetDeleted { id, deleted: false });
    let stale = f.request(Operation::Save(vec![f.description(id)]));
    let mut replacement = (**f.session.catalog.photo(id).unwrap()).clone();
    replacement.content_hash = Some("replacement".into());
    f.apply(Op::RemovePhoto { id });
    f.apply(Op::AddPhoto { photo: Box::new(replacement) });
    assert_eq!(cache.execute(&stale).unwrap().descriptions, 0);
    f.view.invalidate();
    assert_eq!(cache.execute(&request).err().as_deref(), Some("Cancelled"));
}

#[test]
fn vector_inference_cannot_commit_after_deletion_or_a_text_version_change() {
    let mut f = Fixture::new("stale-vector");
    let cache = Cache::default();
    let mut index = Index::default();
    for (i, id) in f.ids.clone().into_iter().enumerate() {
        let photo = PhotoText::from_photo(f.session.catalog.photo(id).unwrap());
        let text = photo.text(&cache.descriptions);
        let op = match i {
            0 => Op::RemovePhoto { id },
            1 => Op::SetDeleted { id, deleted: true },
            _ => Op::SetMeta { id, meta: Box::new(Meta { caption: "changed".into(), ..Meta::default() }) },
        };
        f.apply(op);
        assert!(!cache.record_vector(&f.view, &photo, &text, super::super::tests::vector(0), &mut index));
    }
    assert!(index.records.is_empty());
}

#[test]
fn failed_atomic_cleanup_preserves_both_files_and_retries_without_another_catalog_change() {
    let mut f = Fixture::new("cleanup-retry");
    f.seed();
    let text = std::fs::read(f.dir.join("semantic-descriptions.json")).unwrap();
    let vectors = std::fs::read(f.dir.join("semantic-vectors.bin")).unwrap();
    f.apply(Op::RemovePhoto { id: f.ids[0] });
    let mut cache = Cache::default();
    {
        let _fault = lightcraft_catalog::safe_file::fail_writes_after(0);
        assert!(cache.execute(&f.request(Operation::Reconcile)).is_err());
    }
    assert_eq!(std::fs::read(f.dir.join("semantic-descriptions.json")).unwrap(), text);
    assert_eq!(std::fs::read(f.dir.join("semantic-vectors.bin")).unwrap(), vectors);
    assert!(cache.descriptions_dirty && cache.index_dirty);
    cache.execute(&f.request(Operation::Reconcile)).unwrap();
    f.assert_cached(&f.ids[1..]);
    assert!(!cache.descriptions_dirty && !cache.index_dirty);
}

#[test]
fn each_cache_is_cleaned_even_when_the_other_cannot_be_replaced() {
    for blocked in ["semantic-descriptions.json", "semantic-vectors.bin"] {
        let mut f = Fixture::new(&format!("independent-{blocked}"));
        f.seed();
        let mut cache = Cache::default();
        cache.execute(&f.request(Operation::Reconcile)).unwrap();
        let path = f.dir.join(blocked);
        let backup = path.with_extension("backup");
        std::fs::rename(&path, &backup).unwrap();
        std::fs::create_dir(&path).unwrap();
        f.apply(Op::RemovePhoto { id: f.ids[0] });
        assert!(cache.execute(&f.request(Operation::Reconcile)).is_err());
        if blocked.ends_with("json") {
            assert!(!Index::load(Some(&f.dir)).unwrap().records.contains_key(&f.ids[0].0));
        } else {
            assert!(!Descriptions::load(Some(&f.dir)).unwrap().photos.contains_key(&f.ids[0].0));
        }
        std::fs::remove_dir(&path).unwrap();
        std::fs::rename(backup, path).unwrap();
        cache.execute(&f.request(Operation::Reconcile)).unwrap();
        f.assert_cached(&f.ids[1..]);
    }
}

#[test]
fn corrupt_descriptions_do_not_prevent_orphan_vector_cleanup() {
    let mut f = Fixture::new("corrupt-cleanup");
    f.seed();
    f.apply(Op::RemovePhoto { id: f.ids[0] });
    std::fs::write(f.dir.join("semantic-descriptions.json"), b"{").unwrap();
    let mut cache = Cache::default();
    assert!(cache.execute(&f.request(Operation::Reconcile)).is_err());
    assert!(!Index::load(Some(&f.dir)).unwrap().records.contains_key(&f.ids[0].0));
    assert_eq!(std::fs::read(f.dir.join("semantic-descriptions.json")).unwrap(), b"{");
}

#[test]
fn removing_album_membership_keeps_descriptions_vectors_and_library_search() {
    let mut f = Fixture::new("album-removal");
    let album = lightcraft_catalog::AlbumId(999);
    f.apply(Op::AddAlbum { album: lightcraft_catalog::Album::new(album, "Temporary album") });
    f.apply(Op::SetAlbumPhotos { id: album, photos: f.ids.clone() });
    f.seed();
    f.apply(Op::SetAlbumPhotos { id: album, photos: vec![] });
    let mut cache = Cache::default();
    cache.execute(&f.request(Operation::Reconcile)).unwrap();
    f.assert_cached(&f.ids);
    cache.index = Some(Index::default());
    let photos = f.session.catalog.photos().cloned().collect();
    let reply = cache.execute(&f.request(Operation::Search { photos, query: "dog".into(), minimum: 0.4 })).unwrap();
    for id in &f.ids {
        assert!(reply.hits.iter().any(|h| h.id == id.0));
    }
}
