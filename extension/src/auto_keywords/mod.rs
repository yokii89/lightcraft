//! The Auto Keywords page (design: `extension/docs/00-本地模型自动关键词设计.md`): a local vision
//! model tags photos with one fixed category and a few keywords each, written
//! `{parent}|{category}|{keyword}` through the ordinary keyword system as one undo step.
//!
//! The model runs on the self-hosted Python service (see [`server`], lazy start, quick release:
//! the 「启动模型」 toggle spawns it, and it shuts itself down after five idle minutes). Layout
//! (§5): the UI thread renders the page, enumerates the chosen scope and feeds one [`RenderJob`]
//! at a time to a worker thread; the worker renders, encodes, asks the local model service (see
//! [`vlm`]) and hands the cleaned result back over a channel, which the UI thread drains each
//! frame into review or (in the direct mode) the write. All page state lives in a static
//! `OnceLock<Mutex<State>>`: the page survives being closed mid-run, the worker talks to it only
//! through the channels, and a poisoned mutex is recovered with `into_inner`, never a panic. The
//! write goes through `Session::execute_fn`, which carries the panic guard, the XMP sidecar
//! auto-write and the durable save of any other command.

mod config;
mod prompt;
mod server;
mod vlm;

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, Sender, TryRecvError, channel};
use std::sync::{Mutex, MutexGuard, OnceLock, PoisonError};
use std::time::{Duration, Instant};

use egui::{Align, RichText, Ui};
use lightcraft_catalog::{Meta, Op, PhotoId};
use lightcraft_codecs::{ChromaSubsampling, EncodeImage, EncodeMeta, encode_jpeg};
use lightcraft_engine::RenderJob;
use lightcraft_engine::Session;
use lightcraft_engine::activity::{Cancel, TaskGuard};
use lightcraft_engine::catalog::keywords;
use lightcraft_engine::cmd::keywords as keywords_cmd;
use lightcraft_ui_egui::button::Button;
use lightcraft_ui_egui::icons::Icon;
use lightcraft_ui_egui::progress::{Progress, fraction_of, spinner};
use lightcraft_ui_egui::text_field::TextField;
use lightcraft_ui_egui::theme::Tokens;
use lightcraft_ui_egui::widgets;
use lightcraft_ui_egui::{LightcraftApp, i18n};
use serde_json::Value;

use crate::ui::{card, field, fmt_seconds, hint, section, status_dot, trn, warning};
use config::ApplyMode;

/// The JPEG quality of the image sent to the model (§12): small enough for a quick hand-off to
/// the local service, good enough to classify.
const JPEG_QUALITY: u8 = 85;
/// The long edge of the image sent to the model (§12).
const SEND_EDGE: usize = 1024;
/// How many per-photo durations the ETA averages over (§12).
const ETA_WINDOW: usize = 10;
/// The label of the write's single undo step, and of the activity row (§8). English; the UI
/// translates both from this source string.
const UNDO_LABEL: &str = "Auto Keywords";

/// The page's id in the Extensions panel.
pub(crate) const PAGE_ID: &str = "auto-keywords";

// ---------------------------------------------------------------- state

/// Where the page is (§9). `Idle` and `Ready` both draw the settings page; the rest have their
/// own. `Ready` follows the model service: it is entered and left as the service starts up and
/// (the idle timeout, a manual stop, a crash) goes away again.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum Phase {
    #[default]
    Idle,
    Ready,
    Running,
    Review,
    /// Momentary: the batch is being built and committed (the UI passes through in one frame).
    Applying,
    Done,
    /// The run ended with nothing usable at all (§9 Failed): the failure summary, settings kept.
    Failed,
}

/// One photo's generated keywords.
#[derive(Clone)]
struct PhotoResult {
    id: PhotoId,
    name: String,
    category: String,
    keywords: Vec<String>,
    seconds: f64,
}

/// One photo that did not produce keywords.
struct PhotoFailure {
    id: PhotoId,
    name: String,
    reason: vlm::Reason,
    seconds: f64,
}

/// What a worker job came back as.
enum Outcome {
    Done(PhotoResult),
    Failed(PhotoFailure),
}

/// Everything a job needs, fixed when the run started (mid-run edits of the settings change the
/// next run, not this one).
struct RunConfig {
    endpoint: vlm::Endpoint,
    token: String,
    model: String,
    prompt: String,
    max_keywords: usize,
    parent: String,
    stall_secs: u32,
}

/// UI thread → worker: one photo to render and ask about.
struct Job {
    config: Arc<RunConfig>,
    id: PhotoId,
    name: String,
    render: RenderJob,
}

/// Worker → UI thread: one photo's outcome.
struct WorkerDone {
    outcome: Outcome,
}

/// What the write of a batch came to.
struct Applied {
    photos: usize,
    keywords: usize,
    elapsed: Duration,
    undone: bool,
}

/// The page's whole state, shared by every visit of the page (§5). Only the UI thread locks it;
/// workers talk through the channels.
#[derive(Default)]
struct State {
    settings: config::Settings,
    /// The library generation the settings were loaded for: reopening another library reloads.
    loaded_gen: Option<u64>,
    /// The deadline of the debounced settings save, set when a control changes (§13).
    save_at: Option<Instant>,
    /// The photos the Entire-library scope names, cached per catalog revision (a big library is
    /// not walked every frame).
    lib_count: Option<(u64, usize)>,
    phase: Phase,
    /// The model service's last known status, refreshed from [`server::poll`] every frame.
    service: server::Status,
    // the run
    scope: Scope,
    /// The queue: photo id and display name, videos and (when asked) already-tagged photos left
    /// out (§9).
    entries: Vec<(PhotoId, String)>,
    next_index: usize,
    skipped: usize,
    in_flight: bool,
    current_name: String,
    run_config: Option<Arc<RunConfig>>,
    run_direct: bool,
    run_parent: String,
    results: Vec<PhotoResult>,
    failures: Vec<PhotoFailure>,
    /// Per-photo seconds of the last [`ETA_WINDOW`] photos, for the ETA (§12).
    durations: Vec<f64>,
    started: Option<Instant>,
    job_tx: Option<Sender<Job>>,
    done_rx: Option<Receiver<WorkerDone>>,
    done_feeding: Arc<AtomicBool>,
    cancel: Arc<AtomicBool>,
    activity: Option<TaskGuard>,
    // the review
    excluded: HashSet<u64>,
    removed: HashMap<u64, HashSet<String>>,
    open: HashSet<u64>,
    // the outcome
    applied: Option<Applied>,
    apply_error: Option<String>,
}

impl State {
    /// Every photo the run was started for, skipped ones included (§9: skipped ones count in the
    /// progress).
    fn total(&self) -> usize {
        self.entries.len() + self.skipped
    }

    /// The photos that came back so far, skipped ones included.
    fn processed(&self) -> usize {
        self.results.len() + self.failures.len() + self.skipped
    }

    /// What review would write now: photos with at least one kept keyword, and the kept count.
    fn review_counts(&self) -> (usize, usize) {
        let mut photos = 0;
        let mut count = 0;
        for r in &self.results {
            if self.excluded.contains(&r.id.0) {
                continue;
            }
            let kept = self.kept(r).count();
            if kept > 0 {
                photos += 1;
                count += kept;
            }
        }
        (photos, count)
    }

    /// The keywords of `r` the review keeps: its photo not taken, the removed ones left out.
    fn kept<'a>(&'a self, r: &'a PhotoResult) -> impl Iterator<Item = &'a String> {
        let off = self.removed.get(&r.id.0);
        r.keywords.iter().filter(move |kw| !off.is_some_and(|off| off.contains(&kw.to_lowercase())))
    }

    /// The ETA from the average of the last photos, as (minutes, seconds).
    fn eta(&self) -> Option<(u64, u64)> {
        let remaining = self.total().saturating_sub(self.processed());
        if remaining == 0 || self.durations.is_empty() {
            return None;
        }
        let avg = self.durations.iter().sum::<f64>() / self.durations.len() as f64;
        let secs = (avg * remaining as f64).round() as u64;
        (secs >= 1).then_some((secs / 60, secs % 60))
    }
}

static STATE: OnceLock<Mutex<State>> = OnceLock::new();

fn lock() -> MutexGuard<'static, State> {
    STATE.get_or_init(|| Mutex::new(State::default())).lock().unwrap_or_else(PoisonError::into_inner)
}

// ---------------------------------------------------------------- entry points

/// The Extensions panel's Auto Keywords page.
pub(crate) fn page(app: &mut LightcraftApp, ui: &mut Ui) {
    let mut st = lock();
    load_settings(app, &mut st);
    pump(app, ui.ctx(), &mut st);
    widgets::register(ui, format!("page:{PAGE_ID}"), ui.max_rect());
    draw(app, ui, &mut st);
}

/// Read the settings file when the library changed (§13); without a library on disk, settings
/// stay in memory.
fn load_settings(app: &LightcraftApp, st: &mut State) {
    let library_gen = app.session.library_generation();
    if st.loaded_gen == Some(library_gen) {
        return;
    }
    st.loaded_gen = Some(library_gen);
    if let Some(path) = config::settings_path(library_dir(app).as_deref()) {
        st.settings = config::read(&path);
    }
    st.save_at = None;
}

fn library_dir(app: &LightcraftApp) -> Option<PathBuf> {
    app.session.library.as_ref().filter(|l| l.on_disk).map(|l| l.dir.clone())
}

/// Write the settings file now: the debounced save went due, or a run started (§13).
fn save_settings(app: &mut LightcraftApp, ctx: &egui::Context, st: &mut State) {
    st.save_at = None;
    let Some(path) = config::settings_path(library_dir(app).as_deref()) else { return };
    if let Err(e) = config::write(&path, &st.settings) {
        log::warn!("auto keywords: the settings were not saved: {e}");
        app.toast(ctx, i18n::tr("The settings could not be saved."));
    }
}

fn save_soon(st: &mut State) {
    st.save_at = Some(Instant::now() + Duration::from_millis(500));
}

/// The per-frame work between renders: follow the model service, save what is due, feed and
/// drain the worker, end the run (§5's UI row of the diagram).
fn pump(app: &mut LightcraftApp, ctx: &egui::Context, st: &mut State) {
    // the service's status every frame: it starts, becomes ready, or goes away (the idle
    // timeout, a crash, the toggle) independently of what the page shows (§6.5)
    st.service = server::poll();
    match (st.phase, st.service.phase) {
        (Phase::Idle, server::Phase::Ready) => st.phase = Phase::Ready,
        (Phase::Ready, server::Phase::Ready) => {}
        (Phase::Ready, _) => st.phase = Phase::Idle,
        _ => {}
    }
    if st.phase == Phase::Idle && st.service.phase == server::Phase::Starting {
        // keep the starting page animating while the model downloads or loads
        ctx.request_repaint_after(Duration::from_millis(200));
    } else if st.phase == Phase::Ready {
        // a small heartbeat, so the ready time ticks and a service exit is noticed
        ctx.request_repaint_after(Duration::from_secs(1));
    }
    if st.save_at.is_some_and(|at| Instant::now() >= at) {
        save_settings(app, ctx, st);
    }
    if st.phase != Phase::Running {
        return;
    }
    let cancelled = st.cancel.load(Ordering::Relaxed);
    // drain the worker: every finished photo lands in results or failures (a job dropped by a
    // stop is counted as unprocessed, §9)
    loop {
        let received = st.done_rx.as_ref().map(|rx| rx.try_recv());
        match received {
            Some(Ok(done)) => {
                st.in_flight = false;
                match done.outcome {
                    Outcome::Done(r) => {
                        st.durations.push(r.seconds);
                        st.results.push(r);
                    }
                    Outcome::Failed(f) => {
                        if !matches!(f.reason, vlm::Reason::Stopped) {
                            st.durations.push(f.seconds);
                            st.failures.push(f);
                        }
                    }
                }
                while st.durations.len() > ETA_WINDOW {
                    st.durations.remove(0);
                }
            }
            Some(Err(TryRecvError::Empty)) => break,
            Some(Err(TryRecvError::Disconnected)) => {
                st.done_rx = None;
                break;
            }
            None => break,
        }
    }
    // feed the next photo (§5: the UI thread builds one RenderJob at a time)
    if !st.in_flight && !cancelled {
        if let Some((id, name)) = st.entries.get(st.next_index).cloned() {
            match app.session.render_job(id, SEND_EDGE, SEND_EDGE, false, false) {
                Some(render) => {
                    let sent = match (&st.job_tx, &st.run_config) {
                        (Some(tx), Some(config)) => tx.send(Job { config: config.clone(), id, name: name.clone(), render }).is_ok(),
                        _ => false,
                    };
                    if sent {
                        st.in_flight = true;
                        st.current_name = name.clone();
                        if let Some(guard) = &st.activity {
                            guard.detail(&st.current_name);
                        }
                    } else {
                        // the worker is gone: the run cannot go on, end it with what came back
                        finish_run(app, st);
                        return;
                    }
                }
                None => {
                    // the photo is gone from the catalog since the run started
                    st.failures.push(PhotoFailure { id, name, reason: vlm::Reason::Missing, seconds: 0.0 });
                    st.durations.push(0.0);
                }
            }
            st.next_index += 1;
        } else {
            st.done_feeding.store(true, Ordering::Release);
        }
    }
    if let Some(guard) = &st.activity {
        guard.progress(st.processed() as u64, st.total() as u64);
    }
    // the run is over when nothing is in flight and either everything was fed or a stop was
    // asked — whether or not the worker is still draining its exit (its channel may already be
    // gone: the worker leaves as soon as the queue it was given is worked off)
    if !st.in_flight && (cancelled || st.next_index >= st.entries.len()) {
        finish_run(app, st);
        return;
    }
    ctx.request_repaint_after(Duration::from_millis(150));
}

// ---------------------------------------------------------------- the run

use crate::scope::{self, Scope};

/// Start a run over `entries` (§9 Running): the service's endpoint and token as they are now, the
/// settings as they are now.
fn start_run(app: &mut LightcraftApp, ctx: &egui::Context, st: &mut State, entries: Vec<(PhotoId, String)>, skipped: usize) {
    let Some((endpoint, token)) = server::connection() else {
        // the button is gated on the service being ready; only a service lost between frames
        // lands here
        app.toast(ctx, i18n::tr("The model service is not running. Start it and try again."));
        return;
    };
    let max_keywords = st.settings.max_keywords();
    let config = Arc::new(RunConfig {
        endpoint,
        token,
        model: st.settings.model_repo().to_string(),
        prompt: prompt::prompt_text(max_keywords),
        max_keywords,
        parent: keywords::clean(&st.settings.parent),
        stall_secs: st.settings.timeout(),
    });
    begin_run(app, ctx, st, entries, skipped, config);
}

/// The machinery every run shares: the worker thread, the channels, the cancel flag, the activity
/// row, the state reset. Tests inject the config directly (a dead endpoint, a mock).
fn begin_run(app: &mut LightcraftApp, ctx: &egui::Context, st: &mut State, entries: Vec<(PhotoId, String)>, skipped: usize, config: Arc<RunConfig>) {
    let cancel = Arc::new(AtomicBool::new(false));
    let done_feeding = Arc::new(AtomicBool::new(false));
    let (job_tx, job_rx) = channel();
    let (done_tx, done_rx) = channel();
    let worker_cancel = cancel.clone();
    let worker_feeding = done_feeding.clone();
    let handle = std::thread::Builder::new().name("auto-keywords".to_string()).spawn(move || worker(job_rx, done_tx, worker_feeding, worker_cancel));
    let Ok(_handle) = handle else {
        app.toast(ctx, i18n::tr("The worker thread could not be started."));
        return;
    };
    st.run_config = Some(config);
    st.run_direct = st.settings.apply_mode == ApplyMode::Direct;
    st.run_parent = keywords::clean(&st.settings.parent);
    st.entries = entries;
    st.next_index = 0;
    st.skipped = skipped;
    st.in_flight = false;
    st.current_name.clear();
    st.results.clear();
    st.failures.clear();
    st.durations.clear();
    st.started = Some(Instant::now());
    st.job_tx = Some(job_tx);
    st.done_rx = Some(done_rx);
    st.done_feeding = done_feeding;
    st.cancel = cancel;
    st.excluded.clear();
    st.removed.clear();
    st.open.clear();
    st.applied = None;
    st.apply_error = None;
    st.activity = Some(app.session.activity.start("autoKeywords", UNDO_LABEL, Cancel::Flag(st.cancel.clone())));
    st.phase = Phase::Running;
    // the settings as the run started are saved by the debounced write, which pump lands on the
    // next frame (§13: saved once when a generation starts)
    save_soon(st);
}

/// The worker (§5's lower half): one job at a time — render, encode, ask the model, clean — then
/// the next, until the queue runs dry, a stop is asked, or the page is gone.
fn worker(job_rx: Receiver<Job>, done_tx: Sender<WorkerDone>, done_feeding: Arc<AtomicBool>, cancel: Arc<AtomicBool>) {
    loop {
        if cancel.load(Ordering::Relaxed) {
            return;
        }
        match job_rx.recv_timeout(vlm::READ_POLL) {
            Ok(job) => {
                let outcome = process(job, &cancel);
                if done_tx.send(WorkerDone { outcome }).is_err() {
                    return; // the page is gone (the app is closing)
                }
            }
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                // nothing queued: done when every photo was fed. The queue is checked once more
                // under the same acquire that saw `done_feeding`, so a job sent just before the
                // flag was set is never dropped.
                if done_feeding.load(Ordering::Acquire) && job_rx.try_recv().is_err() {
                    return;
                }
            }
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => return,
        }
    }
}

/// One job: render at the send size (the render's own panic guard catches decoder bugs),
/// JPEG-encode, ask the model, parse. Every failure is this photo's alone (§1).
fn process(job: Job, cancel: &AtomicBool) -> Outcome {
    let started = Instant::now();
    // copies up front, so the failure helper borrows nothing the job still owns (its render is
    // moved out below)
    let (id, name) = (job.id, job.name.clone());
    let failure = |reason: vlm::Reason| Outcome::Failed(PhotoFailure { id, name: name.clone(), reason, seconds: started.elapsed().as_secs_f64() });
    let rendered = job.render.run();
    let image = match rendered.rendered {
        Ok(r) => r.image,
        Err(e) => return failure(vlm::Reason::Other(e)),
    };
    if image.width == 0 || image.height == 0 || image.data.is_empty() {
        return failure(vlm::Reason::Other("the photo rendered to an empty image".to_string()));
    }
    let jpeg = match encode_jpeg(&EncodeImage::rgba8(&image), JPEG_QUALITY, ChromaSubsampling::S420, &EncodeMeta::default()) {
        Ok(jpeg) => jpeg,
        Err(e) => return failure(vlm::Reason::Other(format!("the JPEG encode failed: {e}"))),
    };
    let image_b64 = vlm::base64_encode(&jpeg);
    let config = job.config;
    match vlm::analyze(&config.endpoint, &config.token, &config.model, &config.prompt, &image_b64, config.stall_secs, Some(cancel)) {
        Err(reason) => failure(reason),
        Ok(text) => match prompt::parse(&text, config.max_keywords, &config.parent) {
            Ok(a) => Outcome::Done(PhotoResult { id, name, category: a.category, keywords: a.keywords, seconds: started.elapsed().as_secs_f64() }),
            Err(prompt::ParseError::NoJson) => failure(vlm::Reason::BadReply("the model did not answer with the JSON object asked for".to_string())),
            Err(prompt::ParseError::NoKeywords) => failure(vlm::Reason::NoKeywords),
        },
    }
}

/// Stop the run: no new photo is fed, the request in flight ends at its next read check (§9).
fn stop(st: &mut State) {
    st.cancel.store(true, Ordering::Relaxed);
    st.done_feeding.store(true, Ordering::Release);
}

/// The run ended: review, or write at once (direct mode), or the failure summary (§9).
fn finish_run(app: &mut LightcraftApp, st: &mut State) {
    st.job_tx = None;
    st.done_rx = None;
    let elapsed = st.started.take().unwrap_or_else(Instant::now).elapsed();
    if st.results.is_empty() {
        // nothing usable: the failure summary keeps the settings, a retry starts from it (§9)
        st.activity = None;
        st.run_config = None;
        st.phase = Phase::Failed;
        return;
    }
    if st.run_direct {
        st.phase = Phase::Applying;
        match apply(app, st) {
            Ok((photos, count)) => {
                st.applied = Some(Applied { photos, keywords: count, elapsed, undone: false });
                st.phase = Phase::Done;
            }
            Err(e) => {
                // the write failed: back to review, the error shown, the results kept
                st.phase = Phase::Review;
                st.apply_error = Some(e);
            }
        }
        // the activity row stays until the write is done (§8)
        st.activity = None;
        st.run_config = None;
    } else {
        st.activity = None;
        st.run_config = None;
        st.phase = Phase::Review;
    }
}

// ---------------------------------------------------------------- the write

/// Write the kept results as one `Op::Batch` of `SetMeta`s — one undo step (§8). Only photos with
/// at least one new keyword change; the batch is not committed when none do. Returns the photos
/// and keywords written.
fn apply(app: &mut LightcraftApp, st: &State) -> Result<(usize, usize), String> {
    let parent = st.run_parent.as_str();
    let mut ops: Vec<Op> = Vec::new();
    let mut added: Vec<String> = Vec::new();
    let mut photos = 0usize;
    let mut count = 0usize;
    for r in &st.results {
        if st.excluded.contains(&r.id.0) {
            continue;
        }
        let kept: Vec<&String> = st.kept(r).collect();
        if kept.is_empty() {
            continue;
        }
        let Some(p) = app.session.catalog.photo(r.id) else { continue };
        let mut kws = p.meta.keywords.clone();
        let mut new = 0usize;
        for kw in kept {
            let path = keywords::clean(&format!("{parent}|{}|{kw}", r.category));
            if path.is_empty() || kws.iter().any(|k| keywords::same(k, &path)) {
                continue;
            }
            kws.push(path.clone());
            added.push(path);
            new += 1;
        }
        if new == 0 {
            continue;
        }
        photos += 1;
        count += new;
        ops.push(Op::SetMeta { id: r.id, meta: Box::new(Meta { keywords: kws, ..p.meta.clone() }) });
    }
    if ops.is_empty() {
        return Ok((0, 0));
    }
    let op = Op::Batch { ops };
    let added = added.clone();
    app.session
        .execute_fn("extension.autoKeywords.apply", move |s: &mut Session| {
            s.commit(UNDO_LABEL, op)?;
            // the new keywords join the Recent Keywords list (§8)
            keywords_cmd::note_recent(s, &added);
            Ok(Value::Null)
        })
        .map_err(|e| e.to_string())?;
    Ok((photos, count))
}

/// Write what review keeps (§9 Review → Done).
fn write_review(app: &mut LightcraftApp, st: &mut State) {
    st.phase = Phase::Applying;
    let elapsed = st.started.take().unwrap_or_else(Instant::now).elapsed();
    match apply(app, st) {
        Ok((photos, count)) => {
            st.activity = None;
            st.run_config = None;
            st.applied = Some(Applied { photos, keywords: count, elapsed, undone: false });
            st.apply_error = None;
            st.phase = Phase::Done;
        }
        Err(e) => {
            st.activity = None;
            st.run_config = None;
            st.phase = Phase::Review;
            st.apply_error = Some(e);
        }
    }
}

/// Undo the batch, when it is still the newest step (§10: "Undo this batch" only reverts it
/// while nothing was edited after it).
fn undo_batch(app: &mut LightcraftApp, st: &mut State) {
    let is_last = app.session.undo.last().is_some_and(|e| e.label == UNDO_LABEL);
    if is_last
        && st.applied.as_ref().is_some_and(|a| !a.undone)
        && app.run("edit.undo", Value::Null).is_ok()
        && let Some(a) = &mut st.applied
    {
        a.undone = true;
    }
}

/// Retry exactly the photos that failed (§9 Done). A fresh run over their ids; photos gone from
/// the catalog meanwhile drop out.
fn retry_failed(app: &mut LightcraftApp, ctx: &egui::Context, st: &mut State) {
    let entries: Vec<(PhotoId, String)> =
        st.failures.iter().filter(|f| app.session.catalog.photo(f.id).is_some()).map(|f| (f.id, f.name.clone())).collect();
    if entries.is_empty() {
        return;
    }
    start_run(app, ctx, st, entries, 0);
}

/// Leave the outcome behind and show the settings again (Discard all / Generate more).
fn back_to_idle(st: &mut State) {
    st.results.clear();
    st.failures.clear();
    st.excluded.clear();
    st.removed.clear();
    st.open.clear();
    st.applied = None;
    st.apply_error = None;
    st.run_config = None;
    st.phase = Phase::Idle;
}

// ---------------------------------------------------------------- drawing

fn draw(app: &mut LightcraftApp, ui: &mut Ui, st: &mut State) {
    let t = Tokens::get(ui.ctx());
    egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
        ui.add_space(4.0);
        ui.label(RichText::new(i18n::tr("Auto Keywords")).font(t.semibold(14.0)).color(t.text));
        ui.add_space(2.0);
        ui.label(
            RichText::new(i18n::tr("Generate keywords with a local vision model. Images are sent only to the local model service on this computer."))
                .font(t.font(11.0))
                .color(t.text_dim),
        );
        ui.add_space(10.0);
        match st.phase {
            Phase::Running => draw_running(ui, st),
            Phase::Review => draw_review(app, ui, st),
            Phase::Applying => {
                ui.horizontal(|ui| {
                    spinner(ui, 14.0);
                    ui.label(RichText::new(i18n::tr("Writing the keywords…")).font(t.font(12.0)).color(t.text_label));
                });
            }
            Phase::Done => draw_done(app, ui, st),
            Phase::Failed => draw_failed_all(app, ui, st),
            Phase::Idle | Phase::Ready => draw_settings(app, ui, st),
        }
    });
}

/// The model service card (§10 idle state): the status line with the start/stop toggle (懒启动，
/// 快释放), the interpreter and model fields, the timeout — and, above it, the failure card when
/// the service did not come up.
fn draw_service(app: &mut LightcraftApp, ui: &mut Ui, st: &mut State) {
    let status = st.service.clone();
    if status.phase == server::Phase::Failed {
        draw_service_failed(app, ui, st);
    }
    card(ui, |ui| {
        section(ui, "Model service");
        ui.horizontal(|ui| {
            let t = Tokens::get(ui.ctx());
            match status.phase {
                server::Phase::Starting => {
                    spinner(ui, 12.0);
                    ui.label(RichText::new(i18n::tr("Starting the model service…")).font(t.font(11.5)).color(t.text_label));
                }
                server::Phase::Ready => {
                    status_dot(ui, t.accent);
                    let mut text = i18n::tr("Model ready").to_string();
                    if !status.device.is_empty() {
                        text.push_str(" · ");
                        text.push_str(&status.device);
                    }
                    if status.ready_secs > 0 {
                        text.push_str(" · ");
                        text.push_str(&fmt_seconds(status.ready_secs as f64));
                    }
                    ui.label(RichText::new(text).font(t.font(11.5)).color(t.text_label));
                    ui.with_layout(egui::Layout::right_to_left(Align::Center), |ui| {
                        // stopping the model mid-run would fail every remaining photo
                        let stop_ok = st.phase != Phase::Running;
                        if Button::new(i18n::tr("Stop the model")).secondary().small().enabled(stop_ok).widget("akw:service:stop").show(ui).clicked()
                        {
                            server::stop();
                        }
                    });
                }
                server::Phase::Stopped | server::Phase::Failed => {
                    status_dot(ui, if status.phase == server::Phase::Failed { t.reject } else { t.text_dim });
                    ui.label(RichText::new(i18n::tr("The model is not running")).font(t.font(11.5)).color(t.text_label));
                    ui.with_layout(egui::Layout::right_to_left(Align::Center), |ui| {
                        if Button::new(i18n::tr("Start the model")).primary().small().widget("akw:service:start").show(ui).clicked() {
                            start_service(app, ui.ctx(), st);
                        }
                    });
                }
            }
        });
        match &status.cause {
            server::Cause::IdleTimeout => hint(
                ui,
                &trn(
                    "It shut itself down after {secs} idle seconds to free the machine; press Start to bring it back.",
                    &[("secs", config::IDLE_SECS.to_string())],
                ),
            ),
            server::Cause::ManualStop => hint(ui, i18n::tr("Stopped by hand; press Start to bring it back.")),
            server::Cause::Exited => warning(ui, i18n::tr("The service process is gone.")),
            // a start failure's reason shows in the failure card above
            server::Cause::Error(_) | server::Cause::None => {}
        }
        field(ui, "Model", |ui| {
            let resp = TextField::singleline("akw:model", &mut st.settings.model_repo).hint(config::MODEL_DEFAULT).show(ui);
            if resp.response.changed() || resp.committed() {
                save_soon(st);
            }
        });
        field(ui, "Python interpreter", |ui| {
            let resp = TextField::singleline("akw:python", &mut st.settings.python_path).hint("python / python3 / py").show(ui);
            if resp.response.changed() || resp.committed() {
                save_soon(st);
            }
        });
        field(ui, "Per-photo timeout (seconds)", |ui| {
            if ui.add(egui::DragValue::new(&mut st.settings.timeout_secs).range(config::TIMEOUT_MIN..=config::TIMEOUT_MAX).suffix(" s")).changed() {
                save_soon(st);
            }
        });
        ui.add_space(2.0);
        hint(
            ui,
            &trn(
                "Lazy start: the model loads only after Start, and frees itself after {mins} idle minutes. The first start downloads it (about 5 GB).",
                &[("mins", (config::IDLE_SECS / 60).to_string())],
            ),
        );
        hint(ui, i18n::tr("Images are sent only to the local service on this computer (127.0.0.1)."));
    });
}

/// Start the model service (the toggle's on half): the settings as they are now govern the spawn.
fn start_service(app: &mut LightcraftApp, ctx: &egui::Context, st: &mut State) {
    save_settings(app, ctx, st);
    if let Err(e) = server::start(&st.settings) {
        app.toast(ctx, trn("The model service could not start: {e}", &[("e", e.clone())]));
    }
}

/// The service failure card (§10 fail state): the reason, the service's last stderr lines, how to
/// install the requirements — above the settings, so the fix and the retry are one glance apart.
fn draw_service_failed(app: &mut LightcraftApp, ui: &mut Ui, st: &mut State) {
    let cause = st.service.cause.clone();
    let tail = server::stderr_tail();
    card(ui, |ui| {
        let t = Tokens::get(ui.ctx());
        ui.horizontal(|ui| {
            status_dot(ui, t.reject);
            ui.label(RichText::new(i18n::tr("Could not start the model service")).font(t.semibold(12.0)).color(t.text));
        });
        ui.add_space(4.0);
        let reason = match &cause {
            server::Cause::Error(e) => e.clone(),
            _ => String::new(),
        };
        if !reason.is_empty() {
            ui.label(RichText::new(reason).font(t.font(11.0)).color(t.text_label));
        }
        for line in &tail {
            ui.label(RichText::new(line).font(t.font(10.0)).color(t.text_dim));
        }
        ui.add_space(8.0);
        if Button::new(i18n::tr("Try again")).secondary().small().widget("akw:service:retry").show(ui).clicked() {
            start_service(app, ui.ctx(), st);
        }
        ui.add_space(8.0);
        hint(ui, i18n::tr("Missing Python or the dependencies? Run: pip install -r extension/server/requirements.txt"));
        ui.add_space(6.0);
        warning(ui, i18n::tr("Images are sent only to the local service on this computer (127.0.0.1)."));
    });
}

/// The settings page (§10 idle state): model service, scope, keywords, write mode, the button.
fn draw_settings(app: &mut LightcraftApp, ui: &mut Ui, st: &mut State) {
    draw_service(app, ui, st);
    let (selected, view, library) = counts(app, st);
    card(ui, |ui| {
        section(ui, "Scope");
        if let Some(scope) = scope::selector(ui, st.scope, (selected, view, library)) {
            st.scope = scope;
        }
    });
    card(ui, |ui| {
        section(ui, "Keywords");
        field(ui, "Keywords per photo", |ui| {
            if ui.add(egui::DragValue::new(&mut st.settings.max_keywords).range(config::MAX_KEYWORDS_MIN..=config::MAX_KEYWORDS_MAX)).changed() {
                save_soon(st);
            }
        });
        field(ui, "Parent keyword", |ui| {
            let resp = TextField::singleline("akw:parent", &mut st.settings.parent).hint(config::PARENT_DEFAULT).width(120.0).show(ui);
            if resp.response.changed() || resp.committed() {
                save_soon(st);
            }
        });
        if ui.checkbox(&mut st.settings.skip_tagged, i18n::tr("Skip photos that already have keywords")).changed() {
            save_soon(st);
        }
        ui.add_space(6.0);
        let parent = if st.settings.parent.trim().is_empty() { i18n::tr("Category").to_string() } else { st.settings.parent.trim().to_string() };
        hint(ui, &trn("Saved as Parent|Category|Keyword — for example {path}.", &[("path", format!("{parent}|风景|雪山"))]));
        hint(ui, &format!("{}: {}", i18n::tr("The categories are fixed"), prompt::CATEGORIES.join(" / ")));
    });
    card(ui, |ui| {
        section(ui, "Write mode");
        let modes = [
            (
                "review",
                ApplyMode::Review,
                i18n::tr("Review before applying"),
                i18n::tr("Check each photo's keywords, remove what you don't need, then write the whole batch in one undoable step."),
            ),
            (
                "direct",
                ApplyMode::Direct,
                i18n::tr("Apply immediately"),
                i18n::tr("Each photo's keywords are written when the run ends; the whole batch stays one undo step."),
            ),
        ];
        for (name, mode, label, description) in modes {
            let response = ui.radio(st.settings.apply_mode == mode, label);
            widgets::register(ui, format!("akw:mode:{name}"), response.rect);
            if response.clicked() {
                st.settings.apply_mode = mode;
                save_soon(st);
            }
            hint(ui, description);
        }
    });
    // the one main action (§10): needs a ready model service and something in scope
    let in_scope = scope::count_of(st.scope, (selected, view, library));
    let label = trn("Generate keywords ({n} photos)", &[("n", in_scope.to_string())]);
    let enabled = st.phase == Phase::Ready && in_scope > 0;
    if Button::new(&label).primary().enabled(enabled).widget("akw:generate").show(ui).clicked() {
        let (entries, skipped) = scope::enumerate(app, st.scope, st.settings.skip_tagged);
        if entries.is_empty() {
            app.toast(ui.ctx(), i18n::tr("No photos to tag in this scope."));
        } else {
            start_run(app, ui.ctx(), st, entries, skipped);
        }
    }
    if st.phase == Phase::Ready {
        hint(ui, i18n::tr("Generation uses your computer's GPU; you can keep browsing and editing while it runs."));
    } else {
        hint(ui, i18n::tr("Press Start the model first; generation begins once it is ready."));
    }
}

/// The photo counts of the scope buttons; the library count is cached per catalog revision (a
/// big library would be walked every frame otherwise).
fn counts(app: &mut LightcraftApp, st: &mut State) -> (usize, usize, usize) {
    let selected = app.session.selected().len();
    let view = app.session.visible().len();
    let revision = app.session.catalog.revision;
    let library = match st.lib_count {
        Some((rev, n)) if rev == revision => n,
        _ => {
            let n = app.session.catalog.photos().filter(|p| p.in_library()).count();
            st.lib_count = Some((revision, n));
            n
        }
    };
    (selected, view, library)
}

/// The running page (§10 running state): progress, the photo being asked about, the stop button,
/// the two most recent results.
fn draw_running(ui: &mut Ui, st: &mut State) {
    let done = st.processed();
    let total = st.total();
    card(ui, |ui| {
        let t = Tokens::get(ui.ctx());
        ui.horizontal(|ui| {
            ui.label(RichText::new(done.to_string()).font(t.semibold(15.0)).color(t.text));
            ui.label(RichText::new(trn("/ {total} photos", &[("total", total.to_string())])).font(t.font(11.0)).color(t.text_dim));
            ui.with_layout(egui::Layout::right_to_left(Align::Center), |ui| {
                let pct = done.saturating_mul(100).checked_div(total).unwrap_or_default();
                ui.label(RichText::new(format!("{pct} %")).font(t.font(11.0)).color(t.text_label));
            });
        });
        Progress::new(fraction_of(done as u64, total as u64)).widget("akw:progress").show(ui);
        if !st.current_name.is_empty() {
            ui.label(RichText::new(trn("Analyzing · {name}", &[("name", st.current_name.clone())])).font(t.font(11.0)).color(t.text_label));
        }
        ui.add_space(4.0);
        ui.horizontal_wrapped(|ui| {
            let meta = |ui: &mut Ui, text: String, color: egui::Color32| {
                ui.label(RichText::new(text).font(t.font(10.5)).color(color));
            };
            meta(ui, trn("Generated {n}", &[("n", st.results.len().to_string())]), t.text_dim);
            if !st.failures.is_empty() {
                meta(ui, trn("Failed: {n}", &[("n", st.failures.len().to_string())]), t.reject);
            }
            if st.skipped > 0 {
                meta(ui, trn("Skipped {n}", &[("n", st.skipped.to_string())]), t.text_dim);
            }
            if let Some((m, s)) = st.eta() {
                meta(ui, trn("About {m} min {s} s left", &[("m", m.to_string()), ("s", s.to_string())]), t.text_dim);
            }
        });
        ui.add_space(6.0);
        let kept = st.results.len();
        let text = if st.run_direct {
            trn("Stop and write {n} photos", &[("n", kept.to_string())])
        } else {
            trn("Stop and review {n} photos", &[("n", kept.to_string())])
        };
        if Button::new(&text).secondary().widget("akw:stop").show(ui).clicked() {
            stop(st);
        }
    });
    if !st.results.is_empty() {
        card(ui, |ui| {
            section(ui, "Recent results");
            let recent: Vec<PhotoResult> = st.results.iter().rev().take(2).cloned().collect();
            for r in &recent {
                photo_row(ui, st, r, false);
            }
        });
    }
    hint(ui, i18n::tr("Progress also shows in the activity stack; closing this panel does not stop the task."));
}

/// One photo's row: name, keyword count and seconds, a chevron opening the keyword chips. In the
/// review the row is selectable and its chips removable (§10). `r` is the caller's copy: the rows
/// draw from a snapshot while the clicks edit `st`.
fn photo_row(ui: &mut Ui, st: &mut State, r: &PhotoResult, editable: bool) {
    let t = Tokens::get(ui.ctx());
    let id = r.id.0;
    ui.add_space(2.0);
    ui.horizontal(|ui| {
        if editable {
            let mut on = !st.excluded.contains(&id);
            if ui.checkbox(&mut on, "").changed() {
                if on {
                    st.excluded.remove(&id);
                } else {
                    st.excluded.insert(id);
                }
            }
        }
        ui.vertical(|ui| {
            ui.label(RichText::new(&r.name).font(t.font(11.5)).color(t.text));
            ui.label(
                RichText::new(trn("{k} keywords · {s}", &[("k", st.kept(r).count().to_string()), ("s", fmt_seconds(r.seconds))]))
                    .font(t.font(10.0))
                    .color(t.text_dim),
            );
        });
        ui.with_layout(egui::Layout::right_to_left(Align::Center), |ui| {
            let open = st.open.contains(&id);
            let chevron = if open { Icon::ChevronDown } else { Icon::ChevronRight };
            if Button::icon_only(chevron, "Show the keywords").ghost().small().widget(format!("akw:open:{id}")).show(ui).clicked() {
                if open {
                    st.open.remove(&id);
                } else {
                    st.open.insert(id);
                }
            }
        });
    });
    if st.open.contains(&id) {
        egui::Frame::NONE.inner_margin(egui::Margin { left: 26, right: 0, top: 0, bottom: 0 }).show(ui, |ui| {
            chips(ui, st, id, &r.category, &r.keywords, editable);
        });
    }
    ui.add_space(4.0);
}

/// The category chip and the keyword chips (§10: × removes, another click restores). Not
/// editable while the run is still going.
fn chips(ui: &mut Ui, st: &mut State, id: u64, category: &str, list: &[String], editable: bool) {
    let t = Tokens::get(ui.ctx());
    ui.add_space(2.0);
    ui.horizontal_wrapped(|ui| {
        ui.label(RichText::new(category).font(t.font(10.5)).color(t.accent).background_color(t.accent.gamma_multiply(0.16)));
        ui.label(RichText::new("·").font(t.font(10.5)).color(t.text_dim));
        for kw in list {
            let off = st.removed.get(&id).is_some_and(|off| off.contains(&kw.to_lowercase()));
            let mut text = RichText::new(format!("{kw}  ×")).font(t.font(10.5));
            text = if off { text.strikethrough().weak() } else { text };
            let response = ui.add_enabled(editable, egui::Button::new(text).small());
            widgets::register(ui, format!("akw:chip:{id}:{kw}"), response.rect);
            if response.clicked() {
                let entry = st.removed.entry(id).or_default();
                if off {
                    entry.remove(&kw.to_lowercase());
                } else {
                    entry.insert(kw.to_lowercase());
                }
            }
        }
    });
}

/// The review page (§10 review state): the summary with accept/discard, the rows, the write
/// button.
fn draw_review(app: &mut LightcraftApp, ui: &mut Ui, st: &mut State) {
    let apply_error = st.apply_error.clone();
    if let Some(e) = apply_error {
        warning(ui, &trn("The write failed: {e}", &[("e", e)]));
        ui.add_space(4.0);
    }
    let (photos, count) = st.review_counts();
    card(ui, |ui| {
        let t = Tokens::get(ui.ctx());
        ui.horizontal(|ui| {
            ui.label(
                RichText::new(trn("Ready to write: {p} photos · {k} keywords", &[("p", photos.to_string()), ("k", count.to_string())]))
                    .font(t.font(11.5))
                    .color(t.text_label),
            );
            ui.with_layout(egui::Layout::right_to_left(Align::Center), |ui| {
                if Button::new(i18n::tr("Discard all")).secondary().small().widget("akw:discard").show(ui).clicked() {
                    back_to_idle(st);
                }
                if Button::new(i18n::tr("Accept all")).secondary().small().widget("akw:accept").show(ui).clicked() {
                    st.excluded.clear();
                    st.removed.clear();
                }
            });
        });
    });
    // a snapshot: the rows draw from it while the clicks edit `st`
    let results: Vec<PhotoResult> = st.results.clone();
    for r in &results {
        photo_row(ui, st, r, true);
    }
    if photos > 0 {
        ui.add_space(4.0);
        let text = trn("Write to {n} photos", &[("n", photos.to_string())]);
        if Button::new(&text).primary().widget("akw:write").show(ui).clicked() {
            write_review(app, st);
        }
    }
    ui.add_space(2.0);
    hint(ui, i18n::tr("After the write the keywords are in the Keywords panel and, with auto-write XMP, in the photos' sidecars."));
}

/// The done page (§10 done state): what was written, the undo, the failures with their retry.
fn draw_done(app: &mut LightcraftApp, ui: &mut Ui, st: &mut State) {
    let t = Tokens::get(ui.ctx());
    let Some(a) = st.applied.as_ref().map(|a| (a.photos, a.keywords, a.elapsed, a.undone)) else { return };
    let (photos, keywords, elapsed, undone) = a;
    card(ui, |ui| {
        ui.horizontal(|ui| {
            ui.label(RichText::new("✓").font(t.semibold(13.0)).color(t.accent));
            ui.label(RichText::new(trn("Wrote {n} photos", &[("n", photos.to_string())])).font(t.semibold(12.5)).color(t.text));
        });
        ui.add_space(4.0);
        hint(
            ui,
            &trn(
                "{k} keywords · {t} · {f} failed",
                &[("k", keywords.to_string()), ("t", fmt_seconds(elapsed.as_secs_f64())), ("f", st.failures.len().to_string())],
            ),
        );
        if photos == 0 {
            hint(ui, i18n::tr("Everything the model found was already on the photos."));
        }
        if undone {
            hint(ui, i18n::tr("The batch was undone."));
        } else {
            hint(ui, i18n::tr("All changes are one undo step: Edit ▸ Undo reverts the whole batch."));
        }
        ui.add_space(8.0);
        ui.horizontal(|ui| {
            let undo_ok = !undone && app.session.undo.last().is_some_and(|e| e.label == UNDO_LABEL);
            if Button::new(i18n::tr("Undo this batch")).secondary().small().enabled(undo_ok).widget("akw:undo").show(ui).clicked() {
                undo_batch(app, st);
            }
            if Button::new(i18n::tr("Generate more")).secondary().small().widget("akw:more").show(ui).clicked() {
                back_to_idle(st);
            }
        });
    });
    draw_failures(app, ui, st);
}

/// The failure summary (§9 Failed): nothing usable came out of the run, the settings are kept.
fn draw_failed_all(app: &mut LightcraftApp, ui: &mut Ui, st: &mut State) {
    card(ui, |ui| {
        let t = Tokens::get(ui.ctx());
        ui.label(RichText::new(i18n::tr("The model produced no usable keywords.")).font(t.semibold(12.5)).color(t.text));
        ui.add_space(4.0);
        hint(
            ui,
            &trn(
                "{n} photos failed. The settings are kept; check the model and the service, then try again.",
                &[("n", st.failures.len().to_string())],
            ),
        );
        ui.add_space(8.0);
        if Button::new(i18n::tr("Generate more")).secondary().small().widget("akw:more").show(ui).clicked() {
            back_to_idle(st);
        }
    });
    draw_failures(app, ui, st);
}

/// The failed photos card, shared by Done and the failure summary (§10 done state).
fn draw_failures(app: &mut LightcraftApp, ui: &mut Ui, st: &mut State) {
    if st.failures.is_empty() {
        return;
    }
    card(ui, |ui| {
        section(ui, "Failed photos");
        for f in &st.failures {
            let t = Tokens::get(ui.ctx());
            ui.horizontal(|ui| {
                ui.label(RichText::new(&f.name).font(t.font(11.5)).color(t.text));
                ui.with_layout(egui::Layout::right_to_left(Align::Center), |ui| {
                    ui.label(RichText::new(f.reason.text()).font(t.font(10.5)).color(t.reject));
                });
            });
            ui.add_space(2.0);
        }
        if Button::new(i18n::tr("Retry failed")).secondary().small().widget("akw:retry").show(ui).clicked() {
            retry_failed(app, ui.ctx(), st);
        }
    });
}

#[cfg(test)]
mod tests;
