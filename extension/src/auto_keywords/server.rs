//! The self-hosted model service's manager (§6 of the design): the lazy start, the stop, the
//! watch for the service's own idle shutdown.
//!
//! 「启动模型」是唯一拉起服务的地方（懒启动）：[`start`] writes the embedded Python script to
//! the config dir, finds an interpreter and spawns it; the child binds a random loopback port and
//! announces it with `LISTENING <port>` on stdout, then loads the model and says `READY <device>`.
//! The service shuts itself down after [`config::IDLE_SECS`] seconds without a call (快释放) —
//! [`poll`] watches for that and for any other exit, so the page's status always reflects what
//! the process is doing. [`stop`] kills it on request. An orphaned service (the app died) is
//! still bounded by the service-side idle watchdog.
//!
//! State lives in a static mutex like the page's (recovered with `into_inner`, never a panic);
//! the stdout/stderr reader threads only ever take that lock, and the page takes its own lock
//! before this one, so there is no cycle.

use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::MutexGuard;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock, PoisonError};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use super::config::{self, Settings};
use super::vlm;

/// The Python script, embedded at build time and written to the config dir when it differs from
/// what is there (§6.2): an installed app has no `extension/` sources to run it from.
const SCRIPT: &str = include_str!("../../server/server.py");

/// The Windows spawn flag that keeps a console window from flashing with the service.
#[cfg(windows)]
const CREATE_NO_WINDOW: u32 = 0x0800_0000;

#[cfg(windows)]
use std::os::windows::process::CommandExt as _;

/// The interpreters tried, in order, when the settings name none (`py` is the Windows launcher).
const PYTHON_CANDIDATES: [&str; 3] = ["python", "python3", "py"];

/// The most stderr lines kept for a failure's display.
const STDERR_TAIL_LINES: usize = 8;

/// Where the service is in its life.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Phase {
    #[default]
    /// Nothing is running (also the state after a stop or an idle shutdown).
    Stopped,
    /// The process is up; the model is downloading or loading.
    Starting,
    /// The model is loaded and answers.
    Ready,
    /// It did not come up: [`Status::cause`] says why.
    Failed,
}

/// Why the service is where it is. The UI translates the fixed ones; the error's message shows
/// as-is (it is the service's or the interpreter's own words).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum Cause {
    #[default]
    None,
    /// Stopped by the toggle.
    ManualStop,
    /// The service's idle watchdog ended it ([`config::IDLE_SECS`] without a call).
    IdleTimeout,
    /// The process is gone without having announced why (a crash, a kill from outside).
    Exited,
    /// A start failure: no interpreter, a spawn error, or the model failed to load.
    Error(String),
}

/// A snapshot for the UI: cheap to take every frame.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Status {
    pub phase: Phase,
    /// Where the model runs (`cuda`/`cpu`/`mps`/`stub`), once the service said so.
    pub device: String,
    pub cause: Cause,
    /// Seconds since the model became ready, for the status line.
    pub ready_secs: u64,
}

/// The manager's whole state. `child` is taken only by [`stop`] and [`poll`]; the reader threads
/// touch the rest under the lock.
#[derive(Default)]
struct Managed {
    phase: Phase,
    device: String,
    cause: Cause,
    ready_since: Option<Instant>,
    /// The interpreter that worked last time, so the probe runs once per app session.
    interpreter: Option<String>,
    endpoint: Option<vlm::Endpoint>,
    token: String,
    child: Option<Child>,
    stderr_tail: Vec<String>,
    /// Set by the stdout reader when the service announced its idle shutdown.
    idle_shutdown: bool,
}

static MANAGED: OnceLock<Mutex<Managed>> = OnceLock::new();

fn lock() -> MutexGuard<'static, Managed> {
    MANAGED.get_or_init(|| Mutex::new(Managed::default())).lock().unwrap_or_else(PoisonError::into_inner)
}

/// The service's status now; also reaps the process when it exited by itself (the idle watchdog,
/// a crash) — cheap enough for every frame.
pub fn poll() -> Status {
    let mut m = lock();
    if m.child.is_some()
        && let Some(exit) = m.child.as_mut().and_then(|child| child.try_wait().ok().flatten())
    {
        reap(&mut m, exit.success(), exit.code());
    }
    Status {
        phase: m.phase,
        device: m.device.clone(),
        cause: m.cause.clone(),
        ready_secs: if m.phase == Phase::Ready { m.ready_since.as_ref().map_or(0, |at| at.elapsed().as_secs()) } else { 0 },
    }
}

/// Start the service (§6.2): install the script, find an interpreter, spawn, and hand the child's
/// stdout/stderr to reader threads. `Ok(())` means the process is up and loading; the model may
/// still take a while (a first run downloads it). A failure lands in [`Status::cause::Error`]
/// too, so the page shows it even past the toast.
pub fn start(settings: &Settings) -> Result<(), String> {
    start_with(settings, false)
}

/// The spawn itself. `stub` runs the service's canned-reply test mode (no model, no torch
/// needed) — the tests use it; the page always runs the real service.
fn start_with(settings: &Settings, stub: bool) -> Result<(), String> {
    let mut m = lock();
    if m.phase == Phase::Starting || m.phase == Phase::Ready {
        return Ok(());
    }
    let started = start_inner(&mut m, settings, stub);
    if let Err(e) = &started {
        m.phase = Phase::Failed;
        m.device.clear();
        m.ready_since = None;
        m.cause = Cause::Error(e.clone());
    }
    started
}

fn start_inner(m: &mut Managed, settings: &Settings, stub: bool) -> Result<(), String> {
    let script = install_script()?;
    let interpreter = match m.interpreter.clone() {
        Some(found) => found,
        None => find_interpreter(&settings.python_path)?,
    };
    let token = new_token();
    let mut command = Command::new(&interpreter);
    command
        .arg("-u")
        .arg(&script)
        .arg("--token")
        .arg(&token)
        .arg("--model")
        .arg(settings.model_repo())
        .arg("--idle-timeout")
        .arg(config::IDLE_SECS.to_string())
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    if stub {
        command.arg("--stub");
    }
    #[cfg(windows)]
    command.creation_flags(CREATE_NO_WINDOW);
    let mut child = command.spawn().map_err(|e| format!("{}: {e}", interpreter))?;
    let stdout = child.stdout.take().ok_or("the service's stdout is not readable")?;
    let stderr = child.stderr.take().ok_or("the service's stderr is not readable")?;
    m.phase = Phase::Starting;
    m.device.clear();
    m.cause = Cause::None;
    m.ready_since = None;
    m.endpoint = None;
    m.token = token;
    m.stderr_tail.clear();
    m.idle_shutdown = false;
    m.interpreter = Some(interpreter);
    m.child = Some(child);
    read_stdout(stdout);
    read_stderr(stderr);
    Ok(())
}

/// Stop the service now (the toggle's other half): kill the process; the reader threads see the
/// pipes close and leave, [`poll`] reaps what is left.
pub fn stop() {
    let mut m = lock();
    if let Some(mut child) = m.child.take() {
        let _ = child.kill();
        let _ = child.wait();
    }
    if m.phase != Phase::Failed {
        m.phase = Phase::Stopped;
        m.endpoint = None;
        m.device.clear();
        m.ready_since = None;
        m.cause = Cause::ManualStop;
    }
}

/// The endpoint and token of a listening service, for a run's config.
pub fn connection() -> Option<(vlm::Endpoint, String)> {
    let m = lock();
    m.endpoint.clone().map(|endpoint| (endpoint, m.token.clone()))
}

/// The tail of the service's stderr, for a failure card (the model download's last words).
pub fn stderr_tail() -> Vec<String> {
    lock().stderr_tail.clone()
}

// ---------------------------------------------------------------- internals

/// Reap a child that exited by itself: the idle shutdown is the normal, user-visible end; any
/// other exit before `READY` is a failure, and an unannounced death after ready is reported too.
/// A `LOAD_ERROR` already named the reason — the exit code must not overwrite it.
fn reap(m: &mut Managed, success: bool, code: Option<i32>) {
    m.child = None;
    let idle = m.idle_shutdown;
    let was_ready = m.phase == Phase::Ready;
    m.phase = Phase::Stopped;
    m.endpoint = None;
    m.device.clear();
    m.ready_since = None;
    if idle {
        m.cause = Cause::IdleTimeout;
    } else if !was_ready {
        m.phase = Phase::Failed;
        if matches!(m.cause, Cause::Error(_)) {
            // the service's own LOAD_ERROR line keeps the floor
        } else if success {
            m.cause = Cause::Error("the service exited before the model was ready".to_string());
        } else {
            m.cause = Cause::Error(format!("the service exited (code {})", code.map_or_else(|| "unknown".to_string(), |c| c.to_string())));
        }
    } else {
        m.cause = Cause::Exited;
    }
}

/// The stdout reader: protocol lines only, one [`apply_line`] each.
fn read_stdout(stdout: std::process::ChildStdout) {
    std::thread::Builder::new()
        .name("auto-keywords-service-stdout".to_string())
        .spawn(move || {
            for line in BufReader::new(stdout).lines() {
                let Ok(line) = line else { break };
                apply_line(&mut lock(), &line);
            }
        })
        .map_err(|e| log::warn!("auto keywords: could not start the stdout reader: {e}"))
        .ok();
}

/// What one protocol line does to the state (§6.2): `LISTENING` names the port, `READY` flips the
/// page to ready (with the device the model landed on), `LOAD_ERROR` fails the start,
/// `IDLE_SHUTDOWN` marks the service's own end. Anything else is ignored (the service keeps its
/// stdout to these lines, but a stray line must not break the page).
fn apply_line(m: &mut Managed, line: &str) {
    if let Some(rest) = line.trim().strip_prefix("LISTENING ") {
        match rest.trim().parse::<u16>().map(|port| vlm::parse_endpoint(&format!("http://127.0.0.1:{port}"))) {
            Ok(Ok(endpoint)) => m.endpoint = Some(endpoint),
            _ => log::warn!("auto keywords: the service's LISTENING line is unreadable: {line}"),
        }
    } else if let Some(device) = line.trim().strip_prefix("READY") {
        m.phase = Phase::Ready;
        m.cause = Cause::None;
        m.ready_since = Some(Instant::now());
        m.device = device.trim().to_string();
    } else if let Some(message) = line.trim().strip_prefix("LOAD_ERROR ") {
        m.phase = Phase::Failed;
        m.device.clear();
        m.ready_since = None;
        m.cause = Cause::Error(message.trim().to_string());
    } else if line.trim() == "IDLE_SHUTDOWN" {
        m.idle_shutdown = true;
    }
}

/// The stderr reader: keep the last lines for a failure card; progress and logs are otherwise
/// dropped (draining is what matters, so a chatty download cannot block the child).
fn read_stderr(stderr: std::process::ChildStderr) {
    std::thread::Builder::new()
        .name("auto-keywords-service-stderr".to_string())
        .spawn(move || {
            for line in BufReader::new(stderr).lines() {
                let Ok(line) = line else { break };
                if line.trim().is_empty() {
                    continue;
                }
                // progress bars redraw in place: keep only each line's last segment
                let line = line.rsplit('\r').next().unwrap_or(&line).trim_end();
                let line = if line.chars().count() > 200 { line.chars().take(200).collect::<String>() } else { line.to_string() };
                let mut m = lock();
                if m.stderr_tail.len() >= STDERR_TAIL_LINES {
                    m.stderr_tail.remove(0);
                }
                m.stderr_tail.push(line);
            }
        })
        .map_err(|e| log::warn!("auto keywords: could not start the stderr reader: {e}"))
        .ok();
}

/// Write the embedded script to `<config dir>/extension-server/server.py` when it differs from
/// what is on disk, and return its path. Without a config dir, the temp dir keeps the page
/// working (with a log line).
fn install_script() -> Result<PathBuf, String> {
    let dir = lightcraft_engine::config::config_dir().unwrap_or_else(|| {
        log::warn!("auto keywords: no config dir, the service script goes to the temp dir");
        std::env::temp_dir()
    });
    let dir = dir.join("extension-server");
    std::fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let path = dir.join("server.py");
    if std::fs::read_to_string(&path).is_ok_and(|existing| existing == SCRIPT) {
        return Ok(path);
    }
    std::fs::write(&path, SCRIPT).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(path)
}

/// Find an interpreter: the settings' path wins without probing (its failure shows in the start
/// error); otherwise the first candidate whose `--version` answers like a Python 3.
fn find_interpreter(override_path: &str) -> Result<String, String> {
    let override_path = override_path.trim();
    if !override_path.is_empty() {
        return Ok(override_path.to_string());
    }
    let mut tried: Vec<String> = Vec::new();
    for candidate in PYTHON_CANDIDATES {
        match Command::new(candidate).arg("--version").output() {
            Ok(output) => {
                let said = String::from_utf8_lossy(&output.stdout);
                if said.trim_start().starts_with("Python 3") {
                    return Ok(candidate.to_string());
                }
                tried.push(format!("{candidate}: {}", said.trim()));
            }
            Err(e) => tried.push(format!("{candidate}: {e}")),
        }
    }
    Err(format!(
        "no Python 3 interpreter found (tried {}); install Python and the service's requirements, or set the interpreter path on the page",
        tried.join("; ")
    ))
}

/// A fresh token for one service life: time, process id and a counter folded through FNV-1a. Not
/// cryptographic — it only keeps the machine's other local processes off the service.
fn new_token() -> String {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    let mut fold = |byte: u8| {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    };
    let nanos = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_nanos() as u64).unwrap_or(0);
    for value in [nanos, u64::from(std::process::id()), COUNTER.fetch_add(1, Ordering::Relaxed), nanos.rotate_left(17)] {
        for byte in value.to_le_bytes() {
            fold(byte);
        }
    }
    format!("{hash:016x}")
}

#[cfg(test)]
mod tests;
