//! Tests of the service manager (§6): the protocol-line handling, the reaping, the token — and,
//! when a Python 3 is on the PATH, the real service in `--stub` mode, end to end: spawn, listen,
//! analyze, and the idle shutdown that frees the machine.

use std::io::BufRead;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use lightcraft_testkit::skip;

use super::config::Settings;
use super::{Cause, Managed, Phase, apply_line, connection, drain_lines, find_interpreter, poll, reap, start_with, stop};

// ---------------------------------------------------------------- protocol lines

#[test]
fn protocol_lines_drive_the_state() {
    // the spawn put the manager in Starting before any line arrived (start_inner)
    let mut m = Managed { phase: Phase::Starting, ..Managed::default() };
    apply_line(&mut m, "LISTENING 8321");
    let ep = m.endpoint.clone().expect("the port was parsed");
    assert_eq!((ep.host.as_str(), ep.port), ("127.0.0.1", 8321));
    assert_eq!(m.phase, Phase::Starting, "listening alone is not ready");
    apply_line(&mut m, "READY cuda");
    assert_eq!(m.phase, Phase::Ready);
    assert_eq!(m.device, "cuda");
    assert!(m.ready_since.is_some());
    // a plain READY (no device named) also works
    let mut plain = Managed::default();
    apply_line(&mut plain, "READY");
    assert_eq!((plain.phase, plain.device.as_str()), (Phase::Ready, ""));
    // a load error fails the start with the service's own words
    let mut failed = Managed::default();
    apply_line(&mut failed, "LISTENING 8321");
    apply_line(&mut failed, "LOAD_ERROR torch is not installed");
    assert_eq!(failed.phase, Phase::Failed);
    assert_eq!(failed.cause, Cause::Error("torch is not installed".to_string()));
    // the idle shutdown is remembered for the reap
    apply_line(&mut m, "IDLE_SHUTDOWN");
    assert!(m.idle_shutdown);
    // a stray line changes nothing
    let (phase, cause, endpoint) = (m.phase, m.cause.clone(), m.endpoint.clone());
    apply_line(&mut m, "downloading weights 42 %");
    assert_eq!((m.phase, m.cause, m.endpoint), (phase, cause, endpoint));
}

#[test]
fn an_unreadable_listening_line_is_refused_not_trusted() {
    // the endpoint only ever comes from parsing our own loopback format through the checked
    // constructor; a broken line leaves no endpoint
    let mut m = Managed::default();
    apply_line(&mut m, "LISTENING not-a-port");
    assert_eq!(m.endpoint, None);
    apply_line(&mut m, "LISTENING");
    assert_eq!(m.endpoint, None);
}

#[test]
fn the_reap_tells_the_three_ends_apart() {
    // the service's own idle shutdown: stopped, with its cause
    let mut m = Managed::default();
    apply_line(&mut m, "READY cuda");
    apply_line(&mut m, "IDLE_SHUTDOWN");
    reap(&mut m, true, None);
    assert_eq!(m.phase, Phase::Stopped);
    assert_eq!(m.cause, Cause::IdleTimeout);
    assert_eq!(m.endpoint, None);
    // died before ready: a failure with the exit code
    let mut m = Managed::default();
    apply_line(&mut m, "LISTENING 8321");
    reap(&mut m, false, Some(1));
    assert_eq!(m.phase, Phase::Failed);
    assert!(matches!(m.cause, Cause::Error(message) if message.contains("code 1")));
    // a LOAD_ERROR already named the reason: the exit code does not overwrite it
    let mut m = Managed::default();
    apply_line(&mut m, "LISTENING 8321");
    apply_line(&mut m, "LOAD_ERROR No module named 'torch'");
    reap(&mut m, false, Some(1));
    assert_eq!(m.phase, Phase::Failed);
    assert_eq!(m.cause, Cause::Error("No module named 'torch'".to_string()));
    // a clean exit before ready is still a failure
    let mut m = Managed::default();
    reap(&mut m, true, Some(0));
    assert_eq!(m.phase, Phase::Failed);
    assert!(matches!(m.cause, Cause::Error(message) if message.contains("before the model was ready")));
    // died after ready without announcing why: reported, not failed
    let mut m = Managed::default();
    apply_line(&mut m, "READY cuda");
    reap(&mut m, false, Some(-1));
    assert_eq!(m.phase, Phase::Stopped);
    assert_eq!(m.cause, Cause::Exited);
}

#[test]
fn tokens_differ_between_starts() {
    let first = super::new_token();
    let second = super::new_token();
    assert_ne!(first, second);
    assert_eq!(first.len(), 16);
}

// ---------------------------------------------------------------- the real service

/// A Python 3 for the stub tests, or the skip note (this machine may have none).
fn interpreter_or_skip() -> Option<String> {
    match find_interpreter("") {
        Ok(found) => Some(found),
        Err(why) => {
            skip!(System, "{why}");
            None
        }
    }
}

/// The embedded script's source in this checkout.
fn script() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("server").join("server.py")
}

#[test]
fn the_real_service_answers_and_shuts_itself_down_when_idle() {
    let Some(interpreter) = interpreter_or_skip() else { return };
    let mut child = Command::new(&interpreter)
        .arg("-u")
        .arg(script())
        .arg("--token")
        .arg("test-token")
        .arg("--stub")
        .arg("--idle-timeout")
        .arg("2")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("the stub service spawns");
    let stdout = child.stdout.take().expect("stdout is piped");
    let mut endpoint = None;
    for line in std::io::BufReader::new(stdout).lines() {
        let line = line.expect("the service's stdout reads");
        if let Some(rest) = line.strip_prefix("LISTENING ") {
            endpoint = super::super::vlm::parse_endpoint(&format!("http://127.0.0.1:{}", rest.trim())).ok();
        }
        if line.starts_with("READY") {
            break;
        }
    }
    let endpoint = endpoint.expect("the service announced its port and its readiness");
    let text = super::super::vlm::analyze(&endpoint, "test-token", "m", "p", "aGk=", 5, None).expect("the stub answers");
    assert!(text.contains("雪山"), "the canned reply: {text}");
    // another local process (or a wrong token) is refused
    let wrong = super::super::vlm::analyze(&endpoint, "nope", "m", "p", "aGk=", 5, None).expect_err("a wrong token fails");
    assert!(matches!(wrong, super::super::vlm::Reason::Http { status: 401, .. }), "{wrong:?}");
    // the idle watchdog (2 s here, 5 min in the product) ends the service by itself
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        match child.try_wait().expect("the child's status reads") {
            Some(status) => {
                assert!(status.success(), "the idle shutdown is a clean exit");
                break;
            }
            None => {
                assert!(Instant::now() < deadline, "the service did not shut itself down when idle");
                std::thread::sleep(Duration::from_millis(100));
            }
        }
    }
}

#[test]
fn the_manager_starts_and_stops_the_real_service() {
    let settings = Settings::default();
    if find_interpreter("").is_err() {
        // interpreter_or_skip names the reason; this test exercises the manager, which starts
        // with the same probe
        skip!(System, "no Python interpreter on the PATH");
        return;
    }
    // the stub mode: the manager's full spawn → listen → ready path, without torch or weights
    start_with(&settings, true).expect("the service starts");
    let deadline = Instant::now() + Duration::from_secs(20);
    loop {
        let status = poll();
        if matches!(status.phase, Phase::Ready | Phase::Failed) {
            assert_eq!(status.phase, Phase::Ready, "the stub service comes up: {status:?}");
            break;
        }
        assert!(Instant::now() < deadline, "the service did not become ready: {status:?}");
        std::thread::sleep(Duration::from_millis(100));
    }
    assert!(connection().is_some(), "a ready service has an endpoint and a token");
    stop();
    let status = poll();
    assert_eq!(status.phase, Phase::Stopped);
    assert_eq!(status.cause, Cause::ManualStop);
    assert!(connection().is_none(), "a stopped service offers no endpoint");
}

#[test]
fn the_pipe_drain_survives_bytes_that_are_not_utf8() {
    // A zh-CN Python encoded the "…" of a log line as cp936 (A1 AD): the drain must keep
    // reading — abandoning the pipe once turned the child's next write into [Errno 22] and
    // left a service process hanging past its idle shutdown.
    let mut lines = Vec::new();
    drain_lines(&b"ok one\n\xa1\xad loading \nok two\n"[..], |line| lines.push(line.to_string()));
    assert_eq!(lines, vec!["ok one", "\u{FFFD}\u{FFFD} loading ", "ok two"]);
}
