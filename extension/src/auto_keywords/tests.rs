//! Tests of the Auto Keywords page (§14): the HTTP client against a local mock service, the
//! parsing and cleaning, the write with its undo, the settings file, and headless renders of the
//! page. The service manager's own tests live in [`super::server`].

use std::io::{Read, Write};
use std::net::TcpListener;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::mpsc::channel;
use std::thread::JoinHandle;
use std::time::Duration;

use egui::vec2;
use lightcraft_catalog::{Meta, Op, PhotoId};
use lightcraft_engine::Session;
use lightcraft_engine::activity::Cancel;
use lightcraft_ui_egui::{LightcraftApp, Services};
use serde_json::{Value, json};

use super::config::{self, ApplyMode};
use super::prompt::{self, ParseError};
use super::vlm;
use super::{Outcome, Phase, PhotoFailure, PhotoResult, RunConfig, State};

// ---------------------------------------------------------------- mock server

/// A local HTTP server answering each connection with what `respond` makes of the request. The
/// request is the raw head plus body; the response is raw bytes plus how long the connection is
/// held open after them (a stalled service, for the timeout and cancel tests).
struct Server {
    url: String,
    _handle: Option<JoinHandle<()>>,
}

impl Server {
    fn serve(respond: impl Fn(String) -> (Vec<u8>, u64) + Send + 'static) -> Server {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let handle = std::thread::Builder::new()
            .name("auto-keywords-mock".to_string())
            .spawn(move || {
                for stream in listener.incoming() {
                    let Ok(mut stream) = stream else { continue };
                    let request = read_request(&mut stream);
                    let (response, hold_ms) = respond(request);
                    let _ = stream.write_all(&response);
                    let _ = stream.flush();
                    if hold_ms > 0 {
                        std::thread::sleep(Duration::from_millis(hold_ms));
                    }
                    drop(stream);
                }
            })
            .unwrap();
        Server { url: format!("http://127.0.0.1:{}", addr.port()), _handle: Some(handle) }
    }
}

/// The mock's view of one request: head plus body, read by `Content-Length`.
fn read_request(stream: &mut std::net::TcpStream) -> String {
    let mut buf = Vec::new();
    let mut tmp = [0u8; 4096];
    let head_end = loop {
        if let Some(at) = find(&buf, b"\r\n\r\n") {
            break at + 4;
        }
        let n = match stream.read(&mut tmp) {
            Ok(0) | Err(_) => break buf.len(),
            Ok(n) => n,
        };
        buf.extend_from_slice(&tmp[..n]);
    };
    let head = String::from_utf8_lossy(&buf[..head_end]).into_owned();
    let content_length = head
        .lines()
        .find(|l| l.to_lowercase().starts_with("content-length:"))
        .and_then(|l| l.split(':').nth(1))
        .and_then(|v| v.trim().parse::<usize>().ok())
        .unwrap_or(0);
    while buf.len() < head_end + content_length {
        match stream.read(&mut tmp) {
            Ok(0) | Err(_) => break,
            Ok(n) => buf.extend_from_slice(&tmp[..n]),
        }
    }
    let mut out = head;
    out.push_str(&String::from_utf8_lossy(&buf[head_end..]));
    out
}

fn find(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.len() > haystack.len() {
        return None;
    }
    (0..=haystack.len() - needle.len()).find(|&at| &haystack[at..at + needle.len()] == needle)
}

/// A 200 answer with a JSON body sent by `Content-Length`.
fn http_json(body: &str) -> (Vec<u8>, u64) {
    (
        format!("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len())
            .into_bytes(),
        0,
    )
}

fn http_401() -> (Vec<u8>, u64) {
    (
        format!(
            "HTTP/1.1 401 Unauthorized\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{{\"error\":{{\"message\":\"missing or wrong bearer token\"}}}}",
            "{\"error\":{\"message\":\"missing or wrong bearer token\"}}".len()
        )
        .into_bytes(),
        0,
    )
}

/// A 200 answer sent chunked.
fn http_chunked(body: &str) -> (Vec<u8>, u64) {
    let mut out = b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n".to_vec();
    for chunk in body.as_bytes().chunks(7) {
        out.extend_from_slice(format!("{:x}\r\n", chunk.len()).as_bytes());
        out.extend_from_slice(chunk);
        out.extend_from_slice(b"\r\n");
    }
    out.extend_from_slice(b"0\r\n\r\n");
    (out, 0)
}

/// A 200 answer with no length information: everything until the connection closes.
fn http_close_delimited(body: &str) -> (Vec<u8>, u64) {
    (format!("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nConnection: close\r\n\r\n{body}").into_bytes(), 0)
}

/// A 200 answer that announces more body than it ever sends, holding the connection open:
/// the client must hit its stall limit, not read an early close.
fn http_stalled() -> (Vec<u8>, u64) {
    (b"HTTP/1.1 200 OK\r\nContent-Length: 1000\r\nConnection: close\r\n\r\nshort".to_vec(), 2000)
}

// ---------------------------------------------------------------- endpoint & HTTP

#[test]
fn the_endpoint_refuses_everything_but_local_http() {
    let parse = |url: &str| vlm::parse_endpoint(url);
    assert!(parse("https://127.0.0.1:8321").is_err(), "https is refused with its own explanation");
    assert!(parse("http://example.com:8321").is_err(), "a non-loopback host is refused");
    assert!(parse("ftp://127.0.0.1:8321").is_err(), "another scheme is refused");
    assert!(parse("http://user:pw@127.0.0.1:8321").is_err(), "credentials are refused");
    let ep = parse("http://127.0.0.1:8321/").unwrap();
    assert_eq!((ep.host.as_str(), ep.port, ep.prefix.as_str()), ("127.0.0.1", 8321, ""));
    // the manager always builds the address from the service's LISTENING line, so a port is
    // required, not defaulted
    assert!(parse("http://127.0.0.1").is_err(), "a missing port is refused");
    // localhost and a bracketed IPv6 loopback are accepted
    assert_eq!(parse("http://localhost:8321").unwrap().host, "localhost");
    assert_eq!(parse("http://[::1]:8321").unwrap().host, "::1");
    // a path prefix is kept in front of the API paths
    assert_eq!(parse("http://127.0.0.1:8321/sub").unwrap().prefix, "/sub");
}

#[test]
fn the_chat_request_carries_the_token_and_reads_the_content() {
    let server = Server::serve(|request| {
        assert!(request.starts_with("POST /v1/chat/completions HTTP/1.1"), "the service path: {request}");
        assert!(request.contains("Authorization: Bearer sekrit"), "the service's token rides along: {request}");
        assert!(request.contains("Content-Type: application/json"));
        assert!(request.contains("data:image/jpeg;base64,aGVsbG8="), "the image is a data URL: {request}");
        assert!(request.contains("\"max_tokens\":256"));
        assert!(request.contains("看这张照片"), "the prompt is sent: {request}");
        http_json(r#"{"choices":[{"message":{"content":"{\"category\":\"风景\",\"keywords\":[\"雪山\"]}"}}]}"#)
    });
    let ep = vlm::parse_endpoint(&server.url).unwrap();
    let text = vlm::analyze(&ep, "sekrit", "m", "看这张照片", "aGVsbG8=", 5, None).unwrap();
    assert!(text.contains("雪山"));
}

#[test]
fn a_wrong_token_is_an_http_failure_naming_the_service() {
    let server = Server::serve(|_| http_401());
    let ep = vlm::parse_endpoint(&server.url).unwrap();
    let err = vlm::analyze(&ep, "wrong", "m", "p", "x", 5, None).unwrap_err();
    match err {
        vlm::Reason::Http { status, body } => {
            assert_eq!(status, 401);
            assert!(body.contains("bearer token"), "the service's message is shown: {body}");
        }
        other => panic!("expected an HTTP error, got {other:?}"),
    }
}

#[test]
fn a_reply_arrives_whatever_way_the_body_is_framed() {
    let reply = r#"{"choices":[{"message":{"content":"ok"}}]}"#;
    for (name, make) in [("content-length", http_json(reply)), ("chunked", http_chunked(reply)), ("connection close", http_close_delimited(reply))] {
        let server = Server::serve(move |_| make.clone());
        let ep = vlm::parse_endpoint(&server.url).unwrap();
        let text = vlm::analyze(&ep, "t", "m", "p", "x", 5, None).unwrap_or_else(|e| panic!("{name}: {e:?}"));
        assert_eq!(text, "ok", "{name}");
    }
}

#[test]
fn a_large_response_is_refused() {
    // one byte more than the cap, announced by Content-Length: refused before it is read
    let too_big = 4 << 20;
    let server = Server::serve(move |_| (format!("HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", too_big + 1).into_bytes(), 0));
    let ep = vlm::parse_endpoint(&server.url).unwrap();
    let err = vlm::analyze(&ep, "t", "m", "p", "x", 5, None).unwrap_err();
    assert!(matches!(err, vlm::Reason::BadReply(_)), "{err:?}");
}

#[test]
fn an_http_error_status_names_the_service_answer() {
    let server = Server::serve(|_| (b"HTTP/1.1 404 Not Found\r\nContent-Length: 9\r\nConnection: close\r\n\r\nno model!".to_vec(), 0));
    let ep = vlm::parse_endpoint(&server.url).unwrap();
    let err = vlm::analyze(&ep, "t", "m", "p", "x", 5, None).unwrap_err();
    match err {
        vlm::Reason::Http { status, body } => {
            assert_eq!(status, 404);
            assert!(body.contains("no model"), "the body is shown (truncated): {body}");
        }
        other => panic!("expected an HTTP error, got {other:?}"),
    }
}

#[test]
fn an_unreachable_service_says_so() {
    // a port with no listener: bind, learn the port, drop the listener
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    drop(listener);
    let ep = vlm::parse_endpoint(&format!("http://127.0.0.1:{port}")).unwrap();
    let err = vlm::analyze(&ep, "t", "m", "p", "x", 1, None).unwrap_err();
    assert!(matches!(err, vlm::Reason::Unreachable(_)), "{err:?}");
}

#[test]
fn a_stalled_service_times_out() {
    let server = Server::serve(|_| http_stalled());
    let ep = vlm::parse_endpoint(&server.url).unwrap();
    let started = std::time::Instant::now();
    let err = vlm::analyze(&ep, "t", "m", "p", "x", 1, None).unwrap_err();
    assert!(matches!(err, vlm::Reason::Timeout { .. }), "{err:?}");
    assert!(started.elapsed() >= Duration::from_millis(900), "the stall limit held the request");
}

#[test]
fn the_cancel_flag_stops_a_request_during_the_reads() {
    // the service holds the connection, the flag is set up front: the first poll stops the read
    let server = Server::serve(|_| http_stalled());
    let ep = vlm::parse_endpoint(&server.url).unwrap();
    let cancel = Arc::new(AtomicBool::new(true));
    let err = vlm::analyze(&ep, "t", "m", "p", "x", 5, Some(&cancel)).unwrap_err();
    assert_eq!(err, vlm::Reason::Stopped, "a cancelled request is stopped, not failed");
}

#[test]
fn a_slow_reply_still_arrives_when_the_flag_is_clear() {
    let server = Server::serve(|_| {
        std::thread::sleep(Duration::from_millis(450));
        http_json(r#"{"choices":[{"message":{"content":"in time"}}]}"#)
    });
    let ep = vlm::parse_endpoint(&server.url).unwrap();
    let cancel = Arc::new(AtomicBool::new(false));
    let text = vlm::analyze(&ep, "t", "m", "p", "x", 5, Some(&cancel)).unwrap();
    assert_eq!(text, "in time");
}

// ---------------------------------------------------------------- base64

#[test]
fn base64_matches_the_rfc_4648_vectors() {
    assert_eq!(vlm::base64_encode(b""), "");
    assert_eq!(vlm::base64_encode(b"f"), "Zg==");
    assert_eq!(vlm::base64_encode(b"fo"), "Zm8=");
    assert_eq!(vlm::base64_encode(b"foo"), "Zm9v");
    assert_eq!(vlm::base64_encode(b"foob"), "Zm9vYg==");
    assert_eq!(vlm::base64_encode(b"fooba"), "Zm9vYmE=");
    assert_eq!(vlm::base64_encode(b"foobar"), "Zm9vYmFy");
}

// ---------------------------------------------------------------- parsing & cleaning

#[test]
fn the_reply_is_found_whatever_wraps_it() {
    let parse = |text: &str| prompt::parse(text, 8, "AI").unwrap();
    let direct = parse(r#"{"category":"风景","keywords":["雪山","湖泊"]}"#);
    assert_eq!(direct.category, "风景");
    assert_eq!(direct.keywords, vec!["雪山", "湖泊"]);
    let fenced = parse("```json\n{\"category\":\"风景\",\"keywords\":[\"雪山\"]}\n```");
    assert_eq!(fenced.keywords, vec!["雪山"]);
    let with_prose = parse("好的，结果如下：{\"category\":\"风景\",\"keywords\":[\"湖泊\"]} 请查收。");
    assert_eq!(with_prose.keywords, vec!["湖泊"]);
    // an array root: everything is a keyword, the category is 其他
    let array = parse(r#"["雪山","湖泊"]"#);
    assert_eq!(array.category, prompt::OTHER);
    assert_eq!(array.keywords, vec!["雪山", "湖泊"]);
    // tags and labels are accepted as the keyword field
    let tagged = parse(r#"{"category":"动物","tags":["猫","窗台"]}"#);
    assert_eq!(tagged.keywords, vec!["猫", "窗台"]);
    let labeled = parse(r#"{"category":"食物","labels":["面条"]}"#);
    assert_eq!(labeled.keywords, vec!["面条"]);
}

#[test]
fn categories_outside_the_list_map_or_fall_back() {
    let parse = |text: &str| prompt::parse(text, 8, "AI").unwrap().category;
    assert_eq!(parse(r#"{"category":"人像","keywords":["合影"]}"#), "人物");
    assert_eq!(parse(r#"{"category":"people","keywords":["合影"]}"#), "人物");
    assert_eq!(parse(r#"{"category":"风光","keywords":["湖"]}"#), "风景");
    assert_eq!(parse(r#"{"category":"landscape","keywords":["湖"]}"#), "风景");
    assert_eq!(parse(r#"{"category":"美食","keywords":["面"]}"#), "食物");
    assert_eq!(parse(r#"{"category":"物品","keywords":["杯子"]}"#), "物体");
    // the list members themselves stay
    assert_eq!(parse(r#"{"category":"建筑","keywords":["桥"]}"#), "建筑");
    // anything else lands on 其他
    assert_eq!(parse(r#"{"category":"抽象","keywords":["色块"]}"#), "其他");
    assert_eq!(parse(r#"{"keywords":["谜"]}"#), "其他");
}

#[test]
fn keywords_are_cleaned_of_hierarchy_and_length() {
    // a | must not plant levels: it becomes a space
    let parsed = prompt::parse(r#"{"category":"风景","keywords":["雪|山","湖泊"]}"#, 8, "AI").unwrap();
    assert_eq!(parsed.keywords, vec!["雪 山", "湖泊"]);
    // leading #, trimming, case-insensitive dedup
    let parsed = prompt::parse(r##"{"keywords":["#雪山","  雪山  ","SNow"]}"##, 8, "AI").unwrap();
    assert_eq!(parsed.keywords, vec!["雪山", "SNow"]);
    // over-long keywords are cut at 32 characters
    let long = "很长".repeat(20);
    let parsed = prompt::parse(&format!(r#"{{"keywords":["{long}","湖"]}}"#), 8, "AI").unwrap();
    assert_eq!(parsed.keywords[0].chars().count(), prompt::MAX_KEYWORD_CHARS);
    // the category's and the parent's own names are not keywords
    let parsed = prompt::parse(r##"{"category":"风景","keywords":["风景","AI","雪山"]}"##, 8, "AI").unwrap();
    assert_eq!(parsed.keywords, vec!["雪山"]);
    // at most N keywords are kept
    let many: Vec<String> = (0..20).map(|i| format!("词{i}")).collect();
    let parsed = prompt::parse(&format!(r#"{{"keywords":{}}}"#, json!(many)), 3, "AI").unwrap();
    assert_eq!(parsed.keywords.len(), 3);
}

#[test]
fn an_empty_or_unreadable_answer_is_one_photos_failure() {
    assert_eq!(prompt::parse("对不起，我不知道。", 8, "AI"), Err(ParseError::NoJson));
    assert_eq!(prompt::parse(r#"{"category":"风景","keywords":[]}"#, 8, "AI"), Err(ParseError::NoKeywords));
    assert_eq!(prompt::parse(r#"{"category":"风景"}"#, 8, "AI"), Err(ParseError::NoKeywords));
    assert_eq!(prompt::parse("", 8, "AI"), Err(ParseError::NoJson));
}

#[test]
fn the_prompt_asks_for_the_fixed_categories_and_the_count() {
    let text = prompt::prompt_text(5);
    assert!(text.contains("人物、动物、风景、建筑、食物、物体、文档、其他"));
    assert!(text.contains("2 到 5 个"));
    assert!(text.contains("只输出一个 JSON 对象"));
    assert!(text.contains("不要猜测真实姓名或具体地点"));
}

// ---------------------------------------------------------------- the write

fn demo() -> (LightcraftApp, PhotoId) {
    let app = LightcraftApp::new(Session::with_demo(), Services::default());
    let id = app.session.catalog.photos().next().unwrap().id;
    (app, id)
}

fn state_with(id: PhotoId, keywords: Vec<&str>) -> State {
    State {
        run_parent: "AI".to_string(),
        results: vec![PhotoResult {
            id,
            name: "DSC_0001.NEF".to_string(),
            category: "风景".to_string(),
            keywords: keywords.into_iter().map(str::to_string).collect(),
            seconds: 1.0,
        }],
        ..State::default()
    }
}

#[test]
fn the_write_is_one_batch_one_undo_step_and_undo_restores() {
    let (mut app, id) = demo();
    let before = app.session.catalog.photo(id).unwrap().meta.keywords.clone();
    let undo_before = app.session.undo.len();
    let st = state_with(id, vec!["雪山", "湖泊"]);
    let (photos, count) = super::apply(&mut app, &st).unwrap();
    assert_eq!((photos, count), (1, 2));
    let after = app.session.catalog.photo(id).unwrap().meta.keywords.clone();
    assert!(after.contains(&"AI|风景|雪山".to_string()), "the hierarchical path is written: {after:?}");
    assert!(after.contains(&"AI|风景|湖泊".to_string()));
    assert_eq!(after.len(), before.len() + 2, "the photo's own keywords stay");
    assert_eq!(app.session.undo.len(), undo_before + 1, "the whole batch is one undo step");
    assert_eq!(app.session.undo.last().unwrap().label, "Auto Keywords");
    // the recent keywords learned the new paths (§8)
    assert!(app.session.recent_keywords.iter().any(|k| k == "AI|风景|雪山"));
    // and undo puts the photo back
    app.run("edit.undo", Value::Null).unwrap();
    assert_eq!(app.session.catalog.photo(id).unwrap().meta.keywords, before);
}

#[test]
fn keywords_dedupe_against_the_photo_and_each_other() {
    use lightcraft_engine::catalog::keywords;
    let (mut app, id) = demo();
    // the photo already carries the keyword, spelled in another case
    let meta = Meta { keywords: vec!["ai|风景|雪山".to_string()], ..Default::default() };
    app.session.commit("test", Op::SetMeta { id, meta: Box::new(meta) }).unwrap();
    // the model repeats itself: the batch carries 湖泊 twice
    let st = state_with(id, vec!["湖泊", "湖泊"]);
    let (photos, count) = super::apply(&mut app, &st).unwrap();
    assert_eq!((photos, count), (1, 1), "the batch's repeats are one keyword");
    let after = app.session.catalog.photo(id).unwrap().meta.keywords.clone();
    assert_eq!(after.iter().filter(|k| keywords::same(k, "AI|风景|湖泊")).count(), 1);
    // the existing keyword (in another case) is not written again, a different leaf is
    let st = state_with(id, vec!["雪山", "SNow"]);
    let (photos, count) = super::apply(&mut app, &st).unwrap();
    assert_eq!((photos, count), (1, 1), "only AI|风景|SNow is new: the existing path counts whatever its case");
    let after = app.session.catalog.photo(id).unwrap().meta.keywords.clone();
    assert_eq!(after.iter().filter(|k| keywords::same(k, "ai|风景|雪山")).count(), 1);
    assert!(after.contains(&"AI|风景|SNow".to_string()));
}

#[test]
fn an_empty_parent_writes_two_level_paths() {
    let (mut app, id) = demo();
    let mut st = state_with(id, vec!["雪山"]);
    st.run_parent = String::new();
    super::apply(&mut app, &st).unwrap();
    let after = app.session.catalog.photo(id).unwrap().meta.keywords.clone();
    assert!(after.contains(&"风景|雪山".to_string()), "no empty level is written: {after:?}");
    assert!(after.iter().all(|k| !k.starts_with('|') && !k.contains("||")));
}

#[test]
fn photos_with_nothing_new_are_not_touched_and_an_empty_batch_is_not_committed() {
    let (mut app, id) = demo();
    let meta = Meta { keywords: vec!["AI|风景|雪山".to_string()], ..Default::default() };
    app.session.commit("test", Op::SetMeta { id, meta: Box::new(meta) }).unwrap();
    let undo_before = app.session.undo.len();
    // the only keyword is already on the photo: no SetMeta, no commit
    let st = state_with(id, vec!["雪山"]);
    let (photos, count) = super::apply(&mut app, &st).unwrap();
    assert_eq!((photos, count), (0, 0));
    assert_eq!(app.session.undo.len(), undo_before, "nothing was committed");
}

#[test]
fn excluded_photos_and_removed_keywords_are_not_written() {
    let (mut app, id) = demo();
    let mut st = state_with(id, vec!["雪山", "湖泊"]);
    st.excluded.insert(id.0);
    let (photos, _) = super::apply(&mut app, &st).unwrap();
    assert_eq!(photos, 0, "an un-ticked photo is not written");
    let mut st = state_with(id, vec!["雪山", "湖泊"]);
    st.removed.insert(id.0, ["雪山".to_string()].into_iter().collect());
    let (photos, count) = super::apply(&mut app, &st).unwrap();
    assert_eq!((photos, count), (1, 1), "only the kept keyword is written");
    let after = app.session.catalog.photo(id).unwrap().meta.keywords.clone();
    assert!(!after.contains(&"AI|风景|雪山".to_string()));
    assert!(after.contains(&"AI|风景|湖泊".to_string()));
}

// ---------------------------------------------------------------- settings

#[test]
fn settings_survive_a_roundtrip() {
    let dir = std::env::temp_dir().join(format!("lightcraft-akw-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = config::settings_path(Some(&dir)).unwrap();
    let settings = config::Settings {
        model_repo: "tencent/WeVisOne-VL-4B".to_string(),
        python_path: r"C:\Python314\python.exe".to_string(),
        max_keywords: 12,
        parent: "AI 标签".to_string(),
        skip_tagged: true,
        apply_mode: ApplyMode::Direct,
        timeout_secs: 60,
        ..config::Settings::default()
    };
    config::write(&path, &settings).unwrap();
    let read = config::read(&path);
    assert_eq!(read, settings);
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn a_damaged_settings_file_falls_back_to_the_defaults() {
    let dir = std::env::temp_dir().join(format!("lightcraft-akw-bad-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = config::settings_path(Some(&dir)).unwrap();
    for damaged in ["", "{", "not json at all", "{\"version\": 1, \"modelRepo\": 42}"] {
        std::fs::write(&path, damaged).unwrap();
        let read = config::read(&path);
        assert_eq!(read, config::Settings::default(), "{damaged:?} reads as the defaults");
    }
    // a value outside its range parses (serde defaults fill the rest) and is held at use
    std::fs::write(&path, "{\"maxKeywords\": 99999}").unwrap();
    let read = config::read(&path);
    assert_eq!(read.max_keywords(), config::MAX_KEYWORDS_MAX as usize, "the range holds at use");
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn a_v1_settings_file_still_reads_with_the_new_fields() {
    // version 1 named the retired Ollama backend (endpoint/protocol); those fields are ignored
    // now and the managed service takes over
    let dir = std::env::temp_dir().join(format!("lightcraft-akw-v1-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = config::settings_path(Some(&dir)).unwrap();
    std::fs::write(
        &path,
        r#"{"version": 1, "endpoint": "http://127.0.0.1:11434", "protocol": "ollama", "model": "qwen2.5vl:3b", "maxKeywords": 5, "parent": "AI", "skipTagged": false, "applyMode": "Review", "timeoutSecs": 90}"#,
    )
    .unwrap();
    let read = config::read(&path);
    assert_eq!(read.max_keywords(), 5, "the fields that carry over carry over");
    assert_eq!(read.timeout(), 90);
    assert_eq!(read.model_repo(), config::MODEL_DEFAULT, "the model comes from the service's default now");
    assert!(read.python_path.is_empty());
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn a_missing_settings_file_reads_as_the_defaults() {
    let dir = std::env::temp_dir().join(format!("lightcraft-akw-none-{}", std::process::id()));
    let path = config::settings_path(Some(&dir)).unwrap();
    assert_eq!(config::read(&path), config::Settings::default());
}

#[test]
fn an_empty_model_repo_falls_back_to_the_default() {
    let settings = config::Settings { model_repo: "   ".to_string(), ..config::Settings::default() };
    assert_eq!(settings.model_repo(), config::MODEL_DEFAULT);
}

// ---------------------------------------------------------------- the page

#[test]
fn the_page_renders_headless() {
    // the Auto Keywords page is a level-2 feature of the hub (the hub tests cover the
    // navigation); drawing it directly once must not panic, in the page's opening phase
    let mut app = LightcraftApp::new(Session::with_demo(), Services::default());
    let mut view = lightcraft_ui_egui::headless::HeadlessView::new();
    let input = lightcraft_ui_egui::headless::HeadlessView::raw_input(vec2(360.0, 780.0), 1.0, 0.0, vec![]);
    let _ = view.run(input, |ui| super::page(&mut app, ui));
}

#[test]
fn the_service_card_shows_exactly_one_half_of_the_toggle() {
    // drawn directly on a local state, so the test is deterministic whatever the (global)
    // manager's real service is doing
    let mut app = LightcraftApp::new(Session::with_demo(), Services::default());
    let mut view = lightcraft_ui_egui::headless::HeadlessView::new();
    let input = lightcraft_ui_egui::headless::HeadlessView::raw_input(vec2(420.0, 900.0), 1.0, 0.0, vec![]);
    let mut st = State::default();
    // a stopped service: the start half, no stop
    let _ = view.run(input.clone(), |ui| super::draw_service(&mut app, ui, &mut st));
    let registered: Vec<String> = lightcraft_ui_egui::widgets::take_registry(&view.ctx).into_iter().map(|(id, _)| id).collect();
    assert!(registered.iter().any(|id| id == "akw:service:start"), "the start half shows while stopped: {registered:?}");
    assert!(!registered.iter().any(|id| id == "akw:service:stop"), "no stop while stopped");
    // a ready service: the stop half, no start
    st.service.phase = super::server::Phase::Ready;
    st.service.device = "cpu".to_string();
    let _ = view.run(input, |ui| super::draw_service(&mut app, ui, &mut st));
    let registered: Vec<String> = lightcraft_ui_egui::widgets::take_registry(&view.ctx).into_iter().map(|(id, _)| id).collect();
    assert!(registered.iter().any(|id| id == "akw:service:stop"), "the stop half shows while ready: {registered:?}");
    assert!(!registered.iter().any(|id| id == "akw:service:start"), "no start while ready");
}

#[test]
fn the_page_survives_a_run_against_a_dead_service() {
    // a run whose service is gone: every photo fails, the failure summary shows, the settings
    // survive (§9 Failed)
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    drop(listener);
    let mut app = LightcraftApp::new(Session::with_demo(), Services::default());
    // two photos are enough to see the batch carry on past a failure (a refused connect costs
    // about a second apiece on Windows, so the whole demo library would be minutes)
    let ids: Vec<PhotoId> = app.session.catalog.photos().take(2).map(|p| p.id).collect();
    let entries: Vec<(PhotoId, String)> = ids.iter().map(|&id| (id, "x.jpg".to_string())).collect();
    let config = Arc::new(RunConfig {
        endpoint: vlm::parse_endpoint(&format!("http://127.0.0.1:{port}")).unwrap(),
        token: "t".to_string(),
        model: "m".to_string(),
        prompt: prompt::prompt_text(8),
        max_keywords: 8,
        parent: "AI".to_string(),
        stall_secs: 1,
    });
    {
        let mut st = super::lock();
        let ctx = egui::Context::default();
        super::begin_run(&mut app, &ctx, &mut st, entries, 0, config);
        assert_eq!(st.phase, Phase::Running);
        assert!(st.activity.is_some(), "the run holds an activity row");
    }
    // let the workers fail, then pump the page until it lands on the failure summary
    let ctx = egui::Context::default();
    for _ in 0..600 {
        {
            let mut st = super::lock();
            super::pump(&mut app, &ctx, &mut st);
            if st.phase == Phase::Failed {
                assert!(st.activity.is_none(), "the activity row went when the run ended");
                break;
            }
        }
        std::thread::sleep(Duration::from_millis(25));
    }
    {
        let st = super::lock();
        assert_eq!(st.phase, Phase::Failed, "every photo failed against a dead service");
        assert_eq!(st.failures.len(), ids.len(), "one failure per photo; the batch never stopped");
        assert!(st.results.is_empty());
    }
    // and back to idle, settings kept
    let mut st = super::lock();
    super::back_to_idle(&mut st);
    assert_eq!(st.phase, Phase::Idle);
    assert_eq!(st.settings.model_repo(), config::MODEL_DEFAULT);
}

#[test]
fn a_stopped_photo_is_not_recorded_as_a_failure() {
    // pump's drain: a photo the stop dropped (Reason::Stopped) is counted as unprocessed, not
    // failed (§9); one that genuinely failed is recorded
    let (mut app, _) = demo();
    let (tx, rx) = channel();
    tx.send(worker_done_for(PhotoId(1), "a.jpg", vlm::Reason::NoKeywords)).unwrap();
    tx.send(worker_done_for(PhotoId(2), "b.jpg", vlm::Reason::Stopped)).unwrap();
    drop(tx);
    let mut st = State {
        done_rx: Some(rx),
        phase: Phase::Running,
        activity: Some(app.session.activity.start("autoKeywords", "Auto Keywords", Cancel::No)),
        ..State::default()
    };
    let ctx = egui::Context::default();
    super::pump(&mut app, &ctx, &mut st);
    // nothing was queued, so the run also ends here: the failure summary
    assert_eq!(st.phase, Phase::Failed);
    assert_eq!(st.failures.len(), 1, "the stopped photo is not a failure");
    assert_eq!(st.failures[0].id, PhotoId(1));
}

/// A [`super::WorkerDone`] with one failed photo, for the drain test above.
fn worker_done_for(id: PhotoId, name: &str, reason: vlm::Reason) -> super::WorkerDone {
    super::WorkerDone { outcome: Outcome::Failed(PhotoFailure { id, name: name.to_string(), reason, seconds: 1.0 }) }
}
