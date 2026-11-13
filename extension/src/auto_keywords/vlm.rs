//! The local model service's HTTP client: the self-hosted Python server (see [`super::server`])
//! speaks an OpenAI-shaped chat request on the random loopback port and with the random token the
//! manager started it with.
//!
//! The privacy line the page promises is enforced here: addresses must be `http://` with a
//! loopback host ([`parse_endpoint`]), every resolved socket is checked with
//! [`std::net::IpAddr::is_loopback`] before it is connected to, no proxy is used, no credentials
//! beyond the process-local token are sent, and no redirect is followed. A reply is capped in
//! size and the reading loop watches the cancel flag, so a stalled or hostile service cannot hang
//! the run or the app.
//!
//! `lightcraft-fetch` only does GET (by design, for its update checks), so this module carries
//! its own POST path on `std::net::TcpStream`. The local service is plain HTTP, so no TLS client
//! is needed and the product stays pure Rust with no new dependencies.

use std::io::{ErrorKind, Read, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use serde_json::{Value, json};

/// How long connecting to the service may take.
pub const CONNECT_TIMEOUT: Duration = Duration::from_secs(5);
/// The reads poll at this interval, so the cancel flag is looked at at least this often (§6.4);
/// the worker's idle wait reuses it.
pub const READ_POLL: Duration = Duration::from_millis(200);
/// The longest a whole request may run, however fast the service answers between reads (§6.4).
const MAX_TOTAL: Duration = Duration::from_secs(180);
/// The largest response body read (§6.4): model replies are far smaller.
const MAX_RESPONSE_BYTES: usize = 4 << 20; // 4 MiB
/// The most of an error body kept for display (§6.4).
const MAX_ERROR_BODY: usize = 2 << 10; // 2 KiB
/// The largest response head read before it is called hostile.
const MAX_HEADER_BYTES: usize = 64 << 10; // 64 KiB

/// The only hosts a service address may name (§6.4). Anything else is refused before any bytes
/// are sent.
pub const LOOPBACK_HOSTS: [&str; 3] = ["127.0.0.1", "localhost", "::1"];

/// A parsed service address: the loopback host, the port, and a path prefix (`http://127.0.0.1:8321/sub`),
/// which is kept in front of every API path.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Endpoint {
    pub host: String,
    pub port: u16,
    pub prefix: String,
}

/// Parse the service address. Refuses anything that is not a plain `http://` loopback URL (§6.4):
/// `https://` gets its own explanation, other schemes and hosts their own. The address is built
/// by the service manager ([`super::server`]) and always carries the port it read from the
/// server's `LISTENING` line, so a missing port is an error.
pub fn parse_endpoint(url: &str) -> Result<Endpoint, String> {
    let url = url.trim();
    if url.to_lowercase().starts_with("https://") {
        return Err("A local model service is reached over http://, not https://.".to_string());
    }
    let Some(rest) = url.strip_prefix("http://") else {
        return Err("The service address must start with http://.".to_string());
    };
    let rest = rest.trim_start_matches('/');
    let (authority, path) = match rest.find('/') {
        Some(at) => (&rest[..at], rest[at..].trim_end_matches('/')),
        None => (rest, ""),
    };
    if authority.is_empty() {
        return Err("The service address needs a host, for example http://127.0.0.1:8321.".to_string());
    }
    if authority.contains('@') {
        return Err("Addresses with a user name or password are not supported.".to_string());
    }
    let (host_raw, port) = match authority.rsplit_once(':') {
        Some((host, port)) => {
            let port = port.parse::<u16>().map_err(|_| "The port in the service address is not a valid number.".to_string())?;
            (host, port)
        }
        None => return Err("The service address needs a port, for example http://127.0.0.1:8321.".to_string()),
    };
    // [::1]:11434 — brackets are URL syntax, not part of the host
    let host = host_raw.strip_prefix('[').and_then(|h| h.strip_suffix(']')).unwrap_or(host_raw);
    if !LOOPBACK_HOSTS.contains(&host.to_lowercase().as_str()) {
        return Err("Only a service on this computer is allowed: use 127.0.0.1, localhost or [::1].".to_string());
    }
    Ok(Endpoint { host: host.to_string(), port, prefix: path.to_string() })
}

/// Standard base64 (RFC 4648, with padding), for the JPEG handed to the model. Hand-rolled so the
/// page needs no dependency beyond the workspace's: 26 lines and a table, tested against the
/// RFC's own vectors.
pub fn base64_encode(data: &[u8]) -> String {
    const TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(data.len().div_ceil(3).saturating_mul(4));
    for chunk in data.chunks(3) {
        let b0 = u32::from(chunk[0]);
        let b1 = chunk.get(1).map_or(0, |&b| u32::from(b));
        let b2 = chunk.get(2).map_or(0, |&b| u32::from(b));
        let n = (b0 << 16) | (b1 << 8) | b2;
        out.push(TABLE[(n >> 18) as usize & 63] as char);
        out.push(TABLE[(n >> 12) as usize & 63] as char);
        out.push(if chunk.len() > 1 { TABLE[(n >> 6) as usize & 63] as char } else { '=' });
        out.push(if chunk.len() > 2 { TABLE[n as usize & 63] as char } else { '=' });
    }
    out
}

/// What a request failed with. Made on the worker thread; [`Reason::text`] words it for the UI
/// (in the UI language, on the UI thread).
///
/// [`Reason::text`]: super::Reason::text
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Reason {
    /// The service is not there (refused, unresolvable): the connect-fail page, and one photo's
    /// failure when the service went away mid-run.
    Unreachable(String),
    /// The request ran longer than its stall or total limit.
    Timeout { secs: u64 },
    /// A non-2xx answer, its (truncated) body shown.
    Http { status: u16, body: String },
    /// The service answered 2xx but reported an error itself.
    Service(String),
    /// A 2xx answer that is not what the protocol promised.
    BadReply(String),
    /// The model answered, but nothing usable survived the cleaning.
    NoKeywords,
    /// The photo is no longer in the catalog.
    Missing,
    /// The photo's job was dropped by a stop; recorded nowhere, only counted as not processed.
    Stopped,
    /// Anything else (an OS error's own words).
    Other(String),
}

impl Reason {
    /// The failure in UI words: translated on the UI thread (translations of static texts live in
    /// the locale table; dynamic ones show in English).
    pub fn text(&self) -> String {
        use lightcraft_ui_egui::i18n::tr;
        match self {
            Reason::Unreachable(d) if d == "Connection refused" => tr("The model service refused the connection.").to_string(),
            Reason::Unreachable(d) if d == "could not resolve the address" => tr("The service address could not be resolved.").to_string(),
            Reason::Unreachable(d) => format!("{}: {}", tr("Connection failed"), tr(d)),
            Reason::Timeout { secs } => tr("The request timed out ({secs} s).").replace("{secs}", &secs.to_string()),
            Reason::Http { status, body } => format!("HTTP {status}: {body}"),
            Reason::Service(d) => format!("{}: {d}", tr("The model service reported an error")),
            Reason::BadReply(d) => format!("{} ({})", tr("The model's answer could not be parsed"), tr(d)),
            Reason::NoKeywords => tr("The model did not return usable keywords.").to_string(),
            Reason::Missing => tr("The photo is no longer in the library.").to_string(),
            Reason::Stopped => String::new(),
            Reason::Other(d) => tr(d).to_string(),
        }
    }
}

/// An HTTP answer with its status and body.
struct RawResponse {
    status: u16,
    body: Vec<u8>,
}

fn io_reason(e: &std::io::Error) -> Reason {
    match e.kind() {
        ErrorKind::ConnectionRefused => Reason::Unreachable("Connection refused".to_string()),
        ErrorKind::WouldBlock | ErrorKind::TimedOut => Reason::Timeout { secs: MAX_TOTAL.as_secs() },
        _ => Reason::Other(e.to_string()),
    }
}

/// `needle` inside `haystack`, or `None`. Both are bounded (`MAX_HEADER_BYTES`,
/// `MAX_RESPONSE_BYTES`), so the linear scan stays cheap even if it runs per read.
fn find(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || needle.len() > haystack.len() {
        return None;
    }
    (0..=haystack.len() - needle.len()).find(|&at| &haystack[at..at + needle.len()] == needle)
}

/// One read, with the guards of §6.4 around it: the cancel flag (checked at least every
/// `READ_POLL`, since the socket's read timeout is that), the per-read stall limit, and the total
/// limit. `Ok(None)` is the connection's end.
fn read_step(
    stream: &mut TcpStream,
    buf: &mut [u8],
    last_data: &mut Instant,
    start: &Instant,
    stall: Duration,
    cancel: Option<&AtomicBool>,
) -> Result<Option<usize>, Reason> {
    loop {
        if let Some(flag) = cancel
            && flag.load(Ordering::Relaxed)
        {
            return Err(Reason::Stopped);
        }
        if last_data.elapsed() > stall {
            return Err(Reason::Timeout { secs: stall.as_secs() });
        }
        if start.elapsed() > MAX_TOTAL {
            return Err(Reason::Timeout { secs: MAX_TOTAL.as_secs() });
        }
        match stream.read(buf) {
            Ok(0) => return Ok(None),
            Ok(n) => {
                *last_data = Instant::now();
                return Ok(Some(n));
            }
            // the read timeout fired: no data yet — look at the flags again
            Err(e) if e.kind() == ErrorKind::WouldBlock || e.kind() == ErrorKind::TimedOut => continue,
            Err(e) => return Err(io_reason(&e)),
        }
    }
}

/// The decoded body of a complete chunked response body, or `None` while it is still arriving.
fn decode_chunked(buf: &[u8]) -> Option<Vec<u8>> {
    let mut out = Vec::new();
    let mut at = 0usize;
    loop {
        let line_end = find(&buf[at.min(buf.len())..], b"\r\n")? + at.min(buf.len());
        let size_line = std::str::from_utf8(&buf[at..line_end]).ok()?;
        // chunk extensions (`;name=value`) are legal and ignored
        let size = usize::from_str_radix(size_line.split(';').next()?.trim(), 16).ok()?;
        at = line_end.saturating_add(2);
        if size == 0 {
            // nothing after, or trailers: the blank line ends them
            if buf.get(at..at.saturating_add(2)) == Some(b"\r\n") || find(&buf[at.min(buf.len())..], b"\r\n\r\n").is_some() {
                return Some(out);
            }
            return None;
        }
        let data_end = at.checked_add(size)?;
        let data = buf.get(at..data_end)?;
        out.extend_from_slice(data);
        at = data_end;
        if buf.get(at..at.saturating_add(2)) != Some(b"\r\n") {
            return None;
        }
        at = at.saturating_add(2);
    }
}

/// Read a whole response: head, then the body by `Content-Length`, chunked encoding, or
/// connection close (§6.4).
fn read_response(stream: &mut TcpStream, stall: Duration, cancel: Option<&AtomicBool>) -> Result<RawResponse, Reason> {
    let start = Instant::now();
    let mut last_data = start;
    let mut buf: Vec<u8> = Vec::new();
    let mut tmp = [0u8; 16 * 1024];
    let head_end = loop {
        if let Some(at) = find(&buf, b"\r\n\r\n") {
            break at + 4;
        }
        if buf.len() > MAX_HEADER_BYTES {
            return Err(Reason::BadReply("the response headers are too large".to_string()));
        }
        match read_step(stream, &mut tmp, &mut last_data, &start, stall, cancel)? {
            None => return Err(Reason::BadReply("the connection closed before a response arrived".to_string())),
            Some(n) => buf.extend_from_slice(&tmp[..n]),
        }
    };
    let head = std::str::from_utf8(&buf[..head_end]).map_err(|_| Reason::BadReply("the response head is not text".to_string()))?;
    let mut lines = head.split("\r\n");
    let status_line = lines.next().unwrap_or_default();
    let mut parts = status_line.splitn(3, ' ');
    let _version = parts.next();
    let status: u16 = parts
        .next()
        .and_then(|code| code.trim().parse().ok())
        .ok_or_else(|| Reason::BadReply(format!("the service answered with an unreadable status line: {status_line}")))?;
    let headers: Vec<(String, String)> =
        lines.filter_map(|line| line.split_once(':')).map(|(k, v)| (k.trim().to_lowercase(), v.trim().to_string())).collect();
    let mut body = buf.split_off(head_end);
    let chunked = headers.iter().any(|(k, v)| k == "transfer-encoding" && v.to_lowercase().contains("chunked"));
    let content_length = match headers.iter().find(|(k, _)| k == "content-length") {
        Some((_, v)) => Some(v.parse::<usize>().map_err(|_| Reason::BadReply("the Content-Length header is not a number".to_string()))?),
        None => None,
    };
    if let Some(n) = content_length
        && n > MAX_RESPONSE_BYTES
    {
        return Err(Reason::BadReply(format!("the response is larger than {} MiB", MAX_RESPONSE_BYTES >> 20)));
    }
    loop {
        if chunked {
            if let Some(decoded) = decode_chunked(&body) {
                body = decoded;
                break;
            }
        } else if let Some(n) = content_length
            && body.len() >= n
        {
            body.truncate(n);
            break;
        }
        if body.len() > MAX_RESPONSE_BYTES {
            return Err(Reason::BadReply(format!("the response is larger than {} MiB", MAX_RESPONSE_BYTES >> 20)));
        }
        match read_step(stream, &mut tmp, &mut last_data, &start, stall, cancel)? {
            None => {
                // the connection's end: a chunked or unterminated body must already be complete
                if chunked {
                    body = decode_chunked(&body).ok_or_else(|| Reason::BadReply("the chunked response ended incomplete".to_string()))?;
                } else if content_length.is_some_and(|n| body.len() < n) {
                    return Err(Reason::BadReply("the connection closed before the whole response arrived".to_string()));
                }
                break;
            }
            Some(n) => body.extend_from_slice(&tmp[..n]),
        }
    }
    Ok(RawResponse { status, body })
}

/// Connect to `ep` — only to a socket that really is loopback. The host literal was checked in
/// [`parse_endpoint`]; a `localhost` that resolves somewhere else (a broken hosts file) is still
/// refused here, so the pictures cannot leave the machine.
fn connect(ep: &Endpoint) -> Result<TcpStream, Reason> {
    let addrs: Vec<_> = (ep.host.as_str(), ep.port)
        .to_socket_addrs()
        .map_err(|_| Reason::Unreachable("could not resolve the address".to_string()))?
        .filter(|addr| addr.ip().is_loopback())
        .collect();
    if addrs.is_empty() {
        return Err(Reason::Unreachable("not a local address".to_string()));
    }
    let mut last = Reason::Unreachable("Connection refused".to_string());
    for addr in addrs {
        match TcpStream::connect_timeout(&addr, CONNECT_TIMEOUT) {
            Ok(stream) => {
                let _ = stream.set_nodelay(true);
                // every read returns at the latest after READ_POLL, so the flags above are looked at
                // at least that often (§6.4)
                let _ = stream.set_read_timeout(Some(READ_POLL));
                let _ = stream.set_write_timeout(Some(CONNECT_TIMEOUT));
                return Ok(stream);
            }
            Err(e) => {
                // keep the first meaningful error (a refusal outranks a later timeout)
                if matches!(last, Reason::Unreachable(_)) {
                    last = io_reason(&e);
                }
            }
        }
    }
    Err(last)
}

/// One HTTP request, `body` when it is a POST, always carrying the service's bearer token. Never
/// follows redirects (they cannot reach a loopback service legitimately, and following one could
/// carry the pictures away), never uses a proxy.
fn http(
    ep: &Endpoint,
    method: &str,
    path: &str,
    token: &str,
    body: Option<&str>,
    stall: Duration,
    cancel: Option<&AtomicBool>,
) -> Result<RawResponse, Reason> {
    let mut stream = connect(ep)?;
    let host_header = if ep.host.contains(':') { format!("[{}]:{}", ep.host, ep.port) } else { format!("{}:{}", ep.host, ep.port) };
    let mut request = format!("{method} {path} HTTP/1.1\r\nHost: {host_header}\r\nAccept: application/json\r\n");
    request.push_str(&format!("Authorization: Bearer {token}\r\n"));
    request.push_str("Connection: close\r\n");
    if let Some(body) = body {
        request.push_str(&format!("Content-Type: application/json\r\nContent-Length: {}\r\n", body.len()));
        request.push_str("\r\n");
        request.push_str(body);
    } else {
        request.push_str("\r\n");
    }
    stream.write_all(request.as_bytes()).map_err(|e| io_reason(&e))?;
    stream.flush().map_err(|e| io_reason(&e))?;
    let raw = read_response(&mut stream, stall, cancel)?;
    if !(200..300).contains(&raw.status) {
        let body = String::from_utf8_lossy(&raw.body[..raw.body.len().min(MAX_ERROR_BODY)]).into_owned();
        return Err(Reason::Http { status: raw.status, body });
    }
    Ok(raw)
}

fn with_prefix(ep: &Endpoint, path: &str) -> String {
    format!("{}{}", ep.prefix, path)
}

/// Ask the model about one image (§6.3): the prompt and the image as a `data:` URL in one user
/// message, the reply's `choices[0].message.content` back — a JSON string of category and
/// keywords, cleaned by [`super::prompt::parse`]. The reply size, the timeouts and the cancel
/// flag are all bounded here.
pub fn analyze(
    ep: &Endpoint,
    token: &str,
    model: &str,
    prompt_text: &str,
    image_b64: &str,
    stall_secs: u32,
    cancel: Option<&AtomicBool>,
) -> Result<String, Reason> {
    let stall = Duration::from_secs(u64::from(stall_secs.max(1)));
    let body = json!({
        "model": model,
        "messages": [{ "role": "user", "content": [
            { "type": "text", "text": prompt_text },
            { "type": "image_url", "image_url": { "url": format!("data:image/jpeg;base64,{image_b64}") } },
        ]}],
        "temperature": 0.0,
        "max_tokens": 512,
        "stream": false,
    })
    .to_string();
    let raw = http(ep, "POST", &with_prefix(ep, "/v1/chat/completions"), token, Some(&body), stall, cancel)?;
    let v: Value = serde_json::from_slice(&raw.body).map_err(|_| Reason::BadReply("the reply is not JSON".to_string()))?;
    if let Some(message) = v.pointer("/error/message").and_then(Value::as_str).or_else(|| v.get("error").and_then(Value::as_str)) {
        return Err(Reason::Service(message.to_string()));
    }
    let content = v
        .pointer("/choices/0/message/content")
        .and_then(Value::as_str)
        .ok_or_else(|| Reason::BadReply("the reply has no message content".to_string()))?;
    Ok(content.to_string())
}
