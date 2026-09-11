//! A very small HTTP/1.1 server: enough to serve a status page and a JSON
//! endpoint to one operator, and no more than that.
//!
//! # Why not a web framework
//!
//! This crate's dependency list is argued line by line — `wgpu` gated
//! because a server has no GPU, `rodio` gated because `alsa-sys` needs
//! pkg-config, `tokio` pinned to `io-util` alone because the runtime
//! belongs to whoever drives the connection. Pulling in hyper, axum or
//! even `tiny_http` for four routes would be the first dependency here
//! that nobody could justify in a sentence.
//!
//! What this is *not*: it does not speak HTTP/2, chunked transfer
//! encoding, keep-alive, compression, ranges, or TLS. It reads a request
//! line and headers, discards the body, and writes one response. That is
//! the entire contract, and anything needing more should put a real
//! reverse proxy in front.
//!
//! # Deliberate limits
//!
//! Every bound here exists because this socket is reachable by whatever
//! can reach the loopback interface — which on a shared host is more than
//! just the operator:
//!
//! - The request line and headers are capped at [`MAX_HEADER_BYTES`]. A
//!   peer that opens a connection and streams header bytes forever would
//!   otherwise grow a `Vec` until the process dies.
//! - Reads and writes carry a timeout ([`IO_TIMEOUT`]). A peer that
//!   connects and then says nothing holds a worker thread; with a timeout
//!   it holds one for five seconds.
//! - One thread per connection, capped by [`MAX_CONCURRENT`]. Past the
//!   cap, connections are accepted and closed immediately rather than
//!   queued, so a flood cannot spawn unbounded threads.
//!
//! None of this makes the endpoint safe to expose publicly. It makes it
//! safe to leave running on localhost, which is the documented use.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

/// Longest request line + headers accepted, in bytes.
pub const MAX_HEADER_BYTES: usize = 8 * 1024;

/// How long a single read or write may block before the connection is
/// dropped.
pub const IO_TIMEOUT: Duration = Duration::from_secs(5);

/// Connections served simultaneously. Past this, new ones are closed
/// without being read.
pub const MAX_CONCURRENT: usize = 16;

/// What the caller learned from a request line.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Request {
    /// Uppercase, as it arrived: `GET`, `POST`, anything else.
    pub method: String,
    /// Path only, query string stripped and *not* decoded.
    pub path: String,
}

/// What to send back.
pub struct Response {
    pub status: u16,
    pub content_type: &'static str,
    pub body: Vec<u8>,
}

impl Response {
    pub fn html(body: impl Into<Vec<u8>>) -> Self {
        Self { status: 200, content_type: "text/html; charset=utf-8", body: body.into() }
    }

    pub fn json(body: impl Into<Vec<u8>>) -> Self {
        Self { status: 200, content_type: "application/json; charset=utf-8", body: body.into() }
    }

    pub fn not_found() -> Self {
        Self {
            status: 404,
            content_type: "text/plain; charset=utf-8",
            body: b"not found\n".to_vec(),
        }
    }

    /// 405, naming what *is* allowed — this server is read-only, and a
    /// `POST` arriving is worth answering clearly rather than with a 404
    /// that reads like a typo.
    pub fn method_not_allowed() -> Self {
        Self {
            status: 405,
            content_type: "text/plain; charset=utf-8",
            body: b"read-only: only GET is served\n".to_vec(),
        }
    }

    fn reason(&self) -> &'static str {
        match self.status {
            200 => "OK",
            400 => "Bad Request",
            404 => "Not Found",
            405 => "Method Not Allowed",
            413 => "Payload Too Large",
            503 => "Service Unavailable",
            _ => "Internal Server Error",
        }
    }

    /// Serialise to the wire.
    ///
    /// `Connection: close` on every response, because this server does not
    /// implement keep-alive and claiming otherwise would leave a browser
    /// waiting on a second response that never comes.
    pub fn write_to(&self, w: &mut impl Write) -> std::io::Result<()> {
        let head = format!(
            "HTTP/1.1 {} {}\r\n\
             Content-Type: {}\r\n\
             Content-Length: {}\r\n\
             Connection: close\r\n\
             Cache-Control: no-store\r\n\
             X-Content-Type-Options: nosniff\r\n\
             \r\n",
            self.status,
            self.reason(),
            self.content_type,
            self.body.len(),
        );
        w.write_all(head.as_bytes())?;
        w.write_all(&self.body)?;
        w.flush()
    }
}

/// Parse a request line and drain the headers.
///
/// Returns `None` for anything malformed. The body is deliberately not
/// read: nothing here consumes one, and reading an arbitrary
/// `Content-Length` would be the easiest way to make this endpoint a
/// memory sink.
pub fn read_request(stream: &mut impl Read) -> Option<Request> {
    let mut reader = BufReader::new(stream.take(MAX_HEADER_BYTES as u64));

    let mut line = String::new();
    if reader.read_line(&mut line).ok()? == 0 {
        return None;
    }

    // `GET /path?query HTTP/1.1`
    let mut parts = line.split_whitespace();
    let method = parts.next()?.to_ascii_uppercase();
    let target = parts.next()?;
    // A version is required by HTTP/1.1 but this server does not care
    // which; rejecting on it would break curl's HTTP/1.0 mode for no gain.
    let _version = parts.next();

    let path = target.split('?').next().unwrap_or("/").to_string();
    if !path.starts_with('/') {
        return None;
    }

    // Drain headers to the blank line. Capped by the `take` above, so a
    // peer streaming headers forever hits EOF rather than growing this.
    loop {
        let mut header = String::new();
        match reader.read_line(&mut header) {
            Ok(0) => break,
            Ok(_) if header.trim().is_empty() => break,
            Ok(_) => continue,
            Err(_) => return None,
        }
    }

    Some(Request { method, path })
}

/// A running admin server. Dropping it stops accepting new connections.
///
/// The listener thread is detached rather than joined on drop: it is
/// blocked in `accept`, and waking it would mean either a self-connect or
/// a poll loop, neither of which is worth the complexity for a status
/// page that lives as long as the process.
pub struct Server {
    addr: std::net::SocketAddr,
    live: Arc<AtomicUsize>,
}

impl Server {
    /// The address actually bound, which is what a caller should log.
    ///
    /// Not the address requested: binding port 0 asks the OS to choose,
    /// and the tests rely on that to avoid fighting over a fixed port.
    pub fn addr(&self) -> std::net::SocketAddr {
        self.addr
    }

    /// Connections currently being served.
    pub fn live_connections(&self) -> usize {
        self.live.load(Ordering::Relaxed)
    }
}

/// Bind and start serving, handing every request to `handler`.
///
/// `handler` is called on a worker thread, so it must be `Send + Sync`;
/// it is shared across connections rather than cloned per request.
///
/// # Binding
///
/// Whatever `addr` says — but see [`crate::admin`] for why the documented
/// default is loopback, and why this function does not default it for
/// you: a bind address is a deployment decision, and silently choosing
/// one is how an admin endpoint ends up reachable from outside.
pub fn serve<H>(addr: std::net::SocketAddr, handler: H) -> std::io::Result<Server>
where
    H: Fn(&Request) -> Response + Send + Sync + 'static,
{
    let listener = TcpListener::bind(addr)?;
    let bound = listener.local_addr()?;
    let live = Arc::new(AtomicUsize::new(0));

    let handler = Arc::new(handler);
    let thread_live = live.clone();

    std::thread::Builder::new()
        .name("void-admin".to_string())
        .spawn(move || {
            for stream in listener.incoming() {
                let Ok(stream) = stream else { continue };

                // Shed rather than queue. A queued connection still holds
                // a socket and still gets served eventually, which under a
                // flood means the operator's own request waits behind a
                // thousand others.
                if thread_live.load(Ordering::Relaxed) >= MAX_CONCURRENT {
                    let mut s = stream;
                    let _ = Response {
                        status: 503,
                        content_type: "text/plain; charset=utf-8",
                        body: b"busy\n".to_vec(),
                    }
                    .write_to(&mut s);
                    continue;
                }

                let handler = handler.clone();
                let live = thread_live.clone();
                live.fetch_add(1, Ordering::Relaxed);
                let spawned = std::thread::Builder::new()
                    .name("void-admin-conn".to_string())
                    .spawn(move || {
                        serve_one(stream, handler.as_ref());
                        live.fetch_sub(1, Ordering::Relaxed);
                    });
                if spawned.is_err() {
                    // Could not spawn: undo the count we optimistically
                    // took, or the server would permanently believe it is
                    // busier than it is and shed forever.
                    thread_live.fetch_sub(1, Ordering::Relaxed);
                }
            }
        })?;

    Ok(Server { addr: bound, live })
}

fn serve_one<H>(mut stream: TcpStream, handler: &H)
where
    H: Fn(&Request) -> Response,
{
    let _ = stream.set_read_timeout(Some(IO_TIMEOUT));
    let _ = stream.set_write_timeout(Some(IO_TIMEOUT));

    let response = match read_request(&mut stream) {
        Some(req) if req.method == "GET" => handler(&req),
        Some(_) => Response::method_not_allowed(),
        None => Response {
            status: 400,
            content_type: "text/plain; charset=utf-8",
            body: b"bad request\n".to_vec(),
        },
    };

    let _ = response.write_to(&mut stream);
    // Best-effort: the peer may already be gone, and a failed shutdown on
    // a connection we are dropping anyway is not worth reporting.
    let _ = stream.shutdown(std::net::Shutdown::Both);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn parses_a_plain_get() {
        let mut c = Cursor::new(b"GET /api/status HTTP/1.1\r\nHost: x\r\n\r\n".to_vec());
        let req = read_request(&mut c).expect("a well-formed request must parse");
        assert_eq!(req.method, "GET");
        assert_eq!(req.path, "/api/status");
    }

    /// A query string is not part of the route, and must not turn one path
    /// into another.
    #[test]
    fn strips_the_query_string() {
        let mut c = Cursor::new(b"GET /?refresh=2 HTTP/1.1\r\n\r\n".to_vec());
        assert_eq!(read_request(&mut c).unwrap().path, "/");
    }

    #[test]
    fn lowercase_methods_normalise() {
        let mut c = Cursor::new(b"get / HTTP/1.1\r\n\r\n".to_vec());
        assert_eq!(read_request(&mut c).unwrap().method, "GET");
    }

    #[test]
    fn rejects_garbage() {
        let mut c = Cursor::new(b"\r\n".to_vec());
        assert!(read_request(&mut c).is_none(), "an empty request line is not a request");

        let mut c = Cursor::new(b"GET\r\n\r\n".to_vec());
        assert!(read_request(&mut c).is_none(), "a method with no target is not a request");

        let mut c = Cursor::new(b"GET http://evil/ HTTP/1.1\r\n\r\n".to_vec());
        assert!(read_request(&mut c).is_none(), "an absolute-form target is not served here");
    }

    /// The bound that stops a peer streaming headers until the process
    /// dies. It must terminate, not merely be documented.
    #[test]
    fn a_flood_of_headers_terminates() {
        let mut body = b"GET / HTTP/1.1\r\n".to_vec();
        for i in 0..100_000 {
            body.extend_from_slice(format!("X-Pad-{i}: aaaaaaaaaaaaaaaaaaaa\r\n").as_bytes());
        }
        // No terminating blank line: the only thing that can stop this is
        // the cap.
        let mut c = Cursor::new(body);
        let req = read_request(&mut c);
        assert!(req.is_some(), "the request line itself was well-formed");
        assert_eq!(req.unwrap().path, "/");
    }

    #[test]
    fn responses_carry_length_and_close() {
        let mut out = Vec::new();
        Response::json(b"{}".to_vec()).write_to(&mut out).unwrap();
        let text = String::from_utf8(out).unwrap();
        assert!(text.starts_with("HTTP/1.1 200 OK\r\n"));
        assert!(text.contains("Content-Length: 2\r\n"));
        assert!(text.contains("Connection: close\r\n"));
        assert!(text.contains("Content-Type: application/json"));
        assert!(text.ends_with("\r\n\r\n{}"));
    }

    /// `nosniff` matters here: the status page renders operator-supplied
    /// strings (an asset name, a degraded-writer reason), and a browser
    /// guessing at content type is one of the ways that becomes a script.
    #[test]
    fn responses_refuse_content_sniffing() {
        let mut out = Vec::new();
        Response::html("<p>hi</p>").write_to(&mut out).unwrap();
        let text = String::from_utf8(out).unwrap();
        assert!(text.contains("X-Content-Type-Options: nosniff\r\n"));
        assert!(text.contains("Cache-Control: no-store\r\n"), "a status page must not be cached");
    }

    #[test]
    fn status_codes_carry_their_reason() {
        assert!(String::from_utf8(render(Response::not_found()))
            .unwrap()
            .starts_with("HTTP/1.1 404 Not Found"));
        assert!(String::from_utf8(render(Response::method_not_allowed()))
            .unwrap()
            .starts_with("HTTP/1.1 405 Method Not Allowed"));
    }

    fn render(r: Response) -> Vec<u8> {
        let mut out = Vec::new();
        r.write_to(&mut out).unwrap();
        out
    }
}
