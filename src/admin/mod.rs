//! A read-only operator status page — the `admin` feature.
//!
//! The ledger grew a set of safety mechanisms that nothing called:
//! `audit_zero_sum`, `audit_balances_match_entries`, `health`,
//! `journal_depth`, `reconcile_now`. Each was correct, tested, and
//! unreachable from a running server — `SimCtx` carries `world`, `input`
//! and `dt`, so a game holding a ledger had to invent its own health
//! endpoint to see any of it. This is that endpoint, written once.
//!
//! ```no_run
//! # #[cfg(feature = "admin")] {
//! use std::sync::{Arc, Mutex};
//! use void_engine::admin::{self, Status};
//! use void_engine::persist::ledger::Ledger;
//!
//! // Whatever the game already holds, shared with the page.
//! let ledger = Arc::new(Mutex::new(Ledger::new()));
//! let for_page = ledger.clone();
//!
//! let server = admin::serve(admin::loopback(8080), move || {
//!     let l = for_page.lock().unwrap();
//!     Status::from_ledger(&l, 0)
//! }).expect("bind the admin port");
//!
//! println!("status page on http://{}", server.addr());
//! # }
//! ```
//!
//! # Read-only, and bound where you say
//!
//! There are no mutating routes. `GET /` serves the page, `GET
//! /api/status` serves the same data as JSON, and anything else is a 404;
//! a `POST` gets a 405 explaining why. Triggering a reconciliation from
//! here would be genuinely useful and is deliberately absent — that is a
//! privileged action on a socket, and it needs an authentication story
//! this does not have.
//!
//! [`serve`](crate::admin::serve) takes the bind address as a required
//! argument and does not default it. [`loopback`](crate::admin::loopback)
//! is the address you almost certainly want:
//! this page exposes account balances, audit findings and a database
//! error message, none of which belongs on a public interface. Putting a
//! reverse proxy in front of it — with TLS and real authentication — is
//! the supported way to reach it from elsewhere.
//!
//! # What it costs
//!
//! One thread blocked in `accept`, plus one short-lived thread per
//! request, capped at [`MAX_CONCURRENT`]. No new dependencies: the
//! HTTP/1.1 subset is in [`http`] and the JSON writer in [`json`], both
//! small enough to read in one sitting, because adding a web framework to
//! serve two routes would be the one dependency in this crate that nobody
//! could justify in a sentence.
//!
//! [`MAX_CONCURRENT`]: crate::admin::http::MAX_CONCURRENT
//! [`http`]: crate::admin::http
//! [`json`]: crate::admin::json

pub mod http;
pub mod json;
pub mod page;
pub mod status;

pub use http::{Request, Response, Server};
pub use status::{Drift, Status, TickSummary, Writer, WriterState};

use std::net::SocketAddr;

/// `127.0.0.1:port` — the address this page is designed for.
pub fn loopback(port: u16) -> SocketAddr {
    SocketAddr::from(([127, 0, 0, 1], port))
}

/// Serve the status page, calling `snapshot` once per request.
///
/// The closure is called on a worker thread and must therefore be `Send +
/// Sync`. It is called *per request* rather than cached, so the page
/// always reflects the ledger as it is now — and so a slow snapshot shows
/// up as a slow page rather than as stale data.
///
/// # Locking
///
/// Whatever `snapshot` locks is held for the duration of the snapshot,
/// which on a large ledger means `audit_zero_sum` walking the whole log
/// while the simulation waits. A server that cannot afford that should
/// snapshot into a cache on its own tick and have this closure read the
/// cache.
pub fn serve<F>(addr: SocketAddr, snapshot: F) -> std::io::Result<Server>
where
    F: Fn() -> Status + Send + Sync + 'static,
{
    http::serve(addr, move |req: &Request| match req.path.as_str() {
        "/" | "/index.html" => Response::html(page::render_html(&snapshot())),
        "/api/status" => Response::json(page::render_json(&snapshot())),
        _ => Response::not_found(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};
    use std::net::TcpStream;

    /// Drive the real server over a real socket, so the wiring is tested
    /// rather than the renderers alone.
    fn get(addr: SocketAddr, path: &str) -> String {
        let mut s = TcpStream::connect(addr).expect("connect");
        write!(s, "GET {path} HTTP/1.1\r\nHost: localhost\r\n\r\n").unwrap();
        let mut out = String::new();
        s.read_to_string(&mut out).expect("read");
        out
    }

    fn send_raw(addr: SocketAddr, raw: &str) -> String {
        let mut s = TcpStream::connect(addr).expect("connect");
        s.write_all(raw.as_bytes()).unwrap();
        let mut out = String::new();
        s.read_to_string(&mut out).expect("read");
        out
    }

    #[test]
    fn serves_the_page_and_the_endpoint() {
        // Port 0: let the OS pick, so concurrent test runs cannot collide.
        let server = serve(loopback(0), Status::default).expect("bind");
        let addr = server.addr();

        let page = get(addr, "/");
        assert!(page.starts_with("HTTP/1.1 200 OK"));
        assert!(page.contains("text/html"));
        assert!(page.contains("void_engine — server status"));

        let api = get(addr, "/api/status");
        assert!(api.starts_with("HTTP/1.1 200 OK"));
        assert!(api.contains("application/json"));
        assert!(api.contains(r#""healthy":true"#));
    }

    #[test]
    fn unknown_paths_are_404() {
        let server = serve(loopback(0), Status::default).expect("bind");
        let res = get(server.addr(), "/secrets");
        assert!(res.starts_with("HTTP/1.1 404 Not Found"), "got {res}");
    }

    /// The read-only promise, enforced rather than merely documented.
    #[test]
    fn mutating_methods_are_refused() {
        let server = serve(loopback(0), Status::default).expect("bind");
        for method in ["POST", "PUT", "DELETE", "PATCH"] {
            let res = send_raw(
                server.addr(),
                &format!("{method} /api/status HTTP/1.1\r\nHost: x\r\n\r\n"),
            );
            assert!(
                res.starts_with("HTTP/1.1 405 Method Not Allowed"),
                "{method} was not refused: {res}",
            );
        }
    }

    #[test]
    fn a_malformed_request_gets_400_not_a_panic() {
        let server = serve(loopback(0), Status::default).expect("bind");
        let res = send_raw(server.addr(), "not-a-request\r\n\r\n");
        assert!(res.starts_with("HTTP/1.1 400 Bad Request"), "got {res}");
    }

    /// The snapshot closure runs per request, so the page cannot go stale.
    #[test]
    fn the_snapshot_is_taken_per_request() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;

        let calls = Arc::new(AtomicUsize::new(0));
        let counter = calls.clone();
        let server = serve(loopback(0), move || {
            counter.fetch_add(1, Ordering::SeqCst);
            Status::default()
        })
        .expect("bind");

        get(server.addr(), "/");
        get(server.addr(), "/api/status");
        assert_eq!(calls.load(Ordering::SeqCst), 2, "each request must resample");
    }

    #[test]
    fn loopback_binds_only_the_loopback_interface() {
        assert_eq!(loopback(9).ip().to_string(), "127.0.0.1");
        let server = serve(loopback(0), Status::default).expect("bind");
        assert!(server.addr().ip().is_loopback(), "the default must not be reachable off-box");
    }
}
