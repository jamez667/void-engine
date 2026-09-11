//! Length-prefixed message framing over any async byte stream.
//!
//! QUIC streams (and TCP) are byte streams, not message streams: a
//! `read` can hand you half a message or two-and-a-bit. A u32-LE length
//! prefix restores message boundaries, which is all a control channel
//! needs on top of the transport. Datagrams already carry their own
//! boundaries — this is for the reliable-stream side (handshake, login,
//! chat, anything that must not be dropped).
//!
//! Lifted from `void_proto::framing` in void-claim, with one change that
//! is the whole reason it isn't a verbatim copy: `read_msg` there did
//! `vec![0u8; len]` on a length taken straight off the wire. A hostile
//! peer — or a corrupt frame, or a desynced stream where the reader is
//! four bytes out of phase — sends `0xFFFFFFFF` and the process
//! allocates 4 GiB and dies, from a single unauthenticated packet. The
//! `max_len` argument is a mandatory bound: pick the largest legitimate
//! message your protocol has and refuse anything above it *before*
//! allocating. The cap is a parameter rather than a constant because
//! "largest legitimate message" is a per-protocol fact the engine cannot
//! know.

use tokio::io::{AsyncReadExt, AsyncWriteExt};

/// Write `data` as one length-prefixed frame.
///
/// Errors with `InvalidInput` if `data` exceeds `u32::MAX`, since the
/// prefix could not describe it — a silent truncating cast would desync
/// the stream permanently.
pub async fn write_msg<W: AsyncWriteExt + Unpin>(w: &mut W, data: &[u8]) -> std::io::Result<()> {
    let len = u32::try_from(data.len()).map_err(|_| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("message of {} B exceeds the u32 length prefix", data.len()),
        )
    })?;
    w.write_all(&len.to_le_bytes()).await?;
    w.write_all(data).await
}

/// Read one length-prefixed frame, refusing anything longer than
/// `max_len` before it is allocated.
///
/// An over-cap frame returns `InvalidData` and the stream is left
/// mid-frame — it is not resynchronisable, so callers should treat this
/// as fatal for the connection and close it.
///
/// # This waits forever, on purpose
///
/// There is no timeout here and there cannot usefully be one: this is
/// generic over any `AsyncRead`, and a timer would drag a tokio timer
/// feature onto every caller — including tests that drive it over an
/// in-memory buffer with no runtime at all. A control channel that is
/// legitimately idle for an hour is also not an error, so the right
/// deadline is a per-protocol fact, exactly like `max_len`.
///
/// **A peer that opens a stream and then says nothing will park this
/// future indefinitely.** On a sequential accept loop that parks the
/// whole loop, so every later message from that connection is starved by
/// one silent stream. Callers reading from an untrusted peer must impose
/// their own bound:
///
/// ```ignore
/// // Five seconds to send a 4-byte ack, or the stream is abandoned.
/// match tokio::time::timeout(Duration::from_secs(5), read_msg(&mut recv, MAX_ACK_BYTES)).await {
///     Ok(Ok(bytes)) => handle(bytes),
///     Ok(Err(e))    => log::debug!("malformed frame: {e}"),
///     Err(_elapsed) => log::debug!("peer opened a stream and sent nothing"),
/// }
/// ```
pub async fn read_msg<R: AsyncReadExt + Unpin>(
    r: &mut R,
    max_len: usize,
) -> std::io::Result<Vec<u8>> {
    let mut len_buf = [0u8; 4];
    r.read_exact(&mut len_buf).await?;
    let len = u32::from_le_bytes(len_buf) as usize;
    if len > max_len {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("framed message of {len} B exceeds the {max_len} B cap"),
        ));
    }
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf).await?;
    Ok(buf)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Blocking helper so framing tests don't drag in a tokio runtime
    /// feature the engine otherwise has no use for. The futures here
    /// never yield on an in-memory buffer, so a trivial poll loop with a
    /// no-op waker drives them to completion.
    fn block_on<F: std::future::Future>(mut fut: F) -> F::Output {
        use std::task::{Context, Poll, RawWaker, RawWakerVTable, Waker};
        const VTABLE: RawWakerVTable = RawWakerVTable::new(
            |_| RawWaker::new(std::ptr::null(), &VTABLE),
            |_| {}, |_| {}, |_| {},
        );
        let waker = unsafe { Waker::from_raw(RawWaker::new(std::ptr::null(), &VTABLE)) };
        let mut cx = Context::from_waker(&waker);
        // Safety: `fut` is owned here and never moved after pinning.
        let mut fut = unsafe { std::pin::Pin::new_unchecked(&mut fut) };
        loop {
            if let Poll::Ready(v) = fut.as_mut().poll(&mut cx) {
                return v;
            }
        }
    }

    #[test]
    fn round_trips_a_message() {
        let mut buf: Vec<u8> = Vec::new();
        block_on(write_msg(&mut buf, b"hello")).unwrap();
        assert_eq!(buf.len(), 4 + 5, "u32 prefix then payload");
        let out = block_on(read_msg(&mut buf.as_slice(), 1024)).unwrap();
        assert_eq!(out, b"hello");
    }

    #[test]
    fn round_trips_back_to_back_messages() {
        // The point of framing: two writes into one stream come back as
        // two distinct messages, not one blob.
        let mut buf: Vec<u8> = Vec::new();
        block_on(write_msg(&mut buf, b"one")).unwrap();
        block_on(write_msg(&mut buf, b"two!!")).unwrap();
        let mut r = buf.as_slice();
        assert_eq!(block_on(read_msg(&mut r, 1024)).unwrap(), b"one");
        assert_eq!(block_on(read_msg(&mut r, 1024)).unwrap(), b"two!!");
    }

    #[test]
    fn empty_message_round_trips() {
        let mut buf: Vec<u8> = Vec::new();
        block_on(write_msg(&mut buf, b"")).unwrap();
        assert_eq!(block_on(read_msg(&mut buf.as_slice(), 16)).unwrap(), b"");
    }

    /// The reason this module isn't a verbatim lift: an unbounded length
    /// prefix is a one-packet remote OOM. The cap must be checked before
    /// the allocation, not after the read fails.
    #[test]
    fn oversized_length_is_refused_before_allocating() {
        // A hostile 4 GiB prefix with no body behind it.
        let mut bytes = u32::MAX.to_le_bytes().to_vec();
        bytes.extend_from_slice(b"nothing like that much data");
        let err = block_on(read_msg(&mut bytes.as_slice(), 4096)).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
    }

    #[test]
    fn a_message_exactly_at_the_cap_is_allowed() {
        let payload = vec![7u8; 64];
        let mut buf: Vec<u8> = Vec::new();
        block_on(write_msg(&mut buf, &payload)).unwrap();
        assert_eq!(block_on(read_msg(&mut buf.as_slice(), 64)).unwrap(), payload);
        // ...and one byte over is not.
        let mut buf: Vec<u8> = Vec::new();
        block_on(write_msg(&mut buf, &[7u8; 65])).unwrap();
        assert!(block_on(read_msg(&mut buf.as_slice(), 64)).is_err());
    }

    #[test]
    fn truncated_body_is_an_error_not_a_short_read() {
        // Prefix claims 10 B, stream holds 3. `read_exact` must fail
        // rather than hand back a partial message the caller would
        // try to decode.
        let mut bytes = 10u32.to_le_bytes().to_vec();
        bytes.extend_from_slice(b"abc");
        let err = block_on(read_msg(&mut bytes.as_slice(), 1024)).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::UnexpectedEof);
    }
}
