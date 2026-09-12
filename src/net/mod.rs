//! Netcode primitives. These do not talk to any
//! particular wire format — they're the pieces every fixed-tick netcode
//! setup ends up reinventing (interpolation clock, prediction ring, …).
//!
//! The model these serve is an authoritative server streaming state to
//! clients: [`interp`] hides the latency by rendering slightly in the past,
//! blending between the snapshots that have arrived rather than guessing at
//! the one that has not.
//!
//! A lockstep transport lived here briefly — two peers exchanging inputs and
//! running identical simulations, no server and no state on the wire. It was
//! removed rather than kept alongside: the two models answer opposite
//! questions, nothing in the tree used it, and a second netcode next to the
//! one actually in use is a standing invitation to wire a game to the wrong
//! half.
//!
//! The transport pieces sit alongside it: [`quic`] builds the endpoints,
//! [`framing`] restores message boundaries on the reliable-stream side,
//! and [`chunk`] fits an oversized packet into datagrams without ever
//! dropping a world-state item. All three are lifted from a shipped
//! authoritative-server game and generalised — none of them knows what a
//! snapshot, an entity or a tick is.
//!
//! The `replication` module is the exception, and is gated on its own
//! feature for exactly that reason: deciding what each client should see
//! means knowing about entities, components and ticks. Keeping it behind
//! `replication = ["net", "persist"]` leaves the primitives above usable by
//! a game that wants to write its own, which is why they were generalised
//! in the first place.
//!
//! (Named in prose rather than linked: this header is rendered on the
//! plain `net` axis too, where that module does not exist.)

#[cfg(feature = "replication")]
pub mod bitpack;
pub mod chunk;
pub mod framing;
/// Client-to-server input: held state, one-shot commands, and the rule for
/// reconciling frames that arrive out of order.
///
/// On the plain `net` axis deliberately. It needs no component registry,
/// no entities and no snapshot — only the observation that input has a
/// level half and an edge half that reconcile differently. A game doing
/// its own replication over `chunk` and `quic` wants this too.
pub mod input;
pub mod interp;
pub mod quic;
#[cfg(feature = "replication")]
pub mod replication;
#[cfg(feature = "replication")]
pub mod snapshot;
