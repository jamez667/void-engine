//! Netcode primitives for use inside a client. These do not talk to any
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

pub mod interp;
