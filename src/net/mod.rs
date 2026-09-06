//! Netcode primitives for use inside a client. These do not talk to any
//! particular wire format — they're the pieces every fixed-tick netcode
//! setup ends up reinventing (interpolation clock, prediction ring, …).
//!
//! The two modules here answer opposite questions and are not meant to be
//! combined. [`interp`] smooths *state* streamed from an authoritative
//! server, hiding latency by rendering slightly in the past. [`lockstep`]
//! has no server and streams no state at all: two peers exchange only inputs
//! and run identical simulations. A lockstep peer has nothing to interpolate
//! — it holds the authoritative world itself.

pub mod interp;
pub mod lockstep;
