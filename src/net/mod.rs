//! Netcode primitives for use inside a client. These do not talk to any
//! particular wire format — they're the pieces every fixed-tick netcode
//! setup ends up reinventing (interpolation clock, prediction ring, …).

pub mod interp;
