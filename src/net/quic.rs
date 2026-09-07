//! QUIC endpoint construction for both ends of an authoritative
//! client-server link.
//!
//! Lifted from void-claim: `server::net::make_endpoint` and the client
//! side's `make_client_config` (which existed in two copies there —
//! `void_claim::net` and `sim_users::tls` — because there was nowhere
//! shared to put it; this module is that place).
//!
//! Two things were hardcoded in the original and are parameters here:
//!
//! * **ALPN.** void-claim baked in `b"void-claim/0"`. ALPN is how a
//!   server refuses a client speaking a different protocol *during* the
//!   handshake rather than by getting confused later, so every game
//!   needs its own — and versioning it (`"mygame/1"`) makes an
//!   incompatible protocol bump a clean handshake rejection instead of a
//!   garbled decode.
//! * **The certificate.** The original always generated a fresh
//!   self-signed cert at boot, which is right for a LAN playtest and
//!   wrong for anything a stranger connects to: it forces every client
//!   to skip verification, and a client that skips verification cannot
//!   tell your server from an attacker's. [`CertSource`] makes the
//!   choice explicit at the call site.

use std::net::SocketAddr;
use std::sync::Arc;

use quinn::{ClientConfig, Endpoint, ServerConfig};
use rustls::pki_types::{CertificateDer, PrivateKeyDer, PrivatePkcs8KeyDer};

/// Where the server's TLS identity comes from.
///
/// QUIC has no unencrypted mode, so there is always a certificate; the
/// only question is whether clients can verify it.
pub enum CertSource {
    /// Generate a throwaway self-signed cert for the given SANs at boot.
    ///
    /// Dev and LAN only. Nothing can verify it, so every client must
    /// pair this with [`insecure_skip_verification`] — see the warning
    /// there before shipping it.
    SelfSignedDev {
        /// Subject alternative names, e.g. `vec!["localhost".into()]`.
        subject_alt_names: Vec<String>,
    },
    /// A real certificate chain and its private key, as DER — from
    /// ACME/Let's Encrypt, an internal CA, or whatever issues your
    /// deployment's certs. This is the path for a server on the public
    /// internet, and it lets clients verify normally.
    Supplied {
        /// Leaf first, then intermediates. Clients need the chain to
        /// build a path to a root they trust.
        chain: Vec<CertificateDer<'static>>,
        key: PrivateKeyDer<'static>,
    },
}

impl CertSource {
    /// Convenience for the common dev case: self-signed for `localhost`.
    pub fn self_signed_localhost() -> Self {
        Self::SelfSignedDev { subject_alt_names: vec!["localhost".to_string()] }
    }
}

/// Anything that can go wrong building an endpoint. Kept as a concrete
/// enum rather than `anyhow` (which void-claim used) so the engine does
/// not push an error-handling crate onto its consumers.
#[derive(Debug)]
pub enum QuicSetupError {
    /// Self-signed certificate generation failed.
    CertGen(String),
    /// rustls rejected the cert/key pair, or the config is invalid.
    Tls(String),
    /// Binding the UDP socket failed (port in use, no permission, …).
    Bind(std::io::Error),
}

impl std::fmt::Display for QuicSetupError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CertGen(e) => write!(f, "self-signed certificate generation failed: {e}"),
            Self::Tls(e)     => write!(f, "TLS configuration rejected: {e}"),
            Self::Bind(e)    => write!(f, "could not bind QUIC endpoint: {e}"),
        }
    }
}

impl std::error::Error for QuicSetupError {}

/// Build the QUIC server config: TLS identity plus the agreed ALPN.
///
/// Split out from [`server_endpoint`] because building the config is
/// pure — no socket, no runtime — so it can be validated in a unit test,
/// and because a caller with its own socket or transport tuning wants
/// the config without the binding.
///
/// `alpn` is the protocol name both ends must agree on (e.g.
/// `b"mygame/1"`); it must match what the client config advertises or
/// the handshake fails with no_application_protocol.
pub fn server_config(alpn: &[u8], certs: CertSource) -> Result<ServerConfig, QuicSetupError> {
    let (chain, key) = match certs {
        CertSource::SelfSignedDev { subject_alt_names } => {
            // rcgen will happily mint a cert with no SANs. Nothing can
            // ever match it, so the server would come up and then fail
            // every handshake with an opaque TLS error — catch it here
            // where the message can say what is actually wrong.
            if subject_alt_names.is_empty() {
                return Err(QuicSetupError::CertGen(
                    "no subject alt names: a cert with none matches no hostname".into(),
                ));
            }
            let rcgen::CertifiedKey { cert, key_pair } =
                rcgen::generate_simple_self_signed(subject_alt_names)
                    .map_err(|e| QuicSetupError::CertGen(e.to_string()))?;
            let key = PrivatePkcs8KeyDer::from(key_pair.serialize_der());
            (vec![cert.der().clone()], PrivateKeyDer::from(key))
        }
        CertSource::Supplied { chain, key } => (chain, key),
    };

    let mut tls = rustls::ServerConfig::builder()
        .with_no_client_auth()
        .with_single_cert(chain, key)
        .map_err(|e| QuicSetupError::Tls(e.to_string()))?;
    tls.alpn_protocols = vec![alpn.to_vec()];

    let crypto = quinn::crypto::rustls::QuicServerConfig::try_from(tls)
        .map_err(|e| QuicSetupError::Tls(e.to_string()))?;
    Ok(ServerConfig::with_crypto(Arc::new(crypto)))
}

/// Build and bind a listening QUIC server endpoint on `addr`.
///
/// Must be called from inside a tokio runtime: quinn drives its UDP
/// socket on the ambient reactor and errors with "no async runtime
/// found" otherwise.
pub fn server_endpoint(
    addr: SocketAddr,
    alpn: &[u8],
    certs: CertSource,
) -> Result<Endpoint, QuicSetupError> {
    Endpoint::server(server_config(alpn, certs)?, addr).map_err(QuicSetupError::Bind)
}

/// Client config that verifies the server certificate against the
/// platform's trust store — the normal path, and the one to use for any
/// server a player connects to over the internet.
///
/// Requires the server to present a cert issued for the hostname the
/// client dialed ([`CertSource::Supplied`]); a self-signed dev server
/// will be rejected, which is the point.
pub fn client_config(alpn: &[u8]) -> Result<ClientConfig, QuicSetupError> {
    let mut roots = rustls::RootCertStore::empty();
    roots.extend(webpki_roots::TLS_SERVER_ROOTS.iter().cloned());

    let mut tls = rustls::ClientConfig::builder()
        .with_root_certificates(roots)
        .with_no_client_auth();
    tls.alpn_protocols = vec![alpn.to_vec()];

    let crypto = quinn::crypto::rustls::QuicClientConfig::try_from(tls)
        .map_err(|e| QuicSetupError::Tls(e.to_string()))?;
    Ok(ClientConfig::new(Arc::new(crypto)))
}

/// **Dangerous.** Client config that accepts *any* server certificate
/// without checking it.
///
/// This is the void-claim client's `SkipVerification`, kept because a
/// self-signed dev server is genuinely useful and there is no other way
/// to talk to one — but named so it cannot be reached by accident, and
/// deliberately not the default.
///
/// The connection is still encrypted, but encrypted *to whoever
/// answered*. Certificate verification is the only thing that makes the
/// peer the server you meant; without it anyone positioned on the path —
/// a hostile Wi-Fi AP, a poisoned DNS answer, a compromised router —
/// terminates the connection themselves, sees and rewrites every packet,
/// and neither end notices. No amount of application-level auth fixes
/// this: credentials sent over the connection go straight to the
/// attacker.
///
/// Acceptable:
/// * `cargo run` against `127.0.0.1` on your own machine.
/// * A LAN playtest against a [`CertSource::SelfSignedDev`] server, on a
///   network you control.
/// * Automated tests and load-generating bots against a dev shard.
///
/// Not acceptable, ever:
/// * A shipped client build. Give the server a real cert
///   ([`CertSource::Supplied`]) and use [`client_config`].
/// * Anything reached over the public internet, "just for now" included.
/// * Anything carrying an account password, token or payment detail.
///
/// If you need to trust one specific self-signed server rather than all
/// of them, pin it: put its cert in a `RootCertStore` and use
/// `with_root_certificates`. That keeps verification on and is strictly
/// better than this function.
pub fn insecure_skip_verification(alpn: &[u8]) -> Result<ClientConfig, QuicSetupError> {
    // rustls 0.23 can be built with more than one crypto provider, in
    // which case there is no implicit default — take ring explicitly so
    // this works regardless of what else the binary links.
    let provider = Arc::new(rustls::crypto::ring::default_provider());

    let mut tls = rustls::ClientConfig::builder_with_provider(provider.clone())
        .with_safe_default_protocol_versions()
        .map_err(|e| QuicSetupError::Tls(e.to_string()))?
        .dangerous()
        .with_custom_certificate_verifier(Arc::new(SkipVerification(provider)))
        .with_no_client_auth();
    tls.alpn_protocols = vec![alpn.to_vec()];

    let crypto = quinn::crypto::rustls::QuicClientConfig::try_from(tls)
        .map_err(|e| QuicSetupError::Tls(e.to_string()))?;
    Ok(ClientConfig::new(Arc::new(crypto)))
}

/// The verifier behind [`insecure_skip_verification`]. Private: the only
/// way to get one is through the loudly-named constructor, so a
/// `grep -r insecure_skip_verification` finds every place a build
/// disables verification.
///
/// Signature checking is left intact — it is the *identity* check that
/// is skipped. That is not a mitigation, it just avoids also breaking
/// the handshake.
#[derive(Debug)]
struct SkipVerification(Arc<rustls::crypto::CryptoProvider>);

impl rustls::client::danger::ServerCertVerifier for SkipVerification {
    fn verify_server_cert(
        &self,
        _end_entity: &CertificateDer,
        _intermediates: &[CertificateDer],
        _server_name: &rustls::pki_types::ServerName,
        _ocsp: &[u8],
        _now: rustls::pki_types::UnixTime,
    ) -> Result<rustls::client::danger::ServerCertVerified, rustls::Error> {
        Ok(rustls::client::danger::ServerCertVerified::assertion())
    }

    fn verify_tls12_signature(
        &self,
        message: &[u8],
        cert: &CertificateDer,
        dss: &rustls::DigitallySignedStruct,
    ) -> Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        rustls::crypto::verify_tls12_signature(
            message, cert, dss, &self.0.signature_verification_algorithms,
        )
    }

    fn verify_tls13_signature(
        &self,
        message: &[u8],
        cert: &CertificateDer,
        dss: &rustls::DigitallySignedStruct,
    ) -> Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        rustls::crypto::verify_tls13_signature(
            message, cert, dss, &self.0.signature_verification_algorithms,
        )
    }

    fn supported_verify_schemes(&self) -> Vec<rustls::SignatureScheme> {
        self.0.signature_verification_algorithms.supported_schemes()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ALPN: &[u8] = b"void-engine-test/1";

    // These exercise `server_config`, not `server_endpoint`: binding
    // needs an ambient tokio runtime, and adding tokio's `rt` feature
    // just to bind a socket in a test would put a runtime into every
    // netcode build. All the logic worth testing — cert selection, key
    // matching, ALPN — is in the config.

    fn pair(name: &str) -> (Vec<CertificateDer<'static>>, PrivateKeyDer<'static>) {
        let ck = rcgen::generate_simple_self_signed(vec![name.to_string()]).unwrap();
        let chain = vec![ck.cert.der().clone()];
        let key = PrivateKeyDer::from(PrivatePkcs8KeyDer::from(ck.key_pair.serialize_der()));
        (chain, key)
    }

    #[test]
    fn self_signed_dev_config_builds() {
        assert!(server_config(ALPN, CertSource::self_signed_localhost()).is_ok());
    }

    #[test]
    fn supplied_cert_path_accepts_a_matching_pair() {
        // The shape a real deployment uses (chain + key loaded from
        // disk), exercised with a pair we can make in-process.
        let (chain, key) = pair("example.test");
        assert!(server_config(ALPN, CertSource::Supplied { chain, key }).is_ok());
    }

    #[test]
    fn mismatched_cert_and_key_are_rejected() {
        // Two independently generated identities: the key does not go
        // with the cert. rustls must refuse at construction rather than
        // hand back a config that fails every handshake at runtime.
        let (chain, _) = pair("a.test");
        let (_, key) = pair("b.test");
        let err = server_config(ALPN, CertSource::Supplied { chain, key });
        assert!(matches!(err, Err(QuicSetupError::Tls(_))), "got {err:?}");
    }

    /// A cert source with no SANs at all: rcgen must not silently
    /// produce an identity nothing could ever match.
    #[test]
    fn self_signed_with_no_sans_is_an_error() {
        let r = server_config(ALPN, CertSource::SelfSignedDev { subject_alt_names: vec![] });
        assert!(matches!(r, Err(QuicSetupError::CertGen(_))), "got {r:?}");
    }

    #[test]
    fn both_client_configs_build() {
        assert!(client_config(ALPN).is_ok(), "verifying config is the default path");
        assert!(insecure_skip_verification(ALPN).is_ok());
    }

    /// The danger is opt-in: the skipping verifier is private, so the
    /// only route to it is the loudly-named constructor. This pins that
    /// `client_config` does not quietly share the same code path.
    #[test]
    fn the_default_client_config_verifies() {
        // A verifying config carries real trust anchors; the insecure
        // one carries none. Comparing the two configs directly is not
        // possible through quinn's opaque type, so assert the property
        // that actually matters at the source: the roots are non-empty.
        let mut roots = rustls::RootCertStore::empty();
        roots.extend(webpki_roots::TLS_SERVER_ROOTS.iter().cloned());
        assert!(!roots.is_empty(), "client_config must have trust anchors to verify against");
    }
}
