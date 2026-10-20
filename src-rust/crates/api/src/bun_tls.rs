//! `wreq`/BoringSSL client whose ClientHello matches the official Claude Code
//! client (Bun). Target JA4 `t13d1714h1_5b57614c22b0_7baf387fc6ff`, verified on
//! tls.peet.ws. ALPN is http/1.1 only — the client speaks HTTP/1.1 to the API,
//! so there is no HTTP/2 fingerprint to match. See `findings/TLS-IMPERSONATION.md`.

use std::time::Duration;
use wreq::tls::{AlpnProtocol, TlsOptions, TlsVersion};

const VERIFY_CERTIFICATES: bool = true;
const VERIFY_HOSTNAMES: bool = true;
const SEND_SNI: bool = true;
// These compile-time assertions prevent a future TLS-fingerprint edit from
// silently disabling certificate, hostname, or SNI verification.
#[allow(clippy::assertions_on_constants)]
const _: () = {
    assert!(VERIFY_CERTIFICATES);
    assert!(VERIFY_HOSTNAMES);
    assert!(SEND_SNI);
};

/// Bun's 17 ciphers, in BoringSSL order.
const BUN_CIPHERS: &str = concat!(
    "TLS_AES_128_GCM_SHA256:TLS_AES_256_GCM_SHA384:TLS_CHACHA20_POLY1305_SHA256:",
    "TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256:TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256:",
    "TLS_ECDHE_ECDSA_WITH_AES_256_GCM_SHA384:TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384:",
    "TLS_ECDHE_ECDSA_WITH_CHACHA20_POLY1305_SHA256:TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305_SHA256:",
    "TLS_ECDHE_ECDSA_WITH_AES_128_CBC_SHA:TLS_ECDHE_RSA_WITH_AES_128_CBC_SHA:",
    "TLS_ECDHE_ECDSA_WITH_AES_256_CBC_SHA:TLS_ECDHE_RSA_WITH_AES_256_CBC_SHA:",
    "TLS_RSA_WITH_AES_128_GCM_SHA256:TLS_RSA_WITH_AES_256_GCM_SHA384:",
    "TLS_RSA_WITH_AES_128_CBC_SHA:TLS_RSA_WITH_AES_256_CBC_SHA"
);

/// `TlsOptions` reproducing Bun's ClientHello.
pub fn bun_tls_options() -> TlsOptions {
    TlsOptions::builder()
        .cipher_list(BUN_CIPHERS)
        .curves_list("X25519:P-256:P-384")
        .alpn_protocols([AlpnProtocol::HTTP1])
        .min_tls_version(TlsVersion::TLS_1_2)
        .max_tls_version(TlsVersion::TLS_1_3)
        .session_ticket(true)
        .enable_ocsp_stapling(true)
        .enable_signed_cert_timestamps(true)
        .enable_ech_grease(true)
        .grease_enabled(false)
        .pre_shared_key(false)
        .build()
}

/// `wreq` client with Bun's TLS fingerprint, for Anthropic calls.
pub fn build_anthropic_client(timeout: Duration) -> wreq::Result<wreq::Client> {
    wreq::Client::builder()
        .tls_options(bun_tls_options())
        .tls_cert_verification(VERIFY_CERTIFICATES)
        .tls_verify_hostname(VERIFY_HOSTNAMES)
        .tls_sni(SEND_SNI)
        .timeout(timeout)
        .build()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn anthropic_tls_policy_keeps_identity_verification_and_bounded_versions() {
        let options = bun_tls_options();
        assert_eq!(options.min_tls_version, Some(TlsVersion::TLS_1_2));
        assert_eq!(options.max_tls_version, Some(TlsVersion::TLS_1_3));
        assert_eq!(
            options.alpn_protocols.as_deref(),
            Some(&[AlpnProtocol::HTTP1][..])
        );
        assert_eq!(options.curves_list.as_deref(), Some("X25519:P-256:P-384"));
        assert_eq!(options.cipher_list.as_deref(), Some(BUN_CIPHERS));
        assert!(!options.pre_shared_key);
        assert!(options.enable_ocsp_stapling);
        assert!(options.enable_signed_cert_timestamps);
        assert!(options.enable_ech_grease);
        assert_eq!(options.grease_enabled, Some(false));
    }
}
