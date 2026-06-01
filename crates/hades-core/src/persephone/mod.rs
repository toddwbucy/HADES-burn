//! Persephone service clients.
//!
//! Typed clients for the (HADES-owned) embedding and extraction compute
//! services, which carry the legacy `persephone.*` provider-protocol brand.
//! Transports differ by service:
//!
//! - **embedding** — OpenAI-compatible HTTP at `http://localhost:8087/v1`
//!   (PE-API v1, see `docs/persephone-embedding-api.md`). Do **not**
//!   reintroduce the pre-PR-#70 gRPC Unix-socket pattern at
//!   `/run/hades/embedder.sock`; that path was removed deliberately.
//! - **extraction** — gRPC over a Unix domain socket or TCP.
//!
//! The training client moved out of this module to [`crate::training`] when it
//! was decoupled from the `persephone.*` brand (issue #106).

pub mod embedding;
pub mod extraction;
