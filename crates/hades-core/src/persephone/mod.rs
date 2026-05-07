//! Persephone service clients.
//!
//! Typed clients for the Persephone embedding, extraction, and training
//! services. Transports differ by service:
//!
//! - **embedding** — OpenAI-compatible HTTP at `http://localhost:8087/v1`
//!   (PE-API v1, see `docs/persephone-embedding-api.md`). Do **not**
//!   reintroduce the pre-PR-#70 gRPC Unix-socket pattern at
//!   `/run/hades/embedder.sock`; that path was removed deliberately.
//! - **extraction** — gRPC over a Unix domain socket or TCP.
//! - **training** — gRPC over a Unix domain socket or TCP.

pub mod embedding;
pub mod extraction;
pub mod training;
