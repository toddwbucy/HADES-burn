//! ArangoDB client for HADES.
//!
//! Provides async HTTP access to ArangoDB over Unix domain sockets
//! (primary) or TCP (fallback).

mod error;
mod pool;
mod transport;

pub use error::{ArangoError, ArangoErrorKind};
pub use pool::{ArangoPool, HealthStatus};
pub use transport::ArangoClient;
