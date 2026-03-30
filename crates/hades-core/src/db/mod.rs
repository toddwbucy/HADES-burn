//! ArangoDB client for HADES.
//!
//! Provides async HTTP access to ArangoDB over Unix domain sockets
//! (primary) or TCP (fallback).

pub mod cache;
pub mod collections;
pub mod crud;
mod error;
pub mod index;
pub mod keys;
mod pool;
pub mod query;
mod transport;
pub mod vector;

pub use error::{ArangoError, ArangoErrorKind};
pub use pool::{ArangoPool, HealthStatus};
pub use transport::ArangoClient;
