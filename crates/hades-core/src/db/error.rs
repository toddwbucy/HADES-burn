//! ArangoDB error types.

use serde::Deserialize;

/// Error returned by ArangoDB operations.
#[derive(Debug, thiserror::Error)]
pub enum ArangoError {
    /// ArangoDB returned an HTTP error with a structured error body.
    #[error("ArangoDB {status}: {message} (error {error_num})")]
    Api {
        status: u16,
        error_num: u32,
        message: String,
        /// Full response body for debugging.
        body: serde_json::Value,
    },

    /// HTTP-level error without a parseable ArangoDB body.
    #[error("HTTP {status}: {message}")]
    Http { status: u16, message: String },

    /// Transport-level error (connection refused, timeout, etc).
    #[error("transport error: {0}")]
    Transport(#[from] hyper_util::client::legacy::Error),

    /// Error building or sending the HTTP request.
    #[error("request error: {0}")]
    Request(String),

    /// JSON serialization/deserialization error.
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
}

/// Classify errors for matching without inspecting fields.
impl ArangoError {
    pub fn kind(&self) -> ArangoErrorKind {
        match self {
            Self::Api { status, .. } | Self::Http { status, .. } => match *status {
                401 => ArangoErrorKind::Unauthorized,
                403 => ArangoErrorKind::Forbidden,
                404 => ArangoErrorKind::NotFound,
                409 => ArangoErrorKind::Conflict,
                503 => ArangoErrorKind::Unavailable,
                _ if *status >= 500 => ArangoErrorKind::Server,
                _ => ArangoErrorKind::Client,
            },
            Self::Transport(_) => ArangoErrorKind::Transport,
            Self::Request(_) => ArangoErrorKind::Transport,
            Self::Json(_) => ArangoErrorKind::Client,
        }
    }

    pub fn is_not_found(&self) -> bool {
        self.kind() == ArangoErrorKind::NotFound
    }
}

/// Broad error categories for pattern matching.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArangoErrorKind {
    NotFound,
    Unauthorized,
    Forbidden,
    Conflict,
    Unavailable,
    Server,
    Client,
    Transport,
}

/// ArangoDB's standard JSON error response body.
#[derive(Debug, Deserialize)]
pub(crate) struct ArangoErrorBody {
    #[allow(dead_code)]
    pub error: bool,
    #[serde(rename = "errorNum")]
    pub error_num: u32,
    #[serde(rename = "errorMessage")]
    pub error_message: String,
    #[allow(dead_code)]
    pub code: u16,
}
