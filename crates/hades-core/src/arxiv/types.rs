//! ArXiv domain types — paper metadata and download results.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Metadata for a single arXiv paper, parsed from the Atom XML API response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArxivPaper {
    /// Normalized arXiv ID (no version suffix).
    pub arxiv_id: String,
    /// Paper title (whitespace-normalized).
    pub title: String,
    /// Abstract / summary text (whitespace-normalized).
    pub abstract_text: String,
    /// Author names.
    pub authors: Vec<String>,
    /// All subject categories (e.g. `["cs.AI", "cs.CL"]`).
    pub categories: Vec<String>,
    /// Primary subject category.
    pub primary_category: String,
    /// First publication date.
    pub published: DateTime<Utc>,
    /// Last update date.
    pub updated: DateTime<Utc>,
    /// DOI if assigned.
    pub doi: Option<String>,
    /// Journal reference if published in a journal.
    pub journal_ref: Option<String>,
    /// PDF download URL.
    pub pdf_url: String,
}

impl ArxivPaper {
    /// The e-print (LaTeX source) URL for this paper.
    pub fn eprint_url(&self) -> String {
        format!("https://arxiv.org/e-print/{}", self.arxiv_id)
    }
}

/// Result of downloading a paper's PDF (and optionally LaTeX source).
#[derive(Debug, Clone)]
pub struct DownloadResult {
    /// Whether the download succeeded.
    pub success: bool,
    /// The arXiv ID this result is for.
    pub arxiv_id: String,
    /// Local path to the downloaded PDF, if successful.
    pub pdf_path: Option<PathBuf>,
    /// Local path to the downloaded LaTeX tar.gz, if successful.
    pub latex_path: Option<PathBuf>,
    /// Paper metadata, if fetched successfully.
    pub metadata: Option<ArxivPaper>,
    /// Error description if the download failed.
    pub error_message: Option<String>,
    /// PDF file size in bytes.
    pub file_size_bytes: u64,
}

impl DownloadResult {
    /// Create a failed result.
    pub fn failed(arxiv_id: impl Into<String>, error: impl Into<String>) -> Self {
        Self {
            success: false,
            arxiv_id: arxiv_id.into(),
            pdf_path: None,
            latex_path: None,
            metadata: None,
            error_message: Some(error.into()),
            file_size_bytes: 0,
        }
    }
}
