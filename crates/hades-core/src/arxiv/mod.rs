//! ArXiv API client — search, download, and metadata retrieval.
//!
//! Provides a typed Rust client for the arXiv REST API with rate limiting,
//! retry with exponential backoff, Atom XML response parsing, and streaming
//! PDF/LaTeX downloads.

mod client;
mod types;

pub use client::ArxivClient;
pub use types::{ArxivPaper, DownloadResult};

use regex::Regex;
use std::sync::LazyLock;

/// Regex for new-format arXiv IDs: `YYMM.NNNNN(vN)?`
static NEW_ARXIV_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^\d{4}\.\d{4,5}(v\d+)?$").expect("invalid regex")
});

/// Regex for old-format arXiv IDs: `subject-class(.XX)?/YYMMnnn(vN)?`
static OLD_ARXIV_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^[a-z-]+(\.[A-Z]{2})?/\d{7}(v\d+)?$").expect("invalid regex")
});

/// Version suffix regex.
static VERSION_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"v\d+$").expect("invalid regex")
});

/// Check if a string looks like an arXiv ID (new or old format).
pub fn is_arxiv_id(s: &str) -> bool {
    NEW_ARXIV_RE.is_match(s) || OLD_ARXIV_RE.is_match(s)
}

/// Normalize an arXiv ID by stripping the version suffix.
///
/// # Examples
/// ```
/// # use hades_core::arxiv::normalize_arxiv_id;
/// assert_eq!(normalize_arxiv_id("2501.12345v2"), "2501.12345");
/// assert_eq!(normalize_arxiv_id("hep-th/9901001v1"), "hep-th/9901001");
/// assert_eq!(normalize_arxiv_id("2501.12345"), "2501.12345");
/// ```
pub fn normalize_arxiv_id(id: &str) -> String {
    VERSION_RE.replace(id, "").into_owned()
}

/// Extract the year-month prefix from a new-format arXiv ID.
///
/// Returns `None` for old-format IDs.
///
/// # Examples
/// ```
/// # use hades_core::arxiv::extract_year_month;
/// assert_eq!(extract_year_month("2501.12345"), Some(("25", "01")));
/// ```
pub fn extract_year_month(id: &str) -> Option<(&str, &str)> {
    let normalized = if VERSION_RE.is_match(id) {
        // Can't return references to owned string, so check the original
        id.split('v').next()?
    } else {
        id
    };
    let dot = normalized.find('.')?;
    if dot == 4 {
        Some((&normalized[..2], &normalized[2..4]))
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_arxiv_id_new_format() {
        assert!(is_arxiv_id("2501.12345"));
        assert!(is_arxiv_id("2501.12345v1"));
        assert!(is_arxiv_id("2501.12345v12"));
        assert!(is_arxiv_id("1234.5678"));
    }

    #[test]
    fn test_is_arxiv_id_old_format() {
        assert!(is_arxiv_id("hep-th/9901001"));
        assert!(is_arxiv_id("hep-th/9901001v1"));
        assert!(is_arxiv_id("cs.AI/0601001"));
    }

    #[test]
    fn test_is_arxiv_id_invalid() {
        assert!(!is_arxiv_id("not-an-id"));
        assert!(!is_arxiv_id("/tmp/file.pdf"));
        assert!(!is_arxiv_id("2501"));
        assert!(!is_arxiv_id(""));
    }

    #[test]
    fn test_normalize_arxiv_id() {
        assert_eq!(normalize_arxiv_id("2501.12345v2"), "2501.12345");
        assert_eq!(normalize_arxiv_id("2501.12345"), "2501.12345");
        assert_eq!(normalize_arxiv_id("hep-th/9901001v1"), "hep-th/9901001");
    }

    #[test]
    fn test_extract_year_month() {
        assert_eq!(extract_year_month("2501.12345"), Some(("25", "01")));
        assert_eq!(extract_year_month("2308.99999v2"), Some(("23", "08")));
        assert_eq!(extract_year_month("hep-th/9901001"), None);
    }
}
