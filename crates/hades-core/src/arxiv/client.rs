//! ArXiv REST API client with rate limiting, retry, and Atom XML parsing.

use super::{is_arxiv_id, normalize_arxiv_id, extract_year_month};
use super::types::{ArxivError, ArxivPaper, DownloadResult};

use chrono::{DateTime, Utc};
use regex::Regex;
use reqwest::Client;
use std::path::Path;
use std::sync::LazyLock;
use std::time::{Duration, Instant};
use tokio::sync::Mutex;
use tokio::time::sleep;
use tracing::{debug, info, warn, instrument};

/// Atom XML namespace.
const ATOM_NS: &str = "http://www.w3.org/2005/Atom";
/// ArXiv-specific XML namespace.
const ARXIV_NS: &str = "http://arxiv.org/schemas/atom";

/// ArXiv API base URL.
const API_BASE: &str = "https://export.arxiv.org/api/query";
/// ArXiv PDF base URL.
const PDF_BASE: &str = "https://arxiv.org/pdf";

/// Whitespace normalizer.
static WS_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"\s+").expect("invalid regex")
});

/// Typed client for the arXiv REST API.
///
/// Provides search, metadata retrieval, and paper downloads with built-in
/// rate limiting and exponential-backoff retries.
pub struct ArxivClient {
    http: Client,
    /// Minimum delay between API requests.
    rate_limit_delay: Duration,
    /// Maximum retry attempts for failed requests.
    max_retries: u32,
    /// Tracks the last request time for rate limiting.
    last_request: Mutex<Option<Instant>>,
}

impl ArxivClient {
    /// Create a new client with default settings (3s rate limit, 3 retries, 30s timeout).
    pub fn new() -> Result<Self, ArxivError> {
        Self::with_config(3.0, 3, 30)
    }

    /// Create a client for sync operations (0.5s rate limit).
    pub fn for_sync() -> Result<Self, ArxivError> {
        Self::with_config(0.5, 3, 30)
    }

    /// Create a client with custom configuration.
    ///
    /// # Arguments
    /// * `rate_limit_secs` — minimum seconds between API calls
    /// * `max_retries` — number of retry attempts on failure
    /// * `timeout_secs` — HTTP request timeout in seconds
    pub fn with_config(
        rate_limit_secs: f64,
        max_retries: u32,
        timeout_secs: u64,
    ) -> Result<Self, ArxivError> {
        let http = Client::builder()
            .timeout(Duration::from_secs(timeout_secs))
            .user_agent("HADES-Burn/0.1 (Rust; https://github.com/toddwbucy/HADES-Burn)")
            .build()
            .map_err(ArxivError::ClientBuild)?;

        Ok(Self {
            http,
            rate_limit_delay: Duration::from_secs_f64(rate_limit_secs),
            max_retries,
            last_request: Mutex::new(None),
        })
    }

    /// Fetch metadata for a single paper by arXiv ID.
    ///
    /// Returns `None` if the paper is not found (no entries in the response).
    #[instrument(skip(self))]
    pub async fn get_paper_metadata(
        &self,
        arxiv_id: &str,
    ) -> Result<Option<ArxivPaper>, ArxivError> {
        if !is_arxiv_id(arxiv_id) {
            return Err(ArxivError::InvalidId(arxiv_id.to_string()));
        }
        let normalized = normalize_arxiv_id(arxiv_id);
        let url = format!("{API_BASE}?id_list={normalized}&max_results=1");
        let body = self.request_with_retry(&url).await?;
        let papers = parse_atom_response(&body)?;
        Ok(papers.into_iter().next())
    }

    /// Search arXiv papers by query string.
    ///
    /// Uses the arXiv query syntax (e.g. `cat:cs.AI AND ti:transformer`).
    /// Returns up to `max_results` papers starting from `start`.
    #[instrument(skip(self))]
    pub async fn search(
        &self,
        query: &str,
        start: u32,
        max_results: u32,
    ) -> Result<Vec<ArxivPaper>, ArxivError> {
        let url = format!(
            "{API_BASE}?search_query={}&start={start}&max_results={max_results}",
            urlencoding::encode(query),
        );
        let body = self.request_with_retry(&url).await?;
        parse_atom_response(&body)
    }

    /// Batch-fetch metadata for multiple arXiv IDs.
    ///
    /// Sends IDs in groups of 10 (arXiv API soft limit per request).
    #[instrument(skip(self, arxiv_ids), fields(count = arxiv_ids.len()))]
    pub async fn batch_get_metadata(
        &self,
        arxiv_ids: &[&str],
    ) -> Result<Vec<Option<ArxivPaper>>, ArxivError> {
        let mut results = Vec::with_capacity(arxiv_ids.len());
        for chunk in arxiv_ids.chunks(10) {
            let id_list: Vec<String> = chunk
                .iter()
                .filter(|id| is_arxiv_id(id))
                .map(|id| normalize_arxiv_id(id))
                .collect();

            if id_list.is_empty() {
                results.extend(chunk.iter().map(|_| None));
                continue;
            }

            let url = format!(
                "{API_BASE}?id_list={}&max_results={}",
                id_list.join(","),
                id_list.len(),
            );
            let body = self.request_with_retry(&url).await?;
            let papers = parse_atom_response(&body)?;

            // Match results back to requested IDs.
            for id in chunk {
                let normalized = normalize_arxiv_id(id);
                let found = papers.iter().find(|p| p.arxiv_id == normalized).cloned();
                results.push(found);
            }
        }
        Ok(results)
    }

    /// Download a paper's PDF (and optionally LaTeX source) to local directories.
    ///
    /// Files are organized into `YYMM/` subdirectories by publication date.
    /// If `latex_dir` is `Some`, attempts to download the e-print tar.gz (404 is not an error).
    #[instrument(skip(self))]
    pub async fn download_paper(
        &self,
        arxiv_id: &str,
        pdf_dir: &Path,
        latex_dir: Option<&Path>,
        force: bool,
    ) -> DownloadResult {
        if !is_arxiv_id(arxiv_id) {
            return DownloadResult::failed(arxiv_id, format!("Invalid arXiv ID format: {arxiv_id}"));
        }
        let normalized = normalize_arxiv_id(arxiv_id);

        // Fetch metadata first.
        let metadata = match self.get_paper_metadata(arxiv_id).await {
            Ok(Some(m)) => m,
            Ok(None) => {
                return DownloadResult::failed(&normalized, "Paper not found on arXiv");
            }
            Err(e) => {
                return DownloadResult::failed(
                    &normalized,
                    format!("Failed to fetch paper metadata: {e}"),
                );
            }
        };

        // Determine subdirectory from arXiv ID.
        let subdir = extract_year_month(&normalized)
            .map(|(y, m)| format!("{y}{m}"))
            .unwrap_or_else(|| "other".to_string());

        // -- PDF ---------------------------------------------------------------
        let pdf_subdir = pdf_dir.join(&subdir);
        let pdf_path = pdf_subdir.join(format!("{}.pdf", normalized.replace('/', "_")));

        let file_size_bytes;
        if pdf_path.exists() && !force {
            info!(path = %pdf_path.display(), "PDF already exists, skipping (use force to re-download)");
            file_size_bytes = tokio::fs::metadata(&pdf_path)
                .await
                .map(|m| m.len())
                .unwrap_or(0);
        } else {
            if let Err(e) = tokio::fs::create_dir_all(&pdf_subdir).await {
                return DownloadResult::failed(
                    &normalized,
                    format!("failed to create directory {}: {e}", pdf_subdir.display()),
                );
            }
            let pdf_url = format!("{PDF_BASE}/{normalized}.pdf");
            match self.download_file(&pdf_url, &pdf_path).await {
                Ok(size) => {
                    info!(path = %pdf_path.display(), bytes = size, "PDF downloaded");
                    file_size_bytes = size;
                }
                Err(e) => {
                    return DownloadResult::failed(
                        &normalized,
                        format!("PDF download failed: {e}"),
                    );
                }
            }
        }

        // -- LaTeX (optional) --------------------------------------------------
        let latex_path = if let Some(ldir) = latex_dir {
            let latex_subdir = ldir.join(&subdir);
            let lpath = latex_subdir.join(format!(
                "{}.tar.gz",
                normalized.replace('/', "_")
            ));
            if lpath.exists() && !force {
                Some(lpath)
            } else if let Err(e) = tokio::fs::create_dir_all(&latex_subdir).await {
                warn!(
                    error = %e,
                    dir = %latex_subdir.display(),
                    "failed to create LaTeX directory, skipping LaTeX download"
                );
                None
            } else {
                let eprint_url = metadata.eprint_url();
                match self.download_file(&eprint_url, &lpath).await {
                    Ok(_) => Some(lpath),
                    Err(e) => {
                        debug!(error = %e, "LaTeX source not available (this is normal)");
                        None
                    }
                }
            }
        } else {
            None
        };

        DownloadResult {
            success: true,
            arxiv_id: normalized,
            pdf_path: Some(pdf_path),
            latex_path,
            metadata: Some(metadata),
            error_message: None,
            file_size_bytes,
        }
    }

    // ── Internal helpers ────────────────────────────────────────────────

    /// Enforce rate limiting — sleep if needed to maintain the minimum delay.
    async fn enforce_rate_limit(&self) {
        let mut last = self.last_request.lock().await;
        if let Some(prev) = *last {
            let elapsed = prev.elapsed();
            if elapsed < self.rate_limit_delay {
                sleep(self.rate_limit_delay - elapsed).await;
            }
        }
        *last = Some(Instant::now());
    }

    /// Make a GET request with rate limiting and exponential backoff retries.
    async fn request_with_retry(&self, url: &str) -> Result<String, ArxivError> {
        for attempt in 0..=self.max_retries {
            self.enforce_rate_limit().await;
            match self.http.get(url).send().await {
                Ok(resp) => {
                    let status = resp.status();
                    if status.is_success() {
                        return resp.text().await.map_err(ArxivError::Request);
                    }
                    warn!(status = %status, attempt, url, "HTTP error");
                    if attempt == self.max_retries {
                        return Err(ArxivError::HttpStatus {
                            status: status.as_u16(),
                            url: url.to_string(),
                        });
                    }
                }
                Err(e) => {
                    warn!(error = %e, attempt, url, "request failed");
                    if attempt == self.max_retries {
                        return Err(ArxivError::Request(e));
                    }
                }
            }
            // Exponential backoff: 2^attempt * base delay.
            let backoff = self.rate_limit_delay * 2u32.pow(attempt);
            debug!(backoff_ms = backoff.as_millis(), attempt, "retrying");
            sleep(backoff).await;
        }
        unreachable!()
    }

    /// Download a file by streaming to a temp file, then atomically renaming on success.
    ///
    /// If the download is interrupted or content-length doesn't match, the temp
    /// file is removed and no partial file is left at `dest`.
    async fn download_file(&self, url: &str, dest: &Path) -> Result<u64, ArxivError> {
        use futures::TryStreamExt;
        use tokio::io::AsyncWriteExt;
        use tokio_util::io::StreamReader;

        self.enforce_rate_limit().await;
        let resp = self.http.get(url).send().await?;

        let status = resp.status();
        if !status.is_success() {
            return Err(ArxivError::HttpStatus {
                status: status.as_u16(),
                url: url.to_string(),
            });
        }

        let expected_len = resp.content_length();

        // Write to a temp file first, rename atomically on success.
        let tmp_path = dest.with_extension("part");

        let stream = resp
            .bytes_stream()
            .map_err(std::io::Error::other);
        let mut reader = StreamReader::new(stream);

        let mut file = tokio::fs::File::create(&tmp_path).await.map_err(|e| {
            ArxivError::Io(std::io::Error::new(
                e.kind(),
                format!("failed to create {}: {e}", tmp_path.display()),
            ))
        })?;

        let result = async {
            let written = tokio::io::copy(&mut reader, &mut file).await.map_err(|e| {
                ArxivError::Io(std::io::Error::new(
                    e.kind(),
                    format!("failed to write {}: {e}", tmp_path.display()),
                ))
            })?;
            file.flush().await?;
            file.sync_all().await?;

            // Fail if content-length was advertised and doesn't match.
            if let Some(expected) = expected_len
                && written != expected
            {
                return Err(ArxivError::Io(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    format!(
                        "content-length mismatch for {}: expected {expected}, got {written}",
                        dest.display()
                    ),
                )));
            }

            Ok(written)
        }
        .await;

        match result {
            Ok(written) => {
                // Atomic rename into place.
                tokio::fs::rename(&tmp_path, dest).await.map_err(|e| {
                    ArxivError::Io(std::io::Error::new(
                        e.kind(),
                        format!(
                            "failed to rename {} -> {}: {e}",
                            tmp_path.display(),
                            dest.display()
                        ),
                    ))
                })?;
                Ok(written)
            }
            Err(e) => {
                // Clean up partial temp file.
                let _ = tokio::fs::remove_file(&tmp_path).await;
                Err(e)
            }
        }
    }
}

// ── Atom XML parsing ────────────────────────────────────────────────────

/// Parse an arXiv Atom XML response into a list of papers.
fn parse_atom_response(xml: &str) -> Result<Vec<ArxivPaper>, ArxivError> {
    let doc = roxmltree::Document::parse(xml)
        .map_err(|e| ArxivError::XmlParse(e.to_string()))?;
    let root = doc.root_element();

    let mut papers = Vec::new();
    for entry in root.children().filter(|n| n.has_tag_name((ATOM_NS, "entry"))) {
        // Skip error entries (arXiv returns these for invalid IDs).
        if let Some(id_node) = find_child(&entry, ATOM_NS, "id") {
            let id_text = id_node.text().unwrap_or_default();
            if id_text.contains("api/errors") {
                continue;
            }
        }

        if let Some(paper) = parse_entry(&entry) {
            papers.push(paper);
        }
    }
    Ok(papers)
}

/// Parse a single `<entry>` element into an `ArxivPaper`.
fn parse_entry(entry: &roxmltree::Node) -> Option<ArxivPaper> {
    // Extract arXiv ID from the <id> URL: https://arxiv.org/abs/XXXX.XXXXX(vN)
    // Old-format IDs contain a slash (e.g. hep-th/9901001), so split on "/abs/" not last "/".
    let id_text = find_child_text(entry, ATOM_NS, "id")?;
    let arxiv_id = id_text
        .split("/abs/")
        .nth(1)
        .map(normalize_arxiv_id)
        .unwrap_or_default();

    if arxiv_id.is_empty() {
        return None;
    }

    let title = normalize_ws(&find_child_text(entry, ATOM_NS, "title").unwrap_or_default());
    let abstract_text =
        normalize_ws(&find_child_text(entry, ATOM_NS, "summary").unwrap_or_default());

    // Authors: <author><name>...</name></author>
    let authors: Vec<String> = entry
        .children()
        .filter(|n| n.has_tag_name((ATOM_NS, "author")))
        .filter_map(|a| find_child_text(&a, ATOM_NS, "name"))
        .collect();

    // Primary category from arxiv namespace.
    let primary_category = entry
        .children()
        .find(|n| n.has_tag_name((ARXIV_NS, "primary_category")))
        .and_then(|n| n.attribute("term"))
        .unwrap_or_default()
        .to_string();

    // All categories from atom namespace.
    let categories: Vec<String> = entry
        .children()
        .filter(|n| n.has_tag_name((ATOM_NS, "category")))
        .filter_map(|n| n.attribute("term"))
        .map(|s| s.to_string())
        .collect();

    // Dates — warn if fallback to epoch occurs.
    let pub_raw = find_child_text(entry, ATOM_NS, "published").unwrap_or_default();
    let upd_raw = find_child_text(entry, ATOM_NS, "updated").unwrap_or_default();
    let published_parsed = parse_datetime(&pub_raw);
    let updated_parsed = parse_datetime(&upd_raw);

    if published_parsed.is_none() && !pub_raw.is_empty() {
        warn!(arxiv_id, raw = pub_raw, "failed to parse published date, falling back to epoch");
    }
    if updated_parsed.is_none() && !upd_raw.is_empty() {
        warn!(arxiv_id, raw = upd_raw, "failed to parse updated date, falling back to published/epoch");
    }

    let published = published_parsed.unwrap_or_default();
    let updated = updated_parsed.unwrap_or(published);

    // Optional fields.
    let doi = find_child_text(entry, ARXIV_NS, "doi");
    let journal_ref = find_child_text(entry, ARXIV_NS, "journal_ref");

    let pdf_url = format!("https://arxiv.org/pdf/{arxiv_id}.pdf");

    Some(ArxivPaper {
        arxiv_id,
        title,
        abstract_text,
        authors,
        categories,
        primary_category,
        published,
        updated,
        doi,
        journal_ref,
        pdf_url,
    })
}

// ── XML helper functions ────────────────────────────────────────────────

fn find_child<'a>(
    parent: &'a roxmltree::Node,
    ns: &str,
    local: &str,
) -> Option<roxmltree::Node<'a, 'a>> {
    parent.children().find(|n| n.has_tag_name((ns, local)))
}

fn find_child_text(parent: &roxmltree::Node, ns: &str, local: &str) -> Option<String> {
    find_child(parent, ns, local)
        .and_then(|n| n.text())
        .map(|s| s.to_string())
}

fn normalize_ws(s: &str) -> String {
    WS_RE.replace_all(s.trim(), " ").into_owned()
}

fn parse_datetime(s: &str) -> Option<DateTime<Utc>> {
    if s.is_empty() {
        return None;
    }
    // ArXiv uses ISO 8601 with Z suffix.
    let normalized = s.replace('Z', "+00:00");
    DateTime::parse_from_rfc3339(&normalized)
        .ok()
        .map(|dt| dt.with_timezone(&Utc))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_atom_response_single_entry() {
        let xml = r#"<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"
      xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/2501.12345v1</id>
    <title>Test Paper Title</title>
    <summary>This is the abstract.</summary>
    <author><name>Alice Smith</name></author>
    <author><name>Bob Jones</name></author>
    <published>2025-01-15T00:00:00Z</published>
    <updated>2025-01-16T12:00:00Z</updated>
    <arxiv:primary_category term="cs.AI" />
    <category term="cs.AI" />
    <category term="cs.CL" />
    <arxiv:doi>10.1234/test</arxiv:doi>
  </entry>
</feed>"#;

        let papers = parse_atom_response(xml).unwrap();
        assert_eq!(papers.len(), 1);

        let p = &papers[0];
        assert_eq!(p.arxiv_id, "2501.12345");
        assert_eq!(p.title, "Test Paper Title");
        assert_eq!(p.abstract_text, "This is the abstract.");
        assert_eq!(p.authors, vec!["Alice Smith", "Bob Jones"]);
        assert_eq!(p.primary_category, "cs.AI");
        assert_eq!(p.categories, vec!["cs.AI", "cs.CL"]);
        assert_eq!(p.doi, Some("10.1234/test".to_string()));
        assert_eq!(p.pdf_url, "https://arxiv.org/pdf/2501.12345.pdf");
    }

    #[test]
    fn test_parse_atom_response_skips_errors() {
        let xml = r#"<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/api/errors#invalid_id</id>
    <title>Error</title>
    <summary>Invalid ID</summary>
  </entry>
</feed>"#;

        let papers = parse_atom_response(xml).unwrap();
        assert!(papers.is_empty());
    }

    #[test]
    fn test_parse_atom_response_whitespace_normalization() {
        let xml = r#"<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"
      xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/2501.99999v1</id>
    <title>  Multi
    Line   Title  </title>
    <summary>  Abstract with   lots   of
    whitespace  </summary>
    <author><name>Author</name></author>
    <published>2025-01-01T00:00:00Z</published>
    <updated>2025-01-01T00:00:00Z</updated>
    <arxiv:primary_category term="cs.LG" />
    <category term="cs.LG" />
  </entry>
</feed>"#;

        let papers = parse_atom_response(xml).unwrap();
        assert_eq!(papers[0].title, "Multi Line Title");
        assert_eq!(
            papers[0].abstract_text,
            "Abstract with lots of whitespace"
        );
    }

    #[test]
    fn test_parse_datetime_variants() {
        assert!(parse_datetime("2025-01-15T00:00:00Z").is_some());
        assert!(parse_datetime("2025-01-15T12:30:00+00:00").is_some());
        assert!(parse_datetime("").is_none());
        assert!(parse_datetime("not-a-date").is_none());
    }

    #[test]
    fn test_download_result_failed() {
        let r = DownloadResult::failed("2501.12345", "test error");
        assert!(!r.success);
        assert_eq!(r.arxiv_id, "2501.12345");
        assert_eq!(r.error_message.as_deref(), Some("test error"));
        assert_eq!(r.file_size_bytes, 0);
        assert!(r.pdf_path.is_none());
    }

    #[test]
    fn test_client_constructors() {
        let default = ArxivClient::new().unwrap();
        assert_eq!(default.rate_limit_delay, Duration::from_secs(3));
        assert_eq!(default.max_retries, 3);

        let sync = ArxivClient::for_sync().unwrap();
        assert_eq!(sync.rate_limit_delay, Duration::from_millis(500));
    }

    #[test]
    fn test_parse_old_format_id() {
        let xml = r#"<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"
      xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/hep-th/9901001v1</id>
    <title>Old Format Paper</title>
    <summary>Old format abstract.</summary>
    <author><name>Physicist</name></author>
    <published>1999-01-05T00:00:00Z</published>
    <updated>1999-01-05T00:00:00Z</updated>
    <arxiv:primary_category term="hep-th" />
    <category term="hep-th" />
  </entry>
</feed>"#;

        let papers = parse_atom_response(xml).unwrap();
        assert_eq!(papers.len(), 1);
        assert_eq!(papers[0].arxiv_id, "hep-th/9901001");
    }

    #[test]
    fn test_arxiv_error_variants() {
        let err = ArxivError::InvalidId("bad-id".to_string());
        assert!(err.to_string().contains("bad-id"));

        let err = ArxivError::HttpStatus { status: 503, url: "http://test".to_string() };
        assert!(err.to_string().contains("503"));

        let err = ArxivError::XmlParse("bad xml".to_string());
        assert!(err.to_string().contains("bad xml"));
    }
}
