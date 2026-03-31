//! ArXiv abstract sync — fetch, deduplicate, embed, and store papers.
//!
//! Implements the month-by-month query strategy to stay within arXiv API
//! pagination limits, deduplicates against existing papers in ArangoDB,
//! embeds abstracts via the Persephone embedding service, and stores
//! results across three collections (metadata, abstracts, embeddings).

use chrono::{Datelike, NaiveDate, Utc};
use serde_json::{json, Value};
use tracing::{debug, info, warn};

use crate::arxiv::{extract_year_month, ArxivClient, ArxivPaper};
use crate::db::collections::CollectionProfile;
use crate::db::crud;
use crate::db::keys::normalize_document_key;
use crate::db::ArangoPool;
use crate::db::query::{self, ExecutionTarget};
use crate::persephone::embedding::EmbeddingClient;

/// Maximum results per month query to avoid arXiv pagination ceiling.
const MAX_PER_MONTH: u32 = 5000;
/// Results per API page.
const PAGE_SIZE: u32 = 100;
/// Embedding task for abstract retrieval.
const EMBED_TASK: &str = "retrieval.passage";

/// Result of a sync operation.
#[derive(Debug)]
pub struct SyncResult {
    /// Total papers fetched from arXiv.
    pub fetched: usize,
    /// Papers that already existed (skipped).
    pub duplicates: usize,
    /// Papers successfully embedded and stored.
    pub stored: usize,
    /// Papers that failed during embed/store.
    pub errors: usize,
}

/// Configuration for a sync run.
pub struct SyncConfig {
    /// Start date for the query range.
    pub start_date: NaiveDate,
    /// Optional category filter (e.g. ["cs.AI", "cs.LG"]).
    pub categories: Vec<String>,
    /// Maximum total papers to sync.
    pub max_results: u32,
    /// Embedding batch size.
    pub batch_size: u32,
}

/// Run the full sync pipeline: fetch → deduplicate → embed → store.
pub async fn run_sync(
    pool: &ArangoPool,
    client: &ArxivClient,
    embed_client: &EmbeddingClient,
    config: &SyncConfig,
) -> anyhow::Result<SyncResult> {
    let profile = CollectionProfile::get("sync")
        .expect("sync profile must exist");

    // Phase 1: Fetch papers month by month.
    info!(
        start_date = %config.start_date,
        categories = ?config.categories,
        max = config.max_results,
        "starting arXiv abstract sync"
    );

    let papers = fetch_recent_papers(client, config).await?;
    let fetched = papers.len();
    info!(fetched, "papers fetched from arXiv API");

    if papers.is_empty() {
        return Ok(SyncResult {
            fetched: 0,
            duplicates: 0,
            stored: 0,
            errors: 0,
        });
    }

    // Phase 2: Deduplicate against existing papers.
    let new_papers = filter_existing(pool, profile, &papers).await?;
    let duplicates = fetched - new_papers.len();
    info!(new = new_papers.len(), duplicates, "deduplication complete");

    if new_papers.is_empty() {
        return Ok(SyncResult {
            fetched,
            duplicates,
            stored: 0,
            errors: 0,
        });
    }

    // Phase 3: Embed and store.
    let (stored, errors) = embed_and_store(
        pool,
        embed_client,
        profile,
        &new_papers,
        config.batch_size,
    )
    .await?;

    Ok(SyncResult {
        fetched,
        duplicates,
        stored,
        errors,
    })
}

// ── Phase 1: Fetch ──────────────────────────────────────────────────────

/// Fetch papers month-by-month from start_date to today.
async fn fetch_recent_papers(
    client: &ArxivClient,
    config: &SyncConfig,
) -> Result<Vec<ArxivPaper>, crate::arxiv::ArxivError> {
    let today = Utc::now().date_naive();
    let mut all_papers = Vec::new();
    let mut current = config.start_date;

    while current <= today && (all_papers.len() as u32) < config.max_results {
        // End of this month or today, whichever comes first.
        let month_end = end_of_month(current).min(today);
        let remaining = config.max_results - all_papers.len() as u32;
        let month_max = remaining.min(MAX_PER_MONTH);

        let papers = fetch_month_papers(
            client,
            current,
            month_end,
            &config.categories,
            month_max,
        )
        .await?;

        debug!(
            month = %current.format("%Y-%m"),
            count = papers.len(),
            "month fetch complete"
        );
        all_papers.extend(papers);

        // Advance to first day of next month.
        current = if current.month() == 12 {
            NaiveDate::from_ymd_opt(current.year() + 1, 1, 1)
        } else {
            NaiveDate::from_ymd_opt(current.year(), current.month() + 1, 1)
        }
        .unwrap_or(today);
    }

    Ok(all_papers)
}

/// Fetch papers for a single month range, paginating through results.
async fn fetch_month_papers(
    client: &ArxivClient,
    start: NaiveDate,
    end: NaiveDate,
    categories: &[String],
    max: u32,
) -> Result<Vec<ArxivPaper>, crate::arxiv::ArxivError> {
    let date_range = format!(
        "submittedDate:[{start}0000 TO {end}2359]",
        start = start.format("%Y%m%d"),
        end = end.format("%Y%m%d"),
    );

    let search_query = if categories.is_empty() {
        date_range
    } else {
        let cat_filter = categories
            .iter()
            .map(|c| format!("cat:{c}"))
            .collect::<Vec<_>>()
            .join(" OR ");
        format!("({cat_filter}) AND {date_range}")
    };

    let mut papers = Vec::new();
    let mut start_offset = 0u32;

    while (papers.len() as u32) < max {
        let page_max = PAGE_SIZE.min(max - papers.len() as u32);
        let page = client.search(&search_query, start_offset, page_max).await?;
        let page_len = page.len() as u32;

        papers.extend(page);

        if page_len < page_max {
            // No more results.
            break;
        }
        start_offset += page_len;
    }

    Ok(papers)
}

/// Get the last day of the month containing `date`.
fn end_of_month(date: NaiveDate) -> NaiveDate {
    let (y, m) = if date.month() == 12 {
        (date.year() + 1, 1)
    } else {
        (date.year(), date.month() + 1)
    };
    NaiveDate::from_ymd_opt(y, m, 1)
        .unwrap()
        .pred_opt()
        .unwrap()
}

// ── Phase 2: Deduplication ──────────────────────────────────────────────

/// Filter out papers that already exist in the metadata collection.
async fn filter_existing<'a>(
    pool: &ArangoPool,
    profile: &CollectionProfile,
    papers: &'a [ArxivPaper],
) -> anyhow::Result<Vec<&'a ArxivPaper>> {
    let ids: Vec<String> = papers.iter().map(|p| p.arxiv_id.clone()).collect();

    // Query existing IDs in batches of 500 to avoid huge bind vars.
    let mut existing_ids = std::collections::HashSet::new();
    for chunk in ids.chunks(500) {
        let chunk_list: Vec<Value> = chunk.iter().map(|id| json!(id)).collect();
        let aql = format!(
            "FOR doc IN {} FILTER doc.arxiv_id IN @ids RETURN doc.arxiv_id",
            profile.metadata
        );
        let bind_vars = json!({ "ids": chunk_list });
        let result = query::query(
            pool,
            &aql,
            Some(&bind_vars),
            None,
            false,
            ExecutionTarget::Reader,
        )
        .await?;

        for val in result.results {
            if let Some(id) = val.as_str() {
                existing_ids.insert(id.to_string());
            }
        }
    }

    let new_papers: Vec<&ArxivPaper> = papers
        .iter()
        .filter(|p| !existing_ids.contains(&p.arxiv_id))
        .collect();

    Ok(new_papers)
}

// ── Phase 3: Embed and Store ────────────────────────────────────────────

/// Embed abstracts and store papers in the three sync collections.
async fn embed_and_store(
    pool: &ArangoPool,
    embed_client: &EmbeddingClient,
    profile: &CollectionProfile,
    papers: &[&ArxivPaper],
    batch_size: u32,
) -> anyhow::Result<(usize, usize)> {
    let mut total_stored = 0usize;
    let mut total_errors = 0usize;

    for (batch_idx, batch) in papers.chunks(batch_size as usize).enumerate() {
        // Collect abstracts for embedding.
        let texts: Vec<String> = batch
            .iter()
            .map(|p| p.abstract_text.clone())
            .collect();

        // Embed the batch.
        let embed_result = match embed_client.embed(&texts, EMBED_TASK, Some(batch_size)).await {
            Ok(r) => r,
            Err(e) => {
                warn!(
                    batch = batch_idx,
                    error = %e,
                    count = batch.len(),
                    "embedding batch failed, skipping"
                );
                total_errors += batch.len();
                continue;
            }
        };

        debug!(
            batch = batch_idx,
            count = batch.len(),
            dimension = embed_result.dimension,
            duration_ms = embed_result.duration_ms,
            "batch embedded"
        );

        // Build documents for all three collections.
        let mut meta_docs = Vec::with_capacity(batch.len());
        let mut chunk_docs = Vec::with_capacity(batch.len());
        let mut embed_docs = Vec::with_capacity(batch.len());

        for (i, paper) in batch.iter().enumerate() {
            let doc_key = normalize_document_key(&paper.arxiv_id);
            let (year, month, year_month) = extract_year_month_full(&paper.arxiv_id, paper);

            meta_docs.push(json!({
                "_key": doc_key,
                "arxiv_id": paper.arxiv_id,
                "title": paper.title,
                "authors": paper.authors,
                "categories": paper.categories,
                "primary_category": paper.primary_category,
                "year": year,
                "month": month,
                "year_month": year_month,
                "created_at": Utc::now().to_rfc3339(),
            }));

            chunk_docs.push(json!({
                "_key": doc_key,
                "arxiv_id": paper.arxiv_id,
                "title": paper.title,
                "abstract": paper.abstract_text,
            }));

            embed_docs.push(json!({
                "_key": doc_key,
                "arxiv_id": paper.arxiv_id,
                "combined_embedding": embed_result.embeddings[i],
                "abstract_embedding": [],
                "title_embedding": [],
            }));
        }

        // Insert into all three collections (silently skip duplicates).
        let batch_stored = insert_batch(pool, profile, &meta_docs, &chunk_docs, &embed_docs).await;
        match batch_stored {
            Ok(n) => {
                total_stored += n;
                info!(
                    batch = batch_idx,
                    stored = n,
                    "batch stored"
                );
            }
            Err(e) => {
                warn!(batch = batch_idx, error = %e, "batch insert failed");
                total_errors += batch.len();
            }
        }
    }

    Ok((total_stored, total_errors))
}

/// Insert documents into the three sync collections.
///
/// Uses overwrite=false so duplicates are silently skipped (409 → 0 created).
/// Returns the number of new papers inserted (based on metadata collection).
async fn insert_batch(
    pool: &ArangoPool,
    profile: &CollectionProfile,
    meta_docs: &[Value],
    chunk_docs: &[Value],
    embed_docs: &[Value],
) -> anyhow::Result<usize> {
    let meta_result = crud::insert_documents(pool, profile.metadata, meta_docs, false).await?;
    let _chunk_result = crud::insert_documents(pool, profile.chunks, chunk_docs, false).await?;
    let _embed_result = crud::insert_documents(pool, profile.embeddings, embed_docs, false).await?;

    Ok(meta_result.created as usize)
}

/// Extract year, month, and year_month string from an arXiv paper.
///
/// For new-format IDs (YYMM.NNNNN), parses from the ID directly.
/// Falls back to the published date for old-format IDs.
fn extract_year_month_full(arxiv_id: &str, paper: &ArxivPaper) -> (i32, u32, String) {
    if let Some((y_str, m_str)) = extract_year_month(arxiv_id) {
        let year = 2000 + y_str.parse::<i32>().unwrap_or(0);
        let month = m_str.parse::<u32>().unwrap_or(1);
        let year_month = format!("{year}{month:02}");
        (year, month, year_month)
    } else {
        // Old-format ID — use published date.
        let year = paper.published.year();
        let month = paper.published.month();
        let year_month = format!("{year}{month:02}");
        (year, month, year_month)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_end_of_month() {
        assert_eq!(
            end_of_month(NaiveDate::from_ymd_opt(2025, 1, 15).unwrap()),
            NaiveDate::from_ymd_opt(2025, 1, 31).unwrap()
        );
        assert_eq!(
            end_of_month(NaiveDate::from_ymd_opt(2024, 2, 1).unwrap()),
            NaiveDate::from_ymd_opt(2024, 2, 29).unwrap() // leap year
        );
        assert_eq!(
            end_of_month(NaiveDate::from_ymd_opt(2025, 12, 1).unwrap()),
            NaiveDate::from_ymd_opt(2025, 12, 31).unwrap()
        );
    }

    #[test]
    fn test_extract_year_month_full_new_format() {
        let paper = ArxivPaper {
            arxiv_id: "2501.12345".to_string(),
            title: String::new(),
            abstract_text: String::new(),
            authors: vec![],
            categories: vec![],
            primary_category: String::new(),
            published: Utc::now(),
            updated: Utc::now(),
            doi: None,
            journal_ref: None,
            pdf_url: String::new(),
        };
        let (year, month, ym) = extract_year_month_full("2501.12345", &paper);
        assert_eq!(year, 2025);
        assert_eq!(month, 1);
        assert_eq!(ym, "202501");
    }

    #[test]
    fn test_extract_year_month_full_old_format() {
        use chrono::TimeZone;
        let paper = ArxivPaper {
            arxiv_id: "hep-th/9901001".to_string(),
            title: String::new(),
            abstract_text: String::new(),
            authors: vec![],
            categories: vec![],
            primary_category: String::new(),
            published: Utc.with_ymd_and_hms(1999, 1, 5, 0, 0, 0).unwrap(),
            updated: Utc.with_ymd_and_hms(1999, 1, 5, 0, 0, 0).unwrap(),
            doi: None,
            journal_ref: None,
            pdf_url: String::new(),
        };
        let (year, month, ym) = extract_year_month_full("hep-th/9901001", &paper);
        assert_eq!(year, 1999);
        assert_eq!(month, 1);
        assert_eq!(ym, "199901");
    }
}
