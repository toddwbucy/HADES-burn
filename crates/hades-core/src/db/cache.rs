//! Concurrent cache layer for ArangoDB operations.
//!
//! Provides two caches backed by [`moka::future::Cache`]:
//! - **Document cache**: keyed by `(collection, key)`, TTL 5 minutes.
//! - **Query cache**: keyed by hash of `(AQL, bind_vars, batch_size)`,
//!   TTL 60 seconds.  Bypassed for mutating queries.
//!
//! Both caches are lock-free, async-compatible, and support concurrent
//! access from multiple tokio tasks.

use std::hash::{DefaultHasher, Hash, Hasher};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use moka::future::Cache;
use serde_json::Value;
use tracing::{debug, instrument, trace};

use super::crud;
use super::error::ArangoError;
use super::pool::ArangoPool;
use super::query::{self, ExecutionTarget, QueryResult};

/// Default TTL for cached documents.
const DEFAULT_DOC_TTL: Duration = Duration::from_secs(300); // 5 minutes
/// Default TTL for cached query results.
const DEFAULT_QUERY_TTL: Duration = Duration::from_secs(60); // 1 minute
/// Default maximum entries in the document cache.
const DEFAULT_DOC_MAX_ENTRIES: u64 = 10_000;
/// Default maximum entries in the query cache.
const DEFAULT_QUERY_MAX_ENTRIES: u64 = 1_000;

/// Configuration for the cache layer.
#[derive(Debug, Clone)]
pub struct CacheConfig {
    /// TTL for cached documents.
    pub doc_ttl: Duration,
    /// TTL for cached query results.
    pub query_ttl: Duration,
    /// Maximum entries in the document cache.
    pub doc_max_entries: u64,
    /// Maximum entries in the query cache.
    pub query_max_entries: u64,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            doc_ttl: DEFAULT_DOC_TTL,
            query_ttl: DEFAULT_QUERY_TTL,
            doc_max_entries: DEFAULT_DOC_MAX_ENTRIES,
            query_max_entries: DEFAULT_QUERY_MAX_ENTRIES,
        }
    }
}

/// Cache metrics exposed via tracing.
#[derive(Debug, Default)]
struct Metrics {
    doc_hits: AtomicU64,
    doc_misses: AtomicU64,
    query_hits: AtomicU64,
    query_misses: AtomicU64,
}

/// Cached ArangoDB operations layer.
///
/// Wraps an [`ArangoPool`] with document and query result caches.
/// Write operations (insert, update, delete) invalidate relevant
/// document cache entries.
pub struct CachedPool {
    pool: ArangoPool,
    doc_cache: Cache<String, Value>,
    query_cache: Cache<u64, QueryResult>,
    metrics: Metrics,
}

impl CachedPool {
    /// Create a new cached pool with the given configuration.
    pub fn new(pool: ArangoPool, config: CacheConfig) -> Self {
        let doc_cache = Cache::builder()
            .max_capacity(config.doc_max_entries)
            .time_to_live(config.doc_ttl)
            .build();

        let query_cache = Cache::builder()
            .max_capacity(config.query_max_entries)
            .time_to_live(config.query_ttl)
            .build();

        debug!(
            doc_ttl_secs = config.doc_ttl.as_secs(),
            query_ttl_secs = config.query_ttl.as_secs(),
            doc_max = config.doc_max_entries,
            query_max = config.query_max_entries,
            "cache layer initialized"
        );

        Self {
            pool,
            doc_cache,
            query_cache,
            metrics: Metrics::default(),
        }
    }

    /// Create a new cached pool with default configuration.
    pub fn with_defaults(pool: ArangoPool) -> Self {
        Self::new(pool, CacheConfig::default())
    }

    /// Access the underlying pool directly (bypasses cache).
    pub fn pool(&self) -> &ArangoPool {
        &self.pool
    }

    // -----------------------------------------------------------------------
    // Document operations (cached)
    // -----------------------------------------------------------------------

    /// Get a document, returning a cached copy if available.
    #[instrument(skip(self), fields(db = %self.pool.database()))]
    pub async fn get_document(
        &self,
        collection: &str,
        key: &str,
    ) -> Result<Value, ArangoError> {
        let cache_key = doc_cache_key(collection, key);

        if let Some(cached) = self.doc_cache.get(&cache_key).await {
            self.metrics.doc_hits.fetch_add(1, Ordering::Relaxed);
            trace!(collection, key, "document cache hit");
            return Ok(cached);
        }

        self.metrics.doc_misses.fetch_add(1, Ordering::Relaxed);
        trace!(collection, key, "document cache miss");

        let doc = crud::get_document(&self.pool, collection, key).await?;
        self.doc_cache.insert(cache_key, doc.clone()).await;
        Ok(doc)
    }

    /// Delete a document and invalidate its cache entry.
    #[instrument(skip(self), fields(db = %self.pool.database()))]
    pub async fn delete_document(
        &self,
        collection: &str,
        key: &str,
    ) -> Result<Value, ArangoError> {
        let result = crud::delete_document(&self.pool, collection, key).await?;
        self.doc_cache.invalidate(&doc_cache_key(collection, key)).await;
        trace!(collection, key, "document cache invalidated (delete)");
        Ok(result)
    }

    /// Update a document and invalidate its cache entry.
    #[instrument(skip(self, data), fields(db = %self.pool.database()))]
    pub async fn update_document(
        &self,
        collection: &str,
        key: &str,
        data: &Value,
    ) -> Result<Value, ArangoError> {
        let result = crud::update_document(&self.pool, collection, key, data).await?;
        self.doc_cache.invalidate(&doc_cache_key(collection, key)).await;
        trace!(collection, key, "document cache invalidated (update)");
        Ok(result)
    }

    /// Replace a document and invalidate its cache entry.
    #[instrument(skip(self, data), fields(db = %self.pool.database()))]
    pub async fn replace_document(
        &self,
        collection: &str,
        key: &str,
        data: &Value,
    ) -> Result<Value, ArangoError> {
        let result = crud::replace_document(&self.pool, collection, key, data).await?;
        self.doc_cache.invalidate(&doc_cache_key(collection, key)).await;
        trace!(collection, key, "document cache invalidated (replace)");
        Ok(result)
    }

    /// Insert documents (bulk). Does not populate cache (too many docs),
    /// but invalidates the query cache since collection contents changed.
    #[instrument(skip(self, docs), fields(db = %self.pool.database()))]
    pub async fn insert_documents(
        &self,
        collection: &str,
        docs: &[Value],
        overwrite: bool,
    ) -> Result<crud::ImportResult, ArangoError> {
        let result = crud::insert_documents(&self.pool, collection, docs, overwrite).await?;
        // Invalidate all query cache entries since we can't know which
        // queries are affected by the new documents.
        self.query_cache.invalidate_all();
        trace!(collection, created = result.created, "query cache invalidated (insert)");
        Ok(result)
    }

    // -----------------------------------------------------------------------
    // Query operations (cached)
    // -----------------------------------------------------------------------

    /// Execute an AQL query with caching for read-only queries.
    ///
    /// Write queries (`ExecutionTarget::Writer`) bypass the cache entirely.
    #[instrument(skip(self, bind_vars), fields(db = %self.pool.database()))]
    pub async fn query(
        &self,
        aql: &str,
        bind_vars: Option<&Value>,
        batch_size: Option<u32>,
        full_count: bool,
        target: ExecutionTarget,
    ) -> Result<QueryResult, ArangoError> {
        // Never cache write queries
        if target == ExecutionTarget::Writer {
            return query::query(
                &self.pool, aql, bind_vars, batch_size, full_count, target,
            )
            .await;
        }

        let cache_key = query_cache_key(aql, bind_vars, batch_size);

        if let Some(cached) = self.query_cache.get(&cache_key).await {
            self.metrics.query_hits.fetch_add(1, Ordering::Relaxed);
            trace!("query cache hit");
            return Ok(cached);
        }

        self.metrics.query_misses.fetch_add(1, Ordering::Relaxed);
        trace!("query cache miss");

        let result = query::query(
            &self.pool, aql, bind_vars, batch_size, full_count, target,
        )
        .await?;

        self.query_cache.insert(cache_key, result.clone()).await;
        Ok(result)
    }

    /// Execute an AQL query and return the first result, with caching.
    #[instrument(skip(self, bind_vars), fields(db = %self.pool.database()))]
    pub async fn query_single(
        &self,
        aql: &str,
        bind_vars: Option<&Value>,
        target: ExecutionTarget,
    ) -> Result<Option<Value>, ArangoError> {
        let result = self.query(aql, bind_vars, None, false, target).await?;
        Ok(result.results.into_iter().next())
    }

    // -----------------------------------------------------------------------
    // Cache management
    // -----------------------------------------------------------------------

    /// Invalidate all entries in both caches.
    pub fn invalidate_all(&self) {
        self.doc_cache.invalidate_all();
        self.query_cache.invalidate_all();
        debug!("all caches invalidated");
    }

    /// Invalidate a specific document from the cache.
    pub async fn invalidate_document(&self, collection: &str, key: &str) {
        self.doc_cache.invalidate(&doc_cache_key(collection, key)).await;
    }

    /// Invalidate all query cache entries.
    pub fn invalidate_queries(&self) {
        self.query_cache.invalidate_all();
    }

    /// Get current cache metrics.
    pub fn metrics(&self) -> CacheMetrics {
        CacheMetrics {
            doc_hits: self.metrics.doc_hits.load(Ordering::Relaxed),
            doc_misses: self.metrics.doc_misses.load(Ordering::Relaxed),
            query_hits: self.metrics.query_hits.load(Ordering::Relaxed),
            query_misses: self.metrics.query_misses.load(Ordering::Relaxed),
            doc_entry_count: self.doc_cache.entry_count(),
            query_entry_count: self.query_cache.entry_count(),
        }
    }

    /// Log current cache metrics via tracing.
    pub fn log_metrics(&self) {
        let m = self.metrics();
        debug!(
            doc_hits = m.doc_hits,
            doc_misses = m.doc_misses,
            doc_hit_rate = format_args!("{:.1}%", m.doc_hit_rate() * 100.0),
            query_hits = m.query_hits,
            query_misses = m.query_misses,
            query_hit_rate = format_args!("{:.1}%", m.query_hit_rate() * 100.0),
            doc_entries = m.doc_entry_count,
            query_entries = m.query_entry_count,
            "cache metrics"
        );
    }
}

/// Snapshot of cache hit/miss metrics.
#[derive(Debug, Clone, serde::Serialize)]
pub struct CacheMetrics {
    pub doc_hits: u64,
    pub doc_misses: u64,
    pub query_hits: u64,
    pub query_misses: u64,
    pub doc_entry_count: u64,
    pub query_entry_count: u64,
}

impl CacheMetrics {
    /// Document cache hit rate (0.0–1.0).
    pub fn doc_hit_rate(&self) -> f64 {
        let total = self.doc_hits + self.doc_misses;
        if total == 0 {
            0.0
        } else {
            self.doc_hits as f64 / total as f64
        }
    }

    /// Query cache hit rate (0.0–1.0).
    pub fn query_hit_rate(&self) -> f64 {
        let total = self.query_hits + self.query_misses;
        if total == 0 {
            0.0
        } else {
            self.query_hits as f64 / total as f64
        }
    }
}

// ---------------------------------------------------------------------------
// Cache key helpers
// ---------------------------------------------------------------------------

/// Build a document cache key from collection and document key.
fn doc_cache_key(collection: &str, key: &str) -> String {
    format!("{collection}/{key}")
}

/// Build a query cache key by hashing query parameters.
fn query_cache_key(aql: &str, bind_vars: Option<&Value>, batch_size: Option<u32>) -> u64 {
    let mut hasher = DefaultHasher::new();
    aql.hash(&mut hasher);
    if let Some(vars) = bind_vars {
        vars.to_string().hash(&mut hasher);
    }
    batch_size.hash(&mut hasher);
    hasher.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_doc_cache_key() {
        let key = doc_cache_key("arxiv_metadata", "2501_12345");
        assert_eq!(key, "arxiv_metadata/2501_12345");
    }

    #[test]
    fn test_query_cache_key_deterministic() {
        let vars = serde_json::json!({"key": "value"});
        let k1 = query_cache_key("RETURN 1", Some(&vars), Some(100));
        let k2 = query_cache_key("RETURN 1", Some(&vars), Some(100));
        assert_eq!(k1, k2);
    }

    #[test]
    fn test_query_cache_key_differs_by_aql() {
        let k1 = query_cache_key("RETURN 1", None, None);
        let k2 = query_cache_key("RETURN 2", None, None);
        assert_ne!(k1, k2);
    }

    #[test]
    fn test_query_cache_key_differs_by_bind_vars() {
        let v1 = serde_json::json!({"a": 1});
        let v2 = serde_json::json!({"a": 2});
        let k1 = query_cache_key("RETURN @a", Some(&v1), None);
        let k2 = query_cache_key("RETURN @a", Some(&v2), None);
        assert_ne!(k1, k2);
    }

    #[test]
    fn test_query_cache_key_differs_by_batch_size() {
        let k1 = query_cache_key("RETURN 1", None, Some(100));
        let k2 = query_cache_key("RETURN 1", None, Some(200));
        assert_ne!(k1, k2);
    }

    #[test]
    fn test_cache_config_defaults() {
        let config = CacheConfig::default();
        assert_eq!(config.doc_ttl, Duration::from_secs(300));
        assert_eq!(config.query_ttl, Duration::from_secs(60));
        assert_eq!(config.doc_max_entries, 10_000);
        assert_eq!(config.query_max_entries, 1_000);
    }

    #[test]
    fn test_cache_metrics_hit_rate() {
        let m = CacheMetrics {
            doc_hits: 75,
            doc_misses: 25,
            query_hits: 0,
            query_misses: 0,
            doc_entry_count: 50,
            query_entry_count: 0,
        };
        assert!((m.doc_hit_rate() - 0.75).abs() < 1e-10);
        assert!((m.query_hit_rate() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_cache_metrics_zero_total() {
        let m = CacheMetrics {
            doc_hits: 0,
            doc_misses: 0,
            query_hits: 0,
            query_misses: 0,
            doc_entry_count: 0,
            query_entry_count: 0,
        };
        assert!((m.doc_hit_rate() - 0.0).abs() < 1e-10);
    }
}
