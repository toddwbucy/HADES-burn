//! Core ArangoDB document and collection CRUD operations.
//!
//! All methods operate through [`ArangoPool`], routing reads to the
//! reader client and writes to the writer client.

use serde_json::Value;
use tracing::{debug, instrument, trace};

use super::error::ArangoError;
use super::pool::ArangoPool;

/// Default number of documents per batch for bulk inserts.
const DEFAULT_CHUNK_SIZE: usize = 1000;

// ---------------------------------------------------------------------------
// Document operations
// ---------------------------------------------------------------------------

/// Get a single document by collection and key.
///
/// Returns the full document as JSON, or `ArangoError` with
/// `kind() == NotFound` if the document doesn't exist.
#[instrument(skip(pool), fields(db = %pool.database()))]
pub async fn get_document(
    pool: &ArangoPool,
    collection: &str,
    key: &str,
) -> Result<Value, ArangoError> {
    let path = format!("document/{collection}/{key}");
    pool.reader().get(&path).await
}

/// Insert a single document (or array of documents) into a collection.
///
/// Uses `POST /_api/document/{collection}`.  For a single object the
/// response contains `_key`, `_id`, and `_rev`.  For an array the
/// response is an array of such objects.
#[instrument(skip(pool, data), fields(db = %pool.database()))]
pub async fn insert_document(
    pool: &ArangoPool,
    collection: &str,
    data: &Value,
) -> Result<Value, ArangoError> {
    let path = format!("document/{collection}");
    pool.writer().post(&path, data).await
}

/// Delete a single document by collection and key.
#[instrument(skip(pool), fields(db = %pool.database()))]
pub async fn delete_document(
    pool: &ArangoPool,
    collection: &str,
    key: &str,
) -> Result<Value, ArangoError> {
    let path = format!("document/{collection}/{key}");
    pool.writer().delete(&path).await
}

/// Update (merge-patch) a document's fields.
///
/// Uses PATCH — only the provided fields are updated, the rest are
/// preserved.  Use [`replace_document`] for a full replacement.
#[instrument(skip(pool, data), fields(db = %pool.database()))]
pub async fn update_document(
    pool: &ArangoPool,
    collection: &str,
    key: &str,
    data: &Value,
) -> Result<Value, ArangoError> {
    let path = format!("document/{collection}/{key}");
    pool.writer().patch(&path, data).await
}

/// Replace a document entirely.
///
/// Uses PUT — the entire document body is replaced.
#[instrument(skip(pool, data), fields(db = %pool.database()))]
pub async fn replace_document(
    pool: &ArangoPool,
    collection: &str,
    key: &str,
    data: &Value,
) -> Result<Value, ArangoError> {
    let path = format!("document/{collection}/{key}");
    pool.writer().put(&path, data).await
}

// ---------------------------------------------------------------------------
// Bulk import (NDJSON)
// ---------------------------------------------------------------------------

/// Import result from a single batch.
#[derive(Debug, Clone)]
pub struct ImportResult {
    /// Number of documents created.
    pub created: u64,
    /// Number of errors (if any).
    pub errors: u64,
    /// Number of empty lines ignored.
    pub empty: u64,
    /// Number of documents updated (when overwrite is true).
    pub updated: u64,
}

impl ImportResult {
    fn from_response(resp: &Value) -> Self {
        Self {
            created: resp["created"].as_u64().unwrap_or(0),
            errors: resp["errors"].as_u64().unwrap_or(0),
            empty: resp["empty"].as_u64().unwrap_or(0),
            updated: resp["updated"].as_u64().unwrap_or(0),
        }
    }

    fn empty() -> Self {
        Self {
            created: 0,
            errors: 0,
            empty: 0,
            updated: 0,
        }
    }

    fn merge(&mut self, other: &ImportResult) {
        self.created += other.created;
        self.errors += other.errors;
        self.empty += other.empty;
        self.updated += other.updated;
    }
}

/// Import documents in NDJSON format (atomic batch).
///
/// Uses `POST /_api/import?collection={col}&type=documents&complete=true`.
/// The `complete=true` flag makes the import atomic — all documents are
/// inserted or none are.
///
/// `overwrite` controls whether existing documents (by `_key`) are
/// replaced.
#[instrument(skip(pool, docs), fields(db = %pool.database(), count = docs.len()))]
pub async fn insert_documents(
    pool: &ArangoPool,
    collection: &str,
    docs: &[Value],
    overwrite: bool,
) -> Result<ImportResult, ArangoError> {
    if docs.is_empty() {
        return Ok(ImportResult::empty());
    }

    let path = format!(
        "import?collection={collection}&type=documents&complete=true&overwrite={overwrite}"
    );

    // Build NDJSON body: one JSON object per line, no trailing newline
    let mut ndjson = String::new();
    for (i, doc) in docs.iter().enumerate() {
        if i > 0 {
            ndjson.push('\n');
        }
        // serde_json::to_string produces single-line JSON
        ndjson.push_str(&serde_json::to_string(doc)?);
    }

    debug!(collection, doc_count = docs.len(), overwrite, "importing documents");

    let resp = pool
        .writer()
        .post_raw(&path, &ndjson, "application/x-ndjson")
        .await?;

    let result = ImportResult::from_response(&resp);
    debug!(
        created = result.created,
        errors = result.errors,
        updated = result.updated,
        "import complete"
    );
    Ok(result)
}

/// Error returned when a chunked bulk insert fails partway through.
///
/// Earlier chunks may have been committed (each chunk is atomic via
/// `complete=true`, but the overall bulk insert is **chunk-atomic**, not
/// fully atomic).
#[derive(Debug)]
pub struct PartialImportError {
    /// Aggregated results from chunks that succeeded before the failure.
    pub committed: ImportResult,
    /// The error that caused the failure.
    pub error: ArangoError,
}

impl std::fmt::Display for PartialImportError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "bulk import failed after {} created: {}",
            self.committed.created, self.error
        )
    }
}

impl std::error::Error for PartialImportError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.error)
    }
}

/// Bulk insert documents with automatic chunking.
///
/// Splits `docs` into chunks of `chunk_size` (default 1000) and imports
/// each chunk atomically.  The overall operation is **chunk-atomic**: each
/// chunk is all-or-nothing, but earlier chunks are committed even if a
/// later chunk fails.
///
/// On partial failure, returns `Err(PartialImportError)` containing
/// the aggregated results from committed chunks plus the underlying error.
#[instrument(skip(pool, docs), fields(db = %pool.database(), total = docs.len()))]
pub async fn bulk_insert(
    pool: &ArangoPool,
    collection: &str,
    docs: &[Value],
    chunk_size: Option<usize>,
    overwrite: bool,
) -> Result<ImportResult, PartialImportError> {
    let chunk_size = chunk_size.unwrap_or(DEFAULT_CHUNK_SIZE);
    if chunk_size == 0 {
        return Err(PartialImportError {
            committed: ImportResult::empty(),
            error: ArangoError::Request("chunk_size must be > 0".to_string()),
        });
    }

    let mut total = ImportResult::empty();

    for (i, chunk) in docs.chunks(chunk_size).enumerate() {
        trace!(chunk_index = i, chunk_size = chunk.len(), "importing chunk");
        match insert_documents(pool, collection, chunk, overwrite).await {
            Ok(result) => total.merge(&result),
            Err(error) => {
                return Err(PartialImportError {
                    committed: total,
                    error,
                });
            }
        }
    }

    debug!(
        created = total.created,
        errors = total.errors,
        chunks = docs.len().div_ceil(chunk_size),
        "bulk insert complete"
    );
    Ok(total)
}

// ---------------------------------------------------------------------------
// Collection operations
// ---------------------------------------------------------------------------

/// Collection info returned by [`list_collections`].
#[derive(Debug, Clone, serde::Serialize)]
pub struct CollectionInfo {
    pub name: String,
    /// 2 = document, 3 = edge
    pub collection_type: u32,
}

impl CollectionInfo {
    /// ArangoDB type code for document collections.
    pub const DOCUMENT_TYPE: u32 = 2;
}

/// List collections in the database, optionally filtering system collections.
///
/// System collections have names starting with `_`.
#[instrument(skip(pool), fields(db = %pool.database()))]
pub async fn list_collections(
    pool: &ArangoPool,
    exclude_system: bool,
) -> Result<Vec<CollectionInfo>, ArangoError> {
    let resp = pool.reader().get("collection").await?;
    let arr = resp["result"]
        .as_array()
        .ok_or_else(|| ArangoError::Request(
            "list collections: missing or non-array 'result' field".to_string(),
        ))?;

    let mut collections = Vec::with_capacity(arr.len());
    for c in arr {
        let name = c["name"]
            .as_str()
            .ok_or_else(|| ArangoError::Request(
                "list collections: entry missing 'name' string".to_string(),
            ))?;
        if exclude_system && name.starts_with('_') {
            continue;
        }
        let collection_type = c["type"]
            .as_u64()
            .ok_or_else(|| ArangoError::Request(
                format!("list collections: entry '{name}' missing 'type' integer"),
            ))? as u32;
        collections.push(CollectionInfo {
            name: name.to_string(),
            collection_type,
        });
    }
    Ok(collections)
}

/// Get the document count for a collection.
#[instrument(skip(pool), fields(db = %pool.database()))]
pub async fn count_collection(
    pool: &ArangoPool,
    collection: &str,
) -> Result<u64, ArangoError> {
    let path = format!("collection/{collection}/count");
    let resp = pool.reader().get(&path).await?;
    resp["count"]
        .as_u64()
        .ok_or_else(|| ArangoError::Request(
            format!("count_collection: missing or non-numeric 'count' in response for '{collection}'"),
        ))
}

/// Create a new collection.
///
/// `collection_type`: 2 = document (default), 3 = edge.
#[instrument(skip(pool), fields(db = %pool.database()))]
pub async fn create_collection(
    pool: &ArangoPool,
    name: &str,
    collection_type: Option<u32>,
) -> Result<Value, ArangoError> {
    let mut body = serde_json::json!({ "name": name });
    if let Some(ct) = collection_type {
        body["type"] = Value::from(ct);
    }
    pool.writer().post("collection", &body).await
}

/// Drop a collection.
///
/// Returns `Ok` even if the collection doesn't exist when
/// `ignore_missing` is true.
#[instrument(skip(pool), fields(db = %pool.database()))]
pub async fn drop_collection(
    pool: &ArangoPool,
    name: &str,
    ignore_missing: bool,
) -> Result<Value, ArangoError> {
    let path = format!("collection/{name}");
    match pool.writer().delete(&path).await {
        Ok(resp) => Ok(resp),
        Err(e) if ignore_missing && e.is_not_found() => {
            debug!(collection = name, "collection not found, ignoring");
            Ok(serde_json::json!({"dropped": false, "name": name}))
        }
        Err(e) => Err(e),
    }
}
