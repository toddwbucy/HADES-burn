//! ArangoDB document key normalization.
//!
//! Pure functions for transforming raw identifiers (file paths, external IDs,
//! anything with dots/slashes/version suffixes) into valid ArangoDB document keys.

use regex::Regex;
use sha2::{Digest, Sha256};
use std::sync::LazyLock;

/// Regex to strip trailing version suffix (e.g. `v1`, `v2`, `v12`).
static VERSION_SUFFIX: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"v\d+$").expect("invalid regex"));

/// Normalize a raw identifier into an ArangoDB document key.
///
/// 1. Strip trailing version suffix (`v1`, `v2`, etc.)
/// 2. Replace `.` and `/` with `_`
///
/// # Examples
/// ```
/// # use hades_core::db::keys::normalize_document_key;
/// assert_eq!(normalize_document_key("2501.12345v2"), "2501_12345");
/// assert_eq!(normalize_document_key("hep-th/9901001"), "hep-th_9901001");
/// ```
pub fn normalize_document_key(raw: &str) -> String {
    let stripped = VERSION_SUFFIX.replace(raw, "");
    stripped.replace(['.', '/'], "_")
}

/// Strip a trailing `v\d+` version suffix from an identifier without
/// replacing other delimiters. Works on any string with that versioning
/// convention (e.g. `2501.12345v1` → `2501.12345`, `libfoo-1.0v3` → `libfoo-1.0`).
///
/// # Examples
/// ```
/// # use hades_core::db::keys::strip_version;
/// assert_eq!(strip_version("2501.12345v1"), "2501.12345");
/// assert_eq!(strip_version("2501.12345"), "2501.12345");
/// ```
pub fn strip_version(id: &str) -> String {
    VERSION_SUFFIX.replace(id, "").into_owned()
}

/// Build a chunk key from a normalized document key and chunk index.
///
/// # Examples
/// ```
/// # use hades_core::db::keys::chunk_key;
/// assert_eq!(chunk_key("2501_12345", 3), "2501_12345_chunk_3");
/// ```
pub fn chunk_key(doc_key: &str, index: usize) -> String {
    format!("{doc_key}_chunk_{index}")
}

/// Build an embedding key from a chunk key.
///
/// # Examples
/// ```
/// # use hades_core::db::keys::embedding_key;
/// assert_eq!(embedding_key("2501_12345_chunk_3"), "2501_12345_chunk_3_emb");
/// ```
pub fn embedding_key(chunk_key: &str) -> String {
    format!("{chunk_key}_emb")
}

/// Normalize a file path into an ArangoDB document key.
///
/// Replaces `/` and `.` with `_`. No version stripping.
///
/// # Examples
/// ```
/// # use hades_core::db::keys::file_key;
/// assert_eq!(file_key("core/models.py"), "core_models_py");
/// ```
pub fn file_key(rel_path: &str) -> String {
    rel_path.replace(['.', '/'], "_")
}

/// Build a symbol key from a file key and qualified symbol name.
///
/// Produces a human-readable prefix plus a truncated SHA-256 hash of the
/// original qualified name to prevent collisions from lossy normalization
/// (e.g., `Vec<T>` vs `Vec_T_` would otherwise map to the same key).
///
/// Format: `{file_key}__{readable}__{hash8}`
///
/// # Examples
/// ```
/// # use hades_core::db::keys::{file_key, symbol_key};
/// let key = symbol_key("src_lib_rs", "Config::new");
/// assert!(key.starts_with("src_lib_rs__Config__new__"));
/// assert_eq!(key.len(), "src_lib_rs__Config__new__".len() + 8);
/// ```
pub fn symbol_key(file_key: &str, qualified_name: &str) -> String {
    // Readable prefix: replace :: with __, strip only ArangoDB-invalid chars.
    let readable = qualified_name
        .replace("::", "__")
        .replace(['<', '>', ' ', ',', ':', '\'', '"', '(', ')'], "_");

    // Deterministic 8-char hex hash of the original qualified name.
    let mut hasher = Sha256::new();
    hasher.update(qualified_name.as_bytes());
    let digest = hasher.finalize();
    let hash8 = hex8(&digest);

    format!("{file_key}__{readable}__{hash8}")
}

/// Build a deterministic edge key from source, type, and target.
///
/// Uses a truncated SHA-256 hash of the combined input to keep keys
/// short while preventing collisions from lossy normalization.
///
/// Format: `{from_prefix}__{kind}__{to_prefix}__{hash8}`
///
/// # Examples
/// ```
/// # use hades_core::db::keys::edge_key;
/// let key = edge_key("src_lib_rs__Config__abc12345", "defines", "src_lib_rs__Config__new__def67890");
/// assert!(key.contains("defines"));
/// ```
pub fn edge_key(from: &str, kind: &str, to: &str) -> String {
    // Truncate from/to for readability (first 20 chars each).
    let from_prefix: String = from.chars().take(20).collect();
    let to_prefix: String = to.chars().take(20).collect();

    // Deterministic hash of the full from+kind+to.
    let mut hasher = Sha256::new();
    hasher.update(from.as_bytes());
    hasher.update(b"|");
    hasher.update(kind.as_bytes());
    hasher.update(b"|");
    hasher.update(to.as_bytes());
    let digest = hasher.finalize();
    let hash8 = hex8(&digest);

    format!("{from_prefix}__{kind}__{to_prefix}__{hash8}")
}

/// Build a collision-resistant compliance-edge key from its components.
///
/// Compliance edges link a source document (in any collection) to a
/// `smell_specs` document. The key format is:
///
/// ```text
/// {source_col}__{source_key}__{spec_key}__{hash8}
/// ```
///
/// The hash uses null-byte-separated components, so
/// `("codebase_files", "foo_bar", "smell-001")` and
/// `("codebase", "files_foo_bar", "smell-001")` produce different
/// hashes — the readable prefix can collide under truncation, but the
/// hash disambiguates.
///
/// # Examples
/// ```
/// # use hades_core::db::keys::compliance_edge_key;
/// let key = compliance_edge_key("codebase_files", "foo", "smell-010");
/// assert!(key.starts_with("codebase_files__foo__smell-010__"));
/// // Different inputs that would collide under naive concatenation
/// // produce different keys here:
/// let a = compliance_edge_key("codebase_files", "foo_bar", "s");
/// let b = compliance_edge_key("codebase", "files_foo_bar", "s");
/// assert_ne!(a, b);
/// ```
pub fn compliance_edge_key(source_col: &str, source_key: &str, spec_key: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(source_col.as_bytes());
    hasher.update(b"\0");
    hasher.update(source_key.as_bytes());
    hasher.update(b"\0");
    hasher.update(spec_key.as_bytes());
    let digest = hasher.finalize();
    let hash8 = hex8(&digest);

    let readable: String = format!("{source_col}__{source_key}__{spec_key}")
        .chars()
        .take(40)
        .collect();
    format!("{readable}__{hash8}")
}

/// Compute a deterministic hash of a model identifier for stale-embedding detection.
///
/// Returns the full hex-encoded SHA-256 of the model string. When the model
/// name or version changes, the hash changes, triggering re-embedding.
pub fn model_hash(model_id: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(model_id.as_bytes());
    let digest = hasher.finalize();
    digest.iter().map(|b| format!("{b:02x}")).collect()
}

/// First 8 hex chars of a SHA-256 digest.
fn hex8(digest: &[u8]) -> String {
    // 4 bytes = 8 hex chars
    digest[..4].iter().map(|b| format!("{b:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_document_key() {
        assert_eq!(normalize_document_key("2501.12345v2"), "2501_12345");
        assert_eq!(normalize_document_key("2501.12345v1"), "2501_12345");
        assert_eq!(normalize_document_key("2501.12345"), "2501_12345");
        assert_eq!(normalize_document_key("hep-th/9901001"), "hep-th_9901001");
        assert_eq!(normalize_document_key("simple_key"), "simple_key");
        assert_eq!(
            normalize_document_key("path/to/file.txt"),
            "path_to_file_txt"
        );
        // Version-like substring in the middle should NOT be stripped
        assert_eq!(normalize_document_key("v2_doc.key"), "v2_doc_key");
    }

    #[test]
    fn test_strip_version() {
        // Dotted identifiers with trailing version
        assert_eq!(strip_version("2501.12345v1"), "2501.12345");
        assert_eq!(strip_version("2501.12345v12"), "2501.12345");
        assert_eq!(strip_version("2501.12345"), "2501.12345");
        // Function is generic over any `v\d+` suffix
        assert_eq!(strip_version("libfoo-1.0v3"), "libfoo-1.0");
        assert_eq!(strip_version("libfoo-1.0"), "libfoo-1.0");
        // version-like substring NOT at the end is left alone
        assert_eq!(strip_version("v2_doc.key"), "v2_doc.key");
    }

    #[test]
    fn test_chunk_key() {
        assert_eq!(chunk_key("doc_key", 0), "doc_key_chunk_0");
        assert_eq!(chunk_key("2501_12345", 3), "2501_12345_chunk_3");
    }

    #[test]
    fn test_embedding_key() {
        assert_eq!(embedding_key("doc_chunk_0"), "doc_chunk_0_emb");
        assert_eq!(
            embedding_key("2501_12345_chunk_3"),
            "2501_12345_chunk_3_emb"
        );
    }

    #[test]
    fn test_file_key() {
        assert_eq!(file_key("core/models.py"), "core_models_py");
        assert_eq!(file_key("README.md"), "README_md");
        assert_eq!(file_key("src/lib.rs"), "src_lib_rs");
        assert_eq!(
            file_key("core/persephone/models.py"),
            "core_persephone_models_py"
        );
    }

    #[test]
    fn test_symbol_key() {
        let key = symbol_key("src_lib_rs", "Config::new");
        assert!(key.starts_with("src_lib_rs__Config__new__"), "key: {key}");
        // Hash suffix is 8 hex chars.
        let suffix = key.strip_prefix("src_lib_rs__Config__new__").unwrap();
        assert_eq!(suffix.len(), 8, "hash suffix: {suffix}");

        let key2 = symbol_key("src_lib_rs", "Display for Config");
        assert!(
            key2.starts_with("src_lib_rs__Display_for_Config__"),
            "key: {key2}"
        );

        let key3 = symbol_key("src_lib_rs", "Vec<String>");
        assert!(key3.starts_with("src_lib_rs__Vec_String___"), "key: {key3}");
    }

    #[test]
    fn test_symbol_key_deterministic() {
        // Same input → same key.
        let a = symbol_key("src_lib_rs", "Config::new");
        let b = symbol_key("src_lib_rs", "Config::new");
        assert_eq!(a, b);
    }

    #[test]
    fn test_edge_key_deterministic() {
        let a = edge_key("src_lib_rs__Config__abc", "defines", "src_lib_rs__new__def");
        let b = edge_key("src_lib_rs__Config__abc", "defines", "src_lib_rs__new__def");
        assert_eq!(a, b);
    }

    #[test]
    fn test_edge_key_no_collision() {
        let a = edge_key("src_lib_rs__Config__abc", "defines", "src_lib_rs__new__def");
        let b = edge_key("src_lib_rs__Config__abc", "calls", "src_lib_rs__new__def");
        assert_ne!(a, b, "different edge kinds should produce different keys");
    }

    #[test]
    fn test_symbol_key_no_collision() {
        // Different qualified names → different keys even if readable prefix matches.
        let a = symbol_key("src_lib_rs", "Vec<T>");
        let b = symbol_key("src_lib_rs", "Vec_T_");
        assert_ne!(a, b, "should not collide: a={a}, b={b}");
    }
}
