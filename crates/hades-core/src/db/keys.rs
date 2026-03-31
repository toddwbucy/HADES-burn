//! ArangoDB document key normalization.
//!
//! Pure functions for transforming raw identifiers (arxiv IDs, file paths)
//! into valid ArangoDB document keys.

use regex::Regex;
use std::sync::LazyLock;

/// Regex to strip trailing version suffix (e.g. `v1`, `v2`, `v12`).
static VERSION_SUFFIX: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"v\d+$").expect("invalid regex")
});

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

/// Strip the version suffix from an arxiv ID without replacing delimiters.
///
/// # Examples
/// ```
/// # use hades_core::db::keys::strip_version;
/// assert_eq!(strip_version("2501.12345v1"), "2501.12345");
/// assert_eq!(strip_version("2501.12345"), "2501.12345");
/// ```
pub fn strip_version(arxiv_id: &str) -> String {
    VERSION_SUFFIX.replace(arxiv_id, "").into_owned()
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
/// Replaces unsafe characters (`:`, `<`, `>`, ` `) with `_` and
/// joins with `__` to match the Python HADES key format.
///
/// # Examples
/// ```
/// # use hades_core::db::keys::{file_key, symbol_key};
/// assert_eq!(
///     symbol_key("src_lib_rs", "Config::new"),
///     "src_lib_rs__Config__new"
/// );
/// ```
pub fn symbol_key(file_key: &str, qualified_name: &str) -> String {
    let safe_name = qualified_name.replace("::", "__").replace(['<', '>', ' ', ','], "_");
    format!("{file_key}__{safe_name}")
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
        assert_eq!(normalize_document_key("path/to/file.txt"), "path_to_file_txt");
        // Version-like substring in the middle should NOT be stripped
        assert_eq!(normalize_document_key("v2_doc.key"), "v2_doc_key");
    }

    #[test]
    fn test_strip_version() {
        assert_eq!(strip_version("2501.12345v1"), "2501.12345");
        assert_eq!(strip_version("2501.12345v12"), "2501.12345");
        assert_eq!(strip_version("2501.12345"), "2501.12345");
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
        assert_eq!(
            symbol_key("src_lib_rs", "Config::new"),
            "src_lib_rs__Config__new"
        );
        assert_eq!(
            symbol_key("src_lib_rs", "Display for Config"),
            "src_lib_rs__Display_for_Config"
        );
        assert_eq!(
            symbol_key("src_lib_rs", "Vec<String>"),
            "src_lib_rs__Vec_String_"
        );
    }
}
