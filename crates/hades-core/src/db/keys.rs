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
/// The readable prefix is guaranteed ArangoDB-legal: characters that ArangoDB
/// rejects in `_key` (e.g. `& [ ] #`, non-ASCII) are replaced with `_`, and
/// the prefix is truncated so the full key stays within ArangoDB's 254-byte
/// `_key` limit (issue #180). Keys that were previously storable are
/// unchanged by this sanitization.
///
/// `line` is the symbol's 1-based definition line. It disambiguates symbols
/// that share a qualified name within one file -- e.g. `impl Foo { fn new }` in
/// sibling inline modules, whose qualified name collapses to `Foo::new` (the
/// module prefix is dropped to match the rust-analyzer enrichment key). Both
/// analyzers agree on this line (syn item-span line == rust-analyzer `range`
/// line + 1), so a syn-written vertex and its RA enrichment for the same symbol
/// still derive the same key. See issue #148 and `tests/ra_span_agreement.rs`.
///
/// # Examples
/// ```
/// # use hades_core::db::keys::{file_key, symbol_key};
/// let key = symbol_key("src_lib_rs", "Config::new", 12);
/// assert!(key.starts_with("src_lib_rs__Config__new__"));
/// assert_eq!(key.len(), "src_lib_rs__Config__new__".len() + 8);
/// ```
pub fn symbol_key(file_key: &str, qualified_name: &str, line: usize) -> String {
    // Readable prefix: replace :: with __, then keep only characters that are
    // both ArangoDB-legal AND survived the historical sanitizer. ArangoDB `_key`
    // allows ASCII alphanumerics plus `_ - : . @ ( ) + , = ; $ ! * ' %`; of
    // those, `: ' ( ) ,` (and space, `< > "`) were already replaced before
    // issue #180, so they stay replaced to keep existing keys byte-identical.
    // Everything else — `& [ ] # { }`, non-ASCII, etc. — maps to `_` because it
    // would 400 the import with error 1221 (illegal document key). Characters
    // in the keep-set could never produce an illegal key, and characters
    // outside it could never have been stored, so no stored key changes.
    let mut readable: String = qualified_name
        .replace("::", "__")
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric()
                || matches!(
                    c,
                    '_' | '-' | '.' | '@' | '+' | '=' | ';' | '$' | '!' | '*' | '%'
                )
            {
                c
            } else {
                '_'
            }
        })
        .collect();

    // Deterministic 8-char hex hash of the qualified name plus the definition
    // line. The line is what makes two same-qualified-name symbols in one file
    // (sibling-module impl methods, #148) resolve to distinct keys.
    let mut hasher = Sha256::new();
    hasher.update(qualified_name.as_bytes());
    hasher.update(b"\n");
    // Fixed-width u64 encoding so keys are identical across architectures
    // (usize::to_le_bytes() is 8 bytes on 64-bit, 4 on 32-bit).
    hasher.update((line as u64).to_le_bytes());

    // ArangoDB caps `_key` at 254 bytes. Generic-heavy qualified names (deep
    // candle/tensor impls) can exceed it, which is the same 1221 error as an
    // illegal character. The readable prefix is cosmetic — the hash carries
    // uniqueness — so truncate it to whatever budget the file_key leaves.
    // All kept characters are ASCII, so byte and char counts agree.
    const MAX_KEY_LEN: usize = 254;
    const OVERHEAD: usize = 2 + 2 + 8; // two "__" separators + hash8

    // A file_key so long it leaves no room even for an empty readable prefix
    // could never have produced a storable key (the full key always exceeded
    // 254 bytes → error 1221), so re-deriving is migration-free. Truncate the
    // prefix and fold the full file_key into the hash so two distinct overlong
    // file_keys sharing a truncated prefix cannot collide.
    let fk_budget = MAX_KEY_LEN - OVERHEAD;
    let fk: &str = if file_key.len() > fk_budget {
        hasher.update(b"\n");
        hasher.update(file_key.as_bytes());
        // file_key may contain non-ASCII (it only maps `.` and `/`), so back
        // the cut off to a char boundary rather than slicing blindly.
        let mut cut = fk_budget;
        while !file_key.is_char_boundary(cut) {
            cut -= 1;
        }
        &file_key[..cut]
    } else {
        file_key
    };

    readable.truncate(MAX_KEY_LEN.saturating_sub(fk.len() + OVERHEAD));

    let digest = hasher.finalize();
    let hash8 = hex8(&digest);

    format!("{fk}__{readable}__{hash8}")
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
        let key = symbol_key("src_lib_rs", "Config::new", 12);
        assert!(key.starts_with("src_lib_rs__Config__new__"), "key: {key}");
        // Hash suffix is 8 hex chars.
        let suffix = key.strip_prefix("src_lib_rs__Config__new__").unwrap();
        assert_eq!(suffix.len(), 8, "hash suffix: {suffix}");

        let key2 = symbol_key("src_lib_rs", "Display for Config", 5);
        assert!(
            key2.starts_with("src_lib_rs__Display_for_Config__"),
            "key: {key2}"
        );

        let key3 = symbol_key("src_lib_rs", "Vec<String>", 1);
        assert!(key3.starts_with("src_lib_rs__Vec_String___"), "key: {key3}");
    }

    #[test]
    fn test_symbol_key_deterministic() {
        // Same input (including line) → same key.
        let a = symbol_key("src_lib_rs", "Config::new", 12);
        let b = symbol_key("src_lib_rs", "Config::new", 12);
        assert_eq!(a, b);
    }

    #[test]
    fn test_symbol_key_line_disambiguates() {
        // #148: two impl methods in sibling inline modules collapse to the same
        // qualified name (`Cfg::build`) but sit at different definition lines.
        // The line must make their keys distinct so neither overwrites the
        // other on insert.
        let a = symbol_key("src_lib_rs", "Cfg::build", 19);
        let b = symbol_key("src_lib_rs", "Cfg::build", 26);
        assert_ne!(a, b, "same qualified name, different line must not collide");
        // The readable prefix is identical; only the hash suffix differs.
        assert!(a.starts_with("src_lib_rs__Cfg__build__"));
        assert!(b.starts_with("src_lib_rs__Cfg__build__"));
        // Same line → same key (so syn and RA still agree on the same symbol).
        assert_eq!(a, symbol_key("src_lib_rs", "Cfg::build", 19));
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
        let a = symbol_key("src_lib_rs", "Vec<T>", 1);
        let b = symbol_key("src_lib_rs", "Vec_T_", 1);
        assert_ne!(a, b, "should not collide: a={a}, b={b}");
    }

    /// True iff `c` may appear in an ArangoDB document `_key`.
    fn arango_key_legal(c: char) -> bool {
        c.is_ascii_alphanumeric()
            || matches!(
                c,
                '_' | '-'
                    | ':'
                    | '.'
                    | '@'
                    | '('
                    | ')'
                    | '+'
                    | ','
                    | '='
                    | ';'
                    | '$'
                    | '!'
                    | '*'
                    | '\''
                    | '%'
            )
    }

    #[test]
    fn test_symbol_key_sanitizes_arango_illegal_chars() {
        // #180: rust-analyzer qualified names from generic/impl-heavy code
        // carry characters ArangoDB rejects in `_key` — `&` (references),
        // `[ ]` (slices/arrays), `#` (closure disambiguators), `{ }`.
        // Every one must be sanitized or the atomic import 400s (error 1221).
        for qname in [
            "Module for &Tensor",
            "Index<usize> for [f32]",
            "<&T as Iterator>::next",
            "outer::{closure#0}",
            "impl Deref for Box<[u8; 32]>",
            "unicode_τ::λ",
        ] {
            let key = symbol_key("src_lib_rs", qname, 1);
            for c in key.chars() {
                assert!(
                    arango_key_legal(c),
                    "illegal char {c:?} in key {key:?} for qualified name {qname:?}"
                );
            }
        }
    }

    #[test]
    fn test_symbol_key_previously_storable_keys_unchanged() {
        // The #180 sanitizer must not move any key that was storable before it:
        // stored graphs address symbols by `_key`, so a changed derivation
        // orphans existing documents. Pin the exact pre-#180 outputs.
        assert!(
            symbol_key("src_lib_rs", "Config::new", 12).starts_with("src_lib_rs__Config__new__")
        );
        assert!(
            symbol_key("src_go", "pkg.Type.Method", 3).starts_with("src_go__pkg.Type.Method__"),
            "gopls dotted names passed through before #180 and must still"
        );
        assert!(
            symbol_key("src_lib_rs", "Vec<String>", 1).starts_with("src_lib_rs__Vec_String___"),
            "chars in the historical denylist must sanitize exactly as before"
        );
    }

    #[test]
    fn test_symbol_key_respects_arango_length_limit() {
        // ArangoDB caps `_key` at 254 bytes; a monster generic qualified name
        // must truncate the readable prefix, not produce an illegal key.
        let long_name = format!("Module for {}", "Wrapper<".repeat(60));
        let key = symbol_key("crates_weaver-spu_src_forward_rs", &long_name, 7);
        assert!(key.len() <= 254, "key too long: {} bytes", key.len());

        // Uniqueness must survive truncation: two long names sharing a 254-byte
        // prefix still differ via the hash suffix.
        let a = symbol_key("f", &format!("{}A", "x".repeat(400)), 1);
        let b = symbol_key("f", &format!("{}B", "x".repeat(400)), 1);
        assert_ne!(a, b, "hash suffix must disambiguate truncated prefixes");
        assert!(a.ends_with(|c: char| c.is_ascii_hexdigit()));
    }

    #[test]
    fn test_symbol_key_overlong_file_key() {
        // A pathologically deep path can push file_key alone past the 254-byte
        // budget; the key must still respect the cap, and two distinct overlong
        // file_keys sharing a truncated prefix must not collide (the full
        // file_key is folded into the hash in that case).
        let shared = "d".repeat(300);
        let a = symbol_key(&format!("{shared}_a_rs"), "Config::new", 1);
        let b = symbol_key(&format!("{shared}_b_rs"), "Config::new", 1);
        assert!(a.len() <= 254, "key too long: {} bytes", a.len());
        assert!(b.len() <= 254, "key too long: {} bytes", b.len());
        assert_ne!(a, b, "overlong file_keys must not collide after truncation");

        // Non-ASCII file_key must not panic on the boundary-safe cut.
        let unicode_fk = "π".repeat(200);
        let k = symbol_key(&unicode_fk, "Config::new", 1);
        assert!(k.len() <= 254);
    }
}
