//! Collection profiles mapping logical names to physical ArangoDB collections.
//!
//! Each profile groups three collections: metadata, chunks, and embeddings.
//! Matches the Python `PROFILES` dict in `core/database/collections.py`.

use std::env;

/// A set of three physical ArangoDB collection names for a logical collection.
#[derive(Debug, Clone, serde::Serialize)]
pub struct CollectionProfile {
    /// Document metadata collection (e.g. `arxiv_metadata`).
    pub metadata: &'static str,
    /// Text chunk collection (e.g. `arxiv_abstract_chunks`).
    pub chunks: &'static str,
    /// Embedding vector collection (e.g. `arxiv_abstract_embeddings`).
    pub embeddings: &'static str,
    /// Field name in chunks/embeddings that references the parent metadata
    /// document key (e.g. `"paper_key"` for arxiv, `"file_key"` for codebase).
    pub foreign_key: &'static str,
}

// ---------------------------------------------------------------------------
// Static profile registry
// ---------------------------------------------------------------------------

static ARXIV: CollectionProfile = CollectionProfile {
    metadata: "arxiv_metadata",
    chunks: "arxiv_abstract_chunks",
    embeddings: "arxiv_abstract_embeddings",
    foreign_key: "paper_key",
};

static SYNC: CollectionProfile = CollectionProfile {
    metadata: "arxiv_papers",
    chunks: "arxiv_abstracts",
    embeddings: "arxiv_embeddings",
    foreign_key: "paper_key",
};

static DEFAULT: CollectionProfile = CollectionProfile {
    metadata: "documents",
    chunks: "chunks",
    embeddings: "embeddings",
    foreign_key: "paper_key",
};

static ALL_PROFILES: [(&str, &CollectionProfile); 4] = [
    ("arxiv", &ARXIV),
    ("sync", &SYNC),
    ("default", &DEFAULT),
    ("codebase", &CODEBASE_PROFILE),
];

// ---------------------------------------------------------------------------
// Codebase-specific collections
// ---------------------------------------------------------------------------

/// Base collection profile for codebase ingestion (files → chunks → embeddings).
static CODEBASE_PROFILE: CollectionProfile = CollectionProfile {
    metadata: "codebase_files",
    chunks: "codebase_chunks",
    embeddings: "codebase_embeddings",
    foreign_key: "file_key",
};

/// Extended collection set for codebase ingestion.
///
/// Beyond the standard metadata/chunks/embeddings triple, codebase
/// analysis produces symbol-level metadata and graph edges (imports,
/// defines, calls, implements).
#[derive(Debug, Clone, serde::Serialize)]
pub struct CodebaseCollections {
    /// File-level metadata (language, metrics, symbol_hash).
    pub files: &'static str,
    /// AST-aligned text chunks.
    pub chunks: &'static str,
    /// Embedding vectors per chunk.
    pub embeddings: &'static str,
    /// Symbol-level metadata (name, kind, span, parent file).
    pub symbols: &'static str,
    /// Graph edges (imports, defines, calls, implements).
    /// This is an **edge** collection (type 3) in ArangoDB.
    pub edges: &'static str,
}

/// The singleton codebase collection set.
pub static CODEBASE: CodebaseCollections = CodebaseCollections {
    files: "codebase_files",
    chunks: "codebase_chunks",
    embeddings: "codebase_embeddings",
    symbols: "codebase_symbols",
    edges: "codebase_edges",
};

impl CodebaseCollections {
    /// All collection names, in order suitable for creation.
    ///
    /// Returns `(name, collection_type)` pairs where type 2 = document,
    /// 3 = edge.
    pub fn all_collections(&self) -> [(&str, u32); 5] {
        [
            (self.files, 2),
            (self.chunks, 2),
            (self.embeddings, 2),
            (self.symbols, 2),
            (self.edges, 3),
        ]
    }
}

impl CollectionProfile {
    /// Look up a profile by name.
    pub fn get(name: &str) -> Option<&'static CollectionProfile> {
        ALL_PROFILES
            .iter()
            .find(|(n, _)| *n == name)
            .map(|(_, p)| *p)
    }

    /// Get the default profile.
    ///
    /// Reads `HADES_DEFAULT_COLLECTION` env var, falling back to `"arxiv"`.
    pub fn default_profile() -> &'static CollectionProfile {
        let name = env::var("HADES_DEFAULT_COLLECTION").unwrap_or_else(|_| "arxiv".to_string());
        Self::get(&name).unwrap_or(&ARXIV)
    }

    /// Look up a profile by its metadata collection name.
    ///
    /// Used by purge to find related chunks/embeddings collections
    /// given a document's metadata collection.
    pub fn find_by_metadata(metadata_col: &str) -> Option<&'static CollectionProfile> {
        ALL_PROFILES
            .iter()
            .find(|(_, p)| p.metadata == metadata_col)
            .map(|(_, p)| *p)
    }

    /// All registered profiles as `(name, profile)` pairs.
    pub fn all() -> &'static [(&'static str, &'static CollectionProfile)] {
        &ALL_PROFILES
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arxiv_profile() {
        let p = CollectionProfile::get("arxiv").unwrap();
        assert_eq!(p.metadata, "arxiv_metadata");
        assert_eq!(p.chunks, "arxiv_abstract_chunks");
        assert_eq!(p.embeddings, "arxiv_abstract_embeddings");
    }

    #[test]
    fn test_sync_profile() {
        let p = CollectionProfile::get("sync").unwrap();
        assert_eq!(p.metadata, "arxiv_papers");
        assert_eq!(p.chunks, "arxiv_abstracts");
        assert_eq!(p.embeddings, "arxiv_embeddings");
    }

    #[test]
    fn test_default_profile() {
        let p = CollectionProfile::get("default").unwrap();
        assert_eq!(p.metadata, "documents");
        assert_eq!(p.chunks, "chunks");
        assert_eq!(p.embeddings, "embeddings");
    }

    #[test]
    fn test_nonexistent_profile() {
        assert!(CollectionProfile::get("nonexistent").is_none());
    }

    #[test]
    fn test_default_profile_fallback() {
        // Save, clear, test, restore — ensures determinism regardless
        // of what HADES_DEFAULT_COLLECTION is set to in the environment.
        let saved = env::var("HADES_DEFAULT_COLLECTION").ok();
        // SAFETY: unit tests in this module do not concurrently read
        // this env var from other threads.
        unsafe { env::remove_var("HADES_DEFAULT_COLLECTION") };

        let result = std::panic::catch_unwind(|| {
            let p = CollectionProfile::default_profile();
            assert_eq!(p.metadata, "arxiv_metadata");
        });

        // Restore before propagating any panic
        match saved {
            Some(val) => unsafe { env::set_var("HADES_DEFAULT_COLLECTION", val) },
            None => unsafe { env::remove_var("HADES_DEFAULT_COLLECTION") },
        }

        result.unwrap();
    }

    #[test]
    fn test_all_profiles() {
        let all = CollectionProfile::all();
        assert_eq!(all.len(), 4);
        let names: Vec<&str> = all.iter().map(|(n, _)| *n).collect();
        assert!(names.contains(&"arxiv"));
        assert!(names.contains(&"sync"));
        assert!(names.contains(&"default"));
        assert!(names.contains(&"codebase"));
    }

    #[test]
    fn test_codebase_profile() {
        let p = CollectionProfile::get("codebase").unwrap();
        assert_eq!(p.metadata, "codebase_files");
        assert_eq!(p.chunks, "codebase_chunks");
        assert_eq!(p.embeddings, "codebase_embeddings");
    }

    #[test]
    fn test_codebase_collections() {
        let cols = CODEBASE.all_collections();
        assert_eq!(cols.len(), 5);
        // First 4 are document collections (type 2).
        assert!(cols[..4].iter().all(|(_, t)| *t == 2));
        // Last one (edges) is an edge collection (type 3).
        assert_eq!(cols[4], ("codebase_edges", 3));
    }

    #[test]
    fn test_find_by_metadata_arxiv() {
        let p = CollectionProfile::find_by_metadata("arxiv_metadata").unwrap();
        assert_eq!(p.chunks, "arxiv_abstract_chunks");
        assert_eq!(p.embeddings, "arxiv_abstract_embeddings");
    }

    #[test]
    fn test_find_by_metadata_codebase() {
        let p = CollectionProfile::find_by_metadata("codebase_files").unwrap();
        assert_eq!(p.chunks, "codebase_chunks");
    }

    #[test]
    fn test_find_by_metadata_unknown() {
        assert!(CollectionProfile::find_by_metadata("nonexistent").is_none());
    }
}
