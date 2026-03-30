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
}

// ---------------------------------------------------------------------------
// Static profile registry
// ---------------------------------------------------------------------------

static ARXIV: CollectionProfile = CollectionProfile {
    metadata: "arxiv_metadata",
    chunks: "arxiv_abstract_chunks",
    embeddings: "arxiv_abstract_embeddings",
};

static SYNC: CollectionProfile = CollectionProfile {
    metadata: "arxiv_papers",
    chunks: "arxiv_abstracts",
    embeddings: "arxiv_embeddings",
};

static DEFAULT: CollectionProfile = CollectionProfile {
    metadata: "documents",
    chunks: "chunks",
    embeddings: "embeddings",
};

static ALL_PROFILES: [(&str, &CollectionProfile); 3] = [
    ("arxiv", &ARXIV),
    ("sync", &SYNC),
    ("default", &DEFAULT),
];

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
        // Without HADES_DEFAULT_COLLECTION set, should return arxiv
        let p = CollectionProfile::default_profile();
        assert_eq!(p.metadata, "arxiv_metadata");
    }

    #[test]
    fn test_all_profiles() {
        let all = CollectionProfile::all();
        assert_eq!(all.len(), 3);
        let names: Vec<&str> = all.iter().map(|(n, _)| *n).collect();
        assert!(names.contains(&"arxiv"));
        assert!(names.contains(&"sync"));
        assert!(names.contains(&"default"));
    }
}
