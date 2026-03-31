//! Document chunking engine.
//!
//! Pure Rust text chunking with multiple strategies.  Each strategy
//! implements the [`ChunkingStrategy`] trait, producing [`TextChunk`]
//! values with character offsets for source attribution.
//!
//! Late-chunking support (mean-pooling token-level embeddings per chunk)
//! is in the [`late`] submodule.

mod late;
mod strategies;

pub use late::{late_chunk_embeddings, LateCunkConfig};
pub use strategies::{SentenceChunking, SlidingWindowChunking, TokenChunking};

/// A chunk of text with positional metadata.
#[derive(Debug, Clone, PartialEq)]
pub struct TextChunk {
    /// The chunk text content.
    pub text: String,
    /// Start character offset in the original document (inclusive).
    pub start_char: usize,
    /// End character offset in the original document (exclusive).
    pub end_char: usize,
    /// Zero-based chunk index within the document.
    pub chunk_index: usize,
    /// Total number of chunks produced from the document.
    ///
    /// Set to 0 during chunking; callers fill in after all chunks
    /// are produced (since total is unknown until iteration completes).
    pub total_chunks: usize,
}

/// Strategy for splitting text into chunks.
pub trait ChunkingStrategy {
    /// Split `text` into a vector of [`TextChunk`]s.
    ///
    /// The returned chunks have `total_chunks` already filled in.
    fn chunk(&self, text: &str) -> Vec<TextChunk>;
}
