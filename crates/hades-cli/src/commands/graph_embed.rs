//! `hades graph-embed` subcommands — graph embedding operations.

use clap::Subcommand;

#[derive(Debug, Subcommand)]
pub enum GraphEmbedCmd {
    /// Train graph embeddings.
    Train {
        /// Number of training epochs.
        #[arg(long, default_value_t = 100)]
        epochs: u32,

        /// Embedding dimension.
        #[arg(long, default_value_t = 128)]
        dimension: u32,
    },

    /// Generate embedding for a specific node.
    Embed {
        /// Node ID to embed.
        node_id: String,
    },

    /// Find nearest neighbors of a node in embedding space.
    Neighbors {
        /// Node ID to query.
        node_id: String,

        /// Number of neighbors to return.
        #[arg(short = 'n', long, default_value_t = 10)]
        limit: u32,
    },

    /// Update graph embeddings incrementally.
    Update,
}
