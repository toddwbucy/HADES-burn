//! `hades graph-embed` subcommands — graph embedding operations.

use clap::Subcommand;

#[derive(Debug, Subcommand)]
pub enum GraphEmbedCmd {
    /// Train RGCN graph embeddings via Persephone training service.
    Train {
        /// Maximum number of training epochs.
        #[arg(long, default_value_t = 200)]
        epochs: u32,

        /// Output structural embedding dimension.
        #[arg(long, default_value_t = 128)]
        dimension: u32,

        /// Hidden layer dimension.
        #[arg(long, default_value_t = 256)]
        hidden_dim: u32,

        /// Number of RGCN basis matrices.
        #[arg(long, default_value_t = 21)]
        num_bases: u32,

        /// Dropout rate.
        #[arg(long, default_value_t = 0.2)]
        dropout: f32,

        /// Learning rate.
        #[arg(long, default_value_t = 0.01)]
        lr: f32,

        /// L2 regularization weight decay.
        #[arg(long, default_value_t = 5e-4)]
        weight_decay: f32,

        /// Early stopping patience (epochs without val improvement).
        #[arg(long, default_value_t = 20)]
        patience: u32,

        /// Validation split ratio.
        #[arg(long, default_value_t = 0.1)]
        val_ratio: f64,

        /// Test split ratio.
        #[arg(long, default_value_t = 0.1)]
        test_ratio: f64,

        /// Negative-to-positive sampling ratio.
        #[arg(long, default_value_t = 1.0)]
        neg_ratio: f64,

        /// Export embeddings to a different database after training.
        /// If omitted, exports to the current database.
        #[arg(long)]
        export_to: Option<String>,

        /// Checkpoint directory for model snapshots.
        #[arg(long, default_value = "/tmp/hades-train")]
        checkpoint_dir: String,

        /// Skip embedding export after training.
        #[arg(long)]
        no_export: bool,
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
