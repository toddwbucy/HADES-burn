//! `hades graph-embed` subcommands — graph embedding operations.

use clap::Subcommand;

#[derive(Debug, Subcommand)]
pub enum GraphEmbedCmd {
    /// Train structural graph embeddings via the HADES training service.
    /// The architecture (transductive RGCN or inductive relational GraphSAGE)
    /// is chosen by the database schema's `model_type`, not a flag.
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

        /// Number of basis matrices for the relation-weight decomposition
        /// (used by both RGCN and the inductive GraphSAGE architectures).
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

        /// Checkpoint directory for model snapshots and the graph IPC file.
        /// The training service runs as a separate user (`hades`) and both
        /// reads and writes files here, so this directory must be writable by
        /// BOTH you and that service — e.g. a directory owned by a shared
        /// group with the setgid bit, or any path both can write. Otherwise
        /// the service fails to write the checkpoint or read the graph file.
        /// The default /tmp/hades-train is ephemeral and cleared on reboot;
        /// for production, use a persistent shared path (e.g. /var/lib/hades/train).
        #[arg(long, default_value = "/tmp/hades-train")]
        checkpoint_dir: String,

        /// Validate every N epochs (default: every epoch).
        #[arg(long, default_value_t = 1)]
        val_every: usize,

        /// Prefetch buffer depth for CPU→GPU pipelining.
        #[arg(long, default_value_t = 2)]
        prefetch_depth: usize,

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

    /// Update graph embeddings incrementally (forward pass only, no retraining).
    Update {
        /// Export embeddings to a different database after update.
        /// If omitted, exports to the current database.
        #[arg(long)]
        export_to: Option<String>,

        /// Checkpoint directory containing the trained model. Must be writable
        /// by both you and the `hades` training-service user (see `train`),
        /// since the service reads the graph IPC file written here.
        #[arg(long, default_value = "/tmp/hades-train")]
        checkpoint_dir: String,

        /// Embed only graph nodes whose destination documents do not yet have
        /// `structural_embedding`. Requires an inductive `hetero_sage` schema
        /// and checkpoint; existing embeddings are left untouched.
        #[arg(long)]
        new_nodes: bool,
    },
}
