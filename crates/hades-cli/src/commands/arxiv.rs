//! `hades arxiv` subcommands.

use clap::Subcommand;

#[derive(Debug, Subcommand)]
pub enum ArxivCmd {
    /// Sync abstracts from arXiv API into the knowledge base.
    Sync {
        /// Start date in YYYY-MM-DD format (default: 7 days ago).
        #[arg(short = 'f', long = "from")]
        from_date: Option<String>,

        /// Comma-separated arXiv categories to sync (e.g. cs.AI,cs.LG).
        #[arg(short = 'c', long)]
        categories: Option<String>,

        /// Maximum number of papers to sync.
        #[arg(short = 'm', long = "max", default_value_t = 1000)]
        max_results: u32,

        /// Embedding batch size (tuned for GPU memory).
        #[arg(short = 'b', long, default_value_t = 8)]
        batch_size: u32,

        /// Sync only papers newer than last sync watermark.
        #[arg(short = 'i', long)]
        incremental: bool,
    },

    /// Show sync status and history.
    SyncStatus {
        /// Maximum number of history entries to show.
        #[arg(short = 'n', long, default_value_t = 10)]
        limit: u32,
    },
}
