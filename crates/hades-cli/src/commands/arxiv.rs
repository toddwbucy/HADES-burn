//! `hades arxiv` subcommands.

use clap::Subcommand;

#[derive(Debug, Subcommand)]
pub enum ArxivCmd {
    /// Sync abstracts from arXiv API into the knowledge base.
    Sync {
        /// Run in batch mode (non-interactive).
        #[arg(short = 'b', long)]
        batch: bool,

        /// Maximum number of results to fetch.
        #[arg(short = 'n', long, default_value_t = 100)]
        max_results: u32,

        /// Categories to sync (e.g. cs.AI, cs.LG).
        #[arg(short = 'c', long)]
        categories: Vec<String>,

        /// Number of days to look back.
        #[arg(long)]
        lookback_days: Option<u32>,
    },

    /// Show sync status and history.
    SyncStatus {
        /// Maximum number of history entries to show.
        #[arg(short = 'n', long, default_value_t = 10)]
        limit: u32,
    },
}
