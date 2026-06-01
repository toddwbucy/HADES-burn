//! `hades codebase` subcommands.

use std::path::PathBuf;

use clap::Subcommand;

#[derive(Debug, Subcommand)]
pub enum CodebaseCmd {
    /// Ingest source code into the knowledge graph.
    Ingest {
        /// Path to file or directory to ingest.
        path: PathBuf,

        /// Programming language override (auto-detected if omitted).
        #[arg(short = 'l', long)]
        language: Option<String>,

        /// Run in batch mode.
        #[arg(short = 'b', long)]
        batch: bool,
    },

    /// Update an existing code graph node.
    Update {
        /// Path to file or directory to update.
        path: PathBuf,
    },

    /// Show code ingestion statistics.
    Stats,

    /// Validate codebase graph invariants (ontology spec §10).
    Validate,

    /// Remove orphan symbols and dangling edges left by pre-cascade deletes.
    PruneOrphans {
        /// Report what would be deleted without modifying the graph.
        #[arg(long)]
        dry_run: bool,
    },
}
