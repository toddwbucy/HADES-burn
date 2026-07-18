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

        /// Comma-separated extensions to embed without a parser (e.g.
        /// `wgsl,vert`). Files with these extensions are chunked by size and
        /// embedded as features — no symbol/edge extraction. Their file nodes
        /// are merged (existing fields preserved), not overwritten.
        #[arg(long = "unparsed-ext", value_delimiter = ',')]
        unparsed_ext: Vec<String>,

        /// Path to `compile_commands.json` (or its containing directory) for
        /// compiler-grade C/C++/CUDA include, define, standard, and target
        /// configuration. When omitted, source ancestors and `build/` are
        /// searched automatically.
        #[arg(long = "compile-commands")]
        compile_commands: Option<PathBuf>,

        /// Re-ingest each file even if its change-detection digest is
        /// unchanged. This rebuilds the node's symbols, chunks, and embeddings
        /// in place — it does NOT drop the file node or its inbound edges, so
        /// authored bridge edges survive (unlike `db purge`). Use it to refresh
        /// a node whose stored view has drifted from the source.
        #[arg(short = 'f', long = "force", alias = "no-skip")]
        force: bool,
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
