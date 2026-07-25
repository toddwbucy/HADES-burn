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
        /// a node whose stored view has drifted from the source — in particular
        /// after an edit that touched only bodies, signatures, or comments,
        /// which the name-keyed `symbol_hash` cannot see and which
        /// `codebase drift` reports as `changed`.
        ///
        /// If a rebuild drops a symbol that another file points at, those
        /// inbound edges are reported as `dangling_inbound_edges` — not
        /// deleted, since each records a real dependency. Re-ingest the
        /// dependent files to re-resolve them, or run
        /// `hades codebase prune-orphans` to drop them; until then
        /// `codebase validate` will flag them.
        ///
        /// This never permits an analyzer-fidelity downgrade by itself, so a
        /// file whose stored analysis came from a richer analyzer than the one
        /// available now is still skipped — pass `--allow-analysis-downgrade`
        /// as well to refresh it.
        #[arg(short = 'f', long = "force", alias = "no-skip")]
        force: bool,

        /// Permit a lower-fidelity analyzer to replace previously stored
        /// semantic artifacts. This is separate from `--force` so a temporary
        /// analyzer outage cannot silently degrade the graph.
        #[arg(long = "allow-analysis-downgrade")]
        allow_analysis_downgrade: bool,
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

    /// Remove orphaned symbols, chunks, embeddings, and dangling edges.
    ///
    /// Sweeps child records whose owning file node is already gone. To retire a
    /// file node whose *source file* was deleted, use `codebase retire`.
    PruneOrphans {
        /// Report what would be deleted without modifying the graph.
        #[arg(long)]
        dry_run: bool,
    },

    /// Compare the graph against the source tree it describes (read-only).
    ///
    /// `codebase validate` checks only internal consistency and cannot see any
    /// of this.
    ///
    /// Buckets: `stale` (a file node whose source file no longer exists),
    /// `uningested` (a source file with no node), `changed` (a matched file
    /// whose content differs from what was ingested), and `unhandled` (files
    /// under the root ingest has no handler for, with a reason for each).
    ///
    /// `changed.unverifiable` counts matched files that could not be compared at
    /// all, because they were ingested before `content_hash` existed or are no
    /// longer readable as text.
    ///
    /// `clean` is true only when stale, uningested, changed and
    /// `changed.unverifiable` are all zero. `unhandled` does not gate it, since
    /// every repository contains files no analyzer handles.
    ///
    /// `changed` exists because incremental ingest keys on `symbol_hash`, which
    /// covers symbol *names* only: an edited body, signature or comment leaves
    /// it identical, so a plain `codebase ingest` skips the file while its
    /// stored chunks go stale. Refresh those with `codebase ingest --force`.
    ///
    /// Pass the same discovery flags used at ingest time, and the same root —
    /// keys are relative to the ingest root, so a wrong root reports near-total
    /// drift in both directions rather than a small honest number.
    Drift {
        /// Ingest root the graph was built from.
        path: PathBuf,

        /// Programming language override (must match the ingest invocation).
        #[arg(short = 'l', long)]
        language: Option<String>,

        /// Extensions ingested without a parser (must match the ingest
        /// invocation), e.g. `wgsl,vert`.
        #[arg(long = "unparsed-ext", value_delimiter = ',')]
        unparsed_ext: Vec<String>,

        /// List every key instead of truncating. Use this to feed
        /// `codebase retire --from -`.
        #[arg(long)]
        full: bool,
    },

    /// Retire graph nodes whose source files are gone (complement of --force).
    ///
    /// Removes each target's file node, chunks, embeddings, symbols, and every
    /// codebase edge incident on the file or its symbols. Edges in other
    /// collections (authored bridges such as conformance verdicts) are reported
    /// separately and need `--yes`, since they are irreplaceable if the target
    /// list is wrong.
    ///
    /// Targets are always explicit — use `codebase drift` to discover them.
    Retire {
        /// File node key to retire. Repeatable.
        #[arg(long = "file")]
        files: Vec<String>,

        /// Read newline-separated keys from a file (`-` for stdin).
        /// Blank lines and `#` comments are ignored.
        #[arg(long = "from")]
        from: Option<PathBuf>,

        /// Report what would be removed without modifying the graph.
        #[arg(long)]
        dry_run: bool,

        /// Confirm removal of edges outside the codebase collections.
        #[arg(short = 'y', long)]
        yes: bool,
    },
}
