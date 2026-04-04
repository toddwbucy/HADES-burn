//! CLI subcommand definitions.
//!
//! Each module defines a clap `Subcommand` enum mirroring the
//! corresponding Python Typer command group in HADES.

pub mod arxiv;
pub mod arxiv_sync;
pub mod codebase;
pub mod codebase_ingest;
pub mod daemon;
pub mod db;
pub mod db_read;
pub mod embed;
pub mod graph_embed;
pub mod graph_embed_query;
pub mod graph_embed_train;
pub mod ingest;
pub mod smell;
pub mod task;
