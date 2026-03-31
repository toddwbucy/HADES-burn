//! CLI subcommand definitions.
//!
//! Each module defines a clap `Subcommand` enum mirroring the
//! corresponding Python Typer command group in HADES.

pub mod arxiv;
pub mod codebase;
pub mod db;
pub mod embed;
pub mod graph_embed;
pub mod ingest;
pub mod smell;
pub mod task;
