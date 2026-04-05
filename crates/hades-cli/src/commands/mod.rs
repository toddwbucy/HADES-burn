//! CLI subcommand definitions.
//!
//! Each module defines a clap `Subcommand` enum mirroring the
//! corresponding Python Typer command group in HADES.

pub mod arxiv;
pub mod arxiv_sync;
pub mod codebase;
pub mod codebase_ingest;
pub mod codebase_mgmt;
pub mod daemon;
pub mod db;
pub mod db_graph;
pub mod db_read;
pub mod db_write;
pub mod embed;
pub mod embed_mgmt;
pub mod graph_embed;
pub mod graph_embed_query;
pub mod graph_embed_train;
pub mod ingest;
pub mod smell;
pub mod smell_mgmt;
pub mod system;
pub mod task;
pub mod task_mgmt;
