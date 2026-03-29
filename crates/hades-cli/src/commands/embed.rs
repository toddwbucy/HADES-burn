//! `hades embed` subcommands — embedding generation and service management.

use clap::Subcommand;

#[derive(Debug, Subcommand)]
pub enum EmbedCmd {
    /// Generate an embedding for the given text.
    Text {
        /// Text to embed.
        text: String,

        /// GPU device index.
        #[arg(short = 'g', long)]
        gpu: Option<u32>,

        /// Output format (json, raw).
        #[arg(short = 'f', long, default_value = "json")]
        format: String,
    },

    /// Embedding service management.
    #[command(subcommand)]
    Service(EmbedServiceCmd),

    /// GPU device management.
    #[command(subcommand)]
    Gpu(EmbedGpuCmd),
}

// ── embed service ──────────────────────────────────────────────────

#[derive(Debug, Subcommand)]
pub enum EmbedServiceCmd {
    /// Show embedding service status.
    Status,

    /// Start the embedding service.
    Start {
        /// GPU device index.
        #[arg(short = 'g', long)]
        gpu: Option<u32>,

        /// Run in foreground (don't daemonize).
        #[arg(long)]
        foreground: bool,
    },

    /// Stop the embedding service.
    Stop,
}

// ── embed gpu ──────────────────────────────────────────────────────

#[derive(Debug, Subcommand)]
pub enum EmbedGpuCmd {
    /// Show GPU status and memory usage.
    Status,

    /// List available GPU devices.
    List,
}
