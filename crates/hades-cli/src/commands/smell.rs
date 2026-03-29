//! `hades smell` subcommands — compliance checking.

use std::path::PathBuf;

use clap::Subcommand;

#[derive(Debug, Subcommand)]
pub enum SmellCmd {
    /// Check code for compliance smells.
    Check {
        /// Path to file or directory.
        path: PathBuf,

        /// Output format (text, json).
        #[arg(short = 'f', long, default_value = "text")]
        format: String,

        /// Verbose output.
        #[arg(short = 'V', long)]
        verbose: bool,
    },

    /// Verify compliance status of a path.
    Verify {
        /// Path to file or directory.
        path: PathBuf,

        /// Compliance claims to verify (CS-NN format).
        #[arg(long)]
        claims: Vec<String>,
    },

    /// Generate a compliance report.
    Report {
        /// Path to file or directory.
        path: PathBuf,

        /// Output file path.
        #[arg(short = 'o', long)]
        output: Option<PathBuf>,

        /// Output format (text, json, html).
        #[arg(short = 'f', long, default_value = "text")]
        format: String,
    },
}
