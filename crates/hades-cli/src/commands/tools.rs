//! `hades tools` — external analyzer inventory and health.
//!
//! The enrichment analyzers (rust-analyzer, gopls) are external binaries, and
//! "is it installed" has no directory-independent answer on a rustup system:
//! the shim resolves per rust-toolchain.toml, so the same command can work in
//! one directory and die in another (#164, #167). `tools status` answers the
//! question the way ingest will actually ask it: resolve each analyzer through
//! the same precedence ingest uses (config/env override, then PATH) and probe
//! it FROM a stated workspace directory.
//!
//! `tools install` (HADES-managed, version-pinned binaries) is #167 follow-on
//! work; this subcommand is the inventory/probe half.

use std::path::PathBuf;

use anyhow::{Context, Result};
use serde_json::json;

use hades_core::code::lsp::preflight_binary;
use hades_core::config::HadesConfig;

use super::output::{self, OutputFormat};

#[derive(Debug, clap::Subcommand)]
pub enum ToolsCmd {
    /// Report each analyzer's resolution and a live probe from a workspace.
    Status {
        /// Directory to probe FROM (the shim resolves per-directory). Defaults
        /// to the current directory — pass the ingest root you intend to use.
        #[arg(long)]
        workspace: Option<PathBuf>,
    },
}

pub fn run_status(config: &HadesConfig, workspace: Option<PathBuf>) -> Result<()> {
    let ws = match workspace {
        Some(w) => w
            .canonicalize()
            .with_context(|| format!("workspace not found: {}", w.display()))?,
        None => std::env::current_dir().context("cannot read current directory")?,
    };

    let mut all_ok = true;
    let mut report = serde_json::Map::new();
    for (name, configured) in [
        ("rust-analyzer", config.analyzers.rust_analyzer.as_deref()),
        ("gopls", config.analyzers.gopls.as_deref()),
    ] {
        let (command, source) = match configured {
            Some(p) => (p.to_string(), "config/env"),
            None => (name.to_string(), "PATH"),
        };
        let entry = match preflight_binary(&command, &ws) {
            Ok(version) => json!({
                "command": command, "source": source, "ok": true, "version": version,
            }),
            Err(e) => {
                all_ok = false;
                json!({
                    "command": command, "source": source, "ok": false,
                    "error": e.to_string(),
                })
            }
        };
        report.insert(name.to_string(), entry);
    }
    // libclang is dlopen'd by the clang crate at analysis time, not spawned,
    // so a spawn probe cannot exercise it. Reported for inventory honesty.
    report.insert(
        "libclang".to_string(),
        json!({ "source": "dlopen at analysis time", "ok": null,
                "note": "not probed by tools status; C/C++/CUDA analysis degrades gracefully when absent" }),
    );

    output::print_output_with_success(
        "tools.status",
        json!({ "workspace": ws.display().to_string(), "analyzers": report }),
        &OutputFormat::Json,
        all_ok,
    );
    if !all_ok {
        anyhow::bail!(
            "one or more analyzers failed the probe from {}",
            ws.display()
        );
    }
    Ok(())
}
