//! Subprocess dispatch to the Python `hades` CLI.
//!
//! During the strangler-fig migration, every command starts as a thin
//! pass-through to the existing Python implementation.  As Rust modules
//! mature, individual commands switch from `dispatch()` to native code.

use std::process::{Command, ExitStatus};

use anyhow::{Context, Result};

/// Resolve the path to the Python HADES CLI.
///
/// Checks `HADES_PYTHON_BIN` env var first, then falls back to the
/// known production venv path.  This avoids recursion now that the
/// Rust binary is also named `hades`.
fn python_hades_path() -> std::ffi::OsString {
    std::env::var_os("HADES_PYTHON_BIN").unwrap_or_else(|| {
        let mut p = std::path::PathBuf::from(
            std::env::var_os("HOME").unwrap_or_default(),
        );
        p.push(".local/share/hades-stable/venv/bin/hades");
        p.into_os_string()
    })
}

/// Build and execute a Python `hades` subprocess with the given arguments.
///
/// Inherits stdin/stdout/stderr so the user sees output directly.
/// Returns the child's exit status so the caller can propagate it.
pub fn dispatch(args: &[&str]) -> Result<ExitStatus> {
    let bin = python_hades_path();
    let status = Command::new(&bin)
        .args(args)
        .stdin(std::process::Stdio::inherit())
        .stdout(std::process::Stdio::inherit())
        .stderr(std::process::Stdio::inherit())
        .status()
        .with_context(|| format!(
            "failed to execute Python HADES CLI at {:?} — \
             set HADES_PYTHON_BIN to override",
            bin,
        ))?;
    Ok(status)
}

/// Dispatch with global options prepended.
///
/// Inserts `--db <database>` and/or `--gpu <device>` before the
/// subcommand arguments when they are set.
pub fn dispatch_with_globals(
    database: Option<&str>,
    gpu: Option<u32>,
    args: &[&str],
) -> Result<ExitStatus> {
    let mut full_args: Vec<&str> = Vec::new();

    // Global options must come before subcommands in Typer
    let db_str;
    if let Some(db) = database {
        full_args.push("--db");
        db_str = db.to_string();
        full_args.push(&db_str);
    }

    let gpu_str;
    if let Some(g) = gpu {
        full_args.push("--gpu");
        gpu_str = g.to_string();
        full_args.push(&gpu_str);
    }

    full_args.extend_from_slice(args);
    dispatch(&full_args)
}
