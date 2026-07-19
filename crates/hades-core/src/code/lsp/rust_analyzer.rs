//! rust-analyzer configuration for the shared LSP session.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use serde_json::Value;

use super::LspError;
use super::session::{LanguageServer, LspSession};

pub struct RustAnalyzer;

impl LanguageServer for RustAnalyzer {
    const NAME: &'static str = "rust-analyzer";
    const LANGUAGE_ID: &'static str = "rust";

    fn find_binary() -> Result<String, LspError> {
        find_on_path("rust-analyzer")
            .or_else(|| home_dir().map(|home| home.join(".cargo/bin/rust-analyzer")))
            .filter(|path| path.exists())
            .map(|path| path.to_string_lossy().into_owned())
            .ok_or_else(|| {
                LspError::NotFound(
                    "rust-analyzer; install with `rustup component add rust-analyzer`".into(),
                )
            })
    }

    fn validate_root(root: &Path) -> Result<PathBuf, LspError> {
        let root = root.canonicalize()?;
        if !root.join("Cargo.toml").exists() {
            return Err(LspError::InvalidWorkspace(format!(
                "no Cargo.toml at {}",
                root.display()
            )));
        }
        Ok(root)
    }

    fn initialization_options() -> Value {
        serde_json::json!({ "workDoneProgress": true })
    }

    fn progress_is_ready(value: &Value) -> bool {
        value.get("kind").and_then(Value::as_str) == Some("end")
    }
}

pub type RustAnalyzerSession = LspSession<RustAnalyzer>;

pub fn find_crate_root(file_path: &Path) -> Option<PathBuf> {
    find_workspace_root(file_path, &["Cargo.toml"])
}

pub fn group_files_by_crate(rs_files: &[PathBuf]) -> HashMap<PathBuf, Vec<PathBuf>> {
    group_files(rs_files, find_crate_root, "Cargo.toml")
}

pub(crate) fn find_workspace_root(file_path: &Path, markers: &[&str]) -> Option<PathBuf> {
    let mut dir = if file_path.is_file() {
        file_path.parent()?.to_path_buf()
    } else {
        file_path.to_path_buf()
    };
    loop {
        if markers.iter().any(|marker| dir.join(marker).exists()) {
            return Some(dir);
        }
        if !dir.pop() {
            return None;
        }
    }
}

pub(crate) fn group_files(
    files: &[PathBuf],
    find_root: fn(&Path) -> Option<PathBuf>,
    marker: &str,
) -> HashMap<PathBuf, Vec<PathBuf>> {
    let mut groups = HashMap::new();
    for path in files {
        if let Some(root) = find_root(path) {
            groups
                .entry(root)
                .or_insert_with(Vec::new)
                .push(path.clone());
        } else {
            tracing::warn!(path = %path.display(), marker, "no language workspace found");
        }
    }
    groups
}

pub(crate) fn find_on_path(binary: &str) -> Option<PathBuf> {
    std::process::Command::new("which")
        .arg(binary)
        .output()
        .ok()
        .filter(|output| output.status.success())
        .and_then(|output| {
            let path = String::from_utf8_lossy(&output.stdout).trim().to_string();
            (!path.is_empty()).then(|| PathBuf::from(path))
        })
}

pub(crate) fn home_dir() -> Option<PathBuf> {
    std::env::var_os("HOME").map(PathBuf::from)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_and_groups_crate_files() {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let files = vec![root.join("src/lib.rs"), root.join("src/code/mod.rs")];
        let groups = group_files_by_crate(&files);
        assert_eq!(groups.len(), 1);
        assert_eq!(groups.values().next().unwrap().len(), 2);
    }
}
