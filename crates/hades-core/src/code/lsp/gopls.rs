//! gopls configuration and Go module grouping for semantic analysis.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use serde_json::Value;

use super::LspError;
use super::rust_analyzer::{find_on_path, find_workspace_root, group_files, home_dir};
use super::session::{LanguageServer, LspSession};

pub struct Gopls;

impl LanguageServer for Gopls {
    const NAME: &'static str = "gopls";
    const LANGUAGE_ID: &'static str = "go";

    fn find_binary() -> Result<String, LspError> {
        find_on_path("gopls")
            .or_else(|| home_dir().map(|home| home.join("go/bin/gopls")))
            .filter(|path| path.exists())
            .map(|path| path.to_string_lossy().into_owned())
            .ok_or_else(|| {
                LspError::NotFound(
                    "gopls; install with `go install golang.org/x/tools/gopls@latest`".into(),
                )
            })
    }

    fn validate_root(root: &Path) -> Result<PathBuf, LspError> {
        let root = root.canonicalize()?;
        if !root.join("go.mod").exists() && !root.join("go.work").exists() {
            return Err(LspError::InvalidWorkspace(format!(
                "no go.mod or go.work at {}",
                root.display()
            )));
        }
        Ok(root)
    }

    fn args() -> &'static [&'static str] {
        &["serve"]
    }

    fn initialization_options() -> Value {
        serde_json::json!({
            "symbolMatcher": "FastFuzzy",
            "semanticTokens": true,
            "usePlaceholders": false,
            "completeUnimported": false,
        })
    }

    fn progress_is_ready(value: &Value) -> bool {
        if value.get("kind").and_then(Value::as_str) != Some("end") {
            return false;
        }
        let title = value.get("title").and_then(Value::as_str).unwrap_or("");
        let message = value.get("message").and_then(Value::as_str).unwrap_or("");
        let progress = format!("{title} {message}").to_ascii_lowercase();
        progress.contains("workspace")
            || progress.contains("package")
            || progress.contains("initial")
    }
}

pub type GoplsSession = LspSession<Gopls>;

pub fn find_go_module_root(file_path: &Path) -> Option<PathBuf> {
    find_workspace_root(file_path, &["go.work", "go.mod"])
}

pub fn group_files_by_go_module(go_files: &[PathBuf]) -> HashMap<PathBuf, Vec<PathBuf>> {
    group_files(go_files, find_go_module_root, "go.mod/go.work")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn groups_files_by_nearest_module() {
        let temp = tempfile::tempdir().unwrap();
        std::fs::write(temp.path().join("go.mod"), "module example.test/demo\n").unwrap();
        let file = temp.path().join("main.go");
        std::fs::write(&file, "package main\n").unwrap();
        let groups = group_files_by_go_module(&[file]);
        assert_eq!(groups.len(), 1);
        assert_eq!(groups.values().next().unwrap().len(), 1);
    }
}
