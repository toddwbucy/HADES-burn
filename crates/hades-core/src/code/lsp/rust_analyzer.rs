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

/// The directory rust-analyzer should be pointed at for `file_path`.
///
/// This is the *Cargo workspace* root, not the nearest `Cargo.toml`. The
/// difference is the single biggest cost in the enrichment phase: rust-analyzer
/// runs `cargo metadata` on startup, which from any member directory resolves
/// the whole workspace and returns every member package. Pointing a session at a
/// member crate therefore indexes the entire workspace anyway — so grouping by
/// nearest `Cargo.toml` started one full-workspace session per member and
/// indexed the same code N times, sequentially. On this repository that was five
/// indexes where one would do.
///
/// Walks up looking for the first `Cargo.toml` declaring a `[workspace]` table,
/// which matches Cargo's own resolution: a member's manifest does not declare
/// one, and a nested crate that opts out of its parent workspace does so by
/// declaring an empty `[workspace]` — in which case it really is its own root
/// and the walk stops there, correctly.
///
/// Falls back to the nearest `Cargo.toml` for a standalone crate that belongs to
/// no workspace.
pub fn find_crate_root(file_path: &Path) -> Option<PathBuf> {
    let nearest = find_workspace_root(file_path, &["Cargo.toml"])?;

    let mut dir = nearest.clone();
    loop {
        let manifest = dir.join("Cargo.toml");
        if manifest.is_file()
            && std::fs::read_to_string(&manifest).is_ok_and(|text| declares_workspace_table(&text))
        {
            return Some(dir);
        }
        if !dir.pop() {
            return Some(nearest);
        }
    }
}

/// Does this manifest text declare a `[workspace]` table?
///
/// Matches `[workspace]` and its sub-tables (`[workspace.dependencies]`,
/// `[workspace.package]`, `[workspace.lints]`), all of which only appear in a
/// workspace root. Deliberately not a TOML parse: this runs while walking
/// ancestors of every discovered `.rs` file, and the question is coarse enough
/// that scanning table headers answers it. A `workspace = true` key inside
/// `[package]` is a *member* marker and must not match, which is why this looks
/// for the bracketed header rather than the bare word.
fn declares_workspace_table(manifest: &str) -> bool {
    manifest.lines().any(|line| {
        let line = line.trim();
        line.starts_with("[workspace]") || line.starts_with("[workspace.")
    })
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

#[cfg(test)]
mod workspace_root_tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    /// Files in different members of one workspace must group to ONE root.
    ///
    /// Grouping by nearest `Cargo.toml` produced one rust-analyzer session per
    /// member, and each session's `cargo metadata` resolves the whole workspace
    /// regardless — so the same code was indexed once per member, sequentially.
    #[test]
    fn members_of_one_workspace_share_a_root() {
        let dir = TempDir::new().unwrap();
        let root = dir.path();
        fs::write(
            root.join("Cargo.toml"),
            "[workspace]\nmembers = [\"crates/a\", \"crates/b\"]\n",
        )
        .unwrap();
        for member in ["a", "b"] {
            let src = root.join("crates").join(member).join("src");
            fs::create_dir_all(&src).unwrap();
            fs::write(
                root.join("crates").join(member).join("Cargo.toml"),
                format!("[package]\nname = \"{member}\"\n"),
            )
            .unwrap();
            fs::write(src.join("lib.rs"), "pub fn f() {}\n").unwrap();
        }

        let files = vec![
            root.join("crates/a/src/lib.rs"),
            root.join("crates/b/src/lib.rs"),
        ];
        let groups = group_files_by_crate(&files);

        assert_eq!(
            groups.len(),
            1,
            "one workspace must yield one session, got {:?}",
            groups.keys().collect::<Vec<_>>()
        );
        let (grouped_root, grouped_files) = groups.into_iter().next().unwrap();
        assert_eq!(grouped_root, root.canonicalize().unwrap_or(root.into()));
        assert_eq!(grouped_files.len(), 2);
    }

    /// A standalone crate with no workspace still resolves to its own manifest.
    #[test]
    fn a_standalone_crate_is_its_own_root() {
        let dir = TempDir::new().unwrap();
        let root = dir.path();
        fs::create_dir_all(root.join("src")).unwrap();
        fs::write(root.join("Cargo.toml"), "[package]\nname = \"solo\"\n").unwrap();
        fs::write(root.join("src/lib.rs"), "pub fn f() {}\n").unwrap();

        let found = find_crate_root(&root.join("src/lib.rs")).unwrap();
        assert_eq!(found, root);
    }

    /// A nested crate that opts out of its parent workspace — an empty
    /// `[workspace]` table — is genuinely its own root, and the walk must stop
    /// there rather than climbing to the parent.
    #[test]
    fn a_crate_opting_out_of_its_parent_workspace_is_its_own_root() {
        let dir = TempDir::new().unwrap();
        let root = dir.path();
        fs::write(root.join("Cargo.toml"), "[workspace]\nmembers = []\n").unwrap();
        let nested = root.join("vendor").join("thing");
        fs::create_dir_all(nested.join("src")).unwrap();
        fs::write(
            nested.join("Cargo.toml"),
            "[workspace]\n\n[package]\nname = \"thing\"\n",
        )
        .unwrap();
        fs::write(nested.join("src/lib.rs"), "pub fn f() {}\n").unwrap();

        let found = find_crate_root(&nested.join("src/lib.rs")).unwrap();
        assert_eq!(found, nested, "the opt-out crate is its own workspace");
    }

    /// `workspace = true` under `[package]` is a *member* marker, not a root.
    #[test]
    fn inherited_keys_do_not_make_a_member_look_like_a_root() {
        assert!(!declares_workspace_table(
            "[package]\nname = \"m\"\nversion.workspace = true\nedition.workspace = true\n"
        ));
        assert!(declares_workspace_table("[workspace]\nmembers = []\n"));
        assert!(declares_workspace_table(
            "[workspace.dependencies]\nserde = \"1\"\n"
        ));
    }
}
