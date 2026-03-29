//! Configuration loading for HADES-Burn.
//!
//! Loads configuration from YAML, environment variables, and CLI arguments.
//! Override priority (highest wins):
//!   1. CLI arguments (`--db`, `--gpu`)
//!   2. Environment variables (`ARANGO_*`, `HADES_*`)
//!   3. YAML config file (`hades.yaml`)
//!   4. Compiled-in defaults

mod types;

pub use types::*;

use std::env;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use tracing::debug;

/// Standard locations to search for hades.yaml, in priority order.
const CONFIG_SEARCH_PATHS: &[&str] = &[
    "hades.yaml",
    "core/config/hades.yaml",
];

/// Load the full HADES configuration.
///
/// Resolution order:
/// 1. Find and parse `hades.yaml` (or use defaults if not found)
/// 2. Apply environment variable overrides
///
/// CLI overrides are applied separately via [`HadesConfig::apply_cli_overrides`]
/// after this function returns.
pub fn load_config() -> Result<HadesConfig> {
    let yaml_path = find_config_file();

    let mut config = match yaml_path {
        Some(ref path) => {
            debug!("loading config from {}", path.display());
            let contents = std::fs::read_to_string(path)
                .with_context(|| format!("failed to read config file: {}", path.display()))?;
            serde_yaml::from_str(&contents)
                .with_context(|| format!("failed to parse config file: {}", path.display()))?
        }
        None => {
            debug!("no config file found, using defaults");
            HadesConfig::default()
        }
    };

    config.apply_env_overrides();
    Ok(config)
}

/// Search for hades.yaml in standard locations.
///
/// Priority:
/// 1. `HADES_CONFIG` environment variable (explicit path)
/// 2. `./hades.yaml` (current directory)
/// 3. `./core/config/hades.yaml` (running from HADES repo root)
/// 4. `~/.config/hades/hades.yaml` (user config)
/// 5. `/etc/hades/hades.yaml` (system config)
fn find_config_file() -> Option<PathBuf> {
    // Explicit path via env var
    if let Ok(explicit) = env::var("HADES_CONFIG") {
        let path = PathBuf::from(&explicit);
        if path.exists() {
            return Some(path);
        }
        debug!("HADES_CONFIG={explicit} does not exist, searching standard paths");
    }

    // Current directory and repo-relative paths
    for candidate in CONFIG_SEARCH_PATHS {
        let path = PathBuf::from(candidate);
        if path.exists() {
            return Some(path);
        }
    }

    // User config directory
    if let Some(home) = env::var_os("HOME") {
        let path = Path::new(&home).join(".config/hades/hades.yaml");
        if path.exists() {
            return Some(path);
        }
    }

    // System config
    let system = PathBuf::from("/etc/hades/hades.yaml");
    if system.exists() {
        return Some(system);
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_defaults_without_yaml() {
        let config = HadesConfig::default();
        assert_eq!(config.database.name, "NestedLearning");
        assert_eq!(config.database.host, "localhost");
        assert_eq!(config.database.port, 8529);
        assert_eq!(config.embedding.model.dimension, 2048);
        assert_eq!(config.search.limit, 10);
    }

    #[test]
    fn test_load_from_yaml() {
        let yaml = r#"
database:
  host: db.example.com
  port: 9529
  database: TestDB
  sockets:
    readonly: /tmp/test-ro.sock
    readwrite: /tmp/test-rw.sock

embedding:
  model:
    dimension: 1024

search:
  limit: 25
"#;
        let mut file = NamedTempFile::new().unwrap();
        file.write_all(yaml.as_bytes()).unwrap();

        let contents = std::fs::read_to_string(file.path()).unwrap();
        let config: HadesConfig = serde_yaml::from_str(&contents).unwrap();

        assert_eq!(config.database.host, "db.example.com");
        assert_eq!(config.database.port, 9529);
        assert_eq!(config.database.name, "TestDB");
        assert_eq!(
            config.database.sockets.readonly.as_deref(),
            Some("/tmp/test-ro.sock")
        );
        assert_eq!(config.embedding.model.dimension, 1024);
        assert_eq!(config.search.limit, 25);
        // Unset fields keep defaults
        assert_eq!(config.embedding.model.name, "jinaai/jina-embeddings-v4");
        assert_eq!(config.rocchio.alpha, 1.0);
    }

    #[test]
    fn test_env_overrides() {
        let mut config = HadesConfig::default();

        // SAFETY: test runs single-threaded via cargo test -- --test-threads=1
        // or is isolated enough that env mutation is acceptable.
        unsafe {
            env::set_var("ARANGO_HOST", "override-host");
            env::set_var("ARANGO_PORT", "1234");
            env::set_var("HADES_DATABASE", "OverrideDB");
            env::set_var("ARANGO_PASSWORD", "secret");
        }

        config.apply_env_overrides();

        assert_eq!(config.database.host, "override-host");
        assert_eq!(config.database.port, 1234);
        assert_eq!(config.database.name, "OverrideDB");
        assert_eq!(config.database.password.as_deref(), Some("secret"));

        // Clean up
        unsafe {
            env::remove_var("ARANGO_HOST");
            env::remove_var("ARANGO_PORT");
            env::remove_var("HADES_DATABASE");
            env::remove_var("ARANGO_PASSWORD");
        }
    }

    #[test]
    fn test_cli_overrides() {
        let mut config = HadesConfig::default();
        assert_eq!(config.database.name, "NestedLearning");

        config.apply_cli_overrides(Some("bident_burn"), None);
        assert_eq!(config.database.name, "bident_burn");

        config.apply_cli_overrides(None, Some(1));
        assert_eq!(config.gpu.device, "cuda:1");
    }

    #[test]
    fn test_effective_socket_readonly() {
        let mut config = HadesConfig::default();
        config.database.sockets.readonly = Some("/tmp/ro.sock".into());
        config.database.sockets.readwrite = Some("/tmp/rw.sock".into());

        assert_eq!(config.effective_socket(true), Some("/tmp/ro.sock"));
        assert_eq!(config.effective_socket(false), Some("/tmp/rw.sock"));
    }

    #[test]
    fn test_effective_database() {
        let config = HadesConfig::default();
        assert_eq!(config.effective_database(), "NestedLearning");
    }

    #[test]
    fn test_gpu_disabled_via_env() {
        let mut config = HadesConfig::default();
        assert!(config.gpu.enabled);

        // SAFETY: test-only env mutation
        unsafe { env::set_var("HADES_USE_GPU", "false") };
        config.apply_env_overrides();
        assert!(!config.gpu.enabled);

        unsafe { env::remove_var("HADES_USE_GPU") };
    }
}
