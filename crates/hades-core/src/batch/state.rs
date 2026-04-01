//! Persistent checkpoint state for resumable batch processing.
//!
//! Serializes to `.hades-batch-state.json`, wire-compatible with the
//! Python HADES `BatchState` dataclass in `core/processors/batch.py`.

use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};
use tracing::debug;

/// Persistent state for resumable batch processing.
///
/// Tracks which items have been completed or failed so that a
/// batch can resume from where it left off after a crash or
/// interruption.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchState {
    /// Item IDs that completed successfully.
    pub completed: Vec<String>,
    /// Item IDs that failed, mapped to error messages.
    pub failed: HashMap<String, String>,
    /// ISO-8601 timestamp when this batch started.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub started_at: Option<String>,
    /// ISO-8601 timestamp of last state save.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_updated: Option<String>,
}

impl BatchState {
    /// Create a new empty state.
    pub fn new() -> Self {
        Self {
            completed: Vec::new(),
            failed: HashMap::new(),
            started_at: Some(chrono::Utc::now().to_rfc3339()),
            last_updated: None,
        }
    }

    /// Load state from a JSON file.
    ///
    /// Returns `Ok(None)` if the file does not exist.
    /// Returns `Err` if the file exists but is corrupt.
    pub fn load(path: &Path) -> Result<Option<Self>, super::BatchError> {
        let content = match fs::read_to_string(path) {
            Ok(c) => c,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(e) => return Err(e.into()),
        };
        let state: Self = serde_json::from_str(&content)?;
        debug!(
            completed = state.completed.len(),
            failed = state.failed.len(),
            "loaded batch state"
        );
        Ok(Some(state))
    }

    /// Save state to a JSON file atomically.
    ///
    /// Writes to a temporary file first, then renames to prevent
    /// corruption if the process is killed mid-write.
    pub fn save(&mut self, path: &Path) -> Result<(), super::BatchError> {
        self.last_updated = Some(chrono::Utc::now().to_rfc3339());
        let json = serde_json::to_string_pretty(self)?;

        let tmp_path = path.with_extension("json.tmp");
        fs::write(&tmp_path, json)?;
        fs::rename(&tmp_path, path)?;
        Ok(())
    }

    /// Delete the state file.
    pub fn clear(path: &Path) -> Result<(), super::BatchError> {
        match fs::remove_file(path) {
            Ok(()) => {
                debug!("cleared batch state file");
                Ok(())
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(e) => Err(e.into()),
        }
    }

    /// Record a successful item.
    pub fn mark_completed(&mut self, item_id: String) {
        self.completed.push(item_id);
    }

    /// Record a failed item.
    pub fn mark_failed(&mut self, item_id: String, error: String) {
        self.failed.insert(item_id, error);
    }

    /// Build a set of item IDs to skip (completed + failed).
    pub fn skip_set(&self) -> HashSet<&str> {
        let mut set = HashSet::with_capacity(self.completed.len() + self.failed.len());
        for id in &self.completed {
            set.insert(id.as_str());
        }
        for id in self.failed.keys() {
            set.insert(id.as_str());
        }
        set
    }
}

impl Default for BatchState {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_state_roundtrip() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("state.json");

        let mut state = BatchState::new();
        state.mark_completed("item_1".into());
        state.mark_completed("item_2".into());
        state.mark_failed("item_3".into(), "connection timeout".into());
        state.save(&path).unwrap();

        let loaded = BatchState::load(&path).unwrap().unwrap();
        assert_eq!(loaded.completed.len(), 2);
        assert_eq!(loaded.failed.len(), 1);
        assert_eq!(loaded.failed["item_3"], "connection timeout");
        assert!(loaded.started_at.is_some());
        assert!(loaded.last_updated.is_some());
    }

    #[test]
    fn test_state_load_missing() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("nonexistent.json");
        let result = BatchState::load(&path).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_state_load_corrupt() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("corrupt.json");
        fs::write(&path, "not valid json {{{").unwrap();
        let result = BatchState::load(&path);
        assert!(result.is_err());
    }

    #[test]
    fn test_state_clear() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("state.json");
        fs::write(&path, "{}").unwrap();
        assert!(path.exists());
        BatchState::clear(&path).unwrap();
        assert!(!path.exists());
    }

    #[test]
    fn test_state_clear_missing() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("nonexistent.json");
        // Should not error on missing file.
        BatchState::clear(&path).unwrap();
    }

    #[test]
    fn test_state_atomic_write() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("state.json");

        let mut state = BatchState::new();
        state.mark_completed("a".into());
        state.save(&path).unwrap();

        // Temp file should not linger.
        let tmp = path.with_extension("json.tmp");
        assert!(!tmp.exists());
        // Real file should exist with valid JSON.
        let content = fs::read_to_string(&path).unwrap();
        let _: BatchState = serde_json::from_str(&content).unwrap();
    }

    #[test]
    fn test_python_compat() {
        // Matches Python HADES BatchState.to_dict() output format.
        let json = r#"{
            "completed": ["2501.12345", "2501.67890"],
            "failed": {"2501.99999": "connection timeout"},
            "started_at": "2025-01-15T10:00:00+00:00",
            "last_updated": "2025-01-15T10:15:33.456000+00:00"
        }"#;
        let state: BatchState = serde_json::from_str(json).unwrap();
        assert_eq!(state.completed.len(), 2);
        assert_eq!(state.failed.len(), 1);
        assert_eq!(state.failed["2501.99999"], "connection timeout");
    }

    #[test]
    fn test_skip_set() {
        let mut state = BatchState::new();
        state.mark_completed("a".into());
        state.mark_completed("b".into());
        state.mark_failed("c".into(), "err".into());

        let skip = state.skip_set();
        assert_eq!(skip.len(), 3);
        assert!(skip.contains("a"));
        assert!(skip.contains("b"));
        assert!(skip.contains("c"));
    }
}
