//! Generic batch processor with concurrency, checkpointing, and progress.
//!
//! Processes a list of items through a user-provided async function with:
//! - Per-item error isolation (one failure does not abort the batch)
//! - Resumable checkpoint state (`.hades-batch-state.json`)
//! - Throttled progress reporting to stderr
//! - Bounded concurrency via `Semaphore`
//! - Optional rate limiting

use std::future::Future;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde::Serialize;
use serde_json::Value;
use tokio::sync::Semaphore;
use tokio::task::JoinSet;
use tracing::{debug, error, info, warn};

use super::error::{BatchError, ItemError};
use super::progress::{ProgressReporter, ProgressStatus};
use super::rate_limit::RateLimiter;
use super::state::BatchState;

/// Configuration for a batch processing run.
#[derive(Debug, Clone)]
pub struct BatchProcessorConfig {
    /// Maximum concurrent items in flight.
    pub concurrency: usize,
    /// Path to state file for resume. `None` disables checkpointing.
    pub state_file: Option<PathBuf>,
    /// Whether to resume from an existing state file.
    pub resume: bool,
    /// Whether to reset (delete) existing state before starting.
    pub reset: bool,
    /// Minimum interval between progress reports.
    pub progress_interval: Duration,
    /// Optional rate limiter for throttling requests.
    pub rate_limiter: Option<Arc<RateLimiter>>,
}

impl Default for BatchProcessorConfig {
    fn default() -> Self {
        Self {
            concurrency: 1,
            state_file: Some(PathBuf::from(".hades-batch-state.json")),
            resume: false,
            reset: false,
            progress_interval: Duration::from_secs(1),
            rate_limiter: None,
        }
    }
}

/// Result of processing a single item.
#[derive(Debug, Clone, Serialize)]
pub struct ItemResult {
    /// Item identifier.
    pub item_id: String,
    /// Whether processing succeeded.
    pub success: bool,
    /// Whether this item was skipped (resume / already processed).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub skipped: Option<bool>,
    /// Arbitrary result data (caller's domain-specific result).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
    /// Error details if failed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<ItemError>,
    /// Wall-clock time for this item, in milliseconds.
    pub duration_ms: u64,
}

/// Summary of a completed batch run.
#[derive(Debug, Serialize)]
pub struct BatchSummary {
    /// Total items in the batch (including skipped).
    pub total: usize,
    /// Items completed successfully.
    pub completed: usize,
    /// Items that failed.
    pub failed: usize,
    /// Items skipped (from resume checkpoint).
    pub skipped: usize,
    /// Per-item results.
    pub results: Vec<ItemResult>,
    /// Aggregated error records.
    pub errors: Vec<ItemError>,
    /// Total wall-clock time for the batch, in milliseconds.
    pub duration_ms: u64,
}

/// Generic batch processor with concurrency, checkpointing, and progress.
pub struct BatchProcessor {
    config: BatchProcessorConfig,
}

impl BatchProcessor {
    /// Create a new batch processor with the given configuration.
    pub fn new(config: BatchProcessorConfig) -> Self {
        Self { config }
    }

    /// Process a batch of items through a user-provided async function.
    ///
    /// `items` is a list of `(item_id, item_data)` pairs.
    /// `process_fn` receives the item ID and data, returning a JSON value
    /// on success or an error on failure. Per-item errors are isolated —
    /// one failure does not abort the batch.
    ///
    /// Returns a [`BatchSummary`] with per-item results and aggregate counts.
    pub async fn process<T, F, Fut>(
        &self,
        items: Vec<(String, T)>,
        process_fn: F,
    ) -> Result<BatchSummary, BatchError>
    where
        T: Send + 'static,
        F: Fn(String, T) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<Value, anyhow::Error>> + Send,
    {
        let batch_start = Instant::now();
        let total = items.len();

        // Handle reset.
        if self.config.reset
            && let Some(ref path) = self.config.state_file
        {
            BatchState::clear(path)?;
        }

        // Load or create state.
        let mut state = if self.config.resume {
            if let Some(ref path) = self.config.state_file {
                BatchState::load(path)?.unwrap_or_default()
            } else {
                BatchState::default()
            }
        } else {
            BatchState::default()
        };

        let skip_set = state.skip_set().into_iter().map(String::from).collect::<std::collections::HashSet<String>>();
        let skipped_count = items.iter().filter(|(id, _)| skip_set.contains(id)).count();
        if skipped_count > 0 {
            info!(skipped = skipped_count, "resuming batch, skipping already-processed items");
        }

        // Set up progress reporter.
        let progress = ProgressReporter::new(total, self.config.progress_interval);

        // Set up concurrency control.
        let semaphore = Arc::new(Semaphore::new(self.config.concurrency));
        let process_fn = Arc::new(process_fn);
        let rate_limiter = self.config.rate_limiter.clone();

        // Spawn tasks onto JoinSet.
        let mut join_set: JoinSet<ItemResult> = JoinSet::new();
        let mut results: Vec<ItemResult> = Vec::with_capacity(total);

        for (item_id, item_data) in items {
            // Skip items already processed in a previous run.
            if skip_set.contains(&item_id) {
                progress.inc_completed();
                progress.report(&item_id, ProgressStatus::Skipped, false);
                results.push(ItemResult {
                    item_id,
                    success: true,
                    skipped: Some(true),
                    data: None,
                    error: None,
                    duration_ms: 0,
                });
                continue;
            }

            // Acquire concurrency permit.
            let permit = semaphore.clone().acquire_owned().await.unwrap();
            let fn_clone = process_fn.clone();
            let rl_clone = rate_limiter.clone();
            let id = item_id.clone();

            join_set.spawn(async move {
                // Rate limit if configured.
                if let Some(ref rl) = rl_clone {
                    rl.acquire().await;
                }

                let start = Instant::now();
                let result = fn_clone(id.clone(), item_data).await;
                let duration_ms = start.elapsed().as_millis() as u64;

                drop(permit); // Release semaphore permit.

                match result {
                    Ok(data) => ItemResult {
                        item_id: id,
                        success: true,
                        skipped: None,
                        data: Some(data),
                        error: None,
                        duration_ms,
                    },
                    Err(e) => ItemResult {
                        item_id: id.clone(),
                        success: false,
                        skipped: None,
                        data: None,
                        error: Some(ItemError {
                            item_id: id,
                            stage: "processing".into(),
                            message: e.to_string(),
                            duration_ms,
                        }),
                        duration_ms,
                    },
                }
            });

            // Collect completed tasks as we go (prevents unbounded memory growth).
            while let Some(Ok(result)) = join_set.try_join_next() {
                self.record_result(&mut state, &progress, result, &mut results);
            }
        }

        // Drain remaining tasks.
        while let Some(join_result) = join_set.join_next().await {
            match join_result {
                Ok(result) => {
                    self.record_result(&mut state, &progress, result, &mut results);
                }
                Err(e) => {
                    // Task panicked — record as a generic failure.
                    error!(error = %e, "task panicked");
                    let item_error = ItemError {
                        item_id: "unknown".into(),
                        stage: "spawn".into(),
                        message: format!("task panicked: {e}"),
                        duration_ms: 0,
                    };
                    results.push(ItemResult {
                        item_id: "unknown".into(),
                        success: false,
                        skipped: None,
                        data: None,
                        error: Some(item_error),
                        duration_ms: 0,
                    });
                }
            }
        }

        // Save final state.
        if let Some(ref path) = self.config.state_file {
            let actual_failed = results.iter().filter(|r| !r.success && r.skipped != Some(true)).count();
            if actual_failed == 0 {
                // All succeeded — clean up state file.
                BatchState::clear(path)?;
                debug!("batch complete with zero failures, state file cleared");
            } else {
                state.save(path)?;
                warn!(failures = actual_failed, "batch complete with failures, state file retained");
            }
        }

        progress.report_final();

        // Build summary.
        let errors: Vec<ItemError> = results
            .iter()
            .filter_map(|r| r.error.clone())
            .collect();
        let completed = results.iter().filter(|r| r.success && r.skipped != Some(true)).count();
        let failed = results.iter().filter(|r| !r.success).count();
        let skipped = results.iter().filter(|r| r.skipped == Some(true)).count();

        Ok(BatchSummary {
            total,
            completed,
            failed,
            skipped,
            results,
            errors,
            duration_ms: batch_start.elapsed().as_millis() as u64,
        })
    }

    /// Record a single result: update state, save checkpoint, report progress.
    fn record_result(
        &self,
        state: &mut BatchState,
        progress: &ProgressReporter,
        result: ItemResult,
        results: &mut Vec<ItemResult>,
    ) {
        if result.success {
            state.mark_completed(result.item_id.clone());
            progress.inc_completed();
            progress.report(&result.item_id, ProgressStatus::Completed, false);
        } else {
            let error_msg = result
                .error
                .as_ref()
                .map(|e| e.message.clone())
                .unwrap_or_default();
            state.mark_failed(result.item_id.clone(), error_msg);
            progress.inc_failed();
            progress.report(&result.item_id, ProgressStatus::Failed, false);
        }

        // Checkpoint after every item for crash resilience.
        if let Some(ref path) = self.config.state_file
            && let Err(e) = state.save(path)
        {
            warn!(error = %e, "failed to save batch state (continuing)");
        }

        results.push(result);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_batch_empty() {
        let processor = BatchProcessor::new(BatchProcessorConfig {
            state_file: None,
            ..Default::default()
        });
        let items: Vec<(String, ())> = vec![];
        let summary = processor
            .process(items, |_id, _data| async { Ok(Value::Null) })
            .await
            .unwrap();
        assert_eq!(summary.total, 0);
        assert_eq!(summary.completed, 0);
        assert_eq!(summary.failed, 0);
    }

    #[tokio::test]
    async fn test_batch_all_succeed() {
        let dir = TempDir::new().unwrap();
        let state_path = dir.path().join("state.json");
        let processor = BatchProcessor::new(BatchProcessorConfig {
            state_file: Some(state_path.clone()),
            ..Default::default()
        });

        let items = vec![
            ("a".into(), 1),
            ("b".into(), 2),
            ("c".into(), 3),
        ];

        let summary = processor
            .process(items, |_id, val| async move {
                Ok(serde_json::json!({ "value": val }))
            })
            .await
            .unwrap();

        assert_eq!(summary.total, 3);
        assert_eq!(summary.completed, 3);
        assert_eq!(summary.failed, 0);
        assert_eq!(summary.errors.len(), 0);
        // State file should be cleaned up on success.
        assert!(!state_path.exists());
    }

    #[tokio::test]
    async fn test_batch_partial_failure() {
        let dir = TempDir::new().unwrap();
        let state_path = dir.path().join("state.json");
        let processor = BatchProcessor::new(BatchProcessorConfig {
            state_file: Some(state_path.clone()),
            ..Default::default()
        });

        let items = vec![
            ("ok_1".into(), true),
            ("fail_1".into(), false),
            ("ok_2".into(), true),
        ];

        let summary = processor
            .process(items, |_id, should_succeed| async move {
                if should_succeed {
                    Ok(Value::Null)
                } else {
                    Err(anyhow::anyhow!("intentional failure"))
                }
            })
            .await
            .unwrap();

        assert_eq!(summary.total, 3);
        assert_eq!(summary.completed, 2);
        assert_eq!(summary.failed, 1);
        assert_eq!(summary.errors.len(), 1);
        assert_eq!(summary.errors[0].item_id, "fail_1");
        // State file should be retained on failure.
        assert!(state_path.exists());
    }

    #[tokio::test]
    async fn test_batch_resume_skips_completed() {
        let dir = TempDir::new().unwrap();
        let state_path = dir.path().join("state.json");

        // Simulate a previous run that completed "a" and failed "b".
        let mut prev_state = BatchState::new();
        prev_state.mark_completed("a".into());
        prev_state.mark_failed("b".into(), "previous error".into());
        prev_state.save(&state_path).unwrap();

        let processor = BatchProcessor::new(BatchProcessorConfig {
            state_file: Some(state_path.clone()),
            resume: true,
            ..Default::default()
        });

        let call_count = Arc::new(AtomicUsize::new(0));
        let cc = call_count.clone();

        let items = vec![
            ("a".into(), ()),
            ("b".into(), ()),
            ("c".into(), ()),
        ];

        let summary = processor
            .process(items, move |_id, _data| {
                let cc = cc.clone();
                async move {
                    cc.fetch_add(1, Ordering::Relaxed);
                    Ok(Value::Null)
                }
            })
            .await
            .unwrap();

        // Only "c" should have been processed; "a" and "b" were skipped.
        assert_eq!(call_count.load(Ordering::Relaxed), 1);
        assert_eq!(summary.skipped, 2);
        assert_eq!(summary.completed, 1);
    }

    #[tokio::test]
    async fn test_batch_state_cleared_on_success() {
        let dir = TempDir::new().unwrap();
        let state_path = dir.path().join("state.json");
        let processor = BatchProcessor::new(BatchProcessorConfig {
            state_file: Some(state_path.clone()),
            ..Default::default()
        });

        let items = vec![("a".into(), ())];
        processor
            .process(items, |_id, _data| async { Ok(Value::Null) })
            .await
            .unwrap();

        assert!(!state_path.exists(), "state file should be deleted on success");
    }

    #[tokio::test]
    async fn test_batch_state_persisted_on_failure() {
        let dir = TempDir::new().unwrap();
        let state_path = dir.path().join("state.json");
        let processor = BatchProcessor::new(BatchProcessorConfig {
            state_file: Some(state_path.clone()),
            ..Default::default()
        });

        let items = vec![("a".into(), ())];
        processor
            .process(items, |_id, _data| async {
                Err(anyhow::anyhow!("fail"))
            })
            .await
            .unwrap();

        assert!(state_path.exists(), "state file should be retained on failure");
        let state = BatchState::load(&state_path).unwrap().unwrap();
        assert_eq!(state.failed.len(), 1);
    }

    #[tokio::test]
    async fn test_batch_concurrency_bounded() {
        let max_concurrent = Arc::new(AtomicUsize::new(0));
        let current = Arc::new(AtomicUsize::new(0));

        let processor = BatchProcessor::new(BatchProcessorConfig {
            concurrency: 2,
            state_file: None,
            ..Default::default()
        });

        let items: Vec<(String, ())> = (0..6).map(|i| (format!("item_{i}"), ())).collect();

        let mc = max_concurrent.clone();
        let cur = current.clone();

        processor
            .process(items, move |_id, _data| {
                let mc = mc.clone();
                let cur = cur.clone();
                async move {
                    let prev = cur.fetch_add(1, Ordering::SeqCst);
                    mc.fetch_max(prev + 1, Ordering::SeqCst);
                    tokio::time::sleep(Duration::from_millis(50)).await;
                    cur.fetch_sub(1, Ordering::SeqCst);
                    Ok(Value::Null)
                }
            })
            .await
            .unwrap();

        let observed_max = max_concurrent.load(Ordering::SeqCst);
        assert!(
            observed_max <= 2,
            "max concurrent was {observed_max}, expected <= 2"
        );
    }

    #[tokio::test]
    async fn test_batch_reset() {
        let dir = TempDir::new().unwrap();
        let state_path = dir.path().join("state.json");

        // Create a pre-existing state file.
        let mut prev = BatchState::new();
        prev.mark_completed("old".into());
        prev.save(&state_path).unwrap();

        let processor = BatchProcessor::new(BatchProcessorConfig {
            state_file: Some(state_path.clone()),
            reset: true,
            ..Default::default()
        });

        let call_count = Arc::new(AtomicUsize::new(0));
        let cc = call_count.clone();

        let items = vec![("old".into(), ()), ("new".into(), ())];

        processor
            .process(items, move |_id, _data| {
                let cc = cc.clone();
                async move {
                    cc.fetch_add(1, Ordering::Relaxed);
                    Ok(Value::Null)
                }
            })
            .await
            .unwrap();

        // Both items should be processed (reset cleared the skip set).
        assert_eq!(call_count.load(Ordering::Relaxed), 2);
    }
}
