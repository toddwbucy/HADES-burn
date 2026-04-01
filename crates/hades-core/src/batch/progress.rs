//! Throttled progress reporter for batch processing.
//!
//! Emits structured JSON progress events to stderr at configurable
//! intervals to prevent log spam during large batches.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use serde::Serialize;

/// Progress update emitted to stderr as JSON.
#[derive(Debug, Clone, Serialize)]
pub struct ProgressEvent {
    /// Always "progress".
    #[serde(rename = "type")]
    pub event_type: &'static str,
    /// Total items in the batch.
    pub total: usize,
    /// Items completed so far.
    pub completed: usize,
    /// Items failed so far.
    pub failed: usize,
    /// Currently processing item.
    pub current_item: String,
    /// Status of this progress event.
    pub status: ProgressStatus,
    /// Completion percentage.
    pub percent: f64,
    /// Wall-clock time since batch start, in milliseconds.
    pub elapsed_ms: u64,
    /// Estimated time to completion, in milliseconds.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub eta_ms: Option<u64>,
}

/// Status of a progress event.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ProgressStatus {
    Processing,
    Completed,
    Failed,
    Skipped,
}

/// Throttled progress reporter.
///
/// Thread-safe: uses atomics for counters and a mutex for the
/// last-report timestamp.
pub struct ProgressReporter {
    total: usize,
    completed: AtomicUsize,
    failed: AtomicUsize,
    start: Instant,
    min_interval: Duration,
    last_report: Mutex<Instant>,
}

impl ProgressReporter {
    /// Create a new reporter for a batch of `total` items.
    pub fn new(total: usize, min_interval: Duration) -> Self {
        let now = Instant::now();
        Self {
            total,
            completed: AtomicUsize::new(0),
            failed: AtomicUsize::new(0),
            start: now,
            min_interval,
            // Set last_report far enough in the past to allow the first report.
            // Use checked_sub to avoid panic if min_interval exceeds time since
            // the monotonic clock epoch (e.g., very early in process lifetime).
            last_report: Mutex::new(now.checked_sub(min_interval).unwrap_or(now)),
        }
    }

    /// Increment the completed counter.
    pub fn inc_completed(&self) {
        self.completed.fetch_add(1, Ordering::Relaxed);
    }

    /// Increment the failed counter.
    pub fn inc_failed(&self) {
        self.failed.fetch_add(1, Ordering::Relaxed);
    }

    /// Emit a progress event if the throttle interval has elapsed.
    ///
    /// If `force` is true, the interval check is skipped (used for
    /// the final report).
    pub fn report(&self, item_id: &str, status: ProgressStatus, force: bool) {
        if !force {
            let mut last = self.last_report.lock().unwrap();
            if last.elapsed() < self.min_interval {
                return;
            }
            *last = Instant::now();
        }

        let completed = self.completed.load(Ordering::Relaxed);
        let failed = self.failed.load(Ordering::Relaxed);
        let processed = completed + failed;
        let elapsed = self.start.elapsed();
        let elapsed_ms = elapsed.as_millis() as u64;

        let percent = if self.total > 0 {
            (processed as f64 / self.total as f64) * 100.0
        } else {
            0.0
        };

        let eta_ms = self.estimate_eta(processed, elapsed);

        let event = ProgressEvent {
            event_type: "progress",
            total: self.total,
            completed,
            failed,
            current_item: item_id.to_string(),
            status,
            percent,
            elapsed_ms,
            eta_ms,
        };

        if let Ok(json) = serde_json::to_string(&event) {
            eprintln!("{json}");
        }
    }

    /// Emit a final progress event (always, regardless of throttle).
    pub fn report_final(&self) {
        self.report("", ProgressStatus::Completed, true);
    }

    /// Estimate time to completion.
    ///
    /// Returns `None` if fewer than 2 items have been processed
    /// (insufficient data for a meaningful estimate).
    fn estimate_eta(&self, processed: usize, elapsed: Duration) -> Option<u64> {
        if processed < 2 || processed >= self.total {
            return None;
        }
        let per_item = elapsed.as_millis() as f64 / processed as f64;
        let remaining = (self.total - processed) as f64;
        Some((per_item * remaining) as u64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_progress_throttle() {
        let reporter = ProgressReporter::new(10, Duration::from_secs(10));
        // First report should go through (last_report is in the past).
        reporter.report("item_1", ProgressStatus::Processing, false);
        // Capture last_report after the first (accepted) report.
        let after_first = *reporter.last_report.lock().unwrap();
        // Immediate second report should be suppressed (within 10s interval).
        reporter.report("item_2", ProgressStatus::Processing, false);
        let after_second = *reporter.last_report.lock().unwrap();
        // Throttled report must not update last_report.
        assert_eq!(after_first, after_second, "throttled report should not update last_report");
    }

    #[test]
    fn test_progress_force() {
        let reporter = ProgressReporter::new(10, Duration::from_secs(10));
        reporter.report("item_1", ProgressStatus::Processing, false);
        // Force bypasses throttle.
        reporter.report("item_2", ProgressStatus::Completed, true);
    }

    #[test]
    fn test_eta_calculation() {
        let reporter = ProgressReporter::new(10, Duration::from_secs(1));
        // < 2 items: no ETA.
        assert!(reporter.estimate_eta(0, Duration::from_secs(0)).is_none());
        assert!(reporter.estimate_eta(1, Duration::from_secs(10)).is_none());

        // 2 of 10 done in 20 seconds → 10s/item → 8 remaining → 80s ETA.
        let eta = reporter.estimate_eta(2, Duration::from_secs(20)).unwrap();
        assert_eq!(eta, 80_000); // 80 seconds in milliseconds

        // All done: no ETA.
        assert!(reporter.estimate_eta(10, Duration::from_secs(100)).is_none());
    }

    #[test]
    fn test_progress_counters() {
        let reporter = ProgressReporter::new(5, Duration::from_secs(1));
        reporter.inc_completed();
        reporter.inc_completed();
        reporter.inc_failed();
        assert_eq!(reporter.completed.load(Ordering::Relaxed), 2);
        assert_eq!(reporter.failed.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn test_progress_event_json_format() {
        let event = ProgressEvent {
            event_type: "progress",
            total: 10,
            completed: 3,
            failed: 1,
            current_item: "test_item".into(),
            status: ProgressStatus::Completed,
            percent: 40.0,
            elapsed_ms: 5000,
            eta_ms: Some(7500),
        };
        let json = serde_json::to_string(&event).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed["type"], "progress");
        assert_eq!(parsed["total"], 10);
        assert_eq!(parsed["completed"], 3);
        assert_eq!(parsed["eta_ms"], 7500);
    }
}
