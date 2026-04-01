//! Generic batch processing with concurrency, checkpointing, and progress.
//!
//! Provides [`BatchProcessor`] for running N items through an async pipeline
//! with per-item error isolation, resumable checkpoint state, throttled
//! progress reporting, and optional rate limiting.

mod error;
mod processor;
mod progress;
mod rate_limit;
mod state;

pub use error::{BatchError, ItemError};
pub use processor::{BatchProcessor, BatchProcessorConfig, BatchSummary, ItemResult};
pub use progress::{ProgressEvent, ProgressReporter, ProgressStatus};
pub use rate_limit::RateLimiter;
pub use state::BatchState;
