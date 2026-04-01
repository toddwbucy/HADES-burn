//! Token-bucket rate limiter with exponential backoff.
//!
//! Extracted from the pattern in `arxiv/client.rs` but made
//! standalone and reusable.

use std::time::{Duration, Instant};

use tokio::sync::Mutex;
use tokio::time::sleep;
use tracing::trace;

/// Token-bucket rate limiter with exponential backoff for retries.
#[derive(Debug)]
pub struct RateLimiter {
    /// Minimum delay between permits.
    min_delay: Duration,
    /// Maximum retry attempts.
    max_retries: u32,
    /// Tracks last permit time.
    last_permit: Mutex<Option<Instant>>,
}

impl RateLimiter {
    /// Create a new rate limiter.
    ///
    /// `requests_per_sec` of 0.0 or negative means unlimited (no delay).
    /// `max_retries` controls how many times a retryable error can be retried.
    pub fn new(requests_per_sec: f64, max_retries: u32) -> Self {
        let min_delay = if requests_per_sec > 0.0 {
            Duration::from_secs_f64(1.0 / requests_per_sec)
        } else {
            Duration::ZERO
        };
        Self {
            min_delay,
            max_retries,
            last_permit: Mutex::new(None),
        }
    }

    /// Wait until the rate limit allows the next request.
    ///
    /// If called faster than the configured rate, sleeps for the
    /// remaining interval.
    pub async fn acquire(&self) {
        if self.min_delay.is_zero() {
            return;
        }

        let mut last = self.last_permit.lock().await;
        if let Some(prev) = *last {
            let elapsed = prev.elapsed();
            if elapsed < self.min_delay {
                let wait = self.min_delay - elapsed;
                trace!(wait_ms = wait.as_millis(), "rate limit: waiting");
                sleep(wait).await;
            }
        }
        *last = Some(Instant::now());
    }

    /// Compute the backoff delay for a given retry attempt.
    ///
    /// Uses exponential backoff: `2^attempt * base_delay`.
    /// When rate limiting is unlimited (`min_delay` is zero), a 1-second
    /// floor is applied so that retries still back off meaningfully.
    pub fn backoff_delay(&self, attempt: u32) -> Duration {
        let base = if self.min_delay.is_zero() {
            Duration::from_secs(1)
        } else {
            self.min_delay
        };
        base * 2u32.pow(attempt)
    }

    /// Maximum number of retry attempts.
    pub fn max_retries(&self) -> u32 {
        self.max_retries
    }

    /// Whether an HTTP status code is retryable (429 or 503).
    pub fn is_retryable_status(status: u16) -> bool {
        status == 429 || status == 503
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_rate_limit_acquire() {
        // 10 req/sec → 100ms between permits.
        let limiter = RateLimiter::new(10.0, 3);
        let start = Instant::now();
        limiter.acquire().await; // first: no wait
        limiter.acquire().await; // second: should wait ~100ms
        let elapsed = start.elapsed();
        // Should be at least ~90ms (allowing some scheduling slack).
        assert!(
            elapsed >= Duration::from_millis(80),
            "elapsed: {elapsed:?}"
        );
    }

    #[tokio::test]
    async fn test_rate_limit_unlimited() {
        let limiter = RateLimiter::new(0.0, 3);
        let start = Instant::now();
        limiter.acquire().await;
        limiter.acquire().await;
        limiter.acquire().await;
        // Should be nearly instant.
        assert!(start.elapsed() < Duration::from_millis(10));
    }

    #[test]
    fn test_backoff_delay() {
        let limiter = RateLimiter::new(2.0, 5); // 500ms base delay
        assert_eq!(limiter.backoff_delay(0), Duration::from_millis(500));
        assert_eq!(limiter.backoff_delay(1), Duration::from_millis(1000));
        assert_eq!(limiter.backoff_delay(2), Duration::from_millis(2000));
        assert_eq!(limiter.backoff_delay(3), Duration::from_millis(4000));
    }

    #[test]
    fn test_backoff_delay_unlimited_rate() {
        // With unlimited rate (0 rps), backoff uses a 1-second floor.
        let limiter = RateLimiter::new(0.0, 3);
        assert_eq!(limiter.backoff_delay(0), Duration::from_secs(1));
        assert_eq!(limiter.backoff_delay(1), Duration::from_secs(2));
        assert_eq!(limiter.backoff_delay(2), Duration::from_secs(4));
    }

    #[test]
    fn test_retryable_status() {
        assert!(RateLimiter::is_retryable_status(429));
        assert!(RateLimiter::is_retryable_status(503));
        assert!(!RateLimiter::is_retryable_status(404));
        assert!(!RateLimiter::is_retryable_status(500));
        assert!(!RateLimiter::is_retryable_status(200));
    }

    #[test]
    fn test_max_retries() {
        let limiter = RateLimiter::new(1.0, 5);
        assert_eq!(limiter.max_retries(), 5);
    }
}
