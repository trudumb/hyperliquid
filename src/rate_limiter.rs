use parking_lot::Mutex;
use std::{
    sync::Arc,
    time::{Duration, Instant},
};
use tokio::time::sleep;

/// Rate limit configuration based on Hyperliquid's documented limits
#[derive(Debug, Clone)]
pub struct RateLimitConfig {
    /// Maximum weight per window (default: 1200 for REST API)
    pub max_weight: u32,
    /// Time window duration (default: 60 seconds)
    pub window_duration: Duration,
    /// Safety margin as a percentage (0.0 to 1.0, default: 0.9 = 90%)
    /// Use 90% of the limit to avoid hitting the exact limit
    pub safety_margin: f64,
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        Self {
            max_weight: 1200,
            window_duration: Duration::from_secs(60),
            safety_margin: 0.9, // Use 90% of limit by default
        }
    }
}

impl RateLimitConfig {
    /// Configuration for REST API rate limiting (1200 weight per minute)
    pub fn rest_api() -> Self {
        Self::default()
    }

    /// Configuration for WebSocket message rate limiting (2000 messages per minute)
    pub fn websocket_messages() -> Self {
        Self {
            max_weight: 2000,
            window_duration: Duration::from_secs(60),
            safety_margin: 0.9,
        }
    }

    /// Configuration for EVM JSON-RPC rate limiting (100 requests per minute)
    pub fn evm_rpc() -> Self {
        Self {
            max_weight: 100,
            window_duration: Duration::from_secs(60),
            safety_margin: 0.9,
        }
    }

    /// Get effective max weight with safety margin applied
    pub fn effective_max_weight(&self) -> u32 {
        (self.max_weight as f64 * self.safety_margin) as u32
    }
}

/// Request weight calculator based on Hyperliquid's weight system
#[derive(Debug, Clone, Copy)]
pub enum RequestWeight {
    /// Exchange API request weight: 1 + floor(batch_length / 40)
    Exchange { batch_length: usize },
    /// Info requests with weight 2
    InfoLight, // l2Book, allMids, clearinghouseState, orderStatus, etc.
    /// Info requests with weight 20
    InfoStandard, // Most info requests
    /// Info requests with weight 60
    InfoHeavy, // userRole
    /// Explorer API requests with weight 40
    Explorer,
    /// Custom weight
    Custom(u32),
}

impl RequestWeight {
    /// Calculate the actual weight for this request
    pub fn weight(&self) -> u32 {
        match self {
            RequestWeight::Exchange { batch_length } => 1 + (*batch_length as u32 / 40),
            RequestWeight::InfoLight => 2,
            RequestWeight::InfoStandard => 20,
            RequestWeight::InfoHeavy => 60,
            RequestWeight::Explorer => 40,
            RequestWeight::Custom(w) => *w,
        }
    }

    /// Calculate weight for paginated responses
    /// Additional weight per 20 items for certain endpoints
    pub fn with_pagination_weight(self, items_returned: usize) -> u32 {
        self.weight() + (items_returned as u32 / 20)
    }

    /// Calculate weight for candle snapshot
    /// Additional weight per 60 items
    pub fn with_candle_pagination(self, items_returned: usize) -> u32 {
        self.weight() + (items_returned as u32 / 60)
    }
}

/// Statistics for rate limiter monitoring
#[derive(Debug, Clone, Default)]
pub struct RateLimitStats {
    pub total_requests: u64,
    pub total_weight_used: u64,
    pub times_waited: u64,
    pub total_wait_time_ms: u64,
    pub current_weight_in_window: u32,
    pub requests_in_current_window: u32,
}

/// Token bucket rate limiter with sliding window
///
/// This implementation uses a token bucket algorithm with a sliding window
/// to ensure compliance with Hyperliquid's rate limits.
///
/// # Thread Safety
/// Uses parking_lot::Mutex for efficient concurrent access.
///
/// # Example
/// ```rust
/// use hyperliquid_rust_sdk::rate_limiter::{RateLimiter, RateLimitConfig, RequestWeight};
///
/// let limiter = RateLimiter::new(RateLimitConfig::rest_api());
///
/// // Wait for capacity before making a request
/// limiter.acquire(RequestWeight::Exchange { batch_length: 0 }).await;
///
/// // Make your API call...
/// ```
#[derive(Clone)]
pub struct RateLimiter {
    config: RateLimitConfig,
    state: Arc<Mutex<RateLimiterState>>,
}

struct RateLimiterState {
    /// Sliding window of requests with timestamps
    requests: Vec<(Instant, u32)>,
    /// Total weight currently in the window
    current_weight: u32,
    /// Statistics
    stats: RateLimitStats,
}

impl RateLimiter {
    /// Create a new rate limiter with the given configuration
    pub fn new(config: RateLimitConfig) -> Self {
        Self {
            config,
            state: Arc::new(Mutex::new(RateLimiterState {
                requests: Vec::new(),
                current_weight: 0,
                stats: RateLimitStats::default(),
            })),
        }
    }

    /// Acquire capacity for a request, waiting if necessary
    ///
    /// This method will block until there is sufficient capacity available.
    /// It automatically cleans up old requests outside the time window.
    ///
    /// # Arguments
    /// * `weight` - The weight of the request
    pub async fn acquire(&self, weight: RequestWeight) {
        let weight_value = weight.weight();
        loop {
            let wait_duration = {
                let mut state = self.state.lock();

                // Clean up old requests outside the window
                self.cleanup_old_requests(&mut state);

                // Check if we have capacity
                let effective_max = self.config.effective_max_weight();
                if state.current_weight + weight_value <= effective_max {
                    // We have capacity, add the request
                    let now = Instant::now();
                    state.requests.push((now, weight_value));
                    state.current_weight += weight_value;

                    // Update stats
                    state.stats.total_requests += 1;
                    state.stats.total_weight_used += weight_value as u64;
                    state.stats.current_weight_in_window = state.current_weight;
                    state.stats.requests_in_current_window = state.requests.len() as u32;

                    return; // Successfully acquired
                }

                // Not enough capacity, calculate wait time
                self.calculate_wait_time(&state)
            };

            // Wait outside the lock
            if wait_duration > Duration::ZERO {
                let mut state = self.state.lock();
                state.stats.times_waited += 1;
                state.stats.total_wait_time_ms += wait_duration.as_millis() as u64;
                drop(state);

                sleep(wait_duration).await;
            } else {
                // Small yield to prevent tight spin
                tokio::task::yield_now().await;
            }
        }
    }

    /// Try to acquire capacity without waiting
    ///
    /// # Returns
    /// `true` if capacity was acquired, `false` if rate limited
    pub fn try_acquire(&self, weight: RequestWeight) -> bool {
        let weight_value = weight.weight();
        let mut state = self.state.lock();

        // Clean up old requests
        self.cleanup_old_requests(&mut state);

        // Check capacity
        let effective_max = self.config.effective_max_weight();
        if state.current_weight + weight_value <= effective_max {
            let now = Instant::now();
            state.requests.push((now, weight_value));
            state.current_weight += weight_value;

            // Update stats
            state.stats.total_requests += 1;
            state.stats.total_weight_used += weight_value as u64;
            state.stats.current_weight_in_window = state.current_weight;
            state.stats.requests_in_current_window = state.requests.len() as u32;

            true
        } else {
            false
        }
    }

    /// Get current rate limiter statistics
    pub fn stats(&self) -> RateLimitStats {
        let state = self.state.lock();
        state.stats.clone()
    }

    /// Reset statistics (useful for monitoring specific time periods)
    pub fn reset_stats(&self) {
        let mut state = self.state.lock();
        state.stats.total_requests = 0;
        state.stats.total_weight_used = 0;
        state.stats.times_waited = 0;
        state.stats.total_wait_time_ms = 0;
        // Keep current window stats
    }

    /// Get current capacity utilization (0.0 to 1.0)
    pub fn utilization(&self) -> f64 {
        let state = self.state.lock();
        let effective_max = self.config.effective_max_weight();
        if effective_max == 0 {
            0.0
        } else {
            state.current_weight as f64 / effective_max as f64
        }
    }

    /// Get remaining capacity in the current window
    pub fn remaining_capacity(&self) -> u32 {
        let state = self.state.lock();
        let effective_max = self.config.effective_max_weight();
        effective_max.saturating_sub(state.current_weight)
    }

    /// Estimate time until capacity is available
    pub fn time_until_capacity(&self, weight: RequestWeight) -> Duration {
        let state = self.state.lock();
        let weight_value = weight.weight();
        let effective_max = self.config.effective_max_weight();

        if state.current_weight + weight_value <= effective_max {
            Duration::ZERO
        } else {
            self.calculate_wait_time(&state)
        }
    }

    /// Clean up requests that have fallen outside the time window
    fn cleanup_old_requests(&self, state: &mut RateLimiterState) {
        let now = Instant::now();
        let window_start = now - self.config.window_duration;

        // Remove requests older than the window
        let mut removed_weight = 0u32;
        state.requests.retain(|(timestamp, weight)| {
            if *timestamp < window_start {
                removed_weight += weight;
                false
            } else {
                true
            }
        });

        state.current_weight = state.current_weight.saturating_sub(removed_weight);
        state.stats.current_weight_in_window = state.current_weight;
        state.stats.requests_in_current_window = state.requests.len() as u32;
    }

    /// Calculate how long to wait before retrying
    fn calculate_wait_time(&self, state: &RateLimiterState) -> Duration {
        if state.requests.is_empty() {
            return Duration::ZERO;
        }

        // Find the oldest request
        let oldest = state
            .requests
            .first()
            .map(|(timestamp, _)| *timestamp)
            .unwrap();

        let now = Instant::now();
        let elapsed = now.duration_since(oldest);

        // Wait until the oldest request falls out of the window
        if elapsed < self.config.window_duration {
            self.config.window_duration - elapsed + Duration::from_millis(10) // Small buffer
        } else {
            Duration::from_millis(10) // Small delay
        }
    }
}

/// Rate limiter builder for easy configuration
pub struct RateLimiterBuilder {
    config: RateLimitConfig,
}

impl RateLimiterBuilder {
    pub fn new() -> Self {
        Self {
            config: RateLimitConfig::default(),
        }
    }

    pub fn max_weight(mut self, max_weight: u32) -> Self {
        self.config.max_weight = max_weight;
        self
    }

    pub fn window_duration(mut self, duration: Duration) -> Self {
        self.config.window_duration = duration;
        self
    }

    pub fn safety_margin(mut self, margin: f64) -> Self {
        self.config.safety_margin = margin.clamp(0.0, 1.0);
        self
    }

    pub fn build(self) -> RateLimiter {
        RateLimiter::new(self.config)
    }
}

impl Default for RateLimiterBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_request_weight_calculation() {
        assert_eq!(RequestWeight::Exchange { batch_length: 0 }.weight(), 1);
        assert_eq!(RequestWeight::Exchange { batch_length: 39 }.weight(), 1);
        assert_eq!(RequestWeight::Exchange { batch_length: 40 }.weight(), 2);
        assert_eq!(RequestWeight::Exchange { batch_length: 79 }.weight(), 2);
        assert_eq!(RequestWeight::Exchange { batch_length: 80 }.weight(), 3);

        assert_eq!(RequestWeight::InfoLight.weight(), 2);
        assert_eq!(RequestWeight::InfoStandard.weight(), 20);
        assert_eq!(RequestWeight::InfoHeavy.weight(), 60);
        assert_eq!(RequestWeight::Explorer.weight(), 40);
    }

    #[test]
    fn test_pagination_weight() {
        let weight = RequestWeight::InfoStandard;
        assert_eq!(weight.with_pagination_weight(0), 20);
        assert_eq!(weight.with_pagination_weight(19), 20);
        assert_eq!(weight.with_pagination_weight(20), 21);
        assert_eq!(weight.with_pagination_weight(40), 22);
    }

    #[test]
    fn test_candle_pagination_weight() {
        let weight = RequestWeight::InfoLight;
        assert_eq!(weight.with_candle_pagination(0), 2);
        assert_eq!(weight.with_candle_pagination(59), 2);
        assert_eq!(weight.with_candle_pagination(60), 3);
        assert_eq!(weight.with_candle_pagination(120), 4);
    }

    #[tokio::test]
    async fn test_rate_limiter_basic() {
        let config = RateLimitConfig {
            max_weight: 100,
            window_duration: Duration::from_secs(1),
            safety_margin: 1.0,
        };
        let limiter = RateLimiter::new(config);

        // Should succeed immediately
        limiter.acquire(RequestWeight::Custom(10)).await;
        assert_eq!(limiter.remaining_capacity(), 90);

        // Acquire more
        limiter.acquire(RequestWeight::Custom(20)).await;
        assert_eq!(limiter.remaining_capacity(), 70);

        let stats = limiter.stats();
        assert_eq!(stats.total_requests, 2);
        assert_eq!(stats.total_weight_used, 30);
    }

    #[tokio::test]
    async fn test_rate_limiter_window_cleanup() {
        let config = RateLimitConfig {
            max_weight: 100,
            window_duration: Duration::from_millis(100),
            safety_margin: 1.0,
        };
        let limiter = RateLimiter::new(config);

        // Fill up capacity
        limiter.acquire(RequestWeight::Custom(100)).await;
        assert_eq!(limiter.remaining_capacity(), 0);

        // Wait for window to expire
        sleep(Duration::from_millis(150)).await;

        // Should have capacity again
        limiter.acquire(RequestWeight::Custom(50)).await;
        assert_eq!(limiter.remaining_capacity(), 50);
    }

    #[test]
    fn test_try_acquire() {
        let config = RateLimitConfig {
            max_weight: 100,
            window_duration: Duration::from_secs(1),
            safety_margin: 1.0,
        };
        let limiter = RateLimiter::new(config);

        assert!(limiter.try_acquire(RequestWeight::Custom(50)));
        assert!(limiter.try_acquire(RequestWeight::Custom(50)));
        assert!(!limiter.try_acquire(RequestWeight::Custom(1))); // Should fail
    }

    #[test]
    fn test_utilization() {
        let config = RateLimitConfig {
            max_weight: 100,
            window_duration: Duration::from_secs(1),
            safety_margin: 1.0,
        };
        let limiter = RateLimiter::new(config);

        assert_eq!(limiter.utilization(), 0.0);

        limiter.try_acquire(RequestWeight::Custom(50));
        assert_eq!(limiter.utilization(), 0.5);

        limiter.try_acquire(RequestWeight::Custom(25));
        assert_eq!(limiter.utilization(), 0.75);
    }

    #[test]
    fn test_safety_margin() {
        let config = RateLimitConfig {
            max_weight: 100,
            window_duration: Duration::from_secs(1),
            safety_margin: 0.9,
        };

        // Effective max is 90 with 90% safety margin
        assert_eq!(config.effective_max_weight(), 90);

        let limiter = RateLimiter::new(config);
        assert!(limiter.try_acquire(RequestWeight::Custom(90)));
        assert!(!limiter.try_acquire(RequestWeight::Custom(1)));
    }
}
