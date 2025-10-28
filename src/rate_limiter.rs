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
            max_weight: 1200, // REST API default
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
    /// Info requests with weight 2: l2Book, allMids, clearinghouseState, orderStatus, spotClearinghouseState, exchangeStatus
    InfoLight,
    /// Info requests with weight 20 (Most other info requests)
    InfoStandard,
    /// Info requests with weight 60: userRole
    InfoHeavy,
    /// Explorer API requests with weight 40
    Explorer,
    /// Info request type for candle snapshots (base weight 20)
    InfoCandleSnapshot,
    /// Info request type for endpoints with +1 weight per 20 items (base weight 20)
    InfoPaginatedStandard,
    /// Custom weight
    Custom(u32),
}

impl RequestWeight {
    /// Calculate the base weight for this request type (without pagination)
    pub fn base_weight(&self) -> u32 {
        match self {
            RequestWeight::Exchange { batch_length } => 1 + (*batch_length as u32 / 40),
            RequestWeight::InfoLight => 2,
            RequestWeight::InfoStandard => 20,
            RequestWeight::InfoHeavy => 60,
            RequestWeight::Explorer => 40,
            RequestWeight::InfoCandleSnapshot => 20, // Base weight for candles is standard
            RequestWeight::InfoPaginatedStandard => 20, // Base weight is standard
            RequestWeight::Custom(w) => *w,
        }
    }

    /// Calculate the total weight including pagination costs
    ///
    /// # Arguments
    /// * `items_returned` - The number of items returned in the response (for pagination calculation).
    ///                      Should be 0 if the request type doesn't support pagination.
    ///
    /// # Returns
    /// Total calculated weight for the request.
    pub fn total_weight(&self, items_returned: usize) -> u32 {
        let base = self.base_weight();
        let pagination_cost = match self {
            // Specific pagination rules
            RequestWeight::InfoCandleSnapshot => items_returned as u32 / 60, // +1 per 60 items
            RequestWeight::InfoPaginatedStandard => items_returned as u32 / 20, // +1 per 20 items

            // Endpoints without specific pagination costs mentioned just use base weight
            RequestWeight::Exchange { .. } | // Exchange actions don't have item-based pagination cost
            RequestWeight::InfoLight |
            RequestWeight::InfoStandard | // Assuming standard doesn't paginate unless specified
            RequestWeight::InfoHeavy |
            RequestWeight::Explorer | // Assuming explorer doesn't paginate unless specified
            RequestWeight::Custom(_) => 0,
        };
        base + pagination_cost
    }
}

// ... (RateLimitStats, RateLimiterState, RateLimiter, RateLimiterBuilder remain the same) ...

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

    /// Acquire capacity for a request, waiting if necessary.
    /// Use this for essential requests that must eventually succeed.
    ///
    /// # Arguments
    /// * `weight_type` - The type of request being made.
    /// * `items_returned` - Number of items expected/returned (for pagination cost). Use 0 if not applicable.
    pub async fn acquire(&self, weight_type: RequestWeight, items_returned: usize) {
        let weight_value = weight_type.total_weight(items_returned); // Calculate total weight
        loop {
            let wait_duration = {
                let mut state = self.state.lock();
                self.cleanup_old_requests(&mut state); // Clean before checking

                let effective_max = self.config.effective_max_weight();
                if state.current_weight + weight_value <= effective_max {
                    // Capacity available
                    let now = Instant::now();
                    state.requests.push((now, weight_value));
                    state.current_weight += weight_value;

                    // Update stats
                    state.stats.total_requests += 1;
                    state.stats.total_weight_used += weight_value as u64;
                    state.stats.current_weight_in_window = state.current_weight;
                    state.stats.requests_in_current_window = state.requests.len() as u32;

                    return; // Acquired successfully
                }
                // No capacity, calculate wait time based on oldest request
                self.calculate_wait_time(&state)
            };

            // Wait outside the lock
            if wait_duration > Duration::ZERO {
                { // Scope for lock guard
                    let mut state = self.state.lock();
                    state.stats.times_waited += 1;
                    state.stats.total_wait_time_ms += wait_duration.as_millis() as u64;
                } // Lock released here
                sleep(wait_duration).await;
            } else {
                tokio::task::yield_now().await; // Yield if no wait needed but failed acquire
            }
        }
    }

    /// Try to acquire capacity without waiting.
    /// Use this for non-essential requests or checks.
    ///
    /// # Arguments
    /// * `weight_type` - The type of request being made.
    /// * `items_returned` - Number of items expected/returned (for pagination cost). Use 0 if not applicable.
    ///
    /// # Returns
    /// `true` if capacity was acquired, `false` otherwise.
    pub fn try_acquire(&self, weight_type: RequestWeight, items_returned: usize) -> bool {
        let weight_value = weight_type.total_weight(items_returned); // Calculate total weight
        let mut state = self.state.lock();
        self.cleanup_old_requests(&mut state); // Clean before checking

        let effective_max = self.config.effective_max_weight();
        if state.current_weight + weight_value <= effective_max {
            // Capacity available
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
            // No capacity
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

    /// Estimate time until capacity is available for a given request weight.
    /// Note: This is an estimate, the actual wait time might differ slightly.
    pub fn time_until_capacity(&self, weight_type: RequestWeight, items_returned: usize) -> Duration {
        let weight_value = weight_type.total_weight(items_returned);
        let state = self.state.lock();
        let effective_max = self.config.effective_max_weight();

        if state.current_weight + weight_value <= effective_max {
            Duration::ZERO
        } else {
            // Calculate wait time based on oldest requests needed to free up space
            self.calculate_wait_time_for_weight(&state, weight_value)
        }
    }


    /// Clean up requests that have fallen outside the time window
    fn cleanup_old_requests(&self, state: &mut RateLimiterState) {
        let now = Instant::now();
        let window_start = now.checked_sub(self.config.window_duration);

        // If window_start is None (e.g., duration is zero or negative), don't clean up
        let window_start = match window_start {
             Some(start) => start,
             None => return,
        };


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

    /// Calculate how long to wait until the oldest request expires
    fn calculate_wait_time(&self, state: &RateLimiterState) -> Duration {
        if state.requests.is_empty() {
            return Duration::ZERO;
        }

        // Find the oldest request
        let oldest_timestamp = state.requests[0].0;
        let now = Instant::now();

        // Time elapsed since the oldest request
        let elapsed = now.duration_since(oldest_timestamp);

        // Required wait time for the oldest request to expire
        if elapsed < self.config.window_duration {
             self.config.window_duration - elapsed + Duration::from_millis(5) // Add small buffer
        } else {
             Duration::from_millis(5) // Already expired, small delay before retry
        }
    }

     /// Calculate how long to wait until enough capacity is freed up for `needed_weight`.
     fn calculate_wait_time_for_weight(&self, state: &RateLimiterState, needed_weight: u32) -> Duration {
        if state.requests.is_empty() {
            return Duration::ZERO; // Should not happen if called correctly
        }

        let effective_max = self.config.effective_max_weight();
        let weight_to_free = (state.current_weight + needed_weight).saturating_sub(effective_max);

        if weight_to_free == 0 {
            return Duration::ZERO; // Enough capacity already
        }

        let mut cumulative_weight = 0;
        let now = Instant::now();

        for (timestamp, weight) in state.requests.iter() {
            cumulative_weight += weight;
            if cumulative_weight >= weight_to_free {
                // This is the request that needs to expire (or one after it)
                let elapsed = now.duration_since(*timestamp);
                if elapsed < self.config.window_duration {
                    // Return the time until this request expires
                    return self.config.window_duration - elapsed + Duration::from_millis(5); // Small buffer
                } else {
                    // This request already expired, something might be slightly off, suggest short wait
                    return Duration::from_millis(5);
                }
            }
        }

        // Should theoretically not reach here if weight_to_free > 0
        // Fallback to waiting for the oldest request
        self.calculate_wait_time(state)
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
        // Exchange weights
        assert_eq!(RequestWeight::Exchange { batch_length: 0 }.base_weight(), 1);
        assert_eq!(RequestWeight::Exchange { batch_length: 39 }.base_weight(), 1);
        assert_eq!(RequestWeight::Exchange { batch_length: 40 }.base_weight(), 2);
        assert_eq!(RequestWeight::Exchange { batch_length: 79 }.base_weight(), 2);
        assert_eq!(RequestWeight::Exchange { batch_length: 80 }.base_weight(), 3);

        // Info weights
        assert_eq!(RequestWeight::InfoLight.base_weight(), 2);
        assert_eq!(RequestWeight::InfoStandard.base_weight(), 20);
        assert_eq!(RequestWeight::InfoHeavy.base_weight(), 60);
        assert_eq!(RequestWeight::Explorer.base_weight(), 40);
        assert_eq!(RequestWeight::InfoCandleSnapshot.base_weight(), 20);
        assert_eq!(RequestWeight::InfoPaginatedStandard.base_weight(), 20);
        assert_eq!(RequestWeight::Custom(5).base_weight(), 5);
    }

     #[test]
    fn test_pagination_weight() {
        // Standard pagination (+1 per 20)
        let weight_paginated = RequestWeight::InfoPaginatedStandard;
        assert_eq!(weight_paginated.total_weight(0), 20); // Base
        assert_eq!(weight_paginated.total_weight(19), 20); // Base
        assert_eq!(weight_paginated.total_weight(20), 21); // Base + 1
        assert_eq!(weight_paginated.total_weight(39), 21); // Base + 1
        assert_eq!(weight_paginated.total_weight(40), 22); // Base + 2

        // Candle pagination (+1 per 60)
        let weight_candle = RequestWeight::InfoCandleSnapshot;
        assert_eq!(weight_candle.total_weight(0), 20);   // Base
        assert_eq!(weight_candle.total_weight(59), 20);  // Base
        assert_eq!(weight_candle.total_weight(60), 21);  // Base + 1
        assert_eq!(weight_candle.total_weight(119), 21); // Base + 1
        assert_eq!(weight_candle.total_weight(120), 22); // Base + 2

        // Non-paginated types should ignore items_returned
        let weight_exchange = RequestWeight::Exchange { batch_length: 0 };
        assert_eq!(weight_exchange.total_weight(100), 1); // Only base weight applies
        let weight_light = RequestWeight::InfoLight;
        assert_eq!(weight_light.total_weight(100), 2); // Only base weight applies
    }

    #[tokio::test]
    async fn test_rate_limiter_basic_acquire() {
        let config = RateLimitConfig {
            max_weight: 100,
            window_duration: Duration::from_secs(1),
            safety_margin: 1.0,
        };
        let limiter = RateLimiter::new(config);

        // Acquire weight, no items returned
        limiter.acquire(RequestWeight::Custom(10), 0).await;
        assert_eq!(limiter.remaining_capacity(), 90);

        // Acquire more
        limiter.acquire(RequestWeight::Custom(20), 0).await;
        assert_eq!(limiter.remaining_capacity(), 70);

        let stats = limiter.stats();
        assert_eq!(stats.total_requests, 2);
        assert_eq!(stats.total_weight_used, 30);
    }

    #[tokio::test]
    async fn test_rate_limiter_acquire_with_pagination() {
        let config = RateLimitConfig {
            max_weight: 100,
            window_duration: Duration::from_secs(1),
            safety_margin: 1.0,
        };
        let limiter = RateLimiter::new(config);

        // Acquire paginated weight (base 20 + 2 for 40 items)
        limiter.acquire(RequestWeight::InfoPaginatedStandard, 40).await;
        assert_eq!(limiter.remaining_capacity(), 100 - 22); // 78

        let stats = limiter.stats();
        assert_eq!(stats.total_requests, 1);
        assert_eq!(stats.total_weight_used, 22);
    }

    #[tokio::test]
    async fn test_rate_limiter_wait() {
        let config = RateLimitConfig {
            max_weight: 50,
            window_duration: Duration::from_millis(200),
            safety_margin: 1.0,
        };
        let limiter = RateLimiter::new(config);

        // Acquire initial capacity
        limiter.acquire(RequestWeight::Custom(40), 0).await;
        let start = Instant::now();

        // This should wait
        limiter.acquire(RequestWeight::Custom(20), 0).await;
        let elapsed = start.elapsed();

        // Should have waited roughly until the first request expired
        assert!(elapsed >= Duration::from_millis(190) && elapsed < Duration::from_millis(300));
        assert!(limiter.stats().times_waited >= 1);
    }


    #[test]
    fn test_try_acquire() {
        let config = RateLimitConfig {
            max_weight: 100,
            window_duration: Duration::from_secs(1),
            safety_margin: 1.0,
        };
        let limiter = RateLimiter::new(config);

        assert!(limiter.try_acquire(RequestWeight::Custom(50), 0));
        assert!(limiter.try_acquire(RequestWeight::Custom(50), 0));
        assert!(!limiter.try_acquire(RequestWeight::Custom(1), 0)); // Should fail
    }

     #[test]
    fn test_try_acquire_with_pagination() {
        let config = RateLimitConfig {
            max_weight: 30,
            window_duration: Duration::from_secs(1),
            safety_margin: 1.0,
        };
        let limiter = RateLimiter::new(config);

        // Acquire paginated (base 20 + 1 for 20 items = 21 weight)
        assert!(limiter.try_acquire(RequestWeight::InfoPaginatedStandard, 20));
        assert_eq!(limiter.remaining_capacity(), 30 - 21); // 9 remaining

        // Try to acquire another standard request (weight 20) - should fail
        assert!(!limiter.try_acquire(RequestWeight::InfoStandard, 0));

         // Try to acquire a light request (weight 2) - should succeed
        assert!(limiter.try_acquire(RequestWeight::InfoLight, 0));
        assert_eq!(limiter.remaining_capacity(), 9 - 2); // 7 remaining
    }


    #[test]
    fn test_safety_margin() {
        let config = RateLimitConfig {
            max_weight: 100,
            window_duration: Duration::from_secs(1),
            safety_margin: 0.9, // Effective max weight = 90
        };
        let limiter = RateLimiter::new(config);

        assert_eq!(limiter.remaining_capacity(), 90);
        assert!(limiter.try_acquire(RequestWeight::Custom(90), 0));
        assert_eq!(limiter.remaining_capacity(), 0);
        assert!(!limiter.try_acquire(RequestWeight::Custom(1), 0));
    }

    #[tokio::test]
    async fn test_time_until_capacity() {
        let config = RateLimitConfig {
            max_weight: 50,
            window_duration: Duration::from_millis(500),
            safety_margin: 1.0,
        };
        let limiter = RateLimiter::new(config);

        // Acquire 40 weight
        limiter.acquire(RequestWeight::Custom(40), 0).await;
        sleep(Duration::from_millis(100)).await; // Wait 100ms

        // Check time until 20 weight is available
        // Need to free 10 weight (40 + 20 > 50). Oldest request (40) was 100ms ago.
        // It expires in 500 - 100 = 400ms.
        let wait_time = limiter.time_until_capacity(RequestWeight::Custom(20), 0);
        println!("Estimated wait time: {:?}", wait_time);
        assert!(wait_time > Duration::from_millis(390) && wait_time < Duration::from_millis(410)); // Around 400ms + buffer

         // Check time until 5 weight is available (should be 0)
        let wait_time_small = limiter.time_until_capacity(RequestWeight::Custom(5), 0);
         assert_eq!(wait_time_small, Duration::ZERO);
    }
}
