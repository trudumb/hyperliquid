// ============================================================================
// Intelligent Order Churn Management
// ============================================================================
//
// This component implements model-driven order refresh logic that decides
// when to cancel and replace orders based on:
// - Fill rate statistics (orders getting filled vs. ignored)
// - Market volatility (faster churning in volatile markets)
// - Adverse selection risk (keeping stale orders in trending markets)
// - Queue position deterioration (competitor orders jumping ahead)
// - Spread deviation from optimal (market moved away from quotes)
//
// # Key Innovation: Adaptive Order Lifetime
//
// Instead of canceling orders on every tick (high churn, high fees) or
// using fixed time thresholds (inflexible), this manager uses a dynamic
// model that adapts order lifetime to current market conditions:
//
// - High fill rate → Keep orders longer (they're working)
// - Low fill rate → Refresh sooner (quotes not competitive)
// - High volatility → Refresh faster (market moving rapidly)
// - Low volatility → Keep orders longer (stable environment)
// - High adverse selection → Refresh proactively (being picked off)
// - Trending market → Refresh frequently (don't get run over)
//
// # Example Usage
//
// ```rust
// let mut churn_mgr = OrderChurnManager::new(config);
//
// // Update fill statistics
// churn_mgr.record_fill(level, true, fill_time_ms);
//
// // Check if order should be refreshed
// if churn_mgr.should_refresh_order(order, market_state) {
//     cancel_and_replace(order);
// }
// ```

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// Configuration for order churn management
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderChurnConfig {
    /// Base minimum order lifetime in milliseconds (default: 500ms)
    /// Orders younger than this are NEVER canceled (prevents rapid churn)
    pub min_order_lifetime_ms: u64,

    /// Maximum order lifetime in milliseconds (default: 5000ms)
    /// Orders older than this are ALWAYS refreshed (prevents stale quotes)
    pub max_order_lifetime_ms: u64,

    /// Target fill rate per level (default: 0.2 = 20% of orders should fill)
    /// If actual fill rate is below this, we refresh more aggressively
    pub target_fill_rate: f64,

    /// Lookback window for fill rate calculation in seconds (default: 300s = 5min)
    pub fill_rate_window_sec: u64,

    /// Volatility scaling factor (default: 100.0)
    /// Higher volatility → shorter order lifetime
    /// lifetime_ms *= 1.0 / (1.0 + volatility_bps / volatility_scaling)
    pub volatility_scaling_factor: f64,

    /// Spread deviation threshold in basis points (default: 2.0 bps)
    /// If market moves such that order is >2bps from optimal, refresh it
    pub spread_deviation_threshold_bps: f64,

    /// Adverse selection sensitivity (default: 0.5)
    /// Multiplier for how much adverse selection reduces order lifetime
    /// 0.0 = ignore adverse selection, 1.0 = aggressive refresh under AS
    pub adverse_selection_sensitivity: f64,

    /// Enable queue position tracking (default: false)
    /// If true, refresh orders when LOB depth ahead increases significantly
    pub enable_queue_position_tracking: bool,

    /// Queue deterioration threshold (default: 2.0x)
    /// Refresh if queue size ahead of us increases by this factor
    pub queue_deterioration_threshold: f64,
}

impl Default for OrderChurnConfig {
    fn default() -> Self {
        Self {
            min_order_lifetime_ms: 500,
            max_order_lifetime_ms: 5000,
            target_fill_rate: 0.15,
            fill_rate_window_sec: 300,
            volatility_scaling_factor: 100.0,
            spread_deviation_threshold_bps: 2.0,
            adverse_selection_sensitivity: 0.5,
            enable_queue_position_tracking: false,
            queue_deterioration_threshold: 2.0,
        }
    }
}

/// Statistics for fill rate tracking per level
#[derive(Debug, Clone)]
struct LevelFillStats {
    /// Recent fill events (timestamp_ms, was_filled)
    /// Stores (placement_time, fill_time, side_is_bid)
    recent_fills: VecDeque<(u64, Option<u64>, bool)>,

    /// Recent order placements that timed out without fill
    /// Stores (placement_time, cancel_time, side_is_bid)
    recent_timeouts: VecDeque<(u64, u64, bool)>,
}

impl LevelFillStats {
    fn new() -> Self {
        Self {
            recent_fills: VecDeque::new(),
            recent_timeouts: VecDeque::new(),
        }
    }

    /// Calculate fill rate for the specified side
    fn calculate_fill_rate(&self, is_bid: bool, window_cutoff_ms: u64) -> f64 {
        let fills = self.recent_fills
            .iter()
            .filter(|(placement_time, _, side)| *side == is_bid && *placement_time >= window_cutoff_ms)
            .count();

        let timeouts = self.recent_timeouts
            .iter()
            .filter(|(placement_time, _, side)| *side == is_bid && *placement_time >= window_cutoff_ms)
            .count();

        let total = fills + timeouts;
        if total == 0 {
            return 0.0; // Unknown, assume neutral
        }

        fills as f64 / total as f64
    }

    /// Prune old events outside the window
    fn prune(&mut self, window_cutoff_ms: u64) {
        self.recent_fills.retain(|(t, _, _)| *t >= window_cutoff_ms);
        self.recent_timeouts.retain(|(t, _, _)| *t >= window_cutoff_ms);
    }
}

/// Market state snapshot for churn decisions
#[derive(Debug, Clone)]
pub struct MarketChurnState {
    /// Current timestamp in milliseconds
    pub current_time_ms: u64,

    /// Current L2 mid price
    pub mid_price: f64,

    /// Current volatility estimate in bps
    pub volatility_bps: f64,

    /// Current adverse selection estimate in bps (positive = buying pressure)
    pub adverse_selection_bps: f64,

    /// Order book imbalance (0.5 = balanced, >0.5 = bid heavy)
    pub lob_imbalance: f64,

    /// Best bid price
    pub best_bid: Option<f64>,

    /// Best ask price
    pub best_ask: Option<f64>,

    /// Size ahead of us in queue at each level (if tracking enabled)
    pub queue_depth_ahead: Option<Vec<(f64, f64)>>, // (price, size_ahead)
}

/// Order metadata for churn decisions
#[derive(Debug, Clone)]
pub struct OrderMetadata {
    /// Order ID
    pub oid: u64,

    /// Price
    pub price: f64,

    /// Size
    pub size: f64,

    /// Is buy order
    pub is_buy: bool,

    /// Level (0 = L1, 1 = L2, etc.)
    pub level: usize,

    /// Placement timestamp in milliseconds
    pub placement_time_ms: u64,

    /// Target price (optimal quote from model)
    pub target_price: f64,

    /// Queue position when placed (size ahead in queue)
    pub initial_queue_size_ahead: Option<f64>,
}

/// Intelligent order churn manager
pub struct OrderChurnManager {
    config: OrderChurnConfig,

    /// Fill statistics per level (up to 5 levels)
    level_stats: Vec<LevelFillStats>,

    /// Last update timestamp (for pruning)
    last_prune_time_ms: u64,
}

impl OrderChurnManager {
    /// Create a new order churn manager
    pub fn new(config: OrderChurnConfig) -> Self {
        let level_stats = (0..5).map(|_| LevelFillStats::new()).collect();

        Self {
            config,
            level_stats,
            last_prune_time_ms: 0,
        }
    }

    /// Record a fill event for fill rate tracking
    pub fn record_fill(
        &mut self,
        level: usize,
        is_bid: bool,
        placement_time_ms: u64,
        fill_time_ms: u64,
    ) {
        if level >= self.level_stats.len() {
            return;
        }

        self.level_stats[level]
            .recent_fills
            .push_back((placement_time_ms, Some(fill_time_ms), is_bid));

        // Limit memory
        if self.level_stats[level].recent_fills.len() > 1000 {
            self.level_stats[level].recent_fills.pop_front();
        }
    }

    /// Record a timeout event (order canceled without fill)
    pub fn record_timeout(
        &mut self,
        level: usize,
        is_bid: bool,
        placement_time_ms: u64,
        cancel_time_ms: u64,
    ) {
        if level >= self.level_stats.len() {
            return;
        }

        self.level_stats[level]
            .recent_timeouts
            .push_back((placement_time_ms, cancel_time_ms, is_bid));

        // Limit memory
        if self.level_stats[level].recent_timeouts.len() > 1000 {
            self.level_stats[level].recent_timeouts.pop_front();
        }
    }

    /// Decide if an order should be refreshed based on model
    ///
    /// Returns (should_refresh, reason)
    pub fn should_refresh_order(
        &mut self,
        order: &OrderMetadata,
        market_state: &MarketChurnState,
    ) -> (bool, &'static str) {
        let order_age_ms = market_state.current_time_ms.saturating_sub(order.placement_time_ms);

        // Rule 1: NEVER cancel orders younger than min_lifetime (hard floor)
        if order_age_ms < self.config.min_order_lifetime_ms {
            return (false, "too_young");
        }

        // Rule 2: ALWAYS cancel orders older than max_lifetime (hard ceiling)
        if order_age_ms >= self.config.max_order_lifetime_ms {
            return (true, "max_age");
        }

        // Rule 3: Check spread deviation (order moved away from optimal)
        let spread_deviation_bps = (order.price - order.target_price).abs() / market_state.mid_price * 10000.0;
        if spread_deviation_bps > self.config.spread_deviation_threshold_bps {
            return (true, "spread_deviation");
        }

        // Rule 4: Calculate adaptive lifetime based on market conditions
        let adaptive_lifetime_ms = self.calculate_adaptive_lifetime(order, market_state);

        if order_age_ms >= adaptive_lifetime_ms {
            return (true, "adaptive_model");
        }

        // Rule 5: Check queue position deterioration (if enabled)
        if self.config.enable_queue_position_tracking {
            if let Some(queue_ahead) = self.check_queue_deterioration(order, market_state) {
                if queue_ahead > self.config.queue_deterioration_threshold {
                    return (true, "queue_deterioration");
                }
            }
        }

        (false, "keep")
    }

    /// Calculate adaptive order lifetime based on market conditions
    fn calculate_adaptive_lifetime(
        &mut self,
        order: &OrderMetadata,
        market_state: &MarketChurnState,
    ) -> u64 {
        // Start with base lifetime (midpoint of min/max)
        let base_lifetime_ms = (self.config.min_order_lifetime_ms + self.config.max_order_lifetime_ms) / 2;
        let mut lifetime_ms = base_lifetime_ms as f64;

        // Prune old stats periodically
        if market_state.current_time_ms > self.last_prune_time_ms + 10000 {
            let window_cutoff = market_state.current_time_ms - (self.config.fill_rate_window_sec * 1000);
            for stats in &mut self.level_stats {
                stats.prune(window_cutoff);
            }
            self.last_prune_time_ms = market_state.current_time_ms;
        }

        // Factor 1: Fill rate adjustment
        // Low fill rate → shorter lifetime (quotes not competitive)
        // High fill rate → longer lifetime (quotes working well)
        if order.level < self.level_stats.len() {
            let window_cutoff = market_state.current_time_ms.saturating_sub(self.config.fill_rate_window_sec * 1000);
            let fill_rate = self.level_stats[order.level].calculate_fill_rate(order.is_buy, window_cutoff);

            if fill_rate < self.config.target_fill_rate && fill_rate > 0.0 {
                // Fill rate is low, reduce lifetime to refresh more aggressively
                let fill_rate_ratio = fill_rate / self.config.target_fill_rate;
                lifetime_ms *= 0.5 + (fill_rate_ratio * 0.5); // Scale between 0.5x and 1.0x
            } else if fill_rate > self.config.target_fill_rate {
                // Fill rate is high, extend lifetime (orders working well)
                let fill_rate_ratio = self.config.target_fill_rate / fill_rate.max(0.01);
                lifetime_ms *= 1.0 + (fill_rate_ratio * 0.5); // Scale up to 1.5x
            }
        }

        // Factor 2: Volatility adjustment
        // High volatility → shorter lifetime (market moving fast)
        let vol_factor = 1.0 / (1.0 + market_state.volatility_bps / self.config.volatility_scaling_factor);
        lifetime_ms *= vol_factor;

        // Factor 3: Adverse selection adjustment
        // If order is on the side being adversely selected, reduce lifetime
        let as_factor = if (order.is_buy && market_state.adverse_selection_bps > 5.0)
            || (!order.is_buy && market_state.adverse_selection_bps < -5.0)
        {
            // Order is on the "wrong" side (likely to be picked off)
            1.0 - (market_state.adverse_selection_bps.abs() / 50.0 * self.config.adverse_selection_sensitivity)
        } else {
            1.0
        };
        lifetime_ms *= as_factor.max(0.5); // Don't reduce below 50%

        // Factor 4: LOB imbalance adjustment
        // If LOB is heavily imbalanced against our side, refresh sooner
        let imbalance_factor = if (order.is_buy && market_state.lob_imbalance < 0.3)
            || (!order.is_buy && market_state.lob_imbalance > 0.7)
        {
            0.8 // Reduce lifetime by 20% when LOB is against us
        } else {
            1.0
        };
        lifetime_ms *= imbalance_factor;

        // Clamp to configured bounds
        lifetime_ms
            .max(self.config.min_order_lifetime_ms as f64)
            .min(self.config.max_order_lifetime_ms as f64) as u64
    }

    /// Check if queue position has deteriorated significantly
    fn check_queue_deterioration(
        &self,
        order: &OrderMetadata,
        market_state: &MarketChurnState,
    ) -> Option<f64> {
        if let (Some(initial_size), Some(queue_depths)) =
            (order.initial_queue_size_ahead, &market_state.queue_depth_ahead)
        {
            // Find current queue size at our price level
            if let Some((_, current_size)) = queue_depths
                .iter()
                .find(|(price, _)| (*price - order.price).abs() < 0.0001)
            {
                if initial_size > 0.0 {
                    return Some(*current_size / initial_size);
                }
            }
        }
        None
    }

    /// Get current fill rate for a level and side
    pub fn get_fill_rate(&self, level: usize, is_bid: bool, current_time_ms: u64) -> f64 {
        if level >= self.level_stats.len() {
            return 0.0;
        }

        let window_cutoff = current_time_ms.saturating_sub(self.config.fill_rate_window_sec * 1000);
        self.level_stats[level].calculate_fill_rate(is_bid, window_cutoff)
    }

    /// Get configuration reference
    pub fn config(&self) -> &OrderChurnConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_min_lifetime_prevents_churn() {
        let config = OrderChurnConfig {
            min_order_lifetime_ms: 500,
            ..Default::default()
        };
        let mut mgr = OrderChurnManager::new(config);

        let order = OrderMetadata {
            oid: 123,
            price: 100.0,
            size: 1.0,
            is_buy: true,
            level: 0,
            placement_time_ms: 1000,
            target_price: 100.0,
            initial_queue_size_ahead: None,
        };

        let market = MarketChurnState {
            current_time_ms: 1400, // 400ms old
            mid_price: 100.0,
            volatility_bps: 100.0,
            adverse_selection_bps: 0.0,
            lob_imbalance: 0.5,
            best_bid: Some(99.0),
            best_ask: Some(101.0),
            queue_depth_ahead: None,
        };

        let (should_refresh, reason) = mgr.should_refresh_order(&order, &market);
        assert!(!should_refresh);
        assert_eq!(reason, "too_young");
    }

    #[test]
    fn test_max_lifetime_forces_refresh() {
        let config = OrderChurnConfig {
            min_order_lifetime_ms: 500,
            max_order_lifetime_ms: 5000,
            ..Default::default()
        };
        let mut mgr = OrderChurnManager::new(config);

        let order = OrderMetadata {
            oid: 123,
            price: 100.0,
            size: 1.0,
            is_buy: true,
            level: 0,
            placement_time_ms: 1000,
            target_price: 100.0,
            initial_queue_size_ahead: None,
        };

        let market = MarketChurnState {
            current_time_ms: 7000, // 6000ms old (> max)
            mid_price: 100.0,
            volatility_bps: 100.0,
            adverse_selection_bps: 0.0,
            lob_imbalance: 0.5,
            best_bid: Some(99.0),
            best_ask: Some(101.0),
            queue_depth_ahead: None,
        };

        let (should_refresh, reason) = mgr.should_refresh_order(&order, &market);
        assert!(should_refresh);
        assert_eq!(reason, "max_age");
    }

    #[test]
    fn test_spread_deviation_triggers_refresh() {
        let config = OrderChurnConfig {
            min_order_lifetime_ms: 500,
            spread_deviation_threshold_bps: 2.0,
            ..Default::default()
        };
        let mut mgr = OrderChurnManager::new(config);

        let order = OrderMetadata {
            oid: 123,
            price: 100.0,
            size: 1.0,
            is_buy: true,
            level: 0,
            placement_time_ms: 1000,
            target_price: 100.5, // 50 bps away
            initial_queue_size_ahead: None,
        };

        let market = MarketChurnState {
            current_time_ms: 2000,
            mid_price: 100.0,
            volatility_bps: 100.0,
            adverse_selection_bps: 0.0,
            lob_imbalance: 0.5,
            best_bid: Some(99.0),
            best_ask: Some(101.0),
            queue_depth_ahead: None,
        };

        let (should_refresh, reason) = mgr.should_refresh_order(&order, &market);
        assert!(should_refresh);
        assert_eq!(reason, "spread_deviation");
    }
}
