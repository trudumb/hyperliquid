//! Multi-Level Market Making Optimization
//!
//! This module implements multi-level quoting optimization combined with
//! robust control and dynamic level count optimization.
//!
//! Key Features:
//! - Multi-level quote optimization
//! - Robust control with parameter uncertainty
//! - Dynamic level count optimization
//! - Kelly-inspired size allocation
//! - Momentum-based quote tightening

use serde::{Deserialize, Serialize};
use super::super::ConstrainedTuningParams;
use super::hawkes::HawkesFillModel;

/// Multi-level quote specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiLevelControl {
    /// Bid levels: (offset_bps, size)
    pub bid_levels: Vec<(f64, f64)>,

    /// Ask levels: (offset_bps, size)
    pub ask_levels: Vec<(f64, f64)>,

    /// Taker order rates (for urgent inventory management)
    pub taker_buy_rate: f64,
    pub taker_sell_rate: f64,

    /// Flag for liquidation mode
    pub liquidate: bool,
}

impl MultiLevelControl {
    /// Create symmetric multi-level control
    pub fn symmetric(num_levels: usize, base_offset_bps: f64, level_spacing_bps: f64, base_size: f64) -> Self {
        let mut bid_levels = Vec::new();
        let mut ask_levels = Vec::new();

        for i in 0..num_levels {
            let offset = base_offset_bps + (i as f64 * level_spacing_bps);
            let size = base_size * 0.9_f64.powi(i as i32); // Decreasing size

            bid_levels.push((offset, size));
            ask_levels.push((offset, size));
        }

        Self {
            bid_levels,
            ask_levels,
            taker_buy_rate: 0.0,
            taker_sell_rate: 0.0,
            liquidate: false,
        }
    }

    /// Get total size across all bid levels
    pub fn total_bid_size(&self) -> f64 {
        self.bid_levels.iter().map(|(_, size)| size).sum()
    }

    /// Get total size across all ask levels
    pub fn total_ask_size(&self) -> f64 {
        self.ask_levels.iter().map(|(_, size)| size).sum()
    }

    /// Check if this is valid
    pub fn validate(&self) -> Result<(), String> {
        if self.bid_levels.is_empty() || self.ask_levels.is_empty() {
            return Err("Must have at least one level on each side".to_string());
        }

        // Check that levels are ordered (wider spreads for higher levels)
        for i in 1..self.bid_levels.len() {
            if self.bid_levels[i].0 <= self.bid_levels[i-1].0 {
                return Err("Bid levels must have increasing spreads".to_string());
            }
        }

        for i in 1..self.ask_levels.len() {
            if self.ask_levels[i].0 <= self.ask_levels[i-1].0 {
                return Err("Ask levels must have increasing spreads".to_string());
            }
        }

        Ok(())
    }

    /// Get log string for monitoring
    pub fn to_log_string(&self) -> String {
        let bid_l1 = self.bid_levels.first().map(|(o, s)| format!("{}bps({})", o, s)).unwrap_or_default();
        let ask_l1 = self.ask_levels.first().map(|(o, s)| format!("{}bps({})", o, s)).unwrap_or_default();

        format!(
            "MultiLevel[{} levels | L1: bid={}, ask={} | taker: buy={:.2}, sell={:.2}]",
            self.bid_levels.len(),
            bid_l1,
            ask_l1,
            self.taker_buy_rate,
            self.taker_sell_rate
        )
    }
}

/// Configuration for multi-level optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiLevelConfig {
    /// Maximum number of levels (1-10)
    pub max_levels: usize,

    /// Minimum profitable spread (covers fees + edge)
    pub min_profitable_spread_bps: f64,

    /// Default spacing between levels (bps)
    pub level_spacing_bps: f64,

    /// Total size budget per side
    pub total_size_per_side: f64,

    /// Inventory risk limit (fraction of max position)
    pub inventory_risk_limit: f64,

    /// Directional aggression factor (how much to bias based on signal)
    pub directional_aggression: f64,

    /// Momentum detection threshold (excitation multiplier)
    pub momentum_threshold: f64,

    /// Momentum tightening (bps to reduce spread when momentum detected)
    pub momentum_tightening_bps: f64,

    /// Inventory urgency threshold (start using taker orders)
    pub inventory_urgency_threshold: f64,

    /// Volatility to spread conversion factor (multiplier for volatility_bps -> spread_bps)
    /// Typical range: 0.005 - 0.02
    /// Example: 0.01 means 1000 bps volatility -> 10 bps half-spread
    pub volatility_to_spread_factor: f64,
}

impl Default for MultiLevelConfig {
    fn default() -> Self {
        Self {
            max_levels: 3,
            min_profitable_spread_bps: 4.0,  // 3 bps fees + 1 bps edge
            level_spacing_bps: 2.0,
            total_size_per_side: 1.0,
            inventory_risk_limit: 0.7,
            directional_aggression: 2.0,
            momentum_threshold: 1.3,
            momentum_tightening_bps: 1.0,
            inventory_urgency_threshold: 0.8,
            volatility_to_spread_factor: 0.01,  // Conservative default: 1% of volatility becomes spread
        }
    }
}

/// State information needed for optimization
pub struct OptimizationState<'a> {
    pub mid_price: f64,
    pub inventory: f64,
    pub max_position: f64,
    pub adverse_selection_bps: f64,
    pub lob_imbalance: f64,
    pub volatility_bps: f64,
    pub current_time: f64,
    pub hawkes_model: &'a HawkesFillModel,
}

/// Multi-level optimizer
#[derive(Debug)]
pub struct MultiLevelOptimizer {
    config: MultiLevelConfig,
}

impl MultiLevelOptimizer {
    pub fn new(config: MultiLevelConfig) -> Self {
        Self { config }
    }

    /// Get a reference to the config
    pub fn config(&self) -> &MultiLevelConfig {
        &self.config
    }

    /// Optimize multi-level quotes
    pub fn optimize(
        &self,
        state: &OptimizationState,
        base_half_spread_bps: f64,
        tuning_params: &ConstrainedTuningParams,
    ) -> MultiLevelControl {
        // 1. Calculate directional bias
        let inventory_ratio = state.inventory / state.max_position;
        let (bid_aggression, ask_aggression) = self.calculate_aggression(
            state.adverse_selection_bps,
            inventory_ratio,
            tuning_params,
        );

        // 2. Determine number of levels (could be dynamic in future)
        let num_levels = self.config.max_levels;

        // 3. Calculate sizes for each level
        let (bid_sizes, ask_sizes) = self.allocate_sizes(state, inventory_ratio, num_levels);

        // 4. Build bid levels with aggression and momentum
        let mut bid_levels = Vec::new();
        for level in 0..num_levels {
            let level_offset = level as f64 * self.config.level_spacing_bps;
            let base_offset = base_half_spread_bps.max(self.config.min_profitable_spread_bps / 2.0);

            // Apply directional aggression
            let mut adjusted_offset = base_offset + level_offset - bid_aggression;

            // Check for Hawkes momentum and tighten further
            if state.hawkes_model.has_momentum(level, true, state.current_time, self.config.momentum_threshold) {
                adjusted_offset -= self.config.momentum_tightening_bps;
            }

            // Enforce minimum using tunable min_spread_base_ratio
            let min_offset = base_half_spread_bps * tuning_params.min_spread_base_ratio;
            adjusted_offset = adjusted_offset.max(min_offset).max(self.config.min_profitable_spread_bps / 2.0);

            bid_levels.push((adjusted_offset, bid_sizes[level]));
        }

        // 5. Build ask levels
        let mut ask_levels = Vec::new();
        for level in 0..num_levels {
            let level_offset = level as f64 * self.config.level_spacing_bps;
            let base_offset = base_half_spread_bps.max(self.config.min_profitable_spread_bps / 2.0);

            let mut adjusted_offset = base_offset + level_offset - ask_aggression;

            if state.hawkes_model.has_momentum(level, false, state.current_time, self.config.momentum_threshold) {
                adjusted_offset -= self.config.momentum_tightening_bps;
            }

            // Enforce minimum using tunable min_spread_base_ratio
            let min_offset = base_half_spread_bps * tuning_params.min_spread_base_ratio;
            adjusted_offset = adjusted_offset.max(min_offset).max(self.config.min_profitable_spread_bps / 2.0);

            ask_levels.push((adjusted_offset, ask_sizes[level]));
        }

        // 6. Check for taker urgency
        let (taker_buy_rate, taker_sell_rate, liquidate) = self.check_taker_urgency(
            state.inventory,       // Pass actual inventory to fix double-counting bug
            inventory_ratio,
            state.max_position,
            tuning_params,
        );

        MultiLevelControl {
            bid_levels,
            ask_levels,
            taker_buy_rate,
            taker_sell_rate,
            liquidate,
        }
    }

    /// Calculate aggression factors for each side
    fn calculate_aggression(
        &self,
        adverse_selection_bps: f64,
        inventory_ratio: f64,
        tuning_params: &ConstrainedTuningParams,
    ) -> (f64, f64) {
        // Signal-driven aggression using tunable parameters
        let signal_bid_aggression = adverse_selection_bps.max(0.0)
            * tuning_params.adverse_selection_adjustment_factor
            * tuning_params.adverse_selection_spread_scale;
        let signal_ask_aggression = (-adverse_selection_bps).max(0.0)
            * tuning_params.adverse_selection_adjustment_factor
            * tuning_params.adverse_selection_spread_scale;

        // Inventory-driven aggression (need to reduce position)
        // Use tunable skew_adjustment_factor to control inventory response
        let inv_bid_aggression = if inventory_ratio < -0.3 {
            // Short, want to buy back
            (inventory_ratio.abs() - 0.3) * 5.0 * tuning_params.skew_adjustment_factor
        } else {
            0.0
        };

        let inv_ask_aggression = if inventory_ratio > 0.3 {
            // Long, want to sell
            (inventory_ratio - 0.3) * 5.0 * tuning_params.skew_adjustment_factor
        } else {
            0.0
        };

        // Combine: signal when inventory low, inventory when high
        let weight = 1.0 - inventory_ratio.abs();
        let bid_aggression = signal_bid_aggression * weight + inv_bid_aggression;
        let ask_aggression = signal_ask_aggression * weight + inv_ask_aggression;

        (bid_aggression, ask_aggression)
    }

    /// Allocate sizes across levels
    fn allocate_sizes(
        &self,
        state: &OptimizationState,
        _inventory_ratio: f64,  // Reserved for future use
        num_levels: usize,
    ) -> (Vec<f64>, Vec<f64>) {
        // Base allocation (more size on inner levels)
        let base_allocations = vec![0.45, 0.30, 0.15, 0.07, 0.03];
        let mut bid_alloc = base_allocations[..num_levels.min(5)].to_vec();
        let mut ask_alloc = bid_alloc.clone();

        // Normalize
        let sum: f64 = bid_alloc.iter().sum();
        for x in &mut bid_alloc {
            *x /= sum;
        }
        for x in &mut ask_alloc {
            *x /= sum;
        }

        // Calculate available inventory with proper position limit enforcement
        // SAFETY FIX: Ensure we never exceed max_position even with inventory_risk_limit
        let inventory_capacity = state.max_position * self.config.inventory_risk_limit;

        // Max buy: How much we can buy before hitting our long limit
        // Considers both the risk-adjusted capacity AND the absolute max position
        let max_buy_risk_adjusted = (inventory_capacity - state.inventory).max(0.0);
        let max_buy_absolute = (state.max_position - state.inventory).max(0.0);
        let max_buy = max_buy_risk_adjusted.min(max_buy_absolute);

        // Max sell: How much we can sell before hitting our short limit
        // Considers both the risk-adjusted capacity AND the absolute max position
        let max_sell_risk_adjusted = (inventory_capacity + state.inventory).max(0.0);
        let max_sell_absolute = (state.max_position + state.inventory).max(0.0);
        let max_sell = max_sell_risk_adjusted.min(max_sell_absolute);

        let bid_budget = self.config.total_size_per_side.min(max_buy);
        let ask_budget = self.config.total_size_per_side.min(max_sell);

        // Log position limit constraints for debugging
        if bid_budget < self.config.total_size_per_side * 0.5 {
            log::info!(
                "📊 Bid budget constrained by position limits: {:.2} (wanted {:.2}, max_buy={:.2}, inv={:.2}, max_pos={:.2})",
                bid_budget, self.config.total_size_per_side, max_buy, state.inventory, state.max_position
            );
        }
        if ask_budget < self.config.total_size_per_side * 0.5 {
            log::info!(
                "📊 Ask budget constrained by position limits: {:.2} (wanted {:.2}, max_sell={:.2}, inv={:.2}, max_pos={:.2})",
                ask_budget, self.config.total_size_per_side, max_sell, state.inventory, state.max_position
            );
        }

        // Apply Hawkes excitement multiplier
        let mut bid_sizes = Vec::new();
        let mut ask_sizes = Vec::new();

        for level in 0..num_levels {
            let bid_excitement = state.hawkes_model.excitation_multiplier(level, true, state.current_time);
            let ask_excitement = state.hawkes_model.excitation_multiplier(level, false, state.current_time);

            // More excitement = allocate more size
            // SAFETY: Cap excitement between 0.1 and 2.0 to prevent unbounded amplification
            // This prevents a bug where extreme excitement could suggest massive orders
            let bid_excitement_capped = bid_excitement.min(2.0).max(0.1);
            let ask_excitement_capped = ask_excitement.min(2.0).max(0.1);

            // Log when excitement is being capped (for debugging)
            if bid_excitement > 2.0 {
                log::debug!(
                    "L{} bid excitement capped: {:.2} -> 2.0",
                    level + 1, bid_excitement
                );
            }
            if ask_excitement > 2.0 {
                log::debug!(
                    "L{} ask excitement capped: {:.2} -> 2.0",
                    level + 1, ask_excitement
                );
            }

            bid_sizes.push(bid_budget * bid_alloc[level] * bid_excitement_capped);
            ask_sizes.push(ask_budget * ask_alloc[level] * ask_excitement_capped);
        }

        (bid_sizes, ask_sizes)
    }

    /// Check if inventory is extreme enough to use taker orders
    fn check_taker_urgency(
        &self,
        inventory: f64,        // NEW: actual inventory value
        inventory_ratio: f64,
        max_position: f64,
        tuning_params: &ConstrainedTuningParams,
    ) -> (f64, f64, bool) {
        // Use tunable inventory_urgency_threshold
        if inventory_ratio > tuning_params.inventory_urgency_threshold {
            // Too long - need to sell back to threshold
            // FIX: Calculate excess inventory directly, no double-counting!
            let threshold_position = max_position * tuning_params.inventory_urgency_threshold;
            let excess_inventory = inventory - threshold_position;
            let taker_size = excess_inventory.max(0.0);

            // Optional: apply multiplier to sell faster than 1:1 (use with caution!)
            // let taker_size = (excess_inventory * tuning_params.liquidation_rate_multiplier).max(0.0);

            (0.0, taker_size, true)
        } else if inventory_ratio < -tuning_params.inventory_urgency_threshold {
            // Too short - need to buy to cover
            // FIX: Calculate deficit inventory directly, no double-counting!
            let threshold_position = -max_position * tuning_params.inventory_urgency_threshold;
            let deficit_inventory = threshold_position - inventory;
            let taker_size = deficit_inventory.max(0.0);

            // Optional: apply multiplier to buy faster than 1:1 (use with caution!)
            // let taker_size = (deficit_inventory * tuning_params.liquidation_rate_multiplier).max(0.0);

            (taker_size, 0.0, true)
        } else {
            (0.0, 0.0, false)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::tuning::TuningParams;
    use super::super::hawkes::HawkesFillModel;

    #[test]
    fn test_multi_level_control_symmetric() {
        let control = MultiLevelControl::symmetric(3, 6.0, 2.0, 0.5);

        assert_eq!(control.bid_levels.len(), 3);
        assert_eq!(control.ask_levels.len(), 3);

        // Check spacing
        assert_eq!(control.bid_levels[0].0, 6.0);
        assert_eq!(control.bid_levels[1].0, 8.0);
        assert_eq!(control.bid_levels[2].0, 10.0);

        assert!(control.validate().is_ok());
    }

    #[test]
    fn test_multi_level_optimizer() {
        let config = MultiLevelConfig::default();
        let optimizer = MultiLevelOptimizer::new(config);
        let hawkes = HawkesFillModel::new(3);

        // Use default tuning params
        let tuning_params = TuningParams::default().get_constrained();

        let state = OptimizationState {
            mid_price: 100.0,
            inventory: 0.0,
            max_position: 100.0,
            adverse_selection_bps: 0.0, // Neutral (to avoid extreme aggression in test)
            lob_imbalance: 0.0,
            volatility_bps: 100.0,
            current_time: 0.0,
            hawkes_model: &hawkes,
        };

        let control = optimizer.optimize(&state, 6.0, &tuning_params);

        assert!(control.validate().is_ok());
        assert_eq!(control.bid_levels.len(), 3);
        assert_eq!(control.ask_levels.len(), 3);

        // Levels should have increasing spreads
        assert!(control.bid_levels[0].0 < control.bid_levels[1].0);
        assert!(control.bid_levels[1].0 < control.bid_levels[2].0);
        assert!(control.ask_levels[0].0 < control.ask_levels[1].0);
        assert!(control.ask_levels[1].0 < control.ask_levels[2].0);
    }
}
