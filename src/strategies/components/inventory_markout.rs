//! Inventory-Based Fair Value Adjustments
//!
//! Implements inventory markouts that shift the fair value to discourage
//! position accumulation. Based on inventory aversion and market conditions.

use serde::{Deserialize, Serialize};

/// Configuration for inventory markouts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InventoryMarkoutConfig {
    /// Linear penalty per unit of inventory (in ticks)
    pub linear_penalty_per_unit: f64,

    /// Quadratic penalty coefficient (amplifies at extremes)
    pub quadratic_coefficient: f64,

    /// Cubic penalty coefficient (for aggressive near-limit behavior)
    pub cubic_coefficient: f64,

    /// Volatility scaling factor (higher vol = more aggressive markouts)
    pub volatility_scaling: f64,

    /// Enable inventory urgency (aggressive markouts near limits)
    pub enable_urgency: bool,

    /// Urgency threshold (fraction of max position)
    pub urgency_threshold: f64,

    /// Urgency multiplier (how much to amplify penalty in urgent zone)
    pub urgency_multiplier: f64,
}

impl Default for InventoryMarkoutConfig {
    fn default() -> Self {
        Self {
            linear_penalty_per_unit: 0.005,  // 0.5 bps per unit
            quadratic_coefficient: 0.0001,   // Gentle curve
            cubic_coefficient: 0.000001,     // Aggressive near limits
            volatility_scaling: 0.5,
            enable_urgency: true,
            urgency_threshold: 0.7,          // Start urgency at 70% of max
            urgency_multiplier: 3.0,         // 3x penalty in urgent zone
        }
    }
}

/// Calculator for inventory-based fair value adjustments
pub struct InventoryMarkoutCalculator {
    config: InventoryMarkoutConfig,
    tick_size: f64,
}

impl InventoryMarkoutCalculator {
    pub fn new(config: InventoryMarkoutConfig, tick_size: f64) -> Self {
        Self { config, tick_size }
    }

    /// Calculate fair value adjustment based on inventory
    ///
    /// Returns a price adjustment in absolute terms (not ticks).
    /// Positive = shift up (discourage buys), Negative = shift down (discourage sells)
    pub fn calculate_adjustment(
        &self,
        position: f64,
        max_position: f64,
        volatility_bps: f64,
        _mid_price: f64,
    ) -> InventoryAdjustment {
        if position.abs() < 1e-6 {
            return InventoryAdjustment {
                price_adjustment: 0.0,
                ticks_adjustment: 0.0,
                inventory_ratio: 0.0,
                urgency_zone: false,
                penalty_breakdown: PenaltyBreakdown::default(),
            };
        }

        let inventory_ratio = position / max_position;
        let abs_ratio = inventory_ratio.abs();

        // Base penalties (polynomial curve)
        let linear_penalty = self.config.linear_penalty_per_unit * position.abs();
        let quadratic_penalty = self.config.quadratic_coefficient * position.powi(2);
        let cubic_penalty = self.config.cubic_coefficient * position.powi(3).abs();

        // Volatility scaling (higher vol = more aggressive)
        let vol_factor = 1.0 + (volatility_bps / 100.0) * self.config.volatility_scaling;

        // Urgency multiplier (near limits)
        let urgency_zone = self.config.enable_urgency && abs_ratio > self.config.urgency_threshold;
        let urgency_factor = if urgency_zone {
            let excess = (abs_ratio - self.config.urgency_threshold)
                / (1.0 - self.config.urgency_threshold);
            1.0 + excess * (self.config.urgency_multiplier - 1.0)
        } else {
            1.0
        };

        // Total penalty in ticks
        let ticks_penalty = (linear_penalty + quadratic_penalty + cubic_penalty)
            * vol_factor
            * urgency_factor;

        // Apply sign (positive position = discourage buys = shift fair value UP)
        let signed_ticks = ticks_penalty * position.signum();

        // Convert to price
        let price_adjustment = signed_ticks * self.tick_size;

        InventoryAdjustment {
            price_adjustment,
            ticks_adjustment: signed_ticks,
            inventory_ratio,
            urgency_zone,
            penalty_breakdown: PenaltyBreakdown {
                linear: linear_penalty,
                quadratic: quadratic_penalty,
                cubic: cubic_penalty,
                vol_factor,
                urgency_factor,
                total_ticks: ticks_penalty,
            },
        }
    }

    /// Calculate adjusted fair value (convenience method)
    pub fn adjust_fair_value(
        &self,
        mid_price: f64,
        position: f64,
        max_position: f64,
        volatility_bps: f64,
    ) -> f64 {
        let adjustment = self.calculate_adjustment(
            position,
            max_position,
            volatility_bps,
            mid_price,
        );

        mid_price + adjustment.price_adjustment
    }
}

/// Result of inventory adjustment calculation
#[derive(Debug, Clone)]
pub struct InventoryAdjustment {
    /// Price adjustment (absolute)
    pub price_adjustment: f64,

    /// Adjustment in ticks
    pub ticks_adjustment: f64,

    /// Current inventory ratio (position / max_position)
    pub inventory_ratio: f64,

    /// Whether in urgency zone
    pub urgency_zone: bool,

    /// Breakdown of penalty components
    pub penalty_breakdown: PenaltyBreakdown,
}

#[derive(Debug, Clone, Default)]
pub struct PenaltyBreakdown {
    pub linear: f64,
    pub quadratic: f64,
    pub cubic: f64,
    pub vol_factor: f64,
    pub urgency_factor: f64,
    pub total_ticks: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero_inventory() {
        let config = InventoryMarkoutConfig::default();
        let calc = InventoryMarkoutCalculator::new(config, 0.01);

        let adj = calc.calculate_adjustment(0.0, 10.0, 20.0, 100.0);
        assert_eq!(adj.price_adjustment, 0.0);
    }

    #[test]
    fn test_long_position_shifts_up() {
        let config = InventoryMarkoutConfig::default();
        let calc = InventoryMarkoutCalculator::new(config, 0.01);

        // Long position should shift fair value UP (discourage more buys)
        let adj = calc.calculate_adjustment(5.0, 10.0, 20.0, 100.0);
        assert!(adj.price_adjustment > 0.0);
        assert_eq!(adj.inventory_ratio, 0.5);
    }

    #[test]
    fn test_short_position_shifts_down() {
        let config = InventoryMarkoutConfig::default();
        let calc = InventoryMarkoutCalculator::new(config, 0.01);

        // Short position should shift fair value DOWN (discourage more sells)
        let adj = calc.calculate_adjustment(-5.0, 10.0, 20.0, 100.0);
        assert!(adj.price_adjustment < 0.0);
        assert_eq!(adj.inventory_ratio, -0.5);
    }

    #[test]
    fn test_urgency_zone() {
        let config = InventoryMarkoutConfig::default();
        let calc = InventoryMarkoutCalculator::new(config, 0.01);

        // At 80% of max (above urgency threshold of 70%)
        let adj_urgent = calc.calculate_adjustment(8.0, 10.0, 20.0, 100.0);

        // At 60% of max (below urgency threshold)
        let adj_normal = calc.calculate_adjustment(6.0, 10.0, 20.0, 100.0);

        assert!(adj_urgent.urgency_zone);
        assert!(!adj_normal.urgency_zone);

        // Urgency zone should have larger penalty per unit
        let penalty_per_unit_urgent = adj_urgent.ticks_adjustment / 8.0;
        let penalty_per_unit_normal = adj_normal.ticks_adjustment / 6.0;
        assert!(penalty_per_unit_urgent > penalty_per_unit_normal);
    }

    #[test]
    fn test_volatility_scaling() {
        let config = InventoryMarkoutConfig::default();
        let calc = InventoryMarkoutCalculator::new(config, 0.01);

        let adj_low_vol = calc.calculate_adjustment(5.0, 10.0, 10.0, 100.0);  // 10 bps vol
        let adj_high_vol = calc.calculate_adjustment(5.0, 10.0, 50.0, 100.0); // 50 bps vol

        // Higher vol should create larger adjustments
        assert!(adj_high_vol.price_adjustment.abs() > adj_low_vol.price_adjustment.abs());
    }
}
