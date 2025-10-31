//! Position Manager - Centralized Position State Management
//!
//! This component provides a single source of truth for position state and allowed actions,
//! preventing conflicts between optimizer, strategy, and state manager.

use serde::{Deserialize, Serialize};
use crate::RestingOrder;

/// Position state based on inventory ratio relative to max position
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PositionState {
    /// < 70% of max position - normal trading allowed
    Normal,
    /// 70-85% of max position - reduce trading aggressiveness
    Warning,
    /// 85-100% of max position - start reducing position
    Critical,
    /// > 100% of max position - emergency liquidation required
    OverLimit,
}

/// Allowed actions based on current position state
#[derive(Debug, Clone)]
pub enum AllowedAction {
    /// Normal trading with full position capacity
    FullTrading {
        max_buy: f64,
        max_sell: f64,
    },
    /// Reduced trading with conservative sizes
    ReducedTrading {
        max_buy: f64,
        max_sell: f64,
    },
    /// Only reduce-only orders allowed
    ReduceOnly {
        reduce_size: f64,
    },
    /// Emergency liquidation of entire position
    EmergencyLiquidation {
        full_size: f64,
    },
}

/// Configuration for position manager thresholds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PositionManagerConfig {
    /// Warning threshold (fraction of max position, e.g., 0.7)
    pub warning_threshold: f64,

    /// Critical threshold (fraction of max position, e.g., 0.85)
    pub critical_threshold: f64,

    /// Reduction factor for Warning state (e.g., 0.5 = 50% of normal size)
    pub warning_reduction_factor: f64,

    /// Percentage of position to reduce in Critical state (e.g., 0.3 = 30%)
    pub critical_reduction_percentage: f64,
}

impl Default for PositionManagerConfig {
    fn default() -> Self {
        Self {
            warning_threshold: 0.7,
            critical_threshold: 0.85,
            warning_reduction_factor: 0.5,
            critical_reduction_percentage: 0.3,
        }
    }
}

/// Helper struct to track pending orders
#[derive(Debug, Clone, Default)]
pub struct PendingOrders {
    pub total_buy_size: f64,
    pub total_sell_size: f64,
}

impl PendingOrders {
    /// Create from bid and ask order lists
    pub fn from_orders(bids: &[RestingOrder], asks: &[RestingOrder]) -> Self {
        let total_buy_size: f64 = bids.iter().map(|o| o.size).sum();
        let total_sell_size: f64 = asks.iter().map(|o| o.size).sum();

        Self {
            total_buy_size,
            total_sell_size,
        }
    }

    /// Get net pending exposure (positive = long, negative = short)
    pub fn net_exposure(&self) -> f64 {
        self.total_buy_size - self.total_sell_size
    }
}

/// Position Manager - Central authority for position state and constraints
pub struct PositionManager {
    max_position: f64,
    config: PositionManagerConfig,
}

impl PositionManager {
    /// Create new position manager
    pub fn new(max_position: f64, config: PositionManagerConfig) -> Self {
        Self {
            max_position,
            config,
        }
    }

    /// Create with default config
    pub fn with_max_position(max_position: f64) -> Self {
        Self::new(max_position, PositionManagerConfig::default())
    }

    /// Get current position state based on inventory ratio
    pub fn get_state(&self, current_position: f64) -> PositionState {
        let ratio = current_position.abs() / self.max_position;

        if ratio > 1.0 {
            PositionState::OverLimit
        } else if ratio > self.config.critical_threshold {
            PositionState::Critical
        } else if ratio > self.config.warning_threshold {
            PositionState::Warning
        } else {
            PositionState::Normal
        }
    }

    /// Get allowed action based on current state
    pub fn get_allowed_action(
        &self,
        state: PositionState,
        current_pos: f64,
        pending_orders: &PendingOrders,
    ) -> AllowedAction {
        match state {
            PositionState::Normal => {
                AllowedAction::FullTrading {
                    max_buy: self.calculate_max_buy(current_pos, pending_orders),
                    max_sell: self.calculate_max_sell(current_pos, pending_orders),
                }
            }

            PositionState::Warning => {
                let max_buy = self.calculate_max_buy(current_pos, pending_orders)
                    * self.config.warning_reduction_factor;
                let max_sell = self.calculate_max_sell(current_pos, pending_orders)
                    * self.config.warning_reduction_factor;

                AllowedAction::ReducedTrading {
                    max_buy,
                    max_sell,
                }
            }

            PositionState::Critical => {
                // Reduce by configured percentage of position
                let reduce_size = current_pos.abs() * self.config.critical_reduction_percentage;

                AllowedAction::ReduceOnly {
                    reduce_size,
                }
            }

            PositionState::OverLimit => {
                // Emergency - liquidate entire position
                AllowedAction::EmergencyLiquidation {
                    full_size: current_pos.abs(),
                }
            }
        }
    }

    /// Calculate maximum buy size allowed (considering pending orders)
    fn calculate_max_buy(&self, current_pos: f64, pending: &PendingOrders) -> f64 {
        // Effective long position = current + pending buys
        let effective_long = current_pos + pending.total_buy_size;

        // Available capacity = max - effective_long
        (self.max_position - effective_long).max(0.0)
    }

    /// Calculate maximum sell size allowed (considering pending orders)
    fn calculate_max_sell(&self, current_pos: f64, pending: &PendingOrders) -> f64 {
        // Effective short position = current - pending sells
        let effective_short = current_pos - pending.total_sell_size;

        // Available capacity = max + effective_short (since effective_short is negative when short)
        (self.max_position + effective_short).max(0.0)
    }

    /// Get max position size
    pub fn max_position(&self) -> f64 {
        self.max_position
    }

    /// Update max position (e.g., when margin changes)
    pub fn update_max_position(&mut self, new_max: f64) {
        self.max_position = new_max;
    }

    /// Get config
    pub fn config(&self) -> &PositionManagerConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_position_state_transitions() {
        let manager = PositionManager::with_max_position(10.0);

        assert_eq!(manager.get_state(0.0), PositionState::Normal);
        assert_eq!(manager.get_state(5.0), PositionState::Normal); // 50%
        assert_eq!(manager.get_state(7.5), PositionState::Warning); // 75%
        assert_eq!(manager.get_state(9.0), PositionState::Critical); // 90%
        assert_eq!(manager.get_state(11.0), PositionState::OverLimit); // 110%
    }

    #[test]
    fn test_max_buy_calculation() {
        let manager = PositionManager::with_max_position(10.0);
        let pending = PendingOrders {
            total_buy_size: 2.0,
            total_sell_size: 0.0,
        };

        // Current position: 5.0, Pending buys: 2.0, Effective: 7.0
        // Max buy = 10.0 - 7.0 = 3.0
        let max_buy = manager.calculate_max_buy(5.0, &pending);
        assert_eq!(max_buy, 3.0);
    }

    #[test]
    fn test_max_sell_calculation() {
        let manager = PositionManager::with_max_position(10.0);
        let pending = PendingOrders {
            total_buy_size: 0.0,
            total_sell_size: 2.0,
        };

        // Current position: -5.0, Pending sells: 2.0, Effective: -7.0
        // Max sell = 10.0 - 7.0 = 3.0
        let max_sell = manager.calculate_max_sell(-5.0, &pending);
        assert_eq!(max_sell, 3.0);
    }

    #[test]
    fn test_allowed_action_normal() {
        let manager = PositionManager::with_max_position(10.0);
        let pending = PendingOrders::default();

        let action = manager.get_allowed_action(
            PositionState::Normal,
            0.0,
            &pending,
        );

        match action {
            AllowedAction::FullTrading { max_buy, max_sell } => {
                assert_eq!(max_buy, 10.0);
                assert_eq!(max_sell, 10.0);
            }
            _ => panic!("Expected FullTrading"),
        }
    }

    #[test]
    fn test_allowed_action_overlimit() {
        let manager = PositionManager::with_max_position(10.0);
        let pending = PendingOrders::default();

        let action = manager.get_allowed_action(
            PositionState::OverLimit,
            12.0,  // Over limit
            &pending,
        );

        match action {
            AllowedAction::EmergencyLiquidation { full_size } => {
                assert_eq!(full_size, 12.0);
            }
            _ => panic!("Expected EmergencyLiquidation"),
        }
    }
}
