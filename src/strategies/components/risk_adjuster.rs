// ============================================================================
// Risk Adjuster - Apply Risk-Based Signal Adjustments
// ============================================================================
//
// Takes pure signals from the signal generator and applies risk-based
// adjustments based on position state, margin constraints, and market
// conditions. This is the layer that enforces risk limits.

use log::{info, warn};

use super::{
    PositionManager, AllowedAction, PositionState, PendingOrders,
    signal_generator::QuoteSignal,
    trading_state_store::TradingSnapshot,
};

// ============================================================================
// Adjusted Signal
// ============================================================================

/// Signal after risk adjustments
#[derive(Debug, Clone)]
pub struct AdjustedSignal {
    /// Adjusted bid levels (may be filtered or size-adjusted)
    pub bid_levels: Vec<AdjustedQuoteLevel>,

    /// Adjusted ask levels
    pub ask_levels: Vec<AdjustedQuoteLevel>,

    /// Recommended taker buy rate (may be overridden for risk management)
    pub taker_buy_rate: f64,

    /// Recommended taker sell rate
    pub taker_sell_rate: f64,

    /// Risk adjustment reason (for logging/debugging)
    pub adjustment_reason: String,

    /// Whether signal was modified
    pub was_modified: bool,

    /// Timestamp
    pub timestamp: f64,
}

/// A quote level after risk adjustments
#[derive(Debug, Clone)]
pub struct AdjustedQuoteLevel {
    /// Absolute price (not offset)
    pub price: f64,

    /// Adjusted size
    pub size: f64,

    /// Priority (0-1, higher = more important)
    pub priority: f64,

    /// Whether this is a bid
    pub is_bid: bool,

    /// Original size before adjustment
    pub original_size: f64,

    /// Size adjustment reason
    pub size_adjustment_reason: Option<String>,
}

// ============================================================================
// Margin Calculator (moved from hjb_strategy.rs)
// ============================================================================

/// Calculates margin requirements and available capacity
#[derive(Debug, Clone)]
pub struct MarginCalculator {
    /// Account leverage (1-max_leverage)
    leverage: usize,

    /// Margin safety buffer (0.0-1.0)
    safety_buffer: f64,
}

impl MarginCalculator {
    pub fn new(leverage: usize, safety_buffer: f64) -> Self {
        Self {
            leverage,
            safety_buffer: safety_buffer.clamp(0.0, 0.99),
        }
    }

    /// Calculate initial margin required for a position
    pub fn initial_margin_required(&self, position_size: f64, mark_price: f64) -> f64 {
        (position_size.abs() * mark_price) / self.leverage as f64
    }

    /// Calculate available margin for new positions
    pub fn available_margin(&self, account_equity: f64, margin_used: f64) -> f64 {
        let usable_equity = account_equity * (1.0 - self.safety_buffer);
        (usable_equity - margin_used).max(0.0)
    }

    /// Calculate maximum additional position size that can be opened
    pub fn max_additional_position_size(
        &self,
        account_equity: f64,
        margin_used: f64,
        mark_price: f64,
    ) -> f64 {
        let available = self.available_margin(account_equity, margin_used);
        (available * self.leverage as f64) / mark_price
    }

    /// Adjust order size to fit within margin constraints
    pub fn adjust_order_size_for_margin(
        &self,
        desired_size: f64,
        current_position: f64,
        account_equity: f64,
        margin_used: f64,
        mark_price: f64,
        is_buy: bool,
    ) -> (f64, Option<String>) {
        // Calculate the position delta if this order fills
        let position_delta = if is_buy { desired_size } else { -desired_size };
        let new_position = current_position + position_delta;

        // If order reduces position (opposing direction), no margin check needed
        if new_position.abs() < current_position.abs() {
            return (desired_size, None);
        }

        // Calculate how much position increase is allowed
        let position_increase = new_position.abs() - current_position.abs();
        let max_increase = self.max_additional_position_size(account_equity, margin_used, mark_price);

        if position_increase <= max_increase {
            // Full size fits within margin
            (desired_size, None)
        } else {
            // Reduce size to fit margin constraints
            let adjusted_increase = max_increase.max(0.0);
            let adjusted_size = if current_position.signum() == position_delta.signum() {
                // Same direction: can only add adjusted_increase
                adjusted_increase
            } else {
                // Crossing zero: can close current + open adjusted_increase on other side
                current_position.abs() + adjusted_increase
            };

            let reason = format!(
                "Margin limit: reduced {:.4} -> {:.4} (available: {:.2})",
                desired_size, adjusted_size, max_increase
            );

            (adjusted_size, Some(reason))
        }
    }
}

// ============================================================================
// Risk Adjuster
// ============================================================================

/// Applies risk-based adjustments to signals
pub struct RiskAdjuster {
    /// Position manager for state-based constraints
    position_manager: PositionManager,

    /// Margin calculator
    margin_calculator: MarginCalculator,

    /// Minimum order size (exchange limit)
    min_order_size: f64,
}

impl RiskAdjuster {
    /// Create a new risk adjuster
    pub fn new(
        position_manager: PositionManager,
        margin_calculator: MarginCalculator,
        min_order_size: f64,
    ) -> Self {
        Self {
            position_manager,
            margin_calculator,
            min_order_size,
        }
    }

    /// Adjust signal based on current trading state
    pub fn adjust_signal(
        &self,
        signal: QuoteSignal,
        snapshot: &TradingSnapshot,
    ) -> AdjustedSignal {
        let position = snapshot.risk_metrics.position;
        let pos_state = self.position_manager.get_state(position);

        match pos_state {
            PositionState::Normal => {
                self.apply_normal_adjustments(signal, snapshot)
            }
            PositionState::Warning => {
                info!("[RISK ADJUSTER] Position in Warning state, applying conservative adjustments");
                self.apply_warning_adjustments(signal, snapshot)
            }
            PositionState::Critical => {
                warn!("[RISK ADJUSTER] Position in Critical state, forcing reduction");
                self.create_reduction_signal(snapshot)
            }
            PositionState::OverLimit => {
                warn!("[RISK ADJUSTER] Position over limit, emergency liquidation");
                self.create_liquidation_signal(snapshot)
            }
        }
    }

    /// Apply normal adjustments (margin checks, size limits)
    fn apply_normal_adjustments(
        &self,
        signal: QuoteSignal,
        snapshot: &TradingSnapshot,
    ) -> AdjustedSignal {
        let mut bid_levels = Vec::new();
        let mut ask_levels = Vec::new();
        let mut was_modified = false;

        // Get position state and allowed action
        let pos_state = self.position_manager.get_state(snapshot.risk_metrics.position);
        let pending_orders = PendingOrders {
            total_buy_size: snapshot.total_buy_size,
            total_sell_size: snapshot.total_sell_size,
        };

        let allowed_action = self.position_manager.get_allowed_action(
            pos_state,
            snapshot.risk_metrics.position,
            &pending_orders,
        );

        match allowed_action {
            AllowedAction::FullTrading { max_buy, max_sell } => {
                // Adjust bid levels
                for level in signal.bid_levels {
                    let (adjusted_size, reason) = self.adjust_size_for_constraints(
                        level.size.min(max_buy),
                        snapshot.risk_metrics.position,
                        &snapshot.risk_metrics,
                        true,
                    );

                    if adjusted_size >= self.min_order_size {
                        bid_levels.push(AdjustedQuoteLevel {
                            price: level.price,
                            size: adjusted_size,
                            priority: level.urgency,
                            is_bid: true,
                            original_size: level.size,
                            size_adjustment_reason: reason.clone(),
                        });

                        if reason.is_some() {
                            was_modified = true;
                        }
                    }
                }

                // Adjust ask levels
                for level in signal.ask_levels {
                    let (adjusted_size, reason) = self.adjust_size_for_constraints(
                        level.size.min(max_sell),
                        snapshot.risk_metrics.position,
                        &snapshot.risk_metrics,
                        false,
                    );

                    if adjusted_size >= self.min_order_size {
                        ask_levels.push(AdjustedQuoteLevel {
                            price: level.price,
                            size: adjusted_size,
                            priority: level.urgency,
                            is_bid: false,
                            original_size: level.size,
                            size_adjustment_reason: reason.clone(),
                        });

                        if reason.is_some() {
                            was_modified = true;
                        }
                    }
                }
            }

            AllowedAction::ReducedTrading { max_buy, max_sell } => {
                // Similar to FullTrading but with reduced sizes
                for level in signal.bid_levels {
                    let reduced_size = level.size.min(max_buy);
                    if reduced_size >= self.min_order_size {
                        bid_levels.push(AdjustedQuoteLevel {
                            price: level.price,
                            size: reduced_size,
                            priority: level.urgency,
                            is_bid: true,
                            original_size: level.size,
                            size_adjustment_reason: Some("Reduced trading mode".to_string()),
                        });
                        was_modified = true;
                    }
                }

                for level in signal.ask_levels {
                    let reduced_size = level.size.min(max_sell);
                    if reduced_size >= self.min_order_size {
                        ask_levels.push(AdjustedQuoteLevel {
                            price: level.price,
                            size: reduced_size,
                            priority: level.urgency,
                            is_bid: false,
                            original_size: level.size,
                            size_adjustment_reason: Some("Reduced trading mode".to_string()),
                        });
                        was_modified = true;
                    }
                }
            }

            AllowedAction::ReduceOnly { .. } | AllowedAction::EmergencyLiquidation { .. } => {
                // These are handled by the position state logic below
                was_modified = true;
            }
        }

        let adjustment_reason = if was_modified {
            format!("Normal adjustments applied (state: {:?})", pos_state)
        } else {
            "No adjustments needed".to_string()
        };

        AdjustedSignal {
            bid_levels,
            ask_levels,
            taker_buy_rate: signal.taker_buy_rate,
            taker_sell_rate: signal.taker_sell_rate,
            adjustment_reason,
            was_modified,
            timestamp: signal.timestamp,
        }
    }

    /// Apply warning adjustments (more conservative)
    fn apply_warning_adjustments(
        &self,
        signal: QuoteSignal,
        snapshot: &TradingSnapshot,
    ) -> AdjustedSignal {
        let position = snapshot.risk_metrics.position;
        let mut bid_levels = Vec::new();
        let mut ask_levels = Vec::new();

        // In warning state, reduce sizes and only quote on favorable side
        if position > 0.0 {
            // Long position - favor selling
            for level in signal.ask_levels {
                let reduced_size = level.size * 0.5; // More conservative
                if reduced_size >= self.min_order_size {
                    ask_levels.push(AdjustedQuoteLevel {
                        price: level.price,
                        size: reduced_size,
                        priority: level.urgency,
                        is_bid: false,
                        original_size: level.size,
                        size_adjustment_reason: Some("Warning state: reduced size".to_string()),
                    });
                }
            }
        } else {
            // Short position - favor buying
            for level in signal.bid_levels {
                let reduced_size = level.size * 0.5; // More conservative
                if reduced_size >= self.min_order_size {
                    bid_levels.push(AdjustedQuoteLevel {
                        price: level.price,
                        size: reduced_size,
                        priority: level.urgency,
                        is_bid: true,
                        original_size: level.size,
                        size_adjustment_reason: Some("Warning state: reduced size".to_string()),
                    });
                }
            }
        }

        AdjustedSignal {
            bid_levels,
            ask_levels,
            taker_buy_rate: signal.taker_buy_rate,
            taker_sell_rate: signal.taker_sell_rate,
            adjustment_reason: "Warning state: conservative quoting".to_string(),
            was_modified: true,
            timestamp: signal.timestamp,
        }
    }

    /// Create signal that forces position reduction (Critical state)
    fn create_reduction_signal(&self, snapshot: &TradingSnapshot) -> AdjustedSignal {
        let position = snapshot.risk_metrics.position;
        let reduction_size = position.abs() * 0.3; // Reduce 30% of position

        let mut bid_levels = Vec::new();
        let mut ask_levels = Vec::new();

        // Quote aggressively on the reducing side
        let mid_price = snapshot.market_data.mid_price;
        if mid_price > 0.0 {
            if position > 0.0 {
                // Need to sell - price inside spread
                let aggressive_price = mid_price * (1.0 - 2.0 / 10000.0);
                ask_levels.push(AdjustedQuoteLevel {
                    price: aggressive_price,
                    size: reduction_size,
                    priority: 1.0,
                    is_bid: false,
                    original_size: reduction_size,
                    size_adjustment_reason: Some("Critical: forcing reduction".to_string()),
                });
            } else {
                // Need to buy - price inside spread
                let aggressive_price = mid_price * (1.0 + 2.0 / 10000.0);
                bid_levels.push(AdjustedQuoteLevel {
                    price: aggressive_price,
                    size: reduction_size,
                    priority: 1.0,
                    is_bid: true,
                    original_size: reduction_size,
                    size_adjustment_reason: Some("Critical: forcing reduction".to_string()),
                });
            }
        }

        AdjustedSignal {
            bid_levels,
            ask_levels,
            taker_buy_rate: if position < 0.0 { 0.5 } else { 0.0 },
            taker_sell_rate: if position > 0.0 { 0.5 } else { 0.0 },
            adjustment_reason: "Critical state: forcing position reduction".to_string(),
            was_modified: true,
            timestamp: snapshot.timestamp,
        }
    }

    /// Create emergency liquidation signal (OverLimit state)
    fn create_liquidation_signal(&self, snapshot: &TradingSnapshot) -> AdjustedSignal {
        let position = snapshot.risk_metrics.position;

        // Emergency: use takers to close position immediately
        AdjustedSignal {
            bid_levels: Vec::new(),
            ask_levels: Vec::new(),
            taker_buy_rate: if position < 0.0 { 1.0 } else { 0.0 },
            taker_sell_rate: if position > 0.0 { 1.0 } else { 0.0 },
            adjustment_reason: "EMERGENCY LIQUIDATION".to_string(),
            was_modified: true,
            timestamp: snapshot.timestamp,
        }
    }

    /// Adjust size for margin and position constraints
    fn adjust_size_for_constraints(
        &self,
        desired_size: f64,
        current_position: f64,
        risk_metrics: &super::trading_state_store::RiskMetrics,
        is_buy: bool,
    ) -> (f64, Option<String>) {
        // First check margin
        let (size_after_margin, margin_reason) = self.margin_calculator.adjust_order_size_for_margin(
            desired_size,
            current_position,
            risk_metrics.account_equity,
            risk_metrics.margin_used,
            risk_metrics.max_position_size, // Using as mark price proxy
            is_buy,
        );

        // Then check minimum size
        if size_after_margin < self.min_order_size {
            return (0.0, Some("Below minimum order size".to_string()));
        }

        (size_after_margin, margin_reason)
    }

    /// Get position manager
    pub fn position_manager(&self) -> &PositionManager {
        &self.position_manager
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::PositionManagerConfig;
    use super::super::trading_state_store::{RiskMetrics, MarketData};
    use super::super::signal_generator::{SignalMetadata};
    use std::collections::BTreeMap;

    #[test]
    fn test_normal_adjustments() {
        let config = PositionManagerConfig {
            warning_threshold: 0.7,
            critical_threshold: 0.85,
            warning_reduction_factor: 0.5,
            critical_reduction_percentage: 0.3,
        };

        let position_manager = PositionManager::new(50.0, config);
        let margin_calculator = MarginCalculator::new(5, 0.2);
        let adjuster = RiskAdjuster::new(position_manager, margin_calculator, 0.01);

        let signal = QuoteSignal {
            bid_levels: vec![QuoteLevel {
                offset_bps: -5.0,
                size: 10.0,
                urgency: 0.8,
                is_bid: true,
            }],
            ask_levels: vec![QuoteLevel {
                offset_bps: 5.0,
                size: 10.0,
                urgency: 0.8,
                is_bid: false,
            }],
            urgency: 0.5,
            taker_buy_rate: 0.0,
            taker_sell_rate: 0.0,
            timestamp: 0.0,
            metadata: SignalMetadata {
                mid_price: 100.0,
                volatility_bps: 20.0,
                adverse_selection: 0.0,
                inventory: 0.0,
                optimizer_time_us: 100,
                was_cached: false,
            },
        };

        let snapshot = TradingSnapshot {
            market_data: MarketData::default(),
            risk_metrics: RiskMetrics {
                position: 0.0,
                account_equity: 10000.0,
                margin_used: 0.0,
                margin_available: 8000.0,
                ..Default::default()
            },
            open_orders: BTreeMap::new(),
            num_buy_orders: 0,
            num_sell_orders: 0,
            total_buy_size: 0.0,
            total_sell_size: 0.0,
            timestamp: 0.0,
        };

        let adjusted = adjuster.adjust_signal(signal, &snapshot);

        // Should have both bids and asks in normal state
        assert!(!adjusted.bid_levels.is_empty());
        assert!(!adjusted.ask_levels.is_empty());
    }

    #[test]
    fn test_warning_adjustments() {
        let config = PositionManagerConfig {
            warning_threshold: 0.7,
            critical_threshold: 0.85,
            warning_reduction_factor: 0.5,
            critical_reduction_percentage: 0.3,
        };

        let position_manager = PositionManager::new(50.0, config);
        let margin_calculator = MarginCalculator::new(5, 0.2);
        let adjuster = RiskAdjuster::new(position_manager, margin_calculator, 0.01);

        let signal = QuoteSignal {
            bid_levels: vec![QuoteLevel {
                offset_bps: -5.0,
                size: 10.0,
                urgency: 0.8,
                is_bid: true,
            }],
            ask_levels: vec![QuoteLevel {
                offset_bps: 5.0,
                size: 10.0,
                urgency: 0.8,
                is_bid: false,
            }],
            urgency: 0.5,
            taker_buy_rate: 0.0,
            taker_sell_rate: 0.0,
            timestamp: 0.0,
            metadata: SignalMetadata {
                mid_price: 100.0,
                volatility_bps: 20.0,
                adverse_selection: 0.0,
                inventory: 45.0, // In warning zone
                optimizer_time_us: 100,
                was_cached: false,
            },
        };

        let snapshot = TradingSnapshot {
            market_data: MarketData::default(),
            risk_metrics: RiskMetrics {
                position: 45.0, // Warning zone
                account_equity: 10000.0,
                margin_used: 1000.0,
                margin_available: 7000.0,
                ..Default::default()
            },
            open_orders: BTreeMap::new(),
            num_buy_orders: 0,
            num_sell_orders: 0,
            total_buy_size: 0.0,
            total_sell_size: 0.0,
            timestamp: 0.0,
        };

        let adjusted = adjuster.adjust_signal(signal, &snapshot);

        // In warning with long position, should only have asks
        assert!(adjusted.bid_levels.is_empty());
        assert!(!adjusted.ask_levels.is_empty());
        assert!(adjusted.was_modified);
    }
}
