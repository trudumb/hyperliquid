// ============================================================================
// Order Executor - Order Lifecycle Management
// ============================================================================
//
// Handles the complexity of order placement, tracking, and cancellation.
// Reconciles desired state (from adjusted signals) with actual state
// (open orders on exchange).

use log::debug;

use crate::strategy::StrategyAction;
use crate::{ClientLimit, ClientOrder, ClientOrderRequest, ClientCancelRequest};

use super::risk_adjuster::AdjustedSignal;
use super::trading_state_store::TradingSnapshot;

// ============================================================================
// Execution Error
// ============================================================================

#[derive(Debug, Clone)]
pub enum ExecutionError {
    /// No orders to execute
    NoOrders,

    /// Failed to reconcile orders
    ReconciliationFailed(String),

    /// Invalid order parameters
    InvalidOrder(String),
}

impl std::fmt::Display for ExecutionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExecutionError::NoOrders => write!(f, "No orders to execute"),
            ExecutionError::ReconciliationFailed(msg) => {
                write!(f, "Reconciliation failed: {}", msg)
            }
            ExecutionError::InvalidOrder(msg) => write!(f, "Invalid order: {}", msg),
        }
    }
}

impl std::error::Error for ExecutionError {}

// ============================================================================
// Order Executor
// ============================================================================

/// Executes orders based on adjusted signals
pub struct OrderExecutor {
    /// Minimum price difference to requote (bps)
    requote_threshold_bps: f64,

    /// Asset being traded
    asset: String,

    /// Tick size for price rounding
    tick_size: f64,

    /// Lot size for size rounding
    lot_size: f64,
}

impl OrderExecutor {
    /// Create a new order executor
    pub fn new(
        _state_store: std::sync::Arc<super::trading_state_store::TradingStateStore>,
        _event_bus: std::sync::Arc<super::event_bus::EventBus>,
        asset: String,
        tick_size: f64,
        lot_size: f64,
        requote_threshold_bps: f64,
    ) -> Self {
        Self {
            requote_threshold_bps,
            asset,
            tick_size,
            lot_size,
        }
    }

    /// Execute an adjusted signal - convert to strategy actions
    pub fn execute(&self, signal: AdjustedSignal, snapshot: &TradingSnapshot)
        -> Result<Vec<StrategyAction>, ExecutionError>
    {
        let mut actions = Vec::new();

        // Reconcile bids
        let bid_actions = self.reconcile_side(
            &signal.bid_levels.iter().map(|l| (l.price, l.size)).collect::<Vec<_>>(),
            true,
            snapshot,
        );
        actions.extend(bid_actions);

        // Reconcile asks
        let ask_actions = self.reconcile_side(
            &signal.ask_levels.iter().map(|l| (l.price, l.size)).collect::<Vec<_>>(),
            false,
            snapshot,
        );
        actions.extend(ask_actions);

        // Handle taker orders if needed
        if signal.taker_buy_rate > 0.5 {
            // TODO: Implement taker buy logic
            debug!("[ORDER EXECUTOR] Taker buy rate: {}", signal.taker_buy_rate);
        }

        if signal.taker_sell_rate > 0.5 {
            // TODO: Implement taker sell logic
            debug!("[ORDER EXECUTOR] Taker sell rate: {}", signal.taker_sell_rate);
        }

        Ok(actions)
    }

    /// Reconcile one side (bids or asks)
    fn reconcile_side(
        &self,
        desired_levels: &[(f64, f64)], // (price, size)
        is_bid: bool,
        snapshot: &TradingSnapshot,
    ) -> Vec<StrategyAction> {
        let mut actions = Vec::new();
        let mid_price = snapshot.market_data.mid_price;

        // Get existing orders on this side
        let existing_orders: Vec<_> = snapshot.open_orders.values()
            .filter(|o| o.is_buy == is_bid)
            .collect();

        // Process desired levels: validate, round prices and sizes
        let mut valid_levels: Vec<(f64, f64)> = Vec::new();
        for &(price, size) in desired_levels {
            // Validate and round the price
            if !price.is_finite() || price <= 0.0 {
                debug!("[ORDER EXECUTOR] Skipping invalid price: {}", price);
                continue;
            }

            let rounded_price = self.round_price(price);
            let rounded_size = self.round_size(size);

            if rounded_size > 0.0 {
                valid_levels.push((rounded_price, rounded_size));
            }
        }

        // Calculate price tolerance for matching
        // Use tick_size as minimum tolerance if mid_price is invalid
        let price_tolerance = if mid_price > 0.0 {
            mid_price * self.requote_threshold_bps / 10000.0
        } else {
            self.tick_size * 2.0  // Fallback to 2 ticks tolerance
        };

        // Find orders to cancel (wrong price or size)
        for order in &existing_orders {
            let should_cancel = !valid_levels.iter().any(|(price, _size)| {
                (price - order.price).abs() < price_tolerance
            });

            if should_cancel {
                debug!("[ORDER EXECUTOR] Canceling order {} (price mismatch)", order.order_id);
                actions.push(StrategyAction::Cancel(ClientCancelRequest {
                    asset: self.asset.clone(),
                    oid: order.order_id,
                }));
            }
        }

        // Find prices that need new orders
        let existing_prices: Vec<f64> = existing_orders.iter().map(|o| o.price).collect();

        for &(price, size) in &valid_levels {
            let has_order = existing_prices.iter().any(|&ep| {
                (ep - price).abs() < price_tolerance
            });

            if !has_order {
                debug!("[ORDER EXECUTOR] Creating {} order: price={:.4}, size={:.4}",
                    if is_bid { "bid" } else { "ask" }, price, size);

                actions.push(StrategyAction::Place(ClientOrderRequest {
                    asset: self.asset.clone(),
                    is_buy: is_bid,
                    limit_px: price,
                    sz: size,
                    order_type: ClientOrder::Limit(ClientLimit {
                        tif: "Gtc".to_string(),
                    }),
                    reduce_only: false,
                    cloid: None,
                }));
            }
        }

        actions
    }

    /// Round price to tick size
    fn round_price(&self, price: f64) -> f64 {
        if !price.is_finite() || price <= 0.0 || self.tick_size <= 0.0 {
            return 0.0;  // Invalid input
        }
        (price / self.tick_size).round() * self.tick_size
    }

    /// Round size to lot size
    fn round_size(&self, size: f64) -> f64 {
        if !size.is_finite() || size <= 0.0 || self.lot_size <= 0.0 {
            return 0.0;  // Invalid input
        }
        (size / self.lot_size).round() * self.lot_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_price_rounding() {
        use std::sync::Arc;
        use super::super::{TradingStateStore, EventBus};

        let state_store = Arc::new(TradingStateStore::new(Arc::new(EventBus::new())));
        let event_bus = Arc::new(EventBus::new());
        let executor = OrderExecutor::new(
            state_store,
            event_bus,
            "TEST".to_string(),
            0.01,
            0.001,
            2.0,
        );

        assert_eq!(executor.round_price(100.123), 100.12);
        assert_eq!(executor.round_size(10.1234), 10.123);
    }
}
