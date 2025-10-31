// ============================================================================
// Trading State Store - Single Source of Truth
// ============================================================================
//
// Provides a unified state store for all trading state. Components read from
// and write to this store rather than maintaining their own state copies.
// State changes trigger events on the event bus.

use std::collections::BTreeMap;
use std::sync::Arc;
use parking_lot::RwLock;
use log::{debug, warn};

use super::event_bus::{EventBus, TradingEvent};
use super::order_state_machine::OrderStateMachine;

// ============================================================================
// Market Data State
// ============================================================================

/// Processed market data state
#[derive(Debug, Clone)]
pub struct MarketData {
    /// Current mid price
    pub mid_price: f64,

    /// Best bid price
    pub best_bid: f64,

    /// Best ask price
    pub best_ask: f64,

    /// Bid size at best level
    pub bid_size: f64,

    /// Ask size at best level
    pub ask_size: f64,

    /// Spread in basis points
    pub spread_bps: f64,

    /// Order book imbalance (-1 to +1)
    pub imbalance: f64,

    /// Current volatility estimate in bps
    pub volatility_bps: f64,

    /// Volatility uncertainty in bps
    pub vol_uncertainty_bps: f64,

    /// Adverse selection estimate
    pub adverse_selection: f64,

    /// Last update timestamp
    pub timestamp: f64,
}

impl Default for MarketData {
    fn default() -> Self {
        Self {
            mid_price: 0.0,
            best_bid: 0.0,
            best_ask: 0.0,
            bid_size: 0.0,
            ask_size: 0.0,
            spread_bps: 0.0,
            imbalance: 0.0,
            volatility_bps: 10.0,
            vol_uncertainty_bps: 5.0,
            adverse_selection: 0.0,
            timestamp: 0.0,
        }
    }
}

// ============================================================================
// Risk Metrics
// ============================================================================

/// Risk metrics for position and margin management
#[derive(Debug, Clone)]
pub struct RiskMetrics {
    /// Current position size
    pub position: f64,

    /// Account equity (PnL + deposits)
    pub account_equity: f64,

    /// Margin currently used
    pub margin_used: f64,

    /// Margin available for new positions
    pub margin_available: f64,

    /// Maintenance margin ratio
    pub maintenance_margin_ratio: f64,

    /// Distance to liquidation (0 = liquidation, 1 = safe)
    pub liquidation_distance: f64,

    /// Maximum position size allowed
    pub max_position_size: f64,

    /// Last update timestamp
    pub timestamp: f64,
}

impl Default for RiskMetrics {
    fn default() -> Self {
        Self {
            position: 0.0,
            account_equity: 0.0,
            margin_used: 0.0,
            margin_available: 0.0,
            maintenance_margin_ratio: 0.02,
            liquidation_distance: 1.0,
            max_position_size: 0.0,
            timestamp: 0.0,
        }
    }
}

// ============================================================================
// Order Management State
// ============================================================================

/// Information about an open order
#[derive(Debug, Clone)]
pub struct OpenOrder {
    /// Order ID
    pub order_id: u64,

    /// Client order ID (if any)
    pub client_order_id: Option<String>,

    /// Order size
    pub size: f64,

    /// Order price
    pub price: f64,

    /// Is buy order
    pub is_buy: bool,

    /// Remaining size (after partial fills)
    pub remaining_size: f64,

    /// State machine for this order
    pub state_machine: OrderStateMachine,

    /// Timestamp when order was created
    pub created_at: f64,
}

// ============================================================================
// Trading Snapshot
// ============================================================================

/// Complete snapshot of trading state (for components to read)
#[derive(Debug, Clone)]
pub struct TradingSnapshot {
    /// Market data
    pub market_data: MarketData,

    /// Risk metrics
    pub risk_metrics: RiskMetrics,

    /// Open orders by order ID
    pub open_orders: BTreeMap<u64, OpenOrder>,

    /// Number of open buy orders
    pub num_buy_orders: usize,

    /// Number of open sell orders
    pub num_sell_orders: usize,

    /// Total open buy size
    pub total_buy_size: f64,

    /// Total open sell size
    pub total_sell_size: f64,

    /// Snapshot timestamp
    pub timestamp: f64,
}

// ============================================================================
// Trading State Store
// ============================================================================

/// Central state store for all trading state
pub struct TradingStateStore {
    /// Market data
    market_data: Arc<RwLock<MarketData>>,

    /// Risk metrics
    risk_metrics: Arc<RwLock<RiskMetrics>>,

    /// Open orders (by order ID)
    open_orders: Arc<RwLock<BTreeMap<u64, OpenOrder>>>,

    /// Event bus for publishing state changes
    event_bus: Arc<EventBus>,
}

impl TradingStateStore {
    /// Create a new trading state store
    pub fn new(event_bus: Arc<EventBus>) -> Self {
        Self {
            market_data: Arc::new(RwLock::new(MarketData::default())),
            risk_metrics: Arc::new(RwLock::new(RiskMetrics::default())),
            open_orders: Arc::new(RwLock::new(BTreeMap::new())),
            event_bus,
        }
    }

    // ========================================================================
    // Position Updates
    // ========================================================================

    /// Update position and publish event if changed significantly
    pub fn update_position(&self, new_position: f64, timestamp: f64) -> bool {
        let mut metrics = self.risk_metrics.write();
        let old_position = metrics.position;

        // Check if position changed significantly (more than 0.001)
        if (old_position - new_position).abs() > 0.001 {
            metrics.position = new_position;
            metrics.timestamp = timestamp;

            // Publish event
            self.event_bus.publish(TradingEvent::PositionChanged {
                old: old_position,
                new: new_position,
                timestamp,
            });

            debug!("[STATE STORE] Position updated: {:.4} -> {:.4}", old_position, new_position);
            true
        } else {
            false
        }
    }

    /// Get current position
    pub fn get_position(&self) -> f64 {
        self.risk_metrics.read().position
    }

    // ========================================================================
    // Market Data Updates
    // ========================================================================

    /// Update market data
    pub fn update_market_data(&self, data: MarketData) {
        let mut market_data = self.market_data.write();
        *market_data = data.clone();

        // Publish event
        self.event_bus.publish(TradingEvent::MarketDataUpdated {
            mid_price: data.mid_price,
            spread_bps: data.spread_bps,
            volatility_bps: data.volatility_bps,
            timestamp: data.timestamp,
        });
    }

    /// Get current market data
    pub fn get_market_data(&self) -> MarketData {
        self.market_data.read().clone()
    }

    // ========================================================================
    // Risk Metrics Updates
    // ========================================================================

    /// Update risk metrics
    pub fn update_risk_metrics(&self, metrics: RiskMetrics) {
        let mut risk_metrics = self.risk_metrics.write();
        *risk_metrics = metrics.clone();

        // Publish event
        self.event_bus.publish(TradingEvent::RiskMetricsUpdated {
            position: metrics.position,
            margin_used: metrics.margin_used,
            margin_available: metrics.margin_available,
            timestamp: metrics.timestamp,
        });

        // Check for margin warning
        if metrics.margin_available < metrics.margin_used * 0.2 {
            self.event_bus.publish(TradingEvent::MarginWarning {
                available: metrics.margin_available,
                used: metrics.margin_used,
                threshold: metrics.margin_used * 0.2,
                timestamp: metrics.timestamp,
            });
        }
    }

    /// Get current risk metrics
    pub fn get_risk_metrics(&self) -> RiskMetrics {
        self.risk_metrics.read().clone()
    }

    // ========================================================================
    // Order Management
    // ========================================================================

    /// Add an open order
    pub fn add_order(&self, order: OpenOrder) {
        let order_id = order.order_id;
        let size = order.size;
        let price = order.price;
        let is_buy = order.is_buy;
        let timestamp = order.created_at;

        self.open_orders.write().insert(order_id, order);

        self.event_bus.publish(TradingEvent::OrderCreated {
            order_id,
            size,
            price,
            is_buy,
            timestamp,
        });

        debug!("[STATE STORE] Order added: id={}, size={:.4}, price={:.4}, is_buy={}",
            order_id, size, price, is_buy);
    }

    /// Remove an order (canceled or filled)
    pub fn remove_order(&self, order_id: u64) -> Option<OpenOrder> {
        self.open_orders.write().remove(&order_id)
    }

    /// Update order remaining size (after partial fill)
    pub fn update_order_remaining_size(&self, order_id: u64, remaining_size: f64) {
        if let Some(order) = self.open_orders.write().get_mut(&order_id) {
            order.remaining_size = remaining_size;
        }
    }

    /// Get an open order
    pub fn get_order(&self, order_id: u64) -> Option<OpenOrder> {
        self.open_orders.read().get(&order_id).cloned()
    }

    /// Get all open orders
    pub fn get_all_orders(&self) -> BTreeMap<u64, OpenOrder> {
        self.open_orders.read().clone()
    }

    /// Get count of open orders
    pub fn get_order_count(&self) -> usize {
        self.open_orders.read().len()
    }

    /// Get count of buy orders
    pub fn get_buy_order_count(&self) -> usize {
        self.open_orders.read().values().filter(|o| o.is_buy).count()
    }

    /// Get count of sell orders
    pub fn get_sell_order_count(&self) -> usize {
        self.open_orders.read().values().filter(|o| !o.is_buy).count()
    }

    /// Get total open buy size
    pub fn get_total_buy_size(&self) -> f64 {
        self.open_orders.read().values()
            .filter(|o| o.is_buy)
            .map(|o| o.remaining_size)
            .sum()
    }

    /// Get total open sell size
    pub fn get_total_sell_size(&self) -> f64 {
        self.open_orders.read().values()
            .filter(|o| !o.is_buy)
            .map(|o| o.remaining_size)
            .sum()
    }

    // ========================================================================
    // Snapshot
    // ========================================================================

    /// Get a complete snapshot of current state
    pub fn get_snapshot(&self) -> TradingSnapshot {
        let market_data = self.market_data.read().clone();
        let risk_metrics = self.risk_metrics.read().clone();
        let open_orders = self.open_orders.read().clone();

        let num_buy_orders = open_orders.values().filter(|o| o.is_buy).count();
        let num_sell_orders = open_orders.values().filter(|o| !o.is_buy).count();

        let total_buy_size: f64 = open_orders.values()
            .filter(|o| o.is_buy)
            .map(|o| o.remaining_size)
            .sum();

        let total_sell_size: f64 = open_orders.values()
            .filter(|o| !o.is_buy)
            .map(|o| o.remaining_size)
            .sum();

        TradingSnapshot {
            market_data,
            risk_metrics,
            open_orders,
            num_buy_orders,
            num_sell_orders,
            total_buy_size,
            total_sell_size,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs_f64(),
        }
    }

    /// Clear all state (for testing or reset)
    pub fn clear(&self) {
        *self.market_data.write() = MarketData::default();
        *self.risk_metrics.write() = RiskMetrics::default();
        self.open_orders.write().clear();
    }

    // ========================================================================
    // Order State Machine Integration
    // ========================================================================

    /// Tick all order state machines to check for timeouts
    pub fn tick_order_state_machines(&self, current_time: f64) {
        let mut orders = self.open_orders.write();
        let mut stuck_orders = Vec::new();

        for (order_id, order) in orders.iter_mut() {
            if let Some(duration) = order.state_machine.is_stuck() {
                stuck_orders.push((*order_id, duration));
            }

            if let Err(e) = order.state_machine.tick() {
                warn!("[STATE STORE] Order {} state machine error: {}", order_id, e);
            }
        }

        // Publish stuck order events
        for (order_id, duration) in stuck_orders {
            if let Some(order) = orders.get(&order_id) {
                let state = order.state_machine.state().state_name().to_string();
                self.event_bus.publish(TradingEvent::OrderStuck {
                    order_id,
                    state,
                    duration_secs: duration.as_secs_f64(),
                    timestamp: current_time,
                });
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::order_state_machine::{StateMachineConfig};

    #[test]
    fn test_position_update() {
        let event_bus = Arc::new(EventBus::new());
        let store = TradingStateStore::new(event_bus.clone());

        // Update position
        assert!(store.update_position(10.0, 0.0));
        assert_eq!(store.get_position(), 10.0);

        // Small change should not trigger event
        assert!(!store.update_position(10.0005, 1.0));

        // Significant change should trigger event
        assert!(store.update_position(15.0, 2.0));
        assert_eq!(store.get_position(), 15.0);

        // Check events
        let events = event_bus.get_events_of_type("PositionChanged");
        assert_eq!(events.len(), 2); // Two significant changes
    }

    #[test]
    fn test_order_management() {
        let event_bus = Arc::new(EventBus::new());
        let store = TradingStateStore::new(event_bus.clone());

        // Add order
        let order = OpenOrder {
            order_id: 1,
            client_order_id: Some("test".to_string()),
            size: 10.0,
            price: 100.0,
            is_buy: true,
            remaining_size: 10.0,
            state_machine: OrderStateMachine::from_open_order(
                1, 10.0, 100.0, true,
                StateMachineConfig::default()
            ),
            created_at: 0.0,
        };

        store.add_order(order);

        assert_eq!(store.get_order_count(), 1);
        assert_eq!(store.get_buy_order_count(), 1);
        assert_eq!(store.get_total_buy_size(), 10.0);

        // Remove order
        let removed = store.remove_order(1);
        assert!(removed.is_some());
        assert_eq!(store.get_order_count(), 0);
    }

    #[test]
    fn test_snapshot() {
        let event_bus = Arc::new(EventBus::new());
        let store = TradingStateStore::new(event_bus.clone());

        store.update_position(10.0, 0.0);

        let snapshot = store.get_snapshot();
        assert_eq!(snapshot.risk_metrics.position, 10.0);
        assert_eq!(snapshot.num_buy_orders, 0);
    }
}
