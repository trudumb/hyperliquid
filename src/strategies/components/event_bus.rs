// ============================================================================
// Event Bus - System-wide Event Coordination
// ============================================================================
//
// Implements an event-driven architecture where components publish and
// subscribe to events instead of polling state. This reduces redundant
// checks and provides better observability.

use std::sync::Arc;
use parking_lot::RwLock;
use log::{debug, info, warn};

use crate::strategies::components::PositionState;

// ============================================================================
// Trading Events
// ============================================================================

/// Events that occur during trading operations
#[derive(Debug, Clone)]
pub enum TradingEvent {
    /// Position changed from old to new value
    PositionChanged {
        old: f64,
        new: f64,
        timestamp: f64,
    },

    /// An order was filled
    OrderFilled {
        order_id: u64,
        size: f64,
        price: f64,
        is_buy: bool,
        timestamp: f64,
    },

    /// An order was successfully canceled
    OrderCanceled {
        order_id: u64,
        timestamp: f64,
    },

    /// An order was created
    OrderCreated {
        order_id: u64,
        size: f64,
        price: f64,
        is_buy: bool,
        timestamp: f64,
    },

    /// Position state changed (Normal → Warning → Critical → OverLimit)
    PositionStateChanged {
        old: PositionState,
        new: PositionState,
        position: f64,
        timestamp: f64,
    },

    /// Margin warning - available margin is getting low
    MarginWarning {
        available: f64,
        used: f64,
        threshold: f64,
        timestamp: f64,
    },

    /// Emergency liquidation triggered
    EmergencyLiquidation {
        position: f64,
        reason: String,
        timestamp: f64,
    },

    /// Market data updated
    MarketDataUpdated {
        mid_price: f64,
        spread_bps: f64,
        volatility_bps: f64,
        timestamp: f64,
    },

    /// Risk metrics updated
    RiskMetricsUpdated {
        position: f64,
        margin_used: f64,
        margin_available: f64,
        timestamp: f64,
    },

    /// Order stuck in a bad state (e.g., pending cancel timeout)
    OrderStuck {
        order_id: u64,
        state: String,
        duration_secs: f64,
        timestamp: f64,
    },
}

impl TradingEvent {
    /// Get the timestamp of the event
    pub fn timestamp(&self) -> f64 {
        match self {
            TradingEvent::PositionChanged { timestamp, .. } => *timestamp,
            TradingEvent::OrderFilled { timestamp, .. } => *timestamp,
            TradingEvent::OrderCanceled { timestamp, .. } => *timestamp,
            TradingEvent::OrderCreated { timestamp, .. } => *timestamp,
            TradingEvent::PositionStateChanged { timestamp, .. } => *timestamp,
            TradingEvent::MarginWarning { timestamp, .. } => *timestamp,
            TradingEvent::EmergencyLiquidation { timestamp, .. } => *timestamp,
            TradingEvent::MarketDataUpdated { timestamp, .. } => *timestamp,
            TradingEvent::RiskMetricsUpdated { timestamp, .. } => *timestamp,
            TradingEvent::OrderStuck { timestamp, .. } => *timestamp,
        }
    }

    /// Get a short name for the event type
    pub fn event_type(&self) -> &'static str {
        match self {
            TradingEvent::PositionChanged { .. } => "PositionChanged",
            TradingEvent::OrderFilled { .. } => "OrderFilled",
            TradingEvent::OrderCanceled { .. } => "OrderCanceled",
            TradingEvent::OrderCreated { .. } => "OrderCreated",
            TradingEvent::PositionStateChanged { .. } => "PositionStateChanged",
            TradingEvent::MarginWarning { .. } => "MarginWarning",
            TradingEvent::EmergencyLiquidation { .. } => "EmergencyLiquidation",
            TradingEvent::MarketDataUpdated { .. } => "MarketDataUpdated",
            TradingEvent::RiskMetricsUpdated { .. } => "RiskMetricsUpdated",
            TradingEvent::OrderStuck { .. } => "OrderStuck",
        }
    }
}

// ============================================================================
// Event Subscriber Trait
// ============================================================================

/// Trait for components that want to receive events
pub trait EventSubscriber: Send + Sync {
    /// Called when an event is published
    fn on_event(&self, event: &TradingEvent);

    /// Return the name of this subscriber (for debugging)
    fn name(&self) -> &str;

    /// Whether this subscriber is interested in a specific event type
    /// Default implementation accepts all events
    fn is_interested_in(&self, event: &TradingEvent) -> bool {
        let _ = event;
        true
    }
}

// ============================================================================
// Event Bus
// ============================================================================

/// Central event bus for coordinating trading system components
pub struct EventBus {
    /// List of subscribers
    subscribers: Arc<RwLock<Vec<Arc<dyn EventSubscriber>>>>,

    /// Event history (for debugging and replay)
    history: Arc<RwLock<Vec<TradingEvent>>>,

    /// Maximum history size (to prevent unbounded growth)
    max_history: usize,

    /// Total events published
    total_events: Arc<RwLock<u64>>,
}

impl EventBus {
    /// Create a new event bus
    pub fn new() -> Self {
        Self {
            subscribers: Arc::new(RwLock::new(Vec::new())),
            history: Arc::new(RwLock::new(Vec::new())),
            max_history: 1000, // Keep last 1000 events
            total_events: Arc::new(RwLock::new(0)),
        }
    }

    /// Create a new event bus with custom history size
    pub fn with_history_size(max_history: usize) -> Self {
        Self {
            subscribers: Arc::new(RwLock::new(Vec::new())),
            history: Arc::new(RwLock::new(Vec::new())),
            max_history,
            total_events: Arc::new(RwLock::new(0)),
        }
    }

    /// Subscribe to events
    pub fn subscribe(&self, subscriber: Arc<dyn EventSubscriber>) {
        let mut subs = self.subscribers.write();
        info!("[EVENT BUS] New subscriber: {}", subscriber.name());
        subs.push(subscriber);
    }

    /// Publish an event to all subscribers
    pub fn publish(&self, event: TradingEvent) {
        // Increment counter
        {
            let mut count = self.total_events.write();
            *count += 1;
        }

        // Log event (only important ones)
        match &event {
            TradingEvent::PositionStateChanged { old, new, position, .. } => {
                info!("[EVENT] Position state: {:?} → {:?} (pos: {:.2})", old, new, position);
            }
            TradingEvent::EmergencyLiquidation { position, reason, .. } => {
                warn!("[EVENT] EMERGENCY LIQUIDATION: pos={:.2}, reason={}", position, reason);
            }
            TradingEvent::MarginWarning { available, used, .. } => {
                warn!("[EVENT] Margin warning: available={:.2}, used={:.2}", available, used);
            }
            TradingEvent::OrderStuck { order_id, state, duration_secs, .. } => {
                warn!("[EVENT] Order stuck: oid={}, state={}, duration={:.1}s",
                    order_id, state, duration_secs);
            }
            _ => {
                debug!("[EVENT] {}", event.event_type());
            }
        }

        // Store in history
        {
            let mut history = self.history.write();
            history.push(event.clone());

            // Trim history if too large
            if history.len() > self.max_history {
                let len = history.len();
                history.drain(0..len - self.max_history);
            }
        }

        // Notify subscribers
        let subscribers = self.subscribers.read();
        for subscriber in subscribers.iter() {
            if subscriber.is_interested_in(&event) {
                subscriber.on_event(&event);
            }
        }
    }

    /// Get recent event history
    pub fn get_history(&self, limit: usize) -> Vec<TradingEvent> {
        let history = self.history.read();
        let start = history.len().saturating_sub(limit);
        history[start..].to_vec()
    }

    /// Get total number of events published
    pub fn total_events(&self) -> u64 {
        *self.total_events.read()
    }

    /// Clear event history
    pub fn clear_history(&self) {
        self.history.write().clear();
    }

    /// Get events of a specific type
    pub fn get_events_of_type(&self, event_type: &str) -> Vec<TradingEvent> {
        let history = self.history.read();
        history.iter()
            .filter(|e| e.event_type() == event_type)
            .cloned()
            .collect()
    }
}

impl Default for EventBus {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Example Subscriber Implementations
// ============================================================================

/// Logger subscriber that logs all events
pub struct LoggingSubscriber {
    name: String,
}

impl LoggingSubscriber {
    pub fn new(name: String) -> Self {
        Self { name }
    }
}

impl EventSubscriber for LoggingSubscriber {
    fn on_event(&self, event: &TradingEvent) {
        debug!("[{}] Event: {:?}", self.name, event);
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// Metrics subscriber that tracks event statistics
pub struct MetricsSubscriber {
    name: String,
    fill_count: Arc<RwLock<u64>>,
    cancel_count: Arc<RwLock<u64>>,
    state_change_count: Arc<RwLock<u64>>,
}

impl MetricsSubscriber {
    pub fn new(name: String) -> Self {
        Self {
            name,
            fill_count: Arc::new(RwLock::new(0)),
            cancel_count: Arc::new(RwLock::new(0)),
            state_change_count: Arc::new(RwLock::new(0)),
        }
    }

    pub fn get_fill_count(&self) -> u64 {
        *self.fill_count.read()
    }

    pub fn get_cancel_count(&self) -> u64 {
        *self.cancel_count.read()
    }

    pub fn get_state_change_count(&self) -> u64 {
        *self.state_change_count.read()
    }
}

impl EventSubscriber for MetricsSubscriber {
    fn on_event(&self, event: &TradingEvent) {
        match event {
            TradingEvent::OrderFilled { .. } => {
                let mut count = self.fill_count.write();
                *count += 1;
            }
            TradingEvent::OrderCanceled { .. } => {
                let mut count = self.cancel_count.write();
                *count += 1;
            }
            TradingEvent::PositionStateChanged { .. } => {
                let mut count = self.state_change_count.write();
                *count += 1;
            }
            _ => {}
        }
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn is_interested_in(&self, event: &TradingEvent) -> bool {
        matches!(event,
            TradingEvent::OrderFilled { .. } |
            TradingEvent::OrderCanceled { .. } |
            TradingEvent::PositionStateChanged { .. }
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_event_bus_basic() {
        let bus = EventBus::new();
        let subscriber = Arc::new(MetricsSubscriber::new("test".to_string()));

        bus.subscribe(subscriber.clone());

        // Publish some events
        bus.publish(TradingEvent::OrderFilled {
            order_id: 1,
            size: 10.0,
            price: 100.0,
            is_buy: true,
            timestamp: 0.0,
        });

        bus.publish(TradingEvent::OrderCanceled {
            order_id: 2,
            timestamp: 1.0,
        });

        // Check metrics
        assert_eq!(subscriber.get_fill_count(), 1);
        assert_eq!(subscriber.get_cancel_count(), 1);
        assert_eq!(bus.total_events(), 2);
    }

    #[test]
    fn test_event_history() {
        let bus = EventBus::with_history_size(5);

        // Publish 10 events
        for i in 0..10 {
            bus.publish(TradingEvent::OrderCanceled {
                order_id: i,
                timestamp: i as f64,
            });
        }

        // Should only keep last 5
        let history = bus.get_history(100);
        assert_eq!(history.len(), 5);
        assert_eq!(bus.total_events(), 10);
    }
}
