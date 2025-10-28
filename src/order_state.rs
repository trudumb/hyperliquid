// ============================================================================
// Order State Management
// ============================================================================
//
// This module provides a reusable component for tracking order lifecycle states
// using WebSocket orderUpdates messages. It solves race condition issues where
// fills arrive before cancellation confirmations, or where order state needs to
// be reconciled across multiple event sources.
//
// # Key Features
//
// - Tracks pending orders (placement requested but not confirmed)
// - Maintains active orders with full state information
// - Caches recently completed orders for fill level lookup
// - Bidirectional cloid<->oid mapping
// - Automatic cache pruning to prevent memory leaks
//
// # Usage Example
//
// ```rust
// let mut order_mgr = OrderStateManager::new();
//
// // On order placement
// let cloid = Uuid::new_v4();
// let pending_order = RestingOrder::new(None, Some(cloid), 1.0, 100.0, true, 0);
// order_mgr.add_pending_order(cloid, pending_order);
//
// // On orderUpdate confirmation
// order_mgr.handle_order_update(order_update, &order_book);
//
// // On fill received
// if let Some(level) = order_mgr.get_order_level(fill.oid) {
//     println!("Fill on level {}", level);
// }
// ```

use crate::{OrderState, OrderUpdate, RestingOrder, OrderBook};
use log::{debug, error, info, warn};
use std::collections::HashMap;
use std::time::{Duration, Instant};
use uuid::Uuid;

/// Type alias for Client Order ID
pub type Cloid = Uuid;

/// Manages order state lifecycle using WebSocket orderUpdates
pub struct OrderStateManager {
    /// Maps Cloid -> OID (filled once confirmed by WS)
    cloid_to_oid: HashMap<Cloid, u64>,

    /// Maps OID -> Cloid (reverse lookup)
    oid_to_cloid: HashMap<u64, Cloid>,

    /// Orders pending placement confirmation (Cloid -> RestingOrder)
    pending_place_orders: HashMap<Cloid, RestingOrder>,

    /// Cache for recently completed orders (OID -> RestingOrder)
    /// Stores orders briefly after fill/cancel/reject/expire for level lookup
    recently_completed_orders: HashMap<u64, RestingOrder>,

    /// Last time we pruned the completed orders cache
    last_cache_prune_time: Instant,

    /// Cache TTL in seconds (default: 30s)
    cache_ttl_secs: u64,

    /// Cache prune interval in seconds (default: 10s)
    cache_prune_interval_secs: u64,
}

impl OrderStateManager {
    /// Create a new OrderStateManager with default settings
    pub fn new() -> Self {
        Self {
            cloid_to_oid: HashMap::new(),
            oid_to_cloid: HashMap::new(),
            pending_place_orders: HashMap::new(),
            recently_completed_orders: HashMap::new(),
            last_cache_prune_time: Instant::now(),
            cache_ttl_secs: 30,
            cache_prune_interval_secs: 10,
        }
    }

    /// Create a new OrderStateManager with custom cache settings
    pub fn with_cache_settings(cache_ttl_secs: u64, cache_prune_interval_secs: u64) -> Self {
        Self {
            cloid_to_oid: HashMap::new(),
            oid_to_cloid: HashMap::new(),
            pending_place_orders: HashMap::new(),
            recently_completed_orders: HashMap::new(),
            last_cache_prune_time: Instant::now(),
            cache_ttl_secs,
            cache_prune_interval_secs,
        }
    }

    /// Add a pending order (called when placement request is sent)
    pub fn add_pending_order(&mut self, cloid: Cloid, order: RestingOrder) {
        debug!("Adding pending order: Cloid {}", cloid);
        self.pending_place_orders.insert(cloid, order);
    }

    /// Confirm order placement (called when orderUpdate confirms)
    /// Returns the confirmed order with OID assigned
    pub fn confirm_order_placement(&mut self, cloid: Cloid, oid: u64) -> Option<RestingOrder> {
        if let Some(mut order) = self.pending_place_orders.remove(&cloid) {
            order.oid = Some(oid);
            order.state = OrderState::Active;
            order.timestamp = chrono::Utc::now().timestamp_millis() as u64;

            // Update mappings
            self.cloid_to_oid.insert(cloid, oid);
            self.oid_to_cloid.insert(oid, cloid);

            info!("Order placement confirmed: Cloid {} -> OID {}", cloid, oid);
            Some(order)
        } else {
            warn!("Attempted to confirm unknown pending order: Cloid {}", cloid);
            None
        }
    }

    /// Mark an order as pending cancellation
    pub fn mark_pending_cancel(&mut self, oid: u64, active_orders: &mut [RestingOrder]) -> bool {
        let current_timestamp = chrono::Utc::now().timestamp_millis() as u64;

        if let Some(order) = active_orders.iter_mut().find(|o| o.oid == Some(oid)) {
            if order.state == OrderState::Active || order.state == OrderState::PartiallyFilled {
                order.state = OrderState::PendingCancel;
                order.timestamp = current_timestamp;
                debug!("Marked OID {} as PendingCancel", oid);
                return true;
            }
        }
        false
    }

    /// Remove an order and cache it with a final state
    pub fn remove_and_cache_order(&mut self, oid: u64, final_state: OrderState, active_orders: &mut Vec<RestingOrder>) -> bool {
        let current_timestamp = chrono::Utc::now().timestamp_millis() as u64;

        // Try to find and remove from active orders
        if let Some(pos) = active_orders.iter().position(|o| o.oid == Some(oid)) {
            let mut order = active_orders.remove(pos);
            order.state = final_state.clone();
            order.timestamp = current_timestamp;
            self.recently_completed_orders.insert(oid, order);
            info!("Order OID {} moved to cache with state {:?}", oid, final_state);

            // Clean up mappings
            if let Some(cloid) = self.oid_to_cloid.remove(&oid) {
                self.cloid_to_oid.remove(&cloid);
            }
            return true;
        }

        // Also check pending orders
        if let Some(cloid) = self.oid_to_cloid.get(&oid).copied() {
            if let Some(mut order) = self.pending_place_orders.remove(&cloid) {
                order.oid = Some(oid);
                order.state = final_state.clone();
                order.timestamp = current_timestamp;
                self.recently_completed_orders.insert(oid, order);
                info!("Pending order OID {} moved to cache with state {:?}", oid, final_state);

                // Clean up mappings
                self.oid_to_cloid.remove(&oid);
                self.cloid_to_oid.remove(&cloid);
                return true;
            }
        }

        debug!("Attempted to remove/cache OID {}, but not found in active/pending lists", oid);
        false
    }

    /// Get the level of an order (checks active and cached orders)
    pub fn get_order_level(&self, oid: u64, active_orders: &[RestingOrder]) -> Option<usize> {
        // Check active orders first
        if let Some(order) = active_orders.iter().find(|o| o.oid == Some(oid)) {
            return Some(order.level);
        }

        // Check cache
        if let Some(cached_order) = self.recently_completed_orders.get(&oid) {
            debug!("Order OID {} found in cache (State: {:?}), Level: {}", oid, cached_order.state, cached_order.level);
            return Some(cached_order.level);
        }

        None
    }

    /// Get a cloned copy of order details (checks active and cached orders)
    pub fn get_order(&self, oid: u64, active_orders: &[RestingOrder]) -> Option<RestingOrder> {
        // Check active orders first
        if let Some(order) = active_orders.iter().find(|o| o.oid == Some(oid)) {
            return Some(order.clone());
        }

        // Check cache
        self.recently_completed_orders.get(&oid).cloned()
    }

    /// Handle an orderUpdate message
    /// Returns the updated order if it should be added/updated in active lists
    pub fn handle_order_update(
        &mut self,
        update: &OrderUpdate,
        order_book: Option<&OrderBook>,
    ) -> OrderUpdateResult {
        let oid = update.order.oid;
        let status = update.status.as_str();
        let cloid_opt: Option<Cloid> = update.order.cloid.as_deref().and_then(|s| s.parse().ok());
        let current_timestamp = chrono::Utc::now().timestamp_millis() as u64;

        debug!("Processing OrderUpdate: OID {}, Cloid {:?}, Status: {}", oid, cloid_opt, status);

        match status {
            "open" | "resting" => {
                // Order confirmed active/resting on the book
                if let Some(cloid) = cloid_opt {
                    if let Some(confirmed_order) = self.confirm_order_placement(cloid, oid) {
                        return OrderUpdateResult::AddOrUpdate(confirmed_order);
                    }
                }

                // Order not in pending (might be reconnect/resync)
                // Create from update details
                let price = update.order.limit_px.parse().unwrap_or(0.0);
                let size = update.order.sz.parse().unwrap_or(0.0);
                let is_buy = update.order.side == "B";
                let level = calculate_order_level(price, is_buy, order_book);

                if price > 0.0 && size > 0.0 {
                    let new_order = RestingOrder {
                        oid: Some(oid),
                        cloid: cloid_opt,
                        size,
                        orig_size: update.order.orig_sz.parse().unwrap_or(size),
                        price,
                        is_buy,
                        level,
                        state: OrderState::Active,
                        timestamp: current_timestamp,
                    };

                    if let Some(cloid) = cloid_opt {
                        self.cloid_to_oid.insert(cloid, oid);
                        self.oid_to_cloid.insert(oid, cloid);
                    }

                    return OrderUpdateResult::AddOrUpdate(new_order);
                }
                OrderUpdateResult::NoAction
            }
            "canceled" | "cancelled" => {
                info!("Order OID {} confirmed Canceled", oid);
                OrderUpdateResult::RemoveAndCache(oid, OrderState::Cancelled)
            }
            "rejected" => {
                error!("Order OID {} Rejected!", oid);
                if let Some(cloid) = cloid_opt {
                    if let Some(mut order) = self.pending_place_orders.remove(&cloid) {
                        warn!("Placement Rejected for Cloid {}: {:?}", cloid, order);
                        order.oid = Some(oid);
                        order.state = OrderState::Rejected;
                        order.timestamp = current_timestamp;
                        self.recently_completed_orders.insert(oid, order);
                        return OrderUpdateResult::NoAction;
                    }
                }
                OrderUpdateResult::RemoveAndCache(oid, OrderState::Rejected)
            }
            "filled" => {
                info!("Order OID {} confirmed Fully Filled", oid);
                OrderUpdateResult::RemoveAndCache(oid, OrderState::Filled)
            }
            "partiallyFilled" => {
                debug!("Order OID {} is Partially Filled", oid);
                let remaining_size = update.order.sz.parse().unwrap_or(0.0);
                let price = update.order.limit_px.parse().unwrap_or(0.0);
                let is_buy = update.order.side == "B";
                let level = calculate_order_level(price, is_buy, order_book);

                if price > 0.0 && remaining_size > 0.0 {
                    let updated_order = RestingOrder {
                        oid: Some(oid),
                        cloid: cloid_opt,
                        size: remaining_size,
                        orig_size: update.order.orig_sz.parse().unwrap_or(remaining_size),
                        price,
                        is_buy,
                        level,
                        state: OrderState::PartiallyFilled,
                        timestamp: current_timestamp,
                    };

                    if let Some(cloid) = cloid_opt {
                        self.cloid_to_oid.insert(cloid, oid);
                        self.oid_to_cloid.insert(oid, cloid);
                    }

                    return OrderUpdateResult::UpdatePartial(updated_order);
                }
                OrderUpdateResult::NoAction
            }
            "expired" => {
                info!("Order OID {} Expired", oid);
                OrderUpdateResult::RemoveAndCache(oid, OrderState::Expired)
            }
            "sending" | "pendingCancel" => {
                debug!("Order OID {} has transient status: {}", oid, status);
                OrderUpdateResult::NoAction
            }
            _ => {
                warn!("Received unknown order status '{}' for OID {}", status, oid);
                OrderUpdateResult::NoAction
            }
        }
    }

    /// Prune old orders from the cache if needed
    /// Returns the number of orders pruned
    pub fn prune_cache_if_needed(&mut self) -> usize {
        let now = Instant::now();
        if now.duration_since(self.last_cache_prune_time) > Duration::from_secs(self.cache_prune_interval_secs) {
            let pruned = self.prune_cache();
            self.last_cache_prune_time = now;
            if pruned > 0 {
                debug!("Pruned {} orders from recently_completed_orders cache", pruned);
            }
            pruned
        } else {
            0
        }
    }

    /// Prune old orders from the cache
    /// Returns the number of orders pruned
    fn prune_cache(&mut self) -> usize {
        let cutoff_timestamp = (chrono::Utc::now() - chrono::Duration::seconds(self.cache_ttl_secs as i64))
            .timestamp_millis() as u64;
        let initial_size = self.recently_completed_orders.len();
        self.recently_completed_orders.retain(|_oid, order| order.timestamp >= cutoff_timestamp);
        initial_size - self.recently_completed_orders.len()
    }

    /// Get the number of pending orders
    pub fn pending_count(&self) -> usize {
        self.pending_place_orders.len()
    }

    /// Get the number of cached orders
    pub fn cached_count(&self) -> usize {
        self.recently_completed_orders.len()
    }

    /// Check if an order is pending
    pub fn is_pending(&self, cloid: &Cloid) -> bool {
        self.pending_place_orders.contains_key(cloid)
    }

    /// Get OID from CLOID
    pub fn get_oid(&self, cloid: &Cloid) -> Option<u64> {
        self.cloid_to_oid.get(cloid).copied()
    }

    /// Get CLOID from OID
    pub fn get_cloid(&self, oid: u64) -> Option<Cloid> {
        self.oid_to_cloid.get(&oid).copied()
    }
}

impl Default for OrderStateManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of processing an orderUpdate
#[derive(Debug)]
pub enum OrderUpdateResult {
    /// Order should be added or updated in active lists
    AddOrUpdate(RestingOrder),
    /// Order should have its partial fill updated
    UpdatePartial(RestingOrder),
    /// Order should be removed from active lists and cached
    RemoveAndCache(u64, OrderState),
    /// No action needed
    NoAction,
}

/// Calculate the order book level for a given price
/// Returns 0 if no order book available
fn calculate_order_level(price: f64, is_buy: bool, order_book: Option<&OrderBook>) -> usize {
    if let Some(book) = order_book {
        let levels = if is_buy { &book.bids } else { &book.asks };

        for (idx, level) in levels.iter().enumerate() {
            if let Ok(level_price) = level.px.parse::<f64>() {
                if is_buy {
                    // For bids, higher price = better level
                    if price >= level_price {
                        return idx;
                    }
                } else {
                    // For asks, lower price = better level
                    if price <= level_price {
                        return idx;
                    }
                }
            }
        }
        // If we didn't find a match, it's deeper than the visible book
        return levels.len();
    }
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_and_confirm_pending_order() {
        let mut mgr = OrderStateManager::new();
        let cloid = Uuid::new_v4();
        let order = RestingOrder::new(None, Some(cloid), 1.0, 100.0, true, 0);

        mgr.add_pending_order(cloid, order.clone());
        assert_eq!(mgr.pending_count(), 1);
        assert!(mgr.is_pending(&cloid));

        let confirmed = mgr.confirm_order_placement(cloid, 12345);
        assert!(confirmed.is_some());
        assert_eq!(confirmed.unwrap().oid, Some(12345));
        assert_eq!(mgr.pending_count(), 0);
        assert_eq!(mgr.get_oid(&cloid), Some(12345));
    }

    #[test]
    fn test_cache_pruning() {
        let mut mgr = OrderStateManager::with_cache_settings(1, 1); // 1 second TTL

        let order = RestingOrder {
            oid: Some(123),
            cloid: None,
            size: 1.0,
            orig_size: 1.0,
            price: 100.0,
            is_buy: true,
            level: 0,
            state: OrderState::Filled,
            timestamp: chrono::Utc::now().timestamp_millis() as u64,
        };

        mgr.recently_completed_orders.insert(123, order);
        assert_eq!(mgr.cached_count(), 1);

        // Wait for cache to expire
        std::thread::sleep(std::time::Duration::from_secs(2));
        mgr.last_cache_prune_time = Instant::now() - Duration::from_secs(2);

        let pruned = mgr.prune_cache_if_needed();
        assert_eq!(pruned, 1);
        assert_eq!(mgr.cached_count(), 0);
    }

    #[test]
    fn test_get_order_level() {
        let mut mgr = OrderStateManager::new();
        let order = RestingOrder {
            oid: Some(123),
            cloid: None,
            size: 1.0,
            orig_size: 1.0,
            price: 100.0,
            is_buy: true,
            level: 2,
            state: OrderState::Active,
            timestamp: chrono::Utc::now().timestamp_millis() as u64,
        };

        let active_orders = vec![order.clone()];

        // Should find in active orders
        assert_eq!(mgr.get_order_level(123, &active_orders), Some(2));

        // Move to cache
        let mut active = vec![order];
        mgr.remove_and_cache_order(123, OrderState::Filled, &mut active);

        // Should find in cache
        assert_eq!(mgr.get_order_level(123, &[]), Some(2));
    }
}
