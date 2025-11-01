use parking_lot::Mutex;
use std::sync::Arc;
use uuid::Uuid;

use crate::exchange::{ClientLimit, ClientOrder, ClientOrderRequest};

/// Configuration constants for the order pool
const MIN_POOL_SIZE: usize = 16; // Minimum orders to keep in pool
const MAX_POOL_SIZE: usize = 256; // Maximum orders to cache

/// Statistics about the order pool
#[derive(Debug, Clone)]
pub struct OrderPoolStats {
    pub available: usize,
    pub capacity: usize,
    pub hits: u64,
    pub misses: u64,
}

/// Pre-allocated order pool for zero-copy order handling
///
/// This struct reduces allocations in the hot path by maintaining a pool
/// of reusable ClientOrderRequest objects. This is particularly beneficial
/// for high-frequency trading where order submission latency is critical.
///
/// # Benefits
/// - Eliminates allocations for order objects in the hot path
/// - Uses parking_lot::Mutex for faster locking than std::Mutex
/// - Thread-safe and can be shared across multiple threads
/// - Provides statistics for monitoring pool efficiency
///
/// # Example
/// ```
/// use hyperliquid_rust_sdk::OrderPool;
///
/// let pool = OrderPool::new();
///
/// // Get an order from the pool
/// let mut order = pool.acquire();
/// order.asset = "BTC".to_string();
/// order.is_buy = true;
/// order.sz = 1.0;
/// order.limit_px = 50000.0;
///
/// // Use the order...
///
/// // Return it to the pool for reuse
/// pool.release(order);
/// ```
#[derive(Clone)]
pub struct OrderPool {
    pool: Arc<Mutex<PoolInner>>,
}

struct PoolInner {
    orders: Vec<ClientOrderRequest>,
    hits: u64,
    misses: u64,
}

impl OrderPool {
    /// Create a new order pool with default configuration
    pub fn new() -> Self {
        Self::with_initial_size(MIN_POOL_SIZE)
    }

    /// Create an order pool with a specific initial size
    ///
    /// # Arguments
    /// * `initial_size` - Number of orders to pre-allocate
    pub fn with_initial_size(initial_size: usize) -> Self {
        let mut orders = Vec::with_capacity(initial_size.max(MIN_POOL_SIZE));
        for _ in 0..initial_size {
            orders.push(Self::create_default_order());
        }

        Self {
            pool: Arc::new(Mutex::new(PoolInner {
                orders,
                hits: 0,
                misses: 0,
            })),
        }
    }

    /// Acquire an order from the pool
    ///
    /// If the pool is empty, creates a new order (counted as a "miss").
    /// Otherwise, returns a pre-allocated order (counted as a "hit").
    ///
    /// # Returns
    /// A ClientOrderRequest ready to be configured and used
    pub fn acquire(&self) -> ClientOrderRequest {
        let mut inner = self.pool.lock();
        if let Some(order) = inner.orders.pop() {
            inner.hits += 1;
            order
        } else {
            inner.misses += 1;
            Self::create_default_order()
        }
    }

    /// Return an order to the pool for reuse
    ///
    /// The order is cleared and reset to default values before being
    /// returned to the pool. If the pool is full (at MAX_POOL_SIZE),
    /// the order is dropped instead.
    ///
    /// # Arguments
    /// * `order` - The order to return to the pool
    pub fn release(&self, mut order: ClientOrderRequest) {
        // Clear the order for reuse
        Self::clear_order(&mut order);

        let mut inner = self.pool.lock();
        if inner.orders.len() < MAX_POOL_SIZE {
            inner.orders.push(order);
        }
        // If pool is full, drop the order (let it deallocate)
    }

    /// Get pool statistics for monitoring
    ///
    /// # Returns
    /// OrderPoolStats containing current pool state and usage statistics
    pub fn stats(&self) -> OrderPoolStats {
        let inner = self.pool.lock();
        OrderPoolStats {
            available: inner.orders.len(),
            capacity: inner.orders.capacity(),
            hits: inner.hits,
            misses: inner.misses,
        }
    }

    /// Reset pool statistics
    ///
    /// This is useful for benchmarking or monitoring pool efficiency
    /// over specific time periods.
    pub fn reset_stats(&self) {
        let mut inner = self.pool.lock();
        inner.hits = 0;
        inner.misses = 0;
    }

    /// Calculate the hit rate of the pool (0.0 to 1.0)
    ///
    /// A higher hit rate indicates the pool is well-sized for the workload.
    /// A hit rate below 0.8 might indicate the pool should be larger.
    ///
    /// # Returns
    /// Hit rate as a float, or None if no operations have been performed
    pub fn hit_rate(&self) -> Option<f64> {
        let inner = self.pool.lock();
        let total = inner.hits + inner.misses;
        if total == 0 {
            None
        } else {
            Some(inner.hits as f64 / total as f64)
        }
    }

    /// Create a default order with sensible defaults
    fn create_default_order() -> ClientOrderRequest {
        ClientOrderRequest {
            asset: String::new(),
            is_buy: false,
            reduce_only: false,
            limit_px: 0.0,
            sz: 0.0,
            cloid: None,
            order_type: ClientOrder::Limit(ClientLimit {
                tif: "Gtc".to_string(),
            }),
        }
    }

    /// Clear an order back to default values
    fn clear_order(order: &mut ClientOrderRequest) {
        order.asset.clear();
        order.is_buy = false;
        order.reduce_only = false;
        order.limit_px = 0.0;
        order.sz = 0.0;
        order.cloid = None;
        order.order_type = ClientOrder::Limit(ClientLimit {
            tif: "Gtc".to_string(),
        });
    }
}

impl Default for OrderPool {
    fn default() -> Self {
        Self::new()
    }
}

// Helper methods for building orders from pool
impl OrderPool {
    /// Acquire and configure a limit order
    ///
    /// # Arguments
    /// * `asset` - The asset/coin to trade
    /// * `is_buy` - true for buy, false for sell
    /// * `sz` - Size of the order
    /// * `limit_px` - Limit price
    /// * `tif` - Time in force (default: "Gtc")
    ///
    /// # Returns
    /// Configured ClientOrderRequest ready to submit
    pub fn acquire_limit_order(
        &self,
        asset: String,
        is_buy: bool,
        sz: f64,
        limit_px: f64,
        tif: Option<String>,
    ) -> ClientOrderRequest {
        let mut order = self.acquire();
        order.asset = asset;
        order.is_buy = is_buy;
        order.sz = sz;
        order.limit_px = limit_px;
        order.order_type = ClientOrder::Limit(ClientLimit {
            tif: tif.unwrap_or_else(|| "Gtc".to_string()),
        });
        order
    }

    /// Acquire and configure a limit order with client order ID
    pub fn acquire_limit_order_with_cloid(
        &self,
        asset: String,
        is_buy: bool,
        sz: f64,
        limit_px: f64,
        cloid: Uuid,
        tif: Option<String>,
    ) -> ClientOrderRequest {
        let mut order = self.acquire_limit_order(asset, is_buy, sz, limit_px, tif);
        order.cloid = Some(cloid);
        order
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_order_pool_acquire_release() {
        let pool = OrderPool::with_initial_size(4);

        // Initial state
        let stats = pool.stats();
        assert_eq!(stats.available, 4);
        assert_eq!(stats.hits, 0);
        assert_eq!(stats.misses, 0);

        // Acquire order (should be a hit)
        let order1 = pool.acquire();
        let stats = pool.stats();
        assert_eq!(stats.available, 3);
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 0);

        // Acquire all remaining orders
        let order2 = pool.acquire();
        let order3 = pool.acquire();
        let order4 = pool.acquire();
        let stats = pool.stats();
        assert_eq!(stats.available, 0);
        assert_eq!(stats.hits, 4);

        // Acquire when pool is empty (should be a miss)
        let order5 = pool.acquire();
        let stats = pool.stats();
        assert_eq!(stats.available, 0);
        assert_eq!(stats.hits, 4);
        assert_eq!(stats.misses, 1);

        // Release orders back to pool
        pool.release(order1);
        pool.release(order2);
        let stats = pool.stats();
        assert_eq!(stats.available, 2);

        // Release remaining
        pool.release(order3);
        pool.release(order4);
        pool.release(order5);
        let stats = pool.stats();
        assert_eq!(stats.available, 5);
    }

    #[test]
    fn test_order_pool_max_size() {
        let pool = OrderPool::with_initial_size(0);

        // Create more orders than MAX_POOL_SIZE
        let mut orders = Vec::new();
        for _ in 0..(MAX_POOL_SIZE + 10) {
            orders.push(pool.acquire());
        }

        // Release all orders
        for order in orders {
            pool.release(order);
        }

        // Pool should be capped at MAX_POOL_SIZE
        let stats = pool.stats();
        assert!(stats.available <= MAX_POOL_SIZE);
    }

    #[test]
    fn test_order_clear_on_release() {
        let pool = OrderPool::new();

        let mut order = pool.acquire();
        order.asset = "BTC".to_string();
        order.is_buy = true;
        order.sz = 1.0;
        order.limit_px = 50000.0;

        pool.release(order);

        let order2 = pool.acquire();
        assert_eq!(order2.asset, "");
        assert_eq!(order2.is_buy, false);
        assert_eq!(order2.sz, 0.0);
        assert_eq!(order2.limit_px, 0.0);
    }

    #[test]
    fn test_hit_rate() {
        let pool = OrderPool::with_initial_size(2);

        // Initially no operations
        assert_eq!(pool.hit_rate(), None);

        // Acquire 2 orders (2 hits)
        let o1 = pool.acquire();
        let o2 = pool.acquire();
        assert_eq!(pool.hit_rate(), Some(1.0)); // 2/2 = 100%

        // Acquire 1 more (1 miss)
        let o3 = pool.acquire();
        assert_eq!(pool.hit_rate(), Some(2.0 / 3.0)); // 2/3 = 66.6%

        // Return orders
        pool.release(o1);
        pool.release(o2);
        pool.release(o3);

        // Acquire 3 more (3 hits)
        let _o4 = pool.acquire();
        let _o5 = pool.acquire();
        let _o6 = pool.acquire();
        assert_eq!(pool.hit_rate(), Some(5.0 / 6.0)); // 5/6 = 83.3%
    }

    #[test]
    fn test_reset_stats() {
        let pool = OrderPool::new();

        let o = pool.acquire();
        pool.release(o);

        let stats = pool.stats();
        assert!(stats.hits > 0 || stats.misses > 0);

        pool.reset_stats();

        let stats = pool.stats();
        assert_eq!(stats.hits, 0);
        assert_eq!(stats.misses, 0);
    }

    #[test]
    fn test_acquire_limit_order() {
        let pool = OrderPool::new();

        let order = pool.acquire_limit_order(
            "BTC".to_string(),
            true,
            1.0,
            50000.0,
            Some("Ioc".to_string()),
        );

        assert_eq!(order.asset, "BTC");
        assert_eq!(order.is_buy, true);
        assert_eq!(order.sz, 1.0);
        assert_eq!(order.limit_px, 50000.0);

        match order.order_type {
            ClientOrder::Limit(ref limit) => assert_eq!(limit.tif, "Ioc"),
            _ => panic!("Expected limit order"),
        }
    }

    #[test]
    fn test_acquire_limit_order_with_cloid() {
        let pool = OrderPool::new();
        let cloid = Uuid::new_v4();

        let order = pool.acquire_limit_order_with_cloid(
            "ETH".to_string(),
            false,
            2.5,
            3000.0,
            cloid,
            None,
        );

        assert_eq!(order.asset, "ETH");
        assert_eq!(order.cloid, Some(cloid));
    }
}
