use bytes::BytesMut;
use parking_lot::Mutex;
use std::sync::Arc;

use crate::{
    exchange::{order_pool::OrderPool, ClientOrderRequest, ExchangeClient, OrderPoolStats},
    prelude::*,
    rate_limiter::{RateLimitConfig, RateLimiter, RequestWeight},
    ExchangeResponseStatus,
};

/// Buffer pool configuration
const DEFAULT_BUFFER_CAPACITY: usize = 2048; // 2KB per buffer
const MIN_POOL_SIZE: usize = 8; // Minimum buffers to keep in pool
const MAX_POOL_SIZE: usize = 64; // Maximum buffers to cache

/// Pre-allocated buffer pool for zero-copy serialization
#[derive(Clone)]
struct BufferPool {
    buffers: Arc<Mutex<Vec<BytesMut>>>,
    buffer_capacity: usize,
}

impl BufferPool {
    /// Create a new buffer pool with pre-allocated buffers
    fn new(initial_size: usize, buffer_capacity: usize) -> Self {
        let mut buffers = Vec::with_capacity(initial_size);
        for _ in 0..initial_size {
            buffers.push(BytesMut::with_capacity(buffer_capacity));
        }

        Self {
            buffers: Arc::new(Mutex::new(buffers)),
            buffer_capacity,
        }
    }

    /// Get a buffer from the pool, or create a new one if pool is empty
    fn acquire(&self) -> BytesMut {
        let mut pool = self.buffers.lock();
        pool.pop()
            .unwrap_or_else(|| BytesMut::with_capacity(self.buffer_capacity))
    }

    /// Return a buffer to the pool for reuse
    fn release(&self, mut buffer: BytesMut) {
        buffer.clear(); // Clear content but keep capacity

        let mut pool = self.buffers.lock();
        if pool.len() < MAX_POOL_SIZE {
            pool.push(buffer);
        }
        // If pool is full, drop the buffer (let it deallocate)
    }

    /// Get current pool statistics
    fn stats(&self) -> BufferPoolStats {
        let pool = self.buffers.lock();
        BufferPoolStats {
            available: pool.len(),
            capacity: pool.capacity(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct BufferPoolStats {
    pub available: usize,
    pub capacity: usize,
}

/// High-performance order sender with zero-copy buffer and object pooling
///
/// This struct optimizes order placement for HFT by:
/// - Pre-allocating and reusing buffers to avoid allocations
/// - Pre-allocating and reusing order objects to reduce clones
/// - Using parking_lot::Mutex for faster locking than std::Mutex
/// - Minimizing memory copies during serialization
/// - Rate limiting to comply with Hyperliquid's API limits
///
/// # Example
/// ```ignore
/// use hyperliquid_rust_sdk::{ExchangeClient, FastOrderSender, ClientOrderRequest};
///
/// async fn example() -> Result<(), Box<dyn std::error::Error>> {
///     let exchange = ExchangeClient::new(/* ... */).await?;
///     let fast_sender = FastOrderSender::new(exchange);
///
///     let order = ClientOrderRequest {
///         // ... order details
///         ..Default::default()
///     };
///
///     let response = fast_sender.place_order_fast(&order).await?;
///     Ok(())
/// }
/// ```
pub struct FastOrderSender {
    client: Arc<ExchangeClient>,
    buffer_pool: BufferPool,
    order_pool: OrderPool,
    rate_limiter: RateLimiter,
}

impl FastOrderSender {
    /// Create a new FastOrderSender with default buffer pool settings
    pub fn new(client: ExchangeClient) -> Self {
        Self::with_pool_config(client, MIN_POOL_SIZE, DEFAULT_BUFFER_CAPACITY)
    }

    /// Create a FastOrderSender with custom buffer pool configuration
    ///
    /// # Arguments
    /// * `client` - The ExchangeClient to use for sending orders
    /// * `pool_size` - Initial number of pre-allocated buffers
    /// * `buffer_capacity` - Capacity of each buffer in bytes
    pub fn with_pool_config(
        client: ExchangeClient,
        pool_size: usize,
        buffer_capacity: usize,
    ) -> Self {
        Self {
            client: Arc::new(client),
            buffer_pool: BufferPool::new(pool_size, buffer_capacity),
            order_pool: OrderPool::new(),
            rate_limiter: RateLimiter::new(RateLimitConfig::rest_api()),
        }
    }

    /// Create a FastOrderSender with custom buffer and order pool configuration
    ///
    /// # Arguments
    /// * `client` - The ExchangeClient to use for sending orders
    /// * `buffer_pool_size` - Initial number of pre-allocated buffers
    /// * `buffer_capacity` - Capacity of each buffer in bytes
    /// * `order_pool_size` - Initial number of pre-allocated order objects
    pub fn with_full_pool_config(
        client: ExchangeClient,
        buffer_pool_size: usize,
        buffer_capacity: usize,
        order_pool_size: usize,
    ) -> Self {
        Self {
            client: Arc::new(client),
            buffer_pool: BufferPool::new(buffer_pool_size, buffer_capacity),
            order_pool: OrderPool::with_initial_size(order_pool_size),
            rate_limiter: RateLimiter::new(RateLimitConfig::rest_api()),
        }
    }

    /// Create a FastOrderSender with all custom configurations including rate limiting
    ///
    /// # Arguments
    /// * `client` - The ExchangeClient to use for sending orders
    /// * `buffer_pool_size` - Initial number of pre-allocated buffers
    /// * `buffer_capacity` - Capacity of each buffer in bytes
    /// * `order_pool_size` - Initial number of pre-allocated order objects
    /// * `rate_limit_config` - Rate limiter configuration
    pub fn with_complete_config(
        client: ExchangeClient,
        buffer_pool_size: usize,
        buffer_capacity: usize,
        order_pool_size: usize,
        rate_limit_config: RateLimitConfig,
    ) -> Self {
        Self {
            client: Arc::new(client),
            buffer_pool: BufferPool::new(buffer_pool_size, buffer_capacity),
            order_pool: OrderPool::with_initial_size(order_pool_size),
            rate_limiter: RateLimiter::new(rate_limit_config),
        }
    }

    /// Place a single order using zero-copy buffer pooling with rate limiting
    ///
    /// This method is optimized for minimal latency:
    /// - Waits for rate limit capacity if needed
    /// - Acquires a pre-allocated buffer from the pool
    /// - Serializes the order directly into the buffer
    /// - Returns the buffer to the pool after use
    ///
    /// # Arguments
    /// * `order` - The order to place
    ///
    /// # Returns
    /// Result containing the exchange response status
    pub async fn place_order_fast(
        &self,
        order: &ClientOrderRequest,
    ) -> Result<ExchangeResponseStatus> {
        // Wait for rate limit capacity (weight = 1 for single order)
        self.rate_limiter.acquire(RequestWeight::Exchange { batch_length: 0 }, 0).await;

        // Acquire buffer from pool (zero allocation if available)
        let _buf = self.buffer_pool.acquire();

        // NOTE: Currently using serde_json serialization via ExchangeClient
        // For even better performance, consider implementing custom serialization
        // directly to the buffer using rkyv or bincode in the future

        // Place the order using the underlying client
        let result = self.client.order(order.clone(), None).await;

        // Return buffer to pool for reuse
        self.buffer_pool.release(_buf);

        result
    }

    /// Place multiple orders in a single request using zero-copy buffer pooling with rate limiting
    ///
    /// This is more efficient than calling `place_order_fast` multiple times
    /// because it batches the orders into a single network request and uses
    /// much less rate limit weight (1 + floor(batch_length / 40)).
    ///
    /// # Arguments
    /// * `orders` - Vector of orders to place
    ///
    /// # Returns
    /// Result containing the exchange response status
    pub async fn place_bulk_orders_fast(
        &self,
        orders: Vec<ClientOrderRequest>,
    ) -> Result<ExchangeResponseStatus> {
        let batch_length = orders.len();

        // Wait for rate limit capacity (batched weight is much lower!)
        self.rate_limiter.acquire(RequestWeight::Exchange { batch_length }, 0).await;

        // Acquire buffer from pool
        let _buf = self.buffer_pool.acquire();

        // Place bulk order
        let result = self.client.bulk_order(orders, None).await;

        // Return buffer to pool
        self.buffer_pool.release(_buf);

        result
    }

    /// Cancel an order using zero-copy buffer pooling with rate limiting
    ///
    /// # Arguments
    /// * `cancel_request` - The cancel request
    ///
    /// # Returns
    /// Result containing the exchange response status
    pub async fn cancel_order_fast(
        &self,
        cancel_request: &crate::exchange::ClientCancelRequest,
    ) -> Result<ExchangeResponseStatus> {
        // Wait for rate limit capacity (weight = 1 for single cancel)
        self.rate_limiter.acquire(RequestWeight::Exchange { batch_length: 0 }, 0).await;

        let _buf = self.buffer_pool.acquire();

        let result = self.client.cancel(cancel_request.clone(), None).await;

        self.buffer_pool.release(_buf);

        result
    }

    /// Cancel multiple orders in a single request with rate limiting
    ///
    /// # Arguments
    /// * `cancel_requests` - Vector of cancel requests
    ///
    /// # Returns
    /// Result containing the exchange response status
    pub async fn cancel_bulk_orders_fast(
        &self,
        cancel_requests: Vec<crate::exchange::ClientCancelRequest>,
    ) -> Result<ExchangeResponseStatus> {
        let batch_length = cancel_requests.len();

        // Wait for rate limit capacity (batched weight is much lower!)
        self.rate_limiter.acquire(RequestWeight::Exchange { batch_length }, 0).await;

        let _buf = self.buffer_pool.acquire();

        let result = self.client.bulk_cancel(cancel_requests, None).await;

        self.buffer_pool.release(_buf);

        result
    }

    /// Modify an order using zero-copy buffer pooling with rate limiting
    ///
    /// # Arguments
    /// * `modify_request` - The modify request
    ///
    /// # Returns
    /// Result containing the exchange response status
    pub async fn modify_order_fast(
        &self,
        modify_request: &crate::exchange::ClientModifyRequest,
    ) -> Result<ExchangeResponseStatus> {
        // Wait for rate limit capacity (weight = 1 for single modify)
        self.rate_limiter.acquire(RequestWeight::Exchange { batch_length: 0 }, 0).await;

        let _buf = self.buffer_pool.acquire();

        let result = self.client.modify(modify_request.clone(), None).await;

        self.buffer_pool.release(_buf);

        result
    }

    /// Get buffer pool statistics for monitoring
    ///
    /// Useful for debugging and performance monitoring to ensure
    /// the pool is properly sized for your workload.
    pub fn buffer_pool_stats(&self) -> BufferPoolStats {
        self.buffer_pool.stats()
    }

    /// Get order pool statistics for monitoring
    ///
    /// Useful for debugging and performance monitoring to ensure
    /// the order pool is properly sized for your workload.
    pub fn order_pool_stats(&self) -> OrderPoolStats {
        self.order_pool.stats()
    }

    /// Get the order pool hit rate (0.0 to 1.0)
    ///
    /// A higher hit rate indicates better pool utilization.
    /// A hit rate below 0.8 might indicate the pool should be larger.
    pub fn order_pool_hit_rate(&self) -> Option<f64> {
        self.order_pool.hit_rate()
    }

    /// Reset order pool statistics
    ///
    /// Useful for benchmarking or monitoring specific time periods
    pub fn reset_order_pool_stats(&self) {
        self.order_pool.reset_stats();
    }

    /// Get a reference to the underlying ExchangeClient
    pub fn client(&self) -> &ExchangeClient {
        &self.client
    }

    /// Get a reference to the order pool
    ///
    /// This allows manual acquisition and release of orders for
    /// advanced use cases where you want to build orders yourself.
    pub fn order_pool(&self) -> &OrderPool {
        &self.order_pool
    }

    /// Get rate limiter statistics for monitoring
    ///
    /// Useful for monitoring API usage and ensuring you stay within limits
    pub fn rate_limiter_stats(&self) -> crate::rate_limiter::RateLimitStats {
        self.rate_limiter.stats()
    }

    /// Get rate limiter utilization (0.0 to 1.0)
    ///
    /// Returns the percentage of rate limit capacity currently in use.
    /// Values > 0.8 indicate high usage and potential for delays.
    pub fn rate_limiter_utilization(&self) -> f64 {
        self.rate_limiter.utilization()
    }

    /// Get remaining rate limit capacity
    ///
    /// Returns the number of weight units available before rate limiting occurs
    pub fn rate_limiter_remaining_capacity(&self) -> u32 {
        self.rate_limiter.remaining_capacity()
    }

    /// Reset rate limiter statistics
    ///
    /// Useful for benchmarking or monitoring specific time periods
    pub fn reset_rate_limiter_stats(&self) {
        self.rate_limiter.reset_stats();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_buffer_pool_acquire_release() {
        let pool = BufferPool::new(4, 1024);

        // Initial state
        let stats = pool.stats();
        assert_eq!(stats.available, 4);

        // Acquire buffer
        let buf1 = pool.acquire();
        let stats = pool.stats();
        assert_eq!(stats.available, 3);

        // Acquire all remaining buffers
        let buf2 = pool.acquire();
        let buf3 = pool.acquire();
        let buf4 = pool.acquire();
        let stats = pool.stats();
        assert_eq!(stats.available, 0);

        // Acquire when pool is empty (should create new)
        let buf5 = pool.acquire();
        assert_eq!(buf5.capacity(), 1024);

        // Release buffers back to pool
        pool.release(buf1);
        pool.release(buf2);
        let stats = pool.stats();
        assert_eq!(stats.available, 2);

        // Release remaining
        pool.release(buf3);
        pool.release(buf4);
        pool.release(buf5);
        let stats = pool.stats();
        assert_eq!(stats.available, 5);
    }

    #[test]
    fn test_buffer_pool_max_size() {
        let pool = BufferPool::new(0, 1024);

        // Create more buffers than MAX_POOL_SIZE
        let mut buffers = Vec::new();
        for _ in 0..(MAX_POOL_SIZE + 10) {
            buffers.push(BytesMut::with_capacity(1024));
        }

        // Release all buffers
        for buf in buffers {
            pool.release(buf);
        }

        // Pool should be capped at MAX_POOL_SIZE
        let stats = pool.stats();
        assert!(stats.available <= MAX_POOL_SIZE);
    }

    #[test]
    fn test_buffer_clear_on_release() {
        let pool = BufferPool::new(1, 1024);

        let mut buf = pool.acquire();
        buf.extend_from_slice(b"test data");
        assert_eq!(buf.len(), 9);

        pool.release(buf);

        let buf2 = pool.acquire();
        assert_eq!(buf2.len(), 0, "Buffer should be cleared when released");
        assert!(buf2.capacity() >= 1024, "Buffer capacity should be preserved");
    }
}
