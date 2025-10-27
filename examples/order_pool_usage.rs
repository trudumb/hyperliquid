/// Example demonstrating OrderPool usage for high-frequency trading
///
/// This example shows how to use OrderPool to minimize allocations
/// in the hot path for maximum performance.

use hyperliquid_rust_sdk::{Error, OrderPool};

type Result<T> = std::result::Result<T, Error>;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logger
    env_logger::init();

    println!("=== OrderPool Usage Example ===\n");

    // Example 1: Basic OrderPool usage
    println!("1. Basic OrderPool Usage:");
    basic_order_pool_example();

    // Example 2: FastOrderSender with OrderPool
    println!("\n2. FastOrderSender with OrderPool:");
    fast_order_sender_example().await?;

    // Example 3: Monitoring pool performance
    println!("\n3. Monitoring Pool Performance:");
    monitoring_example();

    Ok(())
}

/// Basic example showing manual order pool usage
fn basic_order_pool_example() {
    // Create an order pool with 16 pre-allocated orders
    let pool = OrderPool::with_initial_size(16);

    // Acquire an order from the pool
    let mut order = pool.acquire();

    // Configure the order
    order.asset = "BTC".to_string();
    order.is_buy = true;
    order.sz = 0.1;
    order.limit_px = 50000.0;

    println!("  Configured order: {:?}", order.asset);
    println!("  Size: {}, Price: {}", order.sz, order.limit_px);

    // After using the order, return it to the pool
    pool.release(order);

    // Check pool stats
    let stats = pool.stats();
    println!("  Pool stats - Available: {}, Hits: {}, Misses: {}",
             stats.available, stats.hits, stats.misses);

    // Helper methods for common patterns
    let order = pool.acquire_limit_order(
        "ETH".to_string(),
        false,  // sell
        1.0,
        3000.0,
        Some("Ioc".to_string()),
    );

    println!("  Created {} {} order",
             if order.is_buy { "buy" } else { "sell" },
             order.asset);

    pool.release(order);
}

/// Example showing FastOrderSender with integrated OrderPool
async fn fast_order_sender_example() -> Result<()> {
    // Note: This requires actual exchange credentials
    // For demonstration purposes, we'll just show the API

    println!("  FastOrderSender can be configured with custom pool sizes:");
    println!("  ```rust");
    println!("  let fast_sender = FastOrderSender::with_full_pool_config(");
    println!("      exchange_client,");
    println!("      buffer_pool_size: 16,    // buffers");
    println!("      buffer_capacity: 2048,    // bytes per buffer");
    println!("      order_pool_size: 32,      // pre-allocated orders");
    println!("  );");
    println!("  ```");

    println!("\n  The order pool is accessible for manual order building:");
    println!("  ```rust");
    println!("  let order = fast_sender.order_pool().acquire_limit_order(");
    println!("      \"BTC\".to_string(),");
    println!("      true,   // buy");
    println!("      0.1,    // size");
    println!("      50000.0, // price");
    println!("      None,   // use default TIF");
    println!("  );");
    println!("  ");
    println!("  // Use the order...");
    println!("  let response = fast_sender.place_order_fast(&order).await?;");
    println!("  ");
    println!("  // Return to pool when done");
    println!("  fast_sender.order_pool().release(order);");
    println!("  ```");

    Ok(())
}

/// Example showing how to monitor pool performance
fn monitoring_example() {
    let pool = OrderPool::with_initial_size(8);

    // Simulate some operations
    let mut orders = Vec::new();
    for i in 0..10 {
        let mut order = pool.acquire();
        order.asset = format!("ASSET{}", i);
        order.sz = (i as f64) * 0.1;
        orders.push(order);
    }

    // Return orders to pool
    for order in orders {
        pool.release(order);
    }

    // Check statistics
    let stats = pool.stats();
    println!("  Pool Statistics:");
    println!("    Available orders: {}", stats.available);
    println!("    Pool capacity: {}", stats.capacity);
    println!("    Cache hits: {}", stats.hits);
    println!("    Cache misses: {}", stats.misses);

    // Calculate hit rate
    if let Some(hit_rate) = pool.hit_rate() {
        println!("    Hit rate: {:.1}%", hit_rate * 100.0);

        if hit_rate < 0.8 {
            println!("    ⚠️  Low hit rate - consider increasing pool size");
        } else if hit_rate > 0.95 {
            println!("    ✓ Excellent hit rate - pool is well-sized");
        } else {
            println!("    ✓ Good hit rate - pool is adequately sized");
        }
    }

    // Reset stats for next measurement period
    pool.reset_stats();
    println!("\n  Stats reset - ready for next monitoring period");
}

/// Performance tips
#[allow(dead_code)]
fn performance_tips() {
    println!("=== Performance Tips ===\n");

    println!("1. Pool Sizing:");
    println!("   - Size the pool based on your maximum concurrent orders");
    println!("   - Monitor hit rate: aim for >80%, ideally >90%");
    println!("   - A hit rate <80% means you should increase pool size");
    println!();

    println!("2. Order Lifecycle:");
    println!("   - Always release orders back to the pool when done");
    println!("   - Don't hold orders longer than necessary");
    println!("   - Consider using a defer pattern to ensure release");
    println!();

    println!("3. Integration with FastOrderSender:");
    println!("   - Use with_full_pool_config() to tune both pools");
    println!("   - Monitor both buffer and order pool statistics");
    println!("   - Adjust pool sizes based on your workload");
    println!();

    println!("4. Advanced Usage:");
    println!("   - For ultra-low latency, pre-warm pools on startup");
    println!("   - Use acquire_limit_order() helpers for common patterns");
    println!("   - Reset stats periodically to track recent performance");
}
