/// Example demonstrating rate limiter usage for Hyperliquid API compliance
///
/// This example shows how to use the RateLimiter to stay within Hyperliquid's
/// documented rate limits for optimal performance without getting throttled.

use hyperliquid_rust_sdk::{
    rate_limiter::{RateLimitConfig, RateLimiter, RequestWeight},
    Error,
};
use std::time::{Duration, Instant};

type Result<T> = std::result::Result<T, Error>;

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::init();

    println!("=== Hyperliquid Rate Limiter Examples ===\n");

    // Example 1: Basic REST API rate limiting
    println!("1. Basic REST API Rate Limiting:");
    basic_rest_example().await?;

    // Example 2: WebSocket message rate limiting
    println!("\n2. WebSocket Message Rate Limiting:");
    websocket_example().await?;

    // Example 3: Batch request optimization
    println!("\n3. Batch Request Optimization:");
    batch_optimization_example().await?;

    // Example 4: Rate limiter monitoring
    println!("\n4. Rate Limiter Monitoring:");
    monitoring_example().await?;

    // Example 5: Custom configuration
    println!("\n5. Custom Configuration:");
    custom_config_example().await?;

    Ok(())
}

/// Basic REST API rate limiting following Hyperliquid's 1200 weight/minute limit
async fn basic_rest_example() -> Result<()> {
    // Create rate limiter with default REST API config (1200 weight/min)
    let limiter = RateLimiter::new(RateLimitConfig::rest_api());

    println!("  REST API Config: {} weight per {} seconds",
             RateLimitConfig::rest_api().max_weight,
             RateLimitConfig::rest_api().window_duration.as_secs());
    println!("  Safety margin: {:.0}%\n",
             RateLimitConfig::rest_api().safety_margin * 100.0);

    // Example: Single order request (weight = 1)
    println!("  Making single order request (weight: 1)...");
    limiter.acquire(RequestWeight::Exchange { batch_length: 0 }).await;
    println!("  ✓ Request completed. Remaining capacity: {}", limiter.remaining_capacity());

    // Example: Info request (weight = 20)
    println!("\n  Making standard info request (weight: 20)...");
    limiter.acquire(RequestWeight::InfoStandard).await;
    println!("  ✓ Request completed. Remaining capacity: {}", limiter.remaining_capacity());

    // Example: Light info request (weight = 2)
    println!("\n  Making light info request - l2Book (weight: 2)...");
    limiter.acquire(RequestWeight::InfoLight).await;
    println!("  ✓ Request completed. Remaining capacity: {}", limiter.remaining_capacity());

    // Example: Heavy info request (weight = 60)
    println!("\n  Making heavy info request - userRole (weight: 60)...");
    limiter.acquire(RequestWeight::InfoHeavy).await;
    println!("  ✓ Request completed. Remaining capacity: {}", limiter.remaining_capacity());

    println!("\n  Current utilization: {:.1}%", limiter.utilization() * 100.0);

    Ok(())
}

/// WebSocket message rate limiting (2000 messages/minute)
async fn websocket_example() -> Result<()> {
    // Create rate limiter for WebSocket messages
    let limiter = RateLimiter::new(RateLimitConfig::websocket_messages());

    println!("  WebSocket Config: {} messages per {} seconds",
             RateLimitConfig::websocket_messages().max_weight,
             RateLimitConfig::websocket_messages().window_duration.as_secs());

    // Simulate sending multiple WebSocket messages
    println!("\n  Simulating 10 WebSocket messages...");
    for i in 1..=10 {
        limiter.acquire(RequestWeight::Custom(1)).await;
        if i % 5 == 0 {
            println!("  Sent {} messages. Remaining capacity: {}", i, limiter.remaining_capacity());
        }
    }

    println!("\n  ✓ All messages sent successfully");
    println!("  Current utilization: {:.2}%", limiter.utilization() * 100.0);

    Ok(())
}

/// Demonstrate batch request optimization
async fn batch_optimization_example() -> Result<()> {
    let limiter = RateLimiter::new(RateLimitConfig::rest_api());

    println!("  Batch Request Weight Calculation:");
    println!("  - Single order: weight = 1");
    println!("  - 40 orders batched: weight = 2");
    println!("  - 79 orders batched: weight = 2");
    println!("  - 80 orders batched: weight = 3");

    // Example: Unbatched vs batched requests
    println!("\n  Scenario: Need to place 80 orders");

    // Unbatched: 80 requests * weight 1 = 80 weight
    let unbatched_weight = 80;
    println!("  - Unbatched (80 separate requests): {} weight", unbatched_weight);

    // Batched: 1 request * weight 3 = 3 weight
    let batched_weight = RequestWeight::Exchange { batch_length: 80 }.weight();
    println!("  - Batched (1 request with 80 orders): {} weight", batched_weight);

    println!("\n  Savings: {} weight ({:.1}x more efficient!)",
             unbatched_weight - batched_weight,
             unbatched_weight as f64 / batched_weight as f64);

    // Make the batched request
    println!("\n  Making batched request...");
    limiter.acquire(RequestWeight::Exchange { batch_length: 80 }).await;
    println!("  ✓ Batch request completed. Remaining capacity: {}", limiter.remaining_capacity());

    Ok(())
}

/// Demonstrate rate limiter monitoring and statistics
async fn monitoring_example() -> Result<()> {
    let limiter = RateLimiter::new(RateLimitConfig::rest_api());

    println!("  Making various requests to demonstrate monitoring...\n");

    // Make several requests
    for i in 1..=5 {
        let weight = match i {
            1 => RequestWeight::Exchange { batch_length: 0 },
            2 => RequestWeight::InfoLight,
            3 => RequestWeight::InfoStandard,
            4 => RequestWeight::InfoHeavy,
            5 => RequestWeight::Explorer,
            _ => unreachable!(),
        };

        println!("  Request {}: weight = {}", i, weight.weight());
        limiter.acquire(weight).await;
    }

    // Get statistics
    let stats = limiter.stats();
    println!("\n  Rate Limiter Statistics:");
    println!("  - Total requests: {}", stats.total_requests);
    println!("  - Total weight used: {}", stats.total_weight_used);
    println!("  - Times waited: {}", stats.times_waited);
    println!("  - Total wait time: {} ms", stats.total_wait_time_ms);
    println!("  - Current window: {} requests, {} weight",
             stats.requests_in_current_window,
             stats.current_weight_in_window);

    // Current state
    println!("\n  Current State:");
    println!("  - Utilization: {:.1}%", limiter.utilization() * 100.0);
    println!("  - Remaining capacity: {}", limiter.remaining_capacity());

    // Check if we can make another heavy request
    let next_weight = RequestWeight::InfoHeavy;
    let time_until = limiter.time_until_capacity(next_weight);
    if time_until.as_secs() > 0 {
        println!("  - Time until next heavy request available: {:?}", time_until);
    } else {
        println!("  - Can make heavy request immediately");
    }

    Ok(())
}

/// Demonstrate custom rate limiter configuration
async fn custom_config_example() -> Result<()> {
    use hyperliquid_rust_sdk::rate_limiter::RateLimiterBuilder;

    println!("  Creating custom rate limiter configurations...\n");

    // Conservative configuration (80% safety margin)
    let conservative = RateLimiterBuilder::new()
        .max_weight(1200)
        .window_duration(Duration::from_secs(60))
        .safety_margin(0.8) // Only use 80% of limit
        .build();

    println!("  Conservative Config (80% safety margin):");
    println!("  - Effective max weight: {}", RateLimitConfig {
        max_weight: 1200,
        window_duration: Duration::from_secs(60),
        safety_margin: 0.8,
    }.effective_max_weight());

    // Aggressive configuration (95% safety margin)
    let _aggressive = RateLimiterBuilder::new()
        .max_weight(1200)
        .window_duration(Duration::from_secs(60))
        .safety_margin(0.95) // Use 95% of limit
        .build();

    println!("\n  Aggressive Config (95% safety margin):");
    println!("  - Effective max weight: {}", RateLimitConfig {
        max_weight: 1200,
        window_duration: Duration::from_secs(60),
        safety_margin: 0.95,
    }.effective_max_weight());

    // Custom short window for burst protection
    let _burst_limiter = RateLimiterBuilder::new()
        .max_weight(100)
        .window_duration(Duration::from_secs(10)) // 10 second window
        .safety_margin(0.9)
        .build();

    println!("\n  Burst Protection Config:");
    println!("  - Max weight: 100 per 10 seconds");
    println!("  - Useful for preventing short bursts that could trigger rate limits");

    // Make requests with conservative limiter
    println!("\n  Testing conservative limiter...");
    for _ in 0..3 {
        conservative.acquire(RequestWeight::InfoStandard).await;
    }
    println!("  ✓ Made 3 requests (60 weight total)");
    println!("  Remaining capacity: {} (conservative approach)",
             conservative.remaining_capacity());

    Ok(())
}

/// Performance benchmarking
#[allow(dead_code)]
async fn benchmark_example() -> Result<()> {
    use tokio::time::sleep;

    println!("=== Performance Benchmark ===\n");

    let limiter = RateLimiter::new(RateLimitConfig {
        max_weight: 1000,
        window_duration: Duration::from_secs(1),
        safety_margin: 1.0,
    });

    println!("  Testing limiter performance with 100 requests...");

    let start = Instant::now();
    let mut waited = 0;

    for i in 0..100 {
        let weight = RequestWeight::Custom(10);

        // Check if we need to wait
        if !limiter.try_acquire(weight) {
            limiter.acquire(weight).await;
            waited += 1;
        }

        // Simulate some work
        if i % 10 == 0 {
            sleep(Duration::from_millis(100)).await;
        }
    }

    let elapsed = start.elapsed();

    println!("\n  Benchmark Results:");
    println!("  - Total time: {:?}", elapsed);
    println!("  - Requests: 100");
    println!("  - Times waited: {}", waited);
    println!("  - Average latency: {:?}", elapsed / 100);

    let stats = limiter.stats();
    println!("\n  Rate Limiter Stats:");
    println!("  - Total wait time: {} ms", stats.total_wait_time_ms);
    println!("  - Average wait time: {:.2} ms",
             if stats.times_waited > 0 {
                 stats.total_wait_time_ms as f64 / stats.times_waited as f64
             } else {
                 0.0
             });

    Ok(())
}

/// Best practices guide
#[allow(dead_code)]
fn best_practices() {
    println!("=== Rate Limiter Best Practices ===\n");

    println!("1. Configuration:");
    println!("   - Use default configs for Hyperliquid's documented limits");
    println!("   - Set safety_margin to 0.9 (90%) to avoid hitting exact limits");
    println!("   - For critical applications, use 0.8 (80%) for more headroom");
    println!();

    println!("2. Batching:");
    println!("   - Always batch orders when possible (much lower weight)");
    println!("   - Weight formula: 1 + floor(batch_length / 40)");
    println!("   - 40 orders batched = weight 2 vs 40 unbatched = weight 40");
    println!();

    println!("3. Request Planning:");
    println!("   - Use try_acquire() for non-critical requests");
    println!("   - Use acquire() for important requests (will wait)");
    println!("   - Check remaining_capacity() before making heavy requests");
    println!();

    println!("4. Monitoring:");
    println!("   - Check utilization() regularly (warn if > 80%)");
    println!("   - Monitor stats.times_waited (high = pool too small)");
    println!("   - Track total_wait_time_ms for latency analysis");
    println!();

    println!("5. WebSocket vs REST:");
    println!("   - Use WebSockets for real-time data (lower latency)");
    println!("   - Separate rate limiters for WS and REST");
    println!("   - WS limit: 2000 messages/min, 100 inflight posts");
    println!();

    println!("6. Multiple IP Addresses:");
    println!("   - Rate limits are per IP address");
    println!("   - Consider multiple IPs for higher throughput");
    println!("   - Each IP gets separate 1200 weight/min");
    println!();

    println!("7. Pagination:");
    println!("   - Remember additional weight for paginated responses");
    println!("   - +1 weight per 20 items for most endpoints");
    println!("   - +1 weight per 60 items for candleSnapshot");
    println!();
}
