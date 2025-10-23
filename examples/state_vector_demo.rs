use hyperliquid_rust_sdk::{
    AssetType, MarketMaker, MarketMakerInput, InventorySkewConfig,
};
use alloy::signers::local::PrivateKeySigner;
use log::info;

/// Example demonstrating the State Vector implementation
/// 
/// This example shows how to:
/// 1. Set up a market maker with state vector tracking
/// 2. Monitor state vector components in real-time
/// 3. Use state vector for decision making
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    
    // Load private key from environment
    let private_key = std::env::var("PRIVATE_KEY")
        .expect("PRIVATE_KEY environment variable not set");
    let wallet: PrivateKeySigner = private_key.parse()
        .expect("Failed to parse private key");
    
    // Asset to market make
    let asset = std::env::var("ASSET").unwrap_or_else(|_| "HYPE".to_string());
    
    info!("Starting State Vector Market Maker Demo");
    info!("Asset: {}", asset);
    
    // Configure inventory skewing to enable LOB analysis
    // This is required for the state vector to track imbalance
    let inventory_skew_config = InventorySkewConfig {
        inventory_skew_factor: 0.5,      // 50% inventory skew factor
        book_imbalance_factor: 0.3,      // 30% book imbalance factor
        depth_analysis_levels: 5,        // Analyze top 5 levels
    };
    
    // Create market maker with state vector
    let mut market_maker = MarketMaker::new(MarketMakerInput {
        asset: asset.clone(),
        target_liquidity: 100.0,           // $100 liquidity per side
        half_spread: 10,                   // 10 bps half spread (0.1%)
        max_bps_diff: 5,                   // Cancel/replace at 5 bps deviation
        max_absolute_position_size: 1000.0, // Max 1000 units position
        asset_type: AssetType::Perp,
        wallet,
        inventory_skew_config: Some(inventory_skew_config),
    })
    .await?;
    
    // Create a shutdown channel (Ctrl+C handler)
    let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel();
    
    // Spawn shutdown handler
    tokio::spawn(async move {
        tokio::signal::ctrl_c().await.ok();
        info!("Shutdown signal received");
        let _ = shutdown_tx.send(());
    });
    
    // Spawn monitoring task to demonstrate state vector access
    tokio::spawn(async move {
        loop {
            tokio::time::sleep(tokio::time::Duration::from_secs(30)).await;
            
            // In a real implementation, you would access the market maker's state vector
            // For this demo, we'll just log a reminder
            info!("=== State Vector Monitoring ===");
            info!("Check logs above for StateVector updates");
            info!("Format: StateVector[S=price, Q=inventory, μ̂=adverse_sel, Δ=spread_bps, I=imbalance]");
            info!("");
            info!("Interpretation Guide:");
            info!("  S (Mid Price): Current market price");
            info!("  Q (Inventory): Your position (+ = long, - = short)");
            info!("  μ̂ (Adverse Selection): Expected drift (+ = bullish, - = bearish)");
            info!("  Δ (Spread): Market spread in bps (higher = more volatile/less liquid)");
            info!("  I (Imbalance): Order book balance (0.5 = balanced, >0.5 = more bids, <0.5 = more asks)");
            info!("================================");
        }
    });
    
    // Start the market maker with shutdown signal
    info!("Market maker starting...");
    info!("State vector will be logged with each update");
    info!("Press Ctrl+C to shutdown gracefully");
    
    market_maker.start_with_shutdown_signal(Some(shutdown_rx)).await;
    
    info!("Market maker shut down successfully");
    Ok(())
}

// Example output you should see:
// 
// [INFO] Starting State Vector Market Maker Demo
// [INFO] Asset: HYPE
// [INFO] Market maker starting...
// [INFO] StateVector[S=0.50, Q=0.0000, μ̂=0.0000, Δ=0.0bps, I=0.000]
// [INFO] Buy for 100.0 HYPE resting at 0.4995
// [INFO] Sell for 100.0 HYPE resting at 0.5005
// [INFO] StateVector[S=0.50, Q=0.0000, μ̂=0.0012, Δ=8.5bps, I=0.623]
// [INFO] Fill: bought 50.0 HYPE (oid: 123456)
// [INFO] StateVector[S=0.50, Q=50.0000, μ̂=0.0015, Δ=8.2bps, I=0.645]
// 
// Interpretation of this sequence:
// 1. Initial state: No position, no drift estimate
// 2. Orders placed around mid
// 3. State updated with imbalance showing buying pressure (I=0.623 > 0.5)
// 4. Adverse selection estimate becomes slightly positive (μ̂=0.0012)
// 5. Fill occurs, inventory increases to 50
// 6. Imbalance increases further, adverse selection strengthens
