/*This is an advanced market making strategy using the V2 implementation.

The V2 market maker uses:
- State Vector (Z_t): Tracks mid-price, inventory, adverse selection, market spread, and LOB imbalance
- Control Vector (u_t): Determines bid/ask offsets and taker order rates
- HJB (Hamilton-Jacobi-Bellman) optimization for optimal quote placement
- Value Function V(Q, Z, t) for inventory management

The algorithm continuously monitors market state and adjusts quotes to maximize expected P&L
while managing inventory risk through state-aware control policies.
*/
use alloy::signers::local::PrivateKeySigner;
use hyperliquid_rust_sdk::{
    AssetType, InventorySkewConfig,
};
use hyperliquid_rust_sdk::market_maker_v2::{MarketMaker, MarketMakerInput};
use std::env;
use tokio::signal;
use log::info;

#[tokio::main]
async fn main() {
    env_logger::init();
    
    // Load environment variables from .env file
    dotenv::dotenv().ok();
    
    // Load private key from environment variable
    let private_key = env::var("PRIVATE_KEY")
        .expect("PRIVATE_KEY environment variable must be set");
    
    let wallet: PrivateKeySigner = private_key
        .parse()
        .expect("Invalid private key format");
    
    // Configure inventory skewing for V2
    // The V2 algorithm integrates this with HJB-based control
    let skew_config = InventorySkewConfig::new(
        0.6,  // inventory_skew_factor: moderate-high aggression (V2 uses this + HJB)
        0.4,  // book_imbalance_factor: react to order book (V2 incorporates into state vector)
        10,   // depth_analysis_levels: analyze top 10 levels for better state estimation
    ).expect("Failed to create skew config");
    
    let market_maker_input = MarketMakerInput {
        asset: "HYPE".to_string(),
        target_liquidity: 0.3,  // Order size: 0.3 per side (allows ~10 fills to reach max position)
        reprice_threshold_ratio: 0.5,  // Reprice when mid moves 50% of current spread (dynamic threshold)
        half_spread: 5,         // DEPRECATED: Ignored - baseline spread now uses real-time market spread
        max_absolute_position_size: 3.0,  // Max position: 3.0
        asset_type: AssetType::Perp, // HYPE is a perpetual
        wallet,
        inventory_skew_config: Some(skew_config),
        enable_trading_gap_threshold_percent: 15.0,  // Enable trading when heuristic is within 15% of optimal
    };
    
    let mut market_maker = MarketMaker::new(market_maker_input).await
        .expect("Failed to create market maker");
    
    info!("=== Market Maker V2 Initialized ===");
    info!("Asset: HYPE");
    info!("Target Liquidity: 0.3 per side");
    info!("Baseline Spread: REAL-TIME MARKET SPREAD (fully market-driven, no arbitrary multipliers)");
    info!("Dynamic Reprice: {:.0}% of current spread (adaptive threshold)", 0.5 * 100.0);
    info!("Max Position: 3.0");
    info!("Trading Enablement: Starts disabled, enables when heuristic gap < {:.1}%", 15.0);
    info!("Features: Market-Driven Spread, Dynamic Reprice, State Vector, Control Vector, HJB Optimization");
    info!("Hot-Reloading: tuning_params.json checked every 10 seconds");
    info!("Adam Optimizer: Autonomous parameter tuning enabled");
    info!("====================================");
    
    // Set up shutdown signal channel
    let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel();
    
    // Set up Ctrl+C handler
    tokio::spawn(async move {
        signal::ctrl_c().await.expect("Failed to install Ctrl+C handler");
        info!("Received Ctrl+C signal, sending shutdown signal...");
        let _ = shutdown_tx.send(());
    });
    
    // Start market maker with shutdown signal
    info!("Starting V2 market maker with HJB-based control...");
    market_maker.start_with_shutdown_signal(Some(shutdown_rx)).await;
    
    info!("Market Maker V2 shutdown complete.");
}
