/*This is an advanced market making strategy using the V2 implementation.

The V2 market maker uses:
- State Vector (Z_t): Tracks mid-price, inventory, adverse selection, market spread, and LOB imbalance
- Control Vector (u_t): Determines bid/ask offsets and taker order rates
- Multi-Level Optimizer: Places multiple orders at different price levels
- Hawkes Fill Model: Self-exciting process for fill rate estimation
- Particle Filter: Stochastic volatility estimation
- Robust HJB: Hamilton-Jacobi-Bellman optimization with uncertainty bounds
- Adam Optimizer: Autonomous parameter tuning

The algorithm continuously monitors market state and adjusts quotes to maximize expected P&L
while managing inventory risk through state-aware control policies.
*/
use alloy::signers::local::PrivateKeySigner;
use hyperliquid_rust_sdk::AssetType;
use hyperliquid_rust_sdk::market_maker_v2::{MarketMaker, MarketMakerInput};
use hyperliquid_rust_sdk::{MultiLevelConfig, RobustConfig};
use hyperliquid_rust_sdk::tui::run_tui;
use std::env;
use tokio::signal;
use log::info;
use tracing_subscriber::{fmt, layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

#[tokio::main]
async fn main() {
    // Create a file appender for JSON logs
    let file_appender = tracing_appender::rolling::never("./", "market_maker.log");
    let (non_blocking_writer, _guard) = tracing_appender::non_blocking(file_appender);

    // Create file layer (JSON format) - NO CONSOLE OUTPUT (using TUI instead)
    let file_layer = fmt::layer()
        .json()
        .with_writer(non_blocking_writer);

    // Get log level filter from RUST_LOG environment variable
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info"));

    // Initialize global logger with ONLY file output (TUI replaces console)
    tracing_subscriber::registry()
        .with(filter)
        .with(file_layer)
        .init();

    // Load environment variables from .env file
    dotenv::dotenv().ok();
    
    // Load private key from environment variable
    let private_key = env::var("PRIVATE_KEY")
        .expect("PRIVATE_KEY environment variable must be set");
    
    let wallet: PrivateKeySigner = private_key
        .parse()
        .expect("Invalid private key format");
    
    // Configure multi-level market making
    let multi_level_config = MultiLevelConfig {
        max_levels: 3,  // Place up to 3 levels on each side
        min_profitable_spread_bps: 4.0,  // Minimum 4 bps total spread to cover fees + edge
        level_spacing_bps: 1.5,  // Space levels 1.5 bps apart
        total_size_per_side: 1.0,  // Total size across all levels: 1.0 HYPE per side
        ..Default::default()  // Use defaults for other parameters
    };
    
    // Configure robust control (uncertainty handling)
    let robust_config = RobustConfig {
        enabled: true,  // Enable robust control
        robustness_level: 0.7,  // 70% robustness (0.0 = no robustness, 1.0 = max robustness)
        ..Default::default()  // Use defaults for other parameters
    };
    
    let market_maker_input = MarketMakerInput {
        asset: "HYPE".to_string(),
        max_absolute_position_size: 50.0,  // Max position: 50 HYPE
        asset_type: AssetType::Perp, // HYPE is a perpetual
        wallet,
        enable_trading_gap_threshold_percent: 30.0,  // Enable trading when performance gap < 30%
        
        // IMPORTANT: Multi-level and robust control configuration
        enable_multi_level: true,  // Enable multi-level market making
        multi_level_config: Some(multi_level_config),  // Provide configuration
        enable_robust_control: true,  // Enable robust control with uncertainty bounds
        robust_config: Some(robust_config),  // Provide configuration
    };
    
    let (mut market_maker, tui_state_rx) = MarketMaker::new(market_maker_input).await
        .expect("Failed to create market maker");

    // Log initialization to file only
    info!("=== Market Maker V2 Initialized ===");
    info!("Asset: HYPE-USD");
    info!("Max Position: 50.0 HYPE");
    info!("Multi-Level: ENABLED (3 levels per side)");
    info!("  - Total size per side: 1.0 HYPE");
    info!("  - Level spacing: 1.5 bps");
    info!("  - Min profitable spread: 4.0 bps");
    info!("Robust Control: ENABLED (70% robustness level)");
    info!("Trading Enablement: Starts disabled, enables when performance gap < 30%");
    info!("Features:");
    info!("  - State Vector with adverse selection estimation");
    info!("  - Multi-Level Optimizer with Hawkes fill model");
    info!("  - Particle Filter for stochastic volatility");
    info!("  - Robust HJB with uncertainty bounds");
    info!("  - Adam Optimizer for autonomous parameter tuning");
    info!("  - TUI Dashboard for real-time monitoring");
    info!("====================================");

    // Spawn TUI dashboard task
    let tui_handle = tokio::spawn(async move {
        if let Err(e) = run_tui(tui_state_rx).await {
            eprintln!("TUI error: {}", e);
        }
    });

    // Set up shutdown signal channel
    let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel();

    // Set up Ctrl+C handler (TUI handles this too with 'q' key)
    tokio::spawn(async move {
        signal::ctrl_c().await.expect("Failed to install Ctrl+C handler");
        info!("Received Ctrl+C signal, sending shutdown signal...");
        let _ = shutdown_tx.send(());
    });

    // Start market maker with shutdown signal
    info!("Starting V2 market maker with multi-level optimization and TUI dashboard...");
    market_maker.start_with_shutdown_signal(Some(shutdown_rx)).await;

    // Wait for TUI to finish
    let _ = tui_handle.await;

    info!("Market Maker V2 shutdown complete.");
}
