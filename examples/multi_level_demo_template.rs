//! Quick Start Example: Multi-Level Market Making
//!
//! This example shows how to run the market maker with multi-level quoting enabled.
//! Copy this to examples/multi_level_demo.rs to run.

use hyperliquid_rust_sdk::{
    MarketMakerInputV2, MarketMakerV2, MultiLevelConfig, RobustConfig,
    AssetType,
};
use alloy::signers::local::PrivateKeySigner;
use log::info;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    
    info!("🚀 Starting Multi-Level Market Maker Demo");
    
    // Load wallet from environment variable or create random for testing
    let wallet = if let Ok(key) = std::env::var("WALLET_PRIVATE_KEY") {
        key.parse::<PrivateKeySigner>()?
    } else {
        info!("⚠️  No WALLET_PRIVATE_KEY found, using random wallet (testnet only!)");
        PrivateKeySigner::random()
    };
    
    // Configure multi-level market making
    let multi_level_config = MultiLevelConfig {
        max_levels: 3,                      // 3 levels of quotes
        min_profitable_spread_bps: 4.0,     // 3 bps fees + 1 bps edge
        level_spacing_bps: 2.0,             // 2 bps between levels
        total_size_per_side: 100.0,         // Total size budget
        inventory_risk_limit: 0.7,          // Use 70% of max position
        directional_aggression: 2.0,        // Moderate signal following
        momentum_threshold: 1.3,            // Detect momentum at 1.3x excitation
        momentum_tightening_bps: 1.0,       // Tighten 1 bps on momentum
        inventory_urgency_threshold: 0.8,   // Taker orders at 80% inventory
    };
    
    // Configure robust control (optional)
    let robust_config = RobustConfig {
        enabled: true,                      // Enable worst-case optimization
        robustness_level: 0.7,              // 70% toward worst case
        min_epsilon_mu: 0.2,                // Apply if drift uncertainty > 0.2 bps
        min_epsilon_sigma: 2.0,             // Apply if vol uncertainty > 2 bps
    };
    
    // Create market maker input
    let input = MarketMakerInputV2 {
        asset: "ETH-USD".to_string(),
        max_absolute_position_size: 100.0,
        asset_type: AssetType::Perp,
        wallet,
        enable_trading_gap_threshold_percent: 15.0,
        
        // NEW: Enable multi-level
        enable_multi_level: true,
        multi_level_config: Some(multi_level_config),
        
        // NEW: Enable robust control
        enable_robust_control: true,
        robust_config: Some(robust_config),
    };
    
    // Create and start the market maker
    info!("Creating MarketMaker with multi-level configuration...");
    let mut market_maker = MarketMakerV2::new(input).await?;
    
    info!("✅ MarketMaker initialized successfully");
    info!("📊 Starting market making loop...");
    
    // Start the market maker (runs forever)
    market_maker.start().await;
    
    Ok(())
}

// ============================================================================
// Expected Output
// ============================================================================

/*

🚀 Starting Multi-Level Market Maker Demo
Creating MarketMaker with multi-level configuration...
Initialized with tuning parameters (constrained): ConstrainedTuningParams { ... }
Adam optimizer will now autonomously tune these parameters
✨ Online Adverse Selection Model enabled: Learning weights via SGD
   Features: [bias, trade_flow, lob_imbalance, spread, volatility]
   Lookback: 10 ticks (~10 sec), Learning rate: 0.001
📊 Adaptive Liu-West SV Filter initialized:
   Particles: 7000, delta: 0.99
   Estimating state (h_t) AND parameters (mu, phi, sigma_eta)
🎯 Multi-level market making ENABLED: 3 levels
🛡️  Robust control ENABLED: robustness_level=70.0%
✅ MarketMaker initialized successfully
📊 Starting market making loop...

... market making logs ...

StateVector[S=68245.50, Q=0.0000, μ̂=0.0000, Δ=12.5bps, I=0.512, σ̂=98.34bps, TF_EMA=0.000]
Multi-level reprice triggered
Bid L1 placed: 50.000 @ 68234.25 (4.50 bps)
Bid L2 placed: 27.000 @ 68220.75 (6.50 bps)
Bid L3 placed: 13.500 @ 68207.25 (8.50 bps)
Ask L1 placed: 50.000 @ 68256.75 (4.50 bps)
Ask L2 placed: 27.000 @ 68270.25 (6.50 bps)
Ask L3 placed: 13.500 @ 68283.75 (8.50 bps)

... after some fills ...

🔥 Hawkes momentum detected! Level=0, Side=BID, Excitation=1.65x
Hawkes L1: bid_fills=3, ask_fills=0, bid_excite=1.65x, ask_excite=1.02x
Multi-level reprice triggered
Bid L1 placed: 50.000 @ 68237.00 (3.50 bps)  ← Tightened due to momentum!
...

*/

// ============================================================================
// Alternative Configurations
// ============================================================================

#[allow(dead_code)]
fn conservative_config() -> MultiLevelConfig {
    MultiLevelConfig {
        max_levels: 2,
        min_profitable_spread_bps: 6.0,
        level_spacing_bps: 4.0,
        total_size_per_side: 50.0,
        inventory_risk_limit: 0.5,
        directional_aggression: 1.0,
        momentum_threshold: 1.5,
        momentum_tightening_bps: 0.5,
        inventory_urgency_threshold: 0.9,
    }
}

#[allow(dead_code)]
fn aggressive_config() -> MultiLevelConfig {
    MultiLevelConfig {
        max_levels: 5,
        min_profitable_spread_bps: 4.0,
        level_spacing_bps: 1.5,
        total_size_per_side: 200.0,
        inventory_risk_limit: 0.8,
        directional_aggression: 3.0,
        momentum_threshold: 1.2,
        momentum_tightening_bps: 2.0,
        inventory_urgency_threshold: 0.7,
    }
}

#[allow(dead_code)]
fn single_level_mode() -> MarketMakerInputV2 {
    // To run in single-level mode (backward compatible):
    // Just set enable_multi_level to false
    let wallet = PrivateKeySigner::random();
    
    MarketMakerInputV2 {
        asset: "ETH-USD".to_string(),
        max_absolute_position_size: 100.0,
        asset_type: AssetType::Perp,
        wallet,
        enable_trading_gap_threshold_percent: 15.0,
        
        // Disable multi-level
        enable_multi_level: false,
        multi_level_config: None,
        enable_robust_control: false,
        robust_config: None,
    }
}
