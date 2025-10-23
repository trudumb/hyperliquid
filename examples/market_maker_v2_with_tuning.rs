// Example: Market Maker V2 with Live Tuning Support
// This example demonstrates how to run the market maker with hot-reloadable parameters
//
// ⚠️  WARNING: This is a DEMONSTRATION ONLY - NOT for live trading!
// ⚠️  This example shows the tuning/monitoring features without executing real trades.
// ⚠️  For live trading, use the actual market_maker_v2 binary with proper configuration.

use hyperliquid_rust_sdk::TuningParams;
use log::info;
use std::time::Duration;
use tokio::time::sleep;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    env_logger::init();

    println!("\n╔════════════════════════════════════════════════════════════╗");
    println!("║  Market Maker V2 - Tuning Parameter Demonstration         ║");
    println!("║  ⚠️  EXAMPLE ONLY - NOT FOR LIVE TRADING                  ║");
    println!("╚════════════════════════════════════════════════════════════╝\n");
    
    println!("This example demonstrates:");
    println!("  1. Hot-reloading parameters from tuning_params.json");
    println!("  2. Real-time parameter monitoring");
    println!("  3. Programmatic parameter updates");
    println!("\nTo run with live trading, use: cargo run --bin market_maker_v2\n");

    // For demonstration, we'll just show parameter management without actual trading
    demonstrate_parameter_features().await?;

    Ok(())
}

/// Demonstrate the parameter management features without live trading
async fn demonstrate_parameter_features() -> Result<(), Box<dyn std::error::Error>> {
    info!("═══════════════════════════════════════════════════════════");
    info!("Feature 1: Loading and Validating Parameters");
    info!("═══════════════════════════════════════════════════════════");
    
    // Show default parameters
    let default_params = TuningParams::default();
    info!("Default Parameters:");
    info!("  skew_adjustment_factor: {}", default_params.skew_adjustment_factor);
    info!("  adverse_selection_adjustment_factor: {}", default_params.adverse_selection_adjustment_factor);
    info!("  adverse_selection_lambda: {}", default_params.adverse_selection_lambda);
    info!("  inventory_urgency_threshold: {}", default_params.inventory_urgency_threshold);
    info!("  liquidation_rate_multiplier: {}", default_params.liquidation_rate_multiplier);
    info!("  min_spread_base_ratio: {}", default_params.min_spread_base_ratio);
    
    sleep(Duration::from_secs(2)).await;
    
    // Try loading from file if it exists
    info!("\n═══════════════════════════════════════════════════════════");
    info!("Feature 2: Loading from Config File");
    info!("═══════════════════════════════════════════════════════════");
    
    let config_path = "tuning_params.json";
    match TuningParams::from_json_file(config_path) {
        Ok(params) => {
            info!("✅ Successfully loaded parameters from {}", config_path);
            info!("Loaded Parameters:");
            info!("  skew_adjustment_factor: {}", params.skew_adjustment_factor);
            info!("  adverse_selection_adjustment_factor: {}", params.adverse_selection_adjustment_factor);
            info!("  adverse_selection_lambda: {}", params.adverse_selection_lambda);
        }
        Err(e) => {
            info!("ℹ️  No config file found ({})", e);
            info!("You can create tuning_params.json to test hot-reloading");
            info!("Example file structure is in tuning_params.example.json");
        }
    }
    
    sleep(Duration::from_secs(2)).await;
    
    // Show validation
    info!("\n═══════════════════════════════════════════════════════════");
    info!("Feature 3: Parameter Validation");
    info!("═══════════════════════════════════════════════════════════");
    
    let mut invalid_params = TuningParams::default();
    invalid_params.skew_adjustment_factor = -0.5; // Invalid!
    
    match invalid_params.validate() {
        Ok(_) => info!("Parameters valid"),
        Err(e) => info!("✅ Validation correctly caught invalid parameter: {}", e),
    }
    
    sleep(Duration::from_secs(2)).await;
    
    // Show programmatic updates
    info!("\n═══════════════════════════════════════════════════════════");
    info!("Feature 4: Programmatic Parameter Updates");
    info!("═══════════════════════════════════════════════════════════");
    
    let mut aggressive_params = TuningParams::default();
    aggressive_params.skew_adjustment_factor = 0.8;
    aggressive_params.inventory_urgency_threshold = 0.6;
    aggressive_params.liquidation_rate_multiplier = 15.0;
    
    info!("Example: Aggressive inventory management parameters:");
    info!("  skew_adjustment_factor: {} (higher = more aggressive skewing)", 
          aggressive_params.skew_adjustment_factor);
    info!("  inventory_urgency_threshold: {} (lower = liquidate earlier)", 
          aggressive_params.inventory_urgency_threshold);
    info!("  liquidation_rate_multiplier: {} (higher = faster liquidation)", 
          aggressive_params.liquidation_rate_multiplier);
    
    sleep(Duration::from_secs(2)).await;
    
    let mut conservative_params = TuningParams::default();
    conservative_params.min_spread_base_ratio = 0.35;
    conservative_params.adverse_selection_lambda = 0.15;
    
    info!("\nExample: Conservative market making parameters:");
    info!("  min_spread_base_ratio: {} (higher = wider spreads)", 
          conservative_params.min_spread_base_ratio);
    info!("  adverse_selection_lambda: {} (higher = faster reaction)", 
          conservative_params.adverse_selection_lambda);
    
    sleep(Duration::from_secs(2)).await;
    
    // Show adaptive tuning scenarios
    info!("\n═══════════════════════════════════════════════════════════");
    info!("Feature 5: Adaptive Tuning Scenarios");
    info!("═══════════════════════════════════════════════════════════");
    
    demonstrate_volatility_adaptation().await;
    
    info!("\n═══════════════════════════════════════════════════════════");
    info!("Demonstration Complete!");
    info!("═══════════════════════════════════════════════════════════");
    info!("\nKey Takeaways:");
    info!("  ✓ Parameters can be loaded from JSON files");
    info!("  ✓ All parameters are validated before application");
    info!("  ✓ Parameters can be updated programmatically");
    info!("  ✓ Live trading bot checks tuning_params.json every 10 seconds");
    info!("  ✓ Changes take effect on the next market data update");
    info!("\nFor live trading: cargo run --bin market_maker_v2");
    
    Ok(())
}

/// Demonstrate adaptive parameter tuning based on market conditions
async fn demonstrate_volatility_adaptation() {
    info!("Scenario: Adapting to different volatility regimes\n");
    
    // High volatility scenario
    let high_vol = 0.025;
    info!("  Market Volatility: {:.3} (HIGH)", high_vol);
    info!("  → Recommended parameters:");
    info!("      adverse_selection_lambda: 0.15 (react faster to adverse selection)");
    info!("      min_spread_base_ratio: 0.30 (wider spreads for safety)");
    info!("      inventory_urgency_threshold: 0.60 (liquidate inventory earlier)");
    
    sleep(Duration::from_secs(2)).await;
    
    // Low volatility scenario
    let low_vol = 0.008;
    info!("\n  Market Volatility: {:.3} (LOW)", low_vol);
    info!("  → Recommended parameters:");
    info!("      adverse_selection_lambda: 0.08 (smoother adjustments)");
    info!("      min_spread_base_ratio: 0.15 (tighter spreads, more competitive)");
    info!("      inventory_urgency_threshold: 0.75 (hold positions longer)");
}
