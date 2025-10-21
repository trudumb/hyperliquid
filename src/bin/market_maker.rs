/*
This is an example of a basic market making strategy.

We subscribe to the current mid price and build a market around this price. Whenever our market becomes outdated, we place and cancel orders to renew it.
*/
use alloy::signers::local::PrivateKeySigner;
use hyperliquid_rust_sdk::{MarketMaker, MarketMakerInput};

#[tokio::main]
async fn main() {
    env_logger::init();
    // Key was randomly generated for testing and shouldn't be used with any real funds
    let wallet: PrivateKeySigner =
        "199d2c0ce0857b4d28a7726e0818ad98e1615fc241c622a667a38de7f934f34c"
            .parse()
            .unwrap();
    let market_maker_input = MarketMakerInput {
        asset: "HYPE".to_string(),
        target_liquidity: 25.0,
        max_bps_diff: 2,
        half_spread: 5,
        max_absolute_position_size: 50.0,
        decimals: 4, // Fixed: Changed to 4 decimal places for optimal precision
        wallet,
        // Risk management parameters
        initial_capital: 1000.0,     // $10,000 initial capital
        max_drawdown_pct: 0.05,       // 5% max drawdown
        max_var_95: 500.0,            // $500 VaR limit
        position_value_limit_pct: 0.1, // 10% of capital per position
        max_daily_loss_pct: 0.02,     // 2% daily loss limit
        heat_limit_pct: 0.03,         // 3% portfolio heat limit
        // Adverse selection protection parameters
        adverse_selection_threshold: 0.001, // Threshold for adverse selection (0.1% of average fill size)
        adverse_selection_history_size: 100, // Keep 100 fills in history
        // Volatility parameters
        base_volatility: 0.50,        // 50% expected annual volatility (crypto typical)
        volatility_lookback_minutes: 60, // Look back 1 hour for volatility calculation
        observation_frequency_secs: 10,  // Expect price updates every ~10 seconds
        // Simulation parameters
        simulation_mode: true,        // Start in simulation mode for testing
    };
    MarketMaker::new(market_maker_input).await.start().await
}
