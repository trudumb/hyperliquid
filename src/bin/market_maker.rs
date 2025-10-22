/*This is an example of a basic market making strategy.

We subscribe to the current mid price and build a market around this price. Whenever our market becomes outdated, we place and cancel orders to renew it.
*/
use alloy::signers::local::PrivateKeySigner;
use hyperliquid_rust_sdk::{AssetType, MarketMaker, MarketMakerInput};
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
    let market_maker_input = MarketMakerInput {
        asset: "HYPE".to_string(),
        target_liquidity: 2.0,
        max_bps_diff: 10,
        half_spread: 5,
        max_absolute_position_size: 3.0,
        asset_type: AssetType::Perp, // HYPE is a perpetual
        wallet,
    };
    
    let mut market_maker = MarketMaker::new(market_maker_input).await
        .expect("Failed to create market maker");
    
    // Set up shutdown signal channel
    let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel();
    
    // Set up Ctrl+C handler
    tokio::spawn(async move {
        signal::ctrl_c().await.expect("Failed to install Ctrl+C handler");
        info!("Received Ctrl+C signal, sending shutdown signal...");
        let _ = shutdown_tx.send(());
    });
    
    // Start market maker with shutdown signal
    market_maker.start_with_shutdown_signal(Some(shutdown_rx)).await;
}