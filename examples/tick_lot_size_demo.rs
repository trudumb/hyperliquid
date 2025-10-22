/*
This example demonstrates the tick and lot size validation functionality.

It shows how to:
1. Create a TickLotValidator for different asset types
2. Validate prices and sizes according to Hyperliquid's rules
3. Round prices and sizes to be valid
4. Handle common validation scenarios
*/
use hyperliquid_rust_sdk::{AssetType, InfoClient, TickLotValidator, BaseUrl};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    
    // Create an info client to fetch real asset metadata
    let info_client = InfoClient::new(None, Some(BaseUrl::Mainnet)).await?;
    
    // Fetch metadata for perps
    let meta = info_client.meta().await?;
    println!("Available perpetual assets:");
    for asset in &meta.universe {
        println!("  {}: sz_decimals = {}", asset.name, asset.sz_decimals);
    }
    
    // Example 1: ETH perpetual (assuming sz_decimals = 1)
    println!("\n=== ETH Perpetual Example ===");
    let eth_validator = TickLotValidator::new("ETH".to_string(), AssetType::Perp, 1);
    
    // Test valid prices
    let valid_prices = vec![1234.5, 0.001234, 123456.0];
    println!("Valid prices:");
    for price in valid_prices {
        match eth_validator.validate_price(price) {
            Ok(_) => println!("  ✓ {} is valid", price),
            Err(e) => println!("  ✗ {} is invalid: {}", price, e),
        }
    }
    
    // Test invalid prices
    let invalid_prices = vec![1234.56, 0.0012345, 0.012345];
    println!("Invalid prices:");
    for price in invalid_prices {
        match eth_validator.validate_price(price) {
            Ok(_) => println!("  ✓ {} is valid", price),
            Err(e) => println!("  ✗ {} is invalid: {}", price, e),
        }
    }
    
    // Test price rounding
    println!("Price rounding:");
    let test_prices = vec![1234.56, 0.0012345, 1.123456];
    for price in test_prices {
        let rounded_down = eth_validator.round_price(price, false);
        let rounded_up = eth_validator.round_price(price, true);
        println!("  {} -> {} (down), {} (up)", price, rounded_down, rounded_up);
    }
    
    // Example 2: Size validation with sz_decimals = 3
    println!("\n=== Size Validation Example (sz_decimals = 3) ===");
    let size_validator = TickLotValidator::new("TEST".to_string(), AssetType::Perp, 3);
    
    let test_sizes = vec![1.001, 1.0001, 10.123, 0.12345];
    println!("Size validation:");
    for size in test_sizes {
        match size_validator.validate_size(size) {
            Ok(_) => println!("  ✓ {} is valid", size),
            Err(e) => println!("  ✗ {} is invalid: {}", size, e),
        }
        
        let rounded = size_validator.round_size(size, false);
        println!("    Rounded: {} -> {}", size, rounded);
    }
    
    // Example 3: Spot asset (if available)
    println!("\n=== Spot Asset Example ===");
    let spot_meta = info_client.spot_meta().await?;
    if !spot_meta.universe.is_empty() {
        println!("Available spot assets:");
        for asset in &spot_meta.universe[..5.min(spot_meta.universe.len())] {
            println!("  {}: index = {}", asset.name, asset.index);
        }
        
        // Create validator for a spot asset (using default sz_decimals for now)
        let spot_validator = TickLotValidator::new("BTC/USDC".to_string(), AssetType::Spot, 2);
        
        println!("Spot price validation (sz_decimals = 2, max_decimals = 8-2 = 6):");
        let spot_prices = vec![50000.123456, 50000.1234567, 0.001234];
        for price in spot_prices {
            match spot_validator.validate_price(price) {
                Ok(_) => println!("  ✓ {} is valid", price),
                Err(e) => println!("  ✗ {} is invalid: {}", price, e),
            }
        }
    }
    
    // Example 4: Real asset from metadata
    if let Some(first_asset) = meta.universe.first() {
        println!("\n=== Real Asset Example: {} ===", first_asset.name);
        let validator = TickLotValidator::from_asset_meta(first_asset, AssetType::Perp);
        
        println!("Asset: {}", validator.asset);
        println!("sz_decimals: {}", validator.sz_decimals);
        println!("Max price decimals: {}", validator.max_price_decimals());
        
        // Test with some realistic values
        let test_values = vec![
            (1000.0, 1.0),
            (1234.56789, 0.123456),
            (0.001234, 10.000001),
        ];
        
        for (price, size) in test_values {
            println!("\nTesting price: {}, size: {}", price, size);
            
            match validator.validate_price(price) {
                Ok(_) => println!("  Price valid ✓"),
                Err(e) => {
                    println!("  Price invalid ✗: {}", e);
                    let rounded = validator.round_price(price, false);
                    println!("  Rounded price: {}", rounded);
                }
            }
            
            match validator.validate_size(size) {
                Ok(_) => println!("  Size valid ✓"),
                Err(e) => {
                    println!("  Size invalid ✗: {}", e);
                    let rounded = validator.round_size(size, false);
                    println!("  Rounded size: {}", rounded);
                }
            }
        }
    }
    
    Ok(())
}