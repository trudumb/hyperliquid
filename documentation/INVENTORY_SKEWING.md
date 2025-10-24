# Inventory Skewing and L2 Book Integration

This document describes the new inventory skewing and L2 book integration features added to the market maker.

## Overview

The market maker now includes sophisticated inventory management through:
1. **L2 Book Integration** - Real-time order book analysis
2. **Inventory Skewing** - Automatic quote adjustment based on position and market conditions

## Features

### 1. L2 Book Integration

The market maker subscribes to real-time order book data and maintains:
- Sorted bid/ask levels
- Depth-weighted prices
- Book imbalance metrics

**Key Components:**
- `OrderBook` struct: Parses and stores L2 book data
- `BookAnalysis` struct: Contains analyzed book metrics
- Depth calculation across configurable number of levels
- Real-time imbalance tracking

### 2. Inventory Skewing

The market maker automatically adjusts quotes based on:

#### Position Signal
- **When long**: Shifts quotes DOWN to make selling easier
- **When short**: Shifts quotes UP to make buying easier
- Magnitude scales with position ratio (current/max)

#### Book Imbalance Signal
- **More bid liquidity**: Shifts quotes UP (encourage selling into demand)
- **More ask liquidity**: Shifts quotes DOWN (encourage buying from supply)

### 3. Smart Quote Adjustment

**Example Scenario:**
```
Position: Long 3 HYPE (max: 5 HYPE)
Position ratio: 60%
Base half spread: 10 bps

Calculations:
- Position component: -0.6 × 0.5 × 10 = -3.0 bps (shift down)
- Book imbalance: +0.2 (more bids)
- Book component: +0.2 × 0.3 × 10 = +0.6 bps (shift up)
- Total skew: -2.4 bps

Result:
- Both bid and ask shift down by ~2.4 bps
- Makes selling easier (to reduce long position)
- Maintains spread integrity
```

## Configuration

### New Parameters

```rust
pub struct InventorySkewConfig {
    /// How aggressively to manage inventory (0.0 to 1.0)
    /// Higher = more aggressive position management
    pub inventory_skew_factor: f64,
    
    /// How much to react to order book imbalance (0.0 to 1.0)
    /// Higher = more reaction to book conditions
    pub book_imbalance_factor: f64,
    
    /// Number of book levels to analyze for depth
    pub depth_analysis_levels: usize,
}
```

**Recommended Settings:**
- `inventory_skew_factor`: 0.5 (moderate)
- `book_imbalance_factor`: 0.3 (conservative)
- `depth_analysis_levels`: 5 (top 5 levels)

### MarketMakerInput

```rust
pub struct MarketMakerInput {
    // ... existing fields ...
    pub inventory_skew_config: Option<InventorySkewConfig>,
}
```

Set to `None` for traditional market making without skewing.

## Usage

### Basic Market Maker (No Skewing)

```rust
use hyperliquid_rust_sdk::{MarketMaker, MarketMakerInput, AssetType};

let input = MarketMakerInput {
    asset: "HYPE".to_string(),
    target_liquidity: 10.0,
    half_spread: 15,
    max_bps_diff: 10,
    max_absolute_position_size: 50.0,
    asset_type: AssetType::Perp,
    wallet: your_wallet,
    inventory_skew_config: None, // Traditional market making
};

let mut mm = MarketMaker::new(input).await?;
mm.start().await;
```

### Market Maker with Inventory Skewing

```rust
use hyperliquid_rust_sdk::{
    MarketMaker, MarketMakerInput, AssetType, InventorySkewConfig
};

let skew_config = InventorySkewConfig::new(
    0.5,  // inventory_skew_factor
    0.3,  // book_imbalance_factor
    5,    // depth_analysis_levels
)?;

let input = MarketMakerInput {
    asset: "HYPE".to_string(),
    target_liquidity: 10.0,
    half_spread: 15,
    max_bps_diff: 10,
    max_absolute_position_size: 50.0,
    asset_type: AssetType::Perp,
    wallet: your_wallet,
    inventory_skew_config: Some(skew_config), // Enable skewing
};

let mut mm = MarketMaker::new(input).await?;
mm.start().await;
```

### Running Examples

```bash
# Traditional market maker
RUST_LOG=info cargo run --bin market_maker

# Market maker with inventory skewing
RUST_LOG=info cargo run --bin market_maker_with_skew
```

## How It Works

### Skew Calculation

The total skew is the sum of two components:

#### 1. Position Component
```
position_component = -(position / max_position) × inventory_skew_factor × base_half_spread
```

**Why negative?**
- When long (+position), we want to shift DOWN to encourage selling
- When short (-position), we want to shift UP to encourage buying
- The negation ensures this behavior

#### 2. Book Component
```
book_component = book_imbalance × book_imbalance_factor × base_half_spread
```

**Imbalance calculation:**
```
imbalance = (bid_depth - ask_depth) / (bid_depth + ask_depth)
```
Range: -1 (all asks) to +1 (all bids)

#### 3. Total Skew
```
total_skew_bps = position_component_bps + book_component_bps
```

Applied to both bid and ask prices:
```
skewed_price = base_price × (1 + total_skew_bps / 10000)
```

## Benefits

### 1. Risk Management
- Automatically prevents dangerous one-sided positions
- Dynamically adjusts to market conditions
- Reduces exposure to adverse selection

### 2. Market Adaptive
- Responds to real-time order book conditions
- Takes advantage of temporary imbalances
- Optimizes fill rates on both sides

### 3. Better Fills
- Get filled more on the profitable side
- Natural inventory rebalancing
- Reduced need for manual intervention

### 4. Smoother PnL
- Reduces volatility in returns
- Better risk-adjusted performance
- More consistent profitability

## Architecture

The implementation is modular with separate concerns:

### `book_analyzer.rs`
- Parses L2 book data
- Calculates depth-weighted prices
- Computes book imbalance metrics
- Provides book statistics

### `inventory_skew.rs`
- Implements skew calculation logic
- Manages configuration
- Provides price adjustment utilities
- Includes comprehensive tests

### `market_maker.rs`
- Integrates book analysis and skewing
- Manages order placement/cancellation
- Handles real-time updates
- Coordinates all components

## Testing

Both modules include comprehensive unit tests:

```bash
# Run all tests
cargo test

# Run specific module tests
cargo test --lib book_analyzer
cargo test --lib inventory_skew
```

## Logging

The market maker provides detailed logging:

```
INFO  Subscribing to L2 book data for inventory skewing
INFO  Book Stats - Mid: 25.50, Bid Depth: 150.25, Ask Depth: 120.30, Imbalance: 0.111, Spread: 9.8 bps
INFO  Inventory Skew - Total: -2.40 bps (Position: -3.00 bps [60.0%], Book: 0.60 bps)
INFO  Buy for 10.0 HYPE resting at 25.48
INFO  Sell for 10.0 HYPE resting at 25.52
```

## Performance Considerations

1. **Book Updates**: L2 book updates are frequent. The implementation efficiently updates state without blocking order placement.

2. **Analysis Depth**: More levels provide better analysis but with marginal diminishing returns. 5 levels is a good balance.

3. **Update Frequency**: The market maker only recomputes prices when significant changes occur (respecting `max_bps_diff`).

## Future Enhancements

Potential improvements:
- [ ] Adaptive skew factors based on market volatility
- [ ] Time-weighted book analysis
- [ ] Multiple asset correlation
- [ ] Machine learning-based skew optimization
- [ ] Backtesting framework

## References

- Order book analysis: Depth-weighted pricing is standard in market microstructure literature
- Inventory management: Based on Avellaneda-Stoikov optimal market making framework
- Risk management: Inspired by modern HFT inventory control techniques
