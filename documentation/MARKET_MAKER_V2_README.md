# Market Maker V2

This is an advanced market making implementation based on **Hamilton-Jacobi-Bellman (HJB)** optimal control theory.

## Overview

The V2 market maker implements a sophisticated algorithmic trading strategy that:

1. **State Vector (Z_t)** - Tracks all relevant market information:
   - `S_t`: Mid-price
   - `Q_t`: Current inventory position
   - `μ̂_t`: Adverse selection estimate (filtered price drift)
   - `Δ_t`: Market spread in basis points
   - `I_t`: Limit Order Book (LOB) imbalance

2. **Control Vector (u_t)** - Determines trading actions:
   - `δ^a_t`: Ask quote offset (distance from mid)
   - `δ^b_t`: Bid quote offset (distance from mid)
   - `ν^a_t`: Taker sell rate (for emergency liquidation)
   - `ν^b_t`: Taker buy rate (for emergency accumulation)

3. **Value Function V(Q, Z, t)** - Estimates maximum achievable P&L from current state

4. **HJB Optimization** - Continuously optimizes control to maximize expected value

## Key Features

### Adverse Selection Management
The algorithm estimates short-term price drift (`μ̂_t`) using LOB imbalance and adjusts quotes accordingly:
- **Bullish signal** → Widen ask spreads (avoid selling too cheap)
- **Bearish signal** → Widen bid spreads (avoid buying too high)

### Inventory Risk Control
Implements quadratic inventory penalties that:
- Widen spreads as absolute inventory increases
- Skew quotes to encourage mean reversion
- Activate taker orders for emergency liquidation at high inventory urgency

### State-Aware Quote Placement
Quotes adapt to:
- Market microstructure (spread, depth)
- Order book imbalance
- Current inventory position
- Estimated adverse selection

## Usage

### Building
```bash
cargo build --bin market_maker_v2 --release
```

### Running
```bash
# Make sure your .env file has PRIVATE_KEY set
RUST_LOG=info cargo run --bin market_maker_v2
```

### Configuration

Edit `src/bin/market_maker_v2.rs` to configure:

```rust
let skew_config = InventorySkewConfig::new(
    0.6,  // inventory_skew_factor: how aggressively to skew (0.0-1.0)
    0.4,  // book_imbalance_factor: sensitivity to LOB imbalance (0.0-1.0)
    10,   // depth_analysis_levels: how many book levels to analyze
).expect("Failed to create skew config");

let market_maker_input = MarketMakerInput {
    asset: "HYPE".to_string(),              // Asset to trade
    target_liquidity: 2.0,                  // Target size per side
    max_bps_diff: 10,                       // Max deviation before updating orders
    half_spread: 5,                         // Base half-spread in bps
    max_absolute_position_size: 3.0,        // Maximum inventory (abs value)
    asset_type: AssetType::Perp,            // Perp or Spot
    wallet,
    inventory_skew_config: Some(skew_config),
};
```

## Algorithm Details

### State Vector Update
On each market data event (mid price, L2 book):
1. Update mid-price `S_t`
2. Update inventory `Q_t` from fills
3. Calculate market spread `Δ_t` from BBO
4. Calculate LOB imbalance `I_t` from bid/ask volume
5. Filter adverse selection estimate `μ̂_t` using EMA

### Control Calculation
Based on current state:
1. Start with symmetric quotes at `half_spread`
2. Apply adverse selection adjustment
3. Apply inventory risk multiplier (quadratic penalty)
4. Apply inventory skewing (asymmetric quotes)
5. Check urgency and activate taker orders if needed
6. Validate control vector constraints

### Order Management
- Cancel and replace orders when deviation exceeds `max_bps_diff`
- Respect tick/lot size constraints
- Track resting order positions
- Handle partial fills correctly

## Monitoring

The algorithm logs:
- State vector updates: `StateVector[S=..., Q=..., μ̂=..., Δ=...bps, I=...]`
- Control vector decisions: `ControlVector[δ^b=...bps, δ^a=...bps, spread=...bps, asymmetry=...bps]`
- Order placements and cancellations
- Fill events

## Comparison with V1

| Feature | V1 (market_maker.rs) | V2 (market_maker_v2.rs) |
|---------|---------------------|-------------------------|
| Quote Placement | Fixed spread + basic skew | HJB-optimized dynamic control |
| Adverse Selection | Not modeled | Filtered estimate with LOB signals |
| Inventory Management | Linear skewing | Quadratic penalty + urgency-based liquidation |
| State Awareness | Minimal | Full state vector (S, Q, μ̂, Δ, I) |
| Theoretical Foundation | Heuristic | Optimal control theory (HJB equation) |

## Advanced Usage

### Full HJB Optimization

By default, V2 uses fast heuristic adjustments. For full HJB optimization:

```rust
// In the main loop, replace calculate_optimal_control() with:
market_maker.calculate_optimal_control_hjb();
```

This performs grid search over control space to find the true optimum (slower but theoretically optimal).

### Accessing State/Control

```rust
// Get current state
let state = market_maker.get_state_vector();
println!("Inventory: {}, Adverse Selection: {}", 
    state.inventory, state.adverse_selection_estimate);

// Get current control
let control = market_maker.get_control_vector();
println!("Spread: {:.1} bps, Asymmetry: {:.1} bps",
    control.total_spread_bps(), control.spread_asymmetry_bps());

// Get expected fill rates
let (lambda_bid, lambda_ask) = market_maker.get_expected_fill_rates();
println!("Expected fills: bid={:.3}/s, ask={:.3}/s", lambda_bid, lambda_ask);
```

## Safety Features

- Automatic shutdown on Ctrl+C
- Position closing on shutdown
- Order cancellation on exit
- Invalid order cleanup
- Tick/lot size validation

## References

- **HJB Framework**: See `HJB_FRAMEWORK.md` for mathematical details
- **State Vector**: See `STATE_VECTOR.md` for implementation
- **Inventory Skewing**: See `INVENTORY_SKEWING.md` for skewing logic
- **Architecture**: See `ARCHITECTURE_DIAGRAM.md` for system design

## Troubleshooting

### Orders not placing
- Check tick/lot size constraints for your asset
- Verify spread is wide enough (min spread validation)
- Check position limits

### Excessive quote updates
- Increase `max_bps_diff` to reduce order churn
- Adjust `inventory_skew_factor` if too aggressive
- Check if market is very volatile

### Large adverse selection estimates
- Normal during trending markets
- Algorithm will automatically widen spreads
- Monitor LOB imbalance for quality

## License

MIT
