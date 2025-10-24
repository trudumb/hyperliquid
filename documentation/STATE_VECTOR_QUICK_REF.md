# State Vector Quick Reference

## Structure

```rust
pub struct StateVector {
    pub mid_price: f64,                      // S_t
    pub inventory: f64,                      // Q_t  
    pub adverse_selection_estimate: f64,     // μ̂_t
    pub market_spread_bps: f64,              // Δ_t
    pub lob_imbalance: f64,                  // I_t
}
```

## Components at a Glance

| Symbol | Field | Range | Meaning |
|--------|-------|-------|---------|
| $S_t$ | `mid_price` | $(0, \infty)$ | Current market price |
| $Q_t$ | `inventory` | $(-\infty, \infty)$ | Position (+ long, - short) |
| $\hat{\mu}_t$ | `adverse_selection_estimate` | $[-1, 1]$ | Expected drift (+ bullish, - bearish) |
| $\Delta_t$ | `market_spread_bps` | $[0, \infty)$ | Market spread (higher = less liquid) |
| $I_t$ | `lob_imbalance` | $[0, 1]$ | Bid/Ask balance (0.5 = balanced) |

## Key Methods

### Update State
```rust
state_vector.update(
    mid_price: f64,
    inventory: f64, 
    book_analysis: Option<&BookAnalysis>,
    order_book: Option<&OrderBook>,
);
```

### Decision Support

```rust
// Get spread adjustment for adverse selection
let adjustment_bps = state_vector.get_adverse_selection_adjustment(base_spread_bps);

// Get inventory risk multiplier (1.0 to 2.0)
let risk_multiplier = state_vector.get_inventory_risk_multiplier(max_inventory);

// Get inventory urgency (0.0 to 1.0)
let urgency = state_vector.get_inventory_urgency(max_inventory);

// Check if market is favorable
let is_good = state_vector.is_market_favorable(max_spread_bps);
```

## Interpretation Guide

### Adverse Selection ($\hat{\mu}_t$)

| Value | Meaning | Action |
|-------|---------|--------|
| $> 0.1$ | Strong bullish signal | Widen sell spread, tighten buy spread |
| $0.0 \text{ to } 0.1$ | Mild bullish | Slight asymmetry |
| $\approx 0$ | Neutral | Symmetric spreads |
| $-0.1 \text{ to } 0.0$ | Mild bearish | Slight asymmetry |
| $< -0.1$ | Strong bearish signal | Widen buy spread, tighten sell spread |

### LOB Imbalance ($I_t$)

| Value | Meaning | Interpretation |
|-------|---------|----------------|
| $0.0 \text{ to } 0.3$ | Heavy selling pressure | Price likely to drop |
| $0.3 \text{ to } 0.45$ | Moderate selling | Weak bearish |
| $0.45 \text{ to } 0.55$ | Balanced | Neutral market |
| $0.55 \text{ to } 0.7$ | Moderate buying | Weak bullish |
| $0.7 \text{ to } 1.0$ | Heavy buying pressure | Price likely to rise |

### Market Spread ($\Delta_t$)

| Range (bps) | Condition | Strategy |
|-------------|-----------|----------|
| $0 \text{ to } 5$ | Very tight | Normal spreads |
| $5 \text{ to } 20$ | Normal | Standard operation |
| $20 \text{ to } 50$ | Wide | Increase spreads |
| $> 50$ | Very wide | Consider pausing |

### Inventory Urgency

| Urgency | Inventory Level | Action Priority |
|---------|-----------------|-----------------|
| $0.0 \text{ to } 0.3$ | Low (0-67%) | Normal operation |
| $0.3 \text{ to } 0.7$ | Moderate (67-89%) | Start reducing |
| $0.7 \text{ to } 0.9$ | High (89-97%) | Aggressive reduction |
| $> 0.9$ | Critical (97-100%) | Emergency exit |

## Usage Patterns

### Pattern 1: Basic Monitoring
```rust
// Log current state
info!("{}", market_maker.get_state_vector().to_log_string());

// Output: StateVector[S=100.50, Q=25.5000, μ̂=0.0234, Δ=8.5bps, I=0.623]
```

### Pattern 2: Spread Adjustment
```rust
let base_spread = 10.0; // 10 bps
let state = market_maker.get_state_vector();

// Apply adverse selection adjustment
let adjustment = state.get_adverse_selection_adjustment(base_spread);
let adjusted_spread = base_spread + adjustment;

// Apply risk multiplier
let risk_multiplier = state.get_inventory_risk_multiplier(max_position);
let final_spread = adjusted_spread * risk_multiplier;
```

### Pattern 3: Inventory Management
```rust
let state = market_maker.get_state_vector();
let urgency = state.get_inventory_urgency(max_position);

if urgency > 0.8 {
    // Emergency: Aggressively reduce position
    let offset = max_position * 0.5; // Cross the spread if needed
} else if urgency > 0.5 {
    // High urgency: Tighten spread on exit side
    let offset = base_spread * (1.0 - urgency);
} else {
    // Normal operation
    let offset = base_spread;
}
```

### Pattern 4: Market Condition Check
```rust
let state = market_maker.get_state_vector();

if !state.is_market_favorable(base_spread * 5.0) {
    // Abnormal conditions - pause trading
    info!("Unfavorable market conditions detected");
    return;
}

// Check specific conditions
if state.market_spread_bps > 100.0 {
    info!("Spread too wide: {:.1} bps", state.market_spread_bps);
}

if state.lob_imbalance < 0.1 || state.lob_imbalance > 0.9 {
    info!("Extreme order book imbalance: {:.3}", state.lob_imbalance);
}
```

### Pattern 5: Combined Decision Making
```rust
let state = market_maker.get_state_vector();

// Calculate optimal quotes
let mut buy_spread = base_spread;
let mut sell_spread = base_spread;

// 1. Apply adverse selection
let adverse_adj = state.get_adverse_selection_adjustment(base_spread);
if adverse_adj > 0.0 {
    buy_spread += adverse_adj;  // Widen buy spread
} else {
    sell_spread -= adverse_adj;  // Widen sell spread
}

// 2. Apply inventory risk
let risk_mult = state.get_inventory_risk_multiplier(max_position);
buy_spread *= risk_mult;
sell_spread *= risk_mult;

// 3. Apply inventory bias
let inventory_ratio = state.inventory / max_position;
if inventory_ratio > 0.5 {
    // Long: Make selling easier
    sell_spread *= 0.8;
} else if inventory_ratio < -0.5 {
    // Short: Make buying easier
    buy_spread *= 0.8;
}

// 4. Final quotes
let buy_price = mid_price - buy_spread * mid_price / 10000.0;
let sell_price = mid_price + sell_spread * mid_price / 10000.0;
```

## Configuration

### Tuning Parameters

```rust
// In StateVector::update_adverse_selection()
const LAMBDA: f64 = 0.1;  // Smoothing (0.01 = smooth, 0.5 = reactive)

// In get_adverse_selection_adjustment()  
const ADJUSTMENT_FACTOR: f64 = 0.5;  // Impact (0.0 = none, 1.0 = full)
```

### Recommended Settings by Strategy

| Strategy | Lambda | Adj Factor | Notes |
|----------|--------|------------|-------|
| Conservative | 0.05 | 0.3 | Slow to react, small adjustments |
| Balanced | 0.10 | 0.5 | Default settings |
| Aggressive | 0.20 | 0.8 | Quick reaction, large adjustments |
| Scalping | 0.30 | 0.3 | Very reactive but small adjustments |

## Performance Metrics

### Key Performance Indicators

1. **Adverse Selection Cost**
   - Measure: PnL on filled orders vs. mid price
   - Target: Minimize losses from being picked off

2. **Inventory Management**
   - Measure: Average absolute inventory / max inventory
   - Target: < 30% for good risk management

3. **Spread Efficiency**  
   - Measure: Realized spread / market spread
   - Target: > 1.0 (capture more than market spread)

4. **Fill Rate**
   - Measure: % of time orders are on the book
   - Target: > 95% uptime

### Monitoring Queries

```rust
// Track adverse selection accuracy
let correlation = correlate(
    adverse_selection_estimates,
    actual_price_changes
);
// Target: > 0.3 (some predictive power)

// Monitor inventory statistics
let mean_inventory = inventories.mean();
let max_inventory = inventories.max();
let inventory_ratio = max_inventory / position_limit;
// Target: mean < 0.3, max < 0.8

// Calculate spread capture
let avg_realized_spread = (sells - buys).mean() / mid_price;
let avg_market_spread = market_spreads.mean();
let spread_capture = avg_realized_spread / avg_market_spread;
// Target: > 1.0
```

## Troubleshooting

| Issue | Possible Cause | Solution |
|-------|---------------|----------|
| $\hat{\mu}_t$ always near 0 | Lambda too low | Increase lambda to 0.15-0.2 |
| $\hat{\mu}_t$ too noisy | Lambda too high | Decrease lambda to 0.05-0.08 |
| Inventory keeps growing | Adjustment too weak | Increase adjustment factor |
| Getting picked off | Not reacting to signals | Increase adjustment factor |
| Low fill rate | Spreads too wide | Reduce risk multiplier impact |
| Imbalance not updating | No LOB subscription | Enable inventory skewing config |

## Advanced Topics

### Multi-Asset State Vectors

For portfolios, track correlation:

```rust
struct PortfolioState {
    assets: HashMap<String, StateVector>,
    correlations: HashMap<(String, String), f64>,
}
```

### Machine Learning Integration

Use state vector as feature input:

```rust
let features = vec![
    state.mid_price / 100.0,  // Normalize
    state.inventory / max_inventory,
    state.adverse_selection_estimate,
    state.market_spread_bps / 100.0,
    state.lob_imbalance,
];

let prediction = model.predict(features);
```

### Event Detection

Detect regime changes:

```rust
if state.market_spread_bps > historical_mean + 3.0 * historical_std {
    // Volatility spike detected
    enter_defensive_mode();
}
```
