# Control Vector Implementation for Market Making

## Overview

This document describes the implementation of the **Control Vector ($\mathbf{u}_t$)** for algorithmic market making in the Hyperliquid Rust SDK. The control vector represents all the actions (levers) the algorithm can pull at any instant to optimize its trading strategy.

## Mathematical Definition

$$\mathbf{u}_t = (\delta^a_t, \delta^b_t, \nu^a_t, \nu^b_t)$$

### Components

#### 1. $\delta^a_t$ - Ask Quote Offset
**Field:** `ask_offset_bps: f64`

The distance from mid price $S_t$ to place our passive ask (sell) order, measured in basis points.

$$P^a_t = S_t \cdot (1 + \delta^a_t / 10000)$$

**Purpose:** Controls where we offer liquidity on the sell side.

**Typical Range:** 5-50 bps (0.05% - 0.5%)

#### 2. $\delta^b_t$ - Bid Quote Offset
**Field:** `bid_offset_bps: f64`

The distance from mid price $S_t$ to place our passive bid (buy) order, measured in basis points.

$$P^b_t = S_t \cdot (1 - \delta^b_t / 10000)$$

**Purpose:** Controls where we offer liquidity on the buy side.

**Typical Range:** 5-50 bps (0.05% - 0.5%)

#### 3. $\nu^a_t$ - Taker Sell Rate
**Field:** `taker_sell_rate: f64`

The rate at which we send aggressive (taker) sell orders to cross the spread, measured in units per second.

**Purpose:** Active inventory liquidation when we're long and need to reduce position urgently.

**Range:** [0, ∞), typically 0-5 units/second

**When Active:**
- Inventory urgency > 0.7
- Position is long (Q_t > 0)
- Need emergency exit

#### 4. $\nu^b_t$ - Taker Buy Rate
**Field:** `taker_buy_rate: f64`

The rate at which we send aggressive (taker) buy orders to cross the spread, measured in units per second.

**Purpose:** Active inventory accumulation when we're short and need to cover position urgently.

**Range:** [0, ∞), typically 0-5 units/second

**When Active:**
- Inventory urgency > 0.7
- Position is short (Q_t < 0)
- Need emergency cover

## Implementation Details

### ControlVector Structure

```rust
pub struct ControlVector {
    pub ask_offset_bps: f64,      // δ^a_t
    pub bid_offset_bps: f64,      // δ^b_t
    pub taker_sell_rate: f64,     // ν^a_t
    pub taker_buy_rate: f64,      // ν^b_t
}
```

### Constructor Methods

#### `new()`
Creates a zero-initialized control vector.

```rust
let control = ControlVector::new();
// All components = 0.0
```

#### `symmetric(half_spread_bps: f64)`
Creates a symmetric market making control (equal offsets, no taker activity).

```rust
let control = ControlVector::symmetric(10.0);
// δ^a_t = δ^b_t = 10 bps
// ν^a_t = ν^b_t = 0
```

**Use Case:** Normal market making with balanced quotes.

#### `asymmetric(ask_offset_bps, bid_offset_bps)`
Creates an asymmetric control with different buy/sell offsets.

```rust
let control = ControlVector::asymmetric(15.0, 10.0);
// δ^a_t = 15 bps (wider ask)
// δ^b_t = 10 bps (tighter bid)
// Bullish bias - encourage buying
```

**Use Case:** Directional bias or inventory management.

#### `with_taker_activity()`
Creates a control with active liquidation.

```rust
let control = ControlVector::with_taker_activity(
    20.0,  // ask_offset_bps
    20.0,  // bid_offset_bps  
    2.0,   // taker_sell_rate
    0.0    // taker_buy_rate
);
// Active selling at 2 units/sec
```

**Use Case:** Emergency inventory reduction.

### Key Methods

#### `calculate_quote_prices(mid_price: f64) -> (f64, f64)`

Converts offsets to actual quote prices.

```rust
let (bid_price, ask_price) = control.calculate_quote_prices(100.0);
// Returns actual prices to quote
```

**Formula:**
```
bid_price = mid_price * (1 - δ^b_t / 10000)
ask_price = mid_price * (1 + δ^a_t / 10000)
```

#### `total_spread_bps() -> f64`

Returns the full spread width.

$$\text{spread} = \delta^a_t + \delta^b_t$$

```rust
let spread = control.total_spread_bps();
// 20 bps if both offsets are 10 bps
```

#### `spread_asymmetry_bps() -> f64`

Returns the spread asymmetry (directional bias).

$$\text{asymmetry} = \delta^a_t - \delta^b_t$$

**Interpretation:**
- Positive: Ask wider than bid (bullish bias)
- Negative: Bid wider than ask (bearish bias)
- Zero: Symmetric quotes

```rust
let asymmetry = control.spread_asymmetry_bps();
// +5 bps = bullish, -5 bps = bearish
```

#### `is_passive_only() -> bool`

Checks if this is a pure market making strategy.

```rust
if control.is_passive_only() {
    println!("Passive-only market making");
}
// true if ν^a_t = ν^b_t = 0
```

#### `is_liquidating() -> bool`

Checks if we're actively managing inventory via taker orders.

```rust
if control.is_liquidating() {
    println!("Active liquidation in progress");
}
// true if ν^a_t > 0 or ν^b_t > 0
```

#### `net_taker_rate() -> f64`

Returns net taker order flow direction.

$$\text{net} = \nu^a_t - \nu^b_t$$

**Interpretation:**
- Positive: Net selling (reducing long position)
- Negative: Net buying (covering short position)
- Zero: No taker activity or balanced

#### `validate(min_spread_bps: f64) -> Result<(), String>`

Validates that the control vector is feasible.

**Checks:**
1. Offsets are non-negative
2. Total spread meets minimum
3. Taker rates are non-negative

```rust
match control.validate(5.0) {
    Ok(_) => println!("Valid control"),
    Err(e) => eprintln!("Invalid: {}", e),
}
```

#### `apply_state_adjustments()`

**THIS IS THE KEY METHOD** - Optimally adjusts control based on state vector.

```rust
control.apply_state_adjustments(
    &state_vector,
    base_spread_bps: f64,
    max_inventory: f64,
);
```

**Adjustment Algorithm:**

1. **Adverse Selection Adjustment**
   ```
   if μ̂_t > 0:  // Bullish signal
       δ^a_t += |adjustment|  // Widen ask
   else:          // Bearish signal
       δ^b_t += |adjustment|  // Widen bid
   ```

2. **Inventory Risk Adjustment**
   ```
   risk_multiplier = 1 + (|Q_t|/Q_max)²
   δ^a_t *= risk_multiplier
   δ^b_t *= risk_multiplier
   ```

3. **Inventory Skew Adjustment**
   ```
   skew = (Q_t / Q_max) * base_spread * 0.5
   δ^a_t -= skew  // Long -> tighter ask
   δ^b_t += skew  // Long -> wider bid
   ```

4. **Active Liquidation Control**
   ```
   if urgency > 0.7:
       if Q_t > 0:  // Long position
           ν^a_t = (urgency - 0.7) * 10
       else:        // Short position
           ν^b_t = (urgency - 0.7) * 10
   ```

## Integration with State Vector

The control vector is the **action** that results from the **state** observation.

```
State Vector (Z_t)  ──▶  Decision Logic  ──▶  Control Vector (u_t)
     │                                              │
     │ Current State                                │ Optimal Action
     │                                              │
     ▼                                              ▼
(S_t, Q_t, μ̂_t, Δ_t, I_t)                  (δ^a_t, δ^b_t, ν^a_t, ν^b_t)
```

### State-Control Mapping

| State Condition | Control Response |
|----------------|------------------|
| $\hat{\mu}_t > 0$ (bullish) | Widen $\delta^a_t$, tighten $\delta^b_t$ |
| $\hat{\mu}_t < 0$ (bearish) | Widen $\delta^b_t$, tighten $\delta^a_t$ |
| $|Q_t| \to Q_{\max}$ (high inventory) | Increase both $\delta^a_t, \delta^b_t$ |
| $Q_t > 0$ (long position) | Decrease $\delta^a_t$, increase $\delta^b_t$ |
| $Q_t < 0$ (short position) | Increase $\delta^a_t$, decrease $\delta^b_t$ |
| Urgency > 0.7 & $Q_t > 0$ | Activate $\nu^a_t$ (sell taker) |
| Urgency > 0.7 & $Q_t < 0$ | Activate $\nu^b_t$ (buy taker) |

## Control Regimes

### 1. Normal Market Making
**Characteristics:**
- Symmetric or mildly asymmetric quotes
- Passive-only ($\nu^a_t = \nu^b_t = 0$)
- Tight spreads (10-20 bps)

**Example:**
```rust
let control = ControlVector::symmetric(10.0);
// δ^a_t = δ^b_t = 10 bps
// ν^a_t = ν^b_t = 0
```

### 2. Directional Bias
**Characteristics:**
- Asymmetric quotes based on signal
- Still passive-only
- Spread asymmetry reflects edge

**Example:**
```rust
let mut control = ControlVector::symmetric(10.0);
// Bullish signal detected
control.ask_offset_bps = 15.0;  // Widen ask
control.bid_offset_bps = 8.0;   // Tighten bid
// Asymmetry = +7 bps (bullish)
```

### 3. Risk Management
**Characteristics:**
- Widened spreads on both sides
- Proportional to inventory risk
- Still passive-only

**Example:**
```rust
// High inventory (80% of max)
let control = ControlVector::symmetric(10.0);
// After risk adjustment: both ~16 bps (1.6x multiplier)
```

### 4. Active Liquidation
**Characteristics:**
- Taker orders active
- Aggressive position reduction
- Triggered by high urgency

**Example:**
```rust
let control = ControlVector::with_taker_activity(
    20.0,  // Wider spreads for safety
    20.0,
    2.5,   // Aggressively selling
    0.0
);
// ν^a_t = 2.5 units/sec
```

## Optimal Control Theory

### The Optimization Problem

The market maker solves:

$$\max_{\mathbf{u}_t} \mathbb{E}\left[\int_0^T \left(\text{PnL}(\mathbf{u}_t, \mathbf{Z}_t) - \gamma Q_t^2\right) dt\right]$$

Subject to:
- $\delta^a_t, \delta^b_t \geq \delta_{\min}$ (minimum spread)
- $\nu^a_t, \nu^b_t \geq 0$ (no negative flow)
- $|Q_t| \leq Q_{\max}$ (position limits)

Where:
- $\gamma$ is inventory aversion parameter
- PnL depends on fill rates, spreads captured
- $Q_t$ evolves based on fills and taker orders

### Hamilton-Jacobi-Bellman Equation

The value function satisfies:

$$0 = \max_{\mathbf{u}} \left\{\mathcal{L}V + f(\mathbf{u}, \mathbf{Z})\right\}$$

Where:
- $\mathcal{L}$ is the infinitesimal generator
- $f$ is the instantaneous reward
- $V(\mathbf{Z}, t)$ is the value function

### Approximate Solution

Our implementation uses a **heuristic approximation**:

1. **Passive Control ($\delta^a_t, \delta^b_t$):**
   - Base on market spread and volatility
   - Adjust for adverse selection (tilt)
   - Scale for inventory risk (widen)
   - Skew for inventory bias (asymmetry)

2. **Active Control ($\nu^a_t, \nu^b_t$):**
   - Activate only when urgency > threshold
   - Rate proportional to urgency excess
   - Direction determined by position sign

## Performance Metrics

### Quote Quality Metrics

**1. Realized Spread**
```
realized_spread = (sell_price - buy_price) / mid_price * 10000
```
Target: > total_spread_bps (capturing full spread)

**2. Spread Efficiency**
```
efficiency = realized_spread / market_spread
```
Target: > 1.0 (better than market)

**3. Asymmetry Tracking**
```
asymmetry_accuracy = corr(spread_asymmetry, price_change)
```
Target: > 0.2 (some predictive power)

### Execution Quality Metrics

**1. Fill Rate**
```
fill_rate = fills_per_hour / quotes_per_hour
```
Target: 20-50% (balanced activity)

**2. Adverse Selection Cost**
```
adverse_cost = avg(mid_price_at_fill - fill_price)
```
Target: Minimize (avoid being picked off)

**3. Liquidation Efficiency**
```
liquidation_cost = avg_taker_price - avg_passive_price
```
Target: < 0.5 * spread (efficient exits)

### Inventory Metrics

**1. Mean Absolute Inventory**
```
mean_inventory = E[|Q_t|] / Q_max
```
Target: < 0.3 (good risk management)

**2. Inventory Autocorrelation**
```
autocorr(Q_t, Q_{t-1})
```
Target: < 0.9 (not persistent)

**3. Urgency Activation Rate**
```
urgency_rate = time_with_urgency > 0.7 / total_time
```
Target: < 10% (rare emergencies)

## Usage Examples

### Example 1: Basic Control

```rust
use hyperliquid_rust_sdk::ControlVector;

// Create symmetric control
let control = ControlVector::symmetric(10.0);

// Calculate quote prices
let (bid, ask) = control.calculate_quote_prices(100.0);
println!("Bid: ${:.2}, Ask: ${:.2}", bid, ask);
// Output: Bid: $99.90, Ask: $100.10

// Check spread
println!("Spread: {:.1} bps", control.total_spread_bps());
// Output: Spread: 20.0 bps
```

### Example 2: State-Driven Control

```rust
use hyperliquid_rust_sdk::{StateVector, ControlVector};

// Observe state
let state = StateVector {
    mid_price: 100.0,
    inventory: 75.0,  // 75% long
    adverse_selection_estimate: 0.05,  // Bullish
    market_spread_bps: 15.0,
    lob_imbalance: 0.65,
};

// Calculate optimal control
let mut control = ControlVector::symmetric(10.0);
control.apply_state_adjustments(&state, 10.0, 100.0);

println!("{}", control.to_log_string());
// Shows adjusted offsets based on state
```

### Example 3: Emergency Liquidation

```rust
// Critical inventory situation
let state = StateVector {
    mid_price: 100.0,
    inventory: 95.0,  // 95% of max!
    adverse_selection_estimate: 0.0,
    market_spread_bps: 20.0,
    lob_imbalance: 0.5,
};

let mut control = ControlVector::symmetric(10.0);
control.apply_state_adjustments(&state, 10.0, 100.0);

if control.is_liquidating() {
    println!("EMERGENCY: Taker sell rate = {:.2} units/sec", 
             control.taker_sell_rate);
    // Output: EMERGENCY: Taker sell rate = 2.50 units/sec
}
```

### Example 4: Validation

```rust
let mut control = ControlVector::asymmetric(5.0, 3.0);

match control.validate(10.0) {
    Ok(_) => println!("Control is valid"),
    Err(e) => {
        println!("Validation failed: {}", e);
        // Output: Validation failed: Total spread 8.00 bps is below minimum 10.00 bps
        
        // Fix it
        control = ControlVector::symmetric(10.0);
    }
}
```

### Example 5: Custom Strategy

```rust
// Implement custom control logic
fn calculate_custom_control(
    state: &StateVector,
    base_spread: f64,
) -> ControlVector {
    let mut control = ControlVector::symmetric(base_spread);
    
    // Custom logic: widen spread in volatile markets
    if state.market_spread_bps > 30.0 {
        control.ask_offset_bps *= 1.5;
        control.bid_offset_bps *= 1.5;
    }
    
    // Custom logic: aggressive when contrarian
    if state.adverse_selection_estimate > 0.1 && state.inventory < 0.0 {
        // Bullish signal + short position = aggressive buying
        control.bid_offset_bps *= 0.7;  // Tighter bid
    }
    
    control
}
```

## Configuration Parameters

### Tunable Parameters

Located in `apply_state_adjustments()`:

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| Adverse Adjustment Factor | 0.5 | [0.0, 1.0] | Impact of μ̂_t on spreads |
| Inventory Skew Factor | 0.5 | [0.0, 1.0] | Max skew from inventory |
| Urgency Threshold | 0.7 | [0.5, 0.9] | When to activate taker |
| Liquidation Scale | 10.0 | [5.0, 20.0] | Taker rate multiplier |
| Min Offset Ratio | 0.2 | [0.1, 0.5] | Min offset vs base |

### Recommended Settings

**Conservative Strategy:**
```rust
// In apply_state_adjustments():
let skew_adjustment = inventory_ratio * base_spread_bps * 0.3;  // 30% max
let urgency_threshold = 0.8;  // Higher threshold
let liquidation_rate = (urgency - threshold) * 5.0;  // Slower rate
```

**Aggressive Strategy:**
```rust
let skew_adjustment = inventory_ratio * base_spread_bps * 0.7;  // 70% max
let urgency_threshold = 0.6;  // Lower threshold
let liquidation_rate = (urgency - threshold) * 15.0;  // Faster rate
```

## Future Enhancements

### 1. Optimal Execution

Implement time-optimal liquidation:

```rust
pub struct ControlVector {
    // ... existing fields
    pub target_inventory: f64,
    pub time_horizon: f64,
}

impl ControlVector {
    pub fn optimal_liquidation_rate(&self, current_inventory: f64) -> f64 {
        // Almgren-Chriss optimal execution
        (current_inventory - self.target_inventory) / self.time_horizon
    }
}
```

### 2. Multi-Level Quoting

Support multiple quote levels:

```rust
pub struct MultiLevelControl {
    pub levels: Vec<QuoteLevel>,
}

pub struct QuoteLevel {
    pub ask_offset_bps: f64,
    pub bid_offset_bps: f64,
    pub size_fraction: f64,
}
```

### 3. Dynamic Spread Targeting

Adjust to market conditions:

```rust
impl ControlVector {
    pub fn dynamic_spread(&self, volatility: f64, liquidity: f64) -> f64 {
        base_spread * (1.0 + volatility) * (1.0 / liquidity)
    }
}
```

### 4. Learning-Based Control

Use reinforcement learning:

```rust
pub struct QLearningControl {
    q_table: HashMap<(StateVector, ControlVector), f64>,
    learning_rate: f64,
    discount_factor: f64,
}
```

## References

### Academic Papers

1. **Avellaneda & Stoikov (2008)**: "High-frequency trading in a limit order book"
   - Foundation for optimal market making
   - Derives optimal bid/ask placement

2. **Guéant et al. (2013)**: "Dealing with the inventory risk"
   - Inventory risk management
   - Closed-form solutions for simple cases

3. **Cartea & Jaimungal (2015)**: "Risk metrics and fine tuning of HFT strategies"
   - Risk metrics for algorithmic trading
   - Control parameter selection

4. **Almgren & Chriss (2001)**: "Optimal execution of portfolio transactions"
   - Optimal liquidation strategies
   - Trade-off between urgency and cost

### Related Documentation

- `STATE_VECTOR.md`: State observation framework
- `INVENTORY_SKEWING.md`: Inventory management
- `STATE_VECTOR_QUICK_REF.md`: Quick reference guide

## Conclusion

The Control Vector ($\mathbf{u}_t$) provides a complete framework for **action selection** in algorithmic market making. By systematically adjusting:

1. **Passive quotes** ($\delta^a_t, \delta^b_t$) for normal operation
2. **Active orders** ($\nu^a_t, \nu^b_t$) for emergencies

The algorithm can optimally balance:
- Profitability (spread capture)
- Risk management (inventory control)
- Adverse selection protection (signal-based tilting)

Combined with the State Vector, this creates a complete **observe-decide-act** loop for sophisticated market making.

---

**Version**: 1.0  
**Date**: October 23, 2025  
**Status**: ✓ Complete and tested
