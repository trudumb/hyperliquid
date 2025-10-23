# State-Control Framework: Complete Reference

## Overview

This document describes how the **State Vector ($\mathbf{Z}_t$)** and **Control Vector ($\mathbf{u}_t$)** work together to create an optimal market making system.

## The Complete Framework

```
┌──────────────────────────────────────────────────────────────────┐
│                      MARKET OBSERVATIONS                          │
│     (Price, Order Book, Inventory, Fills)                        │
└────────────────────────────┬─────────────────────────────────────┘
                             │
                             ▼
                   ┌─────────────────────┐
                   │  STATE VECTOR (Z_t) │
                   │  ─────────────────  │
                   │  • S_t (Mid Price)  │
                   │  • Q_t (Inventory)  │
                   │  • μ̂_t (Adverse Sel)│
                   │  • Δ_t (Spread)     │
                   │  • I_t (Imbalance)  │
                   └──────────┬──────────┘
                              │
                              │ Informs
                              │
                              ▼
                   ┌─────────────────────┐
                   │ DECISION ALGORITHM  │
                   │  ─────────────────  │
                   │ apply_state_        │
                   │ adjustments()       │
                   └──────────┬──────────┘
                              │
                              │ Produces
                              │
                              ▼
                   ┌─────────────────────┐
                   │ CONTROL VECTOR (u_t)│
                   │  ─────────────────  │
                   │  • δ^a_t (Ask Off)  │
                   │  • δ^b_t (Bid Off)  │
                   │  • ν^a_t (Sell Rate)│
                   │  • ν^b_t (Buy Rate) │
                   └──────────┬──────────┘
                              │
                              │ Executes
                              │
                              ▼
                   ┌─────────────────────┐
                   │   ORDER PLACEMENT   │
                   │  ─────────────────  │
                   │  • Passive Quotes   │
                   │  • Taker Orders     │
                   └─────────────────────┘
```

## Mathematical Formulation

### State Space
$$\mathbf{Z}_t = (S_t, Q_t, \hat{\mu}_t, \Delta_t, I_t) \in \mathbb{R}^5$$

### Action Space
$$\mathbf{u}_t = (\delta^a_t, \delta^b_t, \nu^a_t, \nu^b_t) \in \mathbb{R}_+^4$$

### State Transition
$$\mathbf{Z}_{t+1} = f(\mathbf{Z}_t, \mathbf{u}_t, \omega_t)$$

where $\omega_t$ represents market randomness.

### Policy
$$\pi: \mathbf{Z}_t \mapsto \mathbf{u}_t$$

Our policy is defined by `apply_state_adjustments()`.

## Complete State-Control Mapping

| State Condition | Control Response | Reasoning |
|----------------|------------------|-----------|
| **Adverse Selection** |||
| $\hat{\mu}_t > 0$ (bullish) | $\delta^a_t \uparrow$, $\delta^b_t \downarrow$ | Price likely to rise, widen ask, tighten bid |
| $\hat{\mu}_t < 0$ (bearish) | $\delta^a_t \downarrow$, $\delta^b_t \uparrow$ | Price likely to fall, widen bid, tighten ask |
| **Inventory Risk** |||
| $\|Q_t\| \to Q_{\max}$ | $\delta^a_t \uparrow$, $\delta^b_t \uparrow$ | High risk, widen both spreads |
| $\|Q_t\| \to 0$ | $\delta^a_t \to$ base, $\delta^b_t \to$ base | Low risk, normal spreads |
| **Inventory Skew** |||
| $Q_t > 0$ (long) | $\delta^a_t \downarrow$, $\delta^b_t \uparrow$ | Want to sell, tighten ask, widen bid |
| $Q_t < 0$ (short) | $\delta^a_t \uparrow$, $\delta^b_t \downarrow$ | Want to buy, widen ask, tighten bid |
| **Emergency Liquidation** |||
| Urgency > 0.7 & $Q_t > 0$ | $\nu^a_t > 0$ | Aggressively sell to reduce long |
| Urgency > 0.7 & $Q_t < 0$ | $\nu^b_t > 0$ | Aggressively buy to cover short |
| **Market Conditions** |||
| $\Delta_t$ high (wide spread) | $\delta^a_t \uparrow$, $\delta^b_t \uparrow$ | Volatile market, wider quotes |
| $I_t$ extreme (imbalanced) | Adjust $\hat{\mu}_t$ | Update drift estimate |

## Decision Algorithm

### Complete Pseudocode

```python
def calculate_optimal_control(state: StateVector, base_spread: float) -> ControlVector:
    # Initialize with symmetric control
    control = ControlVector.symmetric(base_spread)
    
    # 1. ADVERSE SELECTION ADJUSTMENT
    adverse_adj = state.get_adverse_selection_adjustment(base_spread)
    if adverse_adj > 0:  # Bearish signal
        control.bid_offset_bps += adverse_adj
    else:  # Bullish signal
        control.ask_offset_bps -= adverse_adj
    
    # 2. INVENTORY RISK ADJUSTMENT
    risk_mult = 1 + (state.inventory / max_inventory)^2
    control.ask_offset_bps *= risk_mult
    control.bid_offset_bps *= risk_mult
    
    # 3. INVENTORY SKEW ADJUSTMENT
    inventory_ratio = state.inventory / max_inventory
    skew = inventory_ratio * base_spread * 0.5
    control.ask_offset_bps -= skew  # Long -> tighter ask
    control.bid_offset_bps += skew  # Long -> wider bid
    
    # 4. ENSURE MINIMUM OFFSETS
    min_offset = base_spread * 0.2
    control.ask_offset_bps = max(control.ask_offset_bps, min_offset)
    control.bid_offset_bps = max(control.bid_offset_bps, min_offset)
    
    # 5. ACTIVE LIQUIDATION CONTROL
    urgency = (state.inventory / max_inventory)^3
    if urgency > 0.7:
        liquidation_rate = (urgency - 0.7) * 10.0
        if state.inventory > 0:
            control.taker_sell_rate = liquidation_rate
        else:
            control.taker_buy_rate = liquidation_rate
    
    return control
```

## Example Scenarios

### Scenario 1: Normal Market Making

**State:**
```
S_t = 100.00
Q_t = 0.0 (no position)
μ̂_t = 0.0 (neutral)
Δ_t = 10.0 bps (normal spread)
I_t = 0.5 (balanced)
```

**Control:**
```
δ^a_t = 10.0 bps → Ask = $100.10
δ^b_t = 10.0 bps → Bid = $99.90
ν^a_t = 0.0
ν^b_t = 0.0
```

**Result:** Symmetric quotes, 20 bps spread, passive only.

---

### Scenario 2: Bullish Signal

**State:**
```
S_t = 100.00
Q_t = 0.0
μ̂_t = 0.15 (strong bullish)
Δ_t = 10.0 bps
I_t = 0.75 (heavy bid volume)
```

**Control:**
```
δ^a_t = 13.0 bps → Ask = $100.13 (widened)
δ^b_t = 8.0 bps → Bid = $99.92 (tightened)
ν^a_t = 0.0
ν^b_t = 0.0
```

**Result:** Asymmetric quotes favoring buys, +5 bps bullish bias.

---

### Scenario 3: High Inventory

**State:**
```
S_t = 100.00
Q_t = 80.0 (80% of max)
μ̂_t = 0.0
Δ_t = 10.0 bps
I_t = 0.5
```

**Control:**
```
δ^a_t = 11.0 bps → Ask = $100.11 (tightened to sell)
δ^b_t = 17.0 bps → Bid = $99.83 (widened)
ν^a_t = 0.0
ν^b_t = 0.0
```

**Result:** Skewed quotes to encourage selling, asymmetric spread.

---

### Scenario 4: Emergency Exit

**State:**
```
S_t = 100.00
Q_t = 95.0 (95% of max!)
μ̂_t = -0.05 (bearish signal too!)
Δ_t = 15.0 bps (volatile)
I_t = 0.35 (selling pressure)
```

**Control:**
```
δ^a_t = 18.0 bps → Ask = $100.18 (wide for safety)
δ^b_t = 22.0 bps → Bid = $99.78 (very wide)
ν^a_t = 2.75 units/sec (ACTIVE SELLING!)
ν^b_t = 0.0
```

**Result:** Emergency liquidation via taker orders + protective spreads.

---

### Scenario 5: Volatile Market

**State:**
```
S_t = 100.00
Q_t = 20.0 (modest position)
μ̂_t = 0.0
Δ_t = 50.0 bps (very wide!)
I_t = 0.5
```

**Control:**
```
δ^a_t = 25.0 bps → Ask = $100.25 (wide for protection)
δ^b_t = 25.0 bps → Bid = $99.75
ν^a_t = 0.0
ν^b_t = 0.0
```

**Result:** Wide symmetric quotes for uncertain conditions.

## Performance Metrics

### Joint State-Control Metrics

**1. Policy Effectiveness**
```
effectiveness = E[PnL | (Z_t, u_t)] / E[PnL | Z_t only]
```
Measures improvement from optimal control vs. fixed control.

**2. State-Action Coherence**
```
coherence = corr(u_t - u_{base}, Z_t - Z_{equilibrium})
```
Measures if control adjustments align with state deviations.

**3. Regret**
```
regret = E[PnL_optimal(Z_t) - PnL_actual(Z_t, u_t)]
```
Gap between optimal policy and actual performance.

### Tracking Effectiveness

```rust
// Log state and control together
info!("State: {}", state.to_log_string());
info!("Control: {}", control.to_log_string());

// Example output:
// State: StateVector[S=100.50, Q=75.0000, μ̂=0.0150, Δ=12.5bps, I=0.625]
// Control: ControlVector[δ^b=15.5bps, δ^a=8.5bps, spread=24.0bps, asymmetry=-7.0bps]
```

## Complete Usage Example

```rust
use hyperliquid_rust_sdk::{StateVector, ControlVector};

fn trading_loop(market_maker: &mut MarketMaker) {
    loop {
        // 1. OBSERVE: Update state from market data
        market_maker.update_state_vector();
        let state = market_maker.get_state_vector();
        
        // 2. DECIDE: Calculate optimal control
        market_maker.calculate_optimal_control();
        let control = market_maker.get_control_vector();
        
        // 3. ACT: Place orders based on control
        let (bid_price, ask_price) = control.calculate_quote_prices(state.mid_price);
        
        // 3a. Passive quotes
        place_bid_order(bid_price, liquidity);
        place_ask_order(ask_price, liquidity);
        
        // 3b. Taker orders (if needed)
        if control.taker_sell_rate > 0.0 {
            let size = control.taker_sell_rate * time_interval;
            place_market_sell(size);
        }
        if control.taker_buy_rate > 0.0 {
            let size = control.taker_buy_rate * time_interval;
            place_market_buy(size);
        }
        
        // 4. MONITOR: Log and track
        println!("State: {}", state.to_log_string());
        println!("Control: {}", control.to_log_string());
        
        // 5. WAIT: Next update cycle
        sleep(update_interval);
    }
}
```

## Optimization Criteria

### Primary Objective

$$\max_{\mathbf{u}_t} \mathbb{E}\left[\sum_{t=0}^T \text{PnL}_t - \gamma Q_t^2\right]$$

Subject to:
- State evolution: $\mathbf{Z}_{t+1} = f(\mathbf{Z}_t, \mathbf{u}_t, \omega_t)$
- Control constraints: $\mathbf{u}_t \in \mathcal{U}$
- Position limits: $|Q_t| \leq Q_{\max}$

### Components

**PnL_t:**
```
PnL_t = spread_captured_t - adverse_selection_cost_t - taker_fees_t
```

**Spread Captured:**
```
spread_captured = (δ^a_t + δ^b_t) * fill_rate_t
```

**Adverse Selection Cost:**
```
adverse_cost = |μ̂_t| * position_change_t
```

**Taker Fees:**
```
taker_fees = (ν^a_t + ν^b_t) * fee_rate
```

**Inventory Penalty:**
```
inventory_penalty = γ * Q_t^2
```

## Configuration Guide

### Conservative Settings

For risk-averse market making:

```rust
// State Vector
const LAMBDA: f64 = 0.05;  // Slow adverse selection updates
const ADJUSTMENT_FACTOR: f64 = 0.3;  // Small adjustments

// Control Vector
const INVENTORY_SKEW_FACTOR: f64 = 0.3;  // Modest skewing
const URGENCY_THRESHOLD: f64 = 0.8;  // High threshold
const LIQUIDATION_SCALE: f64 = 5.0;  // Slow liquidation
```

### Aggressive Settings

For active trading:

```rust
// State Vector
const LAMBDA: f64 = 0.2;  // Fast adverse selection updates
const ADJUSTMENT_FACTOR: f64 = 0.8;  // Large adjustments

// Control Vector
const INVENTORY_SKEW_FACTOR: f64 = 0.7;  // Aggressive skewing
const URGENCY_THRESHOLD: f64 = 0.6;  // Low threshold
const LIQUIDATION_SCALE: f64 = 15.0;  // Fast liquidation
```

## Troubleshooting

| Issue | State Symptom | Control Response | Fix |
|-------|---------------|------------------|-----|
| Getting picked off | $\|\hat{\mu}_t\|$ low, losses | Spreads too tight | Increase base spread |
| Low fill rate | Normal | Spreads too wide | Decrease risk multiplier |
| Inventory builds up | $Q_t$ growing | Skew not working | Increase skew factor |
| Frequent liquidations | Urgency often > 0.7 | Threshold too low | Increase threshold |
| Large P&L swings | $Q_t$ volatile | Late liquidation | Decrease threshold |

## Advanced Topics

### 1. Multi-Asset Control

Coordinate control across assets:

```rust
struct PortfolioControl {
    controls: HashMap<String, ControlVector>,
    total_risk_limit: f64,
}

impl PortfolioControl {
    fn allocate_risk(&mut self, states: &HashMap<String, StateVector>) {
        let total_risk: f64 = states.values()
            .map(|s| s.inventory.powi(2))
            .sum();
            
        for (asset, control) in &mut self.controls {
            let risk_fraction = states[asset].inventory.powi(2) / total_risk;
            // Adjust control based on portfolio risk allocation
        }
    }
}
```

### 2. Learning-Based Control

Use RL to learn optimal policy:

```rust
struct QLearning {
    q_table: HashMap<(StateVector, ControlVector), f64>,
}

impl QLearning {
    fn update(&mut self, state: StateVector, control: ControlVector, reward: f64) {
        // Q-learning update rule
        let q_old = self.q_table.get(&(state.clone(), control.clone())).unwrap_or(&0.0);
        let q_new = q_old + learning_rate * (reward + discount * max_q_next - q_old);
        self.q_table.insert((state, control), q_new);
    }
}
```

### 3. Regime-Dependent Control

Different policies for different regimes:

```rust
enum MarketRegime {
    Calm,
    Volatile,
    Trending,
    Crisis,
}

fn select_control_parameters(regime: MarketRegime) -> ControlParameters {
    match regime {
        MarketRegime::Calm => ControlParameters { /* tight spreads */ },
        MarketRegime::Volatile => ControlParameters { /* wide spreads */ },
        MarketRegime::Trending => ControlParameters { /* directional bias */ },
        MarketRegime::Crisis => ControlParameters { /* defensive */ },
    }
}
```

## Conclusion

The State-Control framework provides a complete **observe-decide-act** loop:

1. **State Vector** observes market conditions
2. **Decision Algorithm** processes information
3. **Control Vector** specifies actions
4. **Execution** implements the strategy

This creates a systematic, reproducible, and optimizable market making system that can:
- Adapt to market conditions
- Manage risk effectively
- Protect against adverse selection
- Handle emergency situations

---

**Version**: 1.0  
**Date**: October 23, 2025  
**Status**: ✓ Complete
