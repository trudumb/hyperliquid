# State Vector Implementation for Market Making

## Overview

This document describes the implementation of the **State Vector ($\mathbf{Z}_t$)** for optimal market making decisions in the Hyperliquid Rust SDK. The state vector represents all information needed to make optimal trading decisions at time $t$.

## Mathematical Definition

$$\mathbf{Z}_t = (S_t, Q_t, \hat{\mu}_t, \Delta_t, I_t)$$

### Components

#### 1. $S_t$ - Mid-Price
**Field:** `mid_price: f64`

The core price process representing the current market price. Calculated as:
$$S_t = \frac{S^b_t + S^a_t}{2}$$

where $S^b_t$ is the best bid and $S^a_t$ is the best ask.

**Purpose:** Forms the baseline around which we set our quotes.

#### 2. $Q_t$ - Inventory
**Field:** `inventory: f64`

Our current position in the asset. Positive values indicate long positions, negative values indicate short positions.

**Purpose:** Critical for risk management and position management. Higher absolute inventory increases our risk exposure and should influence our quoting strategy.

#### 3. $\hat{\mu}_t$ - Adverse Selection State
**Field:** `adverse_selection_estimate: f64`

The filtered estimate of the true, unobserved short-term drift. This represents our best guess of $\mathbb{E}[\mu_{\text{true}}]$.

**Interpretation:**
- If $\hat{\mu}_t > 0$: We believe the market is about to move up (bullish signal)
- If $\hat{\mu}_t < 0$: We believe the market is about to move down (bearish signal)
- If $\hat{\mu}_t \approx 0$: No directional bias

**Update Formula:**
$$\hat{\mu}_t = \lambda \cdot \text{signal}_t \cdot \text{spread\_scale}_t + (1-\lambda) \cdot \hat{\mu}_{t-1}$$

where:
- $\lambda = 0.1$ (smoothing parameter, configurable)
- $\text{signal}_t = 2 \cdot (I_t - 0.5)$ (directional signal from imbalance)
- $\text{spread\_scale}_t = \frac{1}{1 + \Delta_t/100}$ (volatility adjustment)

**Purpose:** This is the "secret sauce" that helps us avoid adverse selection. By estimating the short-term drift, we can:
1. Skew our quotes asymmetrically when we detect directional momentum
2. Widen spreads in the direction of expected movement
3. Avoid being picked off by informed traders

#### 4. $\Delta_t$ - Market Spread
**Field:** `market_spread_bps: f64`

The current Best-Bid-Offer (BBO) spread in basis points:
$$\Delta_t = \frac{S^a_t - S^b_t}{S_t} \times 10000$$

**Purpose:** Proxy for market-wide volatility and liquidity conditions. Wider spreads indicate:
- Higher market volatility
- Lower liquidity
- Greater uncertainty
- Need for wider protective spreads

#### 5. $I_t$ - LOB Imbalance
**Field:** `lob_imbalance: f64`

The ratio of volume at the Best Bid and Offer (BBO):
$$I_t = \frac{V^b_t}{V^b_t + V^a_t}$$

where $V^b_t$ is bid volume and $V^a_t$ is ask volume at the BBO.

**Range:** $[0, 1]$
- $I_t = 0$: All volume on ask side (strong selling pressure)
- $I_t = 0.5$: Balanced order book
- $I_t = 1$: All volume on bid side (strong buying pressure)

**Purpose:** Key predictor for short-term price movements. Used to update $\hat{\mu}_t$. Imbalanced order books often predict the direction of the next price move.

## Implementation Details

### StateVector Structure

```rust
pub struct StateVector {
    pub mid_price: f64,
    pub adverse_selection_estimate: f64,
    pub market_spread_bps: f64,
    pub lob_imbalance: f64,
    pub inventory: f64,
}
```

### Key Methods

#### `update()`
Updates all components of the state vector based on current market data:
- Receives mid price, inventory, book analysis, and order book
- Automatically calculates spread and imbalance
- Updates adverse selection estimate using exponential moving average

#### `get_adverse_selection_adjustment()`
Returns optimal spread adjustment in basis points based on adverse selection:
$$\text{adjustment} = -\hat{\mu}_t \times \text{base\_spread} \times \alpha$$

where $\alpha = 0.5$ is an adjustment factor.

**Sign Convention:**
- Positive: Widen buy-side spread (bearish signal)
- Negative: Widen sell-side spread (bullish signal)

#### `get_inventory_risk_multiplier()`
Returns spread widening multiplier based on inventory risk:
$$\text{multiplier} = 1 + \left(\frac{|Q_t|}{Q_{\max}}\right)^2$$

Range: $[1.0, 2.0]$

**Purpose:** Discourage further inventory accumulation by widening spreads as we approach position limits.

#### `get_inventory_urgency()`
Returns urgency score for inventory reduction:
$$\text{urgency} = \left(\frac{|Q_t|}{Q_{\max}}\right)^3$$

Range: $[0.0, 1.0]$

**Purpose:** Cubic function creates rapid increase in urgency near position limits.

#### `is_market_favorable()`
Checks if market conditions are suitable for trading:
- Spread is reasonable (not too wide)
- Order book is not extremely one-sided
- Returns `false` during abnormal market conditions

### Integration with MarketMaker

The state vector is:
1. **Initialized** when creating a new `MarketMaker` instance
2. **Updated** on every:
   - L2 book update
   - Mid price update
   - Fill event (inventory change)
3. **Logged** after each update for monitoring
4. **Queried** to make informed trading decisions

### Usage Examples

#### Example 1: Basic State Vector Access
```rust
// Get current state vector
let state = market_maker.get_state_vector();

// Check adverse selection signal
if state.adverse_selection_estimate > 0.05 {
    println!("Bullish signal detected!");
}

// Monitor inventory risk
let urgency = state.get_inventory_urgency(max_position);
if urgency > 0.7 {
    println!("High inventory urgency - need to reduce position");
}
```

#### Example 2: State-Based Spread Adjustment
```rust
// Calculate optimal spread adjustment
let adjustment = market_maker.calculate_state_based_spread_adjustment();

// Apply to pricing logic
let adjusted_half_spread = base_half_spread + adjustment;
```

#### Example 3: Market Condition Monitoring
```rust
// Check if we should pause trading
if market_maker.should_pause_trading() {
    println!("Unfavorable market conditions - pausing");
    return;
}
```

## Configuration Parameters

### Tunable Parameters

1. **Lambda ($\lambda$)**: Smoothing parameter for adverse selection filter
   - Default: `0.1`
   - Range: `[0.01, 0.5]`
   - Lower = smoother, slower response
   - Higher = more reactive, noisier

2. **Adjustment Factor ($\alpha$)**: Controls adverse selection impact on spreads
   - Default: `0.5`
   - Range: `[0.0, 1.0]`
   - `0.0` = no adjustment
   - `1.0` = full adjustment

3. **Max Spread Threshold**: Maximum acceptable market spread
   - Default: `5x base spread`
   - Purpose: Identify abnormal market conditions

## Theoretical Foundation

### Adverse Selection Problem

In market making, **adverse selection** occurs when informed traders preferentially trade against our quotes, resulting in losses. The adverse selection state $\hat{\mu}_t$ helps mitigate this by:

1. **Detection**: Identifying when informed flow is likely
2. **Protection**: Adjusting spreads asymmetrically
3. **Adaptation**: Dynamically responding to market microstructure

### Information Sources

The state vector aggregates multiple information sources:

| Component | Information Type | Update Frequency |
|-----------|-----------------|------------------|
| $S_t$ | Public price | Every tick |
| $Q_t$ | Private inventory | On fills |
| $\hat{\mu}_t$ | Filtered signal | Every book update |
| $\Delta_t$ | Market liquidity | Every book update |
| $I_t$ | Order flow | Every book update |

### Optimal Control Framework

The state vector enables a **Markov Decision Process (MDP)** formulation:
- **State**: $\mathbf{Z}_t$
- **Action**: Quote placement $(p^b, p^a, q^b, q^a)$
- **Reward**: PnL with inventory penalties
- **Objective**: Maximize $\mathbb{E}[\text{PnL}] - \gamma \mathbb{E}[Q_T^2]$

where $\gamma$ is the inventory aversion parameter.

## Performance Monitoring

### Key Metrics to Track

1. **Adverse Selection Quality**
   - Correlation between $\hat{\mu}_t$ and actual price moves
   - Hit rate of directional predictions

2. **Inventory Management**
   - Mean absolute inventory
   - Inventory half-life (time to reduce by 50%)
   - Maximum inventory reached

3. **Spread Efficiency**
   - Average realized spread
   - Spread vs. volatility ratio
   - Adjustment frequency

### Logging Output

Example log message:
```
StateVector[S=100.50, Q=2.4500, μ̂=0.0234, Δ=8.5bps, I=0.623]
```

Interpretation:
- Mid price: $100.50
- Inventory: 2.45 (long position)
- Adverse selection: +0.0234 (slight bullish bias)
- Market spread: 8.5 bps
- LOB imbalance: 0.623 (more bid volume, buying pressure)

## Future Enhancements

### Potential Improvements

1. **Dynamic Lambda**: Adjust smoothing based on market regime
2. **Multi-Level Imbalance**: Use deeper book levels for $I_t$
3. **Volatility Forecast**: Add $\hat{\sigma}_t$ to state vector
4. **Regime Detection**: Add market regime indicator to state
5. **Machine Learning**: Train neural network to predict $\hat{\mu}_t$

### Advanced Features

1. **Multi-Asset State**: Track correlations across assets
2. **Event Detection**: Identify news/announcements in state
3. **Execution Quality**: Track fill rates and adverse selection costs
4. **Risk Decomposition**: Separate systematic vs. idiosyncratic risk

## References

- Avellaneda, M., & Stoikov, S. (2008). "High-frequency trading in a limit order book"
- Cartea, Á., Jaimungal, S., & Penalva, J. (2015). "Algorithmic and High-Frequency Trading"
- Cont, R., Stoikov, S., & Talreja, R. (2010). "A stochastic model for order book dynamics"

## Conclusion

The state vector $\mathbf{Z}_t$ provides a comprehensive framework for optimal market making decisions. By tracking mid price, inventory, adverse selection, spread, and imbalance, the algorithm can:

1. **Protect** against informed traders
2. **Manage** inventory risk effectively  
3. **Adapt** to changing market conditions
4. **Optimize** spread placement dynamically

This implementation forms the foundation for sophisticated, profitable market making strategies.
