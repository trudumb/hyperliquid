# Dynamic Spread Implementation - Complete Guide

## Current Status: ✅ **FULLY IMPLEMENTED**

Your market maker already has a **sophisticated dynamic spread system** based on real-time market volatility!

---

## 1. What's Already Implemented

### ✅ Volatility Tracking in StateVector

```rust
pub struct StateVector {
    pub volatility_ema_bps: f64,  // Real-time volatility estimate
    pub previous_mid_price: f64,  // For volatility calculation
    // ... other fields
}
```

**How it works:**
- Calculates instantaneous volatility from log returns: `ln(P_t / P_{t-1})`
- Converts to basis points (× 10,000)
- Smooths with EMA: `σ̂_t = λ × vol_t + (1 - λ) × σ̂_{t-1}`
- Rejects outliers (> 1000 bps = 10% moves)
- Updates every time the mid price changes

**Location:** `src/market_maker_v2.rs:285-306`

### ✅ Dynamic Spread Calculation

```rust
fn calculate_optimal_control(&mut self) {
    // Dynamic spread based on real-time volatility
    let base_spread_bps = self.state_vector.volatility_ema_bps 
                        * self.spread_volatility_multiplier;
    
    // Ensure minimum spread
    let base_spread_bps = base_spread_bps.max(5.0);
    
    // ... rest of control calculation
}
```

**How it works:**
- `base_spread = volatility_ema × multiplier`
- Example: If volatility = 10 bps and multiplier = 1.5, then spread = 15 bps
- Automatically widens in volatile markets, tightens in calm markets
- Has a safety floor (min 5 bps) to prevent zero spreads

**Location:** `src/market_maker_v2.rs:1081-1125`

### ✅ Configuration via `spread_volatility_multiplier`

```rust
pub struct MarketMakerInput {
    pub spread_volatility_multiplier: f64,  // e.g., 1.5
    // ... other fields
}
```

**Current settings** (in `src/bin/market_maker_v2.rs`):
```rust
spread_volatility_multiplier: 1.5  // Spread = 1.5 × volatility
```

**Location:** `src/market_maker_v2.rs:1021` and `src/bin/market_maker_v2.rs:49`

---

## 2. How The Dynamic Spread System Works

### Data Flow

```
Mid Price Updates
       ↓
Calculate Log Returns → ln(P_t / P_{t-1})
       ↓
Convert to Basis Points → |log_return| × 10,000
       ↓
EMA Smoothing → σ̂_t = λ × vol_t + (1-λ) × σ̂_{t-1}
       ↓
Dynamic Spread → spread = σ̂_t × multiplier
       ↓
Apply to Quotes → bid/ask offsets
```

### Example Scenario

| Market Condition | Volatility EMA | Multiplier | Resulting Spread | Behavior |
|-----------------|----------------|------------|------------------|----------|
| **Calm Market** | 8 bps | 1.5 | 12 bps | Tight quotes = competitive |
| **Normal Market** | 12 bps | 1.5 | 18 bps | Moderate spread |
| **Volatile Market** | 25 bps | 1.5 | 37.5 bps | Wide spread = protection |
| **Extreme Volatility** | 50 bps | 1.5 | 75 bps | Very wide = safe |

---

## 3. Tuning Recommendations

### Adjusting `spread_volatility_multiplier`

**Current:** `1.5` (moderate-aggressive)

**Recommended ranges:**

| Multiplier | Risk Profile | Use Case |
|-----------|-------------|----------|
| `1.0 - 1.2` | Aggressive | High-frequency, very liquid markets |
| `1.3 - 1.7` | Moderate | **Most markets (recommended)** |
| `1.8 - 2.5` | Conservative | Illiquid or highly volatile assets |
| `2.5+` | Very Conservative | Extreme risk aversion |

**Example configurations:**

```rust
// Ultra-competitive (tight spreads)
spread_volatility_multiplier: 1.2,

// Current (moderate)
spread_volatility_multiplier: 1.5,

// Conservative (wide spreads)
spread_volatility_multiplier: 2.0,
```

### Adjusting EMA Smoothing (`adverse_selection_lambda`)

The volatility EMA uses `tuning_params.adverse_selection_lambda` for smoothing:

**Current:** `0.1` (in `TuningParams::default()`)

```rust
// In tuning_params.json
{
  "adverse_selection_lambda": 0.1  // 10% weight on new observations
}
```

**Tuning guidelines:**

| Lambda | Responsiveness | Use Case |
|--------|----------------|----------|
| `0.01 - 0.05` | Very slow | Ignore short-term noise |
| `0.05 - 0.15` | Moderate | **Balanced (recommended)** |
| `0.15 - 0.30` | Fast | React quickly to regime changes |
| `0.30+` | Very fast | Ultra-responsive (noisy) |

---

## 4. Making `max_bps_diff` Dynamic (RECOMMENDED NEXT STEP)

Currently `max_bps_diff` is **static** (hardcoded to 10 bps in the bin file). This controls when orders are cancelled and replaced due to price deviation.

### Why Make It Dynamic?

**Problem with static `max_bps_diff`:**
- In volatile markets: 10 bps threshold is hit constantly → excessive cancellations
- In calm markets: 10 bps threshold is never hit → stale quotes far from optimal

**Solution: Scale it with volatility**

### Implementation

#### Step 1: Add field to `MarketMakerInput`

```rust
pub struct MarketMakerInput {
    // ... existing fields
    
    // NEW: Dynamic max_bps_diff parameters
    pub requote_threshold_multiplier: f64,  // e.g., 1.0 means requote at 1.0 × volatility
}
```

#### Step 2: Calculate dynamic threshold

```rust
impl MarketMaker {
    fn get_dynamic_requote_threshold(&self) -> f64 {
        // Requote when deviation exceeds volatility × multiplier
        // Example: if vol = 15 bps and multiplier = 1.0, requote at 15 bps deviation
        let threshold = self.state_vector.volatility_ema_bps 
                      * self.requote_threshold_multiplier;
        
        // Ensure reasonable bounds
        threshold.clamp(5.0, 50.0)  // Min 5 bps, max 50 bps
    }
}
```

#### Step 3: Use in `potentially_update()`

Replace this:
```rust
let lower_change = (lower_order_amount - self.lower_resting.position).abs() > EPSILON
    || bps_diff(lower_price, self.lower_resting.price) > self.max_bps_diff;  // STATIC
```

With this:
```rust
let dynamic_threshold = self.get_dynamic_requote_threshold();
let lower_change = (lower_order_amount - self.lower_resting.position).abs() > EPSILON
    || bps_diff(lower_price, self.lower_resting.price) > dynamic_threshold;  // DYNAMIC
```

#### Step 4: Configure in binary

```rust
let market_maker_input = MarketMakerInputV2 {
    // ... existing fields
    
    // Dynamic requoting: requote when price moves 1.0 × volatility
    requote_threshold_multiplier: 1.0,
};
```

### Recommended Multiplier Values

| Multiplier | Requote Frequency | Use Case |
|-----------|------------------|----------|
| `0.5` | Very frequent | HFT, tight inventory control |
| `0.8 - 1.2` | Moderate | **Most strategies (recommended)** |
| `1.5 - 2.0` | Conservative | Reduce transaction costs |
| `2.0+` | Very rare | Minimize exchange calls |

---

## 5. Complete Dynamic Configuration Example

### `src/bin/market_maker_v2.rs`

```rust
let market_maker_input = MarketMakerInputV2 {
    asset: "HYPE".to_string(),
    target_liquidity: 0.3,
    max_absolute_position_size: 3.0,
    asset_type: AssetType::Perp,
    wallet,
    inventory_skew_config: Some(skew_config),
    
    // ===== DYNAMIC PARAMETERS =====
    
    // Base spread: spread = volatility × multiplier
    spread_volatility_multiplier: 1.5,  // 1.5 × vol
    
    // Requote threshold: requote when deviation > volatility × multiplier
    requote_threshold_multiplier: 1.0,  // 1.0 × vol
    
    // DEPRECATED (still needed for backward compatibility)
    max_bps_diff: 10,  // Ignored if dynamic requoting is implemented
    half_spread: 5,    // Ignored - using dynamic spread instead
};
```

### `tuning_params.json`

```json
{
  "skew_adjustment_factor": 0.5,
  "adverse_selection_adjustment_factor": 0.5,
  "adverse_selection_lambda": 0.1,
  "inventory_urgency_threshold": 0.7,
  "liquidation_rate_multiplier": 10.0,
  "min_spread_base_ratio": 0.2,
  "adverse_selection_spread_scale": 100.0,
  "control_gap_threshold": 1.0
}
```

**Note:** `adverse_selection_lambda` controls volatility EMA smoothing.

---

## 6. Monitoring & Validation

### Key Metrics to Log

```rust
info!(
    "Dynamic Spread: vol={:.2}bps × mult={:.2} = {:.2}bps | Requote threshold: {:.2}bps",
    self.state_vector.volatility_ema_bps,
    self.spread_volatility_multiplier,
    base_spread_bps,
    self.get_dynamic_requote_threshold()
);
```

### Expected Behavior

**During market volatility spike:**
- Volatility EMA increases (e.g., 10 → 30 bps)
- Spread automatically widens (e.g., 15 → 45 bps)
- Requote threshold increases (e.g., 10 → 30 bps)
- Result: Wider protection, fewer requotes during chaos

**During market calm:**
- Volatility EMA decreases (e.g., 30 → 8 bps)
- Spread automatically tightens (e.g., 45 → 12 bps)
- Requote threshold decreases (e.g., 30 → 8 bps)
- Result: Competitive quotes, responsive requoting

---

## 7. Performance Considerations

### Current System Efficiency

✅ **Volatility calculation is fast** (log returns + EMA update)
✅ **Dynamic spread calculation is instant** (one multiplication)
✅ **State vector updates happen on every mid price change** (optimal)

### Potential Optimizations

1. **Batch volatility updates:** Only recalculate every N ticks (not recommended - loses accuracy)
2. **Cache base_spread:** Store in MarketMaker struct, update less frequently (saves ~1 multiplication)
3. **Async volatility calculation:** Not needed - already fast enough

**Verdict:** Current implementation is optimal. No changes needed.

---

## 8. Testing Recommendations

### Backtesting Scenarios

1. **Volatile Market Test**
   - Input: High-volatility price series (± 50+ bps moves)
   - Expected: Spreads widen to 40-75 bps, fewer adverse selection losses

2. **Calm Market Test**
   - Input: Low-volatility price series (± 5-10 bps moves)
   - Expected: Spreads tighten to 10-15 bps, high fill rates

3. **Volatility Regime Change**
   - Input: Transition from calm (8 bps) to volatile (40 bps)
   - Expected: Smooth spread widening via EMA, no abrupt changes

### Live Testing Checklist

- [ ] Monitor `volatility_ema_bps` over 24 hours
- [ ] Verify spread tracks volatility (correlation > 0.9)
- [ ] Check minimum spread floor is never violated
- [ ] Validate no excessive requoting (< 10 per minute avg)
- [ ] Measure adverse selection ratio (PnL vs. fills)

---

## 9. Summary

### What You Already Have ✅

| Feature | Status | Location |
|---------|--------|----------|
| Volatility EMA tracking | ✅ **Implemented** | `StateVector::update()` |
| Dynamic spread calculation | ✅ **Implemented** | `calculate_optimal_control()` |
| Configurable multiplier | ✅ **Implemented** | `spread_volatility_multiplier` |
| Real-time adjustment | ✅ **Implemented** | Updates on every mid price change |
| Safety bounds | ✅ **Implemented** | Min 5 bps floor |

### What's Still Static ⚠️

| Feature | Status | Recommendation |
|---------|--------|----------------|
| `max_bps_diff` (requote threshold) | ⚠️ **Static (10 bps)** | Make dynamic (see Section 4) |
| `half_spread` field | ⚠️ **Deprecated but present** | Can be removed (backward compat only) |

### Next Steps

1. **Test current dynamic spread system** in live market
2. **Implement dynamic `max_bps_diff`** (Section 4)
3. **Tune `spread_volatility_multiplier`** based on observed performance
4. **Adjust `adverse_selection_lambda`** to control EMA responsiveness
5. **Remove deprecated `half_spread` field** once stable

---

## 10. Code Snippets for Quick Reference

### Check Current Volatility
```rust
let current_vol = self.state_vector.volatility_ema_bps;
```

### Get Current Dynamic Spread
```rust
let spread = self.state_vector.volatility_ema_bps * self.spread_volatility_multiplier;
```

### Override Multiplier at Runtime
```rust
self.spread_volatility_multiplier = 2.0;  // Make more conservative
self.calculate_optimal_control();  // Recalculate with new multiplier
```

### Monitor Spread Adaptation
```rust
info!(
    "Market state: vol={:.2}bps, spread={:.2}bps, bid_offset={:.2}bps, ask_offset={:.2}bps",
    self.state_vector.volatility_ema_bps,
    self.state_vector.volatility_ema_bps * self.spread_volatility_multiplier,
    self.control_vector.bid_offset_bps,
    self.control_vector.ask_offset_bps
);
```

---

## Conclusion

**You already have a world-class dynamic spread system!** 🎉

The only remaining static parameter is `max_bps_diff`, which controls requoting frequency. Implementing the dynamic requote threshold (Section 4) will complete your fully adaptive market making system.

Your strategy will then automatically:
- ✅ Widen spreads in volatile markets (protection)
- ✅ Tighten spreads in calm markets (competitiveness)
- ✅ Adjust requoting frequency to match market conditions
- ✅ Optimize for profitability across all market regimes

**This is exactly what professional market makers do.**
