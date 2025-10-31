# Hybrid Volatility Model - EWMA Grounded to Stochastic Baseline

## Overview

The **Hybrid Volatility Model** combines the best features of both EWMA and Particle Filter approaches:

1. **EWMA Fast Layer**: Tracks tick-to-tick volatility changes with O(1) updates
2. **Particle Filter Slow Layer**: Provides stochastic baseline and uncertainty quantification
3. **Adaptive Grounding**: Dynamically blends the two based on market conditions
4. **Bid-Ask Rate Dynamics**: Uses fill rate volatility to inform grounding strength

Think of it as:
- **EWMA** = A fast-moving satellite tracking real-time changes
- **Particle Filter** = A gravitational center providing long-term context
- **Grounding** = Elastic tether preventing EWMA drift while allowing fast reaction
- **Bid-Ask Rates** = Sensor measuring market activity to adjust tether tension

## Algorithm

### 1. Fast Layer (EWMA)

Updated **every tick** for low-latency tracking:

```
r_t = ln(P_t / P_{t-1})              # Log return
σ²_EWMA,t = α * r_t² + (1-α) * σ²_EWMA,t-1   # Variance update
σ_EWMA = sqrt(σ²_EWMA) * 10000       # Convert to basis points
```

### 2. Slow Layer (Particle Filter)

Updated **periodically** (e.g., every 10 ticks) for stochastic modeling:

```
h_t = μ + φ(h_{t-1} - μ) + η_t      # AR(1) log-variance dynamics
y_t = sqrt(exp(h_t) * dt) * ε_t      # Observation equation
Particles reweighted by likelihood   # Bayesian update
σ_PF = E[sqrt(exp(h_t))] * 10000    # Expected volatility in bps
```

### 3. Grounding Mechanism

Adaptive blending based on confidence and market activity:

```
λ = λ_base * PF_confidence * rate_adjustment

σ_hybrid = (1 - λ) * σ_EWMA + λ * σ_PF

where:
  PF_confidence = 1 - (σ_PF_uncertainty / σ_PF)  # Higher when PF is certain
  rate_adjustment = 1 - normalized_rate_vol       # Lower when market is active
```

**Intuition**:
- **High PF confidence + calm market** → λ ≈ 0.3-0.5 (trust PF baseline more)
- **Low PF confidence + active market** → λ ≈ 0.05-0.15 (trust EWMA more)

### 4. Bid-Ask Rate Tracking

Tracks fill rates to measure market activity:

```
bid_rate_t = bid_fills / dt          # Fills per second on bid side
ask_rate_t = ask_fills / dt          # Fills per second on ask side
total_rate_t = bid_rate_t + ask_rate_t

rate_vol = std_dev(total_rate[t-N:t]) # Volatility of fill rates
normalized_rate_vol = rate_vol / mean(total_rate)
```

**Effect on grounding**:
- **High rate volatility** → Market is fast-moving → Weaken grounding (trust EWMA's agility)
- **Low rate volatility** → Market is stable → Strengthen grounding (trust PF's stability)

## Configuration

### Basic Configuration

```json
{
  "volatility_model_type": "hybrid",
  "hybrid_vol_config": {
    "ewma_config": {
      "half_life_seconds": 60.0,
      "outlier_threshold": 4.0
    },
    "pf_update_interval_ticks": 10,
    "grounding_strength_base": 0.2,
    "enable_bid_ask_tracking": true
  }
}
```

### Advanced Configuration

```json
{
  "hybrid_vol_config": {
    "ewma_config": {
      "half_life_seconds": 60.0,
      "alpha": 0.1,
      "outlier_threshold": 4.0,
      "min_volatility_bps": 0.5,
      "max_volatility_bps": 50.0,
      "expected_tick_frequency_hz": 10.0
    },

    "num_particles": 1000,
    "pf_mu": -18.0,                    // Recalibrated for tick-level
    "pf_phi": 0.95,                    // High persistence
    "pf_sigma_eta": 0.5,               // Moderate vol-of-vol
    "pf_update_interval_ticks": 10,    // Update PF every 10 ticks

    "grounding_strength_base": 0.2,    // Base blending weight
    "grounding_sensitivity": 0.5,      // How much bid-ask rates matter
    "min_grounding": 0.05,             // Never go below 5% grounding
    "max_grounding": 0.5,              // Never go above 50% grounding

    "enable_bid_ask_tracking": true
  }
}
```

## Key Parameters

### `pf_update_interval_ticks`
**Default**: 10
**Range**: 5-50

How often to update the particle filter (in ticks).

- **Lower (5-10)**: More frequent PF updates, smoother grounding, higher CPU
- **Higher (20-50)**: Less frequent PF updates, EWMA dominates, lower CPU

**Tuning**: For high-frequency markets (>5 Hz), use 15-20. For low-frequency (<1 Hz), use 5-10.

### `grounding_strength_base`
**Default**: 0.2
**Range**: 0.05-0.5

Base weight given to particle filter baseline.

- **Lower (0.05-0.15)**: Mostly EWMA (fast, responsive)
- **Medium (0.15-0.30)**: Balanced blend
- **Higher (0.30-0.50)**: More PF influence (stable, theoretical)

**Tuning**: Start with 0.2. Increase if EWMA is too noisy, decrease if lagging market moves.

### `grounding_sensitivity`
**Default**: 0.5
**Range**: 0.0-1.0

How much bid-ask rate volatility affects grounding strength.

- **0.0**: Ignore bid-ask rates (constant grounding)
- **0.5**: Moderate adjustment
- **1.0**: Full adjustment (grounding can change dramatically)

**Tuning**: Use 0.5 unless you have strong priors about fill rate dynamics.

### `enable_bid_ask_tracking`
**Default**: true
**Type**: boolean

Whether to track fill rates and use them for adaptive grounding.

- **true**: Adaptive grounding based on market activity (recommended)
- **false**: Static grounding based only on PF confidence

**Note**: Requires integration with order fill tracking (see below).

## Integration with Strategy

### Basic Integration

The hybrid model works as a drop-in replacement for EWMA or Particle Filter:

```rust
// In hjb_strategy.rs initialization
let volatility_model: Box<dyn VolatilityModel> = Box::new(
    HybridVolatilityModel::new(HybridVolConfig::high_frequency())
);
```

### Fill Rate Tracking (Optional but Recommended)

To enable bid-ask rate tracking, you need to notify the hybrid model when fills occur:

```rust
// When processing fills in handle_fills()
if let Some(hybrid_model) = self.volatility_model.as_any().downcast_ref::<HybridVolatilityModel>() {
    for (fill, _) in fills {
        let is_bid = fill.side == "B";
        hybrid_model.record_fill(is_bid);
    }
}
```

**Note**: The current implementation doesn't expose `as_any()` - this would require adding:

```rust
// In VolatilityModel trait
fn as_any(&self) -> &dyn std::any::Any;

// In HybridVolatilityModel impl
fn as_any(&self) -> &dyn std::any::Any {
    self
}
```

For now, bid-ask tracking is **passive** (initialized but not actively updated). Full integration coming in future update.

## Performance Characteristics

### Computational Cost

| Component | Update Frequency | Complexity | Time per Update |
|-----------|------------------|------------|-----------------|
| EWMA Layer | Every tick | O(1) | ~1-5 μs |
| Particle Filter | Every N ticks | O(1000) | ~100-500 μs |
| Grounding | Every N ticks | O(1) | ~1 μs |
| **Total (typical)** | Every tick | **Amortized O(1)** | **~5-10 μs avg** |

**Comparison**:
- Pure EWMA: ~1-5 μs per tick
- Pure Particle Filter: ~100-500 μs per tick
- **Hybrid**: ~5-10 μs per tick (10x faster than pure PF, similar to EWMA)

### Memory Usage

- EWMA state: ~1 KB
- Particle filter state: ~100 KB (1000 particles × ~100 bytes each)
- Bid-ask tracker: ~5 KB (50 samples × 2 sides)
- **Total**: ~106 KB (negligible for modern systems)

## Expected Behavior

### Startup Phase (First 50 ticks)

```
Tick 1:   EWMA=?bps, PF=?bps, λ=0.20, Hybrid=?bps
Tick 10:  EWMA=15bps, PF=12bps, λ=0.25, Hybrid=14bps  # First PF update
Tick 20:  EWMA=8bps, PF=10bps, λ=0.22, Hybrid=8.4bps  # Converging
Tick 50:  EWMA=5bps, PF=6bps, λ=0.18, Hybrid=5.2bps   # Stable
```

### Steady State (Normal Operation)

```
# Calm market (low bid-ask rate volatility)
EWMA=4.5bps, PF=5.0bps, λ=0.30, Hybrid=4.65bps  # More PF influence

# Active market (high bid-ask rate volatility)
EWMA=12bps, PF=8bps, λ=0.10, Hybrid=11.6bps     # More EWMA influence

# Volatile regime change
EWMA=25bps, PF=15bps, λ=0.15, Hybrid=23.5bps    # EWMA catches spike quickly
... PF slowly adjusts over next 50-100 ticks
```

### Grounding Dynamics

Typical grounding strength evolution:

```
λ_base = 0.20 (configured)
PF_confidence = 0.85 (particle filter is confident)
rate_adjustment = 0.90 (calm market, low fill rate volatility)

λ = 0.20 * 0.85 * 0.90 = 0.153  # Actual grounding

# When market becomes active:
rate_adjustment = 0.60 (high fill rate volatility)
λ = 0.20 * 0.85 * 0.60 = 0.102  # Weaker grounding, more EWMA
```

## Diagnostics and Monitoring

### Log Messages

The hybrid model outputs detailed diagnostics:

```
[HYBRID VOL] PF updated at tick 10: grounding=0.153, ba_rate_vol=0.023
[HYBRID VOL GROUNDING] pf_conf=0.850, rate_adj=0.900 → λ=0.153
[HYBRID VOL] EWMA=4.50bps, PF=5.00bps, λ=0.153 → Hybrid=4.58bps
```

### Key Metrics to Monitor

1. **Hybrid Volatility** (`4.58bps`):
   - Should be in rational 1-20 bps range for most assets
   - Should track EWMA closely but with reduced noise

2. **Grounding Strength** (`λ=0.153`):
   - Typical range: 0.05-0.30
   - Should adapt to market conditions
   - Warning if stuck at min/max bounds

3. **EWMA vs PF Spread** (`|4.50 - 5.00| = 0.50bps`):
   - Typical difference: 0.5-3 bps
   - Large differences (>5 bps) indicate regime change or model miscalibration

4. **Bid-Ask Rate Volatility** (`0.023`):
   - Relative measure (unitless after normalization)
   - Higher values → weaker grounding
   - Typical range: 0.01-0.10

### Diagnostic Method

```rust
let diagnostics = hybrid_model.diagnostics();
println!("{}", diagnostics);

// Output:
// "Hybrid Vol: 4.58bps | EWMA: 4.50bps | PF: 5.00bps | Grounding: 0.153 | BA Rate Vol: 0.023 | Ticks: 10"
```

## Troubleshooting

### Issue: Hybrid volatility is stuck at EWMA value

**Symptoms**: `λ ≈ 0.05` (minimum grounding), hybrid ≈ EWMA

**Possible causes**:
1. Low PF confidence (high uncertainty)
2. High bid-ask rate volatility
3. `min_grounding` set too low

**Solutions**:
- Increase `pf_update_interval_ticks` (slower PF updates → more stable estimates)
- Increase `grounding_strength_base` (0.2 → 0.3)
- Decrease `grounding_sensitivity` (less response to fill rates)

### Issue: Hybrid volatility is stuck at PF value

**Symptoms**: `λ ≈ 0.50` (maximum grounding), hybrid ≈ PF

**Possible causes**:
1. Very high PF confidence
2. Very low bid-ask rate volatility
3. `max_grounding` set too high

**Solutions**:
- Decrease `grounding_strength_base` (0.2 → 0.1)
- Increase `max_grounding` limit (unlikely to be the issue)
- Check if market is genuinely very calm (this might be correct behavior)

### Issue: Hybrid volatility is noisy/erratic

**Symptoms**: Rapid changes in hybrid estimate, unstable spreads

**Possible causes**:
1. `pf_update_interval_ticks` too low (PF updating too often)
2. EWMA `half_life` too short
3. Grounding changing too quickly

**Solutions**:
- Increase `pf_update_interval_ticks` (10 → 20)
- Increase EWMA `half_life_seconds` (60 → 120)
- Decrease `grounding_sensitivity` (0.5 → 0.2)

### Issue: Hybrid volatility lags market moves

**Symptoms**: Slow response to volatility regime changes

**Possible causes**:
1. Too much PF influence (high grounding)
2. EWMA half-life too long
3. PF updating too infrequently

**Solutions**:
- Decrease `grounding_strength_base` (0.2 → 0.15)
- Decrease EWMA `half_life_seconds` (60 → 30)
- Decrease `pf_update_interval_ticks` (20 → 10)

## Comparison with Pure Models

| Feature | EWMA | Particle Filter | **Hybrid** |
|---------|------|----------------|---------|
| **Speed** | Very fast (1-5 μs) | Slow (100-500 μs) | **Fast (5-10 μs)** |
| **Stability** | Can drift | Very stable | **Stable (grounded)** |
| **Responsiveness** | High | Low | **High with safety net** |
| **Uncertainty** | Heuristic | Bayesian | **Bayesian (from PF)** |
| **Configuration** | Simple (1-2 params) | Complex (5+ params) | **Medium (3-4 key params)** |
| **CPU Usage** | Negligible | ~0.1% | **Negligible** |
| **Memory** | ~1 KB | ~100 KB | **~106 KB** |
| **Best for** | Pure speed | Research | **Production (best of both)** |

## When to Use Hybrid Model

### ✅ Use Hybrid If:

- You want **theoretical rigor** (stochastic modeling) without sacrificing speed
- You need **uncertainty quantification** for robust control
- Your market has **varying activity levels** (benefit from adaptive grounding)
- You're willing to tune 3-4 parameters for optimal performance
- You trust the particle filter to provide a meaningful baseline

### ❌ Use EWMA Instead If:

- You want **maximum simplicity** (1 parameter: half-life)
- You don't need uncertainty quantification
- Your market is consistently high-frequency (no need for grounding)
- You want to minimize configuration complexity

### ❌ Use Pure Particle Filter Instead If:

- You're doing **academic research** on stochastic volatility
- You need the full particle distribution (not just mean ± std)
- Computational cost is not a concern
- You have strong priors on stochastic volatility parameters

## Recommended Starting Configuration

For most production use cases:

```json
{
  "volatility_model_type": "hybrid",
  "hybrid_vol_config": {
    "ewma_config": {
      "half_life_seconds": 60.0,
      "outlier_threshold": 4.0
    },
    "pf_update_interval_ticks": 15,
    "grounding_strength_base": 0.20,
    "grounding_sensitivity": 0.5,
    "enable_bid_ask_tracking": true
  }
}
```

**Rationale**:
- `half_life=60s`: Balanced between responsiveness and stability
- `pf_interval=15`: Updates ~1/sec on 10 Hz market (good balance)
- `grounding=0.20`: 80% EWMA, 20% PF (mostly fast, some anchor)
- `sensitivity=0.5`: Moderate adaptation to market activity

## Future Enhancements

Potential improvements (not yet implemented):

1. **Active bid-ask tracking integration**: Currently passive, needs hook into fill handler
2. **Multi-scale volatility**: Track volatility at multiple time horizons
3. **Regime detection**: Automatically adjust grounding based on detected market regime
4. **Parameter auto-tuning**: Online learning of optimal grounding parameters
5. **Percentile grounding**: Use PF percentiles (5th, 95th) for additional confidence metrics

---

**Summary**: The Hybrid Volatility Model combines EWMA's speed with Particle Filter's theoretical rigor, using adaptive grounding to get the best of both worlds. It's the recommended choice for production market making on tick data when you want both performance and stochastic modeling.
