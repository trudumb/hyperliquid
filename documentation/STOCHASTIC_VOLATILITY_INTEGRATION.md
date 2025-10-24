# Stochastic Volatility Particle Filter Integration Guide

## Overview

The Particle Filter provides a sophisticated method for estimating latent volatility in real-time market making. Unlike simple EMA-based volatility estimates, the Particle Filter:

- Models volatility as a stochastic process with mean reversion
- Adapts to changing market conditions automatically
- Provides uncertainty quantification (confidence intervals)
- Is robust to outliers and measurement noise

## Model Specification

The filter implements a standard Stochastic Volatility (SV) model:

### State Equation (Log-Variance Evolution)
```
h_t = μ + φ(h_{t-1} - μ) + η_t
```
where:
- `h_t = log(σ_t²)` is the log-variance at time t
- `μ` is the long-term mean of log-variance
- `φ ∈ (0,1)` is the persistence parameter (controls mean reversion speed)
- `η_t ~ N(0, σ_η² * dt)` is the process noise

### Measurement Equation (Observed Returns)
```
y_t = √(exp(h_t) * dt) * ε_t
```
where:
- `y_t = ln(P_t / P_{t-1})` is the log return
- `ε_t ~ N(0,1)` is the observation noise
- `dt` is the time interval between observations (in years)

## Integration into MarketMaker

### 1. Add ParticleFilterState to MarketMaker Struct

```rust
use crate::stochastic_volatility::ParticleFilterState;

pub struct MarketMaker {
    // ... existing fields ...
    
    /// Particle filter for stochastic volatility estimation
    pub particle_filter: Arc<RwLock<ParticleFilterState>>,
}
```

### 2. Initialize the Filter

```rust
impl MarketMaker {
    pub fn new(input: MarketMakerInput) -> Result<Self, Error> {
        // ... existing initialization ...
        
        // Initialize particle filter with sensible defaults
        let particle_filter = Arc::new(RwLock::new(ParticleFilterState::new(
            5000,           // num_particles: More = better accuracy but slower
            -9.2,           // mu: ln((0.01)²) ≈ 100 bps annualized vol
            0.98,           // phi: High persistence (slow mean reversion)
            0.15,           // sigma_eta: Volatility of volatility
            -9.2,           // initial_h: Start at long-term mean
            0.5,            // initial_h_std_dev: Moderate initial uncertainty
            42,             // seed: For reproducibility (or use random)
        )));
        
        Ok(Self {
            // ... existing fields ...
            particle_filter,
        })
    }
}
```

### 3. Update Filter on Each AllMids Message

Replace the simple EMA volatility update with the particle filter:

```rust
// In your WebSocket AllMids handler:
Message::AllMids(all_mids) => {
    // ... existing code to extract mid_price ...
    
    // Update particle filter and get new volatility estimate
    let mut pf = self.particle_filter.write().unwrap();
    if let Some(vol_estimate_bps) = pf.update(mid_price) {
        // Update state vector with filter estimate
        self.state_vector.volatility_ema_bps = vol_estimate_bps;
        
        // Optional: Log filter statistics
        info!(
            "Volatility: {:.2} bps (ESS: {:.0}/{}, P5: {:.2}, P95: {:.2})",
            vol_estimate_bps,
            pf.get_ess(),
            pf.particles.len(),
            pf.estimate_volatility_percentile_bps(0.05),
            pf.estimate_volatility_percentile_bps(0.95)
        );
    }
    drop(pf); // Release lock
    
    // ... continue with existing logic ...
}
```

### 4. (Optional) Use Confidence Intervals for Risk Management

The particle filter provides uncertainty quantification:

```rust
let pf = self.particle_filter.read().unwrap();

// Get volatility estimates at different confidence levels
let vol_p5 = pf.estimate_volatility_percentile_bps(0.05);   // 5th percentile (conservative)
let vol_p50 = pf.estimate_volatility_percentile_bps(0.50);  // Median
let vol_mean = pf.estimate_volatility_bps();                 // Mean estimate
let vol_p95 = pf.estimate_volatility_percentile_bps(0.95);  // 95th percentile (aggressive)

// Use conservative estimate for spread calculation during uncertain periods
let spread_volatility_estimate = if pf.get_ess() < 2500.0 {
    vol_p95  // High uncertainty -> use conservative (wide) estimate
} else {
    vol_mean // Normal operation -> use mean estimate
};

let base_spread_bps = spread_volatility_estimate * self.spread_volatility_multiplier;
```

## Parameter Tuning Guide

### Recommended Starting Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `num_particles` | 5000 | 1000-10000 | More particles = smoother estimates, slower performance |
| `mu` | -9.2 | -15 to -7 | ln(σ²) for typical volatility (100 bps ≈ -9.2) |
| `phi` | 0.98 | 0.90-0.99 | Higher = slower mean reversion (more persistent vol) |
| `sigma_eta` | 0.15 | 0.05-0.30 | Higher = faster vol changes (more "vol of vol") |
| `initial_h` | -9.2 | Same as μ | Starting log-variance guess |
| `initial_h_std_dev` | 0.5 | 0.2-1.0 | Initial uncertainty (higher = less confident start) |

### Calibration Tips

1. **For high-frequency assets (e.g., BTC):**
   - Increase `phi` to 0.99 (volatility persists longer)
   - Decrease `sigma_eta` to 0.10 (smoother vol evolution)

2. **For low-frequency/illiquid assets:**
   - Decrease `phi` to 0.95 (faster mean reversion)
   - Increase `sigma_eta` to 0.25 (allow rapid vol changes)

3. **For trending markets:**
   - Increase `sigma_eta` to 0.20+ (capture regime shifts faster)

4. **Monitor ESS (Effective Sample Size):**
   - ESS should stay above 50% of num_particles most of the time
   - Frequent resampling (ESS drops) indicates model mismatch
   - If ESS < 1000 consistently, increase `num_particles` or adjust `sigma_eta`

## Performance Considerations

### Computational Cost

- **Update cost:** O(N) where N = num_particles
- **Typical timing:** ~1ms for 5000 particles on modern CPU
- **Bottlenecks:** Resampling (only when ESS < N/2)

### Optimization Strategies

1. **Reduce particle count:** 1000-2000 particles often sufficient for real-time trading
2. **Sample updates:** Update filter every K ticks instead of every tick:
   ```rust
   if tick_counter % 10 == 0 {
       particle_filter.update(mid_price);
   }
   ```
3. **Background thread:** Run filter updates asynchronously:
   ```rust
   let pf_handle = tokio::spawn(async move {
       particle_filter.update(mid_price)
   });
   ```

## Comparison with EMA Method

| Aspect | EMA Method | Particle Filter |
|--------|-----------|----------------|
| Computational Cost | O(1) | O(N) particles |
| Latency | Instant | ~1ms |
| Accuracy | Moderate | High |
| Adaptability | Fixed decay | Auto-adapts via φ |
| Uncertainty | No | Yes (percentiles) |
| Outlier Robustness | Poor | Good |
| Regime Changes | Slow | Fast (with tuned σ_η) |

**Recommendation:** Use Particle Filter for production market making where accurate volatility estimation is critical. Fall back to EMA only if latency constraints are extreme (<100μs).

## Diagnostics and Monitoring

### Key Metrics to Track

1. **Effective Sample Size (ESS):**
   ```rust
   let ess = particle_filter.get_ess();
   if ess < num_particles * 0.3 {
       warn!("Particle degeneracy detected: ESS = {}", ess);
   }
   ```

2. **Volatility Confidence Interval Width:**
   ```rust
   let vol_iqr = vol_p95 - vol_p5;  // Interquartile range
   if vol_iqr > 100.0 {
       warn!("High volatility uncertainty: IQR = {:.2} bps", vol_iqr);
   }
   ```

3. **Resampling Frequency:**
   - Track how often resampling triggers
   - Too frequent (>50% of updates) suggests model mismatch

### Logging Template

```rust
info!(
    "SV Filter: vol={:.2}bps, ESS={:.0}/{} ({:.1}%), CI=[{:.2}, {:.2}]",
    vol_mean,
    ess,
    num_particles,
    100.0 * ess / (num_particles as f64),
    vol_p5,
    vol_p95
);
```

## Advanced: Multi-Asset Filters

For market makers trading multiple assets, maintain separate filters:

```rust
pub struct MultiAssetMarketMaker {
    particle_filters: HashMap<String, Arc<RwLock<ParticleFilterState>>>,
}

impl MultiAssetMarketMaker {
    pub fn update_volatility(&self, asset: &str, mid_price: f64) {
        if let Some(pf) = self.particle_filters.get(asset) {
            let mut filter = pf.write().unwrap();
            if let Some(vol) = filter.update(mid_price) {
                // Update asset-specific state
            }
        }
    }
}
```

## Testing Your Integration

### Unit Test Template

```rust
#[test]
fn test_particle_filter_integration() {
    let mut mm = create_test_market_maker();
    
    // Simulate price series with known volatility
    let true_vol = 100.0; // 100 bps
    let prices = generate_geometric_brownian_motion(100.0, true_vol, 1000);
    
    for price in prices {
        mm.particle_filter.write().unwrap().update(price);
    }
    
    let estimated_vol = mm.state_vector.volatility_ema_bps;
    assert!((estimated_vol - true_vol).abs() < 20.0, 
            "Estimated vol should be within 20bps of true vol");
}
```

## Troubleshooting

### Problem: Filter estimates are too smooth/lagging

**Solution:** Increase `sigma_eta` to allow faster volatility changes

### Problem: ESS drops to near-zero frequently

**Solution:** 
1. Increase `num_particles` to 10000+
2. Check for outliers in price feed (filter out bad ticks)
3. Adjust `sigma_eta` (try both increasing and decreasing)

### Problem: Estimates are unstable/noisy

**Solution:** 
1. Decrease `sigma_eta` for smoother evolution
2. Increase `phi` for more persistent volatility
3. Apply post-filter EMA smoothing:
   ```rust
   let smoothed_vol = 0.9 * prev_vol + 0.1 * particle_vol;
   ```

### Problem: Performance issues (>5ms per update)

**Solution:**
1. Reduce `num_particles` to 2000 or less
2. Update filter less frequently (every 5-10 ticks)
3. Profile resampling - if triggering too often, adjust threshold

## References

- Original Paper: Kim, S., Shephard, N., & Chib, S. (1998). "Stochastic Volatility: Likelihood Inference and Comparison with ARCH Models"
- Particle Filtering: Doucet, A., & Johansen, A. M. (2009). "A Tutorial on Particle Filtering and Smoothing"
- Market Making Application: Cartea, Á., Jaimungal, S., & Penalva, J. (2015). "Algorithmic and High-Frequency Trading"

## Next Steps

After successful integration:

1. Backtest on historical data to validate estimates match realized volatility
2. Compare P&L using Particle Filter vs EMA volatility
3. Experiment with adaptive parameter tuning (μ, φ, σ_η) based on market regime
4. Consider multi-factor models (e.g., adding jump components)
