# Stochastic Volatility Particle Filter - Quick Reference

## Initialization

```rust
use hyperliquid_rust_sdk::ParticleFilterState;

let filter = ParticleFilterState::new(
    5000,  // num_particles (1000-10000)
    -9.2,  // mu: ln((0.01)²) ≈ 100bps vol
    0.98,  // phi: persistence (0.90-0.99)
    0.15,  // sigma_eta: vol of vol (0.05-0.30)
    -9.2,  // initial_h: starting guess
    0.5,   // initial_h_std_dev: uncertainty
    42,    // seed: for reproducibility
);
```

## Update Loop

```rust
// On each price update (e.g., AllMids message)
if let Some(vol_bps) = filter.update(mid_price) {
    state_vector.volatility_ema_bps = vol_bps;
}
```

## Get Estimates

```rust
// Mean volatility (most common)
let vol = filter.estimate_volatility_bps();

// Confidence intervals
let p5 = filter.estimate_volatility_percentile_bps(0.05);   // Conservative
let p50 = filter.estimate_volatility_percentile_bps(0.50);  // Median
let p95 = filter.estimate_volatility_percentile_bps(0.95);  // Aggressive

// Filter health
let ess = filter.get_ess();  // Should be > num_particles/2
```

## Parameter Cheat Sheet

| Parameter | Low | Medium | High | Purpose |
|-----------|-----|--------|------|---------|
| `num_particles` | 1000 | 5000 | 10000 | Accuracy vs. speed |
| `mu` | -15.0 | -9.2 | -7.0 | 1bps / 100bps / 1000bps expected vol |
| `phi` | 0.90 | 0.98 | 0.99 | Fast / Med / Slow mean reversion |
| `sigma_eta` | 0.05 | 0.15 | 0.30 | Stable / Normal / Volatile markets |

## Common Patterns

### Pattern 1: Basic Integration
```rust
pub struct MarketMaker {
    particle_filter: Arc<RwLock<ParticleFilterState>>,
}

// In AllMids handler:
let mut pf = self.particle_filter.write().unwrap();
if let Some(vol) = pf.update(mid_price) {
    self.state_vector.volatility_ema_bps = vol;
}
```

### Pattern 2: Conservative Spread During Uncertainty
```rust
let vol_estimate = if filter.get_ess() < 2500.0 {
    filter.estimate_volatility_percentile_bps(0.95)  // Wide
} else {
    filter.estimate_volatility_bps()  // Normal
};
```

### Pattern 3: Multi-Asset
```rust
pub struct MultiAssetMM {
    filters: HashMap<String, Arc<RwLock<ParticleFilterState>>>,
}
```

## Monitoring

```rust
info!(
    "Vol: {:.2}bps, ESS: {:.0}/{} ({:.1}%), CI=[{:.2}, {:.2}]",
    filter.estimate_volatility_bps(),
    filter.get_ess(),
    filter.particles.len(),
    100.0 * filter.get_ess() / (filter.particles.len() as f64),
    filter.estimate_volatility_percentile_bps(0.05),
    filter.estimate_volatility_percentile_bps(0.95)
);
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| ESS drops to near-zero | Increase `num_particles` or adjust `sigma_eta` |
| Estimates lag market | Increase `sigma_eta` (faster adaptation) |
| Too noisy | Decrease `sigma_eta`, increase `phi` |
| Slow performance | Reduce `num_particles` to 1000-2000 |

## Quick Calibration

```rust
// High-frequency crypto (BTC, ETH)
ParticleFilterState::new(5000, -9.2, 0.99, 0.10, -9.2, 0.5, seed)

// Mid-frequency altcoins
ParticleFilterState::new(5000, -9.2, 0.98, 0.15, -9.2, 0.5, seed)

// Trending/volatile markets
ParticleFilterState::new(5000, -9.2, 0.95, 0.25, -9.2, 0.5, seed)
```

## Tests

```bash
# Run all tests
cargo test --lib stochastic_volatility -- --nocapture

# Build release
cargo build --release
```

## Files

- **Implementation:** `src/stochastic_volatility.rs`
- **Integration Guide:** `documentation/STOCHASTIC_VOLATILITY_INTEGRATION.md`
- **Summary:** `documentation/STOCHASTIC_VOLATILITY_IMPLEMENTATION_SUMMARY.md`
- **Quick Ref:** `documentation/STOCHASTIC_VOLATILITY_QUICKREF.md` (this file)
