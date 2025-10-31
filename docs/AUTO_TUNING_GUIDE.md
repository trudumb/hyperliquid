# Multi-Objective Auto-Tuning System

## Overview

The auto-tuning system enables **online parameter optimization** for the HJB market making strategy. Instead of manually tweaking 36+ parameters, the system automatically adapts them in real-time based on trading performance.

## Algorithm: SPSA + Adam

### Why SPSA?

**SPSA (Simultaneous Perturbation Stochastic Approximation)** is perfect for high-dimensional parameter optimization when:
- You have 36 parameters to tune
- Each evaluation is expensive (requires trading live for an episode)
- You want O(1) gradient estimates instead of O(n) finite differences

**Key insight**: Instead of perturbing each parameter individually (36 evaluations), SPSA perturbs **all parameters simultaneously in random directions** (2 evaluations: φ+ and φ-).

### Gradient Estimation

```
∇J(φ) ≈ [J(φ + c·Δ) - J(φ - c·Δ)] / (2c) · Δ⁻¹
```

Where:
- `φ ∈ ℝ³⁶` = unconstrained parameter vector
- `Δ ~ Rademacher({-1, +1}³⁶)` = random binary perturbation
- `c` = perturbation magnitude (decays over time: `c_k = c / k^γ`)
- `J(φ)` = multi-objective performance score

### Adam Optimizer

Once we have the gradient estimate, we use **Adam** (Adaptive Moment Estimation) to update parameters:

```
m_t = β₁·m_{t-1} + (1-β₁)·∇J         # First moment (momentum)
v_t = β₂·v_{t-1} + (1-β₂)·(∇J)²      # Second moment (variance)
φ_{t+1} = φ_t + α·m_t / (√v_t + ε)   # Parameter update
```

**Benefits**:
- Adaptive learning rates (faster for flat dimensions, slower for steep ones)
- Momentum helps escape local minima
- Proven to work well for noisy gradients (perfect for trading!)

## Parameter Space

### Unconstrained Space (φ)

The optimizer works in **unconstrained space** where parameters can be any real number:
```
φ ∈ ℝ³⁶  (no bounds, easy to optimize)
```

### Constrained Space (θ)

Parameters are transformed to **constrained space** using sigmoid functions:
```
θ = θ_min + (θ_max - θ_min) · σ(φ)
where σ(φ) = 1 / (1 + exp(-φ))
```

**Example**: `phi` (risk aversion) must be in [0.001, 0.1]
- If `φ_phi = 0.0`, then `phi = 0.001 + (0.1 - 0.001) · 0.5 ≈ 0.05` (centered)
- If `φ_phi = +5.0`, then `phi ≈ 0.1` (upper bound)
- If `φ_phi = -5.0`, then `phi ≈ 0.001` (lower bound)

This **guarantees** all parameters stay within safe ranges, no matter what the optimizer does!

## Multi-Objective Scoring

The system optimizes a **weighted combination of 4 objectives**:

```
J(φ) = 0.4·Profit + 0.3·Risk + 0.2·Efficiency + 0.1·ModelQuality
```

### 1. Profitability (40% weight)

Measures how much money you're making:
```
Profit = 0.4·PnL + 0.3·Sharpe + 0.2·WinRate + 0.1·ProfitFactor
```

**Metrics**:
- **PnL**: Total profit/loss (normalized by account size)
- **Sharpe Ratio**: Risk-adjusted returns = mean(returns) / std(returns)
- **Win Rate**: Fraction of profitable trades
- **Profit Factor**: Gross profit / gross loss

### 2. Risk Control (30% weight)

Measures how well you're controlling risk:
```
Risk = 0.4·MaxDrawdown + 0.3·InventoryVol + 0.3·MarginUsage
```

**Metrics**:
- **Max Drawdown**: Largest peak-to-trough decline (lower is better)
- **Inventory Volatility**: Standard deviation of position size
- **Margin Usage**: Fraction of available margin being used

### 3. Operational Efficiency (20% weight)

Measures trading quality:
```
Efficiency = 0.4·FillRate + 0.3·ChurnRate + 0.3·SpreadCapture
```

**Metrics**:
- **Fill Rate**: Orders filled / orders placed (higher is better)
- **Churn Rate**: Cancellations / placements (lower is better)
- **Spread Capture**: Realized spread / theoretical spread

### 4. Model Quality (10% weight)

Measures forecasting accuracy:
```
ModelQuality = 0.5·VolError + 0.5·ASError
```

**Metrics**:
- **Volatility Prediction Error**: Forecast error for volatility model
- **AS Prediction Error**: Forecast error for adverse selection model

## 36 Tunable Parameters

### Core HJB Strategy (10 params)

| Parameter | Range | Description |
|-----------|-------|-------------|
| `phi` | [0.001, 0.1] | Risk aversion (inventory penalty) |
| `lambda_base` | [0.1, 10.0] | Base fill intensity |
| `max_position` | [1.0, 10.0] | Maximum absolute position |
| `maker_fee_bps` | [0.0, 5.0] | Maker fee in bps |
| `taker_fee_bps` | [0.0, 10.0] | Taker fee in bps |
| `leverage` | [1, 10] | Account leverage |
| `max_leverage` | [10, 50] | Maximum allowed leverage |
| `margin_safety` | [0.1, 0.5] | Margin safety buffer |
| `enable_multi_level` | {true, false} | Enable multi-level quoting |
| `enable_robust_control` | {true, false} | Enable robust control |

### Multi-Level Config (8 params)

| Parameter | Range | Description |
|-----------|-------|-------------|
| `num_levels` | [1, 5] | Number of order book levels |
| `level_spacing_bps` | [5.0, 50.0] | Spacing between levels |
| `min_spread_bps` | [1.0, 10.0] | Minimum profitable spread |
| `vol_to_spread_factor` | [0.001, 0.02] | Volatility-to-spread scaling |
| `base_maker_size` | [0.1, 5.0] | Base order size (L1) |
| `maker_aggression_decay` | [0.1, 0.9] | Size decay per level |
| `taker_size_multiplier` | [0.1, 1.0] | Taker size vs. maker size |
| `min_taker_rate` | [0.01, 0.5] | Minimum taker rate threshold |

### EWMA Volatility Model (6 params)

| Parameter | Range | Description |
|-----------|-------|-------------|
| `ewma_half_life` | [10.0, 600.0] | Half-life in seconds |
| `ewma_alpha` | [0.01, 0.3] | EWMA smoothing factor |
| `ewma_outlier_thresh` | [2.0, 8.0] | Outlier detection threshold |
| `ewma_min_vol` | [0.1, 2.0] | Minimum volatility (bps) |
| `ewma_max_vol` | [20.0, 100.0] | Maximum volatility (bps) |
| `ewma_tick_freq` | [0.1, 100.0] | Expected tick frequency (Hz) |

### Particle Filter Config (4 params)

| Parameter | Range | Description |
|-----------|-------|-------------|
| `pf_mu` | [-20.0, -10.0] | Mean log-variance |
| `pf_phi` | [0.8, 0.99] | AR(1) persistence |
| `pf_sigma_eta` | [0.1, 2.0] | Volatility of volatility |
| `pf_update_interval` | [5, 50] | Ticks between updates |

### Hybrid Grounding Config (8 params)

| Parameter | Range | Description |
|-----------|-------|-------------|
| `grounding_base` | [0.05, 0.5] | Base grounding strength |
| `grounding_sensitivity` | [0.1, 1.0] | Sensitivity to PF confidence |
| `min_grounding` | [0.01, 0.1] | Minimum grounding |
| `max_grounding` | [0.3, 0.8] | Maximum grounding |
| `ba_tracking_window` | [10, 200] | Bid-ask rate window size |
| `ba_ewma_alpha` | [0.01, 0.3] | Bid-ask rate EWMA alpha |
| `ba_vol_window` | [5, 50] | Bid-ask volatility window |
| `ba_rate_scale` | [0.1, 10.0] | Bid-ask rate scaling |

## Tuning Workflow

### Episode-Based Evaluation

An **episode** = N market updates (L2Book ticks). Typical values:
- `updates_per_episode = 500` (about 1 minute at 10 Hz)
- `episodes_per_update = 2` (2 minutes of data per gradient estimate)

### 3-Phase Cycle

1. **Normal Phase**: Trade using current best parameters
2. **Evaluate φ+**: Trade for 1 episode using `φ + c·Δ` parameters, compute score `J+`
3. **Evaluate φ-**: Trade for 1 episode using `φ - c·Δ` parameters, compute score `J-`
4. **Update**: Compute gradient `∇J ≈ (J+ - J-) / (2c·Δ)`, apply Adam update
5. **Repeat**: Go back to step 1 with new parameters

### Typical Timeline

```
t=0:00    Start with initial parameters (from config.json)
t=0:01    Finish episode 1 (φ+ evaluation) → score J+ = 0.723
t=0:02    Finish episode 2 (φ- evaluation) → score J- = 0.698
t=0:02    Compute gradient, update parameters via Adam
t=0:03    Continue trading with new parameters...
```

Every 2 minutes, parameters are refined!

## Configuration

### Minimal Config (Defaults)

```json
{
  "auto_tuning": {
    "enabled": true,
    "mode": "continuous",
    "episodes_per_update": 2,
    "updates_per_episode": 500
  }
}
```

### Full Control Config

```json
{
  "auto_tuning": {
    "enabled": true,
    "mode": "continuous",

    "episodes_per_update": 2,
    "updates_per_episode": 500,

    "spsa_c": 0.1,
    "spsa_gamma": 0.101,

    "adam_alpha": 0.001,
    "adam_beta1": 0.9,
    "adam_beta2": 0.999,
    "adam_epsilon": 1e-8,
    "learning_rate_decay": 0.0,

    "max_param_change": 1.0,
    "min_episodes_for_gradient": 20,

    "objective_weights": {
      "profitability_weight": 0.4,
      "risk_weight": 0.3,
      "efficiency_weight": 0.2,
      "model_quality_weight": 0.1
    },

    "verbose": true
  }
}
```

## Tuning Modes

### 1. Continuous (Recommended)

Update every N episodes:
```json
{
  "mode": "continuous",
  "episodes_per_update": 2
}
```

**Use case**: Continuous adaptation to market conditions

### 2. Scheduled

Update at fixed time intervals:
```json
{
  "mode": "scheduled",
  "update_interval_seconds": 3600
}
```

**Use case**: Hourly/daily reoptimization (less frequent)

### 3. Adaptive

Update when performance drops:
```json
{
  "mode": "adaptive",
  "adaptive_threshold": 0.7
}
```

**Use case**: React to sudden performance degradation

## Safety Features

### 1. Bounded Parameters

Sigmoid transforms **guarantee** all parameters stay within safe ranges:
- No negative spreads
- No infinite positions
- No crazy leverage

### 2. Gradient Clipping

```
max_param_change = 1.0  # Limit L∞ norm of parameter updates
```

Even if the gradient estimate is noisy, parameter changes are bounded.

### 3. Best Params Tracking

The system **always remembers** the best parameters found:
```rust
tuner.get_best_params()  // Revert if needed
```

### 4. Minimum Data Requirements

```
min_episodes_for_gradient = 20
```

Won't update until sufficient data is collected.

## Monitoring

### Log Messages

```
[TUNER] Starting candidate evaluation
[TUNER] Recorded + score: 0.723456
[TUNER] Recorded - score: 0.698234
[TUNER] SPSA gradient norm: 2.341, score_diff: 0.025222
[TUNER] Updated parameters (iteration 42), learning_rate: 0.001000, avg_score: 0.710845
[TUNER] New best score: 0.723456
[TUNER] Applied new parameters: phi=0.0123, lambda=1.45, max_pos=3.2
```

### Export History

```rust
let history_json = tuner_integration.export_history();
// Save to file for analysis
```

## Tuning the Tuner

### Learning Rates

| Scenario | adam_alpha | spsa_c | Description |
|----------|-----------|--------|-------------|
| **Conservative** | 0.0001 | 0.05 | Slow, stable (large accounts) |
| **Moderate** | 0.001 | 0.1 | Balanced (recommended) |
| **Aggressive** | 0.01 | 0.2 | Fast adaptation (testing only) |

### Objective Weights

**Profit-focused**:
```json
{
  "profitability_weight": 0.6,
  "risk_weight": 0.2,
  "efficiency_weight": 0.15,
  "model_quality_weight": 0.05
}
```

**Risk-focused**:
```json
{
  "profitability_weight": 0.3,
  "risk_weight": 0.5,
  "efficiency_weight": 0.15,
  "model_quality_weight": 0.05
}
```

**Balanced** (default):
```json
{
  "profitability_weight": 0.4,
  "risk_weight": 0.3,
  "efficiency_weight": 0.2,
  "model_quality_weight": 0.1
}
```

## FAQ

### Q: Will this make me rich overnight?

**A**: No. Auto-tuning adapts parameters to recent market conditions, but:
- It can't predict black swan events
- It's only as good as the underlying strategy
- Market regime changes may require manual intervention

### Q: How long until I see improvements?

**A**: Typically 1-4 hours (30-120 parameter updates). Watch the logs for:
```
[TUNER] New best score: 0.756  (improving!)
```

### Q: What if parameters go crazy?

**A**: They can't! Sigmoid transforms guarantee bounds. But if you see weird behavior:
1. Check logs for gradient norms (should be < 10)
2. Reduce learning rate by 10x
3. Revert to `tuner.get_best_params()`

### Q: Can I tune only some parameters?

**A**: Not yet, but coming soon! For now, it's all-or-nothing.

### Q: Does this work with all volatility models?

**A**: Yes! EWMA, Particle Filter, and Hybrid all support auto-tuning.

### Q: How much overhead does this add?

**A**: Minimal. Gradient computation is O(1), Adam update is O(36). Performance tracker adds <1% CPU overhead.

## Best Practices

### 1. Start Conservative

```json
{
  "adam_alpha": 0.0001,
  "spsa_c": 0.05,
  "episodes_per_update": 5
}
```

Watch for 1 hour. If stable, increase learning rates.

### 2. Monitor Closely (First 24h)

Check logs every hour:
- Are scores improving?
- Are parameter changes reasonable?
- Are gradients stable (not exploding)?

### 3. Compare to Baseline

Run two strategies in parallel:
- `HYPE_AUTO_TUNED` (auto-tuning enabled)
- `HYPE_BASELINE` (fixed parameters)

Compare performance after 1 week.

### 4. Use Paper Trading First

**Always test with paper trading before using real funds!**

### 5. Adjust Objective Weights

If you care more about risk than profit, increase `risk_weight`.

### 6. Don't Overtune

If `episodes_per_update = 1`, you're updating too fast (overfitting to noise). Use at least 2.

## Advanced Topics

### SPSA Perturbation Decay

```
c_k = c / k^γ
```

- `γ = 0.101` is standard (Spall 1998)
- Larger γ → faster decay (less exploration over time)
- Smaller γ → slower decay (more exploration)

### Adam Bias Correction

```
m_hat = m_t / (1 - β₁^t)
v_hat = v_t / (1 - β₂^t)
```

Corrects for initialization bias (m₀ = v₀ = 0).

### Learning Rate Schedule

```
α_k = α / (1 + decay · k)
```

Optional learning rate decay. Set `learning_rate_decay = 0.0` to disable.

## References

- Spall, J. C. (1998). "Implementation of the simultaneous perturbation algorithm for stochastic optimization"
- Kingma, D. P., & Ba, J. (2014). "Adam: A method for stochastic optimization"
- Cartea, Á., & Jaimungal, S. (2015). "Risk metrics and fine-tuning of high-frequency trading strategies"

## Support

For issues or questions:
1. Check logs for `[TUNER]` messages
2. Export tuning history: `tuner_integration.export_history()`
3. Open GitHub issue with logs + config

---

**Remember**: Auto-tuning is a powerful tool, but it's not magic. Always monitor performance, start conservative, and test thoroughly before deploying with real capital!
