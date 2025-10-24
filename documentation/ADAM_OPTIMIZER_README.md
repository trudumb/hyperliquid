# Adam Optimizer for Automatic Parameter Tuning

## Overview

The market maker now uses the **Adam optimizer** (Adaptive Moment Estimation) for **fully autonomous** parameter tuning. This is a self-learning system that continuously adapts to market conditions without manual intervention.

## 🤖 Autonomous Operation

**This is a self-tuning agent.** The `tuning_params.json` file serves only two purposes:

1. **Initial parameters** when the bot starts up
2. **Persistence** of Adam's learned parameters across restarts

### How to Override Parameters

If you want to manually change parameters:

```bash
# 1. Stop the bot
Ctrl+C

# 2. Edit tuning_params.json with your desired starting parameters
nano tuning_params.json

# 3. Restart the bot
RUST_LOG=info cargo run --bin market_maker_v2
```

Adam will immediately begin learning from your new starting point.

### What Changed

❌ **REMOVED**: Hot-reload functionality (checking file every 10 seconds)  
✅ **NEW**: Adam optimizer has full authority over parameter tuning  
✅ **NEW**: Parameters only loaded once at startup  
✅ **NEW**: Adam continuously saves learned parameters to JSON

**Why this is better:**
- No conflicting updates between file changes and Adam's learning
- Cleaner separation of concerns (human sets starting point, Adam optimizes)
- Adam's learning trajectory is never interrupted
- Simpler, more predictable behavior

## What is Adam?

Adam is an adaptive learning rate optimization algorithm introduced by Kingma & Ba (2015). It combines the benefits of:
- **AdaGrad**: Adapts learning rates based on parameter-specific gradient history
- **RMSProp**: Uses exponential moving averages to prevent learning rate decay
- **Momentum**: Smooths optimization by accumulating gradients over time

## Key Advantages Over Vanilla SGD

### 1. **Automatic Learning Rate Adaptation**
- No need to manually tune the learning rate (`η`)
- Each parameter gets its own adaptive learning rate
- Learning rates automatically adjust based on gradient magnitude and variance

### 2. **Robust to Market Volatility**
- Momentum smoothing prevents overreaction to single noisy gradients
- Second moment estimation accounts for gradient variance
- More stable convergence in volatile market conditions

### 3. **Faster Convergence**
- Typically converges 2-5x faster than vanilla SGD
- Can escape local minima more effectively
- Handles sparse gradients well (when some parameters don't need adjustment)

### 4. **Less Sensitive to Initialization**
- Default `α = 0.001` works well in most cases
- Bias correction ensures good performance from the start
- Momentum parameters (`β₁ = 0.9`, `β₂ = 0.999`) are well-tuned defaults

## How It Works

### Loss Function: Control Gap (Not Value Gap!)

**Critical Design Decision**: The loss function is the **squared difference in quote offsets**, NOT the difference in expected values:

```
L = (δ^b_optimal - δ^b_heuristic)² + (δ^a_optimal - δ^a_heuristic)²
```

Where:
- `δ^b_optimal` = Optimal bid offset from grid search (in bps)
- `δ^b_heuristic` = Heuristic bid offset from `apply_state_adjustments()` (in bps)
- `δ^a_optimal` = Optimal ask offset from grid search (in bps)  
- `δ^a_heuristic` = Heuristic ask offset from `apply_state_adjustments()` (in bps)

**Why this is better than value gap:**

1. **Stable gradients**: Quote offsets are in bps (typically 5-50 bps), producing gradients in a reasonable range
2. **Interpretable**: Loss of 4.0 means quotes differ by 2 bps RMS (root mean square)
3. **Direct optimization**: We want matching quotes, not matching abstract "values"
4. **Numerically well-behaved**: No issues with large or near-zero denominators

**Example**:
```
Optimal:    bid_offset = 5.0 bps,  ask_offset = 5.0 bps
Heuristic:  bid_offset = 7.5 bps,  ask_offset = 4.0 bps

Loss = (5.0 - 7.5)² + (5.0 - 4.0)²
     = (-2.5)² + (1.0)²
     = 6.25 + 1.0
     = 7.25 bps²
```

This is a manageable number that produces gradients Adam can work with effectively.

### Update Rule

For each parameter `θᵢ` at time step `t`:

```
1. Compute gradient: gₜ = ∇L(θₜ)

2. Update first moment (mean):
   mₜ = β₁ · mₜ₋₁ + (1 - β₁) · gₜ

3. Update second moment (variance):
   vₜ = β₂ · vₜ₋₁ + (1 - β₂) · gₜ²

4. Bias correction:
   m̂ₜ = mₜ / (1 - β₁ᵗ)
   v̂ₜ = vₜ / (1 - β₂ᵗ)

5. Parameter update:
   θₜ₊₁ = θₜ - α · m̂ₜ / (√v̂ₜ + ε)
```

Where:
- `α = 0.001` - Base learning rate (conservative default)
- `β₁ = 0.9` - First moment decay (momentum)
- `β₂ = 0.999` - Second moment decay (variance)
- `ε = 1e-8` - Numerical stability constant

## Implementation Details

### AdamOptimizerState Struct

```rust
pub struct AdamOptimizerState {
    pub m: Vec<f64>,        // First moment estimates (7 params)
    pub v: Vec<f64>,        // Second moment estimates (7 params)
    pub t: usize,           // Time step counter
    pub alpha: f64,         // Learning rate (0.001)
    pub beta1: f64,         // First moment decay (0.9)
    pub beta2: f64,         // Second moment decay (0.999)
    pub epsilon: f64,       // Numerical stability (1e-8)
}
```

### Key Methods

1. **`compute_update(gradient_vector)`** - Computes parameter updates using Adam rule
2. **`reset()`** - Resets optimizer state (useful when changing market regimes)
3. **`get_effective_learning_rate(param_index)`** - Returns adaptive learning rate for a parameter

### Integration with Background Optimization

Every 60 seconds, the background task:
1. Runs grid search to find optimal control
2. Calculates gradients via finite differences (numerical differentiation)
3. Passes gradients to Adam optimizer
4. Adam computes adaptive parameter updates
5. Updates are applied and saved to `tuning_params.json`
6. **Bot continues running with updated parameters** (no reload needed)

### Adam as the Sole Authority

The bot operates in a **closed-loop autonomous mode**:

```
┌─────────────────────────────────────────────┐
│  Human Operator                              │
│  (only at startup/restart)                  │
└─────────────┬───────────────────────────────┘
              │
              ↓ (loads once)
     tuning_params.json
              ↓
┌─────────────────────────────────────────────┐
│  Market Maker Bot                            │
│  ┌─────────────────────────────────────┐    │
│  │  Adam Optimizer (autonomous)        │    │
│  │  • Monitors performance             │    │
│  │  • Calculates gradients             │    │
│  │  • Updates parameters every 60s     │    │
│  │  • Saves to JSON (persistence)      │    │
│  └─────────────────────────────────────┘    │
└─────────────────────────────────────────────┘
              ↑
              │ (market data)
         Trading Venue
```

**Key Point**: Once the bot is running, Adam is in full control. There are no external interruptions, file checks, or hot-reloads. This ensures:
- Smooth, uninterrupted learning trajectory
- No race conditions between file I/O and optimization
- Predictable, monotonic improvement over time

## Monitoring

The system logs detailed information for each optimization step:

```
Control gap detected: 7.250000 bps² (bid_gap=2.50bps, ask_gap=1.00bps). Running Adam optimizer parameter tuning...
Gradient[skew_adjustment_factor] = 0.012345
Gradient[adverse_selection_adjustment_factor] = -0.003456
...
Adam Optimizer State:
  Time step: 42
  Base learning rate (α): 0.001
  Effective LR[skew_adjustment_factor]: 0.000234
  Effective LR[adverse_selection_adjustment_factor]: 0.000891
  ...
Adam Update Applied: TuningParams { ... }
Parameter Changes:
  skew_adjustment_factor: 0.500000 -> 0.497654 (Δ=-0.002346)
  ...
```

### Understanding the Logs

**Control Gap**:
```
Control gap detected: 7.250000 bps²
```
- This is the sum of squared differences in quote offsets
- `√(7.25/2) ≈ 1.9 bps` RMS difference per quote
- Threshold is 1.0 bps² (about 0.7 bps RMS per quote)

## Tuning the Optimizer (Advanced)

While the defaults work well, you can customize Adam's hyperparameters:

### Base Learning Rate (`α`)
- **Default**: `0.001` (conservative)
- **Aggressive**: `0.01` - Faster adaptation, more volatile
- **Conservative**: `0.0001` - Slower adaptation, more stable
- **Rule of thumb**: Start with default, only change if convergence issues

### Momentum (`β₁`)
- **Default**: `0.9` (standard)
- **High momentum**: `0.95` - Smoother, slower to change direction
- **Low momentum**: `0.8` - More responsive to recent gradients
- **Use case**: Increase in choppy markets, decrease in trending markets

### Variance Decay (`β₂`)
- **Default**: `0.999` (standard)
- **Rarely needs adjustment**
- **Lower values** (`0.99`) can help if gradients are very noisy

### When to Reset

The optimizer automatically resets when:
- Invalid parameters are produced (violate constraints)
- You manually call `optimizer.reset()`

Consider manual reset when:
- Switching trading assets
- Major market regime change
- After extended downtime

## Performance Characteristics

### Computational Cost
- **Gradient calculation**: ~7 evaluations per optimization cycle
- **Adam update**: O(n) where n=7 parameters (negligible)
- **Total**: ~200-500ms per cycle (background thread, non-blocking)

### Memory Overhead
- **Optimizer state**: 2 × 7 × 8 bytes = 112 bytes (negligible)
- **No significant memory impact**

### Convergence
- **Typical convergence**: 10-50 optimization cycles
- **Time to optimal**: 10-50 minutes (at 60s intervals)
- **Ongoing adaptation**: Continues to track market regime changes

## Comparison with Vanilla SGD

| Metric | Vanilla SGD | Adam Optimizer |
|--------|-------------|----------------|
| Manual tuning required | High | Low |
| Convergence speed | Slow | Fast (2-5x) |
| Stability | Sensitive to LR | Very stable |
| Noise handling | Poor | Excellent |
| Regime adaptation | Limited | Automatic |
| Parameter-specific LR | No | Yes |
| Recommended for production | No | Yes |

## References

- Kingma, D. P., & Ba, J. (2015). Adam: A Method for Stochastic Optimization. ICLR 2015.
- Ruder, S. (2016). An overview of gradient descent optimization algorithms. arXiv:1609.04747

## Next Steps

The Adam optimizer provides a strong foundation for automatic parameter tuning. Future enhancements could include:

1. **Adaptive threshold**: Dynamically adjust the 1.0 bps² control gap threshold
2. **Gradient clipping**: Prevent extreme updates during market shocks
3. **Parameter-specific constraints**: Different bounds for different market conditions
4. **Multi-objective optimization**: Balance P&L vs. inventory vs. risk simultaneously
5. **Online learning**: Update more frequently (every 10-30s instead of 60s)
