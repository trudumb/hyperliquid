# Autonomous Parameter Tuning Guide

## 🤖 Your Bot is Now a Self-Learning Agent

The market maker uses the Adam optimizer to **autonomously tune its parameters** in real-time. You don't need to babysit it or manually adjust parameters during operation.

## Quick Start

### 1. Set Your Initial Parameters (Optional)

If you want to start with custom parameters, edit `tuning_params.json` before starting:

```json
{
  "skew_adjustment_factor": 0.5,
  "adverse_selection_adjustment_factor": 0.5,
  "adverse_selection_lambda": 0.1,
  "inventory_urgency_threshold": 0.7,
  "liquidation_rate_multiplier": 10.0,
  "min_spread_base_ratio": 0.2,
  "adverse_selection_spread_scale": 100.0
}
```

**If you don't create this file**, the bot will use sensible defaults.

### 2. Start the Bot

```bash
RUST_LOG=info cargo run --bin market_maker_v2
```

The bot will:
- Load initial parameters from JSON (or use defaults)
- Start trading with those parameters
- Begin autonomous learning after 60 seconds

### 3. Let Adam Learn

You'll see logs like this every 60 seconds:

```
[INFO] Control gap detected: 7.25 bps² (bid_gap: 2.0 bps, ask_gap: 1.5 bps). Running Adam optimizer parameter tuning...
[INFO] Gradient[skew_adjustment_factor] = 0.012345
[INFO] Adam Optimizer State:
[INFO]   Time step: 42
[INFO]   Base learning rate (α): 0.001
[INFO]   Effective LR[skew_adjustment_factor]: 0.000234
[INFO] Adam Update Applied: TuningParams { ... }
[INFO] Parameter Changes:
[INFO]   skew_adjustment_factor: 0.500000 -> 0.497654 (Δ=-0.002346)
[INFO] Updated tuning parameters saved to tuning_params.json
```

**Adam is learning!** Each update moves the parameters closer to the optimal quote offsets for current market conditions.

## When to Intervene

### ✅ You SHOULD Intervene When:

1. **Changing trading assets** - Different assets may need different starting parameters
2. **Major market regime change** - E.g., bull market → bear market, low → high volatility
3. **After extended downtime** - Market conditions may have shifted significantly
4. **Testing new strategies** - Experimenting with different parameter ranges

### ❌ You Should NOT Intervene When:

1. **Parameters are "drifting"** - This is Adam adapting to market conditions
2. **Short-term performance dip** - Adam needs time to learn (give it 10-50 cycles)
3. **Parameters seem "wrong"** - Trust the optimizer unless you have strong evidence
4. **Curiosity** - Avoid random tweaking; let Adam do its job

## How to Intervene

When you decide to manually adjust parameters:

```bash
# 1. Stop the bot gracefully
Ctrl+C

# 2. Edit tuning_params.json
nano tuning_params.json

# 3. Restart the bot
RUST_LOG=info cargo run --bin market_maker_v2
```

**Important**: Adam will start learning from your new starting point. It will NOT reset its momentum/variance estimates unless parameters violate constraints.

## Understanding the Logs

### Control Gap
```
Control gap detected: 7.25 bps² (bid_gap: 2.0 bps, ask_gap: 1.5 bps)
```
- This measures how different your heuristic quotes are from the optimal quotes
- **Formula**: `(bid_optimal - bid_heuristic)² + (ask_optimal - ask_heuristic)²`
- **Units**: bps² (basis points squared)
- **Interpretation**: RMS quote difference ≈ √(loss/2) = √(7.25/2) ≈ 1.9 bps per quote
- **Larger gap** = Your quotes are further from optimal
- Adam tries to minimize this gap by tuning parameters

### Gradients
```
Gradient[skew_adjustment_factor] = 0.012345
```
- Shows how sensitive the control gap is to each parameter
- **Positive gradient** = Decreasing this parameter will reduce control gap
- **Negative gradient** = Increasing this parameter will reduce control gap
- **Large magnitude** = Parameter has strong effect on quote accuracy

### Effective Learning Rates
```
Effective LR[skew_adjustment_factor]: 0.000234
```
- Adam's adaptive learning rate for this specific parameter
- **Higher LR** = Parameter will change faster
- **Lower LR** = Parameter is more stable
- Changes over time as Adam learns variance

### Parameter Changes
```
skew_adjustment_factor: 0.500000 -> 0.497654 (Δ=-0.002346)
```
- Shows the actual update applied
- **Small changes** = Adam is fine-tuning
- **Large changes** = Adam found a strong signal

## Monitoring Performance

### Good Signs ✅
- Control gap decreasing over time
- Parameter changes becoming smaller (convergence)
- Effective learning rates stabilizing
- No validation errors

### Warning Signs ⚠️
- Control gap increasing consistently
- Parameters hitting constraints repeatedly
- Validation errors after updates
- Erratic parameter changes

### If Things Go Wrong

If you see:
```
[ERROR] Adam optimizer produced invalid parameters: ... Reverting to original.
[INFO] Adam optimizer state has been reset due to invalid parameters
```

**This is OK!** Adam tried something aggressive that violated constraints. It automatically resets and continues learning.

If it happens frequently (>10% of updates), consider:
1. Adjusting parameter constraints in `TuningParams::validate()`
2. Reducing Adam's learning rate (alpha) from 0.001 to 0.0001
3. Checking if your market conditions are extremely volatile

## Advanced: Customizing Adam

To change Adam's hyperparameters, edit the initialization in `market_maker_v2.rs`:

```rust
// Default (conservative)
adam_optimizer: Arc::new(RwLock::new(AdamOptimizerState::default())),

// Custom (more aggressive)
adam_optimizer: Arc::new(RwLock::new(AdamOptimizerState::new(
    0.01,   // alpha (learning rate) - 10x more aggressive
    0.9,    // beta1 (momentum)
    0.999,  // beta2 (variance)
))),
```

**Only do this if you understand the implications!**

## Philosophy: Trust the Optimizer

Modern ML systems like Adam are designed to handle non-stationary, noisy optimization problems - exactly what markets are. The optimizer has:

- **Momentum** to smooth out noise
- **Adaptive learning rates** to handle different parameter scales
- **Bias correction** for proper initialization
- **Gradient clipping** via constraint validation

Your intuition about "good" parameters is likely based on a small sample of market conditions. Adam sees all conditions over time and adapts continuously.

**Trust the process.** Let it run for 24-48 hours before making judgments.

## Example: First 24 Hours

What you might see in the first day:

**Hour 0-1**: Large parameter updates as Adam explores
```
Δ=0.05 to 0.10 (10-20% changes)
```

**Hour 1-6**: Medium updates as Adam finds good regions
```
Δ=0.01 to 0.05 (2-10% changes)
```

**Hour 6-24**: Small updates as Adam fine-tunes
```
Δ=0.001 to 0.01 (0.2-2% changes)
```

**After 24h**: Micro-adjustments tracking market regime
```
Δ=0.0001 to 0.001 (0.02-0.2% changes)
```

This is **healthy convergence**. If you don't see this pattern, something may be wrong.

## FAQ

### Q: Can I watch tuning_params.json change in real-time?

**A:** Yes! Use `watch -n 1 cat tuning_params.json` on Linux/Mac or a file watcher on Windows. You'll see Adam's updates every 60 seconds.

### Q: What if I want to tune faster than every 60 seconds?

**A:** Edit `optimization_interval` in the code:
```rust
let optimization_interval = std::time::Duration::from_secs(30); // 30s instead of 60s
```

**Tradeoff**: Faster tuning = more CPU usage and potentially noisier gradients.

### Q: Can I disable autonomous tuning and just use static parameters?

**A:** Yes! Comment out the background optimization task:
```rust
// if last_optimization_time.elapsed() >= optimization_interval {
//     ... entire optimization block ...
// }
```

The bot will run with whatever parameters are in `tuning_params.json`.

### Q: How do I know if Adam is actually improving performance?

**A:** Watch the logs for:
1. **Control gap trending down** over multiple cycles (quotes getting closer to optimal)
2. **Sharpe ratio improving** (if you track P&L)
3. **Fill rates increasing** at same spreads
4. **Inventory management improving** (less extreme positions)

Track these metrics externally and compare before/after 24-48 hours of Adam tuning.

### Q: What happens if the bot crashes and restarts?

**A:** Adam loads the last saved parameters from `tuning_params.json` and **starts fresh** (t=0, no momentum). This is intentional - after a crash, market conditions may have changed, so starting fresh is safer.

If you want to preserve Adam's state across restarts, you'd need to save/load the `AdamOptimizerState` to JSON as well (currently not implemented).

## Summary

🤖 **Your bot is autonomous**  
📈 **Adam continuously learns and adapts**  
🔧 **You only intervene at major regime changes**  
📊 **Monitor logs to track learning progress**  
🎯 **Trust the process for 24-48h before judging**

Let the machine do what machines do best: tireless optimization of thousands of market observations!
