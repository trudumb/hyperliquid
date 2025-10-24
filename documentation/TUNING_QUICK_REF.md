# Live Tuning Quick Reference Card

## Quick Start

1. **Create config file**: Copy `tuning_params.example.json` to `tuning_params.json`
2. **Edit values**: Adjust parameters while bot is running
3. **Save file**: Changes are automatically detected within 5 seconds (if file watcher enabled)

## Parameter Quick Reference

| Parameter | Default | Range | Quick Description |
|-----------|---------|-------|-------------------|
| `skew_adjustment_factor` | 0.5 | [0.0, 2.0] | Inventory → quote skewing intensity |
| `adverse_selection_adjustment_factor` | 0.5 | [0.0, 2.0] | LOB imbalance → spread adjustment |
| `adverse_selection_lambda` | 0.1 | [0.0, 1.0] | Filter responsiveness (higher = faster) |
| `inventory_urgency_threshold` | 0.7 | [0.0, 1.0] | When to activate market orders |
| `liquidation_rate_multiplier` | 10.0 | [0.0, 100.0] | Market order aggressiveness |
| `min_spread_base_ratio` | 0.2 | [0.0, 1.0] | Minimum quote tightness |
| `adverse_selection_spread_scale` | 100.0 | (0.0, ∞) | Spread normalization factor |

## Common Issues → Solutions

| Problem | Parameter to Change | Direction | New Value |
|---------|---------------------|-----------|-----------|
| 📈 Building long inventory | `skew_adjustment_factor` | ↑ Increase | 0.6 - 0.8 |
| 📉 Building short inventory | `skew_adjustment_factor` | ↑ Increase | 0.6 - 0.8 |
| 😱 Getting adversely selected | `adverse_selection_adjustment_factor` | ↑ Increase | 0.6 - 0.8 |
| 🎢 Too reactive to noise | `adverse_selection_lambda` | ↓ Decrease | 0.05 - 0.08 |
| ⚠️ Inventory too high | `inventory_urgency_threshold` | ↓ Decrease | 0.5 - 0.6 |
| 🐌 Not reducing position fast | `liquidation_rate_multiplier` | ↑ Increase | 15.0 - 20.0 |
| ⚡ Liquidating too quickly | `liquidation_rate_multiplier` | ↓ Decrease | 5.0 - 8.0 |
| 🔀 Quotes crossing spread | `min_spread_base_ratio` | ↑ Increase | 0.25 - 0.3 |

## Preset Configurations

### 🛡️ Conservative (Low Risk)
```json
{
  "skew_adjustment_factor": 0.7,
  "adverse_selection_adjustment_factor": 0.6,
  "adverse_selection_lambda": 0.08,
  "inventory_urgency_threshold": 0.6,
  "liquidation_rate_multiplier": 15.0,
  "min_spread_base_ratio": 0.25,
  "adverse_selection_spread_scale": 100.0
}
```
Use when: High volatility, uncertain market, low risk tolerance

### ⚡ Aggressive (High Turnover)
```json
{
  "skew_adjustment_factor": 0.4,
  "adverse_selection_adjustment_factor": 0.4,
  "adverse_selection_lambda": 0.15,
  "inventory_urgency_threshold": 0.8,
  "liquidation_rate_multiplier": 8.0,
  "min_spread_base_ratio": 0.15,
  "adverse_selection_spread_scale": 100.0
}
```
Use when: Stable market, high liquidity, seeking volume

### 🚀 High-Frequency (Fast Adaptation)
```json
{
  "skew_adjustment_factor": 0.5,
  "adverse_selection_adjustment_factor": 0.7,
  "adverse_selection_lambda": 0.25,
  "inventory_urgency_threshold": 0.7,
  "liquidation_rate_multiplier": 20.0,
  "min_spread_base_ratio": 0.2,
  "adverse_selection_spread_scale": 50.0
}
```
Use when: Fast markets, need quick reaction, comfortable with risk

## Monitoring Commands

Watch logs for these key indicators:

```bash
# State Vector (every update)
StateVector[S=100.50, Q=2.5, μ̂=0.025, Δ=12.5bps, I=0.580]
#           ^price   ^inv ^adverse  ^spread    ^imbalance

# Control Vector (every update)
ControlVector[δ^b=11.2bps, δ^a=13.8bps, spread=25.0bps, asymmetry=2.6bps]
#             ^bid offset  ^ask offset  ^total         ^skew

# Parameter reload notification
🔄 Auto-reloaded tuning parameters from tuning_params.json
```

## Tuning Workflow

1. **Observe** → Watch logs for 30-60 minutes
2. **Identify** → Find the issue from table above  
3. **Change** → Edit ONE parameter in `tuning_params.json`
4. **Save** → File is auto-reloaded (or restart if no watcher)
5. **Monitor** → Watch for 30-60 minutes
6. **Iterate** → Repeat if needed

## Emergency Revert

If performance degrades:
1. Copy `tuning_params.example.json` to `tuning_params.json` (resets to defaults)
2. Or press Ctrl+C and restart bot
3. Check git history for last working config

## API Usage (Programmatic)

```rust
// Get current parameters
let params = market_maker.get_tuning_params();

// Update parameters
let mut new_params = TuningParams::default();
new_params.skew_adjustment_factor = 0.7;
market_maker.update_tuning_params(new_params)?;

// Reload from file (if config_file feature enabled)
market_maker.reload_tuning_params_from_file("tuning_params.json")?;
```

## Validation

All parameters are validated on load. Invalid values will be rejected:
- Out of range → Error with expected range
- Invalid JSON → Parse error
- Missing fields → Uses defaults for missing fields

## Tips

✅ **DO**
- Make one change at a time
- Wait 30-60 min between changes
- Keep notes on what you changed and why
- Test on paper trading first
- Keep backups of working configs

❌ **DON'T**
- Change multiple parameters at once
- Make huge jumps (>50% changes)
- Tune during extreme market events
- Ignore validation errors
- Forget to monitor after changes

---

💡 **Pro Tip**: Keep a spreadsheet tracking parameter changes and their P&L impact over time.

📖 **Full Guide**: See `LIVE_TUNING_GUIDE.md` for detailed explanations and advanced usage.
