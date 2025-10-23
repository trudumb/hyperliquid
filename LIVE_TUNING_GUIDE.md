# Live Tuning Guide for Market Maker V2

This guide explains how to tune the market maker parameters in real-time without restarting the bot.

## Overview

The Market Maker V2 now supports live tuning of key algorithmic parameters through a configurable `TuningParams` structure. This allows you to adjust the bot's behavior dynamically while it's running, enabling rapid iteration and optimization.

## Tunable Parameters

All tunable parameters are defined in the `TuningParams` struct:

### 1. `skew_adjustment_factor` (default: 0.5)
- **Range**: [0.0, 2.0]
- **Description**: Controls how aggressively quotes are skewed based on inventory position
- **Effect**: 
  - Higher values = more aggressive inventory management via quote skewing
  - When long, tightens ask and widens bid more aggressively
  - When short, tightens bid and widens ask more aggressively
- **Tuning**: Increase if inventory builds up too much, decrease if quotes are too asymmetric

### 2. `adverse_selection_adjustment_factor` (default: 0.5)
- **Range**: [0.0, 2.0]  
- **Description**: Controls how much spreads adjust based on adverse selection estimates
- **Effect**:
  - Higher values = more aggressive response to LOB imbalance signals
  - Widens quotes on the side expected to be picked off
- **Tuning**: Increase if you're getting adversely selected often, decrease if too conservative

### 3. `adverse_selection_lambda` (default: 0.1)
- **Range**: [0.0, 1.0]
- **Description**: Smoothing parameter for adverse selection filter (exponential moving average)
- **Effect**:
  - Higher values = more weight on recent observations (more responsive)
  - Lower values = smoother estimate, less reactive to noise
- **Tuning**: Increase for faster markets, decrease for noisy/choppy markets

### 4. `inventory_urgency_threshold` (default: 0.7)
- **Range**: [0.0, 1.0]
- **Description**: Inventory ratio threshold for activating aggressive taker liquidation
- **Effect**:
  - When inventory exceeds this fraction of max, bot sends market orders to reduce position
  - Lower values = activate liquidation earlier
- **Tuning**: Decrease if taking on too much inventory risk, increase if liquidating too early

### 5. `liquidation_rate_multiplier` (default: 10.0)
- **Range**: [0.0, 100.0]
- **Description**: Scales the rate of taker orders when urgency is high
- **Effect**:
  - Higher values = faster liquidation via more aggressive market orders
  - Rate formula: `(urgency - threshold) * multiplier` units per second
- **Tuning**: Increase for faster position reduction, decrease to avoid market impact

### 6. `min_spread_base_ratio` (default: 0.2)
- **Range**: [0.0, 1.0]
- **Description**: Minimum quote offset as fraction of base spread
- **Effect**:
  - Ensures quotes never get tighter than this ratio (prevents crossing spread)
  - 0.2 means minimum 20% of base spread
- **Tuning**: Increase for safer minimum spread, decrease for tighter quoting

### 7. `adverse_selection_spread_scale` (default: 100.0)
- **Range**: (0.0, ∞)
- **Description**: Denominator for normalizing market spread in adverse selection calculation
- **Effect**:
  - Higher values = less sensitivity to wide spreads
  - Lower values = more sensitive to spread changes
- **Tuning**: Adjust based on typical spread levels in your market

## Usage Methods

### Method 1: Programmatic Updates (Recommended for Testing)

You can update parameters directly in code:

```rust
use std::sync::{Arc, RwLock};

// Create custom parameters
let mut custom_params = TuningParams::default();
custom_params.skew_adjustment_factor = 0.7;  // More aggressive inventory management
custom_params.inventory_urgency_threshold = 0.6;  // Activate liquidation earlier

// Update the market maker
market_maker.update_tuning_params(custom_params)?;
```

### Method 2: JSON File Loading (Requires `config_file` feature)

1. **Enable the feature in Cargo.toml**:
```toml
[dependencies]
# ... other dependencies
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"

[features]
config_file = []
```

2. **Create a JSON config file** (e.g., `tuning_params.json`):
```json
{
  "skew_adjustment_factor": 0.6,
  "adverse_selection_adjustment_factor": 0.5,
  "adverse_selection_lambda": 0.15,
  "inventory_urgency_threshold": 0.65,
  "liquidation_rate_multiplier": 12.0,
  "min_spread_base_ratio": 0.2,
  "adverse_selection_spread_scale": 100.0
}
```

3. **Reload parameters while running**:
```rust
// In your trading loop or a separate monitoring thread
match market_maker.reload_tuning_params_from_file("tuning_params.json") {
    Ok(true) => info!("Tuning parameters reloaded successfully"),
    Ok(false) => info!("Config file not found, using current parameters"),
    Err(e) => error!("Failed to reload config: {}", e),
}
```

### Method 3: File Watching (Full Hot-Reload)

For complete hot-reload capability, add a background task that watches the config file:

```rust
use std::time::Duration;
use tokio::time::sleep;

// Spawn a config watcher task
let market_maker_clone = Arc::new(RwLock::new(market_maker));
let config_watcher = {
    let mm = market_maker_clone.clone();
    tokio::spawn(async move {
        let config_path = "tuning_params.json";
        let mut last_modified = std::fs::metadata(config_path)
            .ok()
            .and_then(|m| m.modified().ok());
        
        loop {
            sleep(Duration::from_secs(5)).await; // Check every 5 seconds
            
            if let Ok(metadata) = std::fs::metadata(config_path) {
                if let Ok(modified) = metadata.modified() {
                    if last_modified != Some(modified) {
                        last_modified = Some(modified);
                        
                        let mut mm = mm.write().unwrap();
                        match mm.reload_tuning_params_from_file(config_path) {
                            Ok(true) => info!("🔄 Auto-reloaded tuning parameters"),
                            Ok(false) => {},
                            Err(e) => error!("Failed to auto-reload: {}", e),
                        }
                    }
                }
            }
        }
    })
};

// In your main loop, you can now edit tuning_params.json and changes will be
// automatically picked up within 5 seconds
```

## Tuning Strategy

### 1. Start Conservative
Begin with default parameters and monitor performance for several hours:
- Observe P&L, inventory levels, fill rates
- Look for patterns: consistent adverse selection, inventory buildup, etc.

### 2. Identify Issues
Common issues and their parameter adjustments:

| **Issue** | **Likely Cause** | **Parameter to Adjust** | **Direction** |
|-----------|------------------|-------------------------|---------------|
| Building too much inventory | Quotes not skewed enough | `skew_adjustment_factor` | Increase |
| Getting adversely selected | Not reacting to LOB imbalance | `adverse_selection_adjustment_factor` | Increase |
| Too reactive to noise | Filter too responsive | `adverse_selection_lambda` | Decrease |
| Inventory spikes dangerous | Liquidation threshold too high | `inventory_urgency_threshold` | Decrease |
| Slow to reduce position | Liquidation rate too low | `liquidation_rate_multiplier` | Increase |
| Quotes crossing spread | Min spread too tight | `min_spread_base_ratio` | Increase |

### 3. Iterative Tuning
Make **one change at a time** and observe for 30-60 minutes before making another adjustment.

### 4. Parameter Interaction
Some parameters interact:
- `skew_adjustment_factor` and `inventory_urgency_threshold` both affect inventory management
  - If liquidating too early, increase threshold OR decrease skew factor
- `adverse_selection_adjustment_factor` and `adverse_selection_lambda` both affect adverse selection response
  - For fast adaptive response: high lambda + moderate adjustment factor
  - For smooth stable response: low lambda + high adjustment factor

## Monitoring

Log messages show current state and control vectors:

```
StateVector[S=100.50, Q=2.5000, μ̂=0.0250, Δ=12.5bps, I=0.580]
ControlVector[δ^b=11.2bps, δ^a=13.8bps, spread=25.0bps, asymmetry=2.6bps]
```

Watch for:
- **Q (inventory)**: Should mean-revert to 0
- **μ̂ (adverse selection estimate)**: Should be close to 0 on average
- **Δ (market spread)**: Track market conditions
- **I (LOB imbalance)**: Balanced around 0.5
- **asymmetry**: Indicates quote skewing direction

## Best Practices

1. **Version Control**: Keep your tuning_params.json in version control with timestamps
2. **A/B Testing**: Run two bots with different parameters and compare
3. **Backtesting**: Validate parameter changes on historical data first
4. **Gradual Changes**: Adjust parameters by 10-20% increments, not 100%
5. **Document Changes**: Log why you made each parameter adjustment
6. **Revert Quickly**: If P&L degrades after a change, revert immediately

## Example Configurations

### Conservative (Low Risk)
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

### Aggressive (High Turnover)
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

### High-Frequency (Fast Adaptation)
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

## Validation

All parameters are validated before being applied:
- Out-of-range values will be rejected with an error
- Invalid JSON will be caught during parsing
- The bot continues with previous parameters if validation fails

Check logs for validation errors:
```
ERROR: Invalid parameter: skew_adjustment_factor must be in [0.0, 2.0], got 3.5
```

## Integration with HJB Optimization

The tuning parameters affect the **heuristic control policy**. The bot also runs full HJB grid search optimization in the background every 60 seconds to:

1. Validate heuristic performance
2. Suggest better parameter values
3. Detect performance gaps

Watch for these log messages:
```
Background HJB Optimization Complete: Heuristic_Value=0.1234, Optimal_Value=0.1456, Gap=15.23%
```

If the gap is consistently >10%, consider retuning parameters based on the optimal control vector logged.

## Troubleshooting

**Q: Parameters not updating**
- Check file permissions
- Verify JSON syntax (use `jq` or online validator)
- Check logs for validation errors

**Q: Bot behavior unchanged after parameter update**
- Parameters only affect future decisions, not existing orders
- Wait for next order update cycle (typically a few seconds)
- Verify parameters were actually reloaded (check logs)

**Q: Performance degraded after tuning**
- Revert to previous working configuration
- Make smaller incremental changes
- Validate on paper trading / testnet first

## Advanced: Custom Tuning Logic

For advanced users, you can implement custom tuning logic that adjusts parameters based on market conditions:

```rust
// Example: Adjust parameters based on volatility
let vol = calculate_realized_volatility();

let mut params = market_maker.get_tuning_params();

if vol > HIGH_VOL_THRESHOLD {
    // In high volatility, be more conservative
    params.adverse_selection_lambda = 0.15;  // React faster
    params.min_spread_base_ratio = 0.3;      // Wider minimum
} else if vol < LOW_VOL_THRESHOLD {
    // In low volatility, can be more aggressive
    params.adverse_selection_lambda = 0.08;  // Smoother
    params.min_spread_base_ratio = 0.15;     // Tighter
}

market_maker.update_tuning_params(params)?;
```

---

**Remember**: Live tuning is powerful but requires discipline. Always monitor the impact of changes and be ready to revert if needed.
