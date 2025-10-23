# Tuning Parameters Guide

## Overview

The Market Maker V2 supports **hot-reloading** of tuning parameters. You can adjust strategy behavior in real-time by editing `tuning_params.json` - changes take effect within 10 seconds without restarting the bot.

## Quick Start

1. The bot reads from `tuning_params.json` every 10 seconds
2. Edit the file with your desired parameters
3. Save the file
4. Within 10 seconds, the bot will reload and apply the new parameters
5. Check the logs for confirmation: `"Reloaded tuning parameters from tuning_params.json"`

## Parameters

### `skew_adjustment_factor`
**Range:** `[0.0, 2.0]`  
**Default:** `0.5`  
**Purpose:** Controls how aggressively quotes are skewed based on inventory position.

- **Higher values** (0.7-1.0): More aggressive inventory management
  - Quotes shift more dramatically when inventory is imbalanced
  - Better for risk-averse strategies or volatile markets
  - Helps unload inventory faster
  
- **Lower values** (0.2-0.4): More conservative inventory management
  - Quotes shift less with inventory changes
  - Better for stable markets where you want to maintain positions
  - Allows for larger inventory swings

**Example use case:** Increase to 0.8 during volatile periods to reduce inventory risk.

---

### `adverse_selection_adjustment_factor`
**Range:** `[0.0, 2.0]`  
**Default:** `0.5`  
**Purpose:** Controls how much to widen spreads in response to adverse selection signals.

- **Higher values** (0.7-1.0): More defensive against adverse selection
  - Spreads widen more when detecting informed trading
  - Better protection against getting picked off
  - Reduces fill rate but improves fill quality
  
- **Lower values** (0.2-0.4): Less defensive, more aggressive
  - Tighter spreads even with adverse selection signals
  - Higher fill rate but potentially worse prices
  - Good for liquid markets with low information asymmetry

**Example use case:** Increase to 0.8 during news events or high volatility when information asymmetry is high.

---

### `adverse_selection_lambda`
**Range:** `[0.0, 1.0]`  
**Default:** `0.1`  
**Purpose:** Exponential smoothing parameter for adverse selection filter. Controls how responsive the filter is to new signals.

- **Higher values** (0.15-0.3): More responsive to recent signals
  - Reacts faster to changes in market conditions
  - More volatile adjustments
  - Good for fast-moving markets
  
- **Lower values** (0.05-0.1): Smoother, slower adjustments
  - More weight on historical average
  - Less volatile adjustments
  - Good for stable markets

**Formula:** `new_value = lambda × new_signal + (1 - lambda) × old_value`

**Example use case:** Increase to 0.15-0.2 during volatile hours (US market open) for faster adaptation.

---

### `inventory_urgency_threshold`
**Range:** `[0.0, 1.0]`  
**Default:** `0.7`  
**Purpose:** Inventory ratio (position / max_position) above which to activate urgent liquidation via market orders.

- **Higher values** (0.8-0.9): More patient with inventory
  - Allows larger positions before panic
  - Only liquidates when very close to limits
  - Good when confident in mean reversion
  
- **Lower values** (0.5-0.6): More aggressive liquidation
  - Starts liquidating earlier
  - Maintains tighter inventory control
  - Better for risk-averse strategies

**Example:** With `max_absolute_position_size = 3.0`:
- At 0.7 threshold: Liquidation starts at position ±2.1
- At 0.5 threshold: Liquidation starts at position ±1.5

**Example use case:** Lower to 0.6 before high-impact news events to reduce risk.

---

### `liquidation_rate_multiplier`
**Range:** `[1.0, 100.0]`  
**Default:** `10.0`  
**Purpose:** Scales the aggressiveness of market order liquidation when urgency is triggered.

- **Higher values** (15.0-30.0): Very aggressive liquidation
  - Places larger market orders
  - Exits positions faster
  - Higher slippage but faster risk reduction
  
- **Lower values** (5.0-8.0): Gentler liquidation
  - Smaller market orders
  - Exits positions more gradually
  - Lower slippage but slower risk reduction

**Example use case:** Increase to 20.0+ during market stress when you need to exit positions rapidly.

---

### `min_spread_base_ratio`
**Range:** `[0.0, 1.0]`  
**Default:** `0.2`  
**Purpose:** Minimum quote offset as a fraction of base spread. Ensures quotes don't get too tight during adjustments.

- **Higher values** (0.3-0.5): Wider minimum spreads
  - More conservative, safer margins
  - Lower fill rate
  - Better protection against adverse moves
  
- **Lower values** (0.1-0.2): Tighter minimum spreads
  - More aggressive, competitive quotes
  - Higher fill rate
  - Better for liquid markets

**Example:** With `half_spread = 5 bps`:
- At 0.2 ratio: Minimum spread is 1 bp (0.2 × 5)
- At 0.4 ratio: Minimum spread is 2 bp (0.4 × 5)

**Example use case:** Increase to 0.3-0.4 during low liquidity periods (e.g., Asian hours).

---

### `adverse_selection_spread_scale`
**Range:** `[10.0, 1000.0]`  
**Default:** `100.0`  
**Purpose:** Scaling factor for converting adverse selection estimates into spread adjustments.

- **Higher values** (150.0-200.0): Stronger spread widening response
  - Larger spread adjustments for same adverse selection signal
  - More defensive
  
- **Lower values** (50.0-80.0): Weaker spread widening response
  - Smaller spread adjustments
  - More aggressive

**Example use case:** Increase to 150.0 during high volatility to provide more buffer against adverse selection.

---

### `control_gap_threshold`
**Range:** `[0.1, 50.0]`  
**Default:** `1.0`  
**Purpose:** Minimum control gap (in bps²) required to trigger Adam optimizer parameter tuning. This is a **meta-parameter** that controls when the optimizer runs, not a parameter being optimized itself.

- **Higher values** (5.0-20.0): Conservative tuning
  - Only tune when quotes deviate significantly from optimal
  - Prevents optimizer from chasing noise
  - Better for volatile or noisy markets
  - Reduces computational overhead
  
- **Lower values** (0.5-2.0): Aggressive tuning
  - Tune even for small deviations from optimal
  - Allows extremely fine-grained parameter adjustment
  - Better for stable, quiet markets
  - Higher computational overhead

**How it works:**
- Control gap = (bid_optimal - bid_heuristic)² + (ask_optimal - ask_heuristic)²
- If control gap > threshold, Adam optimizer runs and tunes parameters
- If control gap ≤ threshold, no tuning occurs (heuristic is "close enough")

**Market condition guidelines:**
- **Stable/Quiet markets** (low volatility, tight spreads): 0.5-2.0
  - Small deviations are meaningful and worth correcting
  - Fine-tuning improves performance
  
- **Normal markets** (moderate volatility): 1.0-3.0
  - Default range works well for most conditions
  
- **Volatile/Noisy markets** (high volatility, wide spreads): 5.0-20.0
  - Large deviations are normal and may be transient
  - Prevents overreacting to noise

**Example:** With threshold = 1.0:
- If bid_gap=0.5bps and ask_gap=0.5bps: gap=0.5 (no tuning)
- If bid_gap=0.8bps and ask_gap=0.6bps: gap=1.0 (tuning triggers)
- If bid_gap=1.0bps and ask_gap=1.0bps: gap=2.0 (tuning triggers)

**Example use case:** 
- Increase to 10.0 during volatile market conditions (VIX > 30) to prevent Adam from chasing transient noise
- Decrease to 0.5 during overnight hours with stable conditions for maximum precision

---

## Pre-Configured Scenarios

### 🔴 Conservative (High Risk Aversion)
```json
{
  "skew_adjustment_factor": 0.8,
  "adverse_selection_adjustment_factor": 0.7,
  "adverse_selection_lambda": 0.15,
  "inventory_urgency_threshold": 0.6,
  "liquidation_rate_multiplier": 15.0,
  "min_spread_base_ratio": 0.3,
  "adverse_selection_spread_scale": 150.0,
  "control_gap_threshold": 5.0
}
```
**Use when:** High volatility, news events, low liquidity, uncertain conditions

---

### 🟢 Aggressive (Low Risk Aversion)
```json
{
  "skew_adjustment_factor": 0.3,
  "adverse_selection_adjustment_factor": 0.3,
  "adverse_selection_lambda": 0.08,
  "inventory_urgency_threshold": 0.8,
  "liquidation_rate_multiplier": 8.0,
  "min_spread_base_ratio": 0.15,
  "adverse_selection_spread_scale": 70.0,
  "control_gap_threshold": 0.5
}
```
**Use when:** Low volatility, stable markets, high liquidity, tight spreads beneficial

---

### 🟡 Balanced (Default)
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
**Use when:** Normal market conditions, default starting point

---

### 🟠 High Volatility Mode
```json
{
  "skew_adjustment_factor": 0.7,
  "adverse_selection_adjustment_factor": 0.8,
  "adverse_selection_lambda": 0.2,
  "inventory_urgency_threshold": 0.65,
  "liquidation_rate_multiplier": 12.0,
  "min_spread_base_ratio": 0.25,
  "adverse_selection_spread_scale": 120.0,
  "control_gap_threshold": 10.0
}
```
**Use when:** VIX spike, market crash, flash crash, extreme volatility

---

### 🔵 Inventory Accumulation Mode
```json
{
  "skew_adjustment_factor": 0.2,
  "adverse_selection_adjustment_factor": 0.4,
  "adverse_selection_lambda": 0.08,
  "inventory_urgency_threshold": 0.85,
  "liquidation_rate_multiplier": 5.0,
  "min_spread_base_ratio": 0.18,
  "adverse_selection_spread_scale": 80.0,
  "control_gap_threshold": 0.8
}
```
**Use when:** Building position, confident in direction, want to hold inventory

---

## Time-Based Adjustments

### US Market Hours (9:30 AM - 4:00 PM ET)
- Higher liquidity → Can be more aggressive
- Suggested: Balanced or Aggressive preset
- Consider: `adverse_selection_lambda = 0.12` (slightly faster reactions)

### Asian Hours (7:00 PM - 3:00 AM ET)
- Lower liquidity → Should be more conservative
- Suggested: Conservative preset
- Consider: `min_spread_base_ratio = 0.25` (wider minimum spreads)

### News Events / Economic Releases
- High information asymmetry → Very conservative
- Suggested: Conservative or High Volatility preset
- Consider: `adverse_selection_adjustment_factor = 0.8`

---

## How to Adjust Live

### Example 1: Market becoming more volatile
```bash
# Edit tuning_params.json
nano tuning_params.json

# Change these values:
# "adverse_selection_lambda": 0.15,          # React faster
# "min_spread_base_ratio": 0.25,            # Wider spreads
# "inventory_urgency_threshold": 0.65,      # Liquidate earlier

# Save and exit - bot will reload within 10 seconds
```

### Example 2: Want to accumulate more inventory
```bash
# Edit tuning_params.json
nano tuning_params.json

# Change these values:
# "skew_adjustment_factor": 0.3,            # Less aggressive skewing
# "inventory_urgency_threshold": 0.8,       # Allow more inventory

# Save and exit
```

---

## Monitoring

Watch the bot logs for reload confirmation:
```
[INFO] Reloaded tuning parameters from tuning_params.json: TuningParams { ... }
```

The bot also logs background HJB optimization every 60 seconds:
```
[INFO] Background HJB Optimization Complete: Heuristic_Value=X.XX, Optimal_Value=Y.YY, Gap=Z.Z%
```

If the gap is >10%, consider adjusting parameters to better align with optimal control.

---

## Validation

All parameters are validated before application:
- Out-of-range values will be rejected
- Invalid JSON will be rejected
- On error, the bot continues with previous valid parameters
- Error messages will appear in logs

---

## Best Practices

1. **Start with defaults** - The default balanced preset works well for most conditions
2. **Make small adjustments** - Change one parameter at a time by 10-20%
3. **Monitor for 5-10 minutes** - Observe the effects before making more changes
4. **Keep a backup** - Save your current config before experimenting
5. **Document changes** - Note what you changed and why
6. **Watch the P&L** - Ultimately, profitability determines if changes are good

---

## Troubleshooting

**Q: Changed the file but bot isn't updating?**
- Check logs for error messages
- Verify JSON is valid (use a JSON validator)
- Ensure all parameters are in valid ranges
- Check file permissions

**Q: Bot rejected my changes?**
- Look for validation error in logs
- Check parameter ranges in this guide
- Verify JSON syntax (commas, quotes, brackets)

**Q: Performance got worse after changes?**
- Revert to previous working config
- Make smaller incremental changes
- Consider if market conditions changed (not just parameters)

**Q: How do I know if my tuning is good?**
- Monitor: Fill rate, P&L, inventory control, spread competitiveness
- Compare: Performance before/after parameter changes
- Check: HJB optimization gap (should be <10%)

---

## Support

For issues or questions about tuning:
1. Check this guide thoroughly
2. Review bot logs for clues
3. Test changes in low-risk conditions first
4. Consider consulting the HJB optimization output for guidance
