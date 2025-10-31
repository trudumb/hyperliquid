# HJB Strategy V2 Migration Guide

## Overview

HjbStrategyV2 is a complete refactoring of the original HJB market making strategy with a modular, event-driven architecture. This guide explains how to migrate from HjbStrategy (v1) to HjbStrategyV2.

## What's New in V2?

### Architecture Improvements

1. **Event-Driven Design**
   - Centralized `EventBus` for component coordination
   - `LoggingSubscriber` and `MetricsSubscriber` for observability
   - Full event history for debugging

2. **Single Source of Truth**
   - `TradingStateStore` holds all state (positions, orders, market data)
   - No more conflicting state between components
   - Guaranteed consistency

3. **Proper Order Lifecycle**
   - `OrderStateMachine` tracks all order state transitions
   - Automatic timeout detection for stuck orders
   - No more orders stuck in `PendingCancel` state

4. **Clear Separation of Concerns**
   ```
   Signal Generation → Risk Adjustment → Order Execution
   ```

5. **Position Safety**
   - Single enforcement point in `PositionManager`
   - Consistent limit checking across all components
   - State-based trading (Normal → Warning → Critical → OverLimit)

### Component Architecture

```
MarketUpdate → MarketDataPipeline → TradingStateStore
                                           ↓
         ┌─────────────────────────────────┴──────────────────────────┐
         ↓                                                              ↓
  HjbSignalGenerator                                             EventBus
  (pure HJB optimization)                                              ↓
         ↓                                               Subscribers (logging, metrics)
  RiskAdjuster
  (apply position/margin limits)
         ↓
  OrderExecutor
  (reconcile & execute orders)
         ↓
  StrategyActions
```

## Migration Steps

### Step 1: Update Configuration

Change your `config.json` from:
```json
{
  "asset": "HYPE",
  "strategy_name": "hjb_v1",
  "strategy_params": { ... }
}
```

To:
```json
{
  "asset": "HYPE",
  "strategy_name": "hjb_v2",
  "strategy_params": { ... }
}
```

### Step 2: Configuration Parameters

HjbStrategyV2 uses a simplified configuration:

#### Required Parameters
```json
{
  "max_absolute_position_size": 50.0,
  "phi": 0.01,
  "lambda_base": 1.0,
  "maker_fee_bps": 1.5,
  "taker_fee_bps": 4.5,
  "leverage": 3,
  "margin_safety_buffer": 0.2
}
```

#### Optional Parameters
```json
{
  "requote_threshold_bps": 5.0,
  "ewma_vol_config": {
    "lookback_window_size": 100,
    "alpha": 0.05,
    "min_volatility_bps": 5.0,
    "max_volatility_bps": 500.0
  }
}
```

### Step 3: Gradual Rollout (Recommended)

1. **Shadow Mode** (Week 1)
   - Run V2 alongside V1
   - Log V2 actions but don't execute them
   - Compare outputs

2. **Partial Execution** (Week 2)
   - Execute 10% of orders with V2
   - Monitor metrics closely
   - Compare performance with V1

3. **Increased Execution** (Week 3)
   - Execute 50% of orders with V2
   - Verify no position limit violations
   - Check for stuck orders

4. **Full Migration** (Week 4)
   - Execute 100% with V2
   - Decommission V1
   - Monitor for at least 7 days

### Step 4: Run Both Strategies Simultaneously (Optional)

You can run both strategies for different assets:

```json
{
  "strategies": [
    {
      "asset": "HYPE",
      "strategy_name": "hjb_v2",
      "strategy_params": { ... }
    },
    {
      "asset": "BTC",
      "strategy_name": "hjb_v1",
      "strategy_params": { ... }
    }
  ]
}
```

## Key Differences

### Position Management

**V1 (Old)**
- Position limits enforced in multiple places
- StateManager, Optimizer, and Strategy all check limits
- Can lead to conflicting decisions

**V2 (New)**
- Single `PositionManager` enforces limits
- State-based trading:
  - **Normal** (< 70% of max): Full trading
  - **Warning** (70-85%): Reduced size
  - **Critical** (85-100%): Reduce-only orders
  - **OverLimit** (> 100%): Emergency liquidation

### Order State Tracking

**V1 (Old)**
- Basic `OrderState` enum
- Manual tracking of pending cancels
- Orders can get stuck

**V2 (New)**
- Proper `OrderStateMachine` with transitions
- Automatic timeout detection
- State validation on every transition

### Observability

**V1 (Old)**
- Scattered logging
- Limited metrics
- Hard to debug

**V2 (New)**
- Centralized `EventBus`
- All state changes published as events
- Metrics collection via subscribers
- Event history for debugging

## Monitoring

### Key Metrics to Watch

1. **Position State Changes**
   ```
   Normal → Warning: Position reached 70% of max
   Warning → Critical: Position reached 85% of max
   Critical → OverLimit: Position exceeded max (should never happen!)
   ```

2. **Order Lifecycle**
   ```
   Creating → Open: Order confirmed by exchange
   Open → PendingCancel: Cancel requested
   PendingCancel → Canceled: Cancel confirmed
   ```

3. **Event Counts**
   ```
   Total fills
   Total cancels
   Position state changes
   Margin warnings
   ```

### Logs to Monitor

Look for these log messages:

```
[HJB V2] Position state: Critical, Position: 42.50, Max: 50.00
[HJB V2] Signal adjusted: Margin limit: reduced 5.0000 -> 2.5000
[EVENT] Position state: Warning → Critical (pos: 43.25)
[EVENT] EMERGENCY LIQUIDATION: pos=52.00, reason=Position over limit
```

## Performance Comparison

### Expected Improvements

1. **Position Safety**: 0% position limit violations (vs. occasional violations in V1)
2. **Order Management**: 0% stuck orders (vs. rare stuck PendingCancel in V1)
3. **State Consistency**: 100% consistent state (vs. occasional race conditions in V1)
4. **Debuggability**: 10x easier to debug with event history

### Expected Parity

1. **PnL**: Should be similar (same HJB optimization logic)
2. **Fill Rates**: Should be similar (same quote optimization)
3. **Spreads**: Should be similar (same parameters)

## Troubleshooting

### Issue: Strategy not starting

**Symptom**: Error on startup
```
Unknown strategy: hjb_v2. Available strategies: hjb_v1, hjb_v2
```

**Solution**: Make sure you've updated `market_maker_v3.rs` with the latest code.

### Issue: Different quotes than V1

**Symptom**: V2 generates different quotes

**Solution**: This is expected! V2 has proper risk management. Check:
- Position state (Normal/Warning/Critical)
- Risk adjustment reason in logs
- Margin availability

### Issue: No orders being placed

**Symptom**: V2 logs show signal generation but no orders

**Possible Causes**:
1. Position in Critical state → Only reduce-only orders allowed
2. Margin limit hit → Orders size-adjusted to zero
3. Risk adjustment filtering out all orders

**Debug Steps**:
```
1. Check logs for "[HJB V2] Signal adjusted: ..."
2. Check position state
3. Check margin availability
4. Verify config parameters
```

## Rollback Plan

If you need to rollback to V1:

1. Change `strategy_name` back to `"hjb_v1"` in config
2. Restart the market maker
3. Monitor for 30 minutes to ensure normal operation
4. Report issues for investigation

## Support

For issues or questions:
1. Check logs for `[HJB V2]` messages
2. Check event history for state changes
3. Review metrics from `MetricsSubscriber`
4. Create an issue with:
   - Configuration used
   - Logs from the issue period
   - Event history
   - Expected vs. actual behavior

## Next Steps

Once V2 is stable:
1. Add custom `EventSubscriber` for your monitoring system
2. Tune position manager thresholds based on your risk tolerance
3. Add custom processors to `MarketDataPipeline`
4. Implement custom risk adjustments in `RiskAdjuster`

## References

- Original implementation: `src/strategies/hjb_strategy.rs`
- New implementation: `src/strategies/hjb_strategy_v2.rs`
- Component architecture: `src/strategies/components/`
- Strategy factory: `src/bin/market_maker_v3.rs`
