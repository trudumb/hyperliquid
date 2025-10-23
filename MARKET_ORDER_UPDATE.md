# Market Order Implementation Update

## Summary

Updated `market_maker_v2.rs` to use the SDK's **safer market order functions** (`market_open` and `market_close`) with built-in slippage protection instead of the previous aggressive IOC limit orders.

## Problem with Previous Implementation

The previous code used **aggressive IOC (Immediate or Cancel) limit orders** for taker orders and position closing:

```rust
// OLD: Aggressive IOC limit order
let order = self.exchange_client.order(
    ClientOrderRequest {
        asset,
        is_buy,
        reduce_only: false,
        limit_px: best_bid_price,  // ❌ Price could be stale by the time order reaches exchange
        sz: amount,
        order_type: ClientOrder::Limit(ClientLimit {
            tif: "Ioc".to_string(),
        }),
    },
    None,
).await;
```

### Why This Failed

1. **Stale Price**: By the time the order reached the exchange, `best_bid_price` was often gone
2. **IOC Safety Mechanism**: The IOC flag acted as a safety mechanism - if it couldn't fill at the specified price, it cancelled
3. **Error Message**: This resulted in "Order could not immediately match" errors
4. **Not a True Market Order**: True market orders say "fill me at any price" (dangerous in volatile markets)

### Why This Was Actually Good (But Not Good Enough)

The IOC approach **protected against slippage** but was **too conservative**:
- ✅ **Protected from bad fills** (won't sell at terrible prices)
- ❌ **Failed to execute** when markets moved even slightly
- ❌ **Required manual price calculations** (error-prone)

## New Implementation: SDK Market Orders with Slippage

Now using the SDK's `market_open()` and `market_close()` functions:

```rust
// NEW: Market order with slippage protection
let market_params = MarketOrderParams {
    asset: &asset,
    is_buy,
    sz: amount,
    px: None,                    // ✅ SDK fetches current price
    slippage: Some(0.01),        // ✅ 1% slippage tolerance
    cloid: None,
    wallet: None,
};

let order = self.exchange_client.market_open(market_params).await;
```

### How This Works (Under the Hood)

The SDK's `market_open`/`market_close` functions:

1. **Fetch current market price** from the exchange in real-time
2. **Calculate slippage range**: 
   - For buys: `price * (1 + slippage)` = up to 1% above current price
   - For sells: `price * (1 - slippage)` = down to 1% below current price
3. **Place IOC limit order** at the slippage-adjusted price
4. **Handle all edge cases** (rounding, decimals, validation)

### Benefits

✅ **Fresh prices**: Fetched at order time, not stale from book data  
✅ **Slippage protection**: Won't execute at prices worse than tolerance  
✅ **Simpler code**: No manual price calculation or rounding  
✅ **Better execution**: 1% tolerance allows fills in normal market conditions  
✅ **Still safe**: Won't get disastrously filled like a true market order  

## Changes Made

### 1. Updated `place_taker_order()` Function

**Before**: Manual IOC limit order with best bid/ask price  
**After**: Uses `market_open()` with 1% slippage

```rust
async fn place_taker_order(
    &self,
    asset: String,
    amount: f64,
    _price: f64,  // Now ignored, kept for API compatibility
    is_buy: bool,
) -> f64 {
    let market_params = MarketOrderParams {
        asset: &asset,
        is_buy,
        sz: amount,
        px: None,
        slippage: Some(0.01),  // 1% slippage tolerance
        cloid: None,
        wallet: None,
    };

    self.exchange_client.market_open(market_params).await
    // ... error handling ...
}
```

### 2. Updated `close_position()` Function

**Before**: Manual IOC limit order with 5% price adjustment  
**After**: Uses `market_close()` with 1% slippage

```rust
async fn close_position(&mut self) {
    let market_close_params = MarketCloseParams {
        asset: &self.asset,
        sz: None,              // Close entire position
        px: None,              // SDK fetches current price
        slippage: Some(0.01),  // 1% slippage tolerance
        cloid: None,
        wallet: None,
    };

    self.exchange_client.market_close(market_close_params).await
    // ... error handling ...
}
```

### 3. Updated Taker Order Execution in `potentially_update()`

- Removed manual price calculation from order book
- Added clarifying comments that price is just for logging
- Updated log messages to show "ref_price" instead of "price"

### 4. Added Necessary Imports

```rust
use crate::{
    // ... existing imports ...
    MarketCloseParams, MarketOrderParams,  // Added
    // ... rest ...
};
```

## Slippage Tolerance

Currently set to **1% (0.01)** for all market orders. This is:

- **Conservative enough** to prevent bad fills in most conditions
- **Aggressive enough** to execute in normal market volatility
- **Tunable**: Can be adjusted based on:
  - Asset volatility
  - Market conditions (use state_vector.market_spread_bps)
  - Position urgency
  - Time of day / liquidity

### Recommended Adjustments

For more sophisticated slippage management:

```rust
// Dynamic slippage based on market conditions
let slippage = if self.state_vector.market_spread_bps > 50.0 {
    0.02  // 2% in volatile markets
} else if urgency > 0.9 {
    0.05  // 5% for emergency liquidation
} else {
    0.01  // 1% normal operation
};
```

## Testing Recommendations

1. **Monitor fill rates**: Check if orders are executing more frequently
2. **Track slippage**: Log actual fill price vs expected price
3. **Adjust tolerance**: Increase if orders still failing, decrease if fills are worse than expected
4. **Emergency scenarios**: Test with higher slippage during forced liquidation

## References

- SDK example: `src/bin/tests/market_order_and_cancel.rs`
- SDK implementation: `src/exchange/exchange_client.rs::market_open()` and `market_close()`
- Order types: `src/exchange/order.rs::MarketOrderParams` and `MarketCloseParams`

## Migration Notes

This change is **backward compatible** for external callers:
- `place_taker_order()` still accepts a `price` parameter (now ignored)
- All error handling remains the same
- Position tracking unchanged
- Logging updated but semantically equivalent

No changes required to code calling these functions.
