# BotRunner Refactoring TODO

## Completed ✅
1. Created `OrderStateManager` library component in `src/order_state.rs`
2. Added exports to `src/lib.rs`
3. Updated BotRunner struct to include `order_state_mgr` field
4. Removed redundant individual tracking fields from BotRunner

## Remaining Work 🚧

### 1. Remove Duplicate Helper Methods
**File**: `src/bin/market_maker_v3.rs`

Remove these methods (now handled by OrderStateManager):
- `add_or_update_order()` (lines ~261-306)
- `remove_and_cache_order()` (lines ~308-344)

### 2. Update `execute_actions()` Method
**File**: `src/bin/market_maker_v3.rs` (lines ~992-1083)

Replace direct field access with OrderStateManager calls:

```rust
// OLD:
self.pending_place_orders.insert(cloid, resting_order);

// NEW:
self.order_state_mgr.add_pending_order(cloid, resting_order);
```

For cancel actions:
```rust
// OLD:
if let Some(order) = list.iter_mut().find(|o| o.oid == Some(oid_to_cancel) && o.state == OrderState::Active) {
    order.state = OrderState::PendingCancel;
    // ...
}

// NEW:
let bids_and_asks = [&mut self.current_state.open_bids, &mut self.current_state.open_asks];
for orders in bids_and_asks {
    if self.order_state_mgr.mark_pending_cancel(oid_to_cancel, orders) {
        executor_actions.push(ExecutorAction::Cancel(cancel.clone()));
        break;
    }
}
```

### 3. Update `handle_order_updates()` Method
**File**: `src/bin/market_maker_v3.rs` (lines ~726-898)

Replace entire method body with delegation to OrderStateManager:

```rust
async fn handle_order_updates(&mut self, updates: Vec<OrderUpdate>) {
    for update in updates {
        let result = self.order_state_mgr.handle_order_update(
            &update,
            self.current_state.order_book.as_ref()
        );

        match result {
            OrderUpdateResult::AddOrUpdate(order) => {
                self.add_order_to_current_state(order);
            }
            OrderUpdateResult::UpdatePartial(order) => {
                self.update_partial_fill_in_state(order);
            }
            OrderUpdateResult::RemoveAndCache(oid, final_state) => {
                self.remove_order_from_current_state(oid);
            }
            OrderUpdateResult::NoAction => {}
        }
    }
}
```

Add helper methods:
```rust
fn add_order_to_current_state(&mut self, order: RestingOrder) {
    let orders = if order.is_buy {
        &mut self.current_state.open_bids
    } else {
        &mut self.current_state.open_asks
    };

    if let Some(existing) = orders.iter_mut().find(|o| o.oid == order.oid) {
        *existing = order;
    } else {
        orders.push(order);
    }

    // Keep sorted
    self.current_state.open_bids.sort_by(|a, b| b.price.partial_cmp(&a.price).unwrap_or(std::cmp::Ordering::Equal));
    self.current_state.open_asks.sort_by(|a, b| a.price.partial_cmp(&b.price).unwrap_or(std::cmp::Ordering::Equal));
}

fn update_partial_fill_in_state(&mut self, order: RestingOrder) {
    let orders = if order.is_buy {
        &mut self.current_state.open_bids
    } else {
        &mut self.current_state.open_asks
    };

    if let Some(existing) = orders.iter_mut().find(|o| o.oid == order.oid) {
        existing.size = order.size;
        existing.state = order.state;
        existing.timestamp = order.timestamp;
    }
}

fn remove_order_from_current_state(&mut self, oid: u64) {
    self.current_state.open_bids.retain(|o| o.oid != Some(oid));
    self.current_state.open_asks.retain(|o| o.oid != Some(oid));
}
```

### 4. Update `handle_user_events()` Fill Handling
**File**: `src/bin/market_maker_v3.rs` (lines ~638-665)

Replace level lookup with OrderStateManager call:

```rust
// OLD:
let open_orders = if is_buy { &self.current_state.open_bids } else { &self.current_state.open_asks };
if let Some(order) = open_orders.iter().find(|o| o.oid == Some(oid)) {
    filled_level = Some(order.level);
}
// ... cache check ...

// NEW:
let all_orders: Vec<RestingOrder> = self.current_state.open_bids.iter()
    .chain(self.current_state.open_asks.iter())
    .cloned()
    .collect();

filled_level = self.order_state_mgr.get_order_level(oid, &all_orders);
```

### 5. Update `handle_tick()` Cache Pruning
**File**: `src/bin/market_maker_v3.rs` (lines ~1252-1264)

Replace manual cache pruning with:

```rust
// Remove this section:
// let now = Instant::now();
// if now.duration_since(self.last_cache_prune_time) ...

// Replace with:
self.order_state_mgr.prune_cache_if_needed();
```

### 6. Update `handle_user_events()` NonUserCancel
**File**: `src/bin/market_maker_v3.rs` (lines ~713-720)

```rust
// OLD:
self.remove_and_cache_order(cancel.oid, OrderState::Cancelled);

// NEW:
let mut all_orders = self.current_state.open_bids.clone();
all_orders.extend(self.current_state.open_asks.clone());
self.order_state_mgr.remove_and_cache_order(cancel.oid, OrderState::Cancelled, &mut all_orders);
// Then update current_state lists from all_orders
```

## Testing Checklist

After completing the refactoring:

- [ ] Code compiles without errors
- [ ] All unit tests in `order_state.rs` pass
- [ ] Manual test: Place order → receives confirmation → appears in active list
- [ ] Manual test: Cancel order → receives confirmation → appears in cache
- [ ] Manual test: Fill arrives → level looked up correctly from cache
- [ ] Manual test: Cache prunes after 30 seconds
- [ ] No "unknown OID" warnings during normal operation

## Benefits of This Refactoring

1. **Reusability**: OrderStateManager can be used by other bots
2. **Testability**: Order tracking logic tested independently
3. **Maintainability**: Changes to order tracking in one place
4. **Clarity**: BotRunner focuses on coordination, not state details
5. **Type Safety**: Clear API prevents misuse of internal state

## Notes

- Keep the `calculate_order_level()` method in BotRunner for now (used during placement)
- Consider moving it to OrderStateManager later
- The library component is complete and tested
- BotRunner integration is straightforward but requires careful method-by-method updates
