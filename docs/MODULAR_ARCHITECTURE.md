# Modular Trading System Architecture

## Overview

This document describes the refactored modular architecture for the trading system. The architecture separates concerns into single-responsibility components, uses event-driven communication, and provides a unified state store.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         Main Strategy                            │
│                   (Orchestrates Components)                      │
└───────────────────┬──────────────────────────────────────────────┘
                    │
        ┌───────────┼───────────┐
        │           │           │
        ▼           ▼           ▼
┌───────────┐ ┌──────────┐ ┌────────────────┐
│  Market   │ │  Signal  │ │  Risk          │
│  Data     │→│Generator │→│  Adjuster      │
│  Pipeline │ │          │ │                │
└───────────┘ └──────────┘ └────────┬───────┘
                                     │
                                     ▼
                            ┌────────────────┐
                            │  Order         │
                            │  Executor      │
                            └────────┬───────┘
                                     │
        ┌────────────────────────────┼────────────────────┐
        │                            │                    │
        ▼                            ▼                    ▼
┌───────────────┐          ┌─────────────────┐  ┌───────────────┐
│  Trading      │◄─────────│  Event Bus      │  │ Order State   │
│  State Store  │          │                 │  │ Machines      │
└───────────────┘          └─────────────────┘  └───────────────┘
```

## Core Components

### 1. Event Bus (`event_bus.rs`)

**Purpose**: Centralized event coordination for system-wide communication.

**Key Features**:
- Publish-subscribe pattern for loose coupling
- Event history for debugging and replay
- Specialized subscribers (logging, metrics)
- No polling required - components react to events

**Events**:
```rust
pub enum TradingEvent {
    PositionChanged { old, new, timestamp },
    OrderFilled { order_id, size, price, is_buy, timestamp },
    OrderCanceled { order_id, timestamp },
    PositionStateChanged { old, new, position, timestamp },
    MarginWarning { available, used, threshold, timestamp },
    EmergencyLiquidation { position, reason, timestamp },
    OrderStuck { order_id, state, duration_secs, timestamp },
    // ...
}
```

**Usage**:
```rust
let event_bus = Arc::new(EventBus::new());

// Subscribe to events
event_bus.subscribe(Arc::new(MetricsSubscriber::new("metrics")));

// Publish events
event_bus.publish(TradingEvent::OrderFilled {
    order_id: 123,
    size: 10.0,
    price: 100.0,
    is_buy: true,
    timestamp: now(),
});
```

### 2. Order State Machine (`order_state_machine.rs`)

**Purpose**: Proper order lifecycle management with timeout handling.

**Key Features**:
- Explicit state transitions with validation
- Timeout detection for stuck orders
- Automatic recovery from problematic states
- Complete transition history

**States**:
```rust
pub enum OrderState {
    Creating { requested_at, client_order_id, size, price, is_buy },
    Open { confirmed_at, order_id, remaining_size, price, is_buy },
    PendingCancel { requested_at, order_id, original_size, remaining_size },
    Canceled { completed_at, order_id, filled_size },
    Filled { completed_at, order_id, fill_size, fill_price },
    PartiallyFilled { completed_at, order_id, filled_size, canceled_size },
    Failed { failed_at, order_id, error },
}
```

**Configuration**:
```rust
let config = StateMachineConfig {
    creation_timeout: Duration::from_secs(10),
    cancel_timeout: Duration::from_secs(5),
    auto_recovery: true,
};
```

**Prevents**:
- Orders stuck in PendingCancel
- Untracked order creations
- Invalid state transitions
- Memory leaks from abandoned orders

### 3. Trading State Store (`trading_state_store.rs`)

**Purpose**: Single source of truth for all trading state.

**Key Features**:
- Thread-safe state access with RwLocks
- Automatic event publishing on state changes
- Snapshot capability for consistent reads
- Order tracking with state machines

**State Types**:
```rust
pub struct MarketData {
    mid_price, best_bid, best_ask, spread_bps,
    volatility_bps, vol_uncertainty_bps,
    imbalance, adverse_selection, timestamp
}

pub struct RiskMetrics {
    position, account_equity, margin_used, margin_available,
    maintenance_margin_ratio, liquidation_distance,
    max_position_size, timestamp
}

pub struct OpenOrder {
    order_id, client_order_id, size, price, is_buy,
    remaining_size, state_machine, created_at
}
```

**Usage**:
```rust
let store = TradingStateStore::new(event_bus.clone());

// Update position (publishes event if changed)
store.update_position(10.0, timestamp);

// Get complete snapshot
let snapshot = store.get_snapshot();

// Add order with state machine
store.add_order(OpenOrder {
    order_id: 123,
    size: 10.0,
    price: 100.0,
    is_buy: true,
    state_machine: OrderStateMachine::from_open_order(...),
    // ...
});

// Tick all state machines for timeout checks
store.tick_order_state_machines(current_time);
```

### 4. Signal Generator (`signal_generator.rs`)

**Purpose**: Pure signal generation without side effects.

**Key Features**:
- Separation of calculation from execution
- Signal caching for performance
- Cache invalidation based on state changes
- No order management logic

**Signal Structure**:
```rust
pub struct QuoteSignal {
    bid_levels: Vec<QuoteLevel>,
    ask_levels: Vec<QuoteLevel>,
    urgency: f64,
    taker_buy_rate: f64,
    taker_sell_rate: f64,
    timestamp: f64,
    metadata: SignalMetadata,
}

pub struct QuoteLevel {
    offset_bps: f64,  // Offset from mid price
    size: f64,
    urgency: f64,     // Priority (0-1)
    is_bid: bool,
}
```

**Usage**:
```rust
let mut generator = HjbSignalGenerator::new(
    quote_optimizer,
    volatility_model,
    microprice_as_model,
    hawkes_model,
    lambda_base,
    phi,
    maker_fee_bps,
    taker_fee_bps,
);

// Generate pure signal
let signal = generator.generate_quotes(&market_data, inventory);
```

### 5. Risk Adjuster (`risk_adjuster.rs`)

**Purpose**: Apply risk-based adjustments to pure signals.

**Key Features**:
- Position state-based adjustments
- Margin constraint enforcement
- Size reduction for risk limits
- Emergency liquidation signals

**Adjustment Logic**:
```rust
match position_state {
    Normal => apply_margin_checks(signal),
    Warning => reduce_sizes_and_be_conservative(signal),
    Critical => force_position_reduction(),
    OverLimit => emergency_liquidation(),
}
```

**Usage**:
```rust
let adjuster = RiskAdjuster::new(
    position_manager,
    margin_calculator,
    min_order_size,
);

let adjusted_signal = adjuster.adjust_signal(signal, &snapshot);
```

### 6. Order Executor (`order_executor.rs`)

**Purpose**: Handle order placement and reconciliation.

**Key Features**:
- Reconciles desired vs actual state
- Minimizes order churn
- Price/size rounding
- Taker order execution

**Reconciliation**:
- Identifies orders to cancel (wrong price/size)
- Identifies missing orders to create
- Respects requote thresholds
- Handles exchange rate limits

**Usage**:
```rust
let executor = OrderExecutor::new(
    state_store.clone(),
    event_bus.clone(),
    asset,
    tick_size,
    lot_size,
    requote_threshold_bps,
);

let actions = executor.execute(adjusted_signal, &snapshot)?;
```

### 7. Market Data Pipeline (`market_data_pipeline.rs`)

**Purpose**: Process market updates through specialized processors.

**Key Features**:
- Chain of responsibility pattern
- Specialized processors for each metric
- Extensible processor architecture

**Processors**:
```rust
pub trait MarketDataProcessor {
    fn process(&mut self, data: &mut ProcessedMarketData);
    fn name(&self) -> &str;
}

// Example processors:
- ImbalanceProcessor: Calculates order book imbalance
- VolatilityProcessor: Updates volatility estimates
- AdverseSelectionProcessor: Estimates adverse selection
```

**Usage**:
```rust
let mut pipeline = MarketDataPipeline::new();
pipeline.add_processor(Box::new(ImbalanceProcessor::new()));
pipeline.add_processor(Box::new(VolatilityProcessor::new(vol_model)));
pipeline.add_processor(Box::new(AdverseSelectionProcessor::new()));

let processed = pipeline.process(market_update);
```

## Integration Example

Here's how a complete strategy would use these components:

```rust
pub struct ModularHjbStrategy {
    // State management
    state_store: Arc<TradingStateStore>,
    event_bus: Arc<EventBus>,

    // Processing pipeline
    market_pipeline: MarketDataPipeline,

    // Signal generation and adjustment
    signal_generator: HjbSignalGenerator,
    risk_adjuster: RiskAdjuster,

    // Execution
    order_executor: OrderExecutor,
}

impl Strategy for ModularHjbStrategy {
    fn on_market_update(&mut self, state: &CurrentState, update: &MarketUpdate)
        -> Vec<StrategyAction>
    {
        // 1. Process market data
        let processed_market = self.market_pipeline.process(update.clone());
        self.state_store.update_market_data(processed_market.to_market_data());

        // 2. Update position
        self.state_store.update_position(state.position, processed_market.timestamp);

        // 3. Generate pure signal
        let signal = self.signal_generator.generate_quotes(
            &processed_market.to_market_data(),
            state.position,
        );

        // 4. Apply risk adjustments
        let snapshot = self.state_store.get_snapshot();
        let adjusted = self.risk_adjuster.adjust_signal(signal, &snapshot);

        // 5. Execute (convert to actions)
        match self.order_executor.execute(adjusted, &snapshot) {
            Ok(actions) => actions,
            Err(e) => {
                warn!("Execution error: {}", e);
                Vec::new()
            }
        }
    }

    fn on_user_update(&mut self, update: &UserUpdate) {
        match update {
            UserUpdate::OrderFilled { order_id, filled_size, fill_price, is_buy, .. } => {
                // Publish event
                self.event_bus.publish(TradingEvent::OrderFilled {
                    order_id: *order_id,
                    size: *filled_size,
                    price: *fill_price,
                    is_buy: *is_buy,
                    timestamp: now(),
                });

                // Update order state machine
                if let Some(mut order) = self.state_store.get_order(*order_id) {
                    order.state_machine.transition(OrderEvent::FullFill {
                        filled_size: *filled_size,
                        fill_price: *fill_price,
                    }).ok();
                    self.state_store.remove_order(*order_id);
                }
            },
            // Handle other updates...
            _ => {}
        }
    }
}
```

## Benefits

### Testability
Each component can be tested in isolation:
```rust
#[test]
fn test_signal_generator() {
    let generator = create_test_generator();
    let signal = generator.generate_quotes(&test_market_data(), 0.0);
    assert!(signal.theoretical_spread_bps() > 0.0);
}
```

### Maintainability
- Clear boundaries between components
- Single responsibility per component
- Easy to locate bugs (check relevant component)

### Performance
- No redundant calculations (caching)
- Proper event-driven architecture (no polling)
- Efficient state management (RwLocks)

### Reliability
- Proper state machines prevent stuck orders
- Timeout handling for all async operations
- Event history for debugging

### Observability
- Event bus provides complete audit trail
- Metrics subscribers track key statistics
- Easy to add monitoring/alerting

### Flexibility
- Easy to swap implementations (e.g., different vol models)
- Component-based configuration
- Extensible processor architecture

## Migration Strategy

### Phase 1: Foundation (Complete)
- ✅ Event bus implementation
- ✅ Order state machine
- ✅ Trading state store
- ✅ Module structure

### Phase 2: Core Components (Complete)
- ✅ Signal generator
- ✅ Risk adjuster
- ✅ Order executor
- ✅ Market data pipeline

### Phase 3: Integration (TODO)
- Adapt existing HjbStrategy to use new components
- Fix API mismatches (OptimizerInputs, AllowedAction, etc.)
- Update unit tests
- Integration testing

### Phase 4: Refinement (TODO)
- Performance optimization
- Enhanced error handling
- Comprehensive monitoring
- Documentation and examples

## Known Issues and TODOs

### Type Mismatches
Several components reference types/methods that need adjustment:
- `OptimizerInputs` field names
- `AllowedAction` enum variants
- `PositionManager` method signatures
- `StrategyAction` variants

### Missing Implementations
- Taker order execution in OrderExecutor
- Complete market data processor implementations
- Integration with existing performance tracking
- Comprehensive error recovery

### Testing
- Unit tests for each component (partial)
- Integration tests for component interactions
- Performance benchmarks
- Stress testing

## Configuration

Example configuration for modular strategy:

```json
{
  "strategy_params": {
    "event_bus": {
      "max_history": 1000
    },
    "state_machine": {
      "creation_timeout_secs": 10,
      "cancel_timeout_secs": 5,
      "auto_recovery": true
    },
    "risk_adjuster": {
      "min_order_size": 0.01,
      "margin_safety_buffer": 0.2,
      "leverage": 5
    },
    "order_executor": {
      "requote_threshold_bps": 2.0
    },
    "market_data_pipeline": {
      "processors": ["imbalance", "volatility", "adverse_selection"]
    }
  }
}
```

## Performance Considerations

### Caching
- Signal generator caches results
- Cache invalidation on significant state changes
- Balance between freshness and performance

### Lock Contention
- RwLocks favor readers (many reads, few writes)
- State updates are batched where possible
- Critical sections are minimized

### Event Processing
- Events are processed synchronously
- Consider async event bus for high-frequency systems
- Event history bounded to prevent memory growth

## Debugging

### Event History
```rust
// Get recent events
let events = event_bus.get_history(100);

// Get specific event types
let fills = event_bus.get_events_of_type("OrderFilled");
```

### State Machine History
```rust
// Check order state transitions
let transitions = order.state_machine.get_transitions();
for t in transitions {
    println!("{} -> {} via {}", t.from_state, t.to_state, t.event);
}
```

### State Snapshots
```rust
// Get consistent view of all state
let snapshot = state_store.get_snapshot();
println!("Position: {}, Open Orders: {}",
    snapshot.risk_metrics.position,
    snapshot.open_orders.len());
```

## Conclusion

This modular architecture provides a solid foundation for a professional, maintainable, and reliable trading system. The separation of concerns, event-driven communication, and proper state management address the issues in the monolithic design while improving testability, observability, and performance.

The architecture is designed to scale - components can be moved to separate processes/servers if needed, and the event bus can be replaced with a distributed message queue for larger deployments.
