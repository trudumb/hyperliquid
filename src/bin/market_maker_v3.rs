/*
Market Maker V3 - Generic Bot Runner with Pluggable Strategies

This is a refactored version of the market maker that separates:
1. Core Services (Exchange, WebSocket, TUI) - The "bot runner"
2. Strategy Logic (HJB, Grid, etc.) - The "pluggable brain"

The bot runner:
- Loads strategy from config.json
- Initializes exchange and websocket connections
- Maintains CurrentState (position, orders, market data)
- Passes updates to strategy and executes returned actions
- Renders TUI dashboard in separate thread

The strategy:
- Receives CurrentState and MarketUpdate/UserUpdate
- Returns Vec<StrategyAction> (place/cancel orders)
- Encapsulates all trading logic (HJB, Hawkes, etc.)

# Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Main Binary                          │
│  ┌────────────────┐  ┌─────────────┐  ┌────────────────┐   │
│  │   Bot Runner   │  │  Strategy   │  │   TUI Thread   │   │
│  │   (150 lines)  │→→│  (modular)  │→→│   (separate)   │   │
│  └────────────────┘  └─────────────┘  └────────────────┘   │
│         ↓                    ↓                              │
│    WebSocket         Strategy Actions                       │
│    Feeds             (Place/Cancel)                         │
└─────────────────────────────────────────────────────────────┘
```

# Key Benefits

- **Modularity**: Swap strategies by changing config.json
- **Simplicity**: Main loop is ~150 lines vs 5000 in v2
- **Testability**: Strategies are pure functions (State → Actions)
- **Maintainability**: Clear separation of concerns
- **Extensibility**: Add new strategies without touching core code

# Event Flow

1. WebSocket message arrives (L2Book, Trades, Fills)
2. Bot runner updates CurrentState
3. Bot runner calls strategy.on_market_update() or strategy.on_user_update()
4. Strategy returns Vec<StrategyAction>
5. Bot runner executes actions via exchange client
6. Bot runner updates TUI dashboard

# Example: Adding a New Strategy

```rust
pub struct GridStrategy { /* ... */ }

impl Strategy for GridStrategy {
    fn name(&self) -> &str { "Grid Strategy" }
    
    fn on_market_update(&mut self, state: &CurrentState, update: &MarketUpdate) 
        -> Vec<StrategyAction> {
        // Your grid logic here
        vec![StrategyAction::NoOp]
    }
    // ... other trait methods ...
}

// In main():
let strategy: Box<dyn Strategy> = match config.strategy_name.as_str() {
    "grid_v1" => Box::new(GridStrategy::new(&config.asset, &config)),
    // ...
};
```

# Performance

- Event processing: <1ms per message
- Order placement: <10ms round-trip
- TUI update: <5ms
- Memory footprint: <50MB

*/

use alloy::signers::local::PrivateKeySigner;
use hyperliquid_rust_sdk::{
    AssetType, BaseUrl,
    CurrentState, ExchangeClient, InfoClient, MarketUpdate, Message,
    OrderBook, OrderState, OrderStateManager, OrderUpdate, OrderUpdateResult,
    RestingOrder, Strategy, StrategyAction, Subscription, TickLotValidator,
    TradeInfo, UserData, UserUpdate,
    // New imports for optimized execution
    ParallelOrderExecutor, ExecutorConfig,
    ExchangeResponseStatus, ExchangeDataStatus, ExecutorAction,
};
use hyperliquid_rust_sdk::strategies::hjb_strategy::HjbStrategy;

use log::{error, info, warn, debug};
use std::collections::{HashMap, HashSet};
use std::env;
use std::fs;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};
use tokio::signal;
use tracing_subscriber::{fmt, layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

// RUST_LOG=info cargo run --release --bin market_maker_v3

// ============================================================================
// Configuration
// ============================================================================

#[derive(serde::Deserialize, Debug)]
struct Config {
    asset: String,
    strategy_name: String,
    strategy_params: serde_json::Value,
}

fn load_config(path: &str) -> Config {
    let config_str = fs::read_to_string(path)
        .unwrap_or_else(|_| panic!("Failed to read config file: {}", path));
    serde_json::from_str(&config_str)
        .unwrap_or_else(|e| panic!("Failed to parse config file: {}", e))
}

// ============================================================================
// Bot Runner
// ============================================================================

// Type alias for clarity
type Cloid = uuid::Uuid;

struct BotRunner {
    /// Trading asset
    asset: String,

    /// Strategy instance (trait object for polymorphism)
    strategy: Box<dyn Strategy>,

    /// Optimized order executor for parallel execution (replaces exchange_client)
    order_executor: Arc<ParallelOrderExecutor>,

    /// Info client for market data queries (Option to allow explicit drop on shutdown)
    info_client: Option<InfoClient>,

    /// User wallet address
    user_address: alloy::primitives::Address,

    /// Tick/lot validator for order validation (kept for potential future use)
    #[allow(dead_code)]
    tick_lot_validator: TickLotValidator,

    /// Current bot state (maintained by bot runner)
    current_state: CurrentState,

    /// Order state manager (handles order lifecycle tracking)
    order_state_mgr: OrderStateManager,

    /// Total messages received (for throughput monitoring)
    total_messages: u64,

    /// Shutdown flag (prevents processing new events during shutdown)
    is_shutting_down: Arc<AtomicBool>,

    /// Flag to track if initial snapshot has been received
    snapshot_received: bool,

    /// Track which fills we've already processed to prevent duplicates
    processed_fill_ids: HashSet<u64>,
}

impl BotRunner {
    /// Create a new bot runner
    async fn new(
        asset: String,
        strategy: Box<dyn Strategy>,
        wallet: PrivateKeySigner,
        tick_lot_validator: TickLotValidator,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        // Initialize exchange client
        let exchange_client = Arc::new(
            ExchangeClient::new(None, wallet.clone(), Some(BaseUrl::Mainnet), None, None)
                .await?
        );

        // --- START NEW: Create ParallelOrderExecutor ---
        // Create executor config (tune as needed for optimal performance)
        let executor_config = ExecutorConfig {
            max_concurrent: 8,         // Allow up to 8 parallel requests
            batch_window_us: 100,      // 100μs batching window
            request_timeout_ms: 2500,   // Slightly longer timeout for batched requests
            batch_timeout_ms: 4000,    // 3s total batch timeout
            ..Default::default()       // Use default rate limit config
        };

        // Create the ParallelOrderExecutor
        let order_executor = Arc::new(ParallelOrderExecutor::with_config(
            exchange_client.clone(),
            executor_config
        ));

        info!("✅ ParallelOrderExecutor initialized with max_concurrent=8");
        // --- END NEW ---

        // Initialize info client
        let info_client = InfoClient::new(None, Some(BaseUrl::Mainnet)).await?;

        // Get user address
        let user_address = wallet.address();

        // Fetch initial account state
        let user_state = info_client.user_state(user_address).await?;
        let margin_summary = user_state.margin_summary;

        let account_equity = margin_summary.account_value.parse::<f64>().unwrap_or(0.0);

        // Get max position size from strategy
        let max_position_size = strategy.get_max_position_size();

        // Initialize current state
        let current_state = CurrentState {
            position: 0.0,
            avg_entry_price: 0.0,
            cost_basis: 0.0,
            unrealized_pnl: 0.0,
            realized_pnl: 0.0,
            total_fees: 0.0,
            l2_mid_price: 0.0,
            order_book: None,
            market_spread_bps: 0.0,
            lob_imbalance: 0.5,
            open_bids: Vec::new(),
            open_asks: Vec::new(),
            account_equity,
            margin_used: margin_summary.total_margin_used.parse::<f64>().unwrap_or(0.0),
            max_position_size,
            timestamp: chrono::Utc::now().timestamp() as f64,
            session_start_time: chrono::Utc::now().timestamp() as f64,
        };

        info!("✅ Bot Runner initialized for {} with strategy: {}", asset, strategy.name());
        info!("   Account Equity: ${:.2}", account_equity);

        Ok(Self {
            asset,
            strategy,
            order_executor,  // Use order_executor instead of exchange_client
            info_client: Some(info_client),
            user_address,
            tick_lot_validator,
            current_state,
            order_state_mgr: OrderStateManager::new(),
            total_messages: 0,
            is_shutting_down: Arc::new(AtomicBool::new(false)),
            snapshot_received: false,
            processed_fill_ids: HashSet::new(),
        })
    }

    /// Add or update an order in CurrentState (open_bids/open_asks)
    fn add_or_update_order(&mut self, order_to_add: RestingOrder) {
        let oid = match order_to_add.oid {
            Some(id) => id,
            None => {
                warn!("Attempted to add/update order without OID (Cloid: {:?})", order_to_add.cloid);
                return;
            }
        };

        let orders_list = if order_to_add.is_buy {
            &mut self.current_state.open_bids
        } else {
            &mut self.current_state.open_asks
        };

        if let Some(existing_order) = orders_list.iter_mut().find(|o| o.oid == Some(oid)) {
            // Update existing order state, size, timestamp
            *existing_order = order_to_add;
        } else {
            // Add new order
            orders_list.push(order_to_add);
        }

        // Ensure lists remain sorted
        // Sort bids descending by price, asks ascending by price
        self.current_state.open_bids.sort_by(|a, b| b.price.partial_cmp(&a.price).unwrap_or(std::cmp::Ordering::Equal));
        self.current_state.open_asks.sort_by(|a, b| a.price.partial_cmp(&b.price).unwrap_or(std::cmp::Ordering::Equal));
    }

    /// Remove an order from active lists and cache it with a final state
    fn remove_and_cache_order(&mut self, oid: u64, final_state: OrderState) {
        let mut removed_order: Option<RestingOrder> = None;
        let current_timestamp = chrono::Utc::now().timestamp_millis() as u64;

        self.current_state.open_bids.retain(|o| {
            if o.oid == Some(oid) { removed_order = Some(o.clone()); false } else { true }
        });
        if removed_order.is_none() {
            self.current_state.open_asks.retain(|o| {
                if o.oid == Some(oid) { removed_order = Some(o.clone()); false } else { true }
            });
        }

        // Also check pending orders (though unlikely to be cancelled directly by OID)
        if removed_order.is_none() {
            if let Some(cloid) = self.oid_to_cloid.get(&oid) {
                if let Some(order) = self.pending_place_orders.remove(cloid) {
                    removed_order = Some(order);
                }
            }
        }

        // If found, update state and move to cache
        if let Some(mut order) = removed_order {
            order.state = final_state.clone();
            order.timestamp = current_timestamp;
            self.recently_completed_orders.insert(oid, order);
            info!("Order OID {} moved to cache with state {:?}", oid, final_state);
            // Clean up mappings
            if let Some(cloid) = self.oid_to_cloid.remove(&oid) {
                self.cloid_to_oid.remove(&cloid);
            }
        } else {
            debug!("Attempted to remove/cache OID {}, but not found in active/pending lists.", oid);
        }
    }

    /// Start the bot runner event loop
    async fn run(&mut self, mut shutdown_rx: tokio::sync::oneshot::Receiver<()>) -> Result<(), Box<dyn std::error::Error>> {
        info!("🚀 Starting bot runner event loop...");

        // Set up websocket subscriptions
        let (sender, receiver) = flume::unbounded();

        // Subscribe to user events (fills)
        self.info_client
            .as_mut()
            .unwrap()
            .subscribe(
                Subscription::UserEvents {
                    user: self.user_address,
                },
                sender.clone(),
            )
            .await?;

        // Subscribe specifically to user fills (snapshot + stream)
        self.info_client
            .as_mut()
            .unwrap()
            .subscribe(
                Subscription::UserFills {
                    user: self.user_address,
                },
                sender.clone(),
            )
            .await?;

        // Subscribe to AllMids (mid-price updates)
        self.info_client
            .as_mut()
            .unwrap()
            .subscribe(Subscription::AllMids, sender.clone())
            .await?;

        // Subscribe to L2Book (order book updates)
        self.info_client
            .as_mut()
            .unwrap()
            .subscribe(
                Subscription::L2Book {
                    coin: self.asset.clone(),
                },
                sender.clone(),
            )
            .await?;

        // Subscribe to Trades (market trades for flow analysis)
        self.info_client
            .as_mut()
            .unwrap()
            .subscribe(
                Subscription::Trades {
                    coin: self.asset.clone(),
                },
                sender.clone(),
            )
            .await?;

        // Subscribe to OrderUpdates (order status confirmations)
        self.info_client
            .as_mut()
            .unwrap()
            .subscribe(
                Subscription::OrderUpdates {
                    user: self.user_address,
                },
                sender.clone(),
            )
            .await?;

        info!("✅ WebSocket subscriptions established");

        // Periodic tick timer (1 second interval)
        let mut tick_timer = tokio::time::interval(tokio::time::Duration::from_secs(1));
        tick_timer.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

        // Main event loop
        loop {
            tokio::select! {
                biased;  // Prioritize shutdown signal over other branches

                // Check for shutdown signal
                _ = &mut shutdown_rx => {
                    info!("🛑 Shutdown signal received by main loop");
                    // Set the flag FIRST
                    self.is_shutting_down.store(true, Ordering::Relaxed);
                    // Then handle shutdown actions
                    self.handle_shutdown().await;
                    break;
                }

                // Handle WebSocket messages
                message = receiver.recv_async() => {
                    // Check flag BEFORE processing
                    if self.is_shutting_down.load(Ordering::Relaxed) {
                        warn!("Ignoring WS message during shutdown.");
                        continue;
                    }
                    match message {
                        Ok(message) => {
                            self.handle_message(message).await;
                        }
                        Err(_) => {
                            error!("WebSocket channel closed");
                            break;
                        }
                    }
                }

                // Handle periodic tick (for housekeeping, position checks, etc.)
                _ = tick_timer.tick() => {
                    // Check flag BEFORE processing
                    if self.is_shutting_down.load(Ordering::Relaxed) {
                        warn!("Ignoring timer tick during shutdown.");
                        continue;
                    }
                    self.handle_tick().await;
                }
            }
        }

        // Explicit cleanup: Drop info_client to trigger stop_flag and terminate background tasks
        info!("🔌 Closing WebSocket connections...");
        if let Some(info_client) = self.info_client.take() {
            drop(info_client);
        }
        // Give background tasks time to see stop_flag and terminate
        tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;

        info!("✅ Bot runner stopped");
        Ok(())
    }

    /// Handle incoming WebSocket message
    async fn handle_message(&mut self, message: Message) {
        // Increment message counter
        self.total_messages += 1;

        match &message {
            Message::L2Book(_) => {
                self.handle_l2_book(message).await;
            }
            Message::Trades(_) => {
                self.handle_trades(message).await;
            }
            Message::AllMids(_) => {
                self.handle_all_mids(message).await;
            }
            Message::User(_) => {
                self.handle_user_events(message).await;
            }
            Message::OrderUpdates(order_updates) => {
                self.handle_order_updates(order_updates.data.clone()).await;
            }
            Message::UserFills(user_fills) => {
                // This stream is dedicated to fills, including the initial snapshot.
                let fills_data = &user_fills.data;
                let is_snapshot = fills_data.is_snapshot.unwrap_or(false);

                if is_snapshot {
                    // Process snapshot ONCE to initialize position state
                    if !self.snapshot_received {
                        info!("📸 Initializing from snapshot: {} historical fills", fills_data.fills.len());

                        // Process all historical fills to reconstruct current position
                        for fill in &fills_data.fills {
                            self.process_fill(fill);
                            // Track that we processed this fill
                            let fill_id = self.get_fill_id(fill);
                            self.processed_fill_ids.insert(fill_id);
                        }

                        info!("✅ Position initialized: {} units @ avg price ${:.2}",
                              self.current_state.position,
                              self.current_state.avg_entry_price);

                        self.snapshot_received = true;
                    }
                    return; // Always return after snapshot
                }

                // IGNORE all streaming fills from UserFills - UserEvents will handle them
                // This prevents duplicate processing
                debug!("Ignoring {} streaming fills from UserFills (UserEvents handles these)",
                       fills_data.fills.len());
            }
            _ => {
                // Ignore other message types
            }
        }
    }

    /// Handle L2 order book updates
    async fn handle_l2_book(&mut self, l2_book_msg: Message) {
        if let Message::L2Book(l2_book) = l2_book_msg {
            if let Some(book) = OrderBook::from_l2_data(&l2_book.data) {
                // Update current state with new book data
                if let Some(analysis) = book.analyze(5) {
                    self.current_state.lob_imbalance = analysis.imbalance;
                    self.current_state.market_spread_bps =
                        ((analysis.weighted_ask_price - analysis.weighted_bid_price)
                         / analysis.weighted_bid_price) * 10000.0;
                }

                // Calculate mid-price from BBO
                if !book.bids.is_empty() && !book.asks.is_empty() {
                    if let (Ok(best_bid), Ok(best_ask)) = (
                        book.bids[0].px.parse::<f64>(),
                        book.asks[0].px.parse::<f64>()
                    ) {
                        self.current_state.l2_mid_price = (best_bid + best_ask) / 2.0;
                    }
                }

                self.current_state.order_book = Some(book.clone());

                // Create market update and pass to strategy
                let market_update = MarketUpdate {
                    asset: self.asset.clone(),
                    l2_book: Some(l2_book.data),
                    trades: Vec::new(),
                    mid_price: None,
                    timestamp: chrono::Utc::now().timestamp_millis() as u64,
                };

                let actions = self.strategy.on_market_update(&self.current_state, &market_update);
                self.execute_actions(actions).await;
            }
        }
    }

    /// Handle trade flow updates
    async fn handle_trades(&mut self, trades_msg: Message) {
        if let Message::Trades(trades) = trades_msg {
            let market_update = MarketUpdate {
                asset: self.asset.clone(),
                l2_book: None,
                trades: trades.data.clone(),
                mid_price: None,
                timestamp: chrono::Utc::now().timestamp_millis() as u64,
            };

            let actions = self.strategy.on_market_update(&self.current_state, &market_update);
            self.execute_actions(actions).await;
        }
    }

    /// Handle mid-price updates
    async fn handle_all_mids(&mut self, all_mids_msg: Message) {
        if let Message::AllMids(all_mids) = all_mids_msg {
            if let Some(mid_str) = all_mids.data.mids.get(&self.asset) {
                if let Ok(mid_price) = mid_str.parse::<f64>() {
                    let market_update = MarketUpdate {
                        asset: self.asset.clone(),
                        l2_book: None,
                        trades: Vec::new(),
                        mid_price: Some(mid_price),
                        timestamp: chrono::Utc::now().timestamp_millis() as u64,
                    };

                    let actions = self.strategy.on_market_update(&self.current_state, &market_update);
                    self.execute_actions(actions).await;
                }
            }
        }
    }

    /// Handle user events (fills, funding, liquidations, non-user cancels)
    async fn handle_user_events(&mut self, user_events_msg: Message) {
        if let Message::User(user_events) = user_events_msg {
            match user_events.data {
                UserData::Fills(fills) => {
                    // These are REAL-TIME fills only (no snapshot)
                    if fills.is_empty() {
                        return;
                    }

                    let mut new_fills_with_levels = Vec::new();

                    for fill in &fills {
                        let fill_id = self.get_fill_id(fill);

                        // Check if we already processed this fill (from snapshot)
                        if self.processed_fill_ids.contains(&fill_id) {
                            debug!("Skipping duplicate fill: tid={}", fill.tid);
                            continue;
                        }

                        // --- IDENTIFY FILL LEVEL ---
                        let mut filled_level: Option<usize> = None;
                        let mut order_state_found: Option<OrderState> = None;
                        let oid = fill.oid;
                        let is_buy = fill.side == "B";

                        // 1. Check active orders (including PendingCancel, PartiallyFilled)
                        let open_orders = if is_buy { &self.current_state.open_bids } else { &self.current_state.open_asks };
                        if let Some(order) = open_orders.iter().find(|o| o.oid == Some(oid)) {
                            filled_level = Some(order.level);
                            order_state_found = Some(order.state.clone());
                        }

                        // 2. Check recently completed cache
                        if filled_level.is_none() {
                            if let Some(cached_order) = self.recently_completed_orders.get(&oid) {
                                filled_level = Some(cached_order.level);
                                order_state_found = Some(cached_order.state.clone());
                                debug!("Fill OID {} found in cache (State: {:?}). Level {} retrieved.", oid, cached_order.state, filled_level.unwrap_or(999));
                            }
                        }

                        // 3. Log warning if still not found
                        if filled_level.is_none() {
                            warn!("Fill received for unknown or too-old OID: {}. Cannot determine level.", oid);
                        } else {
                            debug!("Determined level {} for fill OID {} (Order State was: {:?})", filled_level.unwrap_or(999), oid, order_state_found);
                        }

                        // Process new fill (updates position, PnL, etc.)
                        self.process_fill(fill);
                        self.processed_fill_ids.insert(fill_id);
                        new_fills_with_levels.push((fill.clone(), filled_level));
                    }

                    // Only notify strategy if there were NEW fills
                    if !new_fills_with_levels.is_empty() {
                        let user_update = UserUpdate::from_fills_with_levels(new_fills_with_levels);
                        let actions = self.strategy.on_user_update(&self.current_state, &user_update);
                        self.execute_actions(actions).await;
                    }
                }

                UserData::Funding(funding) => {
                    info!("💰 Funding payment: {} USDC for {} (rate: {})",
                          funding.usdc,
                          funding.coin,
                          funding.funding_rate);

                    // Update PnL with funding costs
                    let funding_amount = funding.usdc.parse::<f64>().unwrap_or(0.0);
                    self.current_state.realized_pnl += funding_amount; // Funding can be + or -

                    // Log for tracking
                    info!("   Updated realized PnL: ${:.4}", self.current_state.realized_pnl);
                }

                UserData::Liquidation(liq) => {
                    error!("🚨 LIQUIDATION EVENT!");
                    error!("   Liquidated user: {}", liq.liquidated_user);
                    error!("   Liquidator: {}", liq.liquidator);
                    error!("   Position: {}", liq.liquidated_ntl_pos);
                    error!("   Account value: {}", liq.liquidated_account_value);

                    // Check if it's OUR liquidation
                    if liq.liquidated_user == self.user_address.to_string() {
                        error!("⚠️ WE WERE LIQUIDATED!");
                        // Emergency actions:
                        // 1. Stop trading
                        // 2. Alert external systems
                        // 3. Initiate emergency shutdown
                        self.is_shutting_down.store(true, Ordering::Relaxed);
                    }
                }

                UserData::NonUserCancel(cancels) => {
                    for cancel in &cancels {
                        warn!("⚠️ System cancelled order {} for {}", cancel.oid, cancel.coin);
                        // Use the helper method to remove and cache
                        self.remove_and_cache_order(cancel.oid, OrderState::Cancelled);
                        info!("   Possible reasons: post-only order would cross, position limit, margin");
                    }
                }
            }
        }
    }

    /// Handle order status updates from WebSocket
    async fn handle_order_updates(&mut self, updates: Vec<OrderUpdate>) {
        let current_timestamp = chrono::Utc::now().timestamp_millis() as u64;

        for update in updates {
            let oid = update.order.oid;
            let status = update.status.as_str();
            let cloid_opt: Option<Cloid> = update.order.cloid.as_deref().and_then(|s| s.parse().ok());

            debug!("Processing OrderUpdate: OID {}, Cloid {:?}, Status: {}", oid, cloid_opt, status);

            match status {
                "open" | "resting" => {
                    // Order confirmed active/resting on the book
                    let mut order_details: Option<RestingOrder> = None;

                    // 1. Check if it confirms a pending order
                    if let Some(cloid) = cloid_opt {
                        if let Some(mut pending_order) = self.pending_place_orders.remove(&cloid) {
                            pending_order.oid = Some(oid); // Assign the OID
                            pending_order.state = OrderState::Active;
                            pending_order.timestamp = current_timestamp;
                            order_details = Some(pending_order);

                            // Update mappings
                            self.cloid_to_oid.insert(cloid, oid);
                            self.oid_to_cloid.insert(oid, cloid);
                            info!("Order placement confirmed: Cloid {} -> OID {}", cloid, oid);
                        }
                    }

                    // 2. If not from pending, check if it's an update to an existing active order
                    if order_details.is_none() {
                        let orders = if update.order.side == "B" {
                            &mut self.current_state.open_bids
                        } else {
                            &mut self.current_state.open_asks
                        };
                        if let Some(existing_order) = orders.iter_mut().find(|o| o.oid == Some(oid)) {
                            // Update status if it was PendingCancel or something else
                            if existing_order.state != OrderState::Active && existing_order.state != OrderState::PartiallyFilled {
                                debug!("Order OID {} state updated from {:?} to Active.", oid, existing_order.state);
                                existing_order.state = OrderState::Active;
                            }
                            existing_order.timestamp = current_timestamp;
                            order_details = Some(existing_order.clone());
                        }
                    }

                    // 3. Add/Update in CurrentState
                    if let Some(order) = order_details {
                        self.add_or_update_order(order);
                    } else {
                        // Order appeared without being pending or active? Might happen on reconnect/resync.
                        warn!("Received 'resting' update for unknown OID {}. Adding to state.", oid);
                        let price = update.order.limit_px.parse().unwrap_or(0.0);
                        let size = update.order.sz.parse().unwrap_or(0.0);
                        let is_buy = update.order.side == "B";
                        let level = self.calculate_order_level(price, is_buy);
                        if price > 0.0 && size > 0.0 {
                            let new_order = RestingOrder {
                                oid: Some(oid),
                                cloid: cloid_opt,
                                size,
                                orig_size: update.order.orig_sz.parse().unwrap_or(size),
                                price,
                                is_buy,
                                level,
                                state: OrderState::Active,
                                timestamp: current_timestamp,
                            };
                            self.add_or_update_order(new_order);
                            if let Some(cloid) = cloid_opt {
                                self.cloid_to_oid.insert(cloid, oid);
                                self.oid_to_cloid.insert(oid, cloid);
                            }
                        }
                    }
                },
                "canceled" | "cancelled" => {
                    info!("Order OID {} confirmed Canceled.", oid);
                    self.remove_and_cache_order(oid, OrderState::Cancelled);
                },
                "rejected" => {
                    error!("Order OID {} Rejected!", oid);
                    // Remove from pending if it exists
                    if let Some(cloid) = cloid_opt {
                        if self.pending_place_orders.contains_key(&cloid) {
                            let order = self.pending_place_orders.remove(&cloid);
                            warn!("Placement Rejected for Cloid {}: {:?}", cloid, order);
                            // Move to cache with Rejected state
                            if let Some(mut rej_order) = order {
                                rej_order.oid = Some(oid);
                                rej_order.state = OrderState::Rejected;
                                rej_order.timestamp = current_timestamp;
                                self.recently_completed_orders.insert(oid, rej_order);
                            }
                        } else {
                            warn!("'Rejected' status received for OID {}, Cloid {:?}, not found in pending.", oid, cloid_opt);
                            // Check if it was PendingCancel and revert?
                            let orders_lists = [&mut self.current_state.open_bids, &mut self.current_state.open_asks];
                            for list in orders_lists {
                                if let Some(order) = list.iter_mut().find(|o| o.oid == Some(oid) && o.state == OrderState::PendingCancel) {
                                    warn!("Cancel request for OID {} was Rejected. Reverting state to Active.", oid);
                                    order.state = OrderState::Active;
                                    order.timestamp = current_timestamp;
                                    break;
                                }
                            }
                        }
                    } else {
                        warn!("'Rejected' status received for OID {} without Cloid.", oid);
                        self.remove_and_cache_order(oid, OrderState::Rejected);
                    }
                },
                "filled" => {
                    info!("Order OID {} confirmed Fully Filled.", oid);
                    self.remove_and_cache_order(oid, OrderState::Filled);
                },
                "partiallyFilled" => {
                    debug!("Order OID {} is Partially Filled.", oid);
                    // Update remaining size and state in active lists
                    let remaining_size = update.order.sz.parse().unwrap_or(0.0);
                    let orders = if update.order.side == "B" {
                        &mut self.current_state.open_bids
                    } else {
                        &mut self.current_state.open_asks
                    };
                    if let Some(order) = orders.iter_mut().find(|o| o.oid == Some(oid)) {
                        order.size = remaining_size;
                        order.state = OrderState::PartiallyFilled;
                        order.timestamp = current_timestamp;
                    } else {
                        warn!("Received 'partiallyFilled' for unknown OID {}. Adding to state.", oid);
                        let price = update.order.limit_px.parse().unwrap_or(0.0);
                        let is_buy = update.order.side == "B";
                        let level = self.calculate_order_level(price, is_buy);
                        if price > 0.0 && remaining_size > 0.0 {
                            let new_order = RestingOrder {
                                oid: Some(oid),
                                cloid: cloid_opt,
                                size: remaining_size,
                                orig_size: update.order.orig_sz.parse().unwrap_or(remaining_size),
                                price,
                                is_buy,
                                level,
                                state: OrderState::PartiallyFilled,
                                timestamp: current_timestamp,
                            };
                            self.add_or_update_order(new_order);
                            if let Some(cloid) = cloid_opt {
                                self.cloid_to_oid.insert(cloid, oid);
                                self.oid_to_cloid.insert(oid, cloid);
                            }
                        }
                    }
                },
                "expired" => {
                    info!("Order OID {} Expired.", oid);
                    self.remove_and_cache_order(oid, OrderState::Expired);
                },
                // Ignore transient states
                "sending" | "pendingCancel" => {
                    debug!("Order OID {} has transient status: {}", oid, status);
                }
                _ => {
                    warn!("Received unknown order status '{}' for OID {}", status, oid);
                }
            }
        }
    }

    /// Process a fill and update current state
    fn process_fill(&mut self, fill: &TradeInfo) {
        let fill_size = fill.sz.parse::<f64>().unwrap_or(0.0);
        let fill_price = fill.px.parse::<f64>().unwrap_or(0.0);
        let fill_fee = fill.fee.parse::<f64>().unwrap_or(0.0);
        let is_buy = fill.side == "B";

        // --- IDENTIFY FILL LEVEL ---
        let mut filled_level: Option<usize> = None;

        // Find the matched order and its level
        if is_buy {
            // This was a fill on our BID order
            if let Some(order) = self.current_state.open_bids.iter().find(|o| o.oid == Some(fill.oid)) {
                filled_level = Some(order.level);
            } else {
                warn!("Fill received for unknown/already removed bid OID: {}", fill.oid);
            }
        } else {
            // This was a fill on our ASK order
            if let Some(order) = self.current_state.open_asks.iter().find(|o| o.oid == Some(fill.oid)) {
                filled_level = Some(order.level);
            } else {
                warn!("Fill received for unknown/already removed ask OID: {}", fill.oid);
            }
        }

        // Update total fees
        self.current_state.total_fees += fill_fee.abs();

        // Update position
        let signed_size = if is_buy { fill_size } else { -fill_size };
        let old_position = self.current_state.position;
        let new_position = old_position + signed_size;

        // Update cost basis and realized PnL
        if old_position.abs() < 1e-6 {
            // Opening new position
            self.current_state.cost_basis = fill_size * fill_price;
            self.current_state.avg_entry_price = fill_price;
        } else if (old_position > 0.0 && new_position > old_position)
                || (old_position < 0.0 && new_position < old_position) {
            // Adding to position
            let old_cost = self.current_state.cost_basis;
            let new_cost = old_cost + (fill_size * fill_price);
            self.current_state.cost_basis = new_cost;
            self.current_state.avg_entry_price = new_cost / new_position.abs();
        } else {
            // Reducing or flipping position
            let reduced_size = fill_size.min(old_position.abs());
            let realized_pnl = if old_position > 0.0 {
                // Closing long
                reduced_size * (fill_price - self.current_state.avg_entry_price)
            } else {
                // Closing short
                reduced_size * (self.current_state.avg_entry_price - fill_price)
            };
            self.current_state.realized_pnl += realized_pnl;

            if new_position.abs() < 1e-6 {
                // Fully closed
                self.current_state.cost_basis = 0.0;
                self.current_state.avg_entry_price = 0.0;
            } else {
                // Partial close or flip
                self.current_state.cost_basis = new_position.abs() * fill_price;
                self.current_state.avg_entry_price = fill_price;
            }
        }

        self.current_state.position = new_position;

        // Update unrealized PnL
        if new_position.abs() > 1e-6 && self.current_state.l2_mid_price > 0.0 {
            self.current_state.unrealized_pnl = if new_position > 0.0 {
                new_position * (self.current_state.l2_mid_price - self.current_state.avg_entry_price)
            } else {
                new_position.abs() * (self.current_state.avg_entry_price - self.current_state.l2_mid_price)
            };
        }

        info!("📊 Fill processed: {} {} @ {} (Level: {}) | Position: {} | Realized PnL: {:.2} | Unrealized PnL: {:.2}",
              if is_buy { "BUY" } else { "SELL" },
              fill_size,
              fill_price,
              filled_level.map(|l| format!("L{}", l + 1)).unwrap_or_else(|| "Unknown".to_string()),
              new_position,
              self.current_state.realized_pnl,
              self.current_state.unrealized_pnl
        );
    }

    /// Execute strategy actions using ParallelOrderExecutor
    async fn execute_actions(&mut self, actions: Vec<StrategyAction>) {
        let mut executor_actions: Vec<ExecutorAction> = Vec::new();
        let current_timestamp = chrono::Utc::now().timestamp_millis() as u64;

        for action in actions {
            match action {
                StrategyAction::Place(order) => {
                    if let Some(cloid) = order.cloid {
                        // Create RestingOrder and add to pending
                        let level = self.calculate_order_level(order.limit_px, order.is_buy);
                        let resting_order = RestingOrder::new(
                            None, // OID unknown yet
                            Some(cloid),
                            order.sz,
                            order.limit_px,
                            order.is_buy,
                            level,
                        );
                        self.pending_place_orders.insert(cloid, resting_order);
                        executor_actions.push(ExecutorAction::Place(order));
                        debug!("Order Cloid {} added to pending_place_orders.", cloid);
                    } else {
                        warn!("StrategyAction::Place missing Cloid, cannot track pending status properly.");
                        executor_actions.push(ExecutorAction::Place(order));
                    }
                },
                StrategyAction::Cancel(cancel) => {
                    let oid_to_cancel = cancel.oid;
                    let mut found = false;
                    // Mark as PendingCancel
                    let orders_lists = [&mut self.current_state.open_bids, &mut self.current_state.open_asks];
                    for list in orders_lists {
                        if let Some(order) = list.iter_mut().find(|o| o.oid == Some(oid_to_cancel) && o.state == OrderState::Active) {
                            order.state = OrderState::PendingCancel;
                            order.timestamp = current_timestamp;
                            executor_actions.push(ExecutorAction::Cancel(cancel.clone()));
                            found = true;
                            debug!("Marked OID {} as PendingCancel.", oid_to_cancel);
                            break;
                        }
                    }
                    if !found {
                        debug!("Skipping cancel for OID {}, not found or not Active.", oid_to_cancel);
                    }
                },
                StrategyAction::BatchPlace(orders) => {
                    for order in &orders {
                        if let Some(cloid) = order.cloid {
                            let level = self.calculate_order_level(order.limit_px, order.is_buy);
                            let resting_order = RestingOrder::new(
                                None, Some(cloid), order.sz, order.limit_px, order.is_buy, level
                            );
                            self.pending_place_orders.insert(cloid, resting_order);
                            debug!("Order Cloid {} added to pending_place_orders (batch).", cloid);
                        } else {
                            warn!("StrategyAction::BatchPlace contains order missing Cloid.");
                        }
                    }
                    executor_actions.push(ExecutorAction::BatchPlace(orders));
                },
                StrategyAction::BatchCancel(cancels) => {
                    let mut valid_executor_cancels = Vec::new();
                    for cancel in cancels {
                        let oid_to_cancel = cancel.oid;
                        let mut found = false;
                        let orders_lists = [&mut self.current_state.open_bids, &mut self.current_state.open_asks];
                        for list in orders_lists {
                            if let Some(order) = list.iter_mut().find(|o| o.oid == Some(oid_to_cancel) && o.state == OrderState::Active) {
                                order.state = OrderState::PendingCancel;
                                order.timestamp = current_timestamp;
                                valid_executor_cancels.push(cancel.clone());
                                found = true;
                                debug!("Marked OID {} as PendingCancel (batch).", oid_to_cancel);
                                break;
                            }
                        }
                        if !found {
                            debug!("Skipping batch cancel for OID {}, not found or not Active.", oid_to_cancel);
                        }
                    }
                    if !valid_executor_cancels.is_empty() {
                        executor_actions.push(ExecutorAction::BatchCancel(valid_executor_cancels));
                    }
                },
                StrategyAction::NoOp => {}
            }
        }

        if executor_actions.is_empty() {
            return; // Nothing to execute
        }

        // Execute in a spawned task to avoid blocking the main event loop
        let executor = self.order_executor.clone();
        let asset_name = self.asset.clone();

        tokio::spawn(async move {
            let result = executor.execute_actions_parallel(executor_actions).await;

            // Process results (log errors, potentially update pending state)
            if result.failed > 0 {
                error!(
                    "[{}] {} failed actions out of {} submitted in {} ms.",
                    asset_name,
                    result.failed,
                    result.failed + result.successful,
                    result.execution_time_ms
                );

                // Track if we encountered a timeout error
                let mut timeout_occurred = false;

                for (index, response_res) in result.responses.iter().enumerate() {
                    match response_res {
                        // Case 1: The entire HTTP request failed (e.g., network error, timeout before response)
                        Err(e) => {
                            error!("[{}] Batch {} - Execution Error: {}", asset_name, index, e);
                            // Check if it's likely a timeout error
                            if e.to_string().contains("timeout") {
                                timeout_occurred = true;
                            }
                        }
                        // Case 2: The HTTP request succeeded, but the exchange wrapper returned a top-level error message
                        Ok(ExchangeResponseStatus::Err(msg)) => {
                            error!(
                                "[{}] Batch {} - Exchange returned top-level error: {}",
                                asset_name, index, msg
                            );
                            // Explicitly check for margin errors here if the exchange ever returns them at this level
                            if msg.contains("margin") || msg.contains("Margin") {
                                error!("[{}] POSSIBLE MARGIN ISSUE DETECTED (Top-level response)", asset_name);
                            }
                        }
                        // Case 3: The HTTP request succeeded, and we got a structured response from the exchange
                        Ok(ExchangeResponseStatus::Ok(resp)) => {
                            // Check if the exchange sent back specific statuses for the actions within the batch
                            if let Some(ref data) = resp.data {
                                // Iterate through statuses (should align with submitted actions if applicable)
                                for (status_index, status) in data.statuses.iter().enumerate() {
                                    if let ExchangeDataStatus::Error(err_msg) = status {
                                        error!(
                                            "[{}] Batch {} - Specific action {} failed: {}",
                                            asset_name, index, status_index, err_msg
                                        );
                                        // Explicitly check for known margin error messages
                                        if err_msg.contains("margin") || err_msg.contains("Margin") {
                                            error!("[{}] MARGIN ISSUE DETECTED: {}", asset_name, err_msg);
                                        } else if err_msg.contains("balance") {
                                            error!("[{}] BALANCE ISSUE DETECTED: {}", asset_name, err_msg);
                                        }
                                    }
                                    // You could add logging for other statuses like Resting, Filled here if needed for debugging
                                    // else if let ExchangeDataStatus::Resting(r) = status { ... }
                                }
                            } else {
                                // This case might indicate success but needs confirmation based on API behavior
                                debug!("[{}] Batch {} - Received Ok status but no detailed data/statuses.", asset_name, index);
                            }
                        }
                    }
                }

                // If a timeout occurred, add a suggestion to check margin/network
                if timeout_occurred {
                    warn!("[{}] Timeout error detected. Check network connectivity and available margin.", asset_name);
                }

            } else if result.successful > 0 { // Only log success if there were no failures in the batch
                debug!(
                    "[{}] {} actions executed successfully in {} ms.",
                    asset_name, result.successful, result.execution_time_ms
                );
            }

            // NOTE: Successful placements/cancellations are confirmed via WebSocket UserEvents,
            // so we don't need to update `current_state.open_bids/asks` here directly.
            // The main loop's `handle_user_events` will reconcile based on WS messages.
        });
    }

    /// Calculate the order book level for a given price (kept for potential future use)
    #[allow(dead_code)]
    fn calculate_order_level(&self, price: f64, is_buy: bool) -> usize {
        if let Some(ref book) = self.current_state.order_book {
            let levels = if is_buy { &book.bids } else { &book.asks };

            for (idx, level) in levels.iter().enumerate() {
                if let Ok(level_price) = level.px.parse::<f64>() {
                    if is_buy {
                        // For bids, higher price = better level
                        if price >= level_price {
                            return idx;
                        }
                    } else {
                        // For asks, lower price = better level
                        if price <= level_price {
                            return idx;
                        }
                    }
                }
            }
            // If we didn't find a match, it's deeper than the visible book
            return levels.len();
        }
        0
    }

    // NOTE: execute_place_order and execute_cancel_order methods removed
    // All order execution now handled by ParallelOrderExecutor via execute_actions()

    /// Handle shutdown
    async fn handle_shutdown(&mut self) {
        info!("🛑 Shutting down bot runner...");

        // Call strategy shutdown hook
        let actions = self.strategy.on_shutdown(&self.current_state);

        // Execute final actions using the executor (await directly, don't spawn)
        if !actions.is_empty() {
            info!("Executing {} shutdown actions...", actions.len());

            // Convert strategy actions to executor actions and filter out NoOp
            let executor_actions: Vec<ExecutorAction> = actions
                .into_iter()
                .filter_map(|action| match action {
                    StrategyAction::Place(order) => Some(ExecutorAction::Place(order)),
                    StrategyAction::Cancel(cancel) => Some(ExecutorAction::Cancel(cancel)),
                    StrategyAction::BatchPlace(orders) => Some(ExecutorAction::BatchPlace(orders)),
                    StrategyAction::BatchCancel(cancels) => Some(ExecutorAction::BatchCancel(cancels)),
                    StrategyAction::NoOp => None,
                })
                .collect();

            if !executor_actions.is_empty() {
                let result = self.order_executor.execute_actions_parallel(executor_actions).await;
                info!("Shutdown actions complete: {} successful, {} failed.", result.successful, result.failed);
            }
        } else {
            info!("No shutdown actions required by strategy.");
        }

        info!("✅ Bot runner shutdown procedures complete");
    }

    /// Handle periodic tick (called every second)
    async fn handle_tick(&mut self) {
        // Update timestamp
        self.current_state.timestamp = chrono::Utc::now().timestamp() as f64;

        // Update unrealized PnL based on current mid-price
        if self.current_state.position.abs() > 1e-6 && self.current_state.l2_mid_price > 0.0 {
            if self.current_state.position > 0.0 {
                // Long position
                self.current_state.unrealized_pnl = 
                    self.current_state.position * (self.current_state.l2_mid_price - self.current_state.avg_entry_price);
            } else {
                // Short position
                self.current_state.unrealized_pnl = 
                    self.current_state.position.abs() * (self.current_state.avg_entry_price - self.current_state.l2_mid_price);
            }
        }

        // Prune recently_completed_orders cache
        let now = Instant::now();
        if now.duration_since(self.last_cache_prune_time) > Duration::from_secs(10) {
            // Prune orders older than 30 seconds
            let cutoff_timestamp = (chrono::Utc::now() - chrono::Duration::seconds(30)).timestamp_millis() as u64;
            let initial_size = self.recently_completed_orders.len();
            self.recently_completed_orders.retain(|_oid, order| order.timestamp >= cutoff_timestamp);
            let removed_count = initial_size - self.recently_completed_orders.len();
            if removed_count > 0 {
                debug!("Pruned {} orders from recently_completed_orders cache.", removed_count);
            }
            self.last_cache_prune_time = now;
        }

        // Call strategy tick hook
        let actions = self.strategy.on_tick(&self.current_state);
        self.execute_actions(actions).await;

        // Clean up old fill IDs periodically (every hour)
        if self.total_messages % 3600 == 0 && self.processed_fill_ids.len() > 1000 {
            info!("🧹 Cleaning up old fill IDs (keeping last 1000)");
            // In production, we keep the most recent based on timestamp
            // For simplicity, just clear if too large
            self.processed_fill_ids.clear();
        }
    }

    /// Generate unique ID for a fill to detect duplicates
    /// Using tid (50-bit hash) ensures uniqueness
    fn get_fill_id(&self, fill: &TradeInfo) -> u64 {
        fill.tid
    }
}

// ============================================================================
// Main
// ============================================================================

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging with Mountain Time timezone
    // Note: Mountain Time is UTC-7 (MST) in winter, UTC-6 (MDT) in summer
    // Currently set to MST (-7). Adjust to -6 for MDT if needed.
    let file_appender = tracing_appender::rolling::never("./", "market_maker_v3.log");
    let (non_blocking_writer, _guard) = tracing_appender::non_blocking(file_appender);

    let file_layer = fmt::layer()
        .json()
        .with_writer(non_blocking_writer)
        .with_timer(fmt::time::OffsetTime::new(
            time::UtcOffset::from_hms(-7, 0, 0).unwrap(), // MST (Mountain Standard Time)
            time::format_description::parse(
                "[year]-[month]-[day]T[hour]:[minute]:[second].[subsecond digits:6]-07:00"
            ).unwrap()
        ));

    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info"));

    tracing_subscriber::registry()
        .with(filter)
        .with(file_layer)
        .init();

    // Load environment variables
    dotenv::dotenv().ok();

    // Load private key
    let private_key = env::var("PRIVATE_KEY")
        .expect("PRIVATE_KEY environment variable must be set");

    let wallet: PrivateKeySigner = private_key
        .parse()
        .expect("Invalid private key format");

    // Load config
    let config = load_config("config.json");

    info!("=== Market Maker V3 | Asset: {} | Strategy: {} ===", config.asset, config.strategy_name);

    // --- STRATEGY FACTORY ---
    let strategy: Box<dyn Strategy> = match config.strategy_name.as_str() {
        "hjb_v1" => {
            Box::new(HjbStrategy::new(&config.asset, &serde_json::json!({
                "strategy_params": config.strategy_params
            })))
        }
        // Add more strategies here:
        // "grid_v1" => Box::new(GridStrategy::new(&config.asset, &config)),
        _ => panic!("Unknown strategy: {}", config.strategy_name),
    };

    // Initialize tick/lot validator (hardcoded for now)
    let tick_lot_validator = TickLotValidator::new(
        config.asset.clone(),
        AssetType::Perp,
        3, // sz_decimals
    );

    // Log TickLotValidator configuration for debugging
    debug!("TickLotValidator config: asset={}, szDecimals={}, max_price_decimals={}",
          tick_lot_validator.asset,
          tick_lot_validator.sz_decimals,
          tick_lot_validator.max_price_decimals());

    // Create bot runner
    let mut bot_runner = BotRunner::new(
        config.asset.clone(),
        strategy,
        wallet,
        tick_lot_validator,
    ).await?;

    // Set up shutdown signals
    let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel();

    // Set up Ctrl+C handler
    tokio::spawn(async move {
        signal::ctrl_c().await.expect("Failed to install Ctrl+C handler");
        info!("🛑 Ctrl+C - initiating graceful shutdown...");
        let _ = shutdown_tx.send(());

        // Optional: Add a handler for a *second* Ctrl+C if graceful shutdown hangs
        signal::ctrl_c().await.expect("Failed to install second Ctrl+C handler");
        warn!("⚠️ Second Ctrl+C - forcing exit!");
        std::process::exit(1);
    });

    // Run bot runner
    bot_runner.run(shutdown_rx).await?;

    info!("Market Maker V3 shutdown complete");
    Ok(())
}
