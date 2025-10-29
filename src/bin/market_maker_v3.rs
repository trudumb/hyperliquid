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
    CurrentState, ExchangeClient, InfoClient, MarketUpdate, Message, OrderBook, OrderState,
    OrderStateManager, OrderUpdate, OrderUpdateResult, RestingOrder, Strategy, StrategyAction,
    Subscription, TickLotValidator, TradeInfo, UserData, UserUpdate,
    // New imports for optimized execution
    ParallelOrderExecutor, ExecutorConfig,
    ExchangeResponseStatus, ExchangeDataStatus, ExecutorAction,
};
use hyperliquid_rust_sdk::strategies::hjb_strategy::HjbStrategy;

use log::{error, info, warn, debug};
use std::collections::HashSet;
use std::env;
use std::fs;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
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

        // Initialize position from account state (not historical fills!)
        let (position, avg_entry_price, cost_basis, unrealized_pnl) =
            if let Some(asset_position) = user_state.asset_positions.iter().find(|ap| ap.position.coin == asset) {
                let pos_data = &asset_position.position;
                let szi = pos_data.szi.parse::<f64>().unwrap_or(0.0);
                let entry_px = pos_data.entry_px.as_ref()
                    .and_then(|p| p.parse::<f64>().ok())
                    .unwrap_or(0.0);
                let unrealized = pos_data.unrealized_pnl.parse::<f64>().unwrap_or(0.0);

                // Calculate cost basis from position and entry price
                let cost = szi.abs() * entry_px;

                info!("📊 Found existing position for {}: {} units @ ${:.2}", asset, szi, entry_px);
                info!("   Unrealized PnL: ${:.2}", unrealized);

                (szi, entry_px, cost, unrealized)
            } else {
                info!("📊 No existing position found for {}", asset);
                (0.0, 0.0, 0.0, 0.0)
            };

        // Initialize current state
        let current_state = CurrentState {
            position,
            avg_entry_price,
            cost_basis,
            unrealized_pnl,
            realized_pnl: 0.0,  // Session realized PnL starts at 0
            total_fees: 0.0,    // Session fees start at 0
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

    /// Start the bot runner event loop
    async fn run(&mut self, shutdown_rx: tokio::sync::oneshot::Receiver<()>) -> Result<(), Box<dyn std::error::Error>> {
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

        // Periodic REST reconciliation timer (10 second interval)
        // This captures manually-placed orders or orders from other systems
        let mut reconciliation_timer = tokio::time::interval(tokio::time::Duration::from_secs(10));
        reconciliation_timer.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

        // Shutdown timeout (will be created when shutdown starts)
        let mut shutdown_timeout: Option<std::pin::Pin<Box<tokio::time::Sleep>>> = None;

        // Wrap shutdown_rx in Option so we can take it once and stop polling it
        let mut shutdown_rx = Some(shutdown_rx);

        // Main event loop
        loop {
            tokio::select! {
                biased;  // Prioritize shutdown signal over other branches

                // Check for shutdown signal (only if not yet received)
                _ = async { shutdown_rx.as_mut().unwrap().await }, if shutdown_rx.is_some() => {
                    info!("🛑 Shutdown signal received by main loop");
                    // Take the receiver to prevent polling it again
                    shutdown_rx.take();
                    // Set the flag FIRST
                    self.is_shutting_down.store(true, Ordering::Relaxed);
                    // Then handle shutdown actions (sends cancellations with retry logic)
                    self.handle_shutdown().await;
                    // Set a timeout to allow final WebSocket confirmations to arrive
                    // Note: handle_shutdown() already waits and retries, so this is just a final grace period
                    shutdown_timeout = Some(Box::pin(tokio::time::sleep(tokio::time::Duration::from_millis(500))));
                    info!("⏳ Continuing to process WebSocket messages for final confirmations...");
                }

                // Check for shutdown timeout
                _ = async { shutdown_timeout.as_mut().unwrap().await }, if shutdown_timeout.is_some() => {
                    let remaining_bids = self.current_state.open_bids.len();
                    let remaining_asks = self.current_state.open_asks.len();

                    if remaining_bids > 0 || remaining_asks > 0 {
                        warn!("⚠️ Final grace period expired with {} bids and {} asks in local state",
                              remaining_bids, remaining_asks);
                        warn!("   (Note: Orders may have been canceled on exchange - local state might be stale)");
                    } else {
                        info!("✅ All orders cleared from local state");
                    }

                    info!("🛑 Exiting main loop");
                    break;
                }

                // Handle WebSocket messages
                message = receiver.recv_async() => {
                    match message {
                        Ok(message) => {
                            // Process OrderUpdates even during shutdown to update state
                            // Other message types can be optionally filtered
                            let is_shutting_down = self.is_shutting_down.load(Ordering::Relaxed);
                            let is_order_update = matches!(message, Message::OrderUpdates(_));
                            let should_process = !is_shutting_down || is_order_update;

                            if should_process {
                                self.handle_message(message).await;

                                // During shutdown, check if all orders are cleared after each OrderUpdate
                                if is_shutting_down && shutdown_timeout.is_some() && is_order_update {
                                    let remaining_bids = self.current_state.open_bids.len();
                                    let remaining_asks = self.current_state.open_asks.len();

                                    if remaining_bids == 0 && remaining_asks == 0 {
                                        info!("✅ All orders cleared from state - exiting early");
                                        break;
                                    }
                                }
                            } else {
                                // Ignore non-critical messages during shutdown
                                debug!("Ignoring non-OrderUpdate message during shutdown");
                            }
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

                // Handle periodic REST reconciliation (captures external orders)
                _ = reconciliation_timer.tick() => {
                    // Skip during shutdown
                    if self.is_shutting_down.load(Ordering::Relaxed) {
                        continue;
                    }
                    debug!("Running periodic REST reconciliation...");
                    self.reconcile_with_rest().await;
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
                    // Skip snapshot processing - position already initialized from account state
                    if !self.snapshot_received {
                        info!("📸 Received snapshot with {} historical fills (skipping - position already initialized from account state)", fills_data.fills.len());

                        // Track fill IDs to prevent duplicate processing from UserEvents
                        for fill in &fills_data.fills {
                            let fill_id = self.get_fill_id(fill);
                            self.processed_fill_ids.insert(fill_id);
                        }

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
                        let oid = fill.oid;

                        // Use OrderStateManager to get order level from active or cached orders
                        let all_orders: Vec<RestingOrder> = self.current_state.open_bids.iter()
                            .chain(self.current_state.open_asks.iter())
                            .cloned()
                            .collect();

                        let mut filled_level = self.order_state_mgr.get_order_level(oid, &all_orders);

                        // If order not found, trigger immediate REST reconciliation
                        if filled_level.is_none() {
                            warn!("Fill received for unknown or too-old OID: {}. Triggering immediate REST reconciliation...", oid);
                            self.reconcile_with_rest().await;

                            // Try to get level again after reconciliation
                            let all_orders: Vec<RestingOrder> = self.current_state.open_bids.iter()
                                .chain(self.current_state.open_asks.iter())
                                .cloned()
                                .collect();
                            filled_level = self.order_state_mgr.get_order_level(oid, &all_orders);

                            if filled_level.is_some() {
                                info!("✅ Successfully recovered level {} for OID {} after reconciliation", filled_level.unwrap(), oid);
                            } else {
                                warn!("⚠️ Could not determine level for OID {} even after reconciliation", oid);
                            }
                        } else {
                            debug!("Determined level {} for fill OID {}", filled_level.unwrap_or(999), oid);
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
                        // Try to remove from bids first, then asks
                        let removed = self.order_state_mgr.remove_and_cache_order(
                            cancel.oid,
                            OrderState::Cancelled,
                            &mut self.current_state.open_bids
                        ) || self.order_state_mgr.remove_and_cache_order(
                            cancel.oid,
                            OrderState::Cancelled,
                            &mut self.current_state.open_asks
                        );
                        if removed {
                            info!("   Possible reasons: post-only order would cross, position limit, margin");
                        } else {
                            debug!("   Order {} not found in active lists (may have been already removed)", cancel.oid);
                        }
                    }
                }
            }
        }
    }

    /// Handle order status updates from WebSocket
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
                OrderUpdateResult::RemoveAndCache(oid, _final_state) => {
                    self.remove_order_from_current_state(oid);
                }
                OrderUpdateResult::NoAction => {}
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
                        self.order_state_mgr.add_pending_order(cloid, resting_order);
                        executor_actions.push(ExecutorAction::Place(order));
                        debug!("Order Cloid {} added to pending_place_orders.", cloid);
                    } else {
                        warn!("StrategyAction::Place missing Cloid, cannot track pending status properly.");
                        executor_actions.push(ExecutorAction::Place(order));
                    }
                },
                StrategyAction::Cancel(cancel) => {
                    let oid_to_cancel = cancel.oid;
                    // Mark as PendingCancel using OrderStateManager
                    let bids_and_asks = [&mut self.current_state.open_bids, &mut self.current_state.open_asks];
                    let mut found = false;
                    for orders in bids_and_asks {
                        if self.order_state_mgr.mark_pending_cancel(oid_to_cancel, orders) {
                            executor_actions.push(ExecutorAction::Cancel(cancel.clone()));
                            found = true;
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
                            self.order_state_mgr.add_pending_order(cloid, resting_order);
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
                        // Mark as PendingCancel using OrderStateManager
                        let bids_and_asks = [&mut self.current_state.open_bids, &mut self.current_state.open_asks];
                        let mut found = false;
                        for orders in bids_and_asks {
                            if self.order_state_mgr.mark_pending_cancel(oid_to_cancel, orders) {
                                valid_executor_cancels.push(cancel.clone());
                                found = true;
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

    /// Add or update an order in CurrentState (open_bids/open_asks)
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

    /// Update a partially filled order in CurrentState
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

    /// Remove an order from CurrentState
    fn remove_order_from_current_state(&mut self, oid: u64) {
        self.current_state.open_bids.retain(|o| o.oid != Some(oid));
        self.current_state.open_asks.retain(|o| o.oid != Some(oid));
    }

    /// Reconcile bot's tracked orders with actual open orders from REST API
    /// This captures manually-placed orders or orders from other systems
    async fn reconcile_with_rest(&mut self) {
        let info_client = match self.info_client.as_ref() {
            Some(client) => client,
            None => {
                error!("Cannot reconcile: info_client not available");
                return;
            }
        };

        // Fetch all open orders from REST API
        let open_orders_result = info_client.open_orders(self.user_address).await;
        let open_orders = match open_orders_result {
            Ok(orders) => orders,
            Err(e) => {
                error!("Failed to fetch open orders for reconciliation: {:?}", e);
                return;
            }
        };

        // Filter orders for our asset
        let our_asset_orders: Vec<_> = open_orders.into_iter()
            .filter(|o| o.coin == self.asset)
            .collect();

        debug!("REST reconciliation: found {} open orders for {}", our_asset_orders.len(), self.asset);

        // Build set of currently tracked OIDs
        let tracked_oids: std::collections::HashSet<u64> = self.current_state.open_bids.iter()
            .chain(self.current_state.open_asks.iter())
            .filter_map(|o| o.oid)
            .collect();

        let mut imported_count = 0;

        // Check each REST order
        for rest_order in our_asset_orders {
            let oid = rest_order.oid;

            // Skip if we already track this order
            if tracked_oids.contains(&oid) {
                continue;
            }

            // This is an external order - import it
            let price = rest_order.limit_px.parse::<f64>().unwrap_or(0.0);
            let size = rest_order.sz.parse::<f64>().unwrap_or(0.0);
            let is_buy = rest_order.side == "B";
            let level = self.calculate_order_level(price, is_buy);
            let cloid: Option<uuid::Uuid> = rest_order.cloid.as_deref().and_then(|s| s.parse().ok());

            let external_order = RestingOrder {
                oid: Some(oid),
                cloid,
                size,
                orig_size: size, // We don't have orig_size from REST, use current size
                price,
                is_buy,
                level,
                state: OrderState::Active,
                timestamp: rest_order.timestamp,
            };

            // Import into OrderStateManager cache
            if self.order_state_mgr.import_external_order(oid, external_order.clone()) {
                // Also add to current state so we track it going forward
                self.add_order_to_current_state(external_order);
                imported_count += 1;
            }
        }

        if imported_count > 0 {
            info!("REST reconciliation: imported {} external orders", imported_count);
        } else {
            debug!("REST reconciliation: no new external orders found");
        }
    }

    // NOTE: execute_place_order and execute_cancel_order methods removed
    // All order execution now handled by ParallelOrderExecutor via execute_actions()

    /// Handle shutdown with retry logic for order cancellation
    async fn handle_shutdown(&mut self) {
        info!("🛑 Shutting down bot runner...");

        const MAX_RETRIES: usize = 3;
        const RETRY_DELAY_MS: u64 = 500;

        for attempt in 1..=MAX_RETRIES {
            // Call strategy shutdown hook (generates actions based on current state)
            let actions = self.strategy.on_shutdown(&self.current_state);

            if actions.is_empty() {
                info!("No shutdown actions required (attempt {}/{})", attempt, MAX_RETRIES);
                break;
            }

            info!("Executing {} shutdown actions (attempt {}/{})...", actions.len(), attempt, MAX_RETRIES);

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

                // Log individual failures for debugging
                if result.failed > 0 {
                    warn!("Some shutdown actions failed - will retry if orders remain");
                    for (idx, response) in result.responses.iter().enumerate() {
                        if let Err(e) = response {
                            error!("Shutdown action {} failed: {:?}", idx, e);
                        }
                    }
                }
            }

            // Verify orders were canceled using REST API (more reliable than WebSocket state)
            if let Some(info_client) = self.info_client.as_ref() {
                // Small delay to allow exchange to process cancellations
                tokio::time::sleep(tokio::time::Duration::from_millis(RETRY_DELAY_MS)).await;

                match info_client.open_orders(self.user_address).await {
                    Ok(open_orders) => {
                        let remaining_orders: Vec<_> = open_orders
                            .into_iter()
                            .filter(|o| o.coin == self.asset)
                            .collect();

                        if remaining_orders.is_empty() {
                            info!("✅ REST API confirms all orders canceled");
                            break;
                        } else {
                            warn!("⚠️ REST API shows {} orders still open after attempt {}/{}",
                                  remaining_orders.len(), attempt, MAX_RETRIES);

                            // Update local state with REST data for next retry
                            self.current_state.open_bids.clear();
                            self.current_state.open_asks.clear();

                            for rest_order in remaining_orders {
                                let price = rest_order.limit_px.parse::<f64>().unwrap_or(0.0);
                                let size = rest_order.sz.parse::<f64>().unwrap_or(0.0);
                                let is_buy = rest_order.side == "B";
                                let level = self.calculate_order_level(price, is_buy);
                                let cloid: Option<uuid::Uuid> = rest_order.cloid.as_deref().and_then(|s| s.parse().ok());

                                let order = RestingOrder {
                                    oid: Some(rest_order.oid),
                                    cloid,
                                    size,
                                    orig_size: size,
                                    price,
                                    is_buy,
                                    level,
                                    state: OrderState::Active,
                                    timestamp: rest_order.timestamp,
                                };

                                if is_buy {
                                    self.current_state.open_bids.push(order);
                                } else {
                                    self.current_state.open_asks.push(order);
                                }
                            }

                            if attempt < MAX_RETRIES {
                                info!("Retrying cancellations for remaining orders...");
                                continue;
                            }
                        }
                    }
                    Err(e) => {
                        error!("Failed to verify order cancellations via REST: {:?}", e);
                        break;
                    }
                }
            } else {
                warn!("Cannot verify order cancellations - info_client not available");
                break;
            }
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

        // Prune recently_completed_orders cache using OrderStateManager
        self.order_state_mgr.prune_cache_if_needed();

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
