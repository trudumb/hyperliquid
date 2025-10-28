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
    AssetType, BaseUrl, ClientCancelRequest, ClientOrderRequest,
    CurrentState, ExchangeClient, InfoClient, MarketUpdate, Message,
    OrderBook, RestingOrder, Strategy, StrategyAction, Subscription, TickLotValidator,
    TradeInfo, UserData, UserUpdate,
};
use hyperliquid_rust_sdk::strategies::hjb_strategy::HjbStrategy;

use log::{error, info, warn, debug};
use std::collections::{HashMap, HashSet};
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

    /// Exchange client for order management
    exchange_client: Arc<ExchangeClient>,

    /// Info client for market data queries (Option to allow explicit drop on shutdown)
    info_client: Option<InfoClient>,

    /// User wallet address
    user_address: alloy::primitives::Address,

    /// Tick/lot validator for order validation
    tick_lot_validator: TickLotValidator,

    /// Current bot state (maintained by bot runner)
    current_state: CurrentState,

    /// Order tracking: cloid -> oid mapping
    cloid_to_oid: HashMap<uuid::Uuid, u64>,

    /// Pending order placements (awaiting confirmation)
    pending_orders: HashMap<uuid::Uuid, ClientOrderRequest>,

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
            exchange_client,
            info_client: Some(info_client),
            user_address,
            tick_lot_validator,
            current_state,
            cloid_to_oid: HashMap::new(),
            pending_orders: HashMap::new(),
            total_messages: 0,
            is_shutting_down: Arc::new(AtomicBool::new(false)),
            snapshot_received: false,
            processed_fill_ids: HashSet::new(),
        })
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

                    let mut new_fills = Vec::new();

                    for fill in &fills {
                        let fill_id = self.get_fill_id(fill);

                        // Check if we already processed this fill (from snapshot)
                        if self.processed_fill_ids.contains(&fill_id) {
                            debug!("Skipping duplicate fill: tid={}", fill.tid);
                            continue;
                        }

                        // Process new fill
                        self.process_fill(fill);
                        self.processed_fill_ids.insert(fill_id);
                        new_fills.push(fill.clone());
                    }

                    // Only notify strategy if there were NEW fills
                    if !new_fills.is_empty() {
                        let user_update = UserUpdate::from_fills(new_fills);
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

                        // Remove from our tracking
                        self.current_state.open_bids.retain(|order| order.oid != cancel.oid);
                        self.current_state.open_asks.retain(|order| order.oid != cancel.oid);

                        // Remove from pending orders - need to find by oid
                        let cloid_to_remove: Vec<uuid::Uuid> = self.cloid_to_oid
                            .iter()
                            .filter(|(_, &oid)| oid == cancel.oid)
                            .map(|(&cloid, _)| cloid)
                            .collect();

                        for cloid in cloid_to_remove {
                            self.pending_orders.remove(&cloid);
                            self.cloid_to_oid.remove(&cloid);
                        }

                        // Log why this might have happened
                        info!("   Possible reasons: post-only order would cross, position limit, margin");
                    }
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

        info!("📊 Fill processed: {} {} @ {} | Position: {} | Realized PnL: {:.2} | Unrealized PnL: {:.2}",
              if is_buy { "BUY" } else { "SELL" },
              fill_size,
              fill_price,
              new_position,
              self.current_state.realized_pnl,
              self.current_state.unrealized_pnl
        );
    }

    /// Execute strategy actions
    async fn execute_actions(&mut self, actions: Vec<StrategyAction>) {
        for action in actions {
            match action {
                StrategyAction::Place(order) => {
                    self.execute_place_order(order).await;
                }
                StrategyAction::Cancel(cancel) => {
                    self.execute_cancel_order(cancel).await;
                }
                StrategyAction::BatchPlace(orders) => {
                    for order in orders {
                        self.execute_place_order(order).await;
                    }
                }
                StrategyAction::BatchCancel(cancels) => {
                    for cancel in cancels {
                        self.execute_cancel_order(cancel).await;
                    }
                }
                StrategyAction::NoOp => {
                    // Do nothing
                }
            }
        }
    }

    /// Calculate the order book level for a given price
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

    /// Execute a single order placement
    async fn execute_place_order(&mut self, order: ClientOrderRequest) {
        // Validate order
        let validated_px = self.tick_lot_validator.round_price(order.limit_px, order.is_buy);
        let validated_sz = self.tick_lot_validator.round_size(order.sz, false);

        if validated_sz < 0.001 {
            warn!("Order size too small after rounding: {}", order.sz);
            return;
        }

        let mut validated_order = order.clone();
        validated_order.limit_px = validated_px;
        validated_order.sz = validated_sz;

        // Track pending order
        if let Some(cloid) = validated_order.cloid {
            self.pending_orders.insert(cloid, validated_order.clone());
        }

        // Place order via exchange client
        match self.exchange_client.order(validated_order.clone(), None).await {
            Ok(response) => {
                use hyperliquid_rust_sdk::{ExchangeResponseStatus, ExchangeDataStatus};
                match response {
                    ExchangeResponseStatus::Ok(exchange_response) => {
                        if let Some(data) = exchange_response.data {
                            for status in &data.statuses {
                                match status {
                                    ExchangeDataStatus::Resting(resting) => {
                                        // Order placed successfully
                                        info!("✅ Order placed: {} {} @ {} (oid: {})",
                                              if validated_order.is_buy { "BUY" } else { "SELL" },
                                              validated_order.sz,
                                              validated_order.limit_px,
                                              resting.oid
                                        );

                                        // Track cloid -> oid mapping
                                        if let Some(cloid) = validated_order.cloid {
                                            self.cloid_to_oid.insert(cloid, resting.oid);
                                        }

                                        // Update open orders in current state
                                        let level = self.calculate_order_level(validated_order.limit_px, validated_order.is_buy);
                                        let resting_order = RestingOrder {
                                            oid: resting.oid,
                                            size: validated_order.sz,
                                            price: validated_order.limit_px,
                                            level,
                                            pending_cancel: false,
                                        };

                                        if validated_order.is_buy {
                                            self.current_state.open_bids.push(resting_order);
                                        } else {
                                            self.current_state.open_asks.push(resting_order);
                                        }
                                    }
                                    ExchangeDataStatus::Filled(filled) => {
                                        // Order was immediately filled
                                        info!("✅ Order immediately filled: {} {} @ avg {} (oid: {})",
                                              if validated_order.is_buy { "BUY" } else { "SELL" },
                                              filled.total_sz,
                                              filled.avg_px,
                                              filled.oid
                                        );
                                    }
                                    ExchangeDataStatus::Error(err_msg) => {
                                        error!("❌ Order placement failed: {}", err_msg);
                                    }
                                    _ => {
                                        // Success, WaitingForFill, WaitingForTrigger
                                    }
                                }
                            }
                        }
                    }
                    ExchangeResponseStatus::Err(err_msg) => {
                        error!("❌ Order placement failed: {}", err_msg);
                    }
                }
            }
            Err(e) => {
                error!("❌ Order placement error: {}", e);
            }
        }
    }

    /// Execute a single order cancellation
    async fn execute_cancel_order(&mut self, cancel: ClientCancelRequest) {
        match self.exchange_client.cancel(cancel.clone(), None).await {
            Ok(_) => {
                info!("✅ Order cancelled: oid {}", cancel.oid);

                // Remove from open orders
                self.current_state.open_bids.retain(|o| o.oid != cancel.oid);
                self.current_state.open_asks.retain(|o| o.oid != cancel.oid);
            }
            Err(e) => {
                error!("❌ Order cancellation error: {}", e);
            }
        }
    }

    /// Handle shutdown
    async fn handle_shutdown(&mut self) {
        info!("🛑 Shutting down bot runner...");

        // Call strategy shutdown hook
        let actions = self.strategy.on_shutdown(&self.current_state);
        self.execute_actions(actions).await;

        info!("✅ Bot runner shutdown complete");
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
