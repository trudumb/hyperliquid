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
*/

use alloy::signers::local::PrivateKeySigner;
use hyperliquid_rust_sdk::{
    AssetType, BaseUrl, ClientCancelRequest, ClientOrderRequest,
    CurrentState, ExchangeClient, InfoClient, MarketUpdate, Message,
    OrderBook, RestingOrder, Strategy, StrategyAction, Subscription, TickLotValidator,
    TradeInfo, UserData, UserUpdate,
};
use hyperliquid_rust_sdk::strategies::hjb_strategy::HjbStrategy;
use hyperliquid_rust_sdk::tui::state::DashboardState;

use log::{error, info, warn};
use std::collections::HashMap;
use std::env;
use std::fs;
use std::sync::Arc;
use tokio::signal;
use tokio::sync::{mpsc, watch};
use tracing_subscriber::{fmt, layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

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

    /// Info client for market data queries
    info_client: InfoClient,

    /// User wallet address
    user_address: alloy::primitives::Address,

    /// Tick/lot validator for order validation
    tick_lot_validator: TickLotValidator,

    /// Current bot state (maintained by bot runner)
    current_state: CurrentState,

    /// TUI dashboard state sender
    tui_tx: watch::Sender<DashboardState>,

    /// Order tracking: cloid -> oid mapping
    cloid_to_oid: HashMap<uuid::Uuid, u64>,

    /// Pending order placements (awaiting confirmation)
    pending_orders: HashMap<uuid::Uuid, ClientOrderRequest>,
}

impl BotRunner {
    /// Create a new bot runner
    async fn new(
        asset: String,
        strategy: Box<dyn Strategy>,
        wallet: PrivateKeySigner,
        tick_lot_validator: TickLotValidator,
    ) -> Result<(Self, watch::Receiver<DashboardState>), Box<dyn std::error::Error>> {
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
            max_position_size: 50.0, // Will be updated from strategy config
            timestamp: chrono::Utc::now().timestamp() as f64,
            session_start_time: chrono::Utc::now().timestamp() as f64,
        };

        // Create TUI state channel
        let (tui_tx, tui_rx) = watch::channel(DashboardState::default());

        info!("✅ Bot Runner initialized for {} with strategy: {}", asset, strategy.name());
        info!("   Account Equity: ${:.2}", account_equity);

        Ok((
            Self {
                asset,
                strategy,
                exchange_client,
                info_client,
                user_address,
                tick_lot_validator,
                current_state,
                tui_tx,
                cloid_to_oid: HashMap::new(),
                pending_orders: HashMap::new(),
            },
            tui_rx,
        ))
    }

    /// Start the bot runner event loop
    async fn run(&mut self, mut shutdown_rx: tokio::sync::oneshot::Receiver<()>) -> Result<(), Box<dyn std::error::Error>> {
        info!("🚀 Starting bot runner event loop...");

        // Set up websocket subscriptions
        let (sender, mut receiver) = mpsc::unbounded_channel();

        // Subscribe to user events (fills)
        self.info_client
            .subscribe(
                Subscription::UserEvents {
                    user: self.user_address,
                },
                sender.clone(),
            )
            .await?;

        // Subscribe to AllMids (mid-price updates)
        self.info_client
            .subscribe(Subscription::AllMids, sender.clone())
            .await?;

        // Subscribe to L2Book (order book updates)
        self.info_client
            .subscribe(
                Subscription::L2Book {
                    coin: self.asset.clone(),
                },
                sender.clone(),
            )
            .await?;

        // Subscribe to Trades (market trades for flow analysis)
        self.info_client
            .subscribe(
                Subscription::Trades {
                    coin: self.asset.clone(),
                },
                sender.clone(),
            )
            .await?;

        info!("✅ WebSocket subscriptions established");

        // Main event loop
        loop {
            tokio::select! {
                // Check for shutdown signal
                _ = &mut shutdown_rx => {
                    info!("🛑 Shutdown signal received");
                    self.handle_shutdown().await;
                    break;
                }

                // Handle WebSocket messages
                message = receiver.recv() => {
                    if let Some(message) = message {
                        self.handle_message(message).await;
                    } else {
                        error!("WebSocket channel closed");
                        break;
                    }
                }
            }
        }

        info!("✅ Bot runner event loop terminated");
        Ok(())
    }

    /// Handle incoming WebSocket message
    async fn handle_message(&mut self, message: Message) {
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
            _ => {
                // Ignore other message types
            }
        }

        // Update TUI after every message
        self.update_tui();
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

    /// Handle user events (fills, order updates)
    async fn handle_user_events(&mut self, user_events_msg: Message) {
        if let Message::User(user_events) = user_events_msg {
            if let UserData::Fills(fills) = user_events.data {
                // Update current state with fills
                for fill in &fills {
                    self.process_fill(fill);
                }

                // Create user update and pass to strategy
                let user_update = UserUpdate::from_fills(fills);
                let actions = self.strategy.on_user_update(&self.current_state, &user_update);
                self.execute_actions(actions).await;
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
                                        let resting_order = RestingOrder {
                                            oid: resting.oid,
                                            size: validated_order.sz,
                                            price: validated_order.limit_px,
                                            level: 0, // TODO: Track actual level
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

    /// Update TUI dashboard
    fn update_tui(&self) {
        let dashboard_state = DashboardState {
            cur_position: self.current_state.position,
            avg_entry_price: self.current_state.avg_entry_price,
            unrealized_pnl: self.current_state.unrealized_pnl,
            realized_pnl: self.current_state.realized_pnl,
            total_fees: self.current_state.total_fees,
            total_session_pnl: self.current_state.realized_pnl + self.current_state.unrealized_pnl - self.current_state.total_fees,
            account_equity: self.current_state.account_equity,
            session_start_equity: self.current_state.account_equity, // TODO: Track actual start equity
            sharpe_ratio: 0.0, // TODO: Calculate Sharpe ratio
            l2_mid_price: self.current_state.l2_mid_price,
            all_mids_price: self.current_state.l2_mid_price, // Using L2 mid as AllMids
            market_spread_bps: self.current_state.market_spread_bps,
            lob_imbalance: self.current_state.lob_imbalance,
            volatility_ema_bps: 0.0, // TODO: Get from strategy
            adverse_selection_estimate: 0.0, // TODO: Get from strategy
            trade_flow_ema: 0.0, // TODO: Get from strategy
            pf_ess: 0.0, // TODO: Get from strategy
            pf_max_particles: 1000, // TODO: Get from strategy
            pf_vol_5th: 0.0, // TODO: Get from strategy
            pf_vol_95th: 0.0, // TODO: Get from strategy
            pf_volatility_bps: 0.0, // TODO: Get from strategy
            online_model_mae: 0.0, // TODO: Get from strategy
            online_model_updates: 0, // TODO: Get from strategy
            online_model_lr: 0.001, // TODO: Get from strategy
            online_model_enabled: true, // TODO: Get from strategy
            adam_gradient_samples: 0, // TODO: Get from strategy
            adam_avg_loss: 0.0, // TODO: Get from strategy
            adam_last_update_secs: 0.0, // TODO: Get from strategy
            bid_levels: self.current_state.open_bids.iter().map(|o| {
                hyperliquid_rust_sdk::tui::state::OrderLevel {
                    side: "BID".to_string(),
                    level: o.level,
                    price: o.price,
                    size: o.size,
                    oid: o.oid,
                }
            }).collect(),
            ask_levels: self.current_state.open_asks.iter().map(|o| {
                hyperliquid_rust_sdk::tui::state::OrderLevel {
                    side: "ASK".to_string(),
                    level: o.level,
                    price: o.price,
                    size: o.size,
                    oid: o.oid,
                }
            }).collect(),
            recent_fills: Default::default(),
            uptime_secs: (chrono::Utc::now().timestamp() as f64) - self.current_state.session_start_time,
            total_messages: 0, // TODO: Track total messages
        };

        let _ = self.tui_tx.send(dashboard_state);
    }
}

// ============================================================================
// Main
// ============================================================================

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    let file_appender = tracing_appender::rolling::never("./", "market_maker_v3.log");
    let (non_blocking_writer, _guard) = tracing_appender::non_blocking(file_appender);

    let file_layer = fmt::layer()
        .json()
        .with_writer(non_blocking_writer);

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

    info!("=== Market Maker V3 (Generic Bot Runner) ===");
    info!("Asset: {}", config.asset);
    info!("Strategy: {}", config.strategy_name);
    info!("============================================");

    // --- STRATEGY FACTORY ---
    let strategy: Box<dyn Strategy> = match config.strategy_name.as_str() {
        "hjb_v1" => {
            info!("Loading strategy: HJB Strategy v2");
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

    // Create bot runner
    let (mut bot_runner, tui_rx) = BotRunner::new(
        config.asset.clone(),
        strategy,
        wallet,
        tick_lot_validator,
    ).await?;

    // Set up shutdown signals
    let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel();
    let (tui_shutdown_tx, tui_shutdown_rx) = tokio::sync::oneshot::channel();

    // Spawn TUI task
    let tui_handle = tokio::spawn(async move {
        if let Err(e) = hyperliquid_rust_sdk::tui::run_tui(tui_rx, tui_shutdown_rx).await {
            eprintln!("TUI error: {}", e);
        }
    });

    // Set up Ctrl+C handler
    tokio::spawn(async move {
        signal::ctrl_c().await.expect("Failed to install Ctrl+C handler");
        info!("Received Ctrl+C signal");
        let _ = shutdown_tx.send(());
    });

    // Run bot runner
    bot_runner.run(shutdown_rx).await?;

    // Shutdown TUI
    let _ = tui_shutdown_tx.send(());
    let _ = tui_handle.await;

    info!("Market Maker V3 shutdown complete");
    Ok(())
}
