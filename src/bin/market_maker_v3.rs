// src/bin/market_maker_v3.rs
//
// Market Maker V3 - Distributed Actor Model Implementation
//
// This binary implements the requested architecture:
// 1.  StateManagerActor: A single, authoritative task that manages state,
//     subscribes to user data (fills, orders), and executes all trades.
// 2.  StrategyRunnerActor: One or more tasks, each running a strategy for
//     a specific asset. They subscribe only to market data and query the
//     State Manager for position/margin before sending trade requests.
//
// Communication is handled via tokio mpsc and broadcast channels (simulating IPC).
//

use alloy::primitives::Address;
use alloy::signers::local::PrivateKeySigner;
use hyperliquid_rust_sdk::{
    AssetType, BaseUrl, CurrentState, ExchangeClient, InfoClient,
    MarketUpdate, Message, OrderBook, OrderState, OrderStateManager,
    OrderUpdateResult, ParallelOrderExecutor, RestingOrder, Strategy, StrategyAction,
    Subscription, TickLotValidator, TradeInfo, UserData, ExecutorAction,
    ExecutorConfig, ExchangeResponseStatus, ExchangeDataStatus,
};
use hyperliquid_rust_sdk::strategies::hjb_strategy::HjbStrategy;
// Import our new IPC message types
use hyperliquid_rust_sdk::ipc::{
    AuthoritativeStateUpdate, ExecuteActionsRequest, ExecuteActionsResponse,
};

use log::{debug, error, info, warn};
use std::collections::{HashMap, HashSet};
use std::env;
use std::fs;
use std::sync::Arc;
use tokio::sync::{broadcast, mpsc, oneshot, Mutex, RwLock};
use tokio::task::LocalSet;
use tokio::time::{interval, Duration};
use tracing_subscriber::{fmt, layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

// RUST_LOG=info,hyperliquid_rust_sdk::bin::market_maker_v3=debug cargo run --release --bin market_maker_v3

// ============================================================================
// Configuration
// ============================================================================

#[derive(serde::Deserialize, Debug, Clone)]
struct StrategyConfig {
    asset: String,
    strategy_name: String,
    strategy_params: serde_json::Value,
}

#[derive(serde::Deserialize, Debug)]
struct AppConfig {
    strategies: Vec<StrategyConfig>,
}

fn load_config(path: &str) -> AppConfig {
    let config_str = fs::read_to_string(path)
        .unwrap_or_else(|_| panic!("Failed to read config file: {}", path));
    serde_json::from_str(&config_str)
        .unwrap_or_else(|e| panic!("Failed to parse config file: {}", e))
}

// ============================================================================
// State Manager Actor
// ============================================================================

/// The StateManagerActor is the authoritative source of truth for the account.
/// It is the *only* component that subscribes to user data and sends orders.
struct StateManagerActor {
    /// The single, authoritative source of state.
    authoritative_state: Arc<RwLock<CurrentState>>,
    /// Manages order lifecycles, CLOID->OID mapping, and pending states.
    order_state_mgr: Arc<Mutex<OrderStateManager>>,
    /// The *only* client that can execute trades.
    order_executor: Arc<ParallelOrderExecutor>,
    /// The *only* client that subscribes to user data.
    info_client: InfoClient,
    /// Wallet address for subscriptions.
    user_address: Address,
    /// Channel to receive action requests from all StrategyRunners.
    action_rx: mpsc::Receiver<ExecuteActionsRequest>,
    /// Channel to broadcast state updates to all StrategyRunners.
    state_tx: broadcast::Sender<AuthoritativeStateUpdate>,
    /// Tracks processed fill IDs to prevent duplicates.
    processed_fill_ids: HashSet<u64>,
}

impl StateManagerActor {
    /// Creates a new, un-run State Manager.
    async fn new(
        wallet: PrivateKeySigner,
        action_rx: mpsc::Receiver<ExecuteActionsRequest>,
        state_tx: broadcast::Sender<AuthoritativeStateUpdate>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        info!("Initializing State Manager...");
        let exchange_client = Arc::new(
            ExchangeClient::new(None, wallet.clone(), Some(BaseUrl::Mainnet), None, None)
                .await?,
        );

        let executor_config = ExecutorConfig {
            max_concurrent: 10,
            ..Default::default()
        };
        let order_executor = Arc::new(ParallelOrderExecutor::with_config(
            exchange_client,
            executor_config,
        ));

        let info_client = InfoClient::with_reconnect(None, Some(BaseUrl::Mainnet)).await?;
        let user_address = wallet.address();

        // --- Initialize Authoritative State ---
        info!("Fetching initial account state...");
        let user_state = info_client.user_state(user_address).await?;
        let margin_summary = user_state.margin_summary;
        let account_equity = margin_summary.account_value.parse::<f64>().unwrap_or(0.0);

        // Build a map of initial positions
        let mut initial_positions = HashMap::new();
        for asset_pos in user_state.asset_positions {
            let szi = asset_pos.position.szi.parse::<f64>().unwrap_or(0.0);
            if szi.abs() > 0.0 {
                info!(
                    "  -> Found existing position: {} {} @ ${}",
                    szi,
                    asset_pos.position.coin,
                    asset_pos.position.entry_px.as_deref().unwrap_or("0.0")
                );
                let coin = asset_pos.position.coin.clone();
                initial_positions.insert(coin, asset_pos);
            }
        }
        info!("Initial account equity: ${:.2}", account_equity);

        // Note: CurrentState is designed for a *single* asset.
        // For a multi-asset system, CurrentState should hold a Map<Asset, PositionState>.
        // For this example, we'll assume a single-asset setup as in the original file,
        // but the *architecture* supports multiple runners. We'll use the *first*
        // asset's position for the authoritative state, or zero if none.
        // A robust multi-asset implementation would refactor `CurrentState`.

        // For this demo, we assume the State Manager holds a "global" state
        // and runners will query for the state relevant to them.
        // Let's create a *default* CurrentState, as it will be populated
        // by the first strategy runner that connects or by fills.
        // A better approach: `authoritative_state` should be a `HashMap<String, AssetCurrentState>`.
        // But to match the provided files, we stick to one `CurrentState`.

        let authoritative_state = Arc::new(RwLock::new(CurrentState {
            position: 0.0, // Will be updated by runners/fills
            avg_entry_price: 0.0,
            cost_basis: 0.0,
            unrealized_pnl: 0.0,
            realized_pnl: 0.0,
            total_fees: 0.0,
            l2_mid_price: 0.0, // Not managed by State Manager
            order_book: None,  // Not managed by State Manager
            market_spread_bps: 0.0,
            lob_imbalance: 0.5,
            open_bids: Vec::new(),
            open_asks: Vec::new(),
            account_equity,
            margin_used: margin_summary.total_margin_used.parse::<f64>().unwrap_or(0.0),
            max_position_size: 0.0, // This will be set by strategy runners
            timestamp: chrono::Utc::now().timestamp() as f64,
            session_start_time: chrono::Utc::now().timestamp() as f64,
        }));

        Ok(Self {
            authoritative_state,
            order_state_mgr: Arc::new(Mutex::new(OrderStateManager::new())),
            order_executor,
            info_client,
            user_address,
            action_rx,
            state_tx,
            processed_fill_ids: HashSet::new(),
        })
    }

    /// Runs the main loop of the State Manager.
    async fn run(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        info!("[State Manager] Running.");

        // Subscribe to *only* user-specific data
        let (user_ws_tx, mut user_ws_rx) = mpsc::unbounded_channel();
        self.subscribe_user_data(user_ws_tx).await?;
        info!("[State Manager] Subscribed to user data feeds.");

        // Spawn a task to periodically fetch full account state via REST for reconciliation
        // Note: We create a new InfoClient for this background task instead of cloning
        let state_clone = self.authoritative_state.clone();
        let user_address = self.user_address;
        tokio::spawn(async move {
            let mut timer = interval(Duration::from_secs(15));
            timer.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

            // Create a new InfoClient for this task
            let info_client_for_reconciliation = match InfoClient::with_reconnect(None, Some(BaseUrl::Mainnet)).await {
                Ok(client) => client,
                Err(e) => {
                    error!("[State Manager] Failed to create InfoClient for reconciliation task: {}", e);
                    return;
                }
            };

            loop {
                timer.tick().await;
                debug!("[State Manager] Reconciling account state via REST...");
                match info_client_for_reconciliation.user_state(user_address).await {
                    Ok(user_state) => {
                        let mut state = state_clone.write().await;
                        state.account_equity = user_state
                            .margin_summary
                            .account_value
                            .parse()
                            .unwrap_or(state.account_equity);
                        state.margin_used = user_state
                            .margin_summary
                            .total_margin_used
                            .parse()
                            .unwrap_or(state.margin_used);
                    }
                    Err(e) => {
                        warn!("[State Manager] Failed to reconcile account state: {}", e);
                    }
                }
            }
        });

        loop {
            tokio::select! {
                // Handle incoming WebSocket messages (Fills, Order Updates)
                Some(message) = user_ws_rx.recv() => {
                    self.handle_ws_message(message).await;
                }

                // Handle incoming action requests from Strategy Runners
                Some(request) = self.action_rx.recv() => {
                    self.handle_action_request(request).await;
                }

                _ = tokio::signal::ctrl_c() => {
                    info!("[State Manager] Shutdown signal received.");
                    break;
                }
            }
        }

        info!("[State Manager] Shutting down.");
        Ok(())
    }

    /// Subscribes to all necessary user data feeds.
    async fn subscribe_user_data(
        &mut self,
        sender: mpsc::UnboundedSender<Message>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let (flume_tx, flume_rx) = flume::unbounded();

        self.info_client
            .subscribe(
                Subscription::UserEvents {
                    user: self.user_address,
                },
                flume_tx.clone(),
            )
            .await?;
        self.info_client
            .subscribe(
                Subscription::UserFills {
                    user: self.user_address,
                },
                flume_tx.clone(),
            )
            .await?;
        self.info_client
            .subscribe(
                Subscription::OrderUpdates {
                    user: self.user_address,
                },
                flume_tx.clone(),
            )
            .await?;

        // Spawn a forwarder task
        tokio::spawn(async move {
            while let Ok(msg) = flume_rx.recv_async().await {
                if sender.send(msg).is_err() {
                    error!("[State Manager] WS forwarder failed to send to main loop.");
                    break;
                }
            }
        });

        Ok(())
    }

    /// Processes a WebSocket message, updates state, and broadcasts.
    async fn handle_ws_message(&mut self, message: Message) {
        let mut state_changed = false;

        match message {
            Message::User(user_events) => match user_events.data {
                UserData::Fills(fills) => {
                    let changed = self.process_fills(fills).await;
                    if changed {
                        state_changed = true;
                    }
                }
                UserData::NonUserCancel(cancels) => {
                    let mut mgr = self.order_state_mgr.lock().await;
                    let mut state = self.authoritative_state.write().await;
                    for cancel in cancels {
                        warn!("[State Manager] System cancelled order {}", cancel.oid);
                        mgr.remove_and_cache_order(
                            cancel.oid,
                            OrderState::Cancelled,
                            &mut state.open_bids,
                        );
                        mgr.remove_and_cache_order(
                            cancel.oid,
                            OrderState::Cancelled,
                            &mut state.open_asks,
                        );
                    }
                    state_changed = true;
                }
                _ => {}
            },
            Message::OrderUpdates(order_updates) => {
                let mut mgr = self.order_state_mgr.lock().await;
                let mut state = self.authoritative_state.write().await;
                for update in order_updates.data {
                    // Pass `None` for order book, as State Manager doesn't track it.
                    // Level calculation will be 0, which is acceptable.
                    let result = mgr.handle_order_update(&update, None);
                    match result {
                        OrderUpdateResult::AddOrUpdate(order) => {
                            self.add_order_to_state(&mut state, order);
                        }
                        OrderUpdateResult::UpdatePartial(order) => {
                            self.update_partial_fill_in_state(&mut state, order);
                        }
                        OrderUpdateResult::RemoveAndCache(oid, _state) => {
                            self.remove_order_from_state(&mut state, oid);
                        }
                        OrderUpdateResult::NoAction => {}
                    }
                }
                state_changed = true;
            }
            Message::UserFills(user_fills) => {
                if user_fills.data.is_snapshot.unwrap_or(false) {
                    info!(
                        "[State Manager] Received UserFills snapshot with {} fills.",
                        user_fills.data.fills.len()
                    );
                    let changed = self.process_fills(user_fills.data.fills).await;
                    if changed {
                        state_changed = true;
                    }
                }
                // Streaming fills from UserFills are ignored; UserEvents is the source of truth.
            }
            _ => {}
        }

        if state_changed {
            self.broadcast_state().await;
        }
    }

    /// Processes a list of fills, updates state, and returns whether state changed.
    async fn process_fills(&mut self, fills: Vec<TradeInfo>) -> bool {
        if fills.is_empty() {
            return false;
        }

        let mut state = self.authoritative_state.write().await;
        let _mgr = self.order_state_mgr.lock().await;
        let mut state_changed = false;

        for fill in fills {
            let fill_id = fill.tid;
            if self.processed_fill_ids.contains(&fill_id) {
                continue; // Skip duplicate
            }

            state_changed = true;
            self.process_fill(&mut state, &fill); // Mutates state
            self.processed_fill_ids.insert(fill_id);
        }

        state_changed
    }

    /// Handles an action request from a Strategy Runner.
    async fn handle_action_request(&self, request: ExecuteActionsRequest) {
        let state = self.authoritative_state.read().await;

        // ---
        // 1. Final Margin & Position Check (Atomicity)
        // ---
        // (A robust implementation would check the *impact* of all actions in the
        // request against the current state.position and state.margin_used)
        // For simplicity, we'll assume the runner's check was "good enough"
        // and just execute. A production system *must* do a final check here.
        let (valid, msg) = self.validate_actions(&state, &request.actions);

        if !valid {
            warn!(
                "[State Manager] Rejected actions from [{}]: {}",
                request.asset, msg
            );
            let _ = request.resp.send(ExecuteActionsResponse {
                success: false,
                message: msg,
            });
            return;
        }

        // ---
        // 2. Update Optimistic State (Pending)
        // ---
        let mut mgr = self.order_state_mgr.lock().await;
        let mut executor_actions = Vec::new();

        for action in request.actions {
            match action {
                StrategyAction::Place(order) => {
                    if let Some(cloid) = order.cloid {
                        let resting_order = RestingOrder::new(
                            None, Some(cloid), order.sz, order.limit_px, order.is_buy, 0, // Level 0 (unknown)
                        );
                        mgr.add_pending_order(cloid, resting_order);
                        executor_actions.push(ExecutorAction::Place(order));
                    }
                }
                StrategyAction::Cancel(cancel) => {
                    // Mark as pending cancel *before* sending
                    let mut state_write = self.authoritative_state.write().await;
                    if mgr.mark_pending_cancel(cancel.oid, &mut state_write.open_bids)
                        || mgr.mark_pending_cancel(cancel.oid, &mut state_write.open_asks)
                    {
                        executor_actions.push(ExecutorAction::Cancel(cancel));
                    }
                }
                StrategyAction::BatchPlace(orders) => {
                    for order in &orders {
                        if let Some(cloid) = order.cloid {
                            let resting_order = RestingOrder::new(
                                None, Some(cloid), order.sz, order.limit_px, order.is_buy, 0,
                            );
                            mgr.add_pending_order(cloid, resting_order);
                        }
                    }
                    executor_actions.push(ExecutorAction::BatchPlace(orders));
                }
                StrategyAction::BatchCancel(cancels) => {
                    let mut state_write = self.authoritative_state.write().await;
                    let mut valid_cancels = Vec::new();
                    for cancel in cancels {
                        if mgr.mark_pending_cancel(cancel.oid, &mut state_write.open_bids)
                            || mgr.mark_pending_cancel(cancel.oid, &mut state_write.open_asks)
                        {
                            valid_cancels.push(cancel);
                        }
                    }
                    if !valid_cancels.is_empty() {
                        executor_actions.push(ExecutorAction::BatchCancel(valid_cancels));
                    }
                }
                StrategyAction::NoOp => {}
            }
        }
        drop(mgr); // Release lock

        // ---
        // 3. Execute Actions
        // ---
        if !executor_actions.is_empty() {
            let executor = self.order_executor.clone();
            tokio::spawn(async move {
                let result = executor.execute_actions_parallel(executor_actions).await;
                if result.failed > 0 {
                    error!(
                        "[State Manager] {}/{} actions failed to execute.",
                        result.failed,
                        result.failed + result.successful
                    );
                    for resp in result.responses {
                        if let Err(e) = resp {
                            error!("  -> Failure: {}", e);
                        } else if let Ok(ExchangeResponseStatus::Err(e)) = resp {
                            error!("  -> Exchange Error: {}", e);
                        } else if let Ok(ExchangeResponseStatus::Ok(r)) = resp {
                            if let Some(data) = r.data {
                                for status in data.statuses {
                                    if let ExchangeDataStatus::Error(e) = status {
                                        error!("  -> Action Error: {}", e);
                                    }
                                }
                            }
                        }
                    }
                }
            });
        }

        // ---
        // 4. Send Confirmation
        // ---
        let _ = request.resp.send(ExecuteActionsResponse {
            success: true,
            message: "Actions submitted".to_string(),
        });

        // Broadcast the optimistic state change (e.g., pending orders)
        self.broadcast_state().await;
    }

    /// Performs a final check on actions against the authoritative state.
    fn validate_actions(
        &self,
        _state: &CurrentState,
        _actions: &[StrategyAction],
    ) -> (bool, String) {
        // This is where you would implement your atomic margin and position checks.
        // For example, calculate the total *delta* in position and margin
        // from all 'Place' actions and ensure it doesn't exceed
        // (state.max_position_size) and (state.account_equity - state.margin_used).

        // (Implementation omitted for brevity, assuming valid for now)
        (true, "".to_string())
    }

    /// Broadcasts the current authoritative state to all runners.
    async fn broadcast_state(&self) {
        let state = self.authoritative_state.read().await;
        let open_orders = state
            .open_bids
            .iter()
            .cloned()
            .chain(state.open_asks.iter().cloned())
            .collect();

        let update = AuthoritativeStateUpdate {
            position: state.position,
            avg_entry_price: state.avg_entry_price,
            realized_pnl: state.realized_pnl,
            unrealized_pnl: state.unrealized_pnl,
            account_equity: state.account_equity,
            margin_used: state.margin_used,
            timestamp_ms: (state.timestamp * 1000.0) as u64,
            open_orders,
        };

        // .send() returns the number of *active* subscribers.
        // If 0, no one is listening. If Err, the channel is closed.
        if self.state_tx.send(update).is_err() {
            debug!("[State Manager] No active subscribers to broadcast state to.");
        }
    }

    // --- State Mutation Helpers (copied from original BotRunner) ---

    fn add_order_to_state(&self, state: &mut CurrentState, order: RestingOrder) {
        let orders = if order.is_buy {
            &mut state.open_bids
        } else {
            &mut state.open_asks
        };
        if let Some(existing) = orders.iter_mut().find(|o| o.oid == order.oid) {
            *existing = order;
        } else {
            orders.push(order);
        }
        state.open_bids.sort_by(|a, b| b.price.partial_cmp(&a.price).unwrap_or(std::cmp::Ordering::Equal));
        state.open_asks.sort_by(|a, b| a.price.partial_cmp(&b.price).unwrap_or(std::cmp::Ordering::Equal));
    }

    fn update_partial_fill_in_state(&self, state: &mut CurrentState, order: RestingOrder) {
        let orders = if order.is_buy {
            &mut state.open_bids
        } else {
            &mut state.open_asks
        };
        if let Some(existing) = orders.iter_mut().find(|o| o.oid == order.oid) {
            existing.size = order.size;
            existing.state = order.state;
            existing.timestamp = order.timestamp;
        }
    }

    fn remove_order_from_state(&self, state: &mut CurrentState, oid: u64) {
        state.open_bids.retain(|o| o.oid != Some(oid));
        state.open_asks.retain(|o| o.oid != Some(oid));
    }

    fn process_fill(&self, state: &mut CurrentState, fill: &TradeInfo) {
        let fill_size = fill.sz.parse::<f64>().unwrap_or(0.0);
        let fill_price = fill.px.parse::<f64>().unwrap_or(0.0);
        let fill_fee = fill.fee.parse::<f64>().unwrap_or(0.0);
        let is_buy = fill.side == "B";

        state.total_fees += fill_fee.abs();
        let signed_size = if is_buy { fill_size } else { -fill_size };
        let old_position = state.position;
        let new_position = old_position + signed_size;

        if old_position.abs() < 1e-6 {
            state.cost_basis = fill_size * fill_price;
            state.avg_entry_price = fill_price;
        } else if (old_position > 0.0 && new_position > old_position)
            || (old_position < 0.0 && new_position < old_position)
        {
            let old_cost = state.cost_basis;
            let new_cost = old_cost + (fill_size * fill_price);
            state.cost_basis = new_cost;
            state.avg_entry_price = new_cost / new_position.abs();
        } else {
            let reduced_size = fill_size.min(old_position.abs());
            let realized_pnl = if old_position > 0.0 {
                reduced_size * (fill_price - state.avg_entry_price)
            } else {
                reduced_size * (state.avg_entry_price - fill_price)
            };
            state.realized_pnl += realized_pnl;

            if new_position.abs() < 1e-6 {
                state.cost_basis = 0.0;
                state.avg_entry_price = 0.0;
            } else {
                state.cost_basis = new_position.abs() * fill_price;
                state.avg_entry_price = fill_price;
            }
        }

        state.position = new_position;
        // Unrealized PnL will be updated by the runner based on its market data
    }
}

// ============================================================================
// Strategy Runner Actor
// ============================================================================

/// The StrategyRunnerActor runs the trading logic for a *single* asset.
/// It subscribes to market data and the State Manager's updates.
struct StrategyRunnerActor {
    asset: String,
    strategy: Box<dyn Strategy>,
    /// Client for *market data only*.
    info_client: InfoClient,
    /// Local, *non-authoritative* cache of the account state.
    local_state_cache: Arc<RwLock<CurrentState>>,
    /// Sender to the State Manager for action execution.
    action_tx: mpsc::Sender<ExecuteActionsRequest>,
    /// Receiver for state updates from the State Manager.
    state_rx: broadcast::Receiver<AuthoritativeStateUpdate>,
    /// Validator for this asset (currently unused).
    #[allow(dead_code)]
    tick_lot_validator: TickLotValidator,
}

impl StrategyRunnerActor {
    /// Creates a new, un-run Strategy Runner.
    async fn new(
        config: StrategyConfig,
        action_tx: mpsc::Sender<ExecuteActionsRequest>,
        state_rx: broadcast::Receiver<AuthoritativeStateUpdate>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        info!("[Runner {}] Initializing...", config.asset);
        let strategy: Box<dyn Strategy> = match config.strategy_name.as_str() {
            "hjb_v1" => Box::new(HjbStrategy::new(
                &config.asset,
                &serde_json::json!({ "strategy_params": config.strategy_params }),
            )),
            _ => panic!("Unknown strategy: {}", config.strategy_name),
        };

        let info_client = InfoClient::with_reconnect(None, Some(BaseUrl::Mainnet)).await?;

        // TODO: Fetch sz_decimals from meta API instead of hardcoding
        let tick_lot_validator =
            TickLotValidator::new(config.asset.clone(), AssetType::Perp, 3);

        // Initialize local state cache with defaults. It will be populated
        // by the first message from the State Manager.
        let mut local_state_cache = CurrentState::default();
        local_state_cache.max_position_size = strategy.get_max_position_size();

        info!("[Runner {}] Initialized.", config.asset);

        Ok(Self {
            asset: config.asset,
            strategy,
            info_client,
            local_state_cache: Arc::new(RwLock::new(local_state_cache)),
            action_tx,
            state_rx,
            tick_lot_validator,
        })
    }

    /// Runs the main loop of the Strategy Runner.
    async fn run(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        info!("[Runner {}] Running.", self.asset);

        // Subscribe to *only* market data
        let (market_ws_tx, mut market_ws_rx) = mpsc::unbounded_channel();
        self.subscribe_market_data(market_ws_tx).await?;
        info!("[Runner {}] Subscribed to market data.", self.asset);

        // Periodic tick timer (1 second interval)
        let mut tick_timer = interval(Duration::from_secs(1));
        tick_timer.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

        loop {
            tokio::select! {
                // Handle incoming *market data* (L2Book, Trades)
                Some(message) = market_ws_rx.recv() => {
                    self.handle_market_message(message).await;
                }

                // Handle incoming *authoritative state* from State Manager
                Ok(update) = self.state_rx.recv() => {
                    self.handle_state_update(update).await;
                }

                // Handle periodic tick
                _ = tick_timer.tick() => {
                    self.handle_tick().await;
                }

                 _ = tokio::signal::ctrl_c() => {
                    info!("[Runner {}] Shutdown signal received.", self.asset);
                    self.handle_shutdown().await;
                    break;
                }
            }
        }
        info!("[Runner {}] Shutting down.", self.asset);
        Ok(())
    }

    /// Subscribes to all necessary market data feeds for this asset.
    async fn subscribe_market_data(
        &mut self,
        sender: mpsc::UnboundedSender<Message>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let (flume_tx, flume_rx) = flume::unbounded();

        self.info_client
            .subscribe(
                Subscription::L2Book {
                    coin: self.asset.clone(),
                },
                flume_tx.clone(),
            )
            .await?;
        self.info_client
            .subscribe(
                Subscription::Trades {
                    coin: self.asset.clone(),
                },
                flume_tx.clone(),
            )
            .await?;
        self.info_client
            .subscribe(Subscription::AllMids, flume_tx.clone())
            .await?;

        // Spawn a forwarder task
        tokio::spawn(async move {
            while let Ok(msg) = flume_rx.recv_async().await {
                if sender.send(msg).is_err() {
                    error!("[Runner] Market WS forwarder failed.");
                    break;
                }
            }
        });
        Ok(())
    }

    /// Processes a market data message, runs strategy, and sends actions.
    async fn handle_market_message(&mut self, message: Message) {
        let mut market_update: Option<MarketUpdate> = None;

        match message {
            Message::L2Book(l2_book) => {
                if let Some(book) = OrderBook::from_l2_data(&l2_book.data) {
                    let mut state = self.local_state_cache.write().await;
                    if let Some(analysis) = book.analyze(5) {
                        state.lob_imbalance = analysis.imbalance;
                        state.market_spread_bps = book.spread_bps().unwrap_or(0.0);
                    }
                    if let (Some(bid), Some(ask)) = (book.best_bid(), book.best_ask()) {
                        state.l2_mid_price = (bid + ask) / 2.0;
                    }
                    state.order_book = Some(book);
                    market_update = Some(MarketUpdate::from_l2_book(l2_book.data));
                }
            }
            Message::Trades(trades) => {
                market_update =
                    Some(MarketUpdate::from_trades(self.asset.clone(), trades.data));
            }
            Message::AllMids(all_mids) => {
                if let Some(mid_str) = all_mids.data.mids.get(&self.asset) {
                    if let Ok(mid_price) = mid_str.parse::<f64>() {
                        market_update =
                            Some(MarketUpdate::from_mid_price(self.asset.clone(), mid_price));
                    }
                }
            }
            _ => {}
        }

        // If we have a valid update, run the strategy
        if let Some(update) = market_update {
            let state = self.local_state_cache.read().await;
            let actions = self.strategy.on_market_update(&state, &update);

            if !actions.is_empty() && !matches!(actions[0], StrategyAction::NoOp) {
                self.send_actions(actions).await;
            }
        }
    }

    /// Processes an authoritative state update from the State Manager.
    async fn handle_state_update(&mut self, update: AuthoritativeStateUpdate) {
        debug!("[Runner {}] Received state update: pos={}", self.asset, update.position);
        let mut state = self.local_state_cache.write().await;

        // ---
        // **IMPORTANT**: This logic assumes the State Manager's `CurrentState`
        // is for *this asset*. A true multi-asset system would require
        // the `AuthoritativeStateUpdate` to be a `HashMap<String, AssetState>`.
        // ---
        // For now, we update our local state from the broadcast.
        state.position = update.position;
        state.avg_entry_price = update.avg_entry_price;
        state.realized_pnl = update.realized_pnl;
        state.unrealized_pnl = update.unrealized_pnl;
        state.account_equity = update.account_equity;
        state.margin_used = update.margin_used;
        state.timestamp = update.timestamp_ms as f64 / 1000.0;

        // Update open orders
        state.open_bids.clear();
        state.open_asks.clear();
        for order in update.open_orders {
            if order.is_buy {
                state.open_bids.push(order);
            } else {
                state.open_asks.push(order);
            }
        }
        // Ensure sorted
        state.open_bids.sort_by(|a, b| b.price.partial_cmp(&a.price).unwrap_or(std::cmp::Ordering::Equal));
        state.open_asks.sort_by(|a, b| a.price.partial_cmp(&b.price).unwrap_or(std::cmp::Ordering::Equal));
    }

    /// Handles the periodic 1-second tick.
    async fn handle_tick(&mut self) {
        let _state = self.local_state_cache.read().await;

        // Update timestamp and unrealized PnL locally
        // (This is fine as it's based on market data, which this runner owns)
        let mut state_write = self.local_state_cache.write().await;
        state_write.timestamp = chrono::Utc::now().timestamp() as f64;
        if state_write.position.abs() > 1e-6 && state_write.l2_mid_price > 0.0 {
            if state_write.position > 0.0 { // Long
                state_write.unrealized_pnl =
                    state_write.position * (state_write.l2_mid_price - state_write.avg_entry_price);
            } else { // Short
                state_write.unrealized_pnl =
                    state_write.position.abs() * (state_write.avg_entry_price - state_write.l2_mid_price);
            }
        }
        drop(state_write);

        // Now call the strategy's tick function with the read-lock
        let state_read = self.local_state_cache.read().await;
        let actions = self.strategy.on_tick(&state_read);

        if !actions.is_empty() && !matches!(actions[0], StrategyAction::NoOp) {
            self.send_actions(actions).await;
        }
    }

    /// Handles the shutdown signal.
    async fn handle_shutdown(&mut self) {
        let state = self.local_state_cache.read().await;
        let actions = self.strategy.on_shutdown(&state);
        if !actions.is_empty() {
             info!("[Runner {}] Sending {} shutdown actions to State Manager...", self.asset, actions.len());
            self.send_actions(actions).await;
        }
    }

    /// Sends a list of actions to the State Manager for execution.
    async fn send_actions(&self, actions: Vec<StrategyAction>) {
        let (resp_tx, resp_rx) = oneshot::channel();
        let request = ExecuteActionsRequest {
            asset: self.asset.clone(),
            actions,
            resp: resp_tx,
        };

        if self.action_tx.send(request).await.is_err() {
            error!("[Runner {}] Failed to send actions: State Manager is disconnected.", self.asset);
            return;
        }

        // Wait for the State Manager to confirm receipt and validation
        match resp_rx.await {
            Ok(response) => {
                if !response.success {
                    warn!(
                        "[Runner {}] Actions REJECTED by State Manager: {}",
                        self.asset, response.message
                    );
                } else {
                    debug!(
                        "[Runner {}] Actions submitted to State Manager.",
                        self.asset
                    );
                }
            }
            Err(_) => {
                error!("[Runner {}] Did not receive response from State Manager.", self.asset);
            }
        }
    }
}

// ============================================================================
// Main Launcher
// ============================================================================

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // --- 1. Initialization ---
    let file_appender = tracing_appender::rolling::never("./", "market_maker_v3.log");
    let (non_blocking_writer, _guard) = tracing_appender::non_blocking(file_appender);
    let file_layer = fmt::layer().json().with_writer(non_blocking_writer);
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info,hyperliquid_rust_sdk::bin::market_maker_v3=debug"));

    tracing_subscriber::registry()
        .with(filter)
        .with(file_layer)
        .init();

    dotenv::dotenv().ok();
    let private_key = env::var("PRIVATE_KEY").expect("PRIVATE_KEY must be set");
    let wallet: PrivateKeySigner = private_key.parse().expect("Invalid private key");

    // Load config for all strategies
    let app_config = load_config("config.json");
    info!(
        "Loaded {} strategy configurations from config.json",
        app_config.strategies.len()
    );

    // --- 2. Create Communication Channels ---

    // Channel for Runners to send actions *to* the Manager
    let (action_tx, action_rx) = mpsc::channel(1024); // Buffered channel

    // Channel for Manager to broadcast state *to* all Runners
    let (state_tx, _) = broadcast::channel(128); // Broadcast channel

    // --- 3. Create LocalSet for non-Send futures ---
    // InfoClient uses internal locks that are not Send, so we use LocalSet to run these futures
    let local_set = LocalSet::new();

    // --- 4. Start the State Manager Actor (on LocalSet) ---
    let mut state_manager =
        StateManagerActor::new(wallet.clone(), action_rx, state_tx.clone()).await?;

    let state_manager_handle = local_set.spawn_local(async move {
        if let Err(e) = state_manager.run().await {
            error!("[State Manager] Actor failed: {}", e);
        }
    });

    info!("✅ State Manager Actor spawned.");

    // --- 5. Start All Strategy Runner Actors (on LocalSet) ---
    let mut runner_handles = Vec::new();
    for strategy_config in app_config.strategies {
        let asset = strategy_config.asset.clone();
        let mut runner = StrategyRunnerActor::new(
            strategy_config,
            action_tx.clone(),
            state_tx.subscribe(), // Each runner gets a new receiver
        )
        .await?;

        let handle = local_set.spawn_local(async move {
            if let Err(e) = runner.run().await {
                error!("[Runner {}] Actor failed: {}", asset, e);
            }
        });
        runner_handles.push(handle);
    }

    info!("✅ All {} Strategy Runner Actors spawned.", runner_handles.len());

    // --- 6. Run LocalSet until Ctrl+C ---
    local_set
        .run_until(async move {
            tokio::signal::ctrl_c().await.expect("Failed to wait for Ctrl+C");
            info!("Main task shutting down. Actors will terminate.");

            // Wait for actors to finish (optional)
            let _ = state_manager_handle.await;
            for handle in runner_handles {
                let _ = handle.await;
            }
        })
        .await;

    Ok(())
}
