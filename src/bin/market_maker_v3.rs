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
use hyperliquid_rust_sdk::strategies::hjb_strategy::{HjbStrategy, MarginCalculator};
// Import our new IPC message types
use hyperliquid_rust_sdk::ipc::{
    AssetState, AuthoritativeStateUpdate, ExecuteActionsRequest, ExecuteActionsResponse,
    GlobalAccountState,
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

// RUST_LOG=info,debug cargo run --release --bin market_maker_v3

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
///
/// Multi-Asset Architecture:
/// - Maintains separate state for each trading asset
/// - Tracks global account equity and margin usage
/// - Broadcasts asset-specific updates to prevent cross-contamination
struct StateManagerActor {
    /// Per-asset state tracking (position, orders, PnL for each asset)
    asset_states: Arc<RwLock<HashMap<String, AssetState>>>,
    /// Global account state (equity, margin used)
    global_account_state: Arc<RwLock<GlobalAccountState>>,
    /// Manages order lifecycles, CLOID->OID mapping, and pending states
    order_state_mgr: Arc<Mutex<OrderStateManager>>,
    /// The *only* client that can execute trades
    order_executor: Arc<ParallelOrderExecutor>,
    /// The *only* client that subscribes to user data
    info_client: InfoClient,
    /// Wallet address for subscriptions
    user_address: Address,
    /// Channel to receive action requests from all StrategyRunners
    action_rx: mpsc::Receiver<ExecuteActionsRequest>,
    /// Channel to broadcast state updates to all StrategyRunners
    state_tx: broadcast::Sender<AuthoritativeStateUpdate>,
    /// Tracks processed fill IDs to prevent duplicates
    processed_fill_ids: HashSet<u64>,
    /// Tracks which assets have received their initial UserFills snapshot (to ignore historical data)
    snapshot_received: HashMap<String, bool>,
    /// Margin calculator instance
    margin_calculator: MarginCalculator,
    /// Configurable safety buffer (e.g., 0.1 for 10%)
    safety_buffer: f64,
    /// Leverage (assuming consistent across strategies for now)
    leverage: usize,
    /// Max position size (from config)
    max_position_size: f64,
}

impl StateManagerActor {
    /// Creates a new, un-run State Manager.
    ///
    /// # Arguments
    /// * `wallet` - The wallet for signing transactions
    /// * `assets` - List of assets to track (e.g., ["BTC", "ETH", "HYPE"])
    /// * `action_rx` - Channel to receive action requests from runners
    /// * `state_tx` - Channel to broadcast state updates to runners
    /// * `app_config` - Application configuration containing strategy parameters
    async fn new(
        wallet: PrivateKeySigner,
        assets: Vec<String>,
        action_rx: mpsc::Receiver<ExecuteActionsRequest>,
        state_tx: broadcast::Sender<AuthoritativeStateUpdate>,
        app_config: &AppConfig,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        info!("Initializing State Manager for {} assets...", assets.len());
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

        // --- Get Leverage, Safety Buffer, and Max Position Size from Config ---
        // Read from the first strategy config (assuming consistency)
        let (leverage, safety_buffer, max_position_size) = if let Some(first_strategy_cfg) = app_config.strategies.first() {
            let strat_params = &first_strategy_cfg.strategy_params;
            let leverage = strat_params.get("leverage").and_then(|v| v.as_u64()).unwrap_or(3) as usize;
            let buffer = strat_params.get("margin_safety_buffer").and_then(|v| v.as_f64()).unwrap_or(0.2); // Default 20%
            let max_pos = strat_params.get("max_absolute_position_size").and_then(|v| v.as_f64()).unwrap_or(10.0);
            info!("[State Manager] Using Leverage={}x, SafetyBuffer={:.1}%, MaxPosition={} from first strategy config",
                  leverage, buffer * 100.0, max_pos);
            (leverage, buffer, max_pos)
        } else {
            warn!("[State Manager] No strategies in config, using defaults: Leverage=3x, SafetyBuffer=20%, MaxPosition=10.0");
            (3, 0.2, 10.0)
        };

        // --- Fetch Initial Account State ---
        info!("Fetching initial account state...");
        let user_state = info_client.user_state(user_address).await?;
        let margin_summary = user_state.margin_summary;
        let account_equity = margin_summary.account_value.parse::<f64>().unwrap_or(0.0);
        let margin_used = margin_summary.total_margin_used.parse::<f64>().unwrap_or(0.0);

        info!("Initial account equity: ${:.2}", account_equity);
        info!("Initial margin used: ${:.2}", margin_used);

        // --- Initialize Per-Asset States ---
        let mut asset_states_map = HashMap::new();
        for asset in &assets {
            // Look for existing position for this asset
            let mut asset_state = AssetState {
                asset: asset.clone(),
                position: 0.0,
                avg_entry_price: 0.0,
                realized_pnl: 0.0,
                unrealized_pnl: 0.0,
                total_fees: 0.0,
                cost_basis: 0.0,
                open_bids: Vec::new(),
                open_asks: Vec::new(),
                timestamp: chrono::Utc::now().timestamp() as f64,
            };

            // Check if we have an existing position for this asset
            if let Some(asset_pos) = user_state
                .asset_positions
                .iter()
                .find(|ap| ap.position.coin == *asset)
            {
                let szi = asset_pos.position.szi.parse::<f64>().unwrap_or(0.0);
                if szi.abs() > 1e-6 {
                    let entry_px = asset_pos
                        .position
                        .entry_px
                        .as_deref()
                        .unwrap_or("0.0")
                        .parse::<f64>()
                        .unwrap_or(0.0);

                    asset_state.position = szi;
                    asset_state.avg_entry_price = entry_px;
                    asset_state.cost_basis = szi.abs() * entry_px;

                    info!(
                        "  -> Found existing position for {}: {} @ ${:.3}",
                        asset, szi, entry_px
                    );
                }
            }

            asset_states_map.insert(asset.clone(), asset_state);
        }

        // --- Initialize Global Account State ---
        let global_account_state = Arc::new(RwLock::new(GlobalAccountState {
            account_equity,
            margin_used,
            timestamp_ms: chrono::Utc::now().timestamp_millis() as u64,
        }));

        Ok(Self {
            asset_states: Arc::new(RwLock::new(asset_states_map)),
            global_account_state,
            order_state_mgr: Arc::new(Mutex::new(OrderStateManager::new())),
            order_executor,
            info_client,
            user_address,
            action_rx,
            state_tx,
            processed_fill_ids: HashSet::new(),
            snapshot_received: HashMap::new(),
            // Initialize MarginCalculator
            margin_calculator: MarginCalculator::new(leverage, safety_buffer),
            safety_buffer,
            leverage,
            max_position_size,
        })
    }

    /// Runs the main loop of the State Manager.
    async fn run(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        info!("[State Manager] Running.");

        // Subscribe to *only* user-specific data
        let (user_ws_tx, mut user_ws_rx) = mpsc::unbounded_channel();
        self.subscribe_user_data(user_ws_tx).await?;
        info!("[State Manager] Subscribed to user data feeds.");

        // Give runners a moment to start their event loops and subscribe
        tokio::time::sleep(Duration::from_millis(500)).await;

        // Broadcast the initial account state to all runners
        // This ensures runners have correct account_equity and margin_used before trading
        info!("[State Manager] Broadcasting initial account state to all runners...");
        self.broadcast_state().await;
        info!("[State Manager] Initial state broadcast complete.");

        // Spawn a task to periodically fetch full account state via REST for reconciliation
        // CRITICAL FIX: Use try_write() to avoid blocking the State Manager's event loop
        let global_state_clone = self.global_account_state.clone();
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

                // Fetch state from REST API
                let user_state_result = info_client_for_reconciliation.user_state(user_address).await;

                match user_state_result {
                    Ok(user_state) => {
                        // Parse values before attempting lock
                        let account_equity: f64 = user_state
                            .margin_summary
                            .account_value
                            .parse()
                            .unwrap_or(0.0);
                        let margin_used: f64 = user_state
                            .margin_summary
                            .total_margin_used
                            .parse()
                            .unwrap_or(0.0);
                        let timestamp_ms = chrono::Utc::now().timestamp_millis() as u64;

                        // Try to acquire write lock with exponential backoff
                        let mut backoff_ms = 10;
                        let max_attempts = 5;
                        let mut success = false;

                        for attempt in 1..=max_attempts {
                            match global_state_clone.try_write() {
                                Ok(mut state) => {
                                    // Successfully acquired lock - update quickly
                                    state.account_equity = account_equity;
                                    state.margin_used = margin_used;
                                    state.timestamp_ms = timestamp_ms;
                                    debug!("[State Manager] ✓ Reconciliation complete: equity=${:.2}, margin=${:.2}",
                                        account_equity, margin_used);
                                    success = true;
                                    break;
                                }
                                Err(_) => {
                                    if attempt < max_attempts {
                                        debug!("[State Manager] Reconciliation: Lock contention, retrying in {}ms (attempt {}/{})",
                                            backoff_ms, attempt, max_attempts);
                                        tokio::time::sleep(Duration::from_millis(backoff_ms)).await;
                                        backoff_ms *= 2; // Exponential backoff
                                    }
                                }
                            }
                        }

                        if !success {
                            warn!("[State Manager] ⚠️  Reconciliation skipped: Could not acquire lock after {} attempts", max_attempts);
                        }
                    }
                    Err(e) => {
                        warn!("[State Manager] Failed to reconcile account state: {}", e);
                    }
                }
            }
        });

        // Watchdog timer to detect stalled event loop
        let mut watchdog_timer = interval(Duration::from_secs(30));
        watchdog_timer.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

        // Message timeout detection
        let mut last_ws_message_time = std::time::Instant::now();
        let ws_timeout = Duration::from_secs(60); // Alert if no WS messages for 60 seconds

        let mut healthcheck_timer = interval(Duration::from_secs(10));
        healthcheck_timer.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

        // 🔍 DEADLOCK DETECTION: Track loop iterations
        let mut loop_iterations = 0u64;
        let mut last_iteration_log = std::time::Instant::now();

        loop {
            let loop_start = std::time::Instant::now();
            loop_iterations += 1;

            // Log iteration count every 10 seconds
            if last_iteration_log.elapsed() >= Duration::from_secs(10) {
                info!("[State Manager] 🔄 Event loop alive: {} iterations in last {:.1}s",
                    loop_iterations, last_iteration_log.elapsed().as_secs_f64());
                loop_iterations = 0;
                last_iteration_log = std::time::Instant::now();
            }

            // CRITICAL FIX: Use biased selection to prioritize message processing over timers
            // This prevents timer branches from starving critical message handling
            tokio::select! {
                biased;

                // HIGHEST PRIORITY: Handle incoming WebSocket messages (Fills, Order Updates)
                Some(message) = user_ws_rx.recv() => {
                    debug!("[State Manager] Received WebSocket message (iteration #{})", loop_iterations);
                    last_ws_message_time = std::time::Instant::now();
                    self.handle_ws_message(message).await;
                }

                // HIGH PRIORITY: Handle incoming action requests from Strategy Runners
                Some(request) = self.action_rx.recv() => {
                    debug!("[State Manager] Received action request (iteration #{})", loop_iterations);
                    self.handle_action_request(request).await;
                }

                // LOW PRIORITY: Timeout detection (only fires if no messages for 2 seconds)
                _ = tokio::time::sleep(Duration::from_secs(2)) => {
                    warn!("[State Manager] ⏰ SELECT TIMEOUT: No events processed for 2 seconds!");
                    warn!("[State Manager]   -> This may indicate a deadlock or event starvation");
                }

                // LOW PRIORITY: Watchdog timer
                _ = watchdog_timer.tick() => {
                    info!("[State Manager] Watchdog: Event loop is running normally (iteration #{})", loop_iterations);
                }

                // LOW PRIORITY: Health check timer
                _ = healthcheck_timer.tick() => {
                    let elapsed = last_ws_message_time.elapsed();
                    if elapsed > ws_timeout {
                        warn!("[State Manager] ⚠️  No WebSocket messages received for {:.1}s - connection may be stalled",
                            elapsed.as_secs_f64());
                    } else {
                        debug!("[State Manager] Health check: Last WS message {:.1}s ago", elapsed.as_secs_f64());
                    }
                }

                // CRITICAL: Shutdown signal
                _ = tokio::signal::ctrl_c() => {
                    info!("[State Manager] Shutdown signal received.");
                    break;
                }
            }

            let iteration_duration = loop_start.elapsed();
            if iteration_duration > Duration::from_millis(100) {
                warn!("[State Manager] ⚠️  Slow iteration: took {:.3}s", iteration_duration.as_secs_f64());
            } else {
                debug!("[State Manager] Loop iteration #{} took {:.3}ms", loop_iterations, iteration_duration.as_millis());
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
        info!("[State Manager] Creating flume channel for user data subscriptions...");
        let (flume_tx, flume_rx) = flume::unbounded();

        info!("[State Manager] Subscribing to UserEvents...");
        self.info_client
            .subscribe(
                Subscription::UserEvents {
                    user: self.user_address,
                },
                flume_tx.clone(),
            )
            .await?;
        info!("[State Manager] ✓ UserEvents subscription successful");

        info!("[State Manager] Subscribing to UserFills...");
        self.info_client
            .subscribe(
                Subscription::UserFills {
                    user: self.user_address,
                },
                flume_tx.clone(),
            )
            .await?;
        info!("[State Manager] ✓ UserFills subscription successful");

        info!("[State Manager] Subscribing to OrderUpdates...");
        self.info_client
            .subscribe(
                Subscription::OrderUpdates {
                    user: self.user_address,
                },
                flume_tx.clone(),
            )
            .await?;
        info!("[State Manager] ✓ OrderUpdates subscription successful");

        // Spawn a forwarder task with diagnostic logging
        info!("[State Manager] Spawning WebSocket forwarder task...");
        tokio::spawn(async move {
            info!("[State Manager WS Forwarder] Started - waiting for messages from WebSocket...");
            let mut message_count = 0u64;
            while let Ok(msg) = flume_rx.recv_async().await {
                message_count += 1;
                if message_count == 1 {
                    info!("[State Manager WS Forwarder] 🎉 First message received from WebSocket!");
                } else if message_count % 100 == 0 {
                    debug!("[State Manager WS Forwarder] Forwarded {} messages", message_count);
                }

                // CRITICAL: Explicitly check if channel send succeeds
                match sender.send(msg) {
                    Ok(_) => {
                        if message_count <= 5 || message_count % 100 == 0 {
                            debug!("[State Manager WS Forwarder] ✓ Sent message #{} to main loop", message_count);
                        }
                    }
                    Err(e) => {
                        error!("[State Manager WS Forwarder] ❌ CHANNEL BROKEN: Failed to send message #{} - {:?}", message_count, e);
                        error!("[State Manager WS Forwarder] Main loop receiver has been dropped! Exiting forwarder.");
                        break;
                    }
                }
            }
            warn!("[State Manager WS Forwarder] Exiting - flume channel closed. Total messages forwarded: {}", message_count);
        });

        info!("[State Manager] WebSocket forwarder task spawned successfully");
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
                    let mut asset_states = self.asset_states.write().await;

                    for cancel in cancels {
                        warn!("[State Manager] System cancelled order {}", cancel.oid);

                        // Find which asset this order belongs to and remove it
                        for (_asset, asset_state) in asset_states.iter_mut() {
                            mgr.remove_and_cache_order(
                                cancel.oid,
                                OrderState::Cancelled,
                                &mut asset_state.open_bids,
                            );
                            mgr.remove_and_cache_order(
                                cancel.oid,
                                OrderState::Cancelled,
                                &mut asset_state.open_asks,
                            );
                        }
                    }
                    state_changed = true;
                }
                _ => {}
            },
            Message::OrderUpdates(order_updates) => {
                let mut mgr = self.order_state_mgr.lock().await;
                let mut asset_states = self.asset_states.write().await;

                for update in order_updates.data {
                    // Extract asset from the update (coin field)
                    let asset = update.order.coin.clone();

                    // Pass `None` for order book, as State Manager doesn't track it.
                    // Level calculation will be 0, which is acceptable.
                    let result = mgr.handle_order_update(&update, None);

                    // Find the asset state for this order
                    if let Some(asset_state) = asset_states.get_mut(&asset) {
                        match result {
                            OrderUpdateResult::AddOrUpdate(order) => {
                                self.add_order_to_asset_state(asset_state, order);
                            }
                            OrderUpdateResult::UpdatePartial(order) => {
                                self.update_partial_fill_in_asset_state(asset_state, order);
                            }
                            OrderUpdateResult::RemoveAndCache(oid, _state) => {
                                self.remove_order_from_asset_state(asset_state, oid);
                            }
                            OrderUpdateResult::NoAction => {}
                        }
                    } else {
                        warn!("[State Manager] Received order update for unknown asset: {}", asset);
                    }
                }
                state_changed = true;
            }
            Message::UserFills(user_fills) => {
                if user_fills.data.is_snapshot.unwrap_or(false) {
                    if user_fills.data.fills.is_empty() {
                        return;
                    }

                    // Get the coin from the first fill to identify which asset this snapshot is for
                    let coin = &user_fills.data.fills[0].coin;

                    // Check if we've already received the initial snapshot for this asset
                    if !self.snapshot_received.get(coin).copied().unwrap_or(false) {
                        info!(
                            "[State Manager] Ignoring historical UserFills snapshot for {} ({} fills) - using REST API position as source of truth",
                            coin,
                            user_fills.data.fills.len()
                        );
                        self.snapshot_received.insert(coin.clone(), true);
                        return; // Don't process historical fills
                    }

                    // If we get here, this is a subsequent snapshot (unexpected)
                    warn!(
                        "[State Manager] Received unexpected second UserFills snapshot for {} ({} fills)",
                        coin,
                        user_fills.data.fills.len()
                    );
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

        let mut asset_states = self.asset_states.write().await;
        let _mgr = self.order_state_mgr.lock().await;
        let mut state_changed = false;

        for fill in fills {
            let fill_id = fill.tid;
            if self.processed_fill_ids.contains(&fill_id) {
                continue; // Skip duplicate
            }

            // Route fill to correct asset state
            let asset = fill.coin.clone();
            if let Some(asset_state) = asset_states.get_mut(&asset) {
                state_changed = true;
                self.process_fill_for_asset(asset_state, &fill);
                self.processed_fill_ids.insert(fill_id);

                info!(
                    "[State Manager] Processed fill for {}: {} {} @ ${}, new position: {}",
                    asset,
                    if fill.side == "B" { "BUY" } else { "SELL" },
                    fill.sz,
                    fill.px,
                    asset_state.position
                );
            } else {
                warn!(
                    "[State Manager] Received fill for unknown asset: {} (fill ID: {})",
                    asset, fill_id
                );
            }
        }

        state_changed
    }

    /// Handles an action request from a Strategy Runner.
    async fn handle_action_request(&self, request: ExecuteActionsRequest) {
        let asset = request.asset.clone();
        let action_count = request.actions.len();

        info!("[State Manager] Received {} action(s) from [{}]", action_count, asset);

        // Validate that we're tracking this asset
        {
            let lock_start = std::time::Instant::now();
            let asset_states = self.asset_states.read().await;
            let lock_duration = lock_start.elapsed();
            if lock_duration > Duration::from_millis(10) {
                debug!("[State Manager] Lock acquisition took {:.1}ms for asset validation", lock_duration.as_secs_f64() * 1000.0);
            }

            if !asset_states.contains_key(&asset) {
                warn!(
                    "[State Manager] ❌ Rejected actions from [{}]: Unknown asset",
                    asset
                );
                let _ = request.resp.send(ExecuteActionsResponse {
                    success: false,
                    message: format!("Unknown asset: {}", asset),
                });
                return;
            }
        }

        // ---
        // 1. Final Margin & Position Check (Atomicity)
        // ---
        let lock_start = std::time::Instant::now();
        let asset_states_read = self.asset_states.read().await;
        let global_state_read = self.global_account_state.read().await;
        let lock_duration = lock_start.elapsed();
        if lock_duration > Duration::from_millis(10) {
            warn!("[State Manager] ⚠️  Lock acquisition took {:.1}ms for validation (potential contention)", lock_duration.as_secs_f64() * 1000.0);
        }
        let asset_state = asset_states_read.get(&asset);

        info!("[State Manager] Validating {} actions for [{}]: Equity=${:.2}, Margin Used=${:.2}",
            action_count, asset, global_state_read.account_equity, global_state_read.margin_used);

        let (valid, msg) = self.validate_actions(&global_state_read, asset_state, &request.actions, &asset);

        if !valid {
            warn!(
                "[State Manager] ❌ VALIDATION FAILED for [{}]: {}",
                asset, msg
            );
            let _ = request.resp.send(ExecuteActionsResponse {
                success: false,
                message: msg,
            });
            return;
        }

        info!("[State Manager] ✅ Validation PASSED for [{}], proceeding with execution", asset);

        // Release read locks before acquiring write locks
        drop(asset_states_read);
        drop(global_state_read);

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
                    let mut asset_states = self.asset_states.write().await;
                    if let Some(asset_state) = asset_states.get_mut(&asset) {
                        if mgr.mark_pending_cancel(cancel.oid, &mut asset_state.open_bids)
                            || mgr.mark_pending_cancel(cancel.oid, &mut asset_state.open_asks)
                        {
                            executor_actions.push(ExecutorAction::Cancel(cancel));
                        }
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
                    let mut asset_states = self.asset_states.write().await;
                    if let Some(asset_state) = asset_states.get_mut(&asset) {
                        let mut valid_cancels = Vec::new();
                        for cancel in cancels {
                            if mgr.mark_pending_cancel(cancel.oid, &mut asset_state.open_bids)
                                || mgr.mark_pending_cancel(cancel.oid, &mut asset_state.open_asks)
                            {
                                valid_cancels.push(cancel);
                            }
                        }
                        if !valid_cancels.is_empty() {
                            executor_actions.push(ExecutorAction::BatchCancel(valid_cancels));
                        }
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
            info!("[State Manager] Spawning executor task for {} action(s)", executor_actions.len());
            let executor = self.order_executor.clone();
            let asset_clone = asset.clone();
            tokio::spawn(async move {
                info!("[Executor] Executing {} actions for [{}]...", executor_actions.len(), asset_clone);
                let result = executor.execute_actions_parallel(executor_actions).await;
                if result.failed > 0 {
                    error!(
                        "[Executor] ❌ {}/{} actions FAILED for [{}]",
                        result.failed,
                        result.failed + result.successful,
                        asset_clone
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
                } else {
                    info!("[Executor] ✅ All {} actions SUCCEEDED for [{}]", result.successful, asset_clone);
                }
            });
        } else {
            warn!("[State Manager] No executor actions to process for [{}]", asset);
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

    /// Validates actions against authoritative state to ensure position and margin limits.
    ///
    /// CRITICAL FIX: This version calculates the net effect of cancels AND new orders.
    /// If the batch is net-reducing (or neutral), it always passes validation,
    /// preventing the validation-rejection deadlock loop.
    fn validate_actions(
        &self,
        global_state: &GlobalAccountState,
        asset_state: Option<&AssetState>,
        actions: &[StrategyAction],
        asset_name: &str,
    ) -> (bool, String) {
        debug!(
            "[State Manager] Validating actions for {}: Current Margin Used: {:.2}, Equity: {:.2}",
            asset_name, global_state.margin_used, global_state.account_equity
        );

        let current_position = asset_state.map_or(0.0, |s| s.position);

        // Get the *actual* pending exposure from orders already on the exchange book
        let pending_exposure = asset_state.map_or(0.0, |s| {
            let bid_exposure: f64 = s.open_bids.iter().map(|o| o.size).sum();
            let ask_exposure: f64 = s.open_asks.iter().map(|o| o.size).sum();
            bid_exposure - ask_exposure
        });

        debug!(
            "[State Manager] Validation Check [{}]: Current Position: {:.4}, Pending Orders: {:.4}, Max Position: {:.4}",
            asset_name, current_position, pending_exposure, self.max_position_size
        );

        // --- STEP 1: Calculate exposure being removed by cancels ---
        let mut cancel_exposure = 0.0;
        if let Some(state) = asset_state {
            for action in actions {
                match action {
                    StrategyAction::Cancel(cancel) => {
                        // Find this order and calculate its contribution to exposure
                        if let Some(order) = state.open_bids.iter().find(|o| o.oid == Some(cancel.oid)) {
                            cancel_exposure += order.size; // Bids add positive exposure
                        } else if let Some(order) = state.open_asks.iter().find(|o| o.oid == Some(cancel.oid)) {
                            cancel_exposure -= order.size; // Asks add negative exposure
                        }
                    }
                    StrategyAction::BatchCancel(cancels) => {
                        for cancel in cancels {
                            if let Some(order) = state.open_bids.iter().find(|o| o.oid == Some(cancel.oid)) {
                                cancel_exposure += order.size;
                            } else if let Some(order) = state.open_asks.iter().find(|o| o.oid == Some(cancel.oid)) {
                                cancel_exposure -= order.size;
                            }
                        }
                    }
                    _ => {}
                }
            }
        }

        // --- STEP 2: Calculate exposure being added by new orders ---
        let mut net_size_change_from_new_orders = 0.0;
        let mut estimated_margin_increase = 0.0;

        // Start from current position + pending orders (AFTER cancels) to get true exposure
        let adjusted_pending_exposure = pending_exposure - cancel_exposure;
        let mut cumulative_pos_for_margin = current_position + adjusted_pending_exposure;

        for action in actions {
            match action {
                StrategyAction::Place(order) => {
                    let position_delta = if order.is_buy { order.sz } else { -order.sz };
                    net_size_change_from_new_orders += position_delta;

                    if !order.reduce_only {
                        let next_potential_pos = cumulative_pos_for_margin + position_delta;
                        if next_potential_pos.abs() > cumulative_pos_for_margin.abs() {
                            let size_increase = next_potential_pos.abs() - cumulative_pos_for_margin.abs();
                            estimated_margin_increase += self.margin_calculator.initial_margin_required(size_increase, order.limit_px);
                        }
                        cumulative_pos_for_margin = next_potential_pos;
                    } else {
                        cumulative_pos_for_margin += position_delta;
                    }
                }
                StrategyAction::BatchPlace(orders) => {
                    for order in orders {
                        let position_delta = if order.is_buy { order.sz } else { -order.sz };
                        net_size_change_from_new_orders += position_delta;

                        if !order.reduce_only {
                            let next_potential_pos = cumulative_pos_for_margin + position_delta;
                            if next_potential_pos.abs() > cumulative_pos_for_margin.abs() {
                                let size_increase = next_potential_pos.abs() - cumulative_pos_for_margin.abs();
                                estimated_margin_increase += self.margin_calculator.initial_margin_required(size_increase, order.limit_px);
                            }
                            cumulative_pos_for_margin = next_potential_pos;
                        } else {
                            cumulative_pos_for_margin += position_delta;
                        }
                    }
                }
                // Cancels already processed above
                StrategyAction::Cancel(_) | StrategyAction::BatchCancel(_) | StrategyAction::NoOp => {}
            }
        }

        // --- STEP 3: DEADLOCK FIX - Allow batches that reduce or maintain exposure ---
        // Calculate the net change in total exposure
        let current_total_exposure = (current_position + pending_exposure).abs();
        let final_total_exposure = (current_position + adjusted_pending_exposure + net_size_change_from_new_orders).abs();
        let net_exposure_change = final_total_exposure - current_total_exposure;

        debug!(
            "[State Manager] Exposure analysis [{}]: Current Total={:.4}, Cancel={:.4}, New={:.4}, Final Total={:.4}, Net Change={:.4}",
            asset_name, current_total_exposure, cancel_exposure.abs(), net_size_change_from_new_orders.abs(), final_total_exposure, net_exposure_change
        );

        // If this batch reduces or maintains total exposure, ALWAYS allow it (prevents deadlock)
        if net_exposure_change <= 1e-9 {
            info!(
                "[State Manager] ✅ Validation PASSED for [{}]: Batch is net-reducing/neutral (Change: {:.4})",
                asset_name, net_exposure_change
            );
            return (true, String::new());
        }

        // --- STEP 4: Position Check (for exposure-increasing batches only) ---
        let final_potential_position = current_position + adjusted_pending_exposure + net_size_change_from_new_orders;

        debug!(
            "[State Manager] Position limit check: Current={:.4}, Pending={:.4}, Cancel={:.4}, New={:.4}, Final={:.4}, Max={:.4}",
            current_position, pending_exposure, cancel_exposure, net_size_change_from_new_orders, final_potential_position, self.max_position_size
        );

        if final_potential_position.abs() > self.max_position_size + 1e-9 {
            let msg = format!(
                "Exceeds position limit (Current: {:.4}, Pending: {:.4}, Cancel: {:.4}, New: {:.4}, Final: {:.4}, Max: {:.4})",
                current_position, pending_exposure, cancel_exposure, net_size_change_from_new_orders, final_potential_position, self.max_position_size
            );
            return (false, msg);
        }

        // --- STEP 5: Margin Check (for exposure-increasing batches only) ---
        let potential_margin_used = global_state.margin_used + estimated_margin_increase;
        let max_allowed_margin = global_state.account_equity * (1.0 - self.safety_buffer);

        if potential_margin_used > max_allowed_margin + 1e-9 {
            let msg = format!(
                "Insufficient margin (Current Used: {:.2}, Est. Increase: {:.2}, Potential Used: {:.2}, Max Allowed: {:.2}, Equity: {:.2})",
                global_state.margin_used, estimated_margin_increase, potential_margin_used, max_allowed_margin, global_state.account_equity
            );
            return (false, msg);
        }

        info!(
            "[State Manager] ✅ Validation PASSED for {}: Current={:.4}, Pending={:.4}, Cancel={:.4}, New={:.4}, Final={:.4}, Margin Increase={:.2}",
            asset_name, current_position, pending_exposure, cancel_exposure, net_size_change_from_new_orders, final_potential_position, estimated_margin_increase
        );
        (true, String::new())
    }

    /// Broadcasts the current authoritative state to all runners.
    /// Sends one update per asset so runners can filter by their asset.
    async fn broadcast_state(&self) {
        let lock_start = std::time::Instant::now();
        let asset_states = self.asset_states.read().await;
        let global_state = self.global_account_state.read().await;
        let lock_duration = lock_start.elapsed();

        if lock_duration > Duration::from_millis(10) {
            warn!("[State Manager] ⚠️  broadcast_state: Lock acquisition took {:.1}ms (potential contention)",
                lock_duration.as_secs_f64() * 1000.0);
        }

        // Broadcast state update for each asset
        for (asset, asset_state) in asset_states.iter() {
            let update = AuthoritativeStateUpdate::from_asset_and_global(
                asset_state,
                &global_state,
            );

            // .send() returns the number of *active* subscribers.
            // If 0, no one is listening. If Err, the channel is closed.
            match self.state_tx.send(update) {
                Ok(n) if n > 0 => {
                    debug!("[State Manager] Broadcasted {} state to {} runners", asset, n);
                }
                Ok(_) => {
                    debug!("[State Manager] No active subscribers for {}", asset);
                }
                Err(_) => {
                    debug!("[State Manager] Broadcast channel closed for {}", asset);
                }
            }
        }
    }

    // --- State Mutation Helpers (Per-Asset) ---

    fn add_order_to_asset_state(&self, asset_state: &mut AssetState, order: RestingOrder) {
        let orders = if order.is_buy {
            &mut asset_state.open_bids
        } else {
            &mut asset_state.open_asks
        };
        if let Some(existing) = orders.iter_mut().find(|o| o.oid == order.oid) {
            *existing = order;
        } else {
            orders.push(order);
        }
        asset_state.open_bids.sort_by(|a, b| b.price.partial_cmp(&a.price).unwrap_or(std::cmp::Ordering::Equal));
        asset_state.open_asks.sort_by(|a, b| a.price.partial_cmp(&b.price).unwrap_or(std::cmp::Ordering::Equal));
        asset_state.timestamp = chrono::Utc::now().timestamp() as f64;
    }

    fn update_partial_fill_in_asset_state(&self, asset_state: &mut AssetState, order: RestingOrder) {
        let orders = if order.is_buy {
            &mut asset_state.open_bids
        } else {
            &mut asset_state.open_asks
        };
        if let Some(existing) = orders.iter_mut().find(|o| o.oid == order.oid) {
            existing.size = order.size;
            existing.state = order.state;
            existing.timestamp = order.timestamp;
        }
        asset_state.timestamp = chrono::Utc::now().timestamp() as f64;
    }

    fn remove_order_from_asset_state(&self, asset_state: &mut AssetState, oid: u64) {
        asset_state.open_bids.retain(|o| o.oid != Some(oid));
        asset_state.open_asks.retain(|o| o.oid != Some(oid));
        asset_state.timestamp = chrono::Utc::now().timestamp() as f64;
    }

    fn process_fill_for_asset(&self, asset_state: &mut AssetState, fill: &TradeInfo) {
        let fill_size = fill.sz.parse::<f64>().unwrap_or(0.0);
        let fill_price = fill.px.parse::<f64>().unwrap_or(0.0);
        let fill_fee = fill.fee.parse::<f64>().unwrap_or(0.0);
        let is_buy = fill.side == "B";

        asset_state.total_fees += fill_fee.abs();
        let signed_size = if is_buy { fill_size } else { -fill_size };
        let old_position = asset_state.position;
        let new_position = old_position + signed_size;

        if old_position.abs() < 1e-6 {
            asset_state.cost_basis = fill_size * fill_price;
            asset_state.avg_entry_price = fill_price;
        } else if (old_position > 0.0 && new_position > old_position)
            || (old_position < 0.0 && new_position < old_position)
        {
            let old_cost = asset_state.cost_basis;
            let new_cost = old_cost + (fill_size * fill_price);
            asset_state.cost_basis = new_cost;
            asset_state.avg_entry_price = new_cost / new_position.abs();
        } else {
            let reduced_size = fill_size.min(old_position.abs());
            let realized_pnl = if old_position > 0.0 {
                reduced_size * (fill_price - asset_state.avg_entry_price)
            } else {
                reduced_size * (asset_state.avg_entry_price - fill_price)
            };
            asset_state.realized_pnl += realized_pnl;

            if new_position.abs() < 1e-6 {
                asset_state.cost_basis = 0.0;
                asset_state.avg_entry_price = 0.0;
            } else {
                asset_state.cost_basis = new_position.abs() * fill_price;
                asset_state.avg_entry_price = fill_price;
            }
        }

        asset_state.position = new_position;
        asset_state.timestamp = chrono::Utc::now().timestamp() as f64;
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

        // Watchdog timer to detect stalled event loop
        let mut watchdog_timer = interval(Duration::from_secs(30));
        watchdog_timer.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

        // Message timeout detection
        let mut last_market_message_time = std::time::Instant::now();
        let ws_timeout = Duration::from_secs(60); // Alert if no WS messages for 60 seconds

        let mut healthcheck_timer = interval(Duration::from_secs(15));
        healthcheck_timer.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

        // 🔍 DEADLOCK DETECTION: Track loop iterations
        let mut loop_iterations = 0u64;
        let mut last_iteration_log = std::time::Instant::now();

        loop {
            let loop_start = std::time::Instant::now();
            loop_iterations += 1;

            // Log iteration count every 10 seconds
            if last_iteration_log.elapsed() >= Duration::from_secs(10) {
                info!("[Runner {}] 🔄 Event loop alive: {} iterations in last {:.1}s",
                    self.asset, loop_iterations, last_iteration_log.elapsed().as_secs_f64());
                loop_iterations = 0;
                last_iteration_log = std::time::Instant::now();
            }

            // CRITICAL FIX: Use biased selection to prioritize message processing over timers
            tokio::select! {
                biased;

                // HIGHEST PRIORITY: Handle incoming *authoritative state* from State Manager
                Ok(update) = self.state_rx.recv() => {
                    debug!("[Runner {}] Received state update (iteration #{})", self.asset, loop_iterations);
                    self.handle_state_update(update).await;
                }

                // HIGH PRIORITY: Handle incoming *market data* (L2Book, Trades)
                Some(message) = market_ws_rx.recv() => {
                    debug!("[Runner {}] Received market data message (iteration #{})", self.asset, loop_iterations);
                    last_market_message_time = std::time::Instant::now();
                    self.handle_market_message(message).await;
                }

                // MEDIUM PRIORITY: Handle periodic tick for strategy logic
                _ = tick_timer.tick() => {
                    debug!("[Runner {}] Periodic tick (iteration #{})", self.asset, loop_iterations);
                    self.handle_tick().await;
                }

                // LOW PRIORITY: Timeout detection
                _ = tokio::time::sleep(Duration::from_secs(2)) => {
                    warn!("[Runner {}] ⏰ SELECT TIMEOUT: No events processed for 2 seconds!", self.asset);
                    warn!("[Runner {}]   -> This may indicate a deadlock or event starvation", self.asset);
                }

                // LOW PRIORITY: Watchdog timer
                _ = watchdog_timer.tick() => {
                    info!("[Runner {}] Watchdog: Event loop is running normally (iteration #{})", self.asset, loop_iterations);
                }

                // LOW PRIORITY: Health check timer
                _ = healthcheck_timer.tick() => {
                    let elapsed = last_market_message_time.elapsed();
                    if elapsed > ws_timeout {
                        warn!("[Runner {}] ⚠️  No market data messages received for {:.1}s - connection may be stalled",
                            self.asset, elapsed.as_secs_f64());
                    } else {
                        debug!("[Runner {}] Health check: Last market message {:.1}s ago", self.asset, elapsed.as_secs_f64());
                    }
                }

                // CRITICAL: Shutdown signal
                 _ = tokio::signal::ctrl_c() => {
                    info!("[Runner {}] Shutdown signal received.", self.asset);
                    self.handle_shutdown().await;
                    break;
                }
            }

            let iteration_duration = loop_start.elapsed();
            if iteration_duration > Duration::from_millis(100) {
                warn!("[Runner {}] ⚠️  Slow iteration: took {:.3}s", self.asset, iteration_duration.as_secs_f64());
            } else {
                debug!("[Runner {}] Loop iteration #{} took {:.3}ms", self.asset, loop_iterations, iteration_duration.as_millis());
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
        info!("[Runner {}] Creating flume channel for market data subscriptions...", self.asset);
        let (flume_tx, flume_rx) = flume::unbounded();

        info!("[Runner {}] Subscribing to L2Book...", self.asset);
        self.info_client
            .subscribe(
                Subscription::L2Book {
                    coin: self.asset.clone(),
                },
                flume_tx.clone(),
            )
            .await?;
        info!("[Runner {}] ✓ L2Book subscription successful", self.asset);

        info!("[Runner {}] Subscribing to Trades...", self.asset);
        self.info_client
            .subscribe(
                Subscription::Trades {
                    coin: self.asset.clone(),
                },
                flume_tx.clone(),
            )
            .await?;
        info!("[Runner {}] ✓ Trades subscription successful", self.asset);

        info!("[Runner {}] Subscribing to AllMids...", self.asset);
        self.info_client
            .subscribe(Subscription::AllMids, flume_tx.clone())
            .await?;
        info!("[Runner {}] ✓ AllMids subscription successful", self.asset);

        // Spawn a forwarder task with diagnostic logging
        let asset_for_task = self.asset.clone();
        info!("[Runner {}] Spawning market data forwarder task...", self.asset);
        tokio::spawn(async move {
            info!("[Runner {} Market Forwarder] Started - waiting for messages from WebSocket...", asset_for_task);
            let mut message_count = 0u64;
            while let Ok(msg) = flume_rx.recv_async().await {
                message_count += 1;
                if message_count == 1 {
                    info!("[Runner {} Market Forwarder] 🎉 First message received from WebSocket!", asset_for_task);
                } else if message_count % 500 == 0 {
                    debug!("[Runner {} Market Forwarder] Forwarded {} messages", asset_for_task, message_count);
                }

                // CRITICAL: Explicitly check if channel send succeeds
                match sender.send(msg) {
                    Ok(_) => {
                        if message_count <= 5 || message_count % 500 == 0 {
                            debug!("[Runner {} Market Forwarder] ✓ Sent message #{} to main loop", asset_for_task, message_count);
                        }
                    }
                    Err(e) => {
                        error!("[Runner {} Market Forwarder] ❌ CHANNEL BROKEN: Failed to send message #{} - {:?}", asset_for_task, message_count, e);
                        error!("[Runner {} Market Forwarder] Main loop receiver has been dropped! Exiting forwarder.", asset_for_task);
                        break;
                    }
                }
            }
            warn!("[Runner {} Market Forwarder] Exiting - flume channel closed. Total messages forwarded: {}", asset_for_task, message_count);
        });

        info!("[Runner {}] Market data forwarder task spawned successfully", self.asset);
        Ok(())
    }

    /// Processes a market data message, runs strategy, and sends actions.
    async fn handle_market_message(&mut self, message: Message) {
        let mut market_update: Option<MarketUpdate> = None;

        match message {
            Message::L2Book(l2_book) => {
                if let Some(book) = OrderBook::from_l2_data(&l2_book.data) {
                    let mut state = self.local_state_cache.write().await;
                    let old_mid = state.l2_mid_price;
                    if let Some(analysis) = book.analyze(5) {
                        state.lob_imbalance = analysis.imbalance;
                        state.market_spread_bps = book.spread_bps().unwrap_or(0.0);
                    }
                    if let (Some(bid), Some(ask)) = (book.best_bid(), book.best_ask()) {
                        let new_mid = (bid + ask) / 2.0;

                        // Log first L2Book or if mid price changed significantly
                        if old_mid == 0.0 || (new_mid - old_mid).abs() / old_mid > 0.001 {
                            info!("[Runner {}] L2Book update: bid=${:.3}, ask=${:.3}, mid=${:.3}, imbalance={:.3}",
                                self.asset, bid, ask, new_mid, state.lob_imbalance);
                        } else {
                            debug!("[Runner {}] L2Book: bid=${:.3}, ask=${:.3}, mid=${:.3}, imbalance={:.3}",
                                self.asset, bid, ask, new_mid, state.lob_imbalance);
                        }
                        state.l2_mid_price = new_mid;
                    }
                    state.order_book = Some(book);
                    market_update = Some(MarketUpdate::from_l2_book(l2_book.data));
                } else {
                    warn!("[Runner {}] Failed to parse L2Book data", self.asset);
                }
            }
            Message::Trades(trades) => {
                debug!("[Runner {}] Received {} trade(s)", self.asset, trades.data.len());
                market_update =
                    Some(MarketUpdate::from_trades(self.asset.clone(), trades.data));
            }
            Message::AllMids(all_mids) => {
                if let Some(mid_str) = all_mids.data.mids.get(&self.asset) {
                    if let Ok(mid_price) = mid_str.parse::<f64>() {
                        let mut state = self.local_state_cache.write().await;
                        let old_mid = state.l2_mid_price;

                        // Log first AllMids or if mid price changed significantly
                        if old_mid == 0.0 || (mid_price - old_mid).abs() / old_mid > 0.001 {
                            info!("[Runner {}] AllMids update: mid=${:.3} (was ${:.3})",
                                self.asset, mid_price, old_mid);
                        } else {
                            debug!("[Runner {}] Received AllMids: mid=${:.3}", self.asset, mid_price);
                        }

                        state.l2_mid_price = mid_price;
                        drop(state);

                        market_update =
                            Some(MarketUpdate::from_mid_price(self.asset.clone(), mid_price));
                    } else {
                        warn!("[Runner {}] Failed to parse mid price from AllMids: {}", self.asset, mid_str);
                    }
                } else {
                    debug!("[Runner {}] AllMids does not contain price for this asset", self.asset);
                }
            }
            _ => {
                debug!("[Runner {}] Received unhandled message type", self.asset);
            }
        }

        // If we have a valid update, run the strategy
        if let Some(update) = market_update {
            debug!("[Runner {}] Processing market update, running strategy...", self.asset);
            let state = self.local_state_cache.read().await;
            let actions = self.strategy.on_market_update(&state, &update);

            if !actions.is_empty() && !matches!(actions[0], StrategyAction::NoOp) {
                info!("[Runner {}] Strategy generated {} action(s)", self.asset, actions.len());
                self.send_actions(actions).await;
            } else {
                debug!("[Runner {}] Strategy returned NoOp or empty actions", self.asset);
            }
        } else {
            debug!("[Runner {}] No valid market update generated from message", self.asset);
        }
    }

    /// Processes an authoritative state update from the State Manager.
    async fn handle_state_update(&mut self, update: AuthoritativeStateUpdate) {
        // Filter: Only process updates for OUR asset
        if update.asset != self.asset {
            return; // Ignore updates for other assets
        }

        let is_initial_state = {
            let state = self.local_state_cache.read().await;
            state.account_equity == 0.0 && update.account_equity > 0.0
        };

        let mut state = self.local_state_cache.write().await;

        // Log first meaningful state update (when we get non-zero equity)
        if is_initial_state {
            info!(
                "[Runner {}] Received initial state: equity=${:.2}, margin_used=${:.2}, pos={}",
                self.asset, update.account_equity, update.margin_used, update.position
            );
        } else {
            debug!("[Runner {}] State update: pos={}, equity=${:.2}",
                self.asset, update.position, update.account_equity);
        }

        // Update local state with asset-specific data from State Manager
        state.position = update.position;
        state.avg_entry_price = update.avg_entry_price;
        state.realized_pnl = update.realized_pnl;
        state.unrealized_pnl = update.unrealized_pnl;

        // Update global account data (shared across all assets)
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

        // If this is the initial state update and we have a valid mid price,
        // trigger the strategy to place initial orders
        if is_initial_state && state.l2_mid_price > 0.0 {
            info!(
                "[Runner {}] Triggering initial order placement (mid: ${:.3})",
                self.asset, state.l2_mid_price
            );
            drop(state); // Release write lock before calling strategy

            let state_read = self.local_state_cache.read().await;
            let actions = self.strategy.on_tick(&state_read);
            drop(state_read);

            if !actions.is_empty() && !matches!(actions[0], StrategyAction::NoOp) {
                self.send_actions(actions).await;
            }
        }
    }

    /// Handles the periodic 1-second tick.
    async fn handle_tick(&mut self) {
        debug!("[Runner {}] on_tick called", self.asset);

        // --- Step 1: Acquire WRITE lock for atomic check and update ---
        // We acquire the write lock once to check the price and update state
        // in a single atomic operation.
        let mut state_write = self.local_state_cache.write().await;

        if state_write.l2_mid_price <= 0.0 {
            warn!("[Runner {}] Waiting for valid mid price (current: ${:.3})",
                self.asset, state_write.l2_mid_price);
            // Return, which automatically releases the write lock
            return;
        }

        // Update timestamp and unrealized PnL
        state_write.timestamp = chrono::Utc::now().timestamp() as f64;
        if state_write.position.abs() > 1e-6 {
            if state_write.position > 0.0 { // Long
                state_write.unrealized_pnl =
                    state_write.position * (state_write.l2_mid_price - state_write.avg_entry_price);
            } else { // Short
                state_write.unrealized_pnl =
                    state_write.position.abs() * (state_write.avg_entry_price - state_write.l2_mid_price);
            }
        }

        // Copy out values needed for logging after dropping the lock
        let mid_price = state_write.l2_mid_price;
        let position = state_write.position;
        let has_orders = !state_write.open_bids.is_empty() || !state_write.open_asks.is_empty();
        let num_bids = state_write.open_bids.len();
        let num_asks = state_write.open_asks.len();

        // --- Step 2: Drop WRITE lock *before* running strategy ---
        drop(state_write);

        debug!("[Runner {}] Tick: mid=${:.3}, pos={:.2}, has_orders={}",
            self.asset, mid_price, position, has_orders);

        // --- Step 3: Acquire READ lock to run the strategy ---
        // The state is now guaranteed to be updated from Step 1.
        let state_read = self.local_state_cache.read().await;
        let actions = self.strategy.on_tick(&state_read);
        drop(state_read); // Drop READ lock

        // --- Step 4: Send Actions (no lock held) ---
        if !actions.is_empty() && !matches!(actions[0], StrategyAction::NoOp) {
            info!("[Runner {}] Tick generated {} action(s)", self.asset, actions.len());
            self.send_actions(actions).await;
        } else {
            // Log diagnostic info when no actions are taken
            if mid_price <= 0.0 {
                warn!("[Runner {}] Tick: No actions - invalid mid price ${:.3}", self.asset, mid_price);
            } else if !has_orders {
                warn!("[Runner {}] Tick: No actions despite no orders (mid=${:.3}, pos={:.2})",
                    self.asset, mid_price, position);
            } else {
                debug!("[Runner {}] Tick: NoOp with {} bids, {} asks (mid=${:.3})",
                    self.asset, num_bids, num_asks, mid_price);
            }
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
    /// CRITICAL FIX: This method no longer blocks waiting for a response.
    /// Response handling is done in a separate task to avoid blocking the event loop.
    /// If actions are rejected, clears the local order cache to force state reconciliation.
    async fn send_actions(&self, actions: Vec<StrategyAction>) {
        let place_count = actions.iter().filter(|a| matches!(a, StrategyAction::Place(_) | StrategyAction::BatchPlace(_))).count();
        let cancel_count = actions.iter().filter(|a| matches!(a, StrategyAction::Cancel(_) | StrategyAction::BatchCancel(_))).count();

        info!("[Runner {}] Sending {} action(s) to State Manager: {} place, {} cancel",
            self.asset, actions.len(), place_count, cancel_count);

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

        // CRITICAL FIX: Spawn a separate task to handle the response
        // This prevents blocking the main event loop
        let asset_clone = self.asset.clone();
        let state_cache_clone = self.local_state_cache.clone();
        tokio::spawn(async move {
            match resp_rx.await {
                Ok(response) => {
                    if !response.success {
                        warn!(
                            "[Runner {}] ❌ Actions REJECTED by State Manager: {}",
                            asset_clone, response.message
                        );

                        // SECONDARY FIX: Clear local order cache to force reconciliation
                        // This prevents the runner from getting stuck with stale order state
                        let mut state = state_cache_clone.write().await;
                        let old_bid_count = state.open_bids.len();
                        let old_ask_count = state.open_asks.len();
                        state.open_bids.clear();
                        state.open_asks.clear();
                        drop(state);

                        warn!(
                            "[Runner {}] 🔄 Cleared local order cache ({} bids, {} asks) - waiting for State Manager reconciliation",
                            asset_clone, old_bid_count, old_ask_count
                        );
                    } else {
                        info!(
                            "[Runner {}] ✅ Actions accepted by State Manager: {}",
                            asset_clone, response.message
                        );
                    }
                }
                Err(_) => {
                    error!("[Runner {}] Did not receive response from State Manager.", asset_clone);
                }
            }
        });

        // Log that we've dispatched the request without waiting
        debug!("[Runner {}] Actions dispatched (non-blocking)", self.asset);
    }
}

// ============================================================================
// Main Launcher
// ============================================================================

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // --- 1. Initialization ---
    let file_appender = tracing_appender::rolling::never("./", "market_maker_v3.log");
    // Keep the guard alive for the entire program lifetime to maintain the logging worker thread
    let (non_blocking_writer, _log_guard) = tracing_appender::non_blocking(file_appender);
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

    // Extract list of unique assets from strategy configs
    let assets: Vec<String> = app_config
        .strategies
        .iter()
        .map(|s| s.asset.clone())
        .collect::<std::collections::HashSet<_>>() // Deduplicate
        .into_iter()
        .collect();

    info!("Trading {} unique assets: {:?}", assets.len(), assets);

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
        StateManagerActor::new(wallet.clone(), assets, action_rx, state_tx.clone(), &app_config).await?;

    let state_manager_handle = local_set.spawn_local(async move {
        info!("🚀 [State Manager] Task starting...");
        match state_manager.run().await {
            Ok(_) => {
                warn!("⚠️  [State Manager] Actor exited normally (unexpected unless shutting down)");
            }
            Err(e) => {
                error!("❌ [State Manager] Actor failed with error: {}", e);
                error!("   Backtrace: {:?}", std::backtrace::Backtrace::capture());
            }
        }
        error!("🔴 [State Manager] Task has COMPLETED - this should only happen during shutdown!");
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

        let asset_for_handle = asset.clone();
        let handle = local_set.spawn_local(async move {
            info!("🚀 [Runner {}] Task starting...", asset_for_handle);
            match runner.run().await {
                Ok(_) => {
                    warn!("⚠️  [Runner {}] Actor exited normally (unexpected unless shutting down)", asset_for_handle);
                }
                Err(e) => {
                    error!("❌ [Runner {}] Actor failed with error: {}", asset_for_handle, e);
                    error!("   Backtrace: {:?}", std::backtrace::Backtrace::capture());
                }
            }
            error!("🔴 [Runner {}] Task has COMPLETED - this should only happen during shutdown!", asset_for_handle);
        });
        runner_handles.push(handle);
    }

    info!("✅ All {} Strategy Runner Actors spawned.", runner_handles.len());

    // --- 6. Spawn Global Watchdog (runs outside LocalSet) ---
    // This watchdog runs on the regular tokio runtime and can detect if the LocalSet is blocked
    let watchdog_handle = tokio::spawn(async {
        let mut watchdog_timer = interval(Duration::from_secs(45));
        watchdog_timer.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

        loop {
            watchdog_timer.tick().await;
            info!("🐕 [Global Watchdog] Tokio runtime is alive and processing tasks");
        }
    });

    // --- 7. Run LocalSet with Timeout and Shutdown Handler ---
    info!("🚀 Starting market maker event loop...");

    local_set
        .run_until(async move {
            // Create a shutdown signal channel
            let (shutdown_tx, mut shutdown_rx) = tokio::sync::mpsc::channel::<()>(1);

            // Spawn a task to listen for Ctrl+C
            tokio::spawn(async move {
                tokio::signal::ctrl_c().await.expect("Failed to wait for Ctrl+C");
                info!("🛑 Ctrl+C received, initiating shutdown...");
                let _ = shutdown_tx.send(()).await;
            });

            // Wait for shutdown signal
            let _ = shutdown_rx.recv().await;
            info!("Main task shutting down. Actors will terminate.");

            // Give actors time to clean up (with timeout)
            info!("Waiting for State Manager to finish...");
            match tokio::time::timeout(Duration::from_secs(5), state_manager_handle).await {
                Ok(_) => info!("State Manager shutdown complete"),
                Err(_) => warn!("State Manager shutdown timed out"),
            }

            info!("Waiting for {} Strategy Runners to finish...", runner_handles.len());
            for (i, handle) in runner_handles.into_iter().enumerate() {
                match tokio::time::timeout(Duration::from_secs(3), handle).await {
                    Ok(_) => debug!("Runner {} shutdown complete", i),
                    Err(_) => warn!("Runner {} shutdown timed out", i),
                }
            }

            info!("All actors have been shut down");
        })
        .await;

    // Cancel the watchdog
    watchdog_handle.abort();

    info!("✅ Market maker shutdown complete");
    Ok(())
}
