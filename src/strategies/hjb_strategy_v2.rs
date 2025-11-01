// ============================================================================
// HJB Strategy V2 - Modular Event-Driven Architecture
// ============================================================================
//
// This is a completely refactored version of the HJB strategy that uses a
// modular, event-driven architecture. Key improvements:
//
// 1. **Single Source of Truth**: TradingStateStore holds all state
// 2. **Event-Driven**: Components communicate via EventBus
// 3. **Proper Order Lifecycle**: OrderStateMachine tracks order states
// 4. **Clear Separation**: Signal → Risk → Execution pipeline
// 5. **Position Safety**: PositionManager enforces limits consistently
//
// # Architecture
//
// ```
// MarketUpdate → MarketDataPipeline → TradingStateStore
//                                            ↓
//      ┌─────────────────────────────────────┴──────────────────────────┐
//      ↓                                                                  ↓
// HjbSignalGenerator (pure HJB optimization)                    EventBus
//      ↓                                                                  ↓
// RiskAdjuster (apply position/margin limits)              Subscribers (logging, metrics)
//      ↓
// OrderExecutor (reconcile & execute orders)
//      ↓
// StrategyActions (place/cancel orders)
// ```
//
// # Key Benefits
//
// - Position limits enforced in ONE place (PositionManager)
// - Order states tracked properly (no stuck PendingCancel)
// - Easy to test individual components
// - Easy to swap implementations (e.g., different signal generators)
// - Full observability via event bus

use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use parking_lot::RwLock;
use log::{debug, info, warn};
use serde_json::Value;

use crate::strategy::{CurrentState, MarketUpdate, Strategy, StrategyAction, UserUpdate};
use crate::{
    AssetType, ClientCancelRequest, HawkesFillModel,
};

// Import modular components
use crate::strategies::components::{
    // Event bus
    EventBus, TradingEvent, LoggingSubscriber, MetricsSubscriber,

    // State management
    TradingStateStore, TradingSnapshot, OpenOrder,

    // Signal generation
    HjbSignalGenerator,

    // Risk management
    RiskAdjuster, MarginCalculator,
    PositionManager, PositionManagerConfig, PositionState,

    // Order execution
    OrderExecutor,

    // Market data processing
    MarketDataPipeline, ProcessedMarketData,

    // Order state machine
    OrderStateMachine, StateMachineConfig,

    // Volatility and optimizer
    VolatilityModel, EwmaVolatilityModel, EwmaVolConfig,
    QuoteOptimizer, HjbMultiLevelOptimizer,
    MicropriceAsModel,
};

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for HJB Strategy V2
#[derive(Debug, Clone)]
pub struct HjbStrategyV2Config {
    /// Maximum absolute position size
    pub max_position: f64,

    /// HJB inventory aversion parameter (φ)
    pub phi: f64,

    /// Base Poisson fill rate (λ)
    pub lambda_base: f64,

    /// Maker fee in basis points
    pub maker_fee_bps: f64,

    /// Taker fee in basis points
    pub taker_fee_bps: f64,

    /// Account leverage
    pub leverage: usize,

    /// Margin safety buffer (0.0-1.0)
    pub margin_safety_buffer: f64,

    /// Requote threshold in basis points
    pub requote_threshold_bps: f64,

    /// Asset being traded
    pub asset: String,

    /// Asset type (for tick/lot validation)
    pub asset_type: AssetType,

    /// Tick size
    pub tick_size: f64,

    /// Lot size
    pub lot_size: f64,

    /// Position manager config
    pub position_manager_config: PositionManagerConfig,

    /// EWMA volatility config
    pub ewma_vol_config: EwmaVolConfig,

    /// Order state machine config
    pub state_machine_config: StateMachineConfig,

    /// Auto-tuning configuration (None if disabled)
    pub tuner_config: Option<crate::strategies::components::TunerConfig>,

    /// Number of market updates per tuning episode
    pub tuner_updates_per_episode: usize,
}

impl HjbStrategyV2Config {
    /// Load configuration from JSON
    pub fn from_json(asset: &str, config: &Value) -> Self {
        let params = &config["strategy_params"];

        let max_position = params["max_absolute_position_size"]
            .as_f64()
            .unwrap_or(50.0);

        // Position manager configuration
        let position_manager_config = PositionManagerConfig {
            warning_threshold: 0.7,
            critical_threshold: 0.85,
            warning_reduction_factor: 0.5,
            critical_reduction_percentage: 0.3,
        };

        // EWMA volatility configuration
        let ewma_vol_config = if let Some(vol_config) = params.get("ewma_vol_config") {
            serde_json::from_value(vol_config.clone()).unwrap_or_default()
        } else {
            EwmaVolConfig::default()
        };

        // Order state machine configuration
        let state_machine_config = StateMachineConfig::default();

        // Auto-tuning configuration
        let (tuner_config, tuner_updates_per_episode) = if let Some(auto_tuning) = params.get("auto_tuning") {
            let enabled = auto_tuning.get("enabled")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);

            if enabled {
                // Parse tuner config from JSON or use defaults
                let tuner_cfg = if let Some(tuner_params) = params.get("auto_tuning_config") {
                    serde_json::from_value(tuner_params.clone()).unwrap_or_default()
                } else {
                    crate::strategies::components::TunerConfig::default()
                };

                let updates_per_episode = auto_tuning.get("updates_per_episode")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(100) as usize;

                (Some(tuner_cfg), updates_per_episode)
            } else {
                (None, 100)
            }
        } else {
            (None, 100)
        };

        Self {
            max_position,
            phi: params.get("phi").and_then(|v| v.as_f64()).unwrap_or(0.01),
            lambda_base: params.get("lambda_base").and_then(|v| v.as_f64()).unwrap_or(1.0),
            maker_fee_bps: params.get("maker_fee_bps").and_then(|v| v.as_f64()).unwrap_or(1.5),
            taker_fee_bps: params.get("taker_fee_bps").and_then(|v| v.as_f64()).unwrap_or(4.5),
            leverage: params.get("leverage").and_then(|v| v.as_u64()).unwrap_or(3) as usize,
            margin_safety_buffer: params.get("margin_safety_buffer")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.2),
            requote_threshold_bps: params.get("requote_threshold_bps")
                .and_then(|v| v.as_f64())
                .unwrap_or(5.0),
            asset: asset.to_string(),
            asset_type: AssetType::Perp, // Default to Perp
            tick_size: 0.01, // Default, should be set from exchange metadata
            lot_size: 0.001, // Default, should be set from exchange metadata
            position_manager_config,
            ewma_vol_config,
            state_machine_config,
            tuner_config,
            tuner_updates_per_episode,
        }
    }
}

// ============================================================================
// HJB Strategy V2
// ============================================================================

/// Modular HJB Strategy with event-driven architecture
pub struct HjbStrategyV2 {
    // Configuration
    config: HjbStrategyV2Config,

    // Core infrastructure
    event_bus: Arc<EventBus>,
    state_store: Arc<TradingStateStore>,

    // Components
    market_pipeline: MarketDataPipeline,
    signal_generator: HjbSignalGenerator,
    risk_adjuster: RiskAdjuster,
    position_manager: PositionManager,
    order_executor: OrderExecutor,

    // Metrics
    metrics_subscriber: Arc<MetricsSubscriber>,

    // Auto-tuner actor (None if disabled)
    tuner_actor: Option<crate::strategies::async_tuner_actor::AsyncTunerActorHandle>,

    // Last quote timestamp (to avoid over-quoting)
    last_quote_time: f64,

    // Episode tracking for tuner
    market_updates_in_episode: usize,
}

impl HjbStrategyV2 {
    /// Create a new HJB Strategy V2
    pub fn new(asset: &str, config: &Value) -> Self {
        let config = HjbStrategyV2Config::from_json(asset, config);

        info!("[HJB V2] Initializing modular strategy for asset: {}", asset);
        info!("[HJB V2] Max position: {:.2}, Phi: {:.4}, Lambda: {:.2}",
            config.max_position, config.phi, config.lambda_base);

        // Initialize event bus
        let event_bus = Arc::new(EventBus::with_history_size(2000));

        // Add logging subscriber
        let logging_subscriber = Arc::new(LoggingSubscriber::new(
            format!("HJB_V2_{}", asset)
        ));
        event_bus.subscribe(logging_subscriber);

        // Add metrics subscriber
        let metrics_subscriber = Arc::new(MetricsSubscriber::new(
            format!("Metrics_{}", asset)
        ));
        event_bus.subscribe(metrics_subscriber.clone());

        // Initialize state store
        let state_store = Arc::new(TradingStateStore::new(event_bus.clone()));

        // Initialize market data pipeline with processors
        use crate::strategies::components::{
            RobustConfig, InventorySkewConfig,
            VolatilityProcessor, ImbalanceProcessor, AdverseSelectionProcessor
        };

        let mut market_pipeline = MarketDataPipeline::new();

        // Create volatility model for the pipeline
        let pipeline_vol_model: Box<dyn VolatilityModel> = Box::new(
            EwmaVolatilityModel::new(config.ewma_vol_config.clone())
        );

        // Add processors to pipeline
        market_pipeline.add_processor(Box::new(VolatilityProcessor::new(pipeline_vol_model)));
        market_pipeline.add_processor(Box::new(ImbalanceProcessor::new()));
        market_pipeline.add_processor(Box::new(AdverseSelectionProcessor::new()));

        // Initialize signal generator
        use crate::MultiLevelConfig;
        use super::components::parameter_transforms::StrategyTuningParams;

        // Create separate volatility model for signal generator (for getter methods)
        let volatility_model: Box<dyn VolatilityModel> = Box::new(
            EwmaVolatilityModel::new(config.ewma_vol_config.clone())
        );

        let quote_optimizer: Box<dyn QuoteOptimizer + Send + Sync> = Box::new(
            HjbMultiLevelOptimizer::new(
                MultiLevelConfig::default(),
                RobustConfig::default(),
                InventorySkewConfig::default(),
                config.asset.clone(),
                config.max_position,
                StrategyTuningParams::default().get_constrained(),
            )
        );

        let adverse_selection_model = MicropriceAsModel::with_params(0.15, 8.0);

        // Create Hawkes fill model
        let hawkes_model = Arc::new(RwLock::new(
            HawkesFillModel::new(3) // 3 levels
        ));

        let signal_generator = HjbSignalGenerator::new(
            quote_optimizer,
            volatility_model,
            adverse_selection_model,
            hawkes_model,
            config.lambda_base,
            config.phi,
            config.maker_fee_bps,
            config.taker_fee_bps,
        );

        // Initialize position manager
        let position_manager = PositionManager::new(
            config.max_position,
            config.position_manager_config.clone()
        );

        // Initialize risk adjuster
        let margin_calculator = MarginCalculator::new(
            config.leverage,
            config.margin_safety_buffer,
        );

        // Clone is not needed - just pass position_manager by value after we're done using it
        let risk_adjuster = {
            let pm_for_adjuster = PositionManager::new(
                config.max_position,
                config.position_manager_config.clone()
            );
            RiskAdjuster::new(
                pm_for_adjuster,
                margin_calculator,
                0.001, // min_order_size
            )
        };

        // Initialize order executor
        let order_executor = OrderExecutor::new(
            state_store.clone(),
            event_bus.clone(),
            config.asset.clone(),
            config.tick_size,
            config.lot_size,
            config.requote_threshold_bps,
        );

        // Initialize auto-tuner actor if enabled
        let tuner_actor = if let Some(ref tuner_config) = config.tuner_config {
            info!("[HJB V2] Initializing auto-tuner actor (mode={})", tuner_config.mode);

            // Create initial tuning parameters from defaults
            // The params will be initialized from the unconstrained phi space
            // and the tuner will optimize them from there
            use crate::strategies::components::StrategyTuningParams;
            let initial_params = StrategyTuningParams::default();

            // Generate seed from asset name
            use std::collections::hash_map::DefaultHasher;
            use std::hash::{Hash, Hasher};
            let mut hasher = DefaultHasher::new();
            config.asset.hash(&mut hasher);
            let seed = hasher.finish();

            // Spawn tuner actor
            let actor = crate::strategies::async_tuner_actor::AsyncTunerActor::spawn(
                tuner_config.clone(),
                initial_params,
                config.tuner_updates_per_episode,
                seed,
            );

            info!("[HJB V2] Auto-tuner actor spawned successfully");
            Some(actor)
        } else {
            info!("[HJB V2] Auto-tuning disabled");
            None
        };

        Self {
            config,
            event_bus,
            state_store,
            market_pipeline,
            signal_generator,
            risk_adjuster,
            position_manager,
            order_executor,
            metrics_subscriber,
            tuner_actor,
            last_quote_time: 0.0,
            market_updates_in_episode: 0,
        }
    }

    /// Synchronize external state (from CurrentState) into our state store
    fn sync_external_state(&self, state: &CurrentState) {
        let timestamp = state.timestamp;

        // Update position
        self.state_store.update_position(state.position, timestamp);

        // Update risk metrics
        let mut risk_metrics = self.state_store.get_risk_metrics();
        risk_metrics.position = state.position;
        risk_metrics.account_equity = state.account_equity;
        risk_metrics.margin_used = state.margin_used;
        risk_metrics.max_position_size = state.max_position_size;
        risk_metrics.timestamp = timestamp;

        // Calculate margin available (simplified)
        risk_metrics.margin_available = (state.account_equity - state.margin_used).max(0.0);

        // Calculate liquidation distance (simplified)
        if state.margin_used > 0.0 {
            risk_metrics.liquidation_distance =
                (state.account_equity / state.margin_used).min(1.0);
        } else {
            risk_metrics.liquidation_distance = 1.0;
        }

        self.state_store.update_risk_metrics(risk_metrics);

        // Sync open orders
        self.sync_open_orders(state, timestamp);
    }

    /// Synchronize open orders from CurrentState
    fn sync_open_orders(&self, state: &CurrentState, timestamp: f64) {
        // Get current orders from state store
        let current_orders = self.state_store.get_all_orders();

        // Track which order IDs exist in CurrentState
        let mut state_order_ids = std::collections::HashSet::new();

        // Process bids
        for (_level, bid) in state.open_bids.iter().enumerate() {
            if let Some(oid) = bid.oid {
                state_order_ids.insert(oid);

                // If we don't have this order, add it
                if !current_orders.contains_key(&oid) {
                    let order = OpenOrder {
                        order_id: oid,
                        client_order_id: bid.cloid.map(|id| id.to_string()),
                        size: bid.orig_size,
                        price: bid.price,
                        is_buy: true,
                        remaining_size: bid.size,
                        state_machine: OrderStateMachine::from_open_order(
                            oid,
                            bid.size,
                            bid.price,
                            true,
                            self.config.state_machine_config.clone(),
                        ),
                        created_at: timestamp,
                    };

                    self.state_store.add_order(order);
                } else {
                    // Update remaining size if changed
                    self.state_store.update_order_remaining_size(oid, bid.size);
                }
            }
        }

        // Process asks
        for (_level, ask) in state.open_asks.iter().enumerate() {
            if let Some(oid) = ask.oid {
                state_order_ids.insert(oid);

                // If we don't have this order, add it
                if !current_orders.contains_key(&oid) {
                    let order = OpenOrder {
                        order_id: oid,
                        client_order_id: ask.cloid.map(|id| id.to_string()),
                        size: ask.orig_size,
                        price: ask.price,
                        is_buy: false,
                        remaining_size: ask.size,
                        state_machine: OrderStateMachine::from_open_order(
                            oid,
                            ask.size,
                            ask.price,
                            false,
                            self.config.state_machine_config.clone(),
                        ),
                        created_at: timestamp,
                    };

                    self.state_store.add_order(order);
                } else {
                    // Update remaining size if changed
                    self.state_store.update_order_remaining_size(oid, ask.size);
                }
            }
        }

        // Remove orders that no longer exist in CurrentState
        for (oid, _) in current_orders.iter() {
            if !state_order_ids.contains(oid) {
                self.state_store.remove_order(*oid);
            }
        }
    }

    /// Process market data and update state
    fn process_market_update(&mut self, update: &MarketUpdate) -> ProcessedMarketData {
        // Run through market data pipeline
        self.market_pipeline.process(update.clone())
    }

    /// Generate trading actions based on current snapshot
    fn generate_trading_actions(&mut self, snapshot: &TradingSnapshot, state: &CurrentState) -> Result<Vec<StrategyAction>, String> {
        // 1. Check position state
        let position_state = self.position_manager.get_state(
            snapshot.risk_metrics.position,
        );

        // Log position state changes
        if matches!(position_state, PositionState::Critical | PositionState::OverLimit) {
            warn!("[HJB V2] Position state: {:?}, Position: {:.2}, Max: {:.2}",
                position_state,
                snapshot.risk_metrics.position,
                snapshot.risk_metrics.max_position_size
            );
        }

        // 2. Generate raw signal from HJB optimizer
        let signal = self.signal_generator.generate_quotes(
            &snapshot.market_data,
            state,
        );

        debug!("[HJB V2] Generated signal: {} bids, {} asks, urgency: {:.2}",
            signal.bid_levels.len(),
            signal.ask_levels.len(),
            signal.urgency
        );

        // 3. Apply risk adjustments
        let adjusted_signal = self.risk_adjuster.adjust_signal(
            signal,
            snapshot,
        );

        if adjusted_signal.was_modified {
            debug!("[HJB V2] Signal adjusted: {}", adjusted_signal.adjustment_reason);
        }

        // 4. Execute orders
        match self.order_executor.execute(adjusted_signal, snapshot) {
            Ok(actions) => {
                debug!("[HJB V2] Generated {} strategy actions", actions.len());
                Ok(actions)
            }
            Err(e) => {
                warn!("[HJB V2] Order execution error: {}", e);
                Ok(vec![StrategyAction::NoOp])
            }
        }
    }

    /// Get current timestamp in seconds
    fn current_timestamp() -> f64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs_f64()
    }
}

// ============================================================================
// Strategy Trait Implementation
// ============================================================================

impl Strategy for HjbStrategyV2 {
    fn new(asset: &str, config: &Value) -> Self
    where
        Self: Sized,
    {
        HjbStrategyV2::new(asset, config)
    }

    fn on_market_update(
        &mut self,
        state: &CurrentState,
        update: &MarketUpdate,
    ) -> Vec<StrategyAction> {
        // 0. Track market update for tuner and check for episode completion
        if let Some(ref tuner) = self.tuner_actor {
            use crate::strategies::async_tuner_actor::TunerEvent;

            // Send market update event (non-blocking)
            tuner.send_event(TunerEvent::MarketUpdate);

            // Track episode progress
            self.market_updates_in_episode += 1;

            // Check if episode is complete
            if self.market_updates_in_episode >= self.config.tuner_updates_per_episode {
                tuner.send_event(TunerEvent::EpisodeComplete);
                self.market_updates_in_episode = 0;

                // Read updated parameters (non-blocking RwLock read ~10ns)
                let updated_params = tuner.get_params();

                // Apply updated parameters to signal generator
                // This updates the optimizer and clears the cache
                self.signal_generator.apply_tuning_params(&updated_params);

                info!("[HJB V2 Tuner] ✅ Applied new parameters: phi={:.4}, lambda={:.2}, max_pos={:.1}",
                    updated_params.phi, updated_params.lambda_base, updated_params.max_absolute_position_size);
            }
        }

        // 1. Sync external state into our state store
        self.sync_external_state(state);

        // 2. Process market data
        let processed_data = self.process_market_update(update);

        // 3. Update market data in state store
        let market_data = processed_data.to_market_data();
        self.state_store.update_market_data(market_data);

        // 4. Get complete snapshot
        let snapshot = self.state_store.get_snapshot();

        // 5. Generate trading actions
        match self.generate_trading_actions(&snapshot, state) {
            Ok(actions) => actions,
            Err(e) => {
                warn!("[HJB V2] Error generating actions: {}", e);
                vec![StrategyAction::NoOp]
            }
        }
    }

    fn on_user_update(
        &mut self,
        state: &CurrentState,
        update: &UserUpdate,
    ) -> Vec<StrategyAction> {
        // Sync state first
        self.sync_external_state(state);

        let timestamp = Self::current_timestamp();

        // Process fills
        for (fill_info, _level) in &update.fills {
            let price = fill_info.px.parse::<f64>().unwrap_or(0.0);
            let size = fill_info.sz.parse::<f64>().unwrap_or(0.0);
            let is_buy = fill_info.side == "B";

            // Publish fill event
            self.event_bus.publish(TradingEvent::OrderFilled {
                order_id: fill_info.oid,
                size,
                price,
                is_buy,
                timestamp,
            });

            // Track fill in tuner
            if let Some(ref tuner) = self.tuner_actor {
                use crate::strategies::async_tuner_actor::TunerEvent;

                // Calculate PnL (simplified - would need more context for accurate PnL)
                let pnl_estimate = if is_buy {
                    (state.l2_mid_price - price) * size
                } else {
                    (price - state.l2_mid_price) * size
                };

                tuner.send_event(TunerEvent::Fill {
                    pnl: pnl_estimate,
                    fill_price: price,
                    fill_size: size,
                    is_buy,
                    timestamp,
                });
            }

            // Remove filled order
            self.state_store.remove_order(fill_info.oid);

            debug!("[HJB V2] Order filled: oid={}, size={:.4}, price={:.4}, side={}",
                fill_info.oid, size, price, fill_info.side);
        }

        // Process order placements
        for oid in &update.orders_placed {
            debug!("[HJB V2] Order placed: oid={}", oid);
        }

        // Process cancellations
        for oid in &update.orders_cancelled {
            self.event_bus.publish(TradingEvent::OrderCanceled {
                order_id: *oid,
                timestamp,
            });

            // Track cancellation in tuner
            if let Some(ref tuner) = self.tuner_actor {
                use crate::strategies::async_tuner_actor::TunerEvent;
                tuner.send_event(TunerEvent::Cancel { timestamp });
            }

            self.state_store.remove_order(*oid);

            debug!("[HJB V2] Order canceled: oid={}", oid);
        }

        // Process failures
        for (oid, error) in &update.orders_failed {
            warn!("[HJB V2] Order failed: oid={}, error={}", oid, error);
            self.state_store.remove_order(*oid);
        }

        // Re-quote if we had a fill (position changed)
        if !update.fills.is_empty() {
            let snapshot = self.state_store.get_snapshot();
            match self.generate_trading_actions(&snapshot, state) {
                Ok(actions) => return actions,
                Err(e) => {
                    warn!("[HJB V2] Error generating actions after fill: {}", e);
                }
            }
        }

        vec![StrategyAction::NoOp]
    }

    fn on_tick(&mut self, state: &CurrentState) -> Vec<StrategyAction> {
        // Tick order state machines to check for timeouts
        let timestamp = Self::current_timestamp();
        self.state_store.tick_order_state_machines(timestamp);

        // Periodic requote if we have no orders (safety mechanism)
        let snapshot = self.state_store.get_snapshot();
        if snapshot.open_orders.is_empty() && state.l2_mid_price > 0.0 {
            // Only requote if enough time has passed (avoid spam)
            if timestamp - self.last_quote_time > 1.0 {
                self.last_quote_time = timestamp;

                debug!("[HJB V2] No open orders, requoting...");

                // Create minimal CurrentState for requoting
                let minimal_state = CurrentState {
                    position: snapshot.risk_metrics.position,
                    avg_entry_price: 0.0,
                    cost_basis: 0.0,
                    unrealized_pnl: 0.0,
                    realized_pnl: 0.0,
                    total_fees: 0.0,
                    l2_mid_price: snapshot.market_data.mid_price,
                    order_book: None,
                    market_spread_bps: snapshot.market_data.spread_bps,
                    lob_imbalance: snapshot.market_data.imbalance,
                    open_bids: vec![],
                    open_asks: vec![],
                    account_equity: snapshot.risk_metrics.account_equity,
                    margin_used: snapshot.risk_metrics.margin_used,
                    max_position_size: snapshot.risk_metrics.max_position_size,
                    timestamp,
                    session_start_time: 0.0,
                };

                match self.generate_trading_actions(&snapshot, &minimal_state) {
                    Ok(actions) => return actions,
                    Err(e) => {
                        warn!("[HJB V2] Error generating periodic quote: {}", e);
                    }
                }
            }
        }

        vec![StrategyAction::NoOp]
    }

    fn on_shutdown(&mut self, state: &CurrentState) -> Vec<StrategyAction> {
        info!("[HJB V2] Shutting down strategy");

        // Shutdown tuner actor if running
        if let Some(tuner) = self.tuner_actor.take() {
            use crate::strategies::async_tuner_actor::TunerEvent;

            info!("[HJB V2] Shutting down auto-tuner actor...");

            // Send shutdown event
            tuner.send_event(TunerEvent::Shutdown);

            // Wait for shutdown (this is an async operation, so we spawn a blocking task)
            // Note: In a production system, you might want to handle this differently
            // For now, we just send the shutdown signal and let the actor clean up
            std::thread::spawn(move || {
                let runtime = tokio::runtime::Runtime::new().unwrap();
                runtime.block_on(async {
                    if let Some(history) = tuner.shutdown().await {
                        info!("[HJB V2 Tuner] Final history: {}", history);
                    }
                });
            });

            info!("[HJB V2] Auto-tuner shutdown initiated");
        }

        // Log final metrics
        info!("[HJB V2] Total fills: {}", self.metrics_subscriber.get_fill_count());
        info!("[HJB V2] Total cancels: {}", self.metrics_subscriber.get_cancel_count());
        info!("[HJB V2] Position state changes: {}",
            self.metrics_subscriber.get_state_change_count());
        info!("[HJB V2] Total events: {}", self.event_bus.total_events());

        // Cancel all open orders
        let mut cancel_actions = Vec::new();
        for order in state.open_bids.iter().chain(state.open_asks.iter()) {
            if let Some(order_id) = order.oid {
                cancel_actions.push(StrategyAction::Cancel(ClientCancelRequest {
                    asset: self.config.asset.clone(),
                    oid: order_id,
                }));
            }
        }

        if !cancel_actions.is_empty() {
            info!("[HJB V2] Canceling {} open orders", cancel_actions.len());
        }

        cancel_actions
    }

    fn name(&self) -> &str {
        "HJB Strategy V2 (Modular)"
    }

    fn get_max_position_size(&self) -> f64 {
        self.config.max_position
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strategy_creation() {
        let config = serde_json::json!({
            "strategy_params": {
                "max_absolute_position_size": 10.0,
                "phi": 0.01,
                "lambda_base": 1.0,
            }
        });

        let strategy = HjbStrategyV2::new("TEST", &config);
        assert_eq!(strategy.name(), "HJB Strategy V2 (Modular)");
        assert_eq!(strategy.get_max_position_size(), 10.0);
    }

    #[test]
    fn test_position_synchronization() {
        let config = serde_json::json!({
            "strategy_params": {
                "max_absolute_position_size": 10.0,
            }
        });

        let strategy = HjbStrategyV2::new("TEST", &config);

        // Create a mock CurrentState
        let state = CurrentState {
            position: 5.0,
            avg_entry_price: 100.0,
            cost_basis: 500.0,
            unrealized_pnl: 10.0,
            realized_pnl: 5.0,
            total_fees: 1.0,
            l2_mid_price: 101.0,
            order_book: None,
            market_spread_bps: 5.0,
            lob_imbalance: 0.5,
            open_bids: vec![],
            open_asks: vec![],
            account_equity: 10000.0,
            margin_used: 1000.0,
            max_position_size: 10.0,
            timestamp: 0.0,
            session_start_time: 0.0,
        };

        strategy.sync_external_state(&state);

        // Verify position was synced
        assert_eq!(strategy.state_store.get_position(), 5.0);
    }
}
