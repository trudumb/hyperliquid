// ============================================================================
// HJB Strategy - Advanced Market Making with Stochastic Control
// ============================================================================
//
// This strategy implements sophisticated market making using:
// - HJB (Hamilton-Jacobi-Bellman) optimal control equations
// - Hawkes process fill rate modeling (self-exciting point processes)
// - Particle filter stochastic volatility estimation
// - Robust control with parameter uncertainty sets
// - Multi-level order book optimization
// - Online adverse selection learning via SGD
//
// # Core Components
//
// 1. **State Vector** (Z_t): All observable market state
//    - Mid-price, inventory, adverse selection, LOB imbalance, volatility, trade flow
//
// 2. **Control Vector** (δ_t): All decision levers
//    - Bid/ask offsets (maker quotes), taker buy/sell rates, number of levels
//
// 3. **Value Function** (V(Q,t)): Quadratic penalty on inventory deviation
//    - Guides optimal quoting via HJB partial differential equation
//
// 4. **Multi-Level Optimizer**: Distributes liquidity across L1, L2, L3 levels
//    - Uses Hawkes fill rates to balance execution risk vs. adverse selection
//
// 5. **Robust Control**: Worst-case optimization under parameter uncertainty
//    - Widens spreads when particle filter estimates are uncertain
//
// # References
//
// - Avellaneda & Stoikov (2008): High-frequency trading in a limit order book
// - Guéant, Lehalle, Fernandez-Tapia (2013): Dealing with inventory risk
// - Cartea & Jaimungal (2015): Risk metrics and fine-tuning of high-frequency trading strategies
// - Hawkes (1971): Spectra of some self-exciting and mutually exciting point processes

use std::sync::Arc;
use parking_lot::RwLock;
use log::{debug, info};

use serde_json::Value;

use crate::strategy::{CurrentState, MarketUpdate, Strategy, StrategyAction, UserUpdate};
use crate::{
    AssetType, ClientCancelRequest, ClientLimit, ClientOrder, ClientOrderRequest,
    HawkesFillModel, L2BookData, MultiLevelConfig, MultiLevelOptimizer, OptimizationState,
    OrderBook, ParameterUncertainty, ParticleFilterState, RobustConfig, RobustParameters,
    TickLotValidator, Trade, TradeInfo, EPSILON,
};

// Re-import state vector and related components from the parent module
use crate::market_maker_v2::{
    StateVector, HJBComponents, ValueFunction,
    OnlineAdverseSelectionModel, CachedVolatilityEstimate,
    TuningParams, AdamOptimizerState,
};

// ============================================================================
// Strategy Configuration
// ============================================================================

/// Configuration for the HJB strategy, loaded from JSON
#[derive(Debug, Clone)]
pub struct HjbStrategyConfig {
    /// Asset to trade (e.g., "HYPE", "BTC")
    pub asset: String,

    /// Asset type (Perp or Spot) for tick/lot size validation
    pub asset_type: AssetType,

    /// Maximum absolute position size (inventory limit)
    pub max_absolute_position_size: f64,

    /// Enable multi-level market making (true = L1/L2/L3, false = single level)
    pub enable_multi_level: bool,

    /// Multi-level configuration (spacing, sizes, aggression)
    pub multi_level_config: Option<MultiLevelConfig>,

    /// Enable robust control (uncertainty-aware optimization)
    pub enable_robust_control: bool,

    /// Robust control configuration (robustness level, epsilon bounds)
    pub robust_config: Option<RobustConfig>,

    /// HJB inventory aversion parameter (φ in the HJB equation)
    /// Higher = more aggressive inventory management
    pub phi: f64,

    /// Base Poisson fill rate (λ_base in HJB)
    /// Estimated fills per second at best quotes
    pub lambda_base: f64,

    /// Maker fee in basis points (negative = rebate)
    pub maker_fee_bps: f64,

    /// Taker fee in basis points
    pub taker_fee_bps: f64,

    /// Enable online adverse selection learning
    pub enable_online_learning: bool,

    /// Performance gap threshold (%) to enable trading
    /// E.g., 15.0 = trade when heuristic performance gap < 15%
    pub enable_trading_gap_threshold_percent: f64,
}

impl HjbStrategyConfig {
    /// Load configuration from JSON Value
    pub fn from_json(asset: &str, config: &Value) -> Self {
        let params = &config["strategy_params"];

        // Load multi-level config if enabled
        let enable_multi_level = params.get("enable_multi_level")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);

        let multi_level_config = if enable_multi_level {
            params.get("multi_level_config").and_then(|v| {
                serde_json::from_value(v.clone()).ok()
            })
        } else {
            None
        };

        // Load robust control config if enabled
        let enable_robust_control = params.get("enable_robust_control")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);

        let robust_config = if enable_robust_control {
            params.get("robust_config").and_then(|v| {
                serde_json::from_value(v.clone()).ok()
            })
        } else {
            None
        };

        Self {
            asset: asset.to_string(),
            asset_type: AssetType::Perp,  // Default to Perp (can be configured via separate field if needed)
            max_absolute_position_size: params["max_absolute_position_size"]
                .as_f64()
                .unwrap_or(50.0),
            enable_multi_level,
            multi_level_config,
            enable_robust_control,
            robust_config,
            phi: params.get("phi").and_then(|v| v.as_f64()).unwrap_or(0.01),
            lambda_base: params.get("lambda_base").and_then(|v| v.as_f64()).unwrap_or(1.0),
            maker_fee_bps: params.get("maker_fee_bps").and_then(|v| v.as_f64()).unwrap_or(1.5),
            taker_fee_bps: params.get("taker_fee_bps").and_then(|v| v.as_f64()).unwrap_or(4.5),
            enable_online_learning: params.get("enable_online_learning")
                .and_then(|v| v.as_bool())
                .unwrap_or(true),
            enable_trading_gap_threshold_percent: params.get("enable_trading_gap_threshold_percent")
                .and_then(|v| v.as_f64())
                .unwrap_or(30.0),
        }
    }
}

// ============================================================================
// HJB Strategy Implementation
// ============================================================================

/// HJB-based market making strategy with advanced stochastic control
pub struct HjbStrategy {
    /// Strategy configuration
    config: HjbStrategyConfig,

    /// Tick/lot size validator for order price/size validation
    tick_lot_validator: TickLotValidator,

    /// State Vector (Z_t) - all observable market state
    state_vector: StateVector,

    /// HJB Components (φ, λ, fees)
    hjb_components: HJBComponents,

    /// Value Function V(Q,t) for inventory penalty calculation
    value_function: ValueFunction,

    /// Multi-level optimizer (contains config, Hawkes model, level logic)
    multi_level_optimizer: MultiLevelOptimizer,

    /// Hawkes process model for fill rate estimation
    hawkes_model: Arc<RwLock<HawkesFillModel>>,

    /// Particle filter for stochastic volatility estimation
    particle_filter: Arc<RwLock<ParticleFilterState>>,

    /// Online adverse selection learning model
    online_adverse_selection_model: Arc<RwLock<OnlineAdverseSelectionModel>>,

    /// Cached volatility estimate (updated periodically by background task)
    cached_volatility: Arc<RwLock<CachedVolatilityEstimate>>,

    /// Robust control configuration
    robust_config: RobustConfig,

    /// Current parameter uncertainty estimates
    current_uncertainty: ParameterUncertainty,

    /// Tuning parameters (for self-tuning Adam optimizer)
    tuning_params: Arc<RwLock<TuningParams>>,

    /// Adam optimizer state (for automatic parameter tuning)
    adam_optimizer: Arc<RwLock<AdamOptimizerState>>,

    /// Trading enabled flag (controlled by Adam optimizer)
    trading_enabled: Arc<RwLock<bool>>,

    /// Latest taker buy/sell rates (from multi-level optimizer)
    latest_taker_buy_rate: f64,
    latest_taker_sell_rate: f64,

    /// Timestamps of last taker executions (for rate limiting)
    last_taker_buy_time: f64,
    last_taker_sell_time: f64,

    /// Smoothed taker rates (exponential moving average)
    smoothed_taker_buy_rate: f64,
    smoothed_taker_sell_rate: f64,
}

impl Strategy for HjbStrategy {
    fn new(asset: &str, config: &Value) -> Self
    where
        Self: Sized,
    {
        let strategy_config = HjbStrategyConfig::from_json(asset, config);

        // Initialize tick/lot validator
        // Note: sz_decimals is hardcoded to 3 here - in production, fetch from API
        let tick_lot_validator = TickLotValidator::new(
            strategy_config.asset.clone(),
            strategy_config.asset_type.clone(),
            3, // sz_decimals (default for most assets)
        );

        // Initialize state vector
        let state_vector = StateVector::default();

        // Initialize HJB components
        let hjb_components = HJBComponents {
            lambda_base: strategy_config.lambda_base,
            phi: strategy_config.phi,
            maker_fee_bps: strategy_config.maker_fee_bps,
            taker_fee_bps: strategy_config.taker_fee_bps,
        };

        // Initialize value function
        let value_function = ValueFunction::new(strategy_config.phi, 3600.0);

        // Initialize Hawkes model (3 levels by default)
        let hawkes_model = Arc::new(RwLock::new(HawkesFillModel::new(3)));

        // Initialize multi-level optimizer
        let multi_level_config = strategy_config.multi_level_config.clone()
            .unwrap_or_else(|| MultiLevelConfig::default());

        let multi_level_optimizer = MultiLevelOptimizer::new(multi_level_config);

        // Initialize particle filter for stochastic volatility
        // Parameters: num_particles, mu, phi, sigma_eta, initial_h, initial_h_std_dev, seed
        let particle_filter = Arc::new(RwLock::new(ParticleFilterState::new(
            1000, // num_particles
            0.0,  // mu (mean reversion level)
            0.98, // phi (persistence)
            0.1,  // sigma_eta (volatility of volatility)
            -5.0, // initial_h (log volatility squared)
            0.5,  // initial_h_std_dev
            12345 // seed
        )));

        // Initialize online adverse selection model
        let online_adverse_selection_model = Arc::new(RwLock::new(
            OnlineAdverseSelectionModel::default()
        ));

        // Initialize cached volatility
        let cached_volatility = Arc::new(RwLock::new(CachedVolatilityEstimate::default()));

        // Initialize robust control config
        let robust_config = strategy_config.robust_config.clone()
            .unwrap_or_else(|| RobustConfig::default());

        // Initialize parameter uncertainty
        let current_uncertainty = ParameterUncertainty::default();

        // Initialize tuning parameters
        let tuning_params = Arc::new(RwLock::new(TuningParams::default()));

        // Initialize Adam optimizer
        let adam_optimizer = Arc::new(RwLock::new(AdamOptimizerState::default()));

        // Trading disabled by default (enabled by optimizer)
        let trading_enabled = Arc::new(RwLock::new(false));

        info!("✅ Initialized HJB Strategy for {}", asset);
        info!("   - Multi-level: {}", strategy_config.enable_multi_level);
        info!("   - Robust control: {}", strategy_config.enable_robust_control);
        info!("   - φ (inventory aversion): {}", strategy_config.phi);
        info!("   - λ_base (fill rate): {}", strategy_config.lambda_base);
        info!("   - Max position: {}", strategy_config.max_absolute_position_size);

        Self {
            config: strategy_config,
            tick_lot_validator,
            state_vector,
            hjb_components,
            value_function,
            multi_level_optimizer,
            hawkes_model,
            particle_filter,
            online_adverse_selection_model,
            cached_volatility,
            robust_config,
            current_uncertainty,
            tuning_params,
            adam_optimizer,
            trading_enabled,
            latest_taker_buy_rate: 0.0,
            latest_taker_sell_rate: 0.0,
            last_taker_buy_time: 0.0,
            last_taker_sell_time: 0.0,
            smoothed_taker_buy_rate: 0.0,
            smoothed_taker_sell_rate: 0.0,
        }
    }

    fn on_market_update(
        &mut self,
        state: &CurrentState,
        update: &MarketUpdate,
    ) -> Vec<StrategyAction> {
        // Check if trading is enabled
        if !*self.trading_enabled.read() {
            return vec![StrategyAction::NoOp];
        }

        // Update internal state vector from current state
        self.sync_state_vector(state);

        // Handle different types of market updates
        if let Some(ref l2_book) = update.l2_book {
            self.handle_l2_book_update(state, l2_book);
        }

        if !update.trades.is_empty() {
            self.handle_trades_update(state, &update.trades);
        }

        if update.mid_price.is_some() {
            self.handle_mid_price_update(state, update.mid_price.unwrap());
        }

        // Calculate optimal quotes using multi-level optimization
        let (target_bids, target_asks) = self.calculate_multi_level_targets(state);

        // Reconcile with existing orders
        self.reconcile_orders(state, target_bids, target_asks)
    }

    fn on_user_update(
        &mut self,
        state: &CurrentState,
        update: &UserUpdate,
    ) -> Vec<StrategyAction> {
        // Process fills and update Hawkes model
        if !update.fills.is_empty() {
            self.handle_fills(state, &update.fills);
        }

        // For now, we don't re-quote immediately after fills
        // The next market update will trigger a re-quote
        vec![StrategyAction::NoOp]
    }

    fn on_tick(&mut self, _state: &CurrentState) -> Vec<StrategyAction> {
        // Periodic cleanup and updates
        // Could implement:
        // - Stale order cleanup
        // - Periodic state logging
        // - Performance monitoring

        vec![StrategyAction::NoOp]
    }

    fn on_shutdown(&mut self, state: &CurrentState) -> Vec<StrategyAction> {
        info!("🛑 HJB Strategy shutting down...");

        // Cancel all open orders
        let mut actions = Vec::new();

        for order in &state.open_bids {
            actions.push(StrategyAction::Cancel(ClientCancelRequest {
                asset: self.config.asset.clone(),
                oid: order.oid,
            }));
        }

        for order in &state.open_asks {
            actions.push(StrategyAction::Cancel(ClientCancelRequest {
                asset: self.config.asset.clone(),
                oid: order.oid,
            }));
        }

        actions
    }

    fn name(&self) -> &str {
        "HJB Strategy v2"
    }
}

// ============================================================================
// Strategy Implementation Details
// ============================================================================

impl HjbStrategy {
    /// Sync internal state vector with current bot state
    fn sync_state_vector(&mut self, state: &CurrentState) {
        self.state_vector.mid_price = state.l2_mid_price;
        self.state_vector.inventory = state.position;
        self.state_vector.market_spread_bps = state.market_spread_bps;
        self.state_vector.lob_imbalance = state.lob_imbalance;
    }

    /// Handle L2 book updates
    fn handle_l2_book_update(&mut self, _state: &CurrentState, l2_book: &L2BookData) {
        // Parse order book
        if let Some(book) = OrderBook::from_l2_data(l2_book) {
            // Update state vector with book analysis
            if let Some(analysis) = book.analyze(5) {
                self.state_vector.lob_imbalance = analysis.imbalance;
                self.state_vector.market_spread_bps =
                    ((analysis.weighted_ask_price - analysis.weighted_bid_price) / analysis.weighted_bid_price) * 10000.0;
            }
        }
    }

    /// Handle trade flow updates
    fn handle_trades_update(&mut self, _state: &CurrentState, trades: &[Trade]) {
        let constrained_params = self.tuning_params.read().get_constrained();
        let trades_vec = trades.to_vec();
        self.state_vector.update_trade_flow_ema(&trades_vec, &constrained_params);
    }

    /// Handle mid-price updates
    fn handle_mid_price_update(&mut self, _state: &CurrentState, mid_price: f64) {
        // Update particle filter with new price observation
        let mut pf = self.particle_filter.write();
        pf.update(mid_price);

        // Update online adverse selection model
        let mut model = self.online_adverse_selection_model.write();
        model.update(mid_price);
    }

    /// Handle fills (update Hawkes model)
    fn handle_fills(&mut self, _state: &CurrentState, fills: &[TradeInfo]) {
        let current_time = chrono::Utc::now().timestamp_millis() as f64 / 1000.0;

        let mut hawkes = self.hawkes_model.write();
        for fill in fills {
            let is_bid_fill = fill.side == "B"; // Assuming "B" = bid fill (we got filled on sell)
            // Record fill at level 0 (L1) by default - in production, track actual levels
            hawkes.record_fill(0, is_bid_fill, current_time);
        }
    }

    /// Calculate multi-level target quotes using HJB/Hawkes/Robust control
    fn calculate_multi_level_targets(&mut self, _state: &CurrentState) -> (Vec<(f64, f64)>, Vec<(f64, f64)>) {
        let current_time = chrono::Utc::now().timestamp_millis() as f64 / 1000.0;

        // Get cached volatility uncertainty
        let cached_vol = self.cached_volatility.read();
        let (mu_std, _, _) = cached_vol.param_std_devs;
        let sigma_std = cached_vol.volatility_std_dev_bps;
        drop(cached_vol);

        // Update parameter uncertainty
        self.current_uncertainty = ParameterUncertainty::from_particle_filter_stats(
            mu_std,
            sigma_std,
            0.95,
        );

        // Compute robust parameters
        let robust_params = RobustParameters::compute(
            self.state_vector.adverse_selection_estimate,
            self.state_vector.volatility_ema_bps,
            self.state_vector.inventory,
            &self.current_uncertainty,
            &self.robust_config,
        );

        // Prepare optimization state
        let hawkes_lock = self.hawkes_model.read();
        let opt_state = OptimizationState {
            mid_price: self.state_vector.mid_price,
            inventory: self.state_vector.inventory,
            max_position: self.config.max_absolute_position_size,
            adverse_selection_bps: robust_params.mu_worst_case,
            lob_imbalance: self.state_vector.lob_imbalance,
            volatility_bps: robust_params.sigma_worst_case,
            current_time,
            hawkes_model: &hawkes_lock,
        };

        // Calculate robust base half-spread
        let min_profitable_half_spread = self.multi_level_optimizer.config().min_profitable_spread_bps / 2.0;
        let robust_base_half_spread = (robust_params.sigma_worst_case * 0.1)
            .max(min_profitable_half_spread)
            * robust_params.spread_multiplier;

        // Get tuning parameters
        let current_tuning_params = self.tuning_params.read().get_constrained();

        // Run multi-level optimization
        let multi_level_control = self.multi_level_optimizer.optimize(
            &opt_state,
            robust_base_half_spread,
            &current_tuning_params,
        );
        drop(hawkes_lock);

        // Store taker rates
        self.latest_taker_buy_rate = multi_level_control.taker_buy_rate;
        self.latest_taker_sell_rate = multi_level_control.taker_sell_rate;

        // Convert offsets to prices
        let mut target_bids = Vec::new();
        let mut target_asks = Vec::new();

        for (offset_bps, size_raw) in multi_level_control.bid_levels {
            if size_raw < EPSILON {
                continue;
            }

            let size = self.tick_lot_validator.round_size(size_raw, false);
            if size < EPSILON {
                continue;
            }

            let price_raw = self.state_vector.mid_price * (1.0 - offset_bps / 10000.0);
            let price = self.tick_lot_validator.round_price(price_raw, false);

            if price > 0.0 && (size * price) >= 10.0 {
                target_bids.push((price, size));
            }
        }

        for (offset_bps, size_raw) in multi_level_control.ask_levels {
            if size_raw < EPSILON {
                continue;
            }

            let size = self.tick_lot_validator.round_size(size_raw, false);
            if size < EPSILON {
                continue;
            }

            let price_raw = self.state_vector.mid_price * (1.0 + offset_bps / 10000.0);
            let price = self.tick_lot_validator.round_price(price_raw, true);

            if price > 0.0 && (size * price) >= 10.0 {
                target_asks.push((price, size));
            }
        }

        // Sort bids descending, asks ascending
        target_bids.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        target_asks.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        debug!(
            "[HJB STRATEGY] Targets: Bids({})={:?}, Asks({})={:?}",
            target_bids.len(), target_bids, target_asks.len(), target_asks
        );

        (target_bids, target_asks)
    }

    /// Reconcile existing orders with target quotes
    fn reconcile_orders(
        &self,
        state: &CurrentState,
        target_bids: Vec<(f64, f64)>,
        target_asks: Vec<(f64, f64)>,
    ) -> Vec<StrategyAction> {
        let mut actions = Vec::new();

        // Calculate tolerance for matching orders
        let min_price_step = 10f64.powi(-(self.tick_lot_validator.max_price_decimals() as i32));
        let min_size_step = 10f64.powi(-(self.tick_lot_validator.sz_decimals as i32));
        let price_tolerance = min_price_step * 0.5;
        let size_tolerance = min_size_step * 0.5;

        let mut remaining_target_bids = target_bids.clone();
        let mut remaining_target_asks = target_asks.clone();

        // Check existing bids
        for order in &state.open_bids {
            let matched = remaining_target_bids.iter().position(|(p, s)| {
                (p - order.price).abs() <= price_tolerance &&
                (s - order.size).abs() <= size_tolerance
            });

            if matched.is_some() {
                // Match found, remove from targets
                remaining_target_bids.remove(matched.unwrap());
            } else {
                // No match, cancel this order
                actions.push(StrategyAction::Cancel(ClientCancelRequest {
                    asset: self.config.asset.clone(),
                    oid: order.oid,
                }));
            }
        }

        // Check existing asks
        for order in &state.open_asks {
            let matched = remaining_target_asks.iter().position(|(p, s)| {
                (p - order.price).abs() <= price_tolerance &&
                (s - order.size).abs() <= size_tolerance
            });

            if matched.is_some() {
                // Match found, remove from targets
                remaining_target_asks.remove(matched.unwrap());
            } else {
                // No match, cancel this order
                actions.push(StrategyAction::Cancel(ClientCancelRequest {
                    asset: self.config.asset.clone(),
                    oid: order.oid,
                }));
            }
        }

        // Place remaining target bids
        for (price, size) in remaining_target_bids {
            actions.push(StrategyAction::Place(ClientOrderRequest {
                asset: self.config.asset.clone(),
                is_buy: true,
                reduce_only: false,
                limit_px: price,
                sz: size,
                cloid: Some(uuid::Uuid::new_v4()),
                order_type: ClientOrder::Limit(ClientLimit {
                    tif: "Gtc".to_string(),
                }),
            }));
        }

        // Place remaining target asks
        for (price, size) in remaining_target_asks {
            actions.push(StrategyAction::Place(ClientOrderRequest {
                asset: self.config.asset.clone(),
                is_buy: false,
                reduce_only: false,
                limit_px: price,
                sz: size,
                cloid: Some(uuid::Uuid::new_v4()),
                order_type: ClientOrder::Limit(ClientLimit {
                    tif: "Gtc".to_string(),
                }),
            }));
        }

        debug!("[HJB STRATEGY] Reconcile: {} actions", actions.len());
        actions
    }
}
