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

use crate::strategy::{CurrentState, MarketUpdate, Strategy, StrategyAction, StrategyTuiMetrics, UserUpdate};
use crate::{
    AssetType, ClientCancelRequest, ClientLimit, ClientOrder, ClientOrderRequest,
    HawkesFillModel, L2BookData, MultiLevelConfig, MultiLevelOptimizer,
    OrderBook, ParameterUncertainty, ParticleFilterState, RobustConfig,
    TickLotValidator, Trade, TradeInfo,
};

// Import the component-based architecture
use crate::strategies::components::{
    HjbMultiLevelOptimizer, OptimizerInputs, OptimizerOutput,
};

// ----------------------------------------------------------------------------
// Import HJB implementation details from the sibling hjb_impl module
// ----------------------------------------------------------------------------
use super::hjb_impl::{
    AdamOptimizerState, CachedVolatilityEstimate, ControlVector, HJBComponents,
    OnlineAdverseSelectionModel, StateVector, TuningParams, ValueFunction,
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
// Cached Optimizer Result
// ============================================================================

/// Cached result from the quote optimizer to avoid redundant calculations
#[derive(Debug, Clone)]
struct CachedOptimizerResult {
    /// The optimizer output
    output: OptimizerOutput,
    
    /// Timestamp of this calculation
    timestamp: f64,
    
    /// State hash (for cache invalidation)
    state_hash: u64,
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

    /// **NEW: Component-based quote optimizer**
    /// Replaces the direct use of MultiLevelOptimizer
    quote_optimizer: HjbMultiLevelOptimizer,

    /// Multi-level optimizer (DEPRECATED - kept for backward compatibility)
    /// TODO: Remove once fully migrated to component
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
    
    /// **NEW: Cached optimizer result (for performance)**
    cached_optimizer_result: Option<CachedOptimizerResult>,
    
    /// **NEW: Performance metrics**
    optimization_call_count: u64,
    optimization_cache_hits: u64,
    total_optimization_time_us: u64,
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

        // Initialize robust control config (before using in quote_optimizer)
        let robust_config = strategy_config.robust_config.clone()
            .unwrap_or_else(|| RobustConfig::default());

        // Initialize multi-level optimizer
        let multi_level_config = strategy_config.multi_level_config.clone()
            .unwrap_or_else(|| MultiLevelConfig::default());

        // Initialize the component-based quote optimizer (before moving multi_level_config)
        // Convert tuning params from hjb_impl type to market_maker_v2 type
        let default_tuning_params = TuningParams::default().get_constrained();
        let converted_default_params = crate::market_maker_v2::ConstrainedTuningParams {
            skew_adjustment_factor: default_tuning_params.skew_adjustment_factor,
            adverse_selection_adjustment_factor: default_tuning_params.adverse_selection_adjustment_factor,
            adverse_selection_lambda: default_tuning_params.adverse_selection_lambda,
            inventory_urgency_threshold: default_tuning_params.inventory_urgency_threshold,
            liquidation_rate_multiplier: default_tuning_params.liquidation_rate_multiplier,
            min_spread_base_ratio: default_tuning_params.min_spread_base_ratio,
            adverse_selection_spread_scale: default_tuning_params.adverse_selection_spread_scale,
            control_gap_threshold: default_tuning_params.control_gap_threshold,
        };
        let quote_optimizer = HjbMultiLevelOptimizer::new(
            multi_level_config.clone(),
            robust_config.clone(),
            strategy_config.asset.clone(),
            strategy_config.max_absolute_position_size,
            converted_default_params,
        );

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

        // Initialize parameter uncertainty
        let current_uncertainty = ParameterUncertainty::default();

        // Initialize tuning parameters
        let tuning_params = Arc::new(RwLock::new(TuningParams::default()));

        // Initialize Adam optimizer
        let adam_optimizer = Arc::new(RwLock::new(AdamOptimizerState::default()));

        // Trading enabled by default for v3 (can be disabled via config)
        // In v2, this was controlled by the Adam optimizer, but for v3 we simplify
        let trading_enabled_default = strategy_config.enable_online_learning; // Enable if online learning is on
        let trading_enabled = Arc::new(RwLock::new(trading_enabled_default));

        info!("✅ Initialized HJB Strategy for {}", asset);
        info!("   - Trading enabled: {}", trading_enabled_default);
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
            quote_optimizer,
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
            cached_optimizer_result: None,
            optimization_call_count: 0,
            optimization_cache_hits: 0,
            total_optimization_time_us: 0,
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

    fn get_tui_metrics(&self) -> StrategyTuiMetrics {
        // Read particle filter metrics
        let pf = self.particle_filter.read();
        let pf_ess = pf.get_effective_sample_size();
        let pf_max_particles = pf.get_num_particles();
        let pf_vol_5th = pf.estimate_volatility_percentile_bps(0.05);
        let pf_vol_95th = pf.estimate_volatility_percentile_bps(0.95);
        let pf_volatility_bps = pf.estimate_volatility_bps();
        drop(pf);

        // Read online model metrics
        let model = self.online_adverse_selection_model.read();
        let online_model_mae = model.mean_absolute_error;
        let online_model_updates = model.update_count as u64;
        let online_model_lr = model.learning_rate;
        let online_model_enabled = model.enable_learning;
        drop(model);

        // Read Adam optimizer metrics
        let adam = self.adam_optimizer.read();
        let adam_gradient_samples = adam.t as u64;
        // Calculate average loss from the optimizer's v (second moment)
        let adam_avg_loss = if adam.t > 0 {
            adam.v.iter().map(|&x| x.sqrt()).sum::<f64>() / adam.v.len() as f64
        } else {
            0.0
        };
        drop(adam);

        // Calculate time since last update (for Adam)
        let current_time = chrono::Utc::now().timestamp() as f64;
        let cached_vol = self.cached_volatility.read();
        let adam_last_update_secs = current_time - cached_vol.last_update_time;
        drop(cached_vol);

        // TODO: Calculate Sharpe ratio from historical returns
        // For now, use a placeholder value
        let sharpe_ratio = 0.0;

        StrategyTuiMetrics {
            volatility_ema_bps: self.state_vector.volatility_ema_bps,
            pf_ess,
            pf_max_particles,
            pf_vol_5th,
            pf_vol_95th,
            pf_volatility_bps,
            adverse_selection_estimate: self.state_vector.adverse_selection_estimate,
            trade_flow_ema: self.state_vector.trade_flow_ema,
            online_model_mae,
            online_model_updates,
            online_model_lr,
            online_model_enabled,
            adam_gradient_samples,
            adam_avg_loss,
            adam_last_update_secs,
            sharpe_ratio,
        }
    }

    fn get_max_position_size(&self) -> f64 {
        self.config.max_absolute_position_size
    }
}

// ============================================================================
// Strategy Implementation Details
// ============================================================================

impl HjbStrategy {
    /// Convert from hjb_impl::ConstrainedTuningParams to market_maker_v2::ConstrainedTuningParams
    ///
    /// TODO: Remove this once the type duplication is resolved
    fn convert_tuning_params(
        params: &super::hjb_impl::ConstrainedTuningParams,
    ) -> crate::market_maker_v2::ConstrainedTuningParams {
        crate::market_maker_v2::ConstrainedTuningParams {
            skew_adjustment_factor: params.skew_adjustment_factor,
            adverse_selection_adjustment_factor: params.adverse_selection_adjustment_factor,
            adverse_selection_lambda: params.adverse_selection_lambda,
            inventory_urgency_threshold: params.inventory_urgency_threshold,
            liquidation_rate_multiplier: params.liquidation_rate_multiplier,
            min_spread_base_ratio: params.min_spread_base_ratio,
            adverse_selection_spread_scale: params.adverse_selection_spread_scale,
            control_gap_threshold: params.control_gap_threshold,
        }
    }

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

    /// Calculate multi-level target quotes using HJB/Hawkes/Robust control (Component-based)
    fn calculate_multi_level_targets(&mut self, state: &CurrentState) -> (Vec<(f64, f64)>, Vec<(f64, f64)>) {
        let start = std::time::Instant::now();
        let current_time = chrono::Utc::now().timestamp_millis() as f64 / 1000.0;

        // Increment optimization call counter
        self.optimization_call_count += 1;

        // --- CACHING LOGIC ---
        // Compute state hash for cache invalidation
        let state_hash = self.compute_state_hash(state);
        
        // Check if we can use cached result
        if let Some(cached) = &self.cached_optimizer_result {
            // Cache is valid if:
            // 1. State hasn't changed significantly (same hash)
            // 2. Cache is recent (< 100ms old for fast markets)
            let cache_age_ms = (current_time - cached.timestamp) * 1000.0;
            if cached.state_hash == state_hash && cache_age_ms < 100.0 {
                self.optimization_cache_hits += 1;
                debug!(
                    "[OPTIMIZER CACHE HIT] Age: {:.1}ms, Hit rate: {:.1}%",
                    cache_age_ms,
                    100.0 * self.optimization_cache_hits as f64 / self.optimization_call_count as f64
                );
                
                // Update taker rates from cached result
                self.latest_taker_buy_rate = cached.output.taker_buy_rate;
                self.latest_taker_sell_rate = cached.output.taker_sell_rate;
                
                return (cached.output.target_bids.clone(), cached.output.target_asks.clone());
            }
        }

        // --- MODEL INPUTS PREPARATION ---
        // --- MODEL INPUTS PREPARATION ---
        // Get cached volatility uncertainty
        let cached_vol = self.cached_volatility.read();
        let (_mu_std, _, _) = cached_vol.param_std_devs;
        let sigma_std = cached_vol.volatility_std_dev_bps;
        drop(cached_vol);

        // Prepare optimizer inputs
        let inputs = OptimizerInputs {
            current_time_sec: current_time,
            volatility_bps: self.state_vector.volatility_ema_bps,
            vol_uncertainty_bps: sigma_std,
            adverse_selection_bps: self.state_vector.adverse_selection_estimate,
            lob_imbalance: self.state_vector.lob_imbalance,
        };

        // --- COMPONENT CALL ---
        // Update the component's tuning params from the shared state
        let current_tuning_params = self.tuning_params.read().get_constrained();
        let converted_params = Self::convert_tuning_params(&current_tuning_params);
        self.quote_optimizer.set_tuning_params(converted_params);

        // Call the component with metadata
        let hawkes_lock = self.hawkes_model.read();
        let output = self.quote_optimizer.calculate_target_quotes_with_metadata(
            &inputs,
            state,
            &hawkes_lock,
        );
        drop(hawkes_lock);

        // --- UPDATE STRATEGY STATE ---
        // Store taker rates
        self.latest_taker_buy_rate = output.taker_buy_rate;
        self.latest_taker_sell_rate = output.taker_sell_rate;
        
        if output.liquidate {
            log::warn!("🚨 Liquidation mode triggered by optimizer!");
            log::info!("   Taker rates: buy={:.4}, sell={:.4}", 
                  self.latest_taker_buy_rate, self.latest_taker_sell_rate);
        }

        // --- PERFORMANCE METRICS ---
        let elapsed = start.elapsed();
        self.total_optimization_time_us += elapsed.as_micros() as u64;
        let avg_time_us = self.total_optimization_time_us / self.optimization_call_count;
        
        debug!(
            "[MULTI-LEVEL OPTIMIZE] Time: {}μs (avg: {}μs), Bids: {}, Asks: {}, Cache hit rate: {:.1}%",
            elapsed.as_micros(),
            avg_time_us,
            output.target_bids.len(),
            output.target_asks.len(),
            100.0 * self.optimization_cache_hits as f64 / self.optimization_call_count as f64
        );

        // --- CACHE UPDATE ---
        self.cached_optimizer_result = Some(CachedOptimizerResult {
            output: output.clone(),
            timestamp: current_time,
            state_hash,
        });

        (output.target_bids, output.target_asks)
    }
    
    /// Compute a hash of the current state for cache invalidation.
    /// 
    /// The hash includes key state variables that affect quote calculation:
    /// - Mid price (rounded to tick)
    /// - Inventory (rounded to 0.1)
    /// - Volatility (rounded to 0.1 bps)
    /// - Adverse selection (rounded to 0.1 bps)
    /// - LOB imbalance (rounded to 0.01)
    fn compute_state_hash(&self, state: &CurrentState) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        
        // Round values to avoid cache misses from tiny fluctuations
        let mid_price_ticks = (state.l2_mid_price / 0.01).round() as i64;
        let inventory_tenths = (state.position * 10.0).round() as i64;
        let volatility_tenths = (self.state_vector.volatility_ema_bps * 10.0).round() as i64;
        let adverse_selection_tenths = (self.state_vector.adverse_selection_estimate * 10.0).round() as i64;
        let lob_imbalance_hundredths = (self.state_vector.lob_imbalance * 100.0).round() as i64;
        
        mid_price_ticks.hash(&mut hasher);
        inventory_tenths.hash(&mut hasher);
        volatility_tenths.hash(&mut hasher);
        adverse_selection_tenths.hash(&mut hasher);
        lob_imbalance_hundredths.hash(&mut hasher);
        
        hasher.finish()
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
