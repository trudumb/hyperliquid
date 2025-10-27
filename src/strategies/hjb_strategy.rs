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
    HawkesFillModel, L2BookData, MultiLevelConfig,
    OrderBook, ParameterUncertainty, ParticleFilterState,
    TickLotValidator, Trade, TradeInfo,
};
use crate::strategies::components::{RobustConfig, InventorySkewConfig};

// Import the component-based architecture
use crate::strategies::components::{
    HjbMultiLevelOptimizer, OptimizerInputs, OptimizerOutput,
};

// ----------------------------------------------------------------------------
// Import HJB implementation details from the sibling hjb_impl module
// ----------------------------------------------------------------------------
use super::hjb_impl::{
    AdamOptimizerState, CachedVolatilityEstimate, HJBComponents,
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

    /// Inventory skew configuration (position and book-based quote adjustments)
    pub inventory_skew_config: Option<InventorySkewConfig>,

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

        // Load inventory skew config
        let inventory_skew_config = params.get("inventory_skew_config").and_then(|v| {
            serde_json::from_value(v.clone()).ok()
        });

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
            inventory_skew_config,
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

        // Initialize inventory skew config
        let inventory_skew_config = strategy_config.inventory_skew_config.clone()
            .unwrap_or_else(|| InventorySkewConfig::default());

        // Initialize multi-level optimizer
        let multi_level_config = strategy_config.multi_level_config.clone()
            .unwrap_or_else(|| MultiLevelConfig::default());

        // Initialize the component-based quote optimizer (before moving multi_level_config)
        let default_tuning_params = TuningParams::default().get_constrained();
        let quote_optimizer = HjbMultiLevelOptimizer::new(
            multi_level_config.clone(),
            robust_config.clone(),
            inventory_skew_config,
            strategy_config.asset.clone(),
            strategy_config.max_absolute_position_size,
            default_tuning_params,
        );

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

    fn get_max_position_size(&self) -> f64 {
        self.config.max_absolute_position_size
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
        drop(pf);

        // Update online adverse selection model
        let mut model = self.online_adverse_selection_model.write();
        model.update(mid_price);
        drop(model);

        // Update uncertainty estimates from particle filter
        self.update_uncertainty_estimates();
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
        self.quote_optimizer.set_tuning_params(current_tuning_params);

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

        // Update smoothed taker rates (EMA with alpha=0.3)
        self.update_smoothed_taker_rates(0.3);

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

    /// Update uncertainty estimates from particle filter statistics.
    ///
    /// This method updates `current_uncertainty` with the latest parameter
    /// uncertainty estimates from the particle filter and caches volatility
    /// statistics for use in robust control.
    fn update_uncertainty_estimates(&mut self) {
        let pf = self.particle_filter.read();

        // Extract volatility statistics
        let volatility_bps = pf.estimate_volatility_bps();
        let vol_5th_percentile = pf.estimate_volatility_percentile_bps(0.05);
        let vol_95th_percentile = pf.estimate_volatility_percentile_bps(0.95);

        // Estimate parameter standard deviations from particle spread
        let volatility_std_dev_bps = (vol_95th_percentile - vol_5th_percentile) / 3.29; // ~90% confidence interval / 3.29 ≈ std dev

        // Update cached volatility with all fields
        let current_time = chrono::Utc::now().timestamp() as f64;
        let mut cached_vol = self.cached_volatility.write();
        cached_vol.volatility_bps = volatility_bps;
        cached_vol.vol_5th_percentile = vol_5th_percentile;
        cached_vol.vol_95th_percentile = vol_95th_percentile;
        cached_vol.volatility_std_dev_bps = volatility_std_dev_bps;
        cached_vol.last_update_time = current_time;

        // Estimate parameter uncertainties for robust control
        // These represent the radius of the uncertainty set around point estimates
        let mu_uncertainty = self.state_vector.adverse_selection_estimate * 0.2; // 20% uncertainty
        let sigma_uncertainty = volatility_std_dev_bps;
        let kappa_uncertainty = 0.05; // 5% uncertainty in fill rates

        cached_vol.param_std_devs = (mu_uncertainty, sigma_uncertainty, kappa_uncertainty);
        drop(cached_vol);

        drop(pf);

        // Update current_uncertainty for robust control
        self.current_uncertainty = ParameterUncertainty {
            epsilon_mu: mu_uncertainty,
            epsilon_sigma: sigma_uncertainty,
            confidence: 0.95,
        };

        debug!(
            "[UNCERTAINTY UPDATE] σ: {:.2} bps (±{:.2}), μ: {:.2} bps (±{:.2})",
            volatility_bps, sigma_uncertainty,
            self.state_vector.adverse_selection_estimate, mu_uncertainty
        );
    }

    /// Calculate the value function penalty for current inventory.
    ///
    /// Returns the utility penalty (negative value) for holding the current
    /// inventory position. This uses the HJB value function V(Q, t).
    ///
    /// The value function represents the maximum expected terminal wealth from
    /// the current state. A negative value indicates inventory risk.
    fn get_inventory_penalty(&self) -> f64 {
        // V(Q, state) evaluates the value function given current inventory and state
        self.value_function.evaluate(self.state_vector.inventory, &self.state_vector)
    }

    /// Update smoothed taker rates using exponential moving average.
    ///
    /// This smooths the taker buy/sell rates from the optimizer to avoid
    /// rapid changes that could cause excessive taker executions.
    fn update_smoothed_taker_rates(&mut self, alpha: f64) {
        // EMA: smoothed = α * new + (1 - α) * old
        self.smoothed_taker_buy_rate =
            alpha * self.latest_taker_buy_rate + (1.0 - alpha) * self.smoothed_taker_buy_rate;
        self.smoothed_taker_sell_rate =
            alpha * self.latest_taker_sell_rate + (1.0 - alpha) * self.smoothed_taker_sell_rate;
    }

    /// Check if we should execute a taker order based on rate limiting.
    ///
    /// Returns true if enough time has passed since the last taker execution
    /// and the smoothed taker rate is above the threshold.
    ///
    /// # Arguments
    /// - `is_buy`: true for taker buy (to reduce long), false for taker sell (to reduce short)
    /// - `min_interval_sec`: minimum seconds between taker executions
    /// - `rate_threshold`: minimum smoothed rate to trigger taker execution
    #[allow(dead_code)]
    fn should_execute_taker_order(
        &self,
        is_buy: bool,
        min_interval_sec: f64,
        rate_threshold: f64,
    ) -> bool {
        let current_time = chrono::Utc::now().timestamp_millis() as f64 / 1000.0;

        let (smoothed_rate, last_execution_time) = if is_buy {
            (self.smoothed_taker_buy_rate, self.last_taker_buy_time)
        } else {
            (self.smoothed_taker_sell_rate, self.last_taker_sell_time)
        };

        // Check rate threshold
        if smoothed_rate < rate_threshold {
            return false;
        }

        // Check time since last execution
        let time_since_last = current_time - last_execution_time;
        if time_since_last < min_interval_sec {
            debug!(
                "[TAKER RATE LIMIT] {} order blocked: last executed {:.1}s ago (min: {:.1}s)",
                if is_buy { "BUY" } else { "SELL" },
                time_since_last,
                min_interval_sec
            );
            return false;
        }

        true
    }

    /// Record a taker execution (updates last execution time).
    #[allow(dead_code)]
    fn record_taker_execution(&mut self, is_buy: bool) {
        let current_time = chrono::Utc::now().timestamp_millis() as f64 / 1000.0;
        if is_buy {
            self.last_taker_buy_time = current_time;
        } else {
            self.last_taker_sell_time = current_time;
        }
    }

    /// Perform online learning using the Adam optimizer.
    ///
    /// This method computes gradients of the loss function with respect to
    /// tuning parameters and updates them using the Adam optimizer.
    ///
    /// The loss function is designed to minimize:
    /// - Inventory risk (quadratic penalty on position)
    /// - Adverse selection costs (based on realized vs. expected mid-price moves)
    /// - Spread tightness vs. fill rate tradeoff
    ///
    /// Returns the computed loss value.
    #[allow(dead_code)]
    fn perform_online_learning(&mut self, state: &CurrentState) -> f64 {
        if !self.config.enable_online_learning {
            return 0.0;
        }

        // Compute loss based on current performance
        let inventory_loss = self.hjb_components.phi * state.position.powi(2);
        let adverse_selection_loss = self.state_vector.adverse_selection_estimate.abs();
        let spread_loss = state.market_spread_bps.max(0.0);

        let total_loss = inventory_loss + adverse_selection_loss * 0.1 + spread_loss * 0.01;

        // Compute gradients (simplified finite difference approximation)
        // In a full implementation, this would use automatic differentiation
        let tuning_params = self.tuning_params.read().clone();
        let gradient_vector = self.compute_finite_difference_gradients(&tuning_params, total_loss);

        // Update parameters using Adam optimizer
        let mut adam = self.adam_optimizer.write();
        let updates = adam.compute_update(&gradient_vector);
        drop(adam);

        // Apply updates to tuning parameters
        self.apply_parameter_updates(&updates);

        debug!(
            "[ONLINE LEARNING] Loss: {:.4}, Gradient norm: {:.4}",
            total_loss,
            gradient_vector.iter().map(|g| g.powi(2)).sum::<f64>().sqrt()
        );

        total_loss
    }

    /// Compute finite difference gradients for online learning.
    ///
    /// This is a simplified gradient estimation using forward finite differences.
    /// In production, use automatic differentiation or policy gradient methods.
    fn compute_finite_difference_gradients(
        &self,
        _tuning_params: &crate::TuningParams,
        _current_loss: f64,
    ) -> Vec<f64> {
        // Placeholder: return small random gradients for now
        // In production, implement proper gradient computation
        vec![0.001; 8]
    }

    /// Apply parameter updates from the Adam optimizer.
    fn apply_parameter_updates(&mut self, updates: &[f64]) {
        let mut tuning_params = self.tuning_params.write();

        // Update phi parameters (in unconstrained space)
        let mut params_vec = vec![
            tuning_params.skew_adjustment_factor_phi,
            tuning_params.adverse_selection_adjustment_factor_phi,
            tuning_params.adverse_selection_lambda_phi,
            tuning_params.inventory_urgency_threshold_phi,
            tuning_params.liquidation_rate_multiplier_phi,
            tuning_params.min_spread_base_ratio_phi,
            tuning_params.adverse_selection_spread_scale_phi,
            tuning_params.control_gap_threshold_phi,
        ];

        // Apply updates
        for (i, update) in updates.iter().enumerate() {
            if i < params_vec.len() {
                params_vec[i] -= update; // Gradient descent: θ := θ - α * ∇L
            }
        }

        // Write back
        tuning_params.skew_adjustment_factor_phi = params_vec[0];
        tuning_params.adverse_selection_adjustment_factor_phi = params_vec[1];
        tuning_params.adverse_selection_lambda_phi = params_vec[2];
        tuning_params.inventory_urgency_threshold_phi = params_vec[3];
        tuning_params.liquidation_rate_multiplier_phi = params_vec[4];
        tuning_params.min_spread_base_ratio_phi = params_vec[5];
        tuning_params.adverse_selection_spread_scale_phi = params_vec[6];
        tuning_params.control_gap_threshold_phi = params_vec[7];
    }
}
