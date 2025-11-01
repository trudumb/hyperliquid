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
use log::{debug, info, warn};

use serde_json::Value;

use crate::strategy::{CurrentState, MarketUpdate, Strategy, StrategyAction, UserUpdate};
use crate::{
    AssetType, ClientCancelRequest, ClientLimit, ClientOrder, ClientOrderRequest,
    HawkesFillModel, L2BookData, MultiLevelConfig,
    OrderBook, OrderState, ParameterUncertainty,
    TickLotValidator, Trade, TradeInfo,
};
use crate::strategies::components::{RobustConfig, InventorySkewConfig};

// Import the component-based architecture
use crate::strategies::components::{
    HjbMultiLevelOptimizer, OptimizerInputs, OptimizerOutput,
    OrderChurnManager, OrderChurnConfig, OrderMetadata, MarketChurnState,
    MicropriceAsModel,
    VolatilityModel, EwmaVolatilityModel, EwmaVolConfig, ParticleFilterVolModel,
    HybridVolatilityModel, HybridVolConfig,
    TunerConfig, StrategyTuningParams, MultiObjectiveWeights,
    PositionManager, AllowedAction, PendingOrders,
};

// Import auto-tuning integration
use crate::strategies::tuner_integration::TunerIntegration;

// ----------------------------------------------------------------------------
// Import HJB implementation details from the sibling hjb_impl module
// ----------------------------------------------------------------------------
use super::hjb_impl::{
    CachedVolatilityEstimate, HJBComponents,
    StateVector, ValueFunction,
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

    /// Account leverage setting (1-max_leverage)
    /// Used to calculate margin requirements for position sizing
    pub leverage: usize,

    /// Maximum leverage allowed for this asset (from exchange metadata)
    pub max_leverage: usize,

    /// Margin safety buffer (0.0-1.0)
    /// Reserves this fraction of available margin (e.g., 0.2 = 20% buffer)
    pub margin_safety_buffer: f64,

    /// Order churn management configuration
    pub order_churn_config: Option<OrderChurnConfig>,

    /// Volatility model type: "particle_filter", "ewma", or "hybrid"
    pub volatility_model_type: String,

    /// EWMA volatility configuration (if using EWMA model)
    pub ewma_vol_config: Option<Value>,

    /// Hybrid volatility configuration (if using hybrid model)
    pub hybrid_vol_config: Option<Value>,

    /// Auto-tuning configuration (if enabled)
    pub auto_tuning_config: Option<Value>,
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

        // Load order churn config
        let order_churn_config = params.get("order_churn_config").and_then(|v| {
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
            leverage: params.get("leverage")
                .and_then(|v| v.as_u64())
                .unwrap_or(3) as usize,
            max_leverage: params.get("max_leverage")
                .and_then(|v| v.as_u64())
                .unwrap_or(50) as usize,
            margin_safety_buffer: params.get("margin_safety_buffer")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.2),
            order_churn_config,
            volatility_model_type: params.get("volatility_model_type")
                .and_then(|v| v.as_str())
                .unwrap_or("ewma") // Default to EWMA for tick data
                .to_string(),
            ewma_vol_config: params.get("ewma_vol_config").cloned(),
            hybrid_vol_config: params.get("hybrid_vol_config").cloned(),
            auto_tuning_config: params.get("auto_tuning").cloned(),
        }
    }
}

// ============================================================================
// Margin Calculator
// ============================================================================

/// Calculates margin requirements and available capacity for position sizing
#[derive(Debug, Clone)]
pub struct MarginCalculator {
    /// Account leverage (1-max_leverage)
    leverage: usize,

    /// Margin safety buffer (0.0-1.0)
    safety_buffer: f64,
}

impl MarginCalculator {
    pub fn new(leverage: usize, safety_buffer: f64) -> Self {
        Self {
            leverage,
            safety_buffer: safety_buffer.clamp(0.0, 0.99),
        }
    }

    /// Calculate initial margin required for a position
    /// Formula: position_size * mark_price / leverage
    pub fn initial_margin_required(&self, position_size: f64, mark_price: f64) -> f64 {
        (position_size.abs() * mark_price) / self.leverage as f64
    }

    /// Calculate maintenance margin (50% of initial margin at max leverage)
    /// For liquidation: account_value < maintenance_margin * total_notional
    pub fn maintenance_margin_ratio(&self, max_leverage: usize) -> f64 {
        0.5 / max_leverage as f64
    }

    /// Calculate available margin for new positions
    /// available = account_equity - current_margin_used - buffer
    pub fn available_margin(&self, account_equity: f64, margin_used: f64) -> f64 {
        let usable_equity = account_equity * (1.0 - self.safety_buffer);
        (usable_equity - margin_used).max(0.0)
    }

    /// Calculate maximum additional position size that can be opened
    /// max_size = available_margin * leverage / mark_price
    pub fn max_additional_position_size(
        &self,
        account_equity: f64,
        margin_used: f64,
        mark_price: f64,
    ) -> f64 {
        let available = self.available_margin(account_equity, margin_used);
        (available * self.leverage as f64) / mark_price
    }

    /// Adjust order size to fit within margin constraints
    /// Returns the maximum size that can be safely placed
    pub fn adjust_order_size_for_margin(
        &self,
        desired_size: f64,
        current_position: f64,
        account_equity: f64,
        margin_used: f64,
        mark_price: f64,
        is_buy: bool,
    ) -> f64 {
        // Calculate the position delta if this order fills
        let position_delta = if is_buy { desired_size } else { -desired_size };
        let new_position = current_position + position_delta;

        // If order reduces position (opposing direction), no margin check needed
        if new_position.abs() < current_position.abs() {
            return desired_size;
        }

        // Calculate how much position increase is allowed
        let position_increase = new_position.abs() - current_position.abs();
        let max_increase = self.max_additional_position_size(account_equity, margin_used, mark_price);

        if position_increase <= max_increase {
            // Full size fits within margin
            desired_size
        } else {
            // Reduce size to fit margin constraints
            let adjusted_increase = max_increase.max(0.0);
            let adjusted_size = if current_position.signum() == position_delta.signum() {
                // Same direction: can only add adjusted_increase
                adjusted_increase
            } else {
                // Crossing zero: can close current + open adjusted_increase on other side
                current_position.abs() + adjusted_increase
            };

            debug!(
                "[MARGIN CHECK] Order size reduced: {:.4} -> {:.4} (position: {:.2}, available margin increase: {:.2})",
                desired_size, adjusted_size, current_position, adjusted_increase
            );

            adjusted_size
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

    /// Volatility model (swappable: EWMA or Particle Filter)
    volatility_model: Box<dyn VolatilityModel>,

    /// Microprice-based adverse selection model (stable and production-ready)
    microprice_as_model: MicropriceAsModel,

    /// Cached volatility estimate (updated periodically by background task)
    cached_volatility: Arc<RwLock<CachedVolatilityEstimate>>,

    /// Robust control configuration
    #[allow(dead_code)]
    robust_config: RobustConfig,

    /// Current parameter uncertainty estimates
    current_uncertainty: ParameterUncertainty,

    /// Fixed tuning parameters (no online learning/tuning)
    /// Now using the new StrategyConstrainedParams type
    tuning_params: super::components::parameter_transforms::StrategyConstrainedParams,

    /// Trading enabled flag
    trading_enabled: bool,

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

    /// Margin calculator for position sizing
    margin_calculator: MarginCalculator,

    /// Intelligent order churn manager
    order_churn_manager: OrderChurnManager,

    /// Position manager (centralized position state and constraints)
    position_manager: PositionManager,

    // --- EXPERT ADDITIONS ---
    /// Smoothed volatility (EWMA) for robust spread calculation
    smoothed_volatility_bps: f64,

    /// EMA alpha for volatility smoothing (e.g., 0.1)
    vol_ema_alpha: f64,

    /// Z-Score threshold to trigger a re-quote (e.g., 1.0 = 1 std dev)
    z_score_threshold: f64,

    /// The mid-price from the last *significant* re-quote
    last_z_quote_mid_price: f64,

    /// Auto-tuning integration (None if disabled)
    tuner_integration: Option<TunerIntegration>,
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

        // Use new parameter system (StrategyTuningParams -> StrategyConstrainedParams)
        // The legacy tuning parameters are now handled internally by the optimizer
        use super::components::parameter_transforms::StrategyTuningParams;
        let tuning_params = StrategyTuningParams::default().get_constrained();

        // Initialize the component-based quote optimizer
        let quote_optimizer = HjbMultiLevelOptimizer::new(
            multi_level_config.clone(),
            robust_config.clone(),
            inventory_skew_config,
            strategy_config.asset.clone(),
            strategy_config.max_absolute_position_size,
            tuning_params.clone(),
        );

        // Initialize volatility model based on configuration
        let volatility_model: Box<dyn VolatilityModel> = match strategy_config.volatility_model_type.as_str() {
            "particle_filter" => {
                info!("📊 Using Particle Filter volatility model");
                Box::new(ParticleFilterVolModel::new(
                    1000,   // num_particles
                    -18.0,  // mu (recalibrated for tick-level: ln(σ²) where σ ≈ 3bps instantaneous)
                    0.95,   // phi (high persistence for tick data)
                    0.5,    // sigma_eta (moderate vol-of-vol)
                    -18.5,  // initial_h (starting near mu)
                    1.0,    // initial_h_std_dev (wider uncertainty)
                    12345   // seed
                ))
            }
            "hybrid" => {
                info!("📊 Using Hybrid EWMA-ParticleFilter volatility model (grounded to stochastic baseline)");
                if let Some(ref hybrid_config) = strategy_config.hybrid_vol_config {
                    Box::new(HybridVolatilityModel::from_json(hybrid_config))
                } else {
                    // Default hybrid config for high-frequency ticks
                    Box::new(HybridVolatilityModel::new(HybridVolConfig::high_frequency()))
                }
            }
            "ewma" | _ => {
                info!("📊 Using EWMA volatility model (tick-aware)");
                if let Some(ref ewma_config) = strategy_config.ewma_vol_config {
                    Box::new(EwmaVolatilityModel::from_json(ewma_config))
                } else {
                    // Default EWMA config for high-frequency ticks
                    Box::new(EwmaVolatilityModel::new(EwmaVolConfig::high_frequency()))
                }
            }
        };

        // Initialize cached volatility
        let cached_volatility = Arc::new(RwLock::new(CachedVolatilityEstimate::default()));

        // Get initial volatility for EWMA initialization
        let initial_volatility_bps = cached_volatility.read().volatility_bps;

        // Initialize parameter uncertainty
        let current_uncertainty = ParameterUncertainty::default();

        // Trading enabled by default
        let trading_enabled = true;

        // Initialize margin calculator
        let margin_calculator = MarginCalculator::new(
            strategy_config.leverage,
            strategy_config.margin_safety_buffer,
        );

        // Initialize order churn manager
        let order_churn_config = strategy_config.order_churn_config.clone()
            .unwrap_or_else(|| OrderChurnConfig::default());
        let order_churn_manager = OrderChurnManager::new(order_churn_config);

        info!("✅ Initialized HJB Strategy for {} | Trading: {} | Max Position: {} | Leverage: {}x | Margin Buffer: {:.1}%",
              asset, trading_enabled, strategy_config.max_absolute_position_size,
              strategy_config.leverage, strategy_config.margin_safety_buffer * 100.0);
        info!("📊 Using MicropriceAsModel for stable adverse selection estimation (no SGD/SPSA tuning)");

        // Initialize auto-tuning if enabled
        let tuner_integration = Self::initialize_tuner(&strategy_config);

        // Read auto-tuning parameters from config (or use defaults)
        let (vol_ema_alpha, z_score_threshold) = if let Some(ref auto_tuning) = strategy_config.auto_tuning_config {
            let vol_alpha = auto_tuning.get("vol_ema_alpha")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.1);
            let z_threshold = auto_tuning.get("z_score_threshold")
                .and_then(|v| v.as_f64())
                .unwrap_or(1.0);
            (vol_alpha, z_threshold)
        } else {
            (0.1, 1.0)  // Default values
        };

        info!("🎯 Auto-tuning params: vol_ema_alpha={:.3}, z_score_threshold={:.2}",
              vol_ema_alpha, z_score_threshold);

        // Extract max position before moving strategy_config
        let max_position = strategy_config.max_absolute_position_size;

        Self {
            config: strategy_config,
            tick_lot_validator,
            state_vector,
            hjb_components,
            value_function,
            quote_optimizer,
            hawkes_model,
            volatility_model,
            microprice_as_model: MicropriceAsModel::with_params(0.15, 8.0),  // Smoother, capped at 8bps
            cached_volatility,
            robust_config,
            current_uncertainty,
            tuning_params,
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
            margin_calculator,
            order_churn_manager,

            // Initialize position manager with default config
            position_manager: PositionManager::with_max_position(max_position),

            // --- EXPERT ADDITIONS ---
            // Initialize smoothed vol to the initial estimate
            smoothed_volatility_bps: initial_volatility_bps,
            vol_ema_alpha,
            z_score_threshold,
            last_z_quote_mid_price: 0.0,
            tuner_integration,
        }
    }
    fn on_market_update(
        &mut self,
        state: &CurrentState,
        update: &MarketUpdate,
    ) -> Vec<StrategyAction> {
        debug!("[HJB STRATEGY {}] on_market_update called", self.config.asset);

        // Track market update for auto-tuner
        if let Some(ref mut tuner) = self.tuner_integration {
            tuner.on_market_update();
        }

        // Check if trading is enabled
        if !self.trading_enabled {
            debug!("[HJB STRATEGY {}] Trading disabled, returning NoOp", self.config.asset);
            return vec![StrategyAction::NoOp];
        }

        // Update internal state vector from current state
        self.sync_state_vector(state);

        // Handle different types of market updates
        // IMPORTANT: Always update state (volatility model, L2 book, etc.) regardless of Z-Score filter
        if let Some(ref l2_book) = update.l2_book {
            debug!("[HJB STRATEGY {}] Processing L2 book update", self.config.asset);
            self.handle_l2_book_update(state, l2_book);

            // --- FIX: Trigger volatility update from L2Book's mid-price ---
            // The mid-price was just synced to state_vector by sync_state_vector.
            // We need to update the particle filter with this price to ensure
            // live volatility estimation on high-frequency L2Book ticks.
            if self.state_vector.mid_price > 0.0 {
                // state.l2_mid_price was updated by sync_state_vector earlier
                self.handle_mid_price_update(state, state.l2_mid_price);
            }
            // --- END FIX ---
        }

        if !update.trades.is_empty() {
            debug!("[HJB STRATEGY {}] Processing {} trades", self.config.asset, update.trades.len());
            self.handle_trades_update(state, &update.trades);
        }

        // This 'else if' prevents running the update twice if an AllMids message comes
        else if update.mid_price.is_some() {
            debug!("[HJB STRATEGY {}] Processing mid price update: ${:.3}",
                self.config.asset, update.mid_price.unwrap());
            self.handle_mid_price_update(state, update.mid_price.unwrap());
        }

        // --- EXPERT ADDITION: Z-Score Event Filter ---
        // NOTE: This filter is placed AFTER state updates so that volatility model always updates.
        // It only controls whether we re-quote, not whether we update internal state.
        if let Some(mid_price) = update.mid_price.or(state.order_book.as_ref().map(|b| b.mid_price)) {
            if self.last_z_quote_mid_price > 0.0 && self.smoothed_volatility_bps > 0.0 {
                // Calculate volatility in absolute price terms
                let vol_as_price = (self.smoothed_volatility_bps / 10000.0) * mid_price;

                // Calculate Z-Score of the price move
                let price_delta = (mid_price - self.last_z_quote_mid_price).abs();
                let z_score = price_delta / vol_as_price.max(1e-6);

                if z_score < self.z_score_threshold {
                    // Price move is NOT statistically significant.
                    // DO NOT re-quote. This is just noise.
                    // (But we already updated volatility and state above)
                    debug!("[HJB STRATEGY {}] Market update skipped (Z-Score: {:.2} < {:.2})",
                           self.config.asset, z_score, self.z_score_threshold);
                    return vec![StrategyAction::NoOp];
                }

                // Price move IS significant. Update our anchor price.
                debug!("[HJB STRATEGY {}] Z-Score threshold breached (Z-Score: {:.2}). Re-quoting.",
                       self.config.asset, z_score);
                self.last_z_quote_mid_price = mid_price;
            } else if mid_price > 0.0 {
                // Initialize the anchor price on the first valid mid-price
                self.last_z_quote_mid_price = mid_price;
            }
        }
        // --- End Z-Score Filter ---

        debug!("[HJB STRATEGY {}] Calculating multi-level targets (mid=${:.3}, pos={:.2})",
            self.config.asset, state.l2_mid_price, state.position);

        // --- NEW POSITION MANAGER LOGIC ---
        // Get position state and allowed actions from centralized manager
        let pos_state = self.position_manager.get_state(state.position);
        let pending_orders = PendingOrders::from_orders(&state.open_bids, &state.open_asks);
        let allowed = self.position_manager.get_allowed_action(
            pos_state.clone(),
            state.position,
            &pending_orders,
        );

        debug!("[HJB STRATEGY {}] Position state: {:?}, Allowed action: {:?}",
            self.config.asset, pos_state, allowed);

        // Execute appropriate strategy based on position state
        match allowed {
            AllowedAction::FullTrading { max_buy, max_sell } => {
                // Normal trading - calculate and place optimal quotes
                let (target_bids, target_asks) = self.calculate_multi_level_targets(state);
                self.generate_normal_quotes(state, target_bids, target_asks, max_buy, max_sell)
            }

            AllowedAction::ReducedTrading { max_buy, max_sell } => {
                // Warning state - use more conservative quotes
                let (target_bids, target_asks) = self.calculate_multi_level_targets(state);
                warn!("[HJB STRATEGY {}] ⚠️  WARNING STATE: Reducing trading aggressiveness (max_buy={:.2}, max_sell={:.2})",
                    self.config.asset, max_buy, max_sell);
                self.generate_conservative_quotes(state, target_bids, target_asks, max_buy, max_sell)
            }

            AllowedAction::ReduceOnly { reduce_size } => {
                // Critical state - only reduce position
                warn!("[HJB STRATEGY {}] 🚨 CRITICAL STATE: Reduce-only mode (target reduction: {:.2})",
                    self.config.asset, reduce_size);
                self.generate_reduce_only_orders(state, reduce_size)
            }

            AllowedAction::EmergencyLiquidation { full_size } => {
                // Emergency state - liquidate entire position
                warn!("[HJB STRATEGY {}] 🆘 EMERGENCY LIQUIDATION: Closing entire position ({:.2})",
                    self.config.asset, full_size);
                self.generate_emergency_liquidation(state, full_size)
            }
        }
        // --- END POSITION MANAGER LOGIC ---
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

    fn on_tick(&mut self, state: &CurrentState) -> Vec<StrategyAction> {
        debug!("[HJB STRATEGY {}] on_tick called", self.config.asset);

        // Check for parameter updates from auto-tuner
        if let Some(ref mut tuner) = self.tuner_integration {
            if let Some(new_params) = tuner.on_tick() {
                info!("[HJB STRATEGY {}] 🎛️ Applying new tuned parameters", self.config.asset);
                self.apply_tuned_parameters(new_params);
            }
        }

        // Check if trading is enabled
        if !self.trading_enabled {
            warn!("[HJB STRATEGY {}] Trading disabled, returning NoOp", self.config.asset);
            return vec![StrategyAction::NoOp];
        }

        // Only place orders if we have a valid mid price
        if state.l2_mid_price <= 0.0 {
            warn!("[HJB STRATEGY {}] Waiting for valid mid price (current: ${:.3})",
                self.config.asset, state.l2_mid_price);
            return vec![StrategyAction::NoOp];
        }

        // Sync state vector from current state
        self.sync_state_vector(state);

        // If we have no open orders and a valid mid price, generate initial quotes
        let has_orders = !state.open_bids.is_empty() || !state.open_asks.is_empty();

        debug!("[HJB STRATEGY {}] Tick state: mid=${:.3}, pos={:.2}, has_orders={}, bids={}, asks={}",
            self.config.asset, state.l2_mid_price, state.position, has_orders,
            state.open_bids.len(), state.open_asks.len());

        if !has_orders {
            info!("[HJB STRATEGY {}] No open orders, placing initial orders (mid: ${:.3})",
                self.config.asset, state.l2_mid_price);

            // --- USE POSITION MANAGER (same pattern as on_market_update) ---
            let pos_state = self.position_manager.get_state(state.position);
            let pending_orders = PendingOrders::from_orders(&state.open_bids, &state.open_asks);
            let allowed = self.position_manager.get_allowed_action(
                pos_state.clone(),
                state.position,
                &pending_orders,
            );

            info!("[HJB STRATEGY {}] on_tick: Position state: {:?}", self.config.asset, pos_state);

            return match allowed {
                AllowedAction::FullTrading { max_buy, max_sell } => {
                    let (target_bids, target_asks) = self.calculate_multi_level_targets(state);
                    self.generate_normal_quotes(state, target_bids, target_asks, max_buy, max_sell)
                }

                AllowedAction::ReducedTrading { max_buy, max_sell } => {
                    let (target_bids, target_asks) = self.calculate_multi_level_targets(state);
                    self.generate_conservative_quotes(state, target_bids, target_asks, max_buy, max_sell)
                }

                AllowedAction::ReduceOnly { reduce_size } => {
                    self.generate_reduce_only_orders(state, reduce_size)
                }

                AllowedAction::EmergencyLiquidation { full_size } => {
                    self.generate_emergency_liquidation(state, full_size)
                }
            };
            // --- END POSITION MANAGER LOGIC ---
        }

        // Periodic cleanup and updates for existing orders
        debug!("[HJB STRATEGY {}] Has orders, no action on tick", self.config.asset);
        vec![StrategyAction::NoOp]
    }

    fn on_shutdown(&mut self, state: &CurrentState) -> Vec<StrategyAction> {
        info!("🛑 HJB Strategy shutting down...");

        // Export auto-tuning history if enabled
        if let Some(ref tuner) = self.tuner_integration {
            if let Some(history) = tuner.export_history() {
                info!("🎛️ Auto-tuning history:\n{}", history);
            }
            if let Some(best_params) = tuner.get_best_params() {
                info!("🎛️ Best parameters found: phi={:.4}, lambda={:.2}, max_pos={:.1}",
                    best_params.phi, best_params.lambda_base, best_params.max_absolute_position_size);
            }
        }

        // Cancel all open orders
        let mut actions = Vec::new();

        for order in &state.open_bids {
            if let Some(oid) = order.oid {
                actions.push(StrategyAction::Cancel(ClientCancelRequest {
                    asset: self.config.asset.clone(),
                    oid,
                }));
            }
        }

        for order in &state.open_asks {
            if let Some(oid) = order.oid {
                actions.push(StrategyAction::Cancel(ClientCancelRequest {
                    asset: self.config.asset.clone(),
                    oid,
                }));
            }
        }

        // --- Add Position Closing Logic ---
        if state.position.abs() > crate::EPSILON {
            info!("Current position is {}, creating closing order.", state.position);

            let is_buy_to_close = state.position < 0.0; // Buy if short, sell if long
            let size_to_close = self.tick_lot_validator.round_size(state.position.abs(), false);

            // Define a minimum order size threshold (from tick validator)
            let min_order_size = 10f64.powi(-(self.tick_lot_validator.sz_decimals as i32));

            if size_to_close >= min_order_size && state.l2_mid_price > 0.0 {
                // Calculate an aggressive price for IOC limit order (simulating market order)
                let slippage_factor = if is_buy_to_close { 1.1 } else { 0.9 }; // 10% slippage tolerance
                let aggressive_px = self.tick_lot_validator.round_price(
                    state.l2_mid_price * slippage_factor,
                    is_buy_to_close // Round up for buys, down for sells relative to aggression
                );

                info!("Closing position: {} {} @ aggressive price {} (mid: {})",
                    if is_buy_to_close { "BUY" } else { "SELL" },
                    size_to_close,
                    aggressive_px,
                    state.l2_mid_price);

                let closing_order = ClientOrderRequest {
                    asset: self.config.asset.clone(),
                    is_buy: is_buy_to_close,
                    reduce_only: true, // IMPORTANT: Ensure it only closes position
                    limit_px: aggressive_px,
                    sz: size_to_close,
                    cloid: None, // Or Some(uuid::Uuid::new_v4())
                    order_type: ClientOrder::Limit(ClientLimit {
                        tif: "Ioc".to_string(), // Immediate Or Cancel
                    }),
                };
                actions.push(StrategyAction::Place(closing_order));
            } else {
                warn!("Position size {} is too small to close or mid_price is invalid.", size_to_close);
            }
        } else {
            info!("No position to close.");
        }
        // --- End Position Closing Logic ---

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
// Auto-Tuning Helper Methods
// ============================================================================

impl HjbStrategy {
    /// Initialize auto-tuner if enabled in config
    fn initialize_tuner(config: &HjbStrategyConfig) -> Option<TunerIntegration> {
        let tuning_config = config.auto_tuning_config.as_ref()?;

        // Check if enabled
        let enabled = tuning_config.get("enabled")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        if !enabled {
            info!("🎛️ Auto-tuning: DISABLED");
            return None;
        }

        // Parse tuner config
        let tuner_config = TunerConfig {
            enabled: true,
            mode: tuning_config.get("mode")
                .and_then(|v| v.as_str())
                .unwrap_or("continuous")
                .to_string(),
            episodes_per_update: tuning_config.get("episodes_per_update")
                .and_then(|v| v.as_u64())
                .unwrap_or(2) as usize,
            update_interval_seconds: tuning_config.get("update_interval_seconds")
                .and_then(|v| v.as_f64())
                .unwrap_or(3600.0),
            adaptive_threshold: tuning_config.get("adaptive_threshold")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.7),
            spsa_c: tuning_config.get("spsa_c")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.1),
            spsa_gamma: tuning_config.get("spsa_gamma")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.101),
            adam_alpha: tuning_config.get("adam_alpha")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.001),
            adam_beta1: tuning_config.get("adam_beta1")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.9),
            adam_beta2: tuning_config.get("adam_beta2")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.999),
            adam_epsilon: tuning_config.get("adam_epsilon")
                .and_then(|v| v.as_f64())
                .unwrap_or(1e-8),
            learning_rate_decay: tuning_config.get("learning_rate_decay")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0),
            max_param_change: tuning_config.get("max_param_change")
                .and_then(|v| v.as_f64())
                .unwrap_or(1.0),
            min_episodes_for_gradient: tuning_config.get("min_episodes_for_gradient")
                .and_then(|v| v.as_u64())
                .unwrap_or(20) as usize,
            objective_weights: Self::parse_objective_weights(tuning_config),
            verbose: tuning_config.get("verbose")
                .and_then(|v| v.as_bool())
                .unwrap_or(true),
        };

        // Create initial parameters from current config
        let initial_params = Self::config_to_tuning_params(config);

        // Updates per episode (ticks per episode)
        let updates_per_episode = tuning_config.get("updates_per_episode")
            .and_then(|v| v.as_u64())
            .unwrap_or(500) as usize;

        // Use timestamp as seed for reproducibility with some randomness
        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        info!("🎛️ Auto-tuning: ENABLED (mode={}, episodes_per_update={}, updates_per_episode={}, learning_rate={:.6})",
            tuner_config.mode, tuner_config.episodes_per_update, updates_per_episode, tuner_config.adam_alpha);

        Some(TunerIntegration::new(
            tuner_config,
            initial_params,
            updates_per_episode,
            seed,
        ))
    }

    /// Parse objective weights from config
    fn parse_objective_weights(_tuning_config: &Value) -> MultiObjectiveWeights {
        // For now, just use default MultiObjectiveWeights since the detailed
        // weight parsing requires matching the PerformanceTracker config structure
        MultiObjectiveWeights::default()
    }

    /// Convert current config to tuning parameters
    fn config_to_tuning_params(_config: &HjbStrategyConfig) -> StrategyTuningParams {
        // For now, just use defaults
        // In the future, this should read actual config values and convert to φ space
        StrategyTuningParams::default()
    }

    /// Apply new tuned parameters to strategy
    fn apply_tuned_parameters(&mut self, params: crate::strategies::components::StrategyConstrainedParams) {
        // Update HJB components
        self.hjb_components.phi = params.phi;
        self.hjb_components.lambda_base = params.lambda_base;
        self.hjb_components.maker_fee_bps = params.maker_fee_bps;
        self.hjb_components.taker_fee_bps = params.taker_fee_bps;

        // Update config
        self.config.phi = params.phi;
        self.config.lambda_base = params.lambda_base;
        self.config.max_absolute_position_size = params.max_absolute_position_size;
        self.config.maker_fee_bps = params.maker_fee_bps;
        self.config.taker_fee_bps = params.taker_fee_bps;
        self.config.leverage = params.leverage as usize;
        self.config.max_leverage = params.max_leverage as usize;
        self.config.margin_safety_buffer = params.margin_safety_buffer;
        self.config.enable_multi_level = params.enable_multi_level;
        self.config.enable_robust_control = params.enable_robust_control;

        // Update margin calculator
        self.margin_calculator = MarginCalculator::new(
            params.leverage as usize,
            params.margin_safety_buffer,
        );

        // Update multi-level config if present
        if let Some(ref mut ml_config) = self.config.multi_level_config {
            ml_config.max_levels = params.num_levels;
            ml_config.level_spacing_bps = params.level_spacing_bps;
            ml_config.min_profitable_spread_bps = params.min_profitable_spread_bps;
        }

        // Log the parameter changes
        info!("[HJB STRATEGY {}] 🎛️ Updated parameters: phi={:.4}, lambda={:.2}, max_pos={:.1}, leverage={}x",
            self.config.asset, params.phi, params.lambda_base, params.max_absolute_position_size, params.leverage);

        // Clear cached optimizer result to force recomputation with new params
        self.cached_optimizer_result = None;
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

            // Update microprice-based adverse selection model (no trades yet, will be updated in handle_trades_update)
            self.microprice_as_model.update(Some(&book), &[]);
        }
    }

    /// Handle trade flow updates
    fn handle_trades_update(&mut self, _state: &CurrentState, trades: &[Trade]) {
        let trades_vec = trades.to_vec();

        // Convert new params to legacy format for StateVector
        // (StateVector still uses old ConstrainedTuningParams)
        let legacy_params = HjbMultiLevelOptimizer::to_legacy_params(&self.tuning_params);
        self.state_vector.update_trade_flow_ema(&trades_vec, &legacy_params);

        // Update microprice model with trade flow (no order book here, just trades)
        self.microprice_as_model.update(None, trades);
    }

    /// Handle mid-price updates
    fn handle_mid_price_update(&mut self, _state: &CurrentState, mid_price: f64) {
        // Update volatility model with new price observation
        let market_update = MarketUpdate {
            asset: self.config.asset.clone(),
            mid_price: Some(mid_price),
            l2_book: None,
            trades: vec![],
            timestamp: chrono::Utc::now().timestamp_millis() as u64,
        };
        self.volatility_model.on_market_update(&market_update);

        // Update cached volatility from model
        let new_vol_bps = self.volatility_model.get_volatility_bps();
        let new_uncertainty_bps = self.volatility_model.get_uncertainty_bps();

        {
            let mut cached = self.cached_volatility.write();
            cached.volatility_bps = new_vol_bps;
            cached.volatility_std_dev_bps = new_uncertainty_bps;
        }

        // Update uncertainty estimates from volatility model
        self.update_uncertainty_estimates();

        // --- EXPERT ADDITION: Update Volatility EWMA ---
        let spot_volatility = self.cached_volatility.read().volatility_bps;
        self.smoothed_volatility_bps = (self.vol_ema_alpha * spot_volatility)
            + ((1.0 - self.vol_ema_alpha) * self.smoothed_volatility_bps);

        debug!("[HJB STRATEGY] Volatility updated: Spot={:.2}bps, Smoothed={:.2}bps, Uncertainty={:.2}bps",
            spot_volatility, self.smoothed_volatility_bps, new_uncertainty_bps);
    }

    /// Handle fills (update Hawkes model and order churn manager)
    fn handle_fills(&mut self, state: &CurrentState, fills: &[(TradeInfo, Option<usize>)]) {
        let current_time = chrono::Utc::now().timestamp_millis() as f64 / 1000.0;
        let current_time_ms = (current_time * 1000.0) as u64;

        let mut hawkes = self.hawkes_model.write();
        for (fill, filled_level) in fills {
            let is_bid_fill = fill.side == "B"; // "B" = bid fill (we got filled on our bid)

            // Use the level passed from UserUpdate (already looked up in BotRunner from active orders or cache)
            let level = match filled_level {
                Some(level) => *level, // Level was determined by BotRunner
                None => {
                    // Warning now only triggers if BotRunner genuinely couldn't find the level
                    warn!(
                        "Fill OID {} received by strategy without level (order not found in active or cache). Defaulting to L0 for Hawkes.",
                        fill.oid
                    );
                    0 // Fallback level
                }
            };

            hawkes.record_fill(level, is_bid_fill, current_time);

            // Record fill in order churn manager for fill rate tracking
            // Note: We don't have the exact placement_time_ms from the fill event,
            // so we'll use an estimate based on typical order lifetime
            // A better approach would be to store placement times when placing orders
            let estimated_placement_time_ms = current_time_ms.saturating_sub(2000); // Assume 2s average lifetime
            self.order_churn_manager.record_fill(
                level,
                is_bid_fill,
                estimated_placement_time_ms,
                current_time_ms,
            );

            // Track fill in performance tracker for auto-tuning
            if let Some(ref mut tuner) = self.tuner_integration {
                if let Some(tracker) = tuner.performance_tracker() {
                    // Parse fill price and size
                    if let (Ok(fill_price), Ok(fill_size)) = (fill.px.parse::<f64>(), fill.sz.parse::<f64>()) {
                        tracker.on_fill(is_bid_fill, fill_price, fill_size);

                        // Estimate PnL from this fill
                        let pnl = if is_bid_fill {
                            // Bought at fill_price, current mid is state.l2_mid_price
                            (state.l2_mid_price - fill_price) * fill_size
                        } else {
                            // Sold at fill_price, current mid is state.l2_mid_price
                            (fill_price - state.l2_mid_price) * fill_size
                        };
                        tracker.on_trade(pnl);
                    }
                }
            }

            debug!("Hawkes model updated: level={}, is_bid={}, time={:.2}", level, is_bid_fill, current_time);
            debug!("Churn manager fill recorded: level={}, is_bid={}", level, is_bid_fill);
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
            // 1. State hasn't changed significantly (same hash with wider tolerances)
            // 2. Cache is reasonably recent (< 1000ms = 1 second)
            //
            // Extended cache window reduces unnecessary re-optimizations and churn
            let cache_age_ms = (current_time - cached.timestamp) * 1000.0;
            if cached.state_hash == state_hash && cache_age_ms < 1000.0 {
                self.optimization_cache_hits += 1;
                debug!(
                    "[OPTIMIZER CACHE HIT] Age: {:.1}ms, Hit rate: {:.1}%",
                    cache_age_ms,
                    100.0 * self.optimization_cache_hits as f64 / self.optimization_call_count as f64
                );

                // Update taker rates from cached result
                self.latest_taker_buy_rate = cached.output.taker_buy_rate;
                self.latest_taker_sell_rate = cached.output.taker_sell_rate;

                // Return empty quotes if in liquidation mode (cached result)
                if cached.output.liquidate {
                    return (vec![], vec![]);
                }

                return (cached.output.target_bids.clone(), cached.output.target_asks.clone());
            }
        }

        // --- MODEL INPUTS PREPARATION ---
        // Get volatility AND uncertainty from the Particle Filter cache/state
        let cached_vol = self.cached_volatility.read();
        let volatility_bps_pf = cached_vol.volatility_bps; // Get PF point estimate
        let vol_uncertainty_bps_pf = cached_vol.volatility_std_dev_bps; // Get PF uncertainty
        drop(cached_vol);

        // USE MICROPRICE-BASED ADVERSE SELECTION (more stable than SGD model)
        let microprice_as_bps = self.microprice_as_model.get_adverse_selection_bps();

        // --- EXPERT ADDITION: Get conviction score ---
        let as_conviction = self.microprice_as_model.get_adverse_selection_conviction();
        let confident_as_bps = microprice_as_bps * as_conviction;

        // --- SANITY BOUNDS: Prevent irrational volatility values ---
        // For tick-level data, volatility should be in range [0.5, 50] bps
        // Higher values indicate model miscalibration
        const MIN_VOL_BPS: f64 = 0.5;
        const MAX_VOL_BPS: f64 = 50.0;
        const MAX_VOL_UNCERTAINTY_BPS: f64 = 25.0;

        let bounded_volatility_bps = self.smoothed_volatility_bps.clamp(MIN_VOL_BPS, MAX_VOL_BPS);
        let bounded_vol_uncertainty_bps = vol_uncertainty_bps_pf.clamp(0.0, MAX_VOL_UNCERTAINTY_BPS);

        if (bounded_volatility_bps - self.smoothed_volatility_bps).abs() > 0.1 {
            warn!("[VOL SANITY] Clamping volatility: {:.2}bps -> {:.2}bps (model may need recalibration)",
                self.smoothed_volatility_bps, bounded_volatility_bps);
        }

        // Calculate inventory penalty using value function for monitoring
        let inventory_penalty = self.get_inventory_penalty();

        debug!("[OPTIMIZER INPUTS {}] vol_spot={:.2}bps, vol_smooth={:.2}bps (bounded={:.2}bps), vol_unc={:.2}bps, as_raw={:.2}bps, conviction={:.2}, as_final={:.2}bps, lob_imb={:.3}, pos={:.2}, inv_penalty={:.4}",
            self.config.asset, volatility_bps_pf, self.smoothed_volatility_bps, bounded_volatility_bps, vol_uncertainty_bps_pf,
            microprice_as_bps, as_conviction, confident_as_bps, self.state_vector.lob_imbalance, state.position, inventory_penalty);

        // Prepare optimizer inputs USING SMOOTHED VOLATILITY AND CONVICTION-SCALED AS
        let inputs = OptimizerInputs {
            current_time_sec: current_time,
            volatility_bps: bounded_volatility_bps, // USE BOUNDED VOL (prevent runaway spreads)
            vol_uncertainty_bps: bounded_vol_uncertainty_bps, // USE BOUNDED UNCERTAINTY
            adverse_selection_bps: confident_as_bps, // USE CONVICTION-SCALED AS (more reliable)
            lob_imbalance: self.state_vector.lob_imbalance, // Keep LOB from state_vector
        };

        // --- COMPONENT CALL ---
        // Update the component's tuning params (fixed, no online learning)
        self.quote_optimizer.set_tuning_params(self.tuning_params.clone());

        // Call the component with metadata
        let hawkes_lock = self.hawkes_model.read();
        let output = self.quote_optimizer.calculate_target_quotes_with_metadata(
            &inputs,
            state,
            &hawkes_lock,
            self.config.maker_fee_bps,
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
            log::info!("   Clearing maker quotes - will use ONLY reduce-only taker orders");
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

        // --- LIQUIDATION MODE FIX: Return empty quotes when in liquidation ---
        // This ensures the strategy doesn't try to place maker orders while liquidating
        let (final_bids, final_asks) = if output.liquidate {
            log::info!("   Returning empty maker quotes (liquidation mode)");
            (vec![], vec![])
        } else {
            (output.target_bids.clone(), output.target_asks.clone())
        };

        // --- CACHE UPDATE ---
        // IMPORTANT: Cache the original output (with quotes) for metadata,
        // but we'll return empty quotes above when in liquidation mode
        self.cached_optimizer_result = Some(CachedOptimizerResult {
            output: output.clone(),
            timestamp: current_time,
            state_hash,
        });

        // --- PERFORMANCE TRACKING FOR AUTO-TUNING ---
        if let Some(ref mut tuner) = self.tuner_integration {
            if let Some(tracker) = tuner.performance_tracker() {
                // Track quotes being generated
                if !output.target_bids.is_empty() && !output.target_asks.is_empty() {
                    tracker.on_quote(
                        output.target_bids[0].0,  // best bid price
                        output.target_asks[0].0,  // best ask price
                        output.target_bids[0].1,  // best bid size
                        output.target_asks[0].1,  // best ask size
                    );
                }

                // Track inventory
                tracker.on_inventory_update(state.position);

                // Track margin usage
                let margin_used = self.margin_calculator.initial_margin_required(
                    state.position.abs(),
                    state.l2_mid_price
                );
                let margin_available = state.account_equity;
                tracker.on_margin_update(margin_used, margin_available);

                // Increment tick counter
                tracker.tick();
            }
        }

        (final_bids, final_asks)
    }
    
    /// Compute a hash of the current state for cache invalidation.
    ///
    /// The hash includes key state variables that affect quote calculation:
    /// - Mid price (rounded to 1 bps to allow small moves without re-quote)
    /// - Inventory (rounded to 1.0 full unit)
    /// - Volatility (rounded to 1 bps)
    /// - Adverse selection (rounded to 0.5 bps)
    /// - LOB imbalance (rounded to 0.05 - only major shifts)
    ///
    /// Wider rounding tolerances reduce cache misses and unnecessary re-quotes
    fn compute_state_hash(&self, state: &CurrentState) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();

        // Round values to avoid cache misses from tiny fluctuations
        // Use wider tolerances to reduce churn

        // Mid price: only re-quote if moves by >1 bps (0.01%)
        let mid_price_bps = ((state.l2_mid_price / state.l2_mid_price * 10000.0).round() / 10000.0 * 10000.0).round() as i64;

        // Inventory: only re-quote if changes by >1 full unit
        let inventory_units = state.position.round() as i64;

        // Volatility: only re-quote if changes by >1 bps
        let volatility_bps = self.state_vector.volatility_ema_bps.round() as i64;

        // Adverse selection: only re-quote if changes by >0.5 bps
        let adverse_selection_half_bps = (self.state_vector.adverse_selection_estimate * 2.0).round() as i64;

        // LOB imbalance: only re-quote if major shift (>5%)
        let lob_imbalance_5pct = (self.state_vector.lob_imbalance * 20.0).round() as i64;

        mid_price_bps.hash(&mut hasher);
        inventory_units.hash(&mut hasher);
        volatility_bps.hash(&mut hasher);
        adverse_selection_half_bps.hash(&mut hasher);
        lob_imbalance_5pct.hash(&mut hasher);

        hasher.finish()
    }

    /// Reconcile existing orders with target quotes
    fn reconcile_orders(
        &mut self,
        state: &CurrentState,
        target_bids: Vec<(f64, f64)>,
        target_asks: Vec<(f64, f64)>,
    ) -> Vec<StrategyAction> {
        let mut actions = Vec::new();

        // Calculate tolerance for matching orders with hysteresis to reduce churn
        let min_price_step = 10f64.powi(-(self.tick_lot_validator.max_price_decimals() as i32));
        let min_size_step = 10f64.powi(-(self.tick_lot_validator.sz_decimals as i32));

        // Widen tolerances to prevent canceling orders on minor quote changes
        // Allow up to 3 ticks of price deviation before canceling
        let price_tolerance = min_price_step * 3.0;
        let size_tolerance = min_size_step * 2.0;

        // Get current timestamp for order age calculations
        let current_time_ms = chrono::Utc::now().timestamp_millis() as u64;

        // Create market state for intelligent churn decisions
        let mid = state.l2_mid_price;
        let market_churn_state = MarketChurnState {
            current_time_ms,
            mid_price: mid,
            volatility_bps: self.state_vector.volatility_ema_bps,
            adverse_selection_bps: self.state_vector.adverse_selection_estimate,
            lob_imbalance: self.state_vector.lob_imbalance,
            best_bid: state.order_book.as_ref().and_then(|b| b.best_bid()),
            best_ask: state.order_book.as_ref().and_then(|b| b.best_ask()),
            queue_depth_ahead: None, // Can add queue tracking later
        };

        // --- STRATEGY LOGGING: TARGET QUOTES ---
        if !target_bids.is_empty() || !target_asks.is_empty() {
            log::info!("═══════════════════════════════════════════════════════");
            log::info!("📊 QUOTE RECONCILIATION (Mid: ${:.3})", mid);
            log::info!("───────────────────────────────────────────────────────");

            // Calculate and log spreads for each level
            for (i, (bid_px, bid_sz)) in target_bids.iter().enumerate() {
                let bid_offset_bps = ((mid - bid_px) / mid * 10000.0).abs();
                log::info!("  L{} BID: ${:.3} | Size: {:.2} | Offset: {:.2} bps",
                    i + 1, bid_px, bid_sz, bid_offset_bps);
            }

            for (i, (ask_px, ask_sz)) in target_asks.iter().enumerate() {
                let ask_offset_bps = ((ask_px - mid) / mid * 10000.0).abs();
                log::info!("  L{} ASK: ${:.3} | Size: {:.2} | Offset: {:.2} bps",
                    i + 1, ask_px, ask_sz, ask_offset_bps);
            }

            // Log effective spreads
            if !target_bids.is_empty() && !target_asks.is_empty() {
                let l1_spread_bps = ((target_asks[0].0 - target_bids[0].0) / mid * 10000.0).abs();
                log::info!("  ✅ L1 Full Spread: {:.2} bps", l1_spread_bps);
                log::info!("  💰 Fee Cost: {:.1} bps (maker) + {:.1} bps (taker) = {:.1} bps total",
                    self.config.maker_fee_bps, self.config.taker_fee_bps,
                    self.config.maker_fee_bps + self.config.taker_fee_bps);

                let net_margin_bps = l1_spread_bps - (self.config.maker_fee_bps + self.config.taker_fee_bps);
                if net_margin_bps < 0.0 {
                    log::warn!("  ⚠️  NEGATIVE MARGIN: {:.2} bps (spread too tight!)", net_margin_bps);
                } else {
                    log::info!("  ✅ Net Margin: {:.2} bps", net_margin_bps);
                }
            }

            log::info!("───────────────────────────────────────────────────────");
            log::info!("📋 EXISTING ORDERS: {} bids, {} asks",
                state.open_bids.len(), state.open_asks.len());
        }

        let mut remaining_target_bids = target_bids.clone();
        let mut remaining_target_asks = target_asks.clone();

        let mut matched_bids = 0;
        let mut canceled_bids = 0;
        let mut kept_young_bids = 0;

        // Check existing bids
        for (i, order) in state.open_bids.iter().enumerate() {
            // ✅ FIX: Skip orders that are already PendingCancel
            if order.state == OrderState::PendingCancel {
                debug!(
                    "[HJB STRATEGY] Skipping bid OID {:?} - already PendingCancel",
                    order.oid
                );
                continue;
            }

            let matched = remaining_target_bids.iter().enumerate().position(|(_, (p, s))| {
                (p - order.price).abs() <= price_tolerance &&
                (s - order.size).abs() <= size_tolerance
            });

            if let Some(target_idx) = matched {
                // Match found, remove from targets
                remaining_target_bids.remove(target_idx);
                matched_bids += 1;
            } else {
                // No match - use intelligent churn logic
                let order_meta = OrderMetadata {
                    oid: order.oid.unwrap_or(0),
                    price: order.price,
                    size: order.size,
                    is_buy: true,
                    level: i, // Assuming orders are sorted by level
                    placement_time_ms: order.timestamp,
                    target_price: target_bids.get(i).map(|(p, _)| *p).unwrap_or(order.price),
                    initial_queue_size_ahead: None,
                };

                let (should_refresh, reason) = self.order_churn_manager.should_refresh_order(
                    &order_meta,
                    &market_churn_state,
                );

                if should_refresh {
                    if let Some(oid) = order.oid {
                        actions.push(StrategyAction::Cancel(ClientCancelRequest {
                            asset: self.config.asset.clone(),
                            oid,
                        }));
                        canceled_bids += 1;
                        log::info!("  ❌ Canceling BID OID {} @ ${:.3} (reason: {})",
                            oid, order.price, reason);

                        // Record timeout for fill rate tracking
                        self.order_churn_manager.record_timeout(
                            i,
                            true,
                            order.timestamp,
                            current_time_ms,
                        );
                    }
                } else {
                    // Keep the order (too young or conditions don't warrant refresh)
                    kept_young_bids += 1;
                    debug!(
                        "[HJB STRATEGY] Keeping bid order OID {:?} (reason: {})",
                        order.oid, reason
                    );
                }
            }
        }

        let mut matched_asks = 0;
        let mut canceled_asks = 0;
        let mut kept_young_asks = 0;

        // Check existing asks
        for (i, order) in state.open_asks.iter().enumerate() {
            // ✅ FIX: Skip orders that are already PendingCancel
            if order.state == OrderState::PendingCancel {
                debug!(
                    "[HJB STRATEGY] Skipping ask OID {:?} - already PendingCancel",
                    order.oid
                );
                continue;
            }

            let matched = remaining_target_asks.iter().enumerate().position(|(_, (p, s))| {
                (p - order.price).abs() <= price_tolerance &&
                (s - order.size).abs() <= size_tolerance
            });

            if let Some(target_idx) = matched {
                // Match found, remove from targets
                remaining_target_asks.remove(target_idx);
                matched_asks += 1;
            } else {
                // No match - use intelligent churn logic
                let order_meta = OrderMetadata {
                    oid: order.oid.unwrap_or(0),
                    price: order.price,
                    size: order.size,
                    is_buy: false,
                    level: i, // Assuming orders are sorted by level
                    placement_time_ms: order.timestamp,
                    target_price: target_asks.get(i).map(|(p, _)| *p).unwrap_or(order.price),
                    initial_queue_size_ahead: None,
                };

                let (should_refresh, reason) = self.order_churn_manager.should_refresh_order(
                    &order_meta,
                    &market_churn_state,
                );

                if should_refresh {
                    if let Some(oid) = order.oid {
                        actions.push(StrategyAction::Cancel(ClientCancelRequest {
                            asset: self.config.asset.clone(),
                            oid,
                        }));
                        canceled_asks += 1;
                        log::info!("  ❌ Canceling ASK OID {} @ ${:.3} (reason: {})",
                            oid, order.price, reason);

                        // Record timeout for fill rate tracking
                        self.order_churn_manager.record_timeout(
                            i,
                            false,
                            order.timestamp,
                            current_time_ms,
                        );
                    }
                } else {
                    // Keep the order (too young or conditions don't warrant refresh)
                    kept_young_asks += 1;
                    debug!(
                        "[HJB STRATEGY] Keeping ask order OID {:?} (reason: {})",
                        order.oid, reason
                    );
                }
            }
        }

        log::info!("  ✅ Matched: {} bids, {} asks", matched_bids, matched_asks);
        log::info!("  ❌ Canceled: {} bids, {} asks", canceled_bids, canceled_asks);
        log::info!("  ⏳ Kept young: {} bids, {} asks", kept_young_bids, kept_young_asks);
        log::info!("  🆕 To place: {} bids, {} asks", remaining_target_bids.len(), remaining_target_asks.len());

        // Place remaining target bids (with margin checks)
        for (price, size) in remaining_target_bids {
            // SAFETY CHECK: Prevent orders that would exceed max_position
            let potential_position = state.position + size;
            if potential_position > state.max_position_size {
                log::warn!(
                    "⚠️  Skipping BID order: would exceed max_position (current={:.2}, order_size={:.2}, max={:.2})",
                    state.position, size, state.max_position_size
                );
                continue;
            }

            // Adjust size based on available margin
            let adjusted_size = self.margin_calculator.adjust_order_size_for_margin(
                size,
                state.position,
                state.account_equity,
                state.margin_used,
                state.l2_mid_price,
                true, // is_buy
            );

            // Skip if adjusted size is too small
            let min_size = 10f64.powi(-(self.tick_lot_validator.sz_decimals as i32));
            if adjusted_size < min_size {
                debug!(
                    "[MARGIN CHECK] Skipping bid: adjusted size {:.4} < min {:.4}",
                    adjusted_size, min_size
                );
                continue;
            }

            // FINAL CHECK: Ensure adjusted size also respects position limits
            let final_position = state.position + adjusted_size;
            if final_position > state.max_position_size {
                let safe_size = (state.max_position_size - state.position).max(0.0);
                if safe_size < min_size {
                    log::warn!(
                        "⚠️  Skipping BID order: adjusted size would exceed max_position (available={:.2})",
                        safe_size
                    );
                    continue;
                }
                log::warn!(
                    "⚠️  Reducing BID size from {:.2} to {:.2} to respect max_position",
                    adjusted_size, safe_size
                );

                // ✅ ROBUST FIX: Ensure our bid is NOT a taker order (spread-crossing check)
                if let Some(best_ask) = state.order_book.as_ref().and_then(|b| b.best_ask()) {
                    if price >= best_ask {
                        log::warn!(
                            "⚠️  Skipping BID order: Price {:.3} would cross spread (best ask: {:.3})",
                            price, best_ask
                        );
                        continue; // Do not place this order
                    }
                }

                // Use the safe size instead
                actions.push(StrategyAction::Place(ClientOrderRequest {
                    asset: self.config.asset.clone(),
                    is_buy: true,
                    reduce_only: false,
                    limit_px: price,
                    sz: safe_size,
                    cloid: Some(uuid::Uuid::new_v4()),
                    order_type: ClientOrder::Limit(ClientLimit {
                        tif: "Gtc".to_string(),
                    }),
                }));
                continue;
            }

            // ✅ ROBUST FIX: Ensure our bid is NOT a taker order (spread-crossing check)
            if let Some(best_ask) = state.order_book.as_ref().and_then(|b| b.best_ask()) {
                if price >= best_ask {
                    log::warn!(
                        "⚠️  Skipping BID order: Price {:.3} would cross spread (best ask: {:.3})",
                        price, best_ask
                    );
                    continue; // Do not place this order
                }
            }

            actions.push(StrategyAction::Place(ClientOrderRequest {
                asset: self.config.asset.clone(),
                is_buy: true,
                reduce_only: false,
                limit_px: price,
                sz: adjusted_size,
                cloid: Some(uuid::Uuid::new_v4()),
                order_type: ClientOrder::Limit(ClientLimit {
                    tif: "Gtc".to_string(),
                }),
            }));
        }

        // Place remaining target asks (with margin checks)
        for (price, size) in remaining_target_asks {
            // SAFETY CHECK: Prevent orders that would exceed max_position (short side)
            let potential_position = state.position - size;
            if potential_position < -state.max_position_size {
                log::warn!(
                    "⚠️  Skipping ASK order: would exceed max_position (current={:.2}, order_size={:.2}, max={:.2})",
                    state.position, size, state.max_position_size
                );
                continue;
            }

            // Adjust size based on available margin
            let adjusted_size = self.margin_calculator.adjust_order_size_for_margin(
                size,
                state.position,
                state.account_equity,
                state.margin_used,
                state.l2_mid_price,
                false, // is_buy
            );

            // Skip if adjusted size is too small
            let min_size = 10f64.powi(-(self.tick_lot_validator.sz_decimals as i32));
            if adjusted_size < min_size {
                debug!(
                    "[MARGIN CHECK] Skipping ask: adjusted size {:.4} < min {:.4}",
                    adjusted_size, min_size
                );
                continue;
            }

            // FINAL CHECK: Ensure adjusted size also respects position limits
            let final_position = state.position - adjusted_size;
            if final_position < -state.max_position_size {
                let safe_size = (state.position + state.max_position_size).max(0.0);
                if safe_size < min_size {
                    log::warn!(
                        "⚠️  Skipping ASK order: adjusted size would exceed max_position (available={:.2})",
                        safe_size
                    );
                    continue;
                }
                log::warn!(
                    "⚠️  Reducing ASK size from {:.2} to {:.2} to respect max_position",
                    adjusted_size, safe_size
                );

                // ✅ ROBUST FIX: Ensure our ask is NOT a taker order (spread-crossing check)
                if let Some(best_bid) = state.order_book.as_ref().and_then(|b| b.best_bid()) {
                    if price <= best_bid {
                        log::warn!(
                            "⚠️  Skipping ASK order: Price {:.3} would cross spread (best bid: {:.3})",
                            price, best_bid
                        );
                        continue; // Do not place this order
                    }
                }

                // Use the safe size instead
                actions.push(StrategyAction::Place(ClientOrderRequest {
                    asset: self.config.asset.clone(),
                    is_buy: false,
                    reduce_only: false,
                    limit_px: price,
                    sz: safe_size,
                    cloid: Some(uuid::Uuid::new_v4()),
                    order_type: ClientOrder::Limit(ClientLimit {
                        tif: "Gtc".to_string(),
                    }),
                }));
                continue;
            }

            // ✅ ROBUST FIX: Ensure our ask is NOT a taker order (spread-crossing check)
            if let Some(best_bid) = state.order_book.as_ref().and_then(|b| b.best_bid()) {
                if price <= best_bid {
                    log::warn!(
                        "⚠️  Skipping ASK order: Price {:.3} would cross spread (best bid: {:.3})",
                        price, best_bid
                    );
                    continue; // Do not place this order
                }
            }

            actions.push(StrategyAction::Place(ClientOrderRequest {
                asset: self.config.asset.clone(),
                is_buy: false,
                reduce_only: false,
                limit_px: price,
                sz: adjusted_size,
                cloid: Some(uuid::Uuid::new_v4()),
                order_type: ClientOrder::Limit(ClientLimit {
                    tif: "Gtc".to_string(),
                }),
            }));
        }

        // Final summary
        let new_bids = actions.iter().filter(|a| matches!(a, StrategyAction::Place(req) if req.is_buy)).count();
        let new_asks = actions.iter().filter(|a| matches!(a, StrategyAction::Place(req) if !req.is_buy)).count();
        let total_cancels = actions.iter().filter(|a| matches!(a, StrategyAction::Cancel(_))).count();

        log::info!("───────────────────────────────────────────────────────");
        log::info!("📤 ACTION SUMMARY: {} total ({} cancels, {} new bids, {} new asks)",
            actions.len(), total_cancels, new_bids, new_asks);
        log::info!("═══════════════════════════════════════════════════════\n");

        debug!("[HJB STRATEGY] Reconcile: {} actions", actions.len());
        actions
    }

    /// Create cancel actions for all open orders.
    ///
    /// This is used as part of Tier 3 liquidation to ensure we don't add to position
    /// while trying to urgently exit.
    fn create_cancel_all_orders(
        &self,
        state: &CurrentState,
    ) -> Vec<StrategyAction> {
        let mut actions = Vec::new();

        // Cancel ALL existing orders on BOTH sides
        for order in &state.open_bids {
            if order.state == OrderState::PendingCancel {
                debug!("Skipping bid OID {:?} - already PendingCancel", order.oid);
                continue;
            }
            if let Some(oid) = order.oid {
                actions.push(StrategyAction::Cancel(ClientCancelRequest {
                    asset: self.config.asset.clone(),
                    oid,
                }));
            }
        }

        for order in &state.open_asks {
            if order.state == OrderState::PendingCancel {
                debug!("Skipping ask OID {:?} - already PendingCancel", order.oid);
                continue;
            }
            if let Some(oid) = order.oid {
                actions.push(StrategyAction::Cancel(ClientCancelRequest {
                    asset: self.config.asset.clone(),
                    oid,
                }));
            }
        }

        actions
    }

    /// Create taker liquidation orders based on rates.
    ///
    /// This is Tier 3 of the liquidation logic - aggressive taker orders
    /// used only when inventory exceeds the urgency threshold (e.g., > 90%).
    ///
    /// NOTE: This method is kept for backwards compatibility but is no longer used.
    /// Use generate_emergency_liquidation() instead.
    #[allow(dead_code)]
    fn create_taker_liquidation_orders(
        &self,
        state: &CurrentState,
        taker_buy_rate: f64,
        taker_sell_rate: f64,
    ) -> Vec<StrategyAction> {
        let mut actions = Vec::new();
        let inventory = state.position;
        let is_long = inventory > 0.0;
        let is_short = inventory < 0.0;

        if is_long && taker_sell_rate > 0.0 {
            // TOO LONG: Need to sell aggressively
            self.place_aggressive_sell_orders(state, taker_sell_rate, &mut actions);
        } else if is_short && taker_buy_rate > 0.0 {
            // TOO SHORT: Need to buy aggressively
            self.place_aggressive_buy_orders(state, taker_buy_rate, &mut actions);
        } else if taker_buy_rate > 0.0 || taker_sell_rate > 0.0 {
            log::warn!(
                "⚠️  Taker liquidation triggered but no valid rate (buy={:.4}, sell={:.4})",
                taker_buy_rate, taker_sell_rate
            );
        }

        actions
    }


    /// Place aggressive sell orders to reduce long position.
    ///
    /// Strategy: Place multiple sell orders at increasingly aggressive prices
    /// to maximize fill probability while maintaining some price discipline.
    fn place_aggressive_sell_orders(
        &self,
        state: &CurrentState,
        total_size: f64,
        actions: &mut Vec<StrategyAction>,
    ) {
        // Get current best bid (we want to sell at or near this price)
        let best_bid = state.order_book
            .as_ref()
            .and_then(|book| book.best_bid())
            .unwrap_or(state.l2_mid_price * 0.999);

        let min_price_step = 10f64.powi(-(self.tick_lot_validator.max_price_decimals() as i32));
        let min_size_step = 10f64.powi(-(self.tick_lot_validator.sz_decimals as i32));
        let min_notional = 10.0; // $10 minimum

        // *** NEW LOGIC: ***
        // Check if the *smallest split* (20% of total) will be below the notional limit.
        // If it is, just send the *entire* amount as a single, consolidated order.
        let smallest_split_notional = (total_size * 0.20) * best_bid;

        if smallest_split_notional < min_notional {
            // Small total size, send as one order to avoid splits failing
            if total_size >= min_size_step && (total_size * best_bid) >= min_notional {
                log::info!(
                    "   📤 Liquidation SELL (Consolidated Order): {:.4} @ {:.2} (reduce_only)",
                    total_size, best_bid
                );
                actions.push(StrategyAction::Place(ClientOrderRequest {
                    asset: self.config.asset.clone(),
                    is_buy: false,
                    reduce_only: true,
                    limit_px: best_bid, // Most aggressive price
                    sz: total_size,
                    cloid: Some(uuid::Uuid::new_v4()),
                    order_type: ClientOrder::Limit(ClientLimit {
                        tif: "Ioc".to_string(),
                    }),
                }));
            } else {
                log::warn!(
                    "   ⚠️  Skipping Liquidation SELL: Total size {:.4} (Notional ${:.2}) is below minimums.",
                    total_size, total_size * best_bid
                );
            }
            return; // Exit function
        }
        // *** END NEW LOGIC ***

        // If total_size is large enough, proceed with splitting
        // Price aggressively to cross the spread and walk the book
        let aggressive_price_l1 = self.tick_lot_validator.round_price(best_bid - (min_price_step * 2.0), false); // 2 ticks *below* best bid
        let aggressive_price_l2 = self.tick_lot_validator.round_price(best_bid - (min_price_step * 4.0), false); // 4 ticks *below*

        // Split 60/40
        let levels = vec![
            (aggressive_price_l1, 0.60),
            (aggressive_price_l2, 0.40),
        ];

        for (price, size_fraction) in levels {
            let size = (total_size * size_fraction).max(min_size_step);

            // Adjust size based on available margin
            let adjusted_size = self.margin_calculator.adjust_order_size_for_margin(
                size,
                state.position,
                state.account_equity,
                state.margin_used,
                state.l2_mid_price,
                false, // is_buy = false (selling)
            );

            if adjusted_size >= min_size_step {
                // Check minimum notional value ($10)
                let notional_value = adjusted_size * price;

                if notional_value < min_notional {
                    log::warn!(
                        "   ⚠️  Skipping Liquidation SELL: Notional ${:.2} < min ${:.2}",
                        notional_value, min_notional
                    );
                    continue;
                }

                log::info!(
                    "   📤 Liquidation SELL: {:.4} @ {:.2} (reduce_only)",
                    adjusted_size, price
                );

                actions.push(StrategyAction::Place(ClientOrderRequest {
                    asset: self.config.asset.clone(),
                    is_buy: false,
                    reduce_only: true, // CRITICAL: prevent accidental short
                    limit_px: price,
                    sz: adjusted_size,
                    cloid: Some(uuid::Uuid::new_v4()),
                    order_type: ClientOrder::Limit(ClientLimit {
                        tif: "Ioc".to_string(), // Immediate-or-cancel for urgency
                    }),
                }));
            }
        }
    }

    /// Place aggressive buy orders to reduce short position.
    ///
    /// Strategy: Place multiple buy orders at increasingly aggressive prices
    /// to maximize fill probability while maintaining some price discipline.
    fn place_aggressive_buy_orders(
        &self,
        state: &CurrentState,
        total_size: f64,
        actions: &mut Vec<StrategyAction>,
    ) {
        // Get current best ask (we want to buy at or near this price)
        let best_ask = state.order_book
            .as_ref()
            .and_then(|book| book.best_ask())
            .unwrap_or(state.l2_mid_price * 1.001);

        let min_price_step = 10f64.powi(-(self.tick_lot_validator.max_price_decimals() as i32));
        let min_size_step = 10f64.powi(-(self.tick_lot_validator.sz_decimals as i32));
        let min_notional = 10.0; // $10 minimum

        // *** NEW LOGIC: ***
        // Check if the *smallest split* (20% of total) will be below the notional limit.
        // If it is, just send the *entire* amount as a single, consolidated order.
        let smallest_split_notional = (total_size * 0.20) * best_ask;

        if smallest_split_notional < min_notional {
            // Small total size, send as one order to avoid splits failing
            if total_size >= min_size_step && (total_size * best_ask) >= min_notional {
                log::info!(
                    "   📥 Liquidation BUY (Consolidated Order): {:.4} @ {:.2} (reduce_only)",
                    total_size, best_ask
                );
                actions.push(StrategyAction::Place(ClientOrderRequest {
                    asset: self.config.asset.clone(),
                    is_buy: true,
                    reduce_only: true,
                    limit_px: best_ask, // Most aggressive price
                    sz: total_size,
                    cloid: Some(uuid::Uuid::new_v4()),
                    order_type: ClientOrder::Limit(ClientLimit {
                        tif: "Ioc".to_string(),
                    }),
                }));
            } else {
                log::warn!(
                    "   ⚠️  Skipping Liquidation BUY: Total size {:.4} (Notional ${:.2}) is below minimums.",
                    total_size, total_size * best_ask
                );
            }
            return; // Exit function
        }
        // *** END NEW LOGIC ***

        // If total_size is large enough, proceed with splitting
        // Price aggressively to cross the spread and walk the book
        let aggressive_price_l1 = self.tick_lot_validator.round_price(best_ask + (min_price_step * 2.0), true); // 2 ticks *above* best ask
        let aggressive_price_l2 = self.tick_lot_validator.round_price(best_ask + (min_price_step * 4.0), true); // 4 ticks *above*

        // Split 60/40
        let levels = vec![
            (aggressive_price_l1, 0.60),
            (aggressive_price_l2, 0.40),
        ];

        for (price, size_fraction) in levels {
            let size = (total_size * size_fraction).max(min_size_step);

            // Adjust size based on available margin
            let adjusted_size = self.margin_calculator.adjust_order_size_for_margin(
                size,
                state.position,
                state.account_equity,
                state.margin_used,
                state.l2_mid_price,
                true, // is_buy = true
            );

            if adjusted_size >= min_size_step {
                // Check minimum notional value ($10)
                let notional_value = adjusted_size * price;

                if notional_value < min_notional {
                    log::warn!(
                        "   ⚠️  Skipping Liquidation BUY: Notional ${:.2} < min ${:.2}",
                        notional_value, min_notional
                    );
                    continue;
                }

                log::info!(
                    "   📥 Liquidation BUY: {:.4} @ {:.2} (reduce_only)",
                    adjusted_size, price
                );

                actions.push(StrategyAction::Place(ClientOrderRequest {
                    asset: self.config.asset.clone(),
                    is_buy: true,
                    reduce_only: true, // CRITICAL: prevent accidental long increase
                    limit_px: price,
                    sz: adjusted_size,
                    cloid: Some(uuid::Uuid::new_v4()),
                    order_type: ClientOrder::Limit(ClientLimit {
                        tif: "Ioc".to_string(), // Immediate-or-cancel for urgency
                    }),
                }));
            }
        }
    }

    /// Update uncertainty estimates from particle filter statistics.
    ///
    /// This method updates `current_uncertainty` with the latest parameter
    /// uncertainty estimates from the volatility model and caches volatility
    /// statistics for use in robust control.
    fn update_uncertainty_estimates(&mut self) {
        // Extract volatility statistics from the current model
        let volatility_bps = self.volatility_model.get_volatility_bps();
        let volatility_std_dev_bps = self.volatility_model.get_uncertainty_bps();

        // Estimate percentiles from mean ± std dev (approximate)
        // Assumes normal distribution: 5th ≈ mean - 1.645*std, 95th ≈ mean + 1.645*std
        let vol_5th_percentile = (volatility_bps - 1.645 * volatility_std_dev_bps).max(0.5);
        let vol_95th_percentile = volatility_bps + 1.645 * volatility_std_dev_bps;

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

    // ========================================================================
    // POSITION MANAGER HELPER METHODS
    // ========================================================================
    // These methods provide clean separation of logic for each position state

    /// Generate normal quotes (PositionState::Normal)
    fn generate_normal_quotes(
        &mut self,
        state: &CurrentState,
        target_bids: Vec<(f64, f64)>,
        target_asks: Vec<(f64, f64)>,
        _max_buy: f64,
        _max_sell: f64,
    ) -> Vec<StrategyAction> {
        // Normal trading - just reconcile orders as usual
        // The target quotes from optimizer already respect position limits
        self.reconcile_orders(state, target_bids, target_asks)
    }

    /// Generate conservative quotes (PositionState::Warning)
    fn generate_conservative_quotes(
        &mut self,
        state: &CurrentState,
        target_bids: Vec<(f64, f64)>,
        target_asks: Vec<(f64, f64)>,
        max_buy: f64,
        max_sell: f64,
    ) -> Vec<StrategyAction> {
        // Reduce quote sizes to be more conservative
        let conservative_bids: Vec<(f64, f64)> = target_bids
            .into_iter()
            .map(|(price, size)| {
                let reduced_size = size.min(max_buy);
                (price, reduced_size)
            })
            .filter(|(_, size)| *size > 0.0)
            .collect();

        let conservative_asks: Vec<(f64, f64)> = target_asks
            .into_iter()
            .map(|(price, size)| {
                let reduced_size = size.min(max_sell);
                (price, reduced_size)
            })
            .filter(|(_, size)| *size > 0.0)
            .collect();

        self.reconcile_orders(state, conservative_bids, conservative_asks)
    }

    /// Generate reduce-only orders (PositionState::Critical)
    fn generate_reduce_only_orders(
        &self,
        state: &CurrentState,
        reduce_size: f64,
    ) -> Vec<StrategyAction> {
        let mut actions = Vec::new();

        // Cancel all existing orders first
        actions.extend(self.create_cancel_all_orders(state));

        // Wait for cancels to clear before placing reduce-only orders
        let has_pending_cancels = state.open_bids.iter().any(|o| o.state == OrderState::PendingCancel)
            || state.open_asks.iter().any(|o| o.state == OrderState::PendingCancel);

        if has_pending_cancels {
            debug!("[HJB STRATEGY {}] Waiting for cancels before placing reduce-only orders", self.config.asset);
            return actions;
        }

        // Place reduce-only order to reduce position
        if reduce_size > 0.0 {
            let is_long = state.position > 0.0;
            let best_price = if is_long {
                // Long position - need to sell, use best bid
                state.order_book.as_ref()
                    .and_then(|b| b.best_bid())
                    .unwrap_or(state.l2_mid_price * 0.999)
            } else {
                // Short position - need to buy, use best ask
                state.order_book.as_ref()
                    .and_then(|b| b.best_ask())
                    .unwrap_or(state.l2_mid_price * 1.001)
            };

            // Validate size
            let min_size_step = 10f64.powi(-(self.tick_lot_validator.sz_decimals as i32));
            let adjusted_size = (reduce_size / min_size_step).floor() * min_size_step;

            if adjusted_size >= min_size_step {
                actions.push(StrategyAction::Place(ClientOrderRequest {
                    asset: self.config.asset.clone(),
                    is_buy: !is_long,  // Opposite direction of position
                    reduce_only: true,
                    limit_px: best_price,
                    sz: adjusted_size,
                    cloid: Some(uuid::Uuid::new_v4()),
                    order_type: ClientOrder::Limit(ClientLimit {
                        tif: "Ioc".to_string(),  // Immediate-or-cancel for urgency
                    }),
                }));

                info!("[HJB STRATEGY {}] Placed reduce-only order: {} {:.4} @ {:.2}",
                    self.config.asset,
                    if is_long { "SELL" } else { "BUY" },
                    adjusted_size,
                    best_price
                );
            }
        }

        actions
    }

    /// Generate emergency liquidation orders (PositionState::OverLimit)
    fn generate_emergency_liquidation(
        &self,
        state: &CurrentState,
        full_size: f64,
    ) -> Vec<StrategyAction> {
        let mut actions = Vec::new();

        // Cancel all existing orders immediately
        actions.extend(self.create_cancel_all_orders(state));

        // Wait for cancels to clear
        let has_pending_cancels = state.open_bids.iter().any(|o| o.state == OrderState::PendingCancel)
            || state.open_asks.iter().any(|o| o.state == OrderState::PendingCancel);

        if has_pending_cancels {
            warn!("[HJB STRATEGY {}] Emergency liquidation waiting for cancels", self.config.asset);
            return actions;
        }

        // Place aggressive liquidation orders
        if full_size > 0.0 {
            if state.position > 0.0 {
                // Long position - aggressive sell
                self.place_aggressive_sell_orders(state, full_size, &mut actions);
            } else if state.position < 0.0 {
                // Short position - aggressive buy
                self.place_aggressive_buy_orders(state, full_size, &mut actions);
            }
        }

        actions
    }

}
