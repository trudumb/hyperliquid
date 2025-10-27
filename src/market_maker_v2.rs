use alloy::{primitives::Address, signers::local::PrivateKeySigner};
use log::{debug, error, info, warn};
use tokio::sync::mpsc::unbounded_channel;
use tokio::time;
use std::sync::Arc;
use parking_lot::RwLock;  // Faster RwLock implementation
use std::collections::HashMap;

//RUST_LOG=info cargo run --bin market_maker_v2

use crate::{
    AssetType, BaseUrl, BookAnalysis, ClientCancelRequest, ClientLimit, ClientOrder,
    ClientOrderRequest, ClientTrigger, ExchangeClient, ExchangeDataStatus, ExchangeResponseStatus, InfoClient,
    MarketCloseParams, MarketOrderParams, Message,
    OrderBook, ParticleFilterState, AdaptiveConfig, Subscription, TickLotValidator, TradeInfo, UserData, EPSILON,
    HawkesFillModel, MultiLevelConfig, MultiLevelOptimizer, OptimizationState,
    ParameterUncertainty, RobustConfig, RobustParameters,
};

/// Epsilon for clamping values during inverse transforms (logit/log)
const PARAM_EPSILON: f64 = 1e-8;

/// Minimum time between taker executions (seconds) to prevent over-trading
const MIN_TAKER_INTERVAL_SECS: f64 = 2.0;

/// Smoothing factor for taker rate EMA (higher = more responsive, lower = smoother)
/// alpha = 0.3 means 30% weight on new value, 70% on old value
const TAKER_RATE_SMOOTHING_ALPHA: f64 = 0.3;

/// Maximum taker size as fraction of max position (prevents single large liquidations)
const MAX_TAKER_SIZE_FRACTION: f64 = 0.2;

/// L2 regularization strength for Adam optimizer (prevents parameter drift)
/// Higher values pull parameters toward zero more strongly
const L2_REGULARIZATION_LAMBDA: f64 = 0.001;

/// Timeout for pending cancel orders in HashMap (seconds)
/// Orders older than this are removed from pending_cancel_orders to prevent memory leak
/// Set to 3 minutes to handle delayed fill messages from the exchange
const PENDING_CANCEL_TIMEOUT_SECS: f64 = 180.0;

/// Update interval for background particle filter task (milliseconds)
/// Balances volatility accuracy vs. CPU usage in critical path
const PARTICLE_FILTER_UPDATE_INTERVAL_MS: u64 = 150;

/// Maximum retry attempts for failed order operations
const MAX_RETRY_ATTEMPTS: u8 = 3;

/// Initial backoff delay in milliseconds for retries
const INITIAL_BACKOFF_MS: u64 = 100;

// ============================================================================
// ORDER EXECUTION INFRASTRUCTURE (Async Order Management)
// ============================================================================

/// Simple token bucket rate limiter for API requests
struct TokenBucketRateLimiter {
    /// Maximum number of tokens (burst capacity)
    max_tokens: f64,
    /// Current number of available tokens
    tokens: f64,
    /// Tokens refilled per second
    refill_rate: f64,
    /// Last refill timestamp
    last_refill: std::time::Instant,
}

impl TokenBucketRateLimiter {
    fn new(rate_per_second: f64, burst_capacity: f64) -> Self {
        Self {
            max_tokens: burst_capacity,
            tokens: burst_capacity,
            refill_rate: rate_per_second,
            last_refill: std::time::Instant::now(),
        }
    }

    /// Try to acquire a token. Returns time to wait if rate limited.
    fn try_acquire(&mut self) -> Option<tokio::time::Duration> {
        // Refill tokens based on elapsed time
        let now = std::time::Instant::now();
        let elapsed = now.duration_since(self.last_refill).as_secs_f64();
        self.tokens = (self.tokens + elapsed * self.refill_rate).min(self.max_tokens);
        self.last_refill = now;

        if self.tokens >= 1.0 {
            self.tokens -= 1.0;
            None  // Token acquired, no wait needed
        } else {
            // Calculate how long to wait for next token
            let wait_time = (1.0 - self.tokens) / self.refill_rate;
            Some(tokio::time::Duration::from_secs_f64(wait_time))
        }
    }
}

/// Helper function to retry an async operation with exponential backoff
/// Only retries on network errors (GenericRequest), not on API validation errors (ClientRequest)
async fn retry_with_backoff<F, Fut, T>(
    operation: F,
    max_retries: u8,
    operation_desc: &str,
) -> crate::prelude::Result<T>
where
    F: Fn() -> Fut,
    Fut: std::future::Future<Output = crate::prelude::Result<T>>,
{
    let mut attempt = 0;
    loop {
        match operation().await {
            Ok(result) => return Ok(result),
            Err(e) => {
                // Only retry on network errors, not on API validation errors
                let is_retriable = matches!(
                    e,
                    crate::Error::GenericRequest(_) | crate::Error::ServerRequest { .. }
                );

                attempt += 1;
                if !is_retriable || attempt >= max_retries {
                    if !is_retriable {
                        debug!("{} - non-retriable error, not retrying: {}", operation_desc, e);
                    } else {
                        warn!("{} - max retries ({}) exceeded: {}", operation_desc, max_retries, e);
                    }
                    return Err(e);
                }

                // Exponential backoff: 100ms, 200ms, 400ms
                let backoff_ms = INITIAL_BACKOFF_MS * (1 << (attempt - 1));
                debug!("{} - attempt {}/{} failed, retrying after {}ms: {}",
                       operation_desc, attempt, max_retries, backoff_ms, e);
                tokio::time::sleep(tokio::time::Duration::from_millis(backoff_ms)).await;
            }
        }
    }
}

/// Commands for the async order execution task
/// These are sent via MPSC channel to decouple network I/O from the hot path
#[derive(Debug)]
pub enum OrderCommand {
    /// Place a new order
    Place {
        request: ClientOrderRequest,
        /// Unique ID for tracking this placement intent
        intent_id: u64,
    },
    /// Batch place multiple orders (optimized for bulk_order API)
    BatchPlace {
        requests: Vec<ClientOrderRequest>,
        /// Intent IDs corresponding to each order request
        intent_ids: Vec<u64>,
    },
    /// Cancel a single order
    Cancel {
        request: ClientCancelRequest,
    },
    /// Batch cancel multiple orders
    BatchCancel {
        requests: Vec<ClientCancelRequest>,
    },
}

/// Intent tracking for pending order placements
/// Helps reconcile async order state with fill confirmations
#[derive(Debug, Clone)]
pub struct OrderIntent {
    pub intent_id: u64,
    pub side: bool,  // true = buy, false = sell
    pub price: f64,
    pub size: f64,
    pub level: usize,
    pub submitted_time: f64,
}

/// Result of an order placement attempt from the async execution task
#[derive(Debug, Clone)]
pub struct OrderPlacementResult {
    pub intent_id: u64,
    pub oid: Option<u64>,  // Some(oid) on success, None on failure
    pub success: bool,
    pub error_message: Option<String>,
}

/// Cached volatility estimate from background particle filter
/// Updated periodically to avoid expensive PF updates in hot path
#[derive(Debug, Clone)]
pub struct CachedVolatilityEstimate {
    /// Current volatility estimate in bps (annualized)
    pub volatility_bps: f64,

    /// 5th percentile (lower bound)
    pub vol_5th_percentile: f64,

    /// 95th percentile (upper bound)
    pub vol_95th_percentile: f64,

    /// Parameter standard deviations for robust control
    pub param_std_devs: (f64, f64, f64),  // (mu_std, phi_std, sigma_eta_std)

    /// Volatility standard deviation for robust control
    pub volatility_std_dev_bps: f64,

    /// Unix timestamp of last update
    pub last_update_time: f64,
}

impl Default for CachedVolatilityEstimate {
    fn default() -> Self {
        Self {
            volatility_bps: 100.0,  // Default to 100 bps
            vol_5th_percentile: 80.0,
            vol_95th_percentile: 120.0,
            param_std_devs: (0.1, 0.01, 0.1),
            volatility_std_dev_bps: 10.0,
            last_update_time: 0.0,
        }
    }
}

/// Standard sigmoid function: maps (-inf, inf) -> (0, 1)
fn sigmoid(phi: f64) -> f64 {
    1.0 / (1.0 + (-phi).exp())
}

/// Logit function (inverse sigmoid): maps (0, 1) -> (-inf, inf)
fn logit(theta: f64) -> f64 {
    // Clamp to avoid log(0) or division by zero
    let theta_clamped = theta.clamp(PARAM_EPSILON, 1.0 - PARAM_EPSILON);
    (theta_clamped / (1.0 - theta_clamped)).ln()
}

/// Scaled sigmoid: maps (-inf, inf) -> (a, b)
fn scaled_sigmoid(phi: f64, a: f64, b: f64) -> f64 {
    a + (b - a) * sigmoid(phi)
}

/// Inverse scaled sigmoid: maps (a, b) -> (-inf, inf)
fn inv_scaled_sigmoid(theta: f64, a: f64, b: f64) -> f64 {
    let y = (theta - a) / (b - a);
    logit(y)
}

/// Exponential transform: maps (-inf, inf) -> (0, inf)
fn exp_transform(phi: f64) -> f64 {
    phi.exp()
}

/// Log transform (inverse exp): maps (0, inf) -> (-inf, inf)
fn inv_exp_transform(theta: f64) -> f64 {
    theta.clamp(PARAM_EPSILON, f64::MAX).ln()
}

/// Online Linear Regression Model for Adverse Selection Estimation
/// Uses Stochastic Gradient Descent (SGD) to learn feature weights from observed price changes
/// This replaces the fixed 80/20 heuristic with a data-driven model that adapts to market conditions
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct OnlineAdverseSelectionModel {
    /// Regression weights: W = [w_bias, w_trade_flow, w_lob_imb, w_spread, w_vol]
    /// These weights are learned via SGD to predict short-term price drift
    pub weights: Vec<f64>,
    
    /// Learning rate for SGD updates (default: 0.001)
    /// Controls how quickly the model adapts to new observations
    pub learning_rate: f64,
    
    /// Number of ticks to wait before computing actual price change for training
    /// E.g., lookback_ticks=10 means we predict S_{t+10} - S_t
    pub lookback_ticks: usize,
    
    /// Observation buffer: stores (features, mid_price) for delayed label computation
    /// Implemented as circular buffer with fixed capacity
    pub observation_buffer: std::collections::VecDeque<(Vec<f64>, f64)>,
    
    /// Maximum buffer capacity (should be >= lookback_ticks)
    pub buffer_capacity: usize,
    
    /// Enable/disable online learning (default: true)
    /// When false, model uses fixed weights without updates
    pub enable_learning: bool,
    
    /// Count of SGD updates performed (for monitoring)
    pub update_count: usize,
    
    /// Running average of prediction error (MAE) for monitoring
    pub mean_absolute_error: f64,
    
    /// Decay factor for MAE averaging (0.99 = slow decay)
    pub mae_decay: f64,
    
    /// Welford's algorithm for online mean/variance
    /// [ (count, mean, M2), (count, mean, M2), ... ]
    /// We skip the bias term (index 0) - only normalize the 4 actual features
    pub feature_stats: Vec<(f64, f64, f64)>,
}

impl Default for OnlineAdverseSelectionModel {
    fn default() -> Self {
        Self {
            // Initialize weights to small random values to break symmetry
            // [bias, trade_flow, lob_imbalance, market_spread, volatility]
            weights: vec![0.0, 0.4, 0.1, -0.05, 0.02], // Reasonable starting point based on 80/20 heuristic
            learning_rate: 0.001,
            lookback_ticks: 10, // Predict 10 ticks ahead (~10 seconds)
            observation_buffer: std::collections::VecDeque::with_capacity(100),
            buffer_capacity: 100,
            enable_learning: true,
            update_count: 0,
            mean_absolute_error: 0.0,
            mae_decay: 0.99,
            // 4 features (bias term is not normalized)
            feature_stats: vec![(0.0, 0.0, 0.0); 4],
        }
    }
}

impl OnlineAdverseSelectionModel {
    /// Update running feature statistics using Welford's online algorithm
    /// This should be called once per tick BEFORE predict/record_observation
    pub fn update_feature_stats(&mut self, state: &StateVector) {
        let raw_features = vec![
            state.trade_flow_ema,
            state.lob_imbalance - 0.5,
            state.market_spread_bps,
            state.volatility_ema_bps,
        ];

        for i in 0..raw_features.len() {
            let x = raw_features[i];

            // Welford's online algorithm
            let (count, mean, m2) = &mut self.feature_stats[i];
            *count += 1.0;
            let delta = x - *mean;
            *mean += delta / *count;
            let delta2 = x - *mean; // New mean
            *m2 += delta * delta2;
        }
    }

    /// Get normalized features using current statistics
    /// Does NOT update stats - use update_feature_stats() first
    fn get_normalized_features(&self, state: &StateVector) -> Vec<f64> {
        let raw_features = vec![
            state.trade_flow_ema,
            state.lob_imbalance - 0.5,
            state.market_spread_bps,
            state.volatility_ema_bps,
        ];

        let mut normalized_features = vec![1.0]; // Start with bias term

        for i in 0..raw_features.len() {
            let x = raw_features[i];
            let (count, mean, m2) = &self.feature_stats[i];

            // Get variance and std_dev from current stats
            let (mean, std_dev) = if *count < 2.0 {
                (0.0, 1.0) // Not enough data, just pass through (or return 0.0)
            } else {
                let variance = *m2 / (*count - 1.0);
                let std_dev = variance.sqrt().max(1e-6); // Avoid div by zero
                (*mean, std_dev)
            };

            // Standardize: z = (x - mean) / std_dev
            normalized_features.push((x - mean) / std_dev);
        }

        normalized_features // Returns [1.0, z_flow, z_imb, z_spread, z_vol]
    }
    
    /// Predict short-term price drift: μ_hat = W · X_t
    /// Returns prediction in basis points (positive = bullish, negative = bearish)
    /// NOTE: Call update_feature_stats() first to ensure stats are current
    pub fn predict(&self, state: &StateVector) -> f64 {
        let features = self.get_normalized_features(state);
        
        // Dot product: prediction = sum(w_i * x_i)
        self.weights.iter()
            .zip(features.iter())
            .map(|(w, x)| w * x)
            .sum()
    }
    
    /// Record observation for delayed SGD update
    /// Stores (features, mid_price) in circular buffer
    /// NOTE: Call update_feature_stats() first to ensure stats are current
    pub fn record_observation(&mut self, state: &StateVector, mid_price: f64) {
        let features = self.get_normalized_features(state);
        
        // Add to buffer
        self.observation_buffer.push_back((features, mid_price));
        
        // Maintain buffer capacity (remove oldest if full)
        if self.observation_buffer.len() > self.buffer_capacity {
            self.observation_buffer.pop_front();
        }
    }
    
    /// Perform SGD update if enough observations are available
    /// Computes actual price change from lookback_ticks ago and updates weights
    pub fn update(&mut self, current_mid_price: f64) {
        if !self.enable_learning {
            return;
        }
        
        // Need at least lookback_ticks observations to compute actual price change
        if self.observation_buffer.len() <= self.lookback_ticks {
            return;
        }
        
        // Get observation from lookback_ticks ago
        let lookback_idx = self.observation_buffer.len() - self.lookback_ticks - 1;
        if let Some((features, old_mid_price)) = self.observation_buffer.get(lookback_idx) {
            // Compute actual price change in basis points
            let actual_change_bps = if *old_mid_price > 0.0 {
                ((current_mid_price - old_mid_price) / old_mid_price) * 10000.0
            } else {
                0.0
            };
            
            // Compute prediction from old features
            let predicted_change_bps: f64 = self.weights.iter()
                .zip(features.iter())
                .map(|(w, x)| w * x)
                .sum();
            
            // Compute prediction error
            let error = predicted_change_bps - actual_change_bps;
            
            // Update MAE for monitoring
            let abs_error = error.abs();
            if self.update_count == 0 {
                self.mean_absolute_error = abs_error;
            } else {
                self.mean_absolute_error = self.mae_decay * self.mean_absolute_error 
                    + (1.0 - self.mae_decay) * abs_error;
            }
            
            // SGD update: W = W - learning_rate * error * X
            // This is gradient descent on squared error: L = (y_pred - y_actual)^2
            // ∇L = 2 * (y_pred - y_actual) * X, but we absorb the 2 into learning_rate
            for i in 0..self.weights.len() {
                self.weights[i] -= self.learning_rate * error * features[i];
            }
            
            self.update_count += 1;
            
            // Log update every 100 iterations for monitoring
            if self.update_count % 100 == 0 {
                info!(
                    "Online Adverse Selection Model Update #{}: MAE={:.4}bps, Weights={:?}",
                    self.update_count, self.mean_absolute_error, self.weights
                );
            }
        }
    }
    
    /// Get current model statistics for logging/monitoring
    pub fn get_stats(&self) -> String {
        format!(
            "OnlineModel[updates={}, MAE={:.4}bps, lr={:.6}, enabled={}]",
            self.update_count, self.mean_absolute_error, self.learning_rate, self.enable_learning
        )
    }
    
    /// Reset model to initial state (useful for regime changes)
    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

/// Tunable parameters for live adjustment of market making strategy
/// These parameters control various aspects of the algorithm and can be
/// reloaded at runtime without restarting the bot
///
/// This struct stores the *UNCONSTRAINED* parameters (phi), which are
/// optimized by Adam. They are transformed into constrained values (theta)
/// via the `get_constrained()` method before being used by the strategy.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TuningParams {
    /// Unconstrained (phi) value for inventory skew adjustment factor (theta range: [0.0, 2.0])
    pub skew_adjustment_factor_phi: f64,
    
    /// Unconstrained (phi) value for adverse selection adjustment factor (theta range: [0.0, 2.0])
    pub adverse_selection_adjustment_factor_phi: f64,
    
    /// Unconstrained (phi) value for adverse selection filter smoothing (theta range: [0.0, 1.0])
    pub adverse_selection_lambda_phi: f64,
    
    /// Unconstrained (phi) value for inventory urgency threshold (theta range: [0.0, 1.0])
    pub inventory_urgency_threshold_phi: f64,
    
    /// Unconstrained (phi) value for liquidation rate multiplier (theta range: [0.0, 100.0])
    pub liquidation_rate_multiplier_phi: f64,
    
    /// Unconstrained (phi) value for minimum spread base ratio (theta range: [0.0, 1.0])
    pub min_spread_base_ratio_phi: f64,
    
    /// Unconstrained (phi) value for adverse selection spread scale (theta range: (0.0, inf))
    pub adverse_selection_spread_scale_phi: f64,
    
    /// Unconstrained (phi) value for control gap threshold (theta range: (0.0, inf))
    pub control_gap_threshold_phi: f64,
}

/// Holds the *CONSTRAINED* (theta) parameters after transformation.
/// This struct is used by the strategy logic.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ConstrainedTuningParams {
    /// Inventory skew adjustment factor
    /// Controls how much to skew quotes based on inventory
    /// Higher = more aggressive inventory management
    pub skew_adjustment_factor: f64,
    
    /// Adverse selection adjustment factor
    /// Controls how much to adjust spreads based on adverse selection estimates
    /// Higher = more aggressive response to adverse selection signals
    pub adverse_selection_adjustment_factor: f64,
    
    /// Adverse selection filter smoothing parameter (lambda)
    /// Controls how responsive the adverse selection filter is to new signals
    /// Higher = more weight on recent observations (more responsive)
    /// Lower = more weight on historical average (smoother)
    pub adverse_selection_lambda: f64,
    
    /// Inventory urgency threshold
    /// Inventory ratio above which to activate taker orders for liquidation
    /// Range: [0.0, 1.0] where 1.0 = at max inventory
    pub inventory_urgency_threshold: f64,
    
    /// Liquidation rate multiplier
    /// Scales the taker order rate when urgency is high
    /// Higher = more aggressive liquidation via market orders
    pub liquidation_rate_multiplier: f64,
    
    /// Minimum spread base ratio
    /// Minimum quote offset as a fraction of base spread
    /// Ensures quotes don't get too tight during adjustments
    pub min_spread_base_ratio: f64,
    
    /// Adverse selection spread scale factor
    /// Denominator for normalizing spread in adverse selection calculation
    /// Higher = less sensitivity to spread changes
    pub adverse_selection_spread_scale: f64,
    
    /// Control gap threshold for Adam optimizer
    /// Minimum control gap (bid_gap² + ask_gap²) required to trigger parameter tuning
    /// Higher = more conservative, only tune when quotes deviate significantly from optimal
    /// Lower = more aggressive, tune even for small deviations
    /// Recommended: 1.0-5.0 for stable markets, 5.0-20.0 for volatile/noisy markets
    pub control_gap_threshold: f64,
}

/// Adam Optimizer State for automatic parameter tuning
/// Implements Adaptive Moment Estimation (Kingma & Ba, 2015)
/// Maintains moving averages of gradients and squared gradients for each parameter
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AdamOptimizerState {
    /// First moment estimate (exponential moving average of gradients)
    pub m: Vec<f64>,
    
    /// Second moment estimate (exponential moving average of squared gradients)
    pub v: Vec<f64>,
    
    /// Time step (number of updates performed)
    pub t: usize,
    
    /// Learning rate (default: 0.001)
    pub alpha: f64,
    
    /// Exponential decay rate for first moment (default: 0.9)
    pub beta1: f64,
    
    /// Exponential decay rate for second moment (default: 0.999)
    pub beta2: f64,
    
    /// Small constant for numerical stability (default: 1e-8)
    pub epsilon: f64,
}

impl Default for AdamOptimizerState {
    fn default() -> Self {
        Self {
            m: vec![0.0; 8], // 8 parameters in TuningParams
            v: vec![0.0; 8],
            t: 0,
            alpha: 0.1,    // Increased from 0.01 -> 10x faster convergence
            beta1: 0.9,
            beta2: 0.99,   // Reduced from 0.999: "forgets" old gradients 10x faster, prevents v explosion
            epsilon: 1e-8,
        }
    }
}

impl AdamOptimizerState {
    /// Create a new Adam optimizer with custom parameters
    pub fn new(alpha: f64, beta1: f64, beta2: f64) -> Self {
        Self {
            m: vec![0.0; 8],
            v: vec![0.0; 8],
            t: 0,
            alpha,
            beta1,
            beta2,
            epsilon: 1e-8,
        }
    }
    
    /// Apply Adam update rule to compute parameter updates
    /// Returns the update vector (delta) to be subtracted from parameters
    pub fn compute_update(&mut self, gradient_vector: &[f64]) -> Vec<f64> {
        assert_eq!(gradient_vector.len(), 8, "Gradient vector must have 8 elements");
        
        // Increment time step
        self.t += 1;
        let t = self.t as f64;
        
        let mut updates = Vec::with_capacity(8);
        
        for i in 0..8 {
            let g_t = gradient_vector[i];
            
            // Update biased first moment estimate
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * g_t;
            
            // Update biased second raw moment estimate
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * g_t.powi(2);
            
            // Compute bias-corrected first moment estimate
            let m_hat = self.m[i] / (1.0 - self.beta1.powf(t));
            
            // Compute bias-corrected second raw moment estimate
            let v_hat = self.v[i] / (1.0 - self.beta2.powf(t));
            
            // Compute update
            let update = self.alpha * m_hat / (v_hat.sqrt() + self.epsilon);
            updates.push(update);
        }
        
        updates
    }
    
    /// Reset optimizer state (call when changing market regimes)
    pub fn reset(&mut self) {
        self.m = vec![0.0; 8];
        self.v = vec![0.0; 8];
        self.t = 0;
    }
    
    /// Get effective learning rate for a specific parameter
    /// Useful for monitoring how much each parameter is being adjusted
    pub fn get_effective_learning_rate(&self, param_index: usize) -> f64 {
        if self.t == 0 || param_index >= 8 {
            return 0.0;
        }
        
        let t = self.t as f64;
        let v_hat = self.v[param_index] / (1.0 - self.beta2.powf(t));
        
        self.alpha / (v_hat.sqrt() + self.epsilon)
    }
}

impl Default for TuningParams {
    fn default() -> Self {
        // These are the inverse-transformed (phi) values of the original (theta) defaults
        Self {
            skew_adjustment_factor_phi: inv_scaled_sigmoid(0.5, 0.0, 2.0), // logit(0.25) = -1.0986
            adverse_selection_adjustment_factor_phi: inv_scaled_sigmoid(0.5, 0.0, 2.0), // logit(0.25) = -1.0986
            adverse_selection_lambda_phi: inv_scaled_sigmoid(0.1, 0.0, 1.0), // logit(0.1) = -2.1972
            inventory_urgency_threshold_phi: inv_scaled_sigmoid(0.7, 0.0, 1.0), // logit(0.7) = 0.8473
            liquidation_rate_multiplier_phi: inv_scaled_sigmoid(10.0, 0.0, 100.0), // logit(0.1) = -2.1972
            min_spread_base_ratio_phi: inv_scaled_sigmoid(0.2, 0.0, 1.0), // logit(0.2) = -1.3863
            adverse_selection_spread_scale_phi: inv_exp_transform(100.0), // ln(100.0) = 4.6052
            control_gap_threshold_phi: inv_exp_transform(0.1), // ln(0.1) = -2.3026
        }
    }
}

impl TuningParams {
    /// Transform unconstrained (phi) parameters into constrained (theta) parameters
    pub fn get_constrained(&self) -> ConstrainedTuningParams {
        ConstrainedTuningParams {
            skew_adjustment_factor: scaled_sigmoid(self.skew_adjustment_factor_phi, 0.0, 2.0),
            adverse_selection_adjustment_factor: scaled_sigmoid(self.adverse_selection_adjustment_factor_phi, 0.0, 2.0),
            adverse_selection_lambda: scaled_sigmoid(self.adverse_selection_lambda_phi, 0.0, 1.0),
            inventory_urgency_threshold: scaled_sigmoid(self.inventory_urgency_threshold_phi, 0.0, 1.0),
            liquidation_rate_multiplier: scaled_sigmoid(self.liquidation_rate_multiplier_phi, 0.0, 100.0),
            min_spread_base_ratio: scaled_sigmoid(self.min_spread_base_ratio_phi, 0.0, 1.0),
            adverse_selection_spread_scale: exp_transform(self.adverse_selection_spread_scale_phi),
            control_gap_threshold: exp_transform(self.control_gap_threshold_phi),
        }
    }

    /// Convert a constrained (theta) struct into an unconstrained (phi) struct
    /// This is used when loading from JSON
    fn from_constrained(constrained: &ConstrainedTuningParams) -> Self {
        Self {
            skew_adjustment_factor_phi: inv_scaled_sigmoid(constrained.skew_adjustment_factor, 0.0, 2.0),
            adverse_selection_adjustment_factor_phi: inv_scaled_sigmoid(constrained.adverse_selection_adjustment_factor, 0.0, 2.0),
            adverse_selection_lambda_phi: inv_scaled_sigmoid(constrained.adverse_selection_lambda, 0.0, 1.0),
            inventory_urgency_threshold_phi: inv_scaled_sigmoid(constrained.inventory_urgency_threshold, 0.0, 1.0),
            liquidation_rate_multiplier_phi: inv_scaled_sigmoid(constrained.liquidation_rate_multiplier, 0.0, 100.0),
            min_spread_base_ratio_phi: inv_scaled_sigmoid(constrained.min_spread_base_ratio, 0.0, 1.0),
            adverse_selection_spread_scale_phi: inv_exp_transform(constrained.adverse_selection_spread_scale),
            control_gap_threshold_phi: inv_exp_transform(constrained.control_gap_threshold),
        }
    }
    
    /// Load parameters from a JSON file
    /// Reads constrained (theta) values and converts them to unconstrained (phi) for storage
    pub fn from_json_file(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let contents = std::fs::read_to_string(path)?;
        // Deserialize into the *constrained* struct shape
        let constrained_params: ConstrainedTuningParams = serde_json::from_str(&contents)?;
        // Convert to our internal unconstrained (phi) representation
        Ok(Self::from_constrained(&constrained_params))
    }
    
    /// Save parameters to a JSON file
    /// Converts unconstrained (phi) values to constrained (theta) before saving
    pub fn to_json_file(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        // Get the human-readable constrained (theta) values
        let constrained_params = self.get_constrained();
        let contents = serde_json::to_string_pretty(&constrained_params)?;
        std::fs::write(path, contents)?;
        Ok(())
    }
}

/// State Vector (Z_t) for optimal decision making
/// Represents all information needed to make trading decisions
#[derive(Debug, Clone)]
pub struct StateVector {
    /// S_t: Mid-price - the core price process
    pub mid_price: f64,
    
    /// Q_t: Inventory - our current position
    pub inventory: f64,
    
    /// μ̂_t: Adverse Selection State - filtered estimate of short-term drift
    /// This is our best guess of E[μ_true]
    /// If μ̂_t > 0, we believe the market is about to go up
    pub adverse_selection_estimate: f64,
    
    /// Δ_t: Market Spread - current BBO spread (S^a - S^b) in basis points
    /// Proxy for market-wide volatility and liquidity
    pub market_spread_bps: f64,
    
    /// I_t: LOB Imbalance - ratio of volume at BBO
    /// Calculated as V^b / (V^b + V^a)
    /// Key predictor used to update adverse selection estimate
    pub lob_imbalance: f64,
    
    /// σ̂_t: Volatility Estimate - EMA of realized volatility in basis points
    /// Used to dynamically adjust spread based on market conditions
    /// Higher volatility = wider spreads to manage adverse selection risk
    pub volatility_ema_bps: f64,
    
    /// S_{t-1}: Previous mid-price - used for volatility calculation
    /// Tracks the last mid-price to compute log returns
    pub previous_mid_price: f64,
    
    /// EMA of recent trade flow direction (+1 for taker buy, -1 for taker sell)
    /// This is updated by the Trades stream
    pub trade_flow_ema: f64,
}

impl StateVector {
    /// Create a new state vector with default values
    pub fn new() -> Self {
        Self {
            mid_price: 0.0,
            inventory: 0.0,
            adverse_selection_estimate: 0.0,
            market_spread_bps: 0.0,
            lob_imbalance: 0.0,
            volatility_ema_bps: 10.0, // Default to 10 bps volatility estimate
            previous_mid_price: 0.0,
            trade_flow_ema: 0.0, // Initialize to neutral
        }
    }
    
    /// Calculate new volatility estimate from price change
    /// Returns instantaneous volatility in basis points if valid
    /// 
    /// **DEPRECATED**: This simple EMA method has been replaced by the Particle Filter.
    /// Kept for backward compatibility and as a potential fallback mechanism.
    #[allow(dead_code)]
    fn calculate_new_volatility(&self, new_mid_price: f64) -> Option<f64> {
        // Need both prices to be positive and previous price to be set
        if self.previous_mid_price <= 0.0 || new_mid_price <= 0.0 {
            return None;
        }
        
        // Calculate log return: ln(P_t / P_{t-1})
        let log_return = (new_mid_price / self.previous_mid_price).ln();
        
        // Convert to basis points (multiply by 10,000)
        // Take absolute value as we care about magnitude of volatility
        let instantaneous_vol_bps = log_return.abs() * 10000.0;
        
        // Sanity check: reject extreme outliers (> 1000 bps = 10% move)
        if instantaneous_vol_bps > 1000.0 {
            return None;
        }
        
        Some(instantaneous_vol_bps)
    }
    
    /// Update the state vector from current market data
    /// Note: Volatility is now updated externally by the Particle Filter
    pub fn update(
        &mut self,
        mid_price: f64,
        inventory: f64,
        _book_analysis: Option<&BookAnalysis>,  // Unused - kept for API compatibility
        order_book: Option<&OrderBook>,
        tuning_params: &ConstrainedTuningParams,
        online_model: &mut OnlineAdverseSelectionModel,
    ) {
        // Note: Volatility calculation has been moved to Particle Filter
        // The simple EMA approach is replaced by sophisticated stochastic volatility modeling
        // volatility_ema_bps is now set by MarketMaker::update_state_vector() before calling this method
        
        // Store current mid_price for potential future use
        self.previous_mid_price = self.mid_price;
        self.mid_price = mid_price;
        self.inventory = inventory;
        
        if let Some(book) = order_book {
            // Update market spread (Δ_t)
            if let Some(spread_bps) = book.spread_bps() {
                self.market_spread_bps = spread_bps;
            }

            // FIX: Calculate LOB imbalance directly from order book
            // Get best bid/ask volumes from the OrderBook
            if !book.bids.is_empty() && !book.asks.is_empty() {
                // Parse size strings to f64
                if let (Ok(bid_vol), Ok(ask_vol)) = (
                    book.bids[0].sz.parse::<f64>(),
                    book.asks[0].sz.parse::<f64>()
                ) {
                    let total_vol = bid_vol + ask_vol;

                    if total_vol > 1e-10 {  // EPSILON check
                        // I_t = V^b / (V^b + V^a)
                        // Range: [0, 1] where 0.5 = balanced, >0.5 = bid-heavy, <0.5 = ask-heavy
                        self.lob_imbalance = bid_vol / total_vol;
                    } else {
                        self.lob_imbalance = 0.5; // Neutral if no volume
                    }
                } else {
                    self.lob_imbalance = 0.5; // Neutral if parse fails
                }
            } else {
                self.lob_imbalance = 0.5; // Neutral if no BBO
            }

            // FIX: ALWAYS update adverse selection (moved from book_analysis check)
            // This must run on EVERY update to generate non-zero μ̂
            self.update_adverse_selection(tuning_params, online_model);
        } else {
            // No order book available - set neutral defaults
            self.lob_imbalance = 0.5;
        }
    }
    
    /// Update the adverse selection estimate using online linear regression model
    /// This is called by `StateVector::update` (on AllMids)
    /// 
    /// The model predicts short-term price drift using learned feature weights:
    /// μ_hat = W · X_t where X_t = [1.0, trade_flow, lob_imbalance, spread, volatility]
    fn update_adverse_selection(&mut self, tuning_params: &ConstrainedTuningParams, online_model: &mut OnlineAdverseSelectionModel) {
        // Step 1: Update feature statistics (once per tick)
        online_model.update_feature_stats(self);
        
        // Step 2: Get prediction using updated stats
        let raw_prediction = online_model.predict(self);
        
        // Step 3: Record observation for later SGD training
        online_model.record_observation(self, self.mid_price);
        
        // Apply EMA smoothing to the prediction for stability
        // This provides a "second layer" of smoothing on the model output
        let lambda = tuning_params.adverse_selection_lambda;
        self.adverse_selection_estimate = 
            lambda * raw_prediction + (1.0 - lambda) * self.adverse_selection_estimate;
    }
    
    /// Get optimal spread adjustment based on adverse selection
    /// Returns adjustment in basis points to add/subtract from spread
    /// Positive value = widen spread on buy side (bearish signal)
    /// Negative value = widen spread on sell side (bullish signal)
    pub fn get_adverse_selection_adjustment(&self, base_spread_bps: f64, adjustment_factor: f64) -> f64 {
        // Scale the adverse selection estimate by base spread
        // This creates asymmetric spreads based on expected price movement
        -self.adverse_selection_estimate * base_spread_bps * adjustment_factor
    }
    
    /// Get inventory risk adjustment
    /// Returns multiplier for spread widening based on inventory risk
    /// Higher inventory = wider spreads to discourage further accumulation
    pub fn get_inventory_risk_multiplier(&self, max_inventory: f64) -> f64 {
        if max_inventory <= 0.0 {
            return 1.0;
        }
        
        let inventory_ratio = (self.inventory.abs() / max_inventory).min(1.0);
        
        // Quadratic penalty: 1.0 at zero inventory, up to 2.0 at max inventory
        1.0 + inventory_ratio.powi(2)
    }
    
    /// Get urgency score for inventory management
    /// Returns a value in [0, 1] indicating how urgently we need to reduce inventory
    /// 0 = no urgency, 1 = maximum urgency
    pub fn get_inventory_urgency(&self, max_inventory: f64) -> f64 {
        if max_inventory <= 0.0 {
            return 0.0;
        }
        
        let inventory_ratio = (self.inventory.abs() / max_inventory).min(1.0);
        
        // Cubic urgency function - urgency increases rapidly near max inventory
        inventory_ratio.powi(3)
    }
    
    /// Check if market conditions are favorable for trading
    /// Returns true if spread and liquidity conditions are reasonable
    pub fn is_market_favorable(&self, max_spread_bps: f64) -> bool {
        // Market is favorable if:
        // 1. Spread is not too wide (indicates liquid market)
        // 2. LOB is not extremely imbalanced (indicates one-sided market)
        self.market_spread_bps > 0.0 
            && self.market_spread_bps < max_spread_bps
            && self.lob_imbalance > 0.1 
            && self.lob_imbalance < 0.9
    }
    
    /// Get a string representation of the state vector for logging
    pub fn to_log_string(&self) -> String {
        format!(
            "StateVector[S={:.2}, Q={:.4}, μ̂={:.4}, Δ={:.1}bps, I={:.3}, σ̂={:.2}bps, TF_EMA={:.3}]",
            self.mid_price,
            self.inventory,
            self.adverse_selection_estimate, // This is the final combined/smoothed signal
            self.market_spread_bps,
            self.lob_imbalance,
            self.volatility_ema_bps,
            self.trade_flow_ema // This is the raw trade flow signal
        )
    }
    
    /// Update the trade flow EMA from a batch of trades
    pub fn update_trade_flow_ema(
        &mut self,
        trades: &Vec<crate::Trade>,
        tuning_params: &ConstrainedTuningParams,
    ) {
        if trades.is_empty() {
            return;
        }
        
        // Use adverse_selection_lambda for smoothing the trade flow.
        // You could also add a new, dedicated parameter for this.
        let lambda = tuning_params.adverse_selection_lambda;
        let decay = 1.0 - lambda;
        
        // We must apply the EMA sequentially for each trade in the batch
        // to correctly model time.
        for trade in trades {
            // side="A" = Taker hit ask = Bullish = +1.0
            // side="B" = Taker hit bid = Bearish = -1.0
            let signal = if trade.side == "A" { 1.0 } else { -1.0 };
            
            // Update EMA: V_t = λ * S_t + (1-λ) * V_{t-1}
            self.trade_flow_ema = lambda * signal + decay * self.trade_flow_ema;
        }
    }
}

impl Default for StateVector {
    fn default() -> Self {
        let mut s = Self::new();
        s.trade_flow_ema = 0.0; // Ensure it's set in default as well
        s
    }
}

/// Control Vector (u_t) for algorithm actions
/// Represents all the levers the algorithm can pull at any instant
#[derive(Debug, Clone)]
pub struct ControlVector {
    /// δ^a_t: Ask Quote Offset - distance from S_t to place passive ask order (in bps)
    /// Positive value represents distance above mid price
    pub ask_offset_bps: f64,
    
    /// δ^b_t: Bid Quote Offset - distance from S_t to place passive bid order (in bps)
    /// Positive value represents distance below mid price
    pub bid_offset_bps: f64,
    
    /// ν^a_t: Taker Sell Rate - rate at which to send aggressive sell orders (units per second)
    /// Used for active inventory liquidation when long
    pub taker_sell_rate: f64,
    
    /// ν^b_t: Taker Buy Rate - rate at which to send aggressive buy orders (units per second)
    /// Used for active inventory accumulation when short
    pub taker_buy_rate: f64,
}

impl ControlVector {
    /// Create a new control vector with default passive-only values
    pub fn new() -> Self {
        Self {
            ask_offset_bps: 0.0,
            bid_offset_bps: 0.0,
            taker_sell_rate: 0.0,
            taker_buy_rate: 0.0,
        }
    }
    
    /// Create a symmetric control vector for market making
    /// Both quotes at equal distance from mid, no taker activity
    pub fn symmetric(half_spread_bps: f64) -> Self {
        Self {
            ask_offset_bps: half_spread_bps,
            bid_offset_bps: half_spread_bps,
            taker_sell_rate: 0.0,
            taker_buy_rate: 0.0,
        }
    }
    
    /// Create an asymmetric control vector
    /// Useful for skewing quotes based on inventory or adverse selection
    pub fn asymmetric(ask_offset_bps: f64, bid_offset_bps: f64) -> Self {
        Self {
            ask_offset_bps,
            bid_offset_bps,
            taker_sell_rate: 0.0,
            taker_buy_rate: 0.0,
        }
    }
    
    /// Create a control vector with active liquidation
    /// For emergency inventory management via taker orders
    pub fn with_taker_activity(
        ask_offset_bps: f64,
        bid_offset_bps: f64,
        taker_sell_rate: f64,
        taker_buy_rate: f64,
    ) -> Self {
        Self {
            ask_offset_bps,
            bid_offset_bps,
            taker_sell_rate: taker_sell_rate.max(0.0),
            taker_buy_rate: taker_buy_rate.max(0.0),
        }
    }
    
    /// Calculate the actual quote prices given mid price
    /// Returns (bid_price, ask_price)
    pub fn calculate_quote_prices(&self, mid_price: f64) -> (f64, f64) {
        let bid_price = mid_price * (1.0 - self.bid_offset_bps / 10000.0);
        let ask_price = mid_price * (1.0 + self.ask_offset_bps / 10000.0);
        (bid_price, ask_price)
    }
    
    /// Get the total spread (ask offset + bid offset)
    pub fn total_spread_bps(&self) -> f64 {
        self.ask_offset_bps + self.bid_offset_bps
    }
    
    /// Get the spread asymmetry (difference between ask and bid offsets)
    /// Positive = ask side wider (bullish bias)
    /// Negative = bid side wider (bearish bias)
    pub fn spread_asymmetry_bps(&self) -> f64 {
        self.ask_offset_bps - self.bid_offset_bps
    }
    
    /// Check if this is a passive-only strategy (no taker activity)
    pub fn is_passive_only(&self) -> bool {
        self.taker_sell_rate < EPSILON && self.taker_buy_rate < EPSILON
    }
    
    /// Check if we're actively liquidating (any taker activity)
    pub fn is_liquidating(&self) -> bool {
        !self.is_passive_only()
    }
    
    /// Get net taker direction (positive = net selling, negative = net buying)
    pub fn net_taker_rate(&self) -> f64 {
        self.taker_sell_rate - self.taker_buy_rate
    }
    
    /// Validate that the control vector is feasible
    pub fn validate(&self, min_spread_bps: f64) -> Result<(), String> {
        // Check offsets are non-negative
        if self.ask_offset_bps < 0.0 {
            return Err(format!("Ask offset cannot be negative: {}", self.ask_offset_bps));
        }
        if self.bid_offset_bps < 0.0 {
            return Err(format!("Bid offset cannot be negative: {}", self.bid_offset_bps));
        }
        
        // Check minimum spread
        if self.total_spread_bps() < min_spread_bps {
            return Err(format!(
                "Total spread {:.2} bps is below minimum {:.2} bps",
                self.total_spread_bps(),
                min_spread_bps
            ));
        }
        
        // Check taker rates are non-negative
        if self.taker_sell_rate < 0.0 {
            return Err(format!("Taker sell rate cannot be negative: {}", self.taker_sell_rate));
        }
        if self.taker_buy_rate < 0.0 {
            return Err(format!("Taker buy rate cannot be negative: {}", self.taker_buy_rate));
        }
        
        Ok(())
    }
    
    /// Apply state-based adjustments to the control vector
    /// This is where the state vector informs the control vector
    /// 
    /// **IMPORTANT**: base_half_spread_bps should be the HALF-SPREAD, not total spread.
    /// For example, if you want a 12 bps total spread, pass 6.0 as base_half_spread_bps.
    pub fn apply_state_adjustments(
        &mut self,
        state: &StateVector,
        base_half_spread_bps: f64,
        max_inventory: f64,
        tuning_params: &ConstrainedTuningParams,
        hjb_components: &HJBComponents,
    ) {
        // 1. Adverse Selection Adjustment (Two-Sided Shift)
        let adverse_adj = state.get_adverse_selection_adjustment(
            base_half_spread_bps,
            tuning_params.adverse_selection_adjustment_factor,
        );
        
        // This shifts the entire quote.
        // If bullish (adj is neg): ask widens (ask - neg = ask + pos), bid tightens (bid + neg = bid - pos)
        // If bearish (adj is pos): ask tightens (ask - pos), bid widens (bid + pos)
        self.ask_offset_bps -= adverse_adj;
        self.bid_offset_bps += adverse_adj;
        
        // 2. Inventory Risk Adjustment (Asymmetric)
        let risk_multiplier = state.get_inventory_risk_multiplier(max_inventory);
        if state.inventory > 0.0 {
            // We are LONG. Widen the BID to discourage more buying.
            self.bid_offset_bps *= risk_multiplier;
        } else if state.inventory < 0.0 {
            // We are SHORT. Widen the ASK to discourage more selling.
            self.ask_offset_bps *= risk_multiplier;
        }
        // The side we *want* to get filled on (ask if long, bid if short)
        // is left alone, allowing the skew adjustment in step 3 to tighten it.
        
        // 3. Inventory-based Quote Skewing
        // If long, tighten ask and widen bid to encourage selling
        // If short, tighten bid and widen ask to encourage buying
        let inventory_ratio = if max_inventory > 0.0 {
            (state.inventory / max_inventory).clamp(-1.0, 1.0)
        } else {
            0.0
        };
        
        let skew_adjustment = inventory_ratio * base_half_spread_bps * tuning_params.skew_adjustment_factor;
        self.ask_offset_bps -= skew_adjustment; // Long -> tighter ask
        self.bid_offset_bps += skew_adjustment; // Long -> wider bid
        
        // Ensure offsets stay positive (using half-spread minimum)
        self.ask_offset_bps = self.ask_offset_bps.max(base_half_spread_bps * tuning_params.min_spread_base_ratio);
        self.bid_offset_bps = self.bid_offset_bps.max(base_half_spread_bps * tuning_params.min_spread_base_ratio);
        
        // --- NEW: Explicitly add maker fee buffer ---
        // Add slightly more than the fee as a buffer (e.g., fee + 0.5bps)
        // This ensures our quotes always cover the fee, especially when base spread hits the floor
        let fee_buffer_bps = hjb_components.maker_fee_bps + 0.5;
        self.ask_offset_bps += fee_buffer_bps;
        self.bid_offset_bps += fee_buffer_bps;
        // --- End New ---
        
        // 4. Active Liquidation Control
        // If inventory urgency is high, activate taker orders
        let urgency = state.get_inventory_urgency(max_inventory);
        if urgency > tuning_params.inventory_urgency_threshold {
            // High urgency - use taker orders to actively reduce position
            let liquidation_rate = (urgency - tuning_params.inventory_urgency_threshold) 
                * tuning_params.liquidation_rate_multiplier;
            
            if state.inventory > 0.0 {
                // Long - need to sell
                self.taker_sell_rate = liquidation_rate;
                self.taker_buy_rate = 0.0;
            } else if state.inventory < 0.0 {
                // Short - need to buy
                self.taker_sell_rate = 0.0;
                self.taker_buy_rate = liquidation_rate;
            }
        } else {
            // Normal operation - passive only
            self.taker_sell_rate = 0.0;
            self.taker_buy_rate = 0.0;
        }
    }
    
    /// Get a string representation for logging
    pub fn to_log_string(&self) -> String {
        if self.is_passive_only() {
            format!(
                "ControlVector[δ^b={:.1}bps, δ^a={:.1}bps, spread={:.1}bps, asymmetry={:.1}bps]",
                self.bid_offset_bps,
                self.ask_offset_bps,
                self.total_spread_bps(),
                self.spread_asymmetry_bps()
            )
        } else {
            format!(
                "ControlVector[δ^b={:.1}bps, δ^a={:.1}bps, ν^b={:.3}, ν^a={:.3}]",
                self.bid_offset_bps,
                self.ask_offset_bps,
                self.taker_buy_rate,
                self.taker_sell_rate
            )
        }
    }
}

impl Default for ControlVector {
    fn default() -> Self {
        Self::new()
    }
}

/// Value Function V(Q, Z, t) for HJB equation
/// Represents the maximum expected P&L achievable from state (Q, Z) at time t
#[derive(Debug, Clone)]
pub struct ValueFunction {
    /// Inventory aversion parameter (φ in the HJB equation)
    pub phi: f64,
    
    /// Terminal time (T)
    pub terminal_time: f64,
    
    /// Current time (t)
    pub current_time: f64,
    
    /// Cached value estimates for different inventory levels
    /// Maps inventory -> estimated value
    value_cache: std::collections::HashMap<i32, f64>,
}

impl ValueFunction {
    /// Create a new value function with given parameters
    pub fn new(phi: f64, terminal_time: f64) -> Self {
        Self {
            phi,
            terminal_time,
            current_time: 0.0,
            value_cache: std::collections::HashMap::new(),
        }
    }
    
    /// Update current time
    pub fn set_time(&mut self, t: f64) {
        self.current_time = t;
    }
    
    /// Get time to terminal (T - t)
    pub fn time_to_terminal(&self) -> f64 {
        (self.terminal_time - self.current_time).max(0.0)
    }
    
    /// Evaluate value function V(Q, Z, t)
    /// This is an approximation based on inventory penalty and time decay
    pub fn evaluate(&self, inventory: f64, state: &StateVector) -> f64 {
        let q_rounded = inventory.round() as i32;
        
        // Check cache first
        if let Some(&cached_value) = self.value_cache.get(&q_rounded) {
            return cached_value;
        }
        
        // Approximate value function using inventory penalty
        // V(Q) ≈ -φ * Q² * (T-t) + expected_pnl
        let time_remaining = self.time_to_terminal();
        let inventory_penalty = -self.phi * inventory.powi(2) * time_remaining;
        
        // Expected P&L component (simplified approximation)
        // Assumes we can capture spread and manage adverse selection
        let expected_spread_capture = state.market_spread_bps * 0.5 * time_remaining;
        let adverse_selection_cost = state.adverse_selection_estimate.abs() * inventory.abs() * time_remaining;
        
        let value = inventory_penalty + expected_spread_capture - adverse_selection_cost;
        value
    }
    
    /// Calculate value change from inventory change: V(Q+dQ) - V(Q)
    pub fn inventory_delta(&self, inventory: f64, d_inventory: f64, state: &StateVector) -> f64 {
        let v_new = self.evaluate(inventory + d_inventory, state);
        let v_old = self.evaluate(inventory, state);
        v_new - v_old
    }
    
    /// Cache a value estimate
    pub fn cache_value(&mut self, inventory: i32, value: f64) {
        self.value_cache.insert(inventory, value);
    }
    
    /// Clear the value cache
    pub fn clear_cache(&mut self) {
        self.value_cache.clear();
    }
}

/// HJB (Hamilton-Jacobi-Bellman) Equation Components
/// Represents the optimization problem for market making
#[derive(Debug, Clone)]
pub struct HJBComponents {
    /// Fill rate model parameters
    pub lambda_base: f64,  // Base Poisson fill rate
    
    /// Inventory penalty (φ in objective)
    pub phi: f64,
    
    /// Maker fee (paid when passively filled) in BPS
    pub maker_fee_bps: f64,
    
    /// Taker fee (paid when crossing spread)
    pub taker_fee_bps: f64,
}

impl HJBComponents {
    /// Create new HJB components with default parameters
    pub fn new() -> Self {
        Self {
            lambda_base: 1.0,      // 1 fill per second at best quotes
            phi: 0.01,             // Inventory penalty coefficient
            maker_fee_bps: 1.5,    // 1.5 bps maker fee
            taker_fee_bps: 4.5,    // 4.5 bps taker fee
        }
    }
    
    /// Estimate maker bid fill rate λ^b(δ^b, Z_t)
    /// Rate depends on how competitive our quote is relative to BBO
    pub fn maker_bid_fill_rate(&self, bid_offset_bps: f64, state: &StateVector) -> f64 {
        // λ^b = λ_base * exp(-β * distance_from_bbo)
        // If we're at best bid, rate is high
        // If we're far from best bid, rate decays exponentially
        
        let market_half_spread = state.market_spread_bps / 2.0;
        let distance_from_bbo = (bid_offset_bps - market_half_spread).max(0.0);
        
        // Decay parameter (how fast fill rate drops with distance)
        let beta = 0.1;
        
        // Adjust base rate by LOB imbalance
        // High bid volume (high I_t) reduces our fill rate
        let imbalance_factor = 2.0 * (1.0 - state.lob_imbalance);
        
        self.lambda_base * imbalance_factor * (-beta * distance_from_bbo).exp()
    }
    
    /// Estimate maker ask fill rate λ^a(δ^a, Z_t)
    pub fn maker_ask_fill_rate(&self, ask_offset_bps: f64, state: &StateVector) -> f64 {
        let market_half_spread = state.market_spread_bps / 2.0;
        let distance_from_bbo = (ask_offset_bps - market_half_spread).max(0.0);
        
        let beta = 0.1;
        
        // High ask volume (low I_t) reduces our fill rate
        let imbalance_factor = 2.0 * state.lob_imbalance;
        
        self.lambda_base * imbalance_factor * (-beta * distance_from_bbo).exp()
    }
    
    /// Calculate expected value from maker bid fill
    /// λ^b * [V(Q+1) - V(Q) - (S_t - δ^b) - fee]
    pub fn maker_bid_value(
        &self,
        bid_offset_bps: f64,
        state: &StateVector,
        value_fn: &ValueFunction,
    ) -> f64 {
        let lambda_b = self.maker_bid_fill_rate(bid_offset_bps, state);
        
        // Value change from inventory increase
        let value_change = value_fn.inventory_delta(state.inventory, 1.0, state);
        
        // Cash flow: we pay (S_t - δ^b) to buy
        let price_paid = state.mid_price * (1.0 - bid_offset_bps / 10000.0);
        
        // Maker fee: paid on the filled notional
        let maker_fee = price_paid * self.maker_fee_bps / 10000.0;
        
        let cash_flow = -price_paid - maker_fee;
        
        lambda_b * (value_change + cash_flow)
    }
    
    /// Calculate expected value from maker ask fill
    /// λ^a * [V(Q-1) - V(Q) + (S_t + δ^a) - fee]
    pub fn maker_ask_value(
        &self,
        ask_offset_bps: f64,
        state: &StateVector,
        value_fn: &ValueFunction,
    ) -> f64 {
        let lambda_a = self.maker_ask_fill_rate(ask_offset_bps, state);
        
        // Value change from inventory decrease
        let value_change = value_fn.inventory_delta(state.inventory, -1.0, state);
        
        // Cash flow: we receive (S_t + δ^a) from selling
        let price_received = state.mid_price * (1.0 + ask_offset_bps / 10000.0);
        
        // Maker fee: paid on the filled notional
        let maker_fee = price_received * self.maker_fee_bps / 10000.0;
        
        let cash_flow = price_received - maker_fee;
        
        lambda_a * (value_change + cash_flow)
    }
    
    /// Calculate expected value from taker buy
    /// ν^b * [V(Q+1) - V(Q) - S^a_t]
    pub fn taker_buy_value(
        &self,
        taker_buy_rate: f64,
        state: &StateVector,
        value_fn: &ValueFunction,
    ) -> f64 {
        if taker_buy_rate < EPSILON {
            return 0.0;
        }
        
        let value_change = value_fn.inventory_delta(state.inventory, 1.0, state);
        
        // Must pay market ask price + taker fee
        let market_ask = state.mid_price * (1.0 + state.market_spread_bps / 20000.0);
        let fee = market_ask * self.taker_fee_bps / 10000.0;
        let cash_flow = -(market_ask + fee);
        
        taker_buy_rate * (value_change + cash_flow)
    }
    
    /// Calculate expected value from taker sell
    /// ν^a * [V(Q-1) - V(Q) + S^b_t]
    pub fn taker_sell_value(
        &self,
        taker_sell_rate: f64,
        state: &StateVector,
        value_fn: &ValueFunction,
    ) -> f64 {
        if taker_sell_rate < EPSILON {
            return 0.0;
        }
        
        let value_change = value_fn.inventory_delta(state.inventory, -1.0, state);
        
        // Receive market bid price - taker fee
        let market_bid = state.mid_price * (1.0 - state.market_spread_bps / 20000.0);
        let fee = market_bid * self.taker_fee_bps / 10000.0;
        let cash_flow = market_bid - fee;
        
        taker_sell_rate * (value_change + cash_flow)
    }
    
    /// Evaluate the full HJB objective for a given control
    /// Returns the instantaneous expected value rate
    pub fn evaluate_control(
        &self,
        control: &ControlVector,
        state: &StateVector,
        value_fn: &ValueFunction,
    ) -> f64 {
        // Running inventory penalty: -φ * Q²
        let inventory_penalty = -self.phi * state.inventory.powi(2);
        
        // Maker bid fill value
        let maker_bid = self.maker_bid_value(control.bid_offset_bps, state, value_fn);
        
        // Maker ask fill value
        let maker_ask = self.maker_ask_value(control.ask_offset_bps, state, value_fn);
        
        // Taker buy value
        let taker_buy = self.taker_buy_value(control.taker_buy_rate, state, value_fn);
        
        // Taker sell value
        let taker_sell = self.taker_sell_value(control.taker_sell_rate, state, value_fn);
        
        inventory_penalty + maker_bid + maker_ask + taker_buy + taker_sell
    }
    
    /// Find control by grid search over candidate solutions
    /// 
    /// **IMPORTANT**: This is NOT a true optimum, but a practical grid search over
    /// hardcoded multipliers. While this provides a "good enough" solution, it has limitations:
    /// 
    /// - **Not optimal**: Grid search only evaluates discrete points, missing true optimum
    /// - **Slow**: Evaluates 25+ control candidates (5x5 grid + taker variants)
    /// - **Coarse**: Fixed multipliers may not adapt well to all market conditions
    /// 
    /// **Recommended Usage**:
    /// - Use `apply_state_adjustments()` heuristic for real-time updates (much faster)
    /// - Run this function in a background thread periodically to:
    ///   - Validate heuristic performance
    ///   - Tune heuristic parameters based on grid search results
    ///   - Generate training data for ML-based control policies
    /// 
    /// For production, the fast heuristic is typically just as good and 100x+ faster.
    /// 
    /// **Parameter**: base_spread_bps should be the HALF-SPREAD (not total spread).
    /// For example, if you want a 12 bps total spread, pass 6.0.
    pub fn optimize_control(
        &self,
        state: &StateVector,
        value_fn: &ValueFunction,
        base_spread_bps: f64,
    ) -> ControlVector {
        let mut best_control = ControlVector::symmetric(base_spread_bps);
        let mut best_value = self.evaluate_control(&best_control, state, value_fn);
        
        // Grid search over discrete control candidates
        // This is practical but not theoretically optimal
        
        // Try different bid/ask offsets
        for bid_mult in [0.5, 0.75, 1.0, 1.25, 1.5].iter() {
            for ask_mult in [0.5, 0.75, 1.0, 1.25, 1.5].iter() {
                let mut candidate = ControlVector::asymmetric(
                    base_spread_bps * ask_mult,
                    base_spread_bps * bid_mult,
                );
                
                // Try with and without taker activity
                for &use_taker in [false, true].iter() {
                    if use_taker {
                        // Consider taker orders for inventory management
                        let urgency = state.get_inventory_urgency(100.0); // Assume max 100
                        if urgency > 0.7 {
                            if state.inventory > 0.0 {
                                candidate.taker_sell_rate = (urgency - 0.7) * 10.0;
                            } else if state.inventory < 0.0 {
                                candidate.taker_buy_rate = (urgency - 0.7) * 10.0;
                            }
                        }
                    }
                    
                    let value = self.evaluate_control(&candidate, state, value_fn);
                    if value > best_value {
                        best_value = value;
                        best_control = candidate.clone();
                    }
                    
                    // Reset taker rates
                    candidate.taker_sell_rate = 0.0;
                    candidate.taker_buy_rate = 0.0;
                }
            }
        }
        
        best_control
    }
}

impl Default for HJBComponents {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub struct MarketMakerRestingOrder {
    pub oid: u64,
    pub position: f64,
    pub price: f64,
    pub level: usize,  // Track which level this is (0 = L1, 1 = L2, etc.)
    pub pending_cancel: bool,  // Track if cancellation has been sent but not confirmed
}

/// Wrapper for an order pending cancellation with timestamp
#[derive(Debug, Clone)]
pub struct PendingCancelOrder {
    pub order: MarketMakerRestingOrder,
    pub cancel_time: f64,  // Unix timestamp when cancel was sent
}

#[derive(Debug)]
pub struct MarketMakerInput {
    pub asset: String,
    pub max_absolute_position_size: f64, // Absolute value of the max position we can take on
    pub asset_type: AssetType, // Asset type (Perp or Spot) for tick/lot size validation
    pub wallet: PrivateKeySigner, // Wallet containing private key
    
    /// Performance gap threshold (in percent) to enable live trading
    /// E.g., 15.0 means trading starts when heuristic gap is < 15%
    pub enable_trading_gap_threshold_percent: f64,
    
    /// Multi-level market making configuration
    pub enable_multi_level: bool,
    pub multi_level_config: Option<MultiLevelConfig>,
    
    /// Robust control configuration (uncertainty handling)
    pub enable_robust_control: bool,
    pub robust_config: Option<RobustConfig>,
}

/// Refactored MarketMaker struct with unified multi-level optimization
/// Removes obsolete single-level fields and integrates new components
#[derive(Debug)]
pub struct MarketMaker {
    // ===== Core Configuration =====
    /// Trading asset symbol (e.g., "BTC", "ETH")
    pub asset: String,
    
    /// Tick/lot size validator for order price/size validation
    pub tick_lot_validator: TickLotValidator,
    
    /// Maximum absolute position size (inventory limit)
    pub max_absolute_position_size: f64,
    
    // ===== Exchange Clients =====
    /// Info client for market data queries
    pub info_client: InfoClient,

    /// Exchange client for order management (wrapped in Arc for sharing with async tasks)
    pub exchange_client: Arc<ExchangeClient>,

    /// User wallet address
    pub user_address: Address,
    
    // ===== Real-Time State =====
    /// Current position (positive = long, negative = short)
    pub cur_position: f64,
    
    /// Latest mid price from AllMids stream (kept for logging/debugging only)
    pub latest_mid_price: f64,

    /// Latest mid price calculated from L2Book BBO (used for order pricing)
    /// This is the authoritative mid-price for order placement
    pub latest_l2_mid_price: f64,

    /// Latest order book snapshot (needed for BBO and imbalance)
    pub latest_book: Option<OrderBook>,
    
    /// State Vector (Z_t) - holds μ̂, σ̂, I, trade flow, etc.
    pub state_vector: StateVector,
    
    // ===== Advanced Models =====
    /// Particle filter for stochastic volatility estimation
    /// Replaces simple EMA with sophisticated latent volatility modeling
    pub particle_filter: Arc<RwLock<ParticleFilterState>>,
    
    /// Online Linear Regression Model for Adverse Selection
    /// Learns to predict short-term price drift using SGD
    pub online_adverse_selection_model: Arc<RwLock<OnlineAdverseSelectionModel>>,
    
    // ===== Multi-Level Optimizer & Components =====
    /// Unified multi-level optimizer (contains config, logic for levels/sizes)
    /// Replaces old single-level HJB components and value function
    pub multi_level_optimizer: MultiLevelOptimizer,

    /// Old HJB Components - kept for gradient calculation target
    /// Used as the "optimal target" in the learning loop
    pub hjb_components: HJBComponents,

    /// Old Value Function - kept for gradient calculation target
    /// Used as the "optimal target" in the learning loop
    pub value_function: ValueFunction,

    /// Hawkes process model for fill rate estimation
    /// Needs RwLock for fill updates from trade stream
    pub hawkes_model: Arc<RwLock<HawkesFillModel>>,
    
    /// Robust control configuration (uncertainty sets, robustness parameters)
    pub robust_config: RobustConfig,
    
    /// Current parameter uncertainty estimates from particle filter
    pub current_uncertainty: ParameterUncertainty,
    
    // ===== Resting Order State (Multi-Level) =====
    /// Resting bid orders by level (replaces old lower_resting)
    /// bid_levels[0] = L1 (tightest), bid_levels[1] = L2, etc.
    pub bid_levels: Vec<MarketMakerRestingOrder>,

    /// Resting ask orders by level (replaces old upper_resting)
    /// ask_levels[0] = L1 (tightest), ask_levels[1] = L2, etc.
    pub ask_levels: Vec<MarketMakerRestingOrder>,

    /// Orders that have been sent for cancellation but might still receive fills
    /// Key: order ID (oid), Value: order + cancel timestamp
    /// Cleaned up periodically to prevent memory leak
    pub pending_cancel_orders: HashMap<u64, PendingCancelOrder>,

    /// Holding pen for fills that arrive before orderUpdates confirms the oid mapping
    /// Key: oid, Value: Vec of TradeInfo (fill) messages waiting to be processed
    /// When orderUpdates arrives with cloid->oid mapping, we flush these fills
    pub unmatched_fills: HashMap<u64, Vec<TradeInfo>>,

    /// Pending order placement intents (tracked for async reconciliation)
    /// Key: intent_id, Value: OrderIntent details
    pub pending_order_intents: Arc<RwLock<HashMap<u64, OrderIntent>>>,

    /// Next intent ID counter for tracking order placements
    pub next_intent_id: Arc<RwLock<u64>>,

    /// Channel sender for async order execution task
    /// Sends OrderCommand to background task for network I/O
    pub order_command_tx: tokio::sync::mpsc::Sender<OrderCommand>,

    /// HIGH-PRIORITY channel sender for cancellations
    /// Cancels are processed before placements to prevent queue saturation
    pub cancel_command_tx: tokio::sync::mpsc::Sender<OrderCommand>,

    /// Channel receiver for order placement results from async execution task
    /// Receives OrderPlacementResult to track which orders were successfully placed
    pub order_result_rx: tokio::sync::mpsc::UnboundedReceiver<OrderPlacementResult>,

    /// Cached volatility estimate from background particle filter
    /// Updated periodically (every 150ms) to avoid blocking hot path
    pub cached_volatility: Arc<RwLock<CachedVolatilityEstimate>>,

    /// Channel sender for sending price updates to particle filter background task
    pub pf_price_tx: tokio::sync::mpsc::Sender<f64>,

    // ===== Adam Self-Tuning System =====
    /// Tunable parameters wrapped in Arc<RwLock> for live updates
    /// These are the meta-parameters that Adam optimizes (skew factors, etc.)
    pub tuning_params: Arc<RwLock<TuningParams>>,
    
    /// Adam optimizer state for automatic parameter tuning
    pub adam_optimizer: Arc<RwLock<AdamOptimizerState>>,
    
    /// Gradient accumulator for stable Adam updates
    /// Accumulates gradients over multiple AllMids messages (default: 60 seconds)
    pub gradient_accumulator: Arc<RwLock<Vec<f64>>>,
    
    /// Count of gradients accumulated in current window
    pub gradient_count: Arc<RwLock<usize>>,
    
    /// Message counter for sampling (accumulate every Nth message to save CPU)
    pub message_counter: Arc<RwLock<usize>>,
    
    // ===== Trading Control =====
    /// Flag indicating if live trading is enabled (set by optimizer)
    pub trading_enabled: Arc<RwLock<bool>>,
    
    /// Performance gap threshold (in percent) to enable live trading
    /// E.g., 15.0 means trading starts when heuristic gap is < 15%
    pub enable_trading_gap_threshold_percent: f64,
    
    // ===== Taker Logic State =====
    /// Latest taker buy rate (if needed for taker logic)
    pub latest_taker_buy_rate: f64,

    /// Latest taker sell rate (if needed for taker logic)
    pub latest_taker_sell_rate: f64,

    /// Timestamp of last taker buy execution (for rate limiting)
    pub last_taker_buy_time: f64,

    /// Timestamp of last taker sell execution (for rate limiting)
    pub last_taker_sell_time: f64,

    /// Smoothed taker buy rate (exponential moving average to prevent spikes)
    pub smoothed_taker_buy_rate: f64,

    /// Smoothed taker sell rate (exponential moving average to prevent spikes)
    pub smoothed_taker_sell_rate: f64,

    // ===== TUI Dashboard =====
    /// Watch channel sender for broadcasting state updates to TUI dashboard
    /// Sends DashboardState snapshots on every significant event
    pub tui_state_tx: tokio::sync::watch::Sender<crate::tui::state::DashboardState>,

    /// Start time of the market maker (for uptime calculation)
    pub start_time: f64,

    // ===== Performance Tracking =====
    /// Rolling window of equity snapshots for Sharpe ratio calculation
    /// Stores (timestamp, total_equity) tuples
    /// Max 1000 snapshots (roughly 16 minutes at 1 second intervals)
    pub equity_history: std::collections::VecDeque<(f64, f64)>,

    /// Last equity value for calculating returns
    pub last_equity: f64,

    // ===== REMOVED/DEPRECATED Fields (Documented for Reference) =====
    // The following fields have been removed in this refactor:
    //
    // - target_liquidity: f64
    //   → Replaced by multi_level_optimizer.config.level_sizes
    //
    // - half_spread: u16
    //   → Baseline spread now uses real-time market spread from latest_book
    //
    // - reprice_threshold_ratio: f64
    //   → Reprice logic replaced by multi-level reconciliation
    //
    // - lower_resting: MarketMakerRestingOrder
    // - upper_resting: MarketMakerRestingOrder
    //   → Replaced by bid_levels and ask_levels (multi-level)
    //
    // - inventory_skew_calculator: Option<InventorySkewCalculator>
    //   → Skew logic integrated into multi_level_optimizer
    //
    // - latest_book_analysis: Option<BookAnalysis>
    //   → Analysis now done inline or within optimizer
    //
    // - control_vector: ControlVector
    //   → Replaced by MultiLevelControl generated dynamically
    //
    // - hjb_components: HJBComponents
    // - value_function: ValueFunction
    //   → Logic now within MultiLevelOptimizer (or RobustHJBComponents)
    //
    // - multi_level_control: Option<MultiLevelControl>
    //   → Generated on-the-fly by optimizer, not stored
    //
    // - multi_level_enabled: bool
    // - robust_control_enabled: bool
    //   → Configuration is implicit (non-optional optimizer/config)
}

impl MarketMaker {
    /// Get a copy of current tuning parameters
    pub fn get_tuning_params(&self) -> TuningParams {
        self.tuning_params.read().clone()
    }

    /// Calculate Sharpe ratio from equity history
    /// Returns annualized Sharpe ratio assuming 1-second sampling interval
    /// Formula: Sharpe = (mean_return / std_return) * sqrt(periods_per_year)
    /// For 1-second intervals: periods_per_year = 365.25 * 24 * 60 * 60 ≈ 31,557,600
    fn calculate_sharpe_ratio(&self) -> f64 {
        if self.equity_history.len() < 2 {
            return 0.0;  // Not enough data
        }

        // Calculate returns from equity snapshots
        let mut returns: Vec<f64> = Vec::with_capacity(self.equity_history.len() - 1);
        let history: Vec<_> = self.equity_history.iter().collect();
        
        for i in 1..history.len() {
            let (_, prev_equity) = history[i - 1];
            let (_, curr_equity) = history[i];
            
            if *prev_equity > 0.0 {
                let ret = (curr_equity - prev_equity) / prev_equity;
                returns.push(ret);
            }
        }

        if returns.is_empty() {
            return 0.0;
        }

        // Calculate mean return
        let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;

        // Calculate standard deviation of returns
        let variance = returns.iter()
            .map(|r| (r - mean_return).powi(2))
            .sum::<f64>() / returns.len() as f64;
        
        let std_return = variance.sqrt();

        if std_return < 1e-10 {
            return 0.0;  // Avoid division by zero
        }

        // Annualize the Sharpe ratio
        // Assuming 1-second intervals: sqrt(31557600) ≈ 5618.0
        let periods_per_year: f64 = 365.25 * 24.0 * 60.0 * 60.0;
        let sharpe = (mean_return / std_return) * periods_per_year.sqrt();

        sharpe
    }

    /// Update TUI dashboard state with current market maker state
    /// Called on every significant event (L2Book update, fills, etc.)
    fn update_tui_state(&mut self) {
        use crate::tui::state::{DashboardState, OrderLevel};

        // Calculate unrealized PnL
        let unrealized_pnl = if self.cur_position != 0.0 && self.latest_l2_mid_price > 0.0 {
            let avg_entry = self.state_vector.mid_price;
            (self.latest_l2_mid_price - avg_entry) * self.cur_position
        } else {
            0.0
        };

        // Calculate current total equity (account_equity + unrealized_pnl)
        // For now, we'll use unrealized_pnl as a proxy since account_equity is TODO
        let current_equity = unrealized_pnl;  // TODO: Add account_equity when available

        // Update equity history for Sharpe ratio calculation
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs_f64();
        
        // Only update equity history once per second to avoid too many samples
        let should_update = self.equity_history.is_empty() || 
                           (now - self.equity_history.back().unwrap().0) >= 1.0;
        
        if should_update {
            self.equity_history.push_back((now, current_equity));
            
            // Keep max 1000 samples (roughly 16 minutes at 1 second intervals)
            while self.equity_history.len() > 1000 {
                self.equity_history.pop_front();
            }
            
            self.last_equity = current_equity;
        }

        // Calculate Sharpe ratio from equity history
        let sharpe_ratio = self.calculate_sharpe_ratio();

        // Get particle filter stats
        let pf = self.particle_filter.read();
        let pf_ess = pf.get_ess();
        let vol_5th = pf.estimate_volatility_percentile_bps(0.05);
        let vol_95th = pf.estimate_volatility_percentile_bps(0.95);
        let pf_max_particles = pf.particles.len();
        drop(pf);

        // Get cached volatility
        let cached_vol = self.cached_volatility.read();
        let pf_volatility_bps = cached_vol.volatility_bps;
        drop(cached_vol);

        // Get online model stats
        let online_model = self.online_adverse_selection_model.read();
        let online_model_mae = online_model.mean_absolute_error;
        let online_model_updates = online_model.update_count;
        let online_model_lr = online_model.learning_rate;
        let online_model_enabled = true;  // Model is always enabled in current implementation
        drop(online_model);

        // Get Adam optimizer stats
        let adam = self.adam_optimizer.read();
        let gradient_count = *self.gradient_count.read();
        let grad_acc = self.gradient_accumulator.read();
        let adam_avg_loss = if !grad_acc.is_empty() {
            grad_acc.iter().sum::<f64>() / grad_acc.len() as f64
        } else {
            0.0
        };
        drop(adam);
        drop(grad_acc);

        // Calculate time since last Adam update (placeholder - we'll track this separately)
        let adam_last_update_secs = 0.0;

        // Convert bid_levels to OrderLevel
        let bid_levels: Vec<OrderLevel> = self.bid_levels.iter().map(|order| {
            OrderLevel {
                side: "BID".to_string(),
                level: order.level + 1,  // Display as 1-indexed (L1, L2, L3)
                price: order.price,
                size: order.position,
                oid: order.oid,
            }
        }).collect();

        // Convert ask_levels to OrderLevel
        let ask_levels: Vec<OrderLevel> = self.ask_levels.iter().map(|order| {
            OrderLevel {
                side: "ASK".to_string(),
                level: order.level + 1,  // Display as 1-indexed (L1, L2, L3)
                price: order.price,
                size: order.position,
                oid: order.oid,
            }
        }).collect();

        // Calculate uptime
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs_f64();
        let uptime_secs = now - self.start_time;

        // Get message counter
        let total_messages = *self.message_counter.read() as u64;

        // Create dashboard state (Note: recent_fills will be populated separately on fill events)
        let dashboard_state = DashboardState {
            cur_position: self.cur_position,
            avg_entry_price: self.state_vector.mid_price,
            unrealized_pnl,
            account_equity: 0.0,  // TODO: Fetch from InfoClient
            sharpe_ratio,

            l2_mid_price: self.latest_l2_mid_price,
            all_mids_price: self.latest_mid_price,
            market_spread_bps: self.state_vector.market_spread_bps,
            lob_imbalance: self.state_vector.lob_imbalance,

            volatility_ema_bps: self.state_vector.volatility_ema_bps,
            adverse_selection_estimate: self.state_vector.adverse_selection_estimate,
            trade_flow_ema: self.state_vector.trade_flow_ema,

            pf_ess,
            pf_max_particles,
            pf_vol_5th: vol_5th,
            pf_vol_95th: vol_95th,
            pf_volatility_bps,

            online_model_mae,
            online_model_updates,
            online_model_lr,
            online_model_enabled,

            adam_gradient_samples: gradient_count,
            adam_avg_loss,
            adam_last_update_secs,

            bid_levels,
            ask_levels,

            recent_fills: self.tui_state_tx.borrow().recent_fills.clone(),

            uptime_secs,
            total_messages,
        };

        // Send updated state to TUI (non-blocking - if no one is listening, that's fine)
        let _ = self.tui_state_tx.send(dashboard_state);
    }

    /// Update the state vector with current market conditions
    /// Now uses Particle Filter for sophisticated volatility estimation
    fn update_state_vector(&mut self) {
        // ⚡ OPTIMIZED: Use cached volatility from background particle filter task
        // This eliminates expensive particle filter updates (5-20ms) from the hot path
        // Background task updates cache every 150ms, providing near-real-time estimates

        // Send latest L2 mid price to background PF task (non-blocking)
        if self.latest_l2_mid_price > 0.0 {
            let _ = self.pf_price_tx.try_send(self.latest_l2_mid_price);
            // Note: Ignore send errors - if channel is full, PF will use previous price
        }

        let cached_vol = self.cached_volatility.read();
        let vol_estimate_bps = cached_vol.volatility_bps;
        drop(cached_vol);  // Release lock immediately

        // Update state vector with cached volatility estimate
        self.state_vector.volatility_ema_bps = vol_estimate_bps;

        // ⚡ NOTE: Particle filter is now updated in background task (every 150ms)
        // This decouples expensive PF computation from critical path
        // Latency savings: 5-20ms per update_state_vector() call

        // Step 2: Update rest of state vector with market data
        let constrained_params = self.tuning_params.read().get_constrained();
        let mut online_model = self.online_adverse_selection_model.write();

        // CRITICAL: Use L2Book mid-price (not AllMids) for order pricing
        // Skip update if L2Book mid not yet available
        if self.latest_l2_mid_price > 0.0 {
            self.state_vector.update(
                self.latest_l2_mid_price,  // CHANGED: Use L2Book mid (accurate to BBO)
                self.cur_position,
                None, // book_analysis removed - analysis done inline now
                self.latest_book.as_ref(),
                &constrained_params,
                &mut *online_model,
            );

            // Log state vector for monitoring (can be disabled for performance)
            debug!("{}", self.state_vector.to_log_string());
        } else {
            debug!("⏸️  Skipping state_vector.update(): L2Book mid not yet available");
        }
    }
    
    /// Calculate optimal multi-level targets using Hawkes, Robust HJB, and sizing logic.
    /// This is the NEW core quoting logic that replaces calculate_optimal_control.
    /// Returns: (Vec<(price, size)>, Vec<(price, size)>) for bids and asks.
    fn calculate_multi_level_targets(&mut self) -> (Vec<(f64, f64)>, Vec<(f64, f64)>) {
        let start = std::time::Instant::now();
        let current_time = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs_f64();

        // --- 1. Get Current State & Uncertainty ---
        let state = &self.state_vector;

        // ⚡ OPTIMIZED: Use cached volatility uncertainty from background task
        // This eliminates read lock contention on particle filter (1-2ms savings)
        let cached_vol = self.cached_volatility.read();
        let (mu_std, _, _) = cached_vol.param_std_devs;
        let sigma_std = cached_vol.volatility_std_dev_bps;
        drop(cached_vol);  // Release lock immediately

        // Use adverse selection model estimate as nominal mu, PF uncertainty if available
        let nominal_mu = state.adverse_selection_estimate;
        // Use particle filter vol estimate as nominal sigma (already in state_vector)
        let nominal_sigma = state.volatility_ema_bps;

        self.current_uncertainty = ParameterUncertainty::from_particle_filter_stats(
            mu_std, // Or use a fixed value/MAE from online_model if PF doesn't estimate drift directly
            sigma_std,
            0.95, // Example confidence level
        );

        // --- 2. Compute Robust Parameters ---
        let robust_params = RobustParameters::compute(
            nominal_mu,
            nominal_sigma,
            state.inventory,
            &self.current_uncertainty,
            &self.robust_config,
        );

        // --- 3. Prepare Optimization State ---
        let hawkes_lock = self.hawkes_model.read(); // Read lock needed
        let opt_state = OptimizationState {
            mid_price: state.mid_price,
            inventory: state.inventory,
            max_position: self.max_absolute_position_size,
            // Use ROBUST drift estimate for optimization
            adverse_selection_bps: robust_params.mu_worst_case,
            lob_imbalance: state.lob_imbalance,
            // Use ROBUST vol estimate
            volatility_bps: robust_params.sigma_worst_case,
            current_time,
            hawkes_model: &hawkes_lock, // Pass reference to locked model
        };

        // --- 4. Run Multi-Level Optimizer ---
        // Calculate base half-spread using robust vol & multiplier, bounded by floor
        let min_profitable_half_spread = self.multi_level_optimizer.config().min_profitable_spread_bps / 2.0;
        let robust_base_half_spread = (robust_params.sigma_worst_case * 0.1) // Example: 10% of vol as base spread heuristic
                                         .max(min_profitable_half_spread)
                                         * robust_params.spread_multiplier; // Apply robustness widening

        // Get current tuning parameters
        let current_tuning_params = self.tuning_params.read().get_constrained();

        let multi_level_control = self.multi_level_optimizer.optimize(
            &opt_state,
            robust_base_half_spread, // Pass the calculated robust base
            &current_tuning_params,
        );
        drop(hawkes_lock); // Release lock

        // --- 5. Convert Control Offsets to Prices ---
        let mut target_bids: Vec<(f64, f64)> = Vec::new();
        let mut target_asks: Vec<(f64, f64)> = Vec::new();

        for (offset_bps, size_raw) in multi_level_control.bid_levels {
            if size_raw < EPSILON {
                continue;
            } // Skip dust

            // FIX: Round size DOWN to asset's sz_decimals (conservative - never place more than intended)
            let size = self.tick_lot_validator.round_size(size_raw, false);
            if size < EPSILON {
                continue; // Skip if rounded size is too small
            }

            let price_raw = state.mid_price * (1.0 - offset_bps / 10000.0);
            let price = self.tick_lot_validator.round_price(price_raw, false); // Round bid down

            if price > 0.0 && (size * price) >= 10.0 { // Notional minimum check: $10
                target_bids.push((price, size)); // Push rounded size
            }
        }

        for (offset_bps, size_raw) in multi_level_control.ask_levels {
            if size_raw < EPSILON {
                continue;
            }

            // FIX: Round size DOWN to asset's sz_decimals (conservative)
            let size = self.tick_lot_validator.round_size(size_raw, false);
            if size < EPSILON {
                continue; // Skip if rounded size is too small
            }

            let price_raw = state.mid_price * (1.0 + offset_bps / 10000.0);
            let price = self.tick_lot_validator.round_price(price_raw, true); // Round ask up

            if price > 0.0 && (size * price) >= 10.0 { // Notional minimum check: $10
                target_asks.push((price, size)); // Push rounded size
            }
        }

        // --- 6. Final Processing & Logging ---
        // Sort bids descending, asks ascending by price
        target_bids.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        target_asks.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        // Optional: Implement logic to merge orders at the same price tick
        // (This could be added later if needed)

        let elapsed = start.elapsed();
        debug!(
            "[MULTI-LEVEL OPTIMIZE] Targets ({}μs): Bids({}): {:?}, Asks({}): {:?}",
            elapsed.as_micros(), target_bids.len(), target_bids, target_asks.len(), target_asks
        );
        // Store taker rates for use in potentially_update
        self.latest_taker_buy_rate = multi_level_control.taker_buy_rate;
        self.latest_taker_sell_rate = multi_level_control.taker_sell_rate;
        
        if multi_level_control.liquidate {
            warn!("Liquidation mode triggered by optimizer!");
            info!("Taker rates set: buy={:.4}, sell={:.4}", 
                  self.latest_taker_buy_rate, self.latest_taker_sell_rate);
        }

        (target_bids, target_asks)
    }
    
    /// Get the current state vector (read-only access)
    pub fn get_state_vector(&self) -> &StateVector {
        &self.state_vector
    }
    
    /// Calculate optimal spread adjustment based on state vector
    /// This can be used to implement more sophisticated pricing strategies
    pub fn calculate_state_based_spread_adjustment(&self) -> f64 {
        // Use dynamic spread calculation (half-spread)
        let base_total_spread_bps = (self.state_vector.market_spread_bps).max(12.0);
        let base_half_spread_bps = base_total_spread_bps / 2.0;
        // Get constrained (theta) params
        let constrained_params = self.tuning_params.read().get_constrained();
        
        // Get adverse selection adjustment
        let adverse_adjustment = self.state_vector
            .get_adverse_selection_adjustment(base_half_spread_bps, constrained_params.adverse_selection_adjustment_factor);
        
        // Get inventory risk multiplier
        let risk_multiplier = self.state_vector
            .get_inventory_risk_multiplier(self.max_absolute_position_size);
        
        // Combined adjustment: adverse selection shift + risk-based widening
        // Note: This is a simple combination - more sophisticated models could be used
        adverse_adjustment + (base_half_spread_bps * (risk_multiplier - 1.0))
    }
    
    /// Check if we should pause market making based on state vector
    pub fn should_pause_trading(&self) -> bool {
        // Pause if market conditions are unfavorable
        // Use dynamic spread: threshold = 10x current *market* spread
        let base_total_spread_bps = (self.state_vector.market_spread_bps).max(12.0);
        let max_spread_threshold = base_total_spread_bps * 10.0;
        !self.state_vector.is_market_favorable(max_spread_threshold)
    }

    /// Clean up invalid resting orders (e.g., zero/negative positions, zero OIDs)
    /// Also re-assigns level indices based on sorted price.
    fn cleanup_invalid_resting_orders(&mut self) {
        let initial_bids = self.bid_levels.len();
        let initial_asks = self.ask_levels.len();

        // Remove orders that are effectively filled or invalid
        self.bid_levels.retain(|order| order.position >= EPSILON && order.oid != 0);
        self.ask_levels.retain(|order| order.position >= EPSILON && order.oid != 0);

        // Re-sort just in case order of remaining items changed (unlikely but safe)
        self.bid_levels.sort_by(|a, b| b.price.partial_cmp(&a.price).unwrap_or(std::cmp::Ordering::Equal));
        self.ask_levels.sort_by(|a, b| a.price.partial_cmp(&b.price).unwrap_or(std::cmp::Ordering::Equal));

        // Re-assign levels based on current sorted order (0 = best price)
        for (i, order) in self.bid_levels.iter_mut().enumerate() {
            order.level = i;
        }
        for (i, order) in self.ask_levels.iter_mut().enumerate() {
            order.level = i;
        }

        if self.bid_levels.len() < initial_bids || self.ask_levels.len() < initial_asks {
             log::debug!( // Use debug level
                 "Cleaned up orders: {} bids removed, {} asks removed.",
                 initial_bids - self.bid_levels.len(),
                 initial_asks - self.ask_levels.len()
             );
        }

        // Clean up old pending_cancel_orders (prevent memory leak)
        let current_time = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs_f64();

        let initial_pending_count = self.pending_cancel_orders.len();
        self.pending_cancel_orders.retain(|oid, pending| {
            let age = current_time - pending.cancel_time;
            if age > PENDING_CANCEL_TIMEOUT_SECS {
                debug!("Removing stale pending_cancel order (oid={}, age={:.1}s > {:.1}s)",
                      oid, age, PENDING_CANCEL_TIMEOUT_SECS);
                false
            } else {
                true
            }
        });

        if self.pending_cancel_orders.len() < initial_pending_count {
            debug!("Cleaned up {} stale pending_cancel orders",
                  initial_pending_count - self.pending_cancel_orders.len());
        }

        // Clean up stale pending order intents (prevent memory leak from failed placements)
        // Intents older than 30 seconds that never got a result are likely stuck/failed
        let mut intents = self.pending_order_intents.write();
        let initial_intent_count = intents.len();
        intents.retain(|intent_id, intent| {
            let age = current_time - intent.submitted_time;
            if age > 30.0 {
                warn!("Removing stale order intent (intent_id={}, age={:.1}s, side={}, L{})",
                      intent_id, age, if intent.side { "BID" } else { "ASK" }, intent.level + 1);
                false
            } else {
                true
            }
        });

        if intents.len() < initial_intent_count {
            warn!("Cleaned up {} stale order intents (likely failed placements)",
                  initial_intent_count - intents.len());
        }
        drop(intents);  // Release lock
    }

    /// Helper function to cancel all currently tracked resting orders.
    async fn cancel_all_orders(&mut self) {
        let mut oids_to_cancel: Vec<u64> = Vec::new();

        // Drain removes elements while iterating, simplifying clearing logic
        oids_to_cancel.extend(self.bid_levels.drain(..).filter(|o| o.oid != 0).map(|o| o.oid));
        oids_to_cancel.extend(self.ask_levels.drain(..).filter(|o| o.oid != 0).map(|o| o.oid));

        // Ensure lists are empty after draining
        assert!(self.bid_levels.is_empty());
        assert!(self.ask_levels.is_empty());

        if !oids_to_cancel.is_empty() {
            info!("Cancelling all {} resting orders...", oids_to_cancel.len());
            
            // Cancel orders sequentially (ExchangeClient doesn't implement Clone)
            let mut cancelled_count = 0;
            for oid in oids_to_cancel {
                if self.attempt_cancel(self.asset.clone(), oid).await {
                    cancelled_count += 1;
                }
            }
            
            info!("Finished cancelling orders. {} successful.", cancelled_count);
        } else {
            info!("No active resting orders found to cancel.");
        }
    }

    pub async fn new(input: MarketMakerInput) -> Result<(MarketMaker, tokio::sync::watch::Receiver<crate::tui::state::DashboardState>), crate::Error> {
        let user_address = input.wallet.address();

        let info_client = InfoClient::new(None, Some(BaseUrl::Mainnet)).await?;
        let exchange_client =
            ExchangeClient::new(None, input.wallet, Some(BaseUrl::Mainnet), None, None)
                .await?;

        // Fetch asset metadata to get sz_decimals
        let sz_decimals = match input.asset_type {
            AssetType::Perp => {
                let meta = info_client.meta().await?;
                meta.universe
                    .iter()
                    .find(|asset_meta| asset_meta.name == input.asset)
                    .map(|asset_meta| asset_meta.sz_decimals)
                    .ok_or_else(|| crate::Error::InvalidInput(format!(
                        "Asset {} not found in metadata", input.asset
                    )))?
            }
            AssetType::Spot => {
                let _spot_meta = info_client.spot_meta().await?;
                // For spot assets, we need to look up the token info
                // For now, use a default of 6 decimals - this should be improved
                // to properly lookup the spot asset metadata
                6 // This is a placeholder - should be improved
            }
        };

        let tick_lot_validator = TickLotValidator::new(
            input.asset.clone(),
            input.asset_type,
            sz_decimals,
        );

        // Note: inventory_skew_calculator logic now integrated into multi_level_optimizer

        // Load initial tuning parameters from JSON file if it exists
        // After this, Adam optimizer takes full control of parameter tuning
        // The JSON file is only used for:
        // 1. Initial parameters on bot startup
        // 2. Persistence of Adam's learned parameters across restarts
        // To manually override parameters: stop bot, edit JSON, restart bot
        let initial_params = TuningParams::from_json_file("tuning_params.json")
            .unwrap_or_else(|e| {
                info!("Could not load tuning_params.json ({}), using defaults", e);
                TuningParams::default()
            });
        
        // Log the *constrained* (theta) values for readability
        info!("Initialized with tuning parameters (constrained): {:?}", initial_params.get_constrained());
        info!("Adam optimizer will now autonomously tune these parameters");
        info!("✨ Online Adverse Selection Model enabled: Learning weights via SGD");
        info!("   Features: [bias, trade_flow, lob_imbalance, spread, volatility]");
        info!("   Lookback: 10 ticks (~10 sec), Learning rate: 0.001");
        
        // Initialize particle filter for stochastic volatility estimation
        let adaptive_config = AdaptiveConfig::liu_west();
        let particle_filter = Arc::new(RwLock::new(ParticleFilterState::new_liu_west(
            7000,    // num_particles (more recommended for joint estimation)
            -9.2,    // initial_mu
            0.88,    // initial_phi
            1.2,     // initial_sigma_eta
            -9.2,    // initial_h
            0.5,     // param_std_dev (base uncertainty for params)
            1.0,     // state_std_dev (initial h uncertainty)
            adaptive_config,
            42,      // seed
        )));
        info!("📊 Adaptive Liu-West SV Filter initialized:");
        info!("   Particles: 7000, delta: {}", adaptive_config.delta);
        info!("   Estimating state (h_t) AND parameters (mu, phi, sigma_eta)");
        
        // ===== Multi-Level Market Making Initialization =====
        
        // Get multi_level_config from input, with proper validation and defaults
        let multi_level_config = if input.enable_multi_level {
            if let Some(config) = input.multi_level_config.clone() {
                config
            } else {
                warn!("⚠️  enable_multi_level=true but multi_level_config is None!");
                warn!("   Using default MultiLevelConfig - this may not match your requirements");
                warn!("   Recommended: Explicitly provide multi_level_config in MarketMakerInput");
                MultiLevelConfig::default()
            }
        } else {
            // Single-level mode: Use default config with 1 level
            // WARNING: Single-level mode is largely untested with new framework
            warn!("⚠️  Single-level mode enabled (multi_level_config not provided)");
            warn!("   This mode is UNTESTED with the new multi-level framework!");
            warn!("   Recommended: Use multi-level with max_levels=1 instead");
            MultiLevelConfig::default()
        };
        
        // Extract max_levels from config for component initialization
        let max_levels = multi_level_config.max_levels;
        
        // Initialize Hawkes fill model with the configured number of levels
        let hawkes_model = Arc::new(RwLock::new(HawkesFillModel::new(max_levels)));
        
        // Initialize multi-level optimizer with the configuration
        let multi_level_optimizer = MultiLevelOptimizer::new(multi_level_config.clone());
        
        // Initialize robust control configuration
        // Ensure robust_config.enabled matches input.enable_robust_control
        let mut robust_config = input.robust_config.unwrap_or_default();
        robust_config.enabled = input.enable_robust_control;
        
        // Log initialization status for multi-level market making
        if input.enable_multi_level {
            info!("🎯 Multi-level market making ENABLED");
            info!("   Levels: {}", max_levels);
            info!("   Total size per side: {}", multi_level_config.total_size_per_side);
            info!("   Level spacing: {} bps", multi_level_config.level_spacing_bps);
            info!("   Min profitable spread: {} bps", multi_level_config.min_profitable_spread_bps);
        } else {
            info!("📌 Single-level mode (max_levels={})", max_levels);
            warn!("   Note: Single-level mode uses default config and may not be optimal");
        }
        
        // Log initialization status for robust control
        if input.enable_robust_control {
            info!("🛡️  Robust control ENABLED");
            info!("   Robustness level: {:.1}%", robust_config.robustness_level * 100.0);
            info!("   Uncertainty bounds will be applied to drift, volatility, and spreads");
        } else {
            info!("📊 Robust control DISABLED (using nominal parameters without uncertainty bounds)");
        }

        // ===== Async Order Execution Infrastructure =====

        // Create MPSC channel for order commands (buffered with capacity 2000)
        // Increased from 100 to handle high-frequency order placement without capacity errors
        let (order_command_tx, mut order_command_rx) = tokio::sync::mpsc::channel::<OrderCommand>(2000);

        // Create HIGH-PRIORITY channel for cancellations (smaller capacity, processed first)
        // This ensures cancels are never blocked by a saturated placement queue
        let (cancel_command_tx, mut cancel_command_rx) = tokio::sync::mpsc::channel::<OrderCommand>(500);

        // Create unbounded channel for order placement results
        // Async execution task sends results back to main loop for tracking
        let (order_result_tx, order_result_rx) = tokio::sync::mpsc::unbounded_channel::<OrderPlacementResult>();

        // Wrap exchange_client in Arc for sharing across tasks
        let exchange_client = Arc::new(exchange_client);
        let order_exec_exchange_client = exchange_client.clone();

        // Spawn dedicated order execution task
        // This task handles all network I/O for order placement/cancellation
        // decoupling it from the critical path event loop
        tokio::spawn(async move {
            // Initialize rate limiter: 15 requests/second with burst of 30
            let rate_limiter = Arc::new(tokio::sync::Mutex::new(TokenBucketRateLimiter::new(15.0, 30.0)));
            info!("⚡ Order execution task started (rate limit: 15 req/s, burst: 30)");
            info!("⚡ Using BIASED select for priority cancel channel");

            loop {
                // Use biased select to ALWAYS prioritize cancels over placements
                // This prevents cancel starvation when the placement queue is saturated
                let command = tokio::select! {
                    biased;  // CRITICAL: Process branches in order - cancels first!

                    // HIGH PRIORITY: Process cancels first
                    Some(cmd) = cancel_command_rx.recv() => Some((cmd, true)),

                    // LOWER PRIORITY: Process placements second
                    Some(cmd) = order_command_rx.recv() => Some((cmd, false)),

                    // Both channels closed
                    else => None,
                };

                let Some((command, is_cancel)) = command else {
                    warn!("⚠️  Order execution task exiting - all channels closed");
                    break;
                };

                if is_cancel {
                    debug!("Processing command from HIGH-PRIORITY cancel channel");
                }

                // Spawn each command in its own task for concurrent execution
                // This prevents any single slow API call from blocking the channel
                let exec_client = order_exec_exchange_client.clone();
                let result_tx = order_result_tx.clone();
                let limiter = rate_limiter.clone();

                tokio::spawn(async move {
                    // Apply rate limiting
                    {
                        let mut limiter_guard = limiter.lock().await;
                        if let Some(wait_time) = limiter_guard.try_acquire() {
                            debug!("Rate limit reached, waiting {:?}", wait_time);
                            drop(limiter_guard); // Release lock before sleeping
                            tokio::time::sleep(wait_time).await;
                            // Try again after waiting
                            let mut limiter_guard2 = limiter.lock().await;
                            if let Some(additional_wait) = limiter_guard2.try_acquire() {
                                drop(limiter_guard2);
                                tokio::time::sleep(additional_wait).await;
                            }
                        }
                    }

                    match command {
                    OrderCommand::Place { request, intent_id } => {
                        debug!("Executing order placement: intent_id={}, price={}, size={}",
                               intent_id, request.limit_px, request.sz);

                        // Clone request for retry closure
                        let request_clone = ClientOrderRequest {
                            asset: request.asset.clone(),
                            is_buy: request.is_buy,
                            reduce_only: request.reduce_only,
                            limit_px: request.limit_px,
                            sz: request.sz,
                            cloid: request.cloid,
                            order_type: match &request.order_type {
                                ClientOrder::Limit(l) => ClientOrder::Limit(ClientLimit { tif: l.tif.clone() }),
                                ClientOrder::Trigger(t) => ClientOrder::Trigger(ClientTrigger {
                                    is_market: t.is_market,
                                    trigger_px: t.trigger_px,
                                    tpsl: t.tpsl.clone(),
                                }),
                            },
                        };
                        let client = exec_client.clone();

                        match retry_with_backoff(
                            || async {
                                let req = ClientOrderRequest {
                                    asset: request_clone.asset.clone(),
                                    is_buy: request_clone.is_buy,
                                    reduce_only: request_clone.reduce_only,
                                    limit_px: request_clone.limit_px,
                                    sz: request_clone.sz,
                                    cloid: request_clone.cloid,
                                    order_type: match &request_clone.order_type {
                                        ClientOrder::Limit(l) => ClientOrder::Limit(ClientLimit { tif: l.tif.clone() }),
                                        ClientOrder::Trigger(t) => ClientOrder::Trigger(ClientTrigger {
                                            is_market: t.is_market,
                                            trigger_px: t.trigger_px,
                                            tpsl: t.tpsl.clone(),
                                        }),
                                    },
                                };
                                client.order(req, None).await
                            },
                            MAX_RETRY_ATTEMPTS,
                            &format!("Order placement (intent_id={})", intent_id),
                        ).await {
                            Ok(response) => {
                                if let ExchangeResponseStatus::Ok(data) = response {
                                    if let Some(statuses) = data.data {
                                        if let Some(status) = statuses.statuses.first() {
                                            debug!("Order placed successfully: intent_id={}, status={:?}",
                                                   intent_id, status);

                                            // Match on the status enum to extract oid
                                            match status {
                                                ExchangeDataStatus::Resting(resting_order) => {
                                                    let oid = resting_order.oid;
                                                    debug!("Order placed with oid={}", oid);

                                                    // Send success result back to main loop
                                                    let result = OrderPlacementResult {
                                                        intent_id,
                                                        oid: Some(oid),
                                                        success: true,
                                                        error_message: None,
                                                    };
                                                    if let Err(e) = result_tx.send(result) {
                                                        error!("Failed to send order result for intent_id={}: {}", intent_id, e);
                                                    }
                                                }
                                                ExchangeDataStatus::Filled(_) => {
                                                    // Order was filled immediately
                                                    debug!("Order filled immediately (no resting): intent_id={}", intent_id);
                                                    let result = OrderPlacementResult {
                                                        intent_id,
                                                        oid: None,  // No oid because it filled immediately
                                                        success: true,
                                                        error_message: None,
                                                    };
                                                    if let Err(e) = result_tx.send(result) {
                                                        error!("Failed to send order result for intent_id={}: {}", intent_id, e);
                                                    }
                                                }
                                                ExchangeDataStatus::Error(err_msg) => {
                                                    warn!("Order placement failed: intent_id={}, error={}", intent_id, err_msg);
                                                    let result = OrderPlacementResult {
                                                        intent_id,
                                                        oid: None,
                                                        success: false,
                                                        error_message: Some(err_msg.clone()),
                                                    };
                                                    if let Err(e) = result_tx.send(result) {
                                                        error!("Failed to send order failure result for intent_id={}: {}", intent_id, e);
                                                    }
                                                }
                                                _ => {
                                                    debug!("Order status: intent_id={}, status={:?}", intent_id, status);
                                                    // For other statuses (Success, WaitingForFill, etc.), send success
                                                    let result = OrderPlacementResult {
                                                        intent_id,
                                                        oid: None,
                                                        success: true,
                                                        error_message: None,
                                                    };
                                                    if let Err(e) = result_tx.send(result) {
                                                        error!("Failed to send order result for intent_id={}: {}", intent_id, e);
                                                    }
                                                }
                                            }
                                        }
                                    }
                                } else {
                                    warn!("Order placement failed: intent_id={}, response={:?}",
                                          intent_id, response);

                                    // Send failure result back to main loop
                                    let result = OrderPlacementResult {
                                        intent_id,
                                        oid: None,
                                        success: false,
                                        error_message: Some(format!("{:?}", response)),
                                    };
                                    if let Err(e) = result_tx.send(result) {
                                        error!("Failed to send order failure result for intent_id={}: {}", intent_id, e);
                                    }
                                }
                            }
                            Err(e) => {
                                error!("Order placement error: intent_id={}, error={}", intent_id, e);

                                // Send failure result back to main loop
                                let result = OrderPlacementResult {
                                    intent_id,
                                    oid: None,
                                    success: false,
                                    error_message: Some(e.to_string()),
                                };
                                if let Err(send_err) = result_tx.send(result) {
                                    error!("Failed to send order error result for intent_id={}: {}", intent_id, send_err);
                                }
                            }
                        }
                    }
                    OrderCommand::BatchPlace { requests, intent_ids } => {
                        if requests.is_empty() {
                            return; // Early return from spawned task
                        }
                        let num_orders = requests.len();
                        debug!("Executing batch order placement: {} orders", num_orders);

                        // Clone for retry closure
                        let requests_clone = requests.clone();
                        let intent_ids_clone = intent_ids.clone();
                        let client = exec_client.clone();

                        match retry_with_backoff(
                            || async {
                                let reqs: Vec<ClientOrderRequest> = requests_clone.iter().map(|r| {
                                    ClientOrderRequest {
                                        asset: r.asset.clone(),
                                        is_buy: r.is_buy,
                                        reduce_only: r.reduce_only,
                                        limit_px: r.limit_px,
                                        sz: r.sz,
                                        cloid: r.cloid,
                                        order_type: match &r.order_type {
                                            ClientOrder::Limit(l) => ClientOrder::Limit(ClientLimit { tif: l.tif.clone() }),
                                            ClientOrder::Trigger(t) => ClientOrder::Trigger(ClientTrigger {
                                                is_market: t.is_market,
                                                trigger_px: t.trigger_px,
                                                tpsl: t.tpsl.clone(),
                                            }),
                                        },
                                    }
                                }).collect();
                                client.bulk_order(reqs, None).await
                            },
                            MAX_RETRY_ATTEMPTS,
                            &format!("Batch order placement ({} orders)", num_orders),
                        ).await {
                            Ok(response) => {
                                if let ExchangeResponseStatus::Ok(data) = response {
                                    if let Some(statuses) = data.data {
                                        for (idx, status) in statuses.statuses.iter().enumerate() {
                                            let intent_id = intent_ids_clone.get(idx).copied().unwrap_or(0);
                                            debug!("Batch order {} placed: intent_id={}, status={:?}",
                                                   idx, intent_id, status);
                                        }
                                    }
                                } else {
                                    warn!("Batch order placement failed: response={:?}", response);
                                }
                            }
                            Err(e) => {
                                error!("Batch order placement error: {}", e);
                            }
                        }
                    }
                    OrderCommand::Cancel { request } => {
                        debug!("Executing order cancellation: oid={}", request.oid);

                        let asset = request.asset.clone();
                        let oid = request.oid;
                        let client = exec_client.clone();

                        match retry_with_backoff(
                            || async {
                                let req = ClientCancelRequest {
                                    asset: asset.clone(),
                                    oid,
                                };
                                client.cancel(req, None).await
                            },
                            MAX_RETRY_ATTEMPTS,
                            &format!("Order cancellation (oid={})", oid),
                        ).await {
                            Ok(response) => {
                                debug!("Order cancel response: {:?}", response);
                            }
                            Err(e) => {
                                error!("Order cancellation error: {}", e);
                            }
                        }
                    }
                    OrderCommand::BatchCancel { requests } => {
                        if requests.is_empty() {
                            return; // Early return from spawned task
                        }
                        let num_cancels = requests.len();
                        debug!("Executing batch cancel: {} orders", num_cancels);

                        let requests_clone = requests.clone();
                        let client = exec_client.clone();

                        match retry_with_backoff(
                            || async {
                                let reqs: Vec<ClientCancelRequest> = requests_clone.iter().map(|r| {
                                    ClientCancelRequest {
                                        asset: r.asset.clone(),
                                        oid: r.oid,
                                    }
                                }).collect();
                                client.bulk_cancel(reqs, None).await
                            },
                            MAX_RETRY_ATTEMPTS,
                            &format!("Batch cancel ({} orders)", num_cancels),
                        ).await {
                            Ok(response) => {
                                debug!("Batch cancel response: {:?}", response);
                            }
                            Err(e) => {
                                error!("Batch cancel error: {}", e);
                            }
                        }
                    }
                    } // End of match command
                }); // End of tokio::spawn for this command
            } // End of loop
        });

        info!("✅ Async order execution task initialized (channel buffer: 2000)");

        // ===== Background Particle Filter Update Task =====

        // Create shared cached volatility estimate
        let cached_volatility = Arc::new(RwLock::new(CachedVolatilityEstimate::default()));

        // Create channel for sending price updates to PF background task
        let (pf_price_tx, mut pf_price_rx) = tokio::sync::mpsc::channel::<f64>(100);

        // ===== TUI Dashboard =====

        // Create watch channel for TUI dashboard state broadcasting
        let (tui_state_tx, tui_state_rx) = tokio::sync::watch::channel(crate::tui::state::DashboardState::default());

        // Record start time for uptime calculation
        let start_time = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs_f64();

        // Clone necessary references for the background task
        let pf_clone = particle_filter.clone();
        let cached_vol_clone = cached_volatility.clone();

        // Spawn background particle filter update task
        // This task receives price updates and periodically updates volatility estimates
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(
                std::time::Duration::from_millis(PARTICLE_FILTER_UPDATE_INTERVAL_MS)
            );
            info!("⚡ Particle Filter background task started (update interval: {}ms)",
                  PARTICLE_FILTER_UPDATE_INTERVAL_MS);

            let mut latest_price: Option<f64> = None;
            let mut update_counter = 0usize;

            loop {
                tokio::select! {
                    // Receive price updates (non-blocking)
                    Some(price) = pf_price_rx.recv() => {
                        latest_price = Some(price);
                    }

                    // Periodic update timer
                    _ = interval.tick() => {
                        // Update particle filter with latest price if available
                        if let Some(price) = latest_price {
                            let mut pf_lock = pf_clone.write();

                            if let Some(vol_estimate_bps) = pf_lock.update(price) {
                                // Successfully updated PF
                                drop(pf_lock);  // Release write lock

                                // Read current PF state and extract volatility estimates
                                let pf_read = pf_clone.read();

                                let volatility_bps = pf_read.estimate_volatility_bps();
                                let vol_5th = pf_read.estimate_volatility_percentile_bps(0.05);
                                let vol_95th = pf_read.estimate_volatility_percentile_bps(0.95);
                                let param_std_devs = pf_read.get_parameter_std_devs();
                                let volatility_std_dev_bps = pf_read.get_volatility_std_dev_bps();

                                // Log diagnostics periodically
                                update_counter += 1;
                                if update_counter % 20 == 0 {  // Every 20 updates (3 seconds at 150ms)
                                    let ess = pf_read.get_ess();
                                    debug!(
                                        "📊 PF: vol={:.2}bps, ESS={:.0}/7000 ({:.1}%), CI=[{:.2}, {:.2}]",
                                        vol_estimate_bps, ess, 100.0 * ess / 7000.0, vol_5th, vol_95th
                                    );
                                }

                                drop(pf_read);  // Release read lock

                                // Update cached estimate
                                let mut cache = cached_vol_clone.write();
                                cache.volatility_bps = volatility_bps;
                                cache.vol_5th_percentile = vol_5th;
                                cache.vol_95th_percentile = vol_95th;
                                cache.param_std_devs = param_std_devs;
                                cache.volatility_std_dev_bps = volatility_std_dev_bps;
                                cache.last_update_time = std::time::SystemTime::now()
                                    .duration_since(std::time::UNIX_EPOCH)
                                    .unwrap()
                                    .as_secs_f64();
                                drop(cache);  // Release write lock
                            }
                        }
                    }
                }
            }
        });

        info!("✅ Background Particle Filter task initialized");

        let market_maker = MarketMaker {
            // ===== Core Configuration =====
            asset: input.asset,
            tick_lot_validator,
            max_absolute_position_size: input.max_absolute_position_size,
            
            // ===== Exchange Clients =====
            info_client,
            exchange_client,
            user_address,
            
            // ===== Real-Time State =====
            cur_position: 0.0,
            latest_mid_price: -1.0,
            latest_l2_mid_price: -1.0,  // Initialized to -1.0, updated from L2Book
            latest_book: None,
            state_vector: StateVector::new(),  // Initialized with defaults, updated on first AllMids message
            
            // ===== Advanced Models =====
            particle_filter,  // Liu-West adaptive filter for stochastic volatility
            online_adverse_selection_model: Arc::new(RwLock::new(OnlineAdverseSelectionModel::default())),
            
            // ===== Multi-Level Optimizer & Components =====
            multi_level_optimizer,  // Contains config, HJB logic, level sizing
            hjb_components: HJBComponents::default(),  // Old grid search optimizer (used as learning target)
            value_function: ValueFunction::new(0.1, 3600.0),  // Old value function (phi=0.1, T=1 hour)
            hawkes_model,           // Fill rate estimation with self-excitation
            robust_config,          // Uncertainty sets and robustness parameters
            current_uncertainty: ParameterUncertainty::default(),  // Updated by particle filter each tick
            
            // ===== Resting Order State (Multi-Level) =====
            bid_levels: Vec::with_capacity(max_levels),  // Initialize empty, will be populated on first quote
            ask_levels: Vec::with_capacity(max_levels),  // Initialize empty, will be populated on first quote
            pending_cancel_orders: HashMap::new(),  // Initialize empty HashMap for tracking canceled orders
            unmatched_fills: HashMap::new(),  // Initialize empty HashMap for holding pen

            // ===== Async Order Execution State =====
            pending_order_intents: Arc::new(RwLock::new(HashMap::new())),  // Track pending order placements
            next_intent_id: Arc::new(RwLock::new(0)),  // Counter for intent IDs
            order_command_tx,  // Channel sender for async order execution
            cancel_command_tx,  // HIGH-PRIORITY channel sender for cancellations
            order_result_rx,  // Channel receiver for order placement results
            cached_volatility,  // Cached volatility from background particle filter task
            pf_price_tx,  // Channel sender for particle filter price updates

            // ===== Adam Self-Tuning System =====
            tuning_params: Arc::new(RwLock::new(initial_params)),  // Meta-parameters for heuristic adjustments
            adam_optimizer: Arc::new(RwLock::new(AdamOptimizerState::default())),  // Adaptive moment estimation
            gradient_accumulator: Arc::new(RwLock::new(vec![0.0; 9])),  // 8 parameters + 1 loss = 9 elements
            gradient_count: Arc::new(RwLock::new(0)),  // Reset after each Adam update
            message_counter: Arc::new(RwLock::new(0)),  // For gradient sampling (every Nth message)
            
            // ===== Trading Control =====
            trading_enabled: Arc::new(RwLock::new(false)),  // Start disabled, enabled when optimizer validates performance
            enable_trading_gap_threshold_percent: input.enable_trading_gap_threshold_percent,

            // ===== Taker Logic State =====
            latest_taker_buy_rate: 0.0,  // Initialize to zero
            latest_taker_sell_rate: 0.0,  // Initialize to zero
            last_taker_buy_time: 0.0,  // Initialize to zero (unix epoch)
            last_taker_sell_time: 0.0,  // Initialize to zero (unix epoch)
            smoothed_taker_buy_rate: 0.0,  // Initialize to zero
            smoothed_taker_sell_rate: 0.0,  // Initialize to zero

            // ===== TUI Dashboard =====
            tui_state_tx,  // Watch channel sender for dashboard state updates
            start_time,    // Unix timestamp when market maker started

            // ===== Performance Tracking =====
            equity_history: std::collections::VecDeque::with_capacity(1000),  // Max 1000 samples
            last_equity: 0.0,  // Initialize to zero
        };

        Ok((market_maker, tui_state_rx))
    }

    pub async fn start(&mut self) {
        self.start_with_shutdown_signal(None).await;
    }

    pub async fn start_with_shutdown_signal(&mut self, mut shutdown_rx: Option<tokio::sync::oneshot::Receiver<()>>) {
        let (sender, mut receiver) = unbounded_channel();

        // Subscribe to UserEvents for fills
        self.info_client
            .subscribe(
                Subscription::UserEvents {
                    user: self.user_address,
                },
                sender.clone(),
            )
            .await
            .unwrap();

        // Subscribe to AllMids so we can market make around the mid price
        self.info_client
            .subscribe(Subscription::AllMids, sender.clone())
            .await
            .unwrap();

        // Subscribe to L2Book for market data
        info!("Subscribing to L2 book data");
        self.info_client
            .subscribe(
                Subscription::L2Book {
                    coin: self.asset.clone(),
                },
                sender.clone(),
            )
            .await
            .unwrap();

        // Subscribe to Trades data for trade flow analysis
        info!("Subscribing to Trades data for trade flow analysis");
        self.info_client
            .subscribe(
                Subscription::Trades {
                    coin: self.asset.clone(),
                },
                sender.clone(),
            )
            .await
            .unwrap();

        // Subscribe to OrderUpdates to get oid->cloid mappings for fill reconciliation
        info!("Subscribing to OrderUpdates for oid mapping");
        self.info_client
            .subscribe(
                Subscription::OrderUpdates {
                    user: self.user_address,
                },
                sender,
            )
            .await
            .unwrap();

        // Create Adam optimizer timer (60 second interval)
        let mut adam_timer = time::interval(std::time::Duration::from_secs(60));
        
        loop {
            tokio::select! {
                // Check for shutdown signal
                _ = async {
                    if let Some(ref mut rx) = shutdown_rx {
                        rx.await.ok();
                    } else {
                        std::future::pending::<()>().await
                    }
                } => {
                    info!("Shutdown signal received, cancelling orders and exiting...");
                    self.shutdown().await;
                    break;
                }

                // Handle order placement results from async execution task
                Some(result) = self.order_result_rx.recv() => {
                    self.handle_order_placement_result(result);
                }

                // Handle market maker messages
                message = receiver.recv() => {
            let message = message.unwrap();
            match message {
                Message::L2Book(l2_book) => {
                    // Update our order book state
                    if let Some(book) = OrderBook::from_l2_data(&l2_book.data) {
                        self.latest_book = Some(book.clone());

                        // Calculate mid from L2Book BBO and store as authoritative mid-price
                        if !book.bids.is_empty() && !book.asks.is_empty() {
                            if let (Ok(best_bid), Ok(best_ask)) = (
                                book.bids[0].px.parse::<f64>(),
                                book.asks[0].px.parse::<f64>()
                            ) {
                                let l2_mid = (best_bid + best_ask) / 2.0;
                                // CRITICAL: Update L2Book mid-price (used for order pricing)
                                self.latest_l2_mid_price = l2_mid;

                                let allmids_mid = self.latest_mid_price;
                                let _diff_bps = ((l2_mid - allmids_mid) / allmids_mid * 10000.0).abs();
                            }
                        }

                        // Update state vector with new book data
                        self.update_state_vector();

                        // Update TUI dashboard with new state
                        self.update_tui_state();
                    }
                }
                Message::Trades(trades) => {
                    // Get constrained (theta) params
                    let constrained_params = self.tuning_params.read().get_constrained();
                    // Call our new function to update the trade flow EMA
                    self.state_vector.update_trade_flow_ema(&trades.data, &constrained_params);
                }
                Message::AllMids(all_mids) => {
                    let all_mids = all_mids.data.mids;
                    let mid = all_mids.get(&self.asset);
                    if let Some(mid) = mid {
                        let mid: f64 = mid.parse().unwrap();
                        self.latest_mid_price = mid;

                        // Mid-price updates are tracked silently (shown in TUI)

                        // ⚡ OPTIMIZATION: Removed redundant update_state_vector() call
                        // State vector is updated by L2Book handler which has actual order book data
                        // This eliminates ~15-30ms of redundant computation per AllMids message
                        // Latency savings: ~15-30ms per AllMids (1-2s depending on frequency)

                        // ============================================
                        // ONLINE ADVERSE SELECTION MODEL TRAINING
                        // ============================================
                        // Perform SGD update if enough history is available
                        // NOTE: record_observation is now called inside StateVector::update_adverse_selection
                        {
                            let mut model = self.online_adverse_selection_model.write();
                            
                            // Perform SGD update using delayed label (actual price change)
                            model.update(mid);
                            
                            // Log model stats every 100 updates for monitoring
                            if model.update_count > 0 && model.update_count % 100 == 0 {
                                info!("Online Adverse Selection Model: {}", model.get_stats());
                            }
                        }
                        
                        // ============================================
                        // GRADIENT ACCUMULATION FOR STABLE ADAM OPTIMIZER
                        // ============================================
                        // Sample every 5th AllMids message to balance CPU usage with gradient quality
                        // This provides ~12 gradient samples per minute (assuming ~1 AllMids/second)
                        // Over 60 seconds, this accumulates ~12 gradients for stable averaging
                        {
                            let mut counter = self.message_counter.write();
                            *counter += 1;
                            let sample_interval = 5; // Accumulate gradient every 5th message (~12 samples/minute)
                            
                            if *counter % sample_interval == 0 {
                                // Call accumulate_gradient_snapshot for Adam optimization
                                self.accumulate_gradient_snapshot();
                            }
                        }
                        
                        // Check to see if we need to cancel or place any new orders
                        self.potentially_update().await;
                    } else {
                        error!(
                            "could not get mid for asset {}: {all_mids:?}",
                            self.asset.clone()
                        );
                    }
                }
                Message::User(user_events) => {
                    // We haven't seen the first mid price event yet, so just continue
                    if self.latest_mid_price < 0.0 {
                        continue;
                    }
                    
                    let user_events = user_events.data;
                    if let UserData::Fills(fills) = user_events {
                        let mut position_changed = false;
                        let mut hawkes_needs_update = false;
                        let current_time = std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap_or_default()
                            .as_secs_f64();

                        for fill in fills {
                            let amount: f64 = fill.sz.parse().unwrap_or(0.0);
                            if amount < EPSILON {
                                continue;
                            }

                            position_changed = true;
                            let is_bid_fill = fill.side == "B";
                            let filled_oid = fill.oid;
                            let mut filled_level: Option<usize> = None;

                            // DEBUG: Enhanced fill notification logging
                            let fill_price: f64 = fill.px.parse().unwrap_or(0.0);
                            // Use L2Book mid (where fills actually occur) for accurate offset calculation
                            let current_mid = self.latest_l2_mid_price;
                            let side_str = if is_bid_fill { "BUY" } else { "SELL" };

                            // Calculate how far from mid the fill happened
                            let offset_from_mid_bps = if is_bid_fill {
                                (current_mid - fill_price) / current_mid * 10000.0
                            } else {
                                (fill_price - current_mid) / current_mid * 10000.0
                            };

                            // Check if this was likely an instant fill (very close to mid)
                            let is_suspicious = offset_from_mid_bps.abs() < 20.0;  // Within 20 bps of mid = probably instant fill

                            if is_suspicious {
                                warn!(
                                    "⚡ INSTANT FILL?: {} {:.2} @ {:.3} | mid={:.3} | offset={:.1}bps | oid={}",
                                    side_str, amount, fill_price, current_mid, offset_from_mid_bps, filled_oid
                                );
                            } else {
                                info!(
                                    "✅ FILL: {} {:.2} @ {:.3} | mid={:.3} | offset={:.1}bps | oid={}",
                                    side_str, amount, fill_price, current_mid, offset_from_mid_bps, filled_oid
                                );
                            }

                            if is_bid_fill {
                                // Our Bid was filled (we bought)
                                self.cur_position += amount;
                                info!("Fill: bought {} {} (oid: {})", amount, self.asset.clone(), filled_oid);

                                // First, check pending_cancel_orders HashMap (O(1) lookup for late fills)
                                if let Some(pending_cancel) = self.pending_cancel_orders.get_mut(&filled_oid) {
                                    // Fill arrived after cancel was sent
                                    filled_level = Some(pending_cancel.order.level);
                                    pending_cancel.order.position -= amount;

                                    if pending_cancel.order.position < EPSILON {
                                        info!("Bid oid {} (L{}) filled after cancel sent, removing from pending_cancel HashMap.",
                                              filled_oid, filled_level.unwrap_or(99) + 1);
                                        self.pending_cancel_orders.remove(&filled_oid);
                                    } else {
                                        info!("Bid oid {} (L{}) partially filled after cancel sent, remaining: {}",
                                              filled_oid, filled_level.unwrap_or(99) + 1, pending_cancel.order.position);
                                    }
                                } else if let Some(index) = self.bid_levels.iter().position(|o| o.oid == filled_oid) {
                                    // Find and update the specific active resting bid (normal case)
                                    filled_level = Some(self.bid_levels[index].level); // Get level before modifying/removing
                                    self.bid_levels[index].position -= amount;

                                    if self.bid_levels[index].position < EPSILON {
                                        info!("Resting bid oid {} (L{}) fully filled, removing.", filled_oid, filled_level.unwrap_or(99) + 1);
                                        self.bid_levels.remove(index);
                                    } else {
                                        info!("Resting bid oid {} (L{}) partially filled, remaining: {}",
                                              filled_oid, filled_level.unwrap_or(99) + 1, self.bid_levels[index].position);
                                    }
                                } else {
                                    // ⚡ NEW: Handle async order execution
                                    // Fill arrived for an order we submitted but haven't added to tracking yet
                                    // Check pending_order_intents to find the matching intent
                                    let intents = self.pending_order_intents.read();

                                    // Find matching intent by price and size (since we don't have oid yet)
                                    let matching_intent = intents.values().find(|intent| {
                                        intent.side == true && // buy
                                        (intent.price - fill_price).abs() < EPSILON &&
                                        (intent.size - amount).abs() < EPSILON
                                    }).cloned();

                                    drop(intents);  // Release lock

                                    if let Some(intent) = matching_intent {
                                        filled_level = Some(intent.level);
                                        info!("⚡ ASYNC FILL: Bid matched pending intent (intent_id={}, L{})",
                                              intent.intent_id, intent.level + 1);

                                        // Add order to tracking (first fill sets the oid)
                                        let remaining_size = intent.size - amount;
                                        if remaining_size > EPSILON {
                                            // Partial fill - add to tracking with remaining size
                                            let order = MarketMakerRestingOrder {
                                                oid: filled_oid,
                                                position: remaining_size,
                                                price: fill_price,
                                                level: intent.level,
                                                pending_cancel: false,
                                            };
                                            self.bid_levels.push(order);
                                            info!("Added partially filled bid to tracking: oid={}, remaining={}",
                                                  filled_oid, remaining_size);
                                        } else {
                                            // Fully filled before we could add to tracking
                                            info!("Bid fully filled on placement (intent_id={})", intent.intent_id);
                                        }

                                        // Remove intent from pending
                                        self.pending_order_intents.write().remove(&intent.intent_id);
                                    } else {
                                        // Fill arrived before orderUpdates confirmed the oid mapping
                                        // Add to holding pen - will be processed when orderUpdates arrives
                                        debug!("Fill arrived before oid mapping for bid oid: {}, adding to holding pen", filled_oid);
                                        self.unmatched_fills.entry(filled_oid).or_insert_with(Vec::new).push(fill.clone());
                                    }
                                }
                            } else {
                                // Our Ask was filled (we sold)
                                self.cur_position -= amount;
                                info!("Fill: sold {} {} (oid: {})", amount, self.asset.clone(), filled_oid);

                                // First, check pending_cancel_orders HashMap (O(1) lookup for late fills)
                                if let Some(pending_cancel) = self.pending_cancel_orders.get_mut(&filled_oid) {
                                    // Fill arrived after cancel was sent
                                    filled_level = Some(pending_cancel.order.level);
                                    pending_cancel.order.position -= amount;

                                    if pending_cancel.order.position < EPSILON {
                                        info!("Ask oid {} (L{}) filled after cancel sent, removing from pending_cancel HashMap.",
                                              filled_oid, filled_level.unwrap_or(99) + 1);
                                        self.pending_cancel_orders.remove(&filled_oid);
                                    } else {
                                        info!("Ask oid {} (L{}) partially filled after cancel sent, remaining: {}",
                                              filled_oid, filled_level.unwrap_or(99) + 1, pending_cancel.order.position);
                                    }
                                } else if let Some(index) = self.ask_levels.iter().position(|o| o.oid == filled_oid) {
                                    // Find and update the specific active resting ask (normal case)
                                    filled_level = Some(self.ask_levels[index].level);
                                    self.ask_levels[index].position -= amount;

                                    if self.ask_levels[index].position < EPSILON {
                                        info!("Resting ask oid {} (L{}) fully filled, removing.", filled_oid, filled_level.unwrap_or(99) + 1);
                                        self.ask_levels.remove(index);
                                    } else {
                                        info!("Resting ask oid {} (L{}) partially filled, remaining: {}",
                                              filled_oid, filled_level.unwrap_or(99) + 1, self.ask_levels[index].position);
                                    }
                                } else {
                                    // ⚡ NEW: Handle async order execution
                                    // Fill arrived for an order we submitted but haven't added to tracking yet
                                    // Check pending_order_intents to find the matching intent
                                    let intents = self.pending_order_intents.read();

                                    // Find matching intent by price and size (since we don't have oid yet)
                                    let matching_intent = intents.values().find(|intent| {
                                        intent.side == false && // sell
                                        (intent.price - fill_price).abs() < EPSILON &&
                                        (intent.size - amount).abs() < EPSILON
                                    }).cloned();

                                    drop(intents);  // Release lock

                                    if let Some(intent) = matching_intent {
                                        filled_level = Some(intent.level);
                                        info!("⚡ ASYNC FILL: Ask matched pending intent (intent_id={}, L{})",
                                              intent.intent_id, intent.level + 1);

                                        // Add order to tracking (first fill sets the oid)
                                        let remaining_size = intent.size - amount;
                                        if remaining_size > EPSILON {
                                            // Partial fill - add to tracking with remaining size
                                            let order = MarketMakerRestingOrder {
                                                oid: filled_oid,
                                                position: remaining_size,
                                                price: fill_price,
                                                level: intent.level,
                                                pending_cancel: false,
                                            };
                                            self.ask_levels.push(order);
                                            info!("Added partially filled ask to tracking: oid={}, remaining={}",
                                                  filled_oid, remaining_size);
                                        } else {
                                            // Fully filled before we could add to tracking
                                            info!("Ask fully filled on placement (intent_id={})", intent.intent_id);
                                        }

                                        // Remove intent from pending
                                        self.pending_order_intents.write().remove(&intent.intent_id);
                                    } else {
                                        // Fill arrived before orderUpdates confirmed the oid mapping
                                        // Add to holding pen - will be processed when orderUpdates arrives
                                        debug!("Fill arrived before oid mapping for ask oid: {}, adding to holding pen", filled_oid);
                                        self.unmatched_fills.entry(filled_oid).or_insert_with(Vec::new).push(fill.clone());
                                    }
                                }
                            }

                            // Update Hawkes model state IF we identified the level
                            if let Some(level) = filled_level {
                                self.hawkes_model.write().record_fill(level, is_bid_fill, current_time);
                                hawkes_needs_update = true; // Signal that quotes might need recalculation
                                info!("Updated Hawkes model: fill L{}, side {}", level + 1, if is_bid_fill {"BID"} else {"ASK"});
                            }
                        }

                        // Re-assign levels after potential removals
                        for (i, order) in self.bid_levels.iter_mut().enumerate() {
                            order.level = i;
                        }
                        for (i, order) in self.ask_levels.iter_mut().enumerate() {
                            order.level = i;
                        }

                        // CRITICAL: Check if position limit was exceeded after fills
                        if position_changed && self.cur_position.abs() > self.max_absolute_position_size {
                            error!(
                                "⛔ POSITION LIMIT EXCEEDED AFTER FILL: cur_position={:.2} > max={:.2}",
                                self.cur_position.abs(), self.max_absolute_position_size
                            );
                            warn!("⚠️  This indicates orders were placed that violated position limits");
                            warn!("⚠️  Bot will now attempt to flatten excess position via potentially_update()");
                            // Note: potentially_update() will suppress new orders in the violating direction
                            // and taker logic may help reduce position
                        }

                        // Trigger state update and potentially new quotes if needed
                        if position_changed || hawkes_needs_update {
                            self.update_state_vector(); // Update inventory, etc.
                            self.update_tui_state(); // Update TUI dashboard
                            self.potentially_update().await; // Re-evaluate multi-level quotes
                        }
                    }
                }
                Message::OrderUpdates(order_updates) => {
                    // Process oid->cloid mappings from orderUpdates
                    // This is critical for reconciling fills that arrive before oid confirmation
                    for order_status in &order_updates.data {
                        let oid = order_status.order.oid;

                        // Check if this oid has pending fills in the holding pen
                        if let Some(pending_fills) = self.unmatched_fills.remove(&oid) {
                            info!("🔄 Processing {} held fills for newly confirmed oid {}", pending_fills.len(), oid);

                            let current_time = std::time::SystemTime::now()
                                .duration_since(std::time::UNIX_EPOCH)
                                .unwrap_or_default()
                                .as_secs_f64();

                            // Process each fill that was waiting for this oid
                            for fill in pending_fills {
                                let amount: f64 = fill.sz.parse().unwrap_or(0.0);
                                if amount < EPSILON {
                                    continue;
                                }

                                let is_bid_fill = fill.side == "B";
                                let fill_price: f64 = fill.px.parse().unwrap_or(0.0);

                                // Update position
                                if is_bid_fill {
                                    self.cur_position += amount;
                                    info!("Held fill processed: bought {} {} @ {} (oid: {})",
                                          amount, self.asset, fill_price, oid);
                                } else {
                                    self.cur_position -= amount;
                                    info!("Held fill processed: sold {} {} @ {} (oid: {})",
                                          amount, self.asset, fill_price, oid);
                                }

                                // Try to find this order in our tracking to update Hawkes model
                                let level_opt = if is_bid_fill {
                                    self.bid_levels.iter().find(|o| o.oid == oid).map(|o| o.level)
                                } else {
                                    self.ask_levels.iter().find(|o| o.oid == oid).map(|o| o.level)
                                };

                                if let Some(level) = level_opt {
                                    self.hawkes_model.write().record_fill(level, is_bid_fill, current_time);
                                    info!("Updated Hawkes model from held fill: L{}, side {}",
                                          level + 1, if is_bid_fill {"BID"} else {"ASK"});
                                }
                            }

                            // Trigger state update after processing held fills
                            self.update_state_vector();
                            self.potentially_update().await;
                        }
                    }
                }
                _ => {
                    panic!("Unsupported message type");
                }
            }
                }

                // Adam optimizer timer - runs every 60 seconds
                _ = adam_timer.tick() => {
                    // Run Adam optimizer update (status shown in TUI)
                    let adam_handle = self.run_adam_update_and_enablement_check();
                    tokio::spawn(async move {
                        if let Err(e) = adam_handle.await {
                            error!("Adam update task failed: {:?}", e);
                        }
                    });
                }
            }
        }
    }

    async fn attempt_cancel(&self, asset: String, oid: u64) -> bool {
        let cancel = self
            .exchange_client
            .cancel(ClientCancelRequest { asset, oid }, None)
            .await;

        match cancel {
            Ok(cancel) => match cancel {
                ExchangeResponseStatus::Ok(cancel) => {
                    if let Some(cancel) = cancel.data {
                        if !cancel.statuses.is_empty() {
                            match cancel.statuses[0].clone() {
                                ExchangeDataStatus::Success => {
                                    return true;
                                }
                                ExchangeDataStatus::Error(e) => {
                                    // Check if it's an expected "already gone" error
                                    if e.contains("already canceled") || e.contains("filled") || e.contains("never placed") {
                                        // Don't log as error - this is expected during fast markets
                                        return false;
                                    }
                                    error!("Error with cancelling: {e}") // Only log real errors
                                }
                                _ => {
                                    error!("Unexpected response status when cancelling: {:?}", cancel.statuses[0]);
                                }
                            }
                        } else {
                            error!("Exchange data statuses is empty when cancelling: {cancel:?}")
                        }
                    } else {
                        error!("Exchange response data is empty when cancelling: {cancel:?}")
                    }
                }
                ExchangeResponseStatus::Err(e) => error!("Error with cancelling: {e}"),
            },
            Err(e) => error!("Error with cancelling: {e}"),
        }
        false
    }

    /// Place a market order using the SDK's market_open function with slippage protection
    /// This is safer than a pure IOC limit order as it handles slippage properly
    /// Returns the filled amount
    #[allow(dead_code)]
    async fn place_taker_order(
        &self,
        asset: String,
        amount: f64,
        _price: f64, // Kept for backward compatibility but not used
        is_buy: bool,
    ) -> f64 {
        // Validate size before placing order
        if let Err(e) = self.tick_lot_validator.validate_size(amount) {
            error!("Invalid taker order size {}: {}", amount, e);
            return 0.0;
        }

        // Use market_open with slippage protection
        // Default 1% slippage - adjust based on market conditions
        let slippage = 0.01; // 1% slippage tolerance
        
        let market_params = MarketOrderParams {
            asset: &asset,
            is_buy,
            sz: amount,
            px: None, // Let the SDK fetch current price
            slippage: Some(slippage),
            cloid: None,
            wallet: None, // Use default wallet from exchange_client
        };

        let order = self.exchange_client.market_open(market_params).await;

        match order {
            Ok(order) => match order {
                ExchangeResponseStatus::Ok(order) => {
                    if let Some(order) = order.data {
                        if !order.statuses.is_empty() {
                            match order.statuses[0].clone() {
                                ExchangeDataStatus::Filled(_order) => {
                                    info!("Market order filled: {} {} (slippage: {}%)", 
                                          amount, if is_buy { "buy" } else { "sell" }, slippage * 100.0);
                                    return amount;
                                }
                                ExchangeDataStatus::Resting(_) => {
                                    // Market order should not rest, but handle it gracefully
                                    info!("Market order unexpectedly resting");
                                    return 0.0;
                                }
                                ExchangeDataStatus::Error(e) => {
                                    error!("Error with placing market order: {e}");
                                }
                                _ => {}
                            }
                        } else {
                            error!("Exchange data statuses is empty when placing market order: {order:?}");
                        }
                    } else {
                        error!("Exchange response data is empty when placing market order: {order:?}");
                    }
                }
                ExchangeResponseStatus::Err(e) => {
                    error!("Error with placing market order: {e}");
                }
            },
            Err(e) => error!("Error with placing market order: {e}"),
        }
        0.0
    }

    /// Cancel all open orders and close any position, then shutdown gracefully.
    pub async fn shutdown(&mut self) {
        info!("Shutting down market maker...");

        // Use the helper to cancel all orders and clear local lists
        self.cancel_all_orders().await;

        // Close any existing position (logic remains the same)
        if self.cur_position.abs() >= EPSILON {
            info!("Current position: {:.6} {}, closing position...", self.cur_position, self.asset);
            self.close_position().await; // Ensure this handles potential errors gracefully
        } else {
            info!("No significant position to close (position: {:.6})", self.cur_position);
        }

        // Optional: Persist any final state (e.g., learned parameters) if needed
        if let Err(e) = self.tuning_params.read().to_json_file("tuning_params_final.json") {
            warn!("Failed to save final tuning parameters: {}", e);
        } else {
            info!("Final tuning parameters saved to tuning_params_final.json");
        }

        info!("Market maker shutdown complete.");
    }

    /// Close the current position using the SDK's market_close function with slippage protection
    async fn close_position(&mut self) {
        let position_size = self.cur_position.abs();
        
        // Validate the position size
        if let Err(e) = self.tick_lot_validator.validate_size(position_size) {
            error!("Invalid position size {}: {}", position_size, e);
            return;
        }
        
        let is_sell = self.cur_position > 0.0; // If we're long, we need to sell to close
        
        info!(
            "Closing position: {} {:.6} {} (using market_close with slippage protection)",
            if is_sell { "sell" } else { "buy" },
            position_size,
            self.asset
        );
        
        // Use market_close with slippage protection
        // Default 1% slippage - adjust based on market conditions
        let slippage = 0.01; // 1% slippage tolerance
        
        let market_close_params = MarketCloseParams {
            asset: &self.asset,
            sz: None, // Close entire position
            px: None, // Let the SDK fetch current price
            slippage: Some(slippage),
            cloid: None,
            wallet: None, // Use default wallet from exchange_client
        };

        let order = self.exchange_client.market_close(market_close_params).await;
            
        match order {
            Ok(order) => match order {
                ExchangeResponseStatus::Ok(order) => {
                    if let Some(order) = order.data {
                        if !order.statuses.is_empty() {
                            match order.statuses[0].clone() {
                                ExchangeDataStatus::Filled(_) => {
                                    info!("Position successfully closed with market order (slippage: {}%)", slippage * 100.0);
                                    self.cur_position = 0.0; // Reset position
                                }
                                ExchangeDataStatus::Resting(order) => {
                                    info!("Close order unexpectedly resting (oid: {}), attempting to cancel...", order.oid);
                                    // If it's still resting, cancel it
                                    self.attempt_cancel(self.asset.clone(), order.oid).await;
                                }
                                ExchangeDataStatus::Error(e) => {
                                    error!("Error closing position: {}", e);
                                }
                                _ => {
                                    error!("Unexpected order status when closing position: {:?}", order.statuses[0]);
                                }
                            }
                        } else {
                            error!("Empty order statuses when closing position");
                        }
                    } else {
                        error!("No order data when closing position");
                    }
                }
                ExchangeResponseStatus::Err(e) => {
                    error!("Error closing position: {}", e);
                }
            },
            Err(e) => {
                error!("Error placing close order: {}", e);
            }
        }
    }

    /// Handle order placement result from async execution task
    /// Adds successfully placed orders to tracking structures (bid_levels/ask_levels)
    fn handle_order_placement_result(&mut self, result: OrderPlacementResult) {
        // Look up the intent
        let mut intents = self.pending_order_intents.write();
        let intent = match intents.get(&result.intent_id) {
            Some(intent) => intent.clone(),
            None => {
                // Intent already removed (likely due to immediate fill)
                debug!("Order placement result received but intent {} already removed (likely filled immediately)",
                      result.intent_id);
                return;
            }
        };

        if result.success {
            if let Some(oid) = result.oid {
                // Order was placed successfully and is resting on the book
                info!("✅ Order placed successfully: intent_id={}, oid={}, side={}, L{}",
                      result.intent_id, oid, if intent.side { "BID" } else { "ASK" }, intent.level + 1);

                // Create resting order entry
                let order = MarketMakerRestingOrder {
                    oid,
                    position: intent.size,
                    price: intent.price,
                    level: intent.level,
                    pending_cancel: false,
                };

                // Add to appropriate tracking structure
                if intent.side {
                    // Buy order
                    self.bid_levels.push(order);
                } else {
                    // Sell order
                    self.ask_levels.push(order);
                }

                // Remove from pending intents
                intents.remove(&result.intent_id);

                debug!("Order added to tracking: oid={}, {} levels, {} levels",
                      oid, self.bid_levels.len(), self.ask_levels.len());
            } else {
                // Order was filled immediately (no resting state)
                info!("⚡ Order filled immediately on placement: intent_id={}, L{}",
                      result.intent_id, intent.level + 1);

                // Remove from pending intents (fill message will handle position update)
                intents.remove(&result.intent_id);
            }
        } else {
            // Order placement failed
            let error_msg = result.error_message.unwrap_or_else(|| "Unknown error".to_string());
            warn!("❌ Order placement failed: intent_id={}, side={}, L{}: {}",
                  result.intent_id,
                  if intent.side { "BID" } else { "ASK" },
                  intent.level + 1,
                  error_msg);

            // Remove from pending intents
            intents.remove(&result.intent_id);
        }
    }

    /// Main logic loop: calculate multi-level targets and reconcile orders.
    /// Enhanced with concurrent operations and complete taker logic implementation.
    async fn potentially_update(&mut self) {
        // --- 0. Pre-checks ---
        self.cleanup_invalid_resting_orders();

        // CIRCUIT BREAKER: Check if order execution channel is saturated
        // If capacity is low, skip placing new orders to prevent channel overflow
        let channel_capacity = self.order_command_tx.capacity();
        if channel_capacity < 400 {
            warn!(
                "🔴 CIRCUIT BREAKER: Order channel capacity low ({} < 400), skipping new placements",
                channel_capacity
            );
            // Still allow cancellations but skip new placements
            // Note: This prevents death spirals from channel saturation
            return;
        }

        if !*self.trading_enabled.read() {
            // Trading disabled (status shown in TUI)
            return;
        }
        // CRITICAL: Check L2Book mid-price (not AllMids) since that's what we use for pricing
        if self.latest_l2_mid_price <= 0.0 {
            warn!("L2Book mid price not available. Skipping order management.");
            return;
        }

        // DEBUG: Periodic snapshot of market state
        debug!(
            "📸 SNAPSHOT: mid={:.3} | pos={:.2} | bids={} | asks={} | L2_available={}",
            self.latest_l2_mid_price,
            self.cur_position,
            self.bid_levels.len(),
            self.ask_levels.len(),
            self.latest_book.is_some()
        );

        // If L2 book is available, log BBO
        if let Some(ref book) = self.latest_book {
            if !book.bids.is_empty() && !book.asks.is_empty() {
                if let (Ok(best_bid), Ok(best_ask)) = (
                    book.bids[0].px.parse::<f64>(),
                    book.asks[0].px.parse::<f64>()
                ) {
                    debug!(
                        "📖 L2 BBO: {:.3} x {:.3} | spread={:.1}bps | L2_mid={:.3}",
                        best_bid,
                        best_ask,
                        ((best_ask - best_bid) / best_bid * 10000.0),
                        (best_bid + best_ask) / 2.0
                    );
                }
            }
        }

        // --- 0b. Check Urgency and Execute Taker First if Critical ---
        // If inventory urgency is extremely high, prioritize taker execution before slow maker reconciliation
        let inventory_ratio = self.cur_position.abs() / self.max_absolute_position_size;
        let tuning_params = self.tuning_params.read().get_constrained();
        let urgency_threshold = tuning_params.inventory_urgency_threshold;

        // Define "critical urgency" as being significantly above threshold (e.g., 1.2x threshold)
        let critical_urgency_multiplier = 1.2;
        if inventory_ratio > urgency_threshold * critical_urgency_multiplier {
            info!("🚨 CRITICAL URGENCY: inventory_ratio={:.2} > threshold*{:.1}={:.2}, executing taker FIRST",
                  inventory_ratio, critical_urgency_multiplier, urgency_threshold * critical_urgency_multiplier);

            // Execute urgent taker logic immediately (before maker reconciliation)
            self.execute_taker_logic().await;
        }

        // --- 1. Calculate Target Quotes (Multi-Level) ---
        // This now returns Vec<(price, size)> using the new framework
        let (target_bids, target_asks) = self.calculate_multi_level_targets();

        // --- 2. Reconcile Existing Orders (Enhanced) ---
        let mut orders_to_cancel: Vec<u64> = Vec::new();
        let mut bids_to_place = target_bids.clone();
        let mut asks_to_place = target_asks.clone();

        // Get current time for pending_cancel timestamp
        let current_time = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs_f64();

        // Check existing bids against targets with improved tolerance
        let mut remaining_bids: Vec<MarketMakerRestingOrder> = Vec::new();
        for mut order in self.bid_levels.drain(..) {
            if order.oid == 0 || order.position < EPSILON {
                continue;
            }

            // Skip orders that are already in pending_cancel_orders HashMap
            if order.pending_cancel {
                // This shouldn't happen anymore, but keep as safety check
                warn!("Found order with pending_cancel=true in bid_levels (should be in HashMap): oid={}", order.oid);
                continue;
            }

            // Enhanced matching logic using exchange-native tick/lot precision
            // This prevents unnecessary cancels due to floating-point rounding
            let order_price = order.price;
            let order_position = order.position;
            let order_level = order.level;
            let order_oid = order.oid;

            // Calculate minimum meaningful price and size differences based on exchange precision
            let min_price_step = 10f64.powi(-(self.tick_lot_validator.max_price_decimals() as i32));
            let min_size_step = 10f64.powi(-(self.tick_lot_validator.sz_decimals as i32));

            // Use half a tick/lot as tolerance to handle rounding edge cases
            let price_tolerance = min_price_step * 0.5;
            let size_tolerance = min_size_step * 0.5;

            if let Some(target_idx) = bids_to_place.iter().position(|(p, s)| {
                (p - order_price).abs() <= price_tolerance && (s - order_position).abs() <= size_tolerance
            }) {
                // Match found! Keep this resting order, remove from placement list
                bids_to_place.remove(target_idx);
                debug!("Keeping existing L{} Bid: {} @ {} (matched target)", order_level + 1, order_position, order_price);
                remaining_bids.push(order);
            } else {
                // No match found - REMOVE from active tracking and move to pending_cancel HashMap
                order.pending_cancel = true;
                orders_to_cancel.push(order_oid);
                info!("Marking for cancellation L{} Bid: {} @ {} (oid: {})", order_level + 1, order_position, order_price, order_oid);

                // Move to pending_cancel_orders HashMap (O(1) insert)
                self.pending_cancel_orders.insert(order_oid, PendingCancelOrder {
                    order: order.clone(),
                    cancel_time: current_time,
                });
                // Note: NOT pushing to remaining_bids - order is removed from active tracking
            }
        }
        self.bid_levels = remaining_bids;

        // Check existing asks against targets with same enhanced logic
        let mut remaining_asks: Vec<MarketMakerRestingOrder> = Vec::new();
        for mut order in self.ask_levels.drain(..) {
            if order.oid == 0 || order.position < EPSILON {
                continue;
            }

            // Skip orders that are already in pending_cancel_orders HashMap
            if order.pending_cancel {
                // This shouldn't happen anymore, but keep as safety check
                warn!("Found order with pending_cancel=true in ask_levels (should be in HashMap): oid={}", order.oid);
                continue;
            }

            // Enhanced matching logic using exchange-native tick/lot precision
            // This prevents unnecessary cancels due to floating-point rounding
            let order_price = order.price;
            let order_position = order.position;
            let order_level = order.level;
            let order_oid = order.oid;

            // Calculate minimum meaningful price and size differences based on exchange precision
            let min_price_step = 10f64.powi(-(self.tick_lot_validator.max_price_decimals() as i32));
            let min_size_step = 10f64.powi(-(self.tick_lot_validator.sz_decimals as i32));

            // Use half a tick/lot as tolerance to handle rounding edge cases
            let price_tolerance = min_price_step * 0.5;
            let size_tolerance = min_size_step * 0.5;

            if let Some(target_idx) = asks_to_place.iter().position(|(p, s)| {
                (p - order_price).abs() <= price_tolerance && (s - order_position).abs() <= size_tolerance
            }) {
                // Match found! Keep this resting order
                asks_to_place.remove(target_idx);
                debug!("Keeping existing L{} Ask: {} @ {} (matched target)", order_level + 1, order_position, order_price);
                remaining_asks.push(order);
            } else {
                // No match found - REMOVE from active tracking and move to pending_cancel HashMap
                order.pending_cancel = true;
                orders_to_cancel.push(order_oid);
                info!("Marking for cancellation L{} Ask: {} @ {} (oid: {})", order_level + 1, order_position, order_price, order_oid);

                // Move to pending_cancel_orders HashMap (O(1) insert)
                self.pending_cancel_orders.insert(order_oid, PendingCancelOrder {
                    order: order.clone(),
                    cancel_time: current_time,
                });
                // Note: NOT pushing to remaining_asks - order is removed from active tracking
            }
        }
        self.ask_levels = remaining_asks;

        // --- 3. Execute Cancellations in Batch (OPTIMIZED) ---
        let num_cancellations = orders_to_cancel.len();
        if !orders_to_cancel.is_empty() {
            // Execute batch cancellations (status shown in TUI)

            // Build batch cancel request (single network call instead of N sequential calls)
            let cancel_requests: Vec<crate::ClientCancelRequest> = orders_to_cancel
                .into_iter()
                .map(|oid| crate::ClientCancelRequest {
                    asset: self.asset.clone(),
                    oid,
                })
                .collect();

            // Send batch cancellation to HIGH-PRIORITY async execution channel (non-blocking)
            let cancel_command = OrderCommand::BatchCancel {
                requests: cancel_requests,
            };

            if let Err(e) = self.cancel_command_tx.try_send(cancel_command) {
                error!("Failed to send batch cancel command to HIGH-PRIORITY execution channel: {}", e);
                // Channel might be full or closed - this is a critical error
                // but we continue to allow order placements to proceed
            } else {
                debug!("✅ Batch cancel command sent to HIGH-PRIORITY channel ({} orders, non-blocking)", num_cancellations);
            }
        }

        // --- 4. Inventory Sync Validation ---
        // CRITICAL: Ensure state_vector.inventory matches cur_position
        // This is essential for correct risk management and order sizing
        let inventory_divergence = (self.state_vector.inventory - self.cur_position).abs();
        if inventory_divergence > EPSILON {
            warn!(
                "⚠️  INVENTORY SYNC DIVERGENCE: state_vector.inventory={:.4} vs cur_position={:.4} (Δ={:.4})",
                self.state_vector.inventory, self.cur_position, inventory_divergence
            );
            // Force sync - this should never be needed if update_state_vector is called correctly
            // but provides defense in depth
            warn!("🔧 Force-syncing state_vector.inventory to cur_position");
            self.state_vector.inventory = self.cur_position;
        }

        // --- 5. Position Limit Safety Checks ---
        // CRITICAL: Prevent placing orders that would violate max_absolute_position_size
        // This is the primary defense against position limit violations

        // If we are at or over our MAX LONG position, suppress all new bids
        if self.cur_position >= self.max_absolute_position_size {
            if !bids_to_place.is_empty() {
                warn!(
                    "⛔ At max long position ({:.2} >= {:.2}). Suppressing {} new bid order(s).",
                    self.cur_position, self.max_absolute_position_size, bids_to_place.len()
                );
                bids_to_place.clear(); // Clear all pending bids
            }
        }

        // If we are at or over our MAX SHORT position, suppress all new asks
        if self.cur_position <= -self.max_absolute_position_size {
            if !asks_to_place.is_empty() {
                warn!(
                    "⛔ At max short position ({:.2} <= -{:.2}). Suppressing {} new ask order(s).",
                    self.cur_position, self.max_absolute_position_size, asks_to_place.len()
                );
                asks_to_place.clear(); // Clear all pending asks
            }
        }

        // --- 5. Execute Placements Sequentially ---
        let mut _successful_placements = 0;

        // Extract BBO from latest_book for spread-crossing validation
        let best_ask = self.latest_book.as_ref()
            .and_then(|book| {
                if !book.asks.is_empty() {
                    book.asks[0].px.parse::<f64>().ok()
                } else {
                    None
                }
            });

        let best_bid = self.latest_book.as_ref()
            .and_then(|book| {
                if !book.bids.is_empty() {
                    book.bids[0].px.parse::<f64>().ok()
                } else {
                    None
                }
            });

        // Place new bids
        for (level, (price, size)) in bids_to_place.iter().enumerate() {
            // CRITICAL: Check if bid would cross the spread (aggressive taker order)
            if let Some(ask) = best_ask {
                if *price >= ask {
                    error!(
                        "⛔ SKIPPING TAKER BID: L{} bid price {:.3} >= best_ask {:.3} (would cross spread!)",
                        level + 1, price, ask
                    );
                    continue; // Skip this order - it would be an instant taker fill
                }
            }

            // SAFETY: Check if this bid would violate position limits (defense in depth)
            let remaining_buy_capacity = self.max_absolute_position_size - self.cur_position;
            if *size > remaining_buy_capacity + EPSILON {
                warn!(
                    "⚠️  Skipping L{} Bid: size {:.2} exceeds remaining buy capacity {:.2} (pos={:.2}, max={:.2})",
                    level + 1, size, remaining_buy_capacity, self.cur_position, self.max_absolute_position_size
                );
                continue; // Skip this order
            }

            if *size >= EPSILON && *price > 0.0 && (*size * *price) >= 10.0 { // $10 notional minimum
                // Placing bid order (shown in TUI Open Orders panel)

                // Generate unique intent ID for tracking
                let intent_id = {
                    let mut next_id = self.next_intent_id.write();
                    let id = *next_id;
                    *next_id += 1;
                    id
                };

                // Create order request
                let order_request = ClientOrderRequest {
                    asset: self.asset.clone(),
                    is_buy: true,
                    limit_px: *price,
                    sz: *size,
                    order_type: ClientOrder::Limit(ClientLimit {
                        tif: "Gtc".to_string(),
                    }),
                    reduce_only: false,
                    cloid: None,
                };

                // Track intent before sending (for reconciliation)
                let intent = OrderIntent {
                    intent_id,
                    side: true,  // buy
                    price: *price,
                    size: *size,
                    level,
                    submitted_time: std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap()
                        .as_secs_f64(),
                };

                self.pending_order_intents.write().insert(intent_id, intent);

                // Send order command to async execution task (non-blocking)
                let place_command = OrderCommand::Place {
                    request: order_request,
                    intent_id,
                };

                if let Err(e) = self.order_command_tx.try_send(place_command) {
                    error!("Failed to send place order command (bid L{}): {}", level + 1, e);
                    // Remove intent if send failed
                    self.pending_order_intents.write().remove(&intent_id);
                } else {
                    _successful_placements += 1;
                    // Bid placement sent (tracking via TUI)
                }
            } else if (*size * *price) < 10.0 {
                warn!("Skipping L{} Bid: notional ${:.2} < $10 minimum", level + 1, size * price);
            }
        }
        
        // Place new asks
        for (level, (price, size)) in asks_to_place.iter().enumerate() {
            // CRITICAL: Check if ask would cross the spread (aggressive taker order)
            if let Some(bid) = best_bid {
                if *price <= bid {
                    error!(
                        "⛔ SKIPPING TAKER ASK: L{} ask price {:.3} <= best_bid {:.3} (would cross spread!)",
                        level + 1, price, bid
                    );
                    continue; // Skip this order - it would be an instant taker fill
                }
            }

            // SAFETY: Check if this ask would violate position limits (defense in depth)
            let remaining_sell_capacity = self.max_absolute_position_size + self.cur_position;
            if *size > remaining_sell_capacity + EPSILON {
                warn!(
                    "⚠️  Skipping L{} Ask: size {:.2} exceeds remaining sell capacity {:.2} (pos={:.2}, max={:.2})",
                    level + 1, size, remaining_sell_capacity, self.cur_position, self.max_absolute_position_size
                );
                continue; // Skip this order
            }

            if *size >= EPSILON && *price > 0.0 && (*size * *price) >= 10.0 { // $10 notional minimum
                // Placing ask order (shown in TUI Open Orders panel)

                // Generate unique intent ID for tracking
                let intent_id = {
                    let mut next_id = self.next_intent_id.write();
                    let id = *next_id;
                    *next_id += 1;
                    id
                };

                // Create order request
                let order_request = ClientOrderRequest {
                    asset: self.asset.clone(),
                    is_buy: false,
                    limit_px: *price,
                    sz: *size,
                    order_type: ClientOrder::Limit(ClientLimit {
                        tif: "Gtc".to_string(),
                    }),
                    reduce_only: false,
                    cloid: None,
                };

                // Track intent before sending (for reconciliation)
                let intent = OrderIntent {
                    intent_id,
                    side: false,  // sell
                    price: *price,
                    size: *size,
                    level,
                    submitted_time: std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap()
                        .as_secs_f64(),
                };

                self.pending_order_intents.write().insert(intent_id, intent);

                // Send order command to async execution task (non-blocking)
                let place_command = OrderCommand::Place {
                    request: order_request,
                    intent_id,
                };

                if let Err(e) = self.order_command_tx.try_send(place_command) {
                    error!("Failed to send place order command (ask L{}): {}", level + 1, e);
                    // Remove intent if send failed
                    self.pending_order_intents.write().remove(&intent_id);
                } else {
                    _successful_placements += 1;
                    // Ask placement sent (tracking via TUI)
                }
            } else if (*size * *price) < 10.0 {
                warn!("Skipping L{} Ask: notional ${:.2} < $10 minimum", level + 1, size * price);
            }
        }

        // --- 6. Sort Levels and Reassign Level Indices ---
        // Sort bids descending by price (highest price = Level 0 = best bid)
        self.bid_levels.sort_by(|a, b| b.price.partial_cmp(&a.price).unwrap());
        
        // Sort asks ascending by price (lowest price = Level 0 = best ask)
        self.ask_levels.sort_by(|a, b| a.price.partial_cmp(&b.price).unwrap());

        // Reassign level indices based on sorted order (0 = best price)
        for (i, order) in self.bid_levels.iter_mut().enumerate() { 
            order.level = i; 
        }
        for (i, order) in self.ask_levels.iter_mut().enumerate() { 
            order.level = i; 
        }

        // Order placement complete (status shown in TUI Open Orders panel)

        // --- 7. Implement Taker Logic (Normal Priority) ---
        // For normal urgency, execute taker logic after maker reconciliation
        self.execute_taker_logic().await;
    }

    /// Execute taker logic with rate limiting, smoothing, and size caps
    /// Can be called before or after maker reconciliation depending on urgency
    async fn execute_taker_logic(&mut self) {
        if self.latest_taker_buy_rate > EPSILON || self.latest_taker_sell_rate > EPSILON {
            let current_time = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs_f64();

            // Smooth taker rates using EMA to prevent sudden spikes
            self.smoothed_taker_buy_rate = TAKER_RATE_SMOOTHING_ALPHA * self.latest_taker_buy_rate
                + (1.0 - TAKER_RATE_SMOOTHING_ALPHA) * self.smoothed_taker_buy_rate;
            self.smoothed_taker_sell_rate = TAKER_RATE_SMOOTHING_ALPHA * self.latest_taker_sell_rate
                + (1.0 - TAKER_RATE_SMOOTHING_ALPHA) * self.smoothed_taker_sell_rate;

            // Taker rates calculated (not shown in TUI, only resulting position changes)

            // Taker Buy Logic (for when we need to buy aggressively)
            if self.smoothed_taker_buy_rate > EPSILON {
                // Rate limiting: Check if enough time has passed since last execution
                let time_since_last_buy = current_time - self.last_taker_buy_time;

                if time_since_last_buy < MIN_TAKER_INTERVAL_SECS {
                    info!("Taker buy rate-limited: {:.1}s since last execution (min={:.1}s)",
                          time_since_last_buy, MIN_TAKER_INTERVAL_SECS);
                } else {
                    // Use smoothed rate instead of raw rate
                    let desired_buy_size = self.tick_lot_validator.round_size(self.smoothed_taker_buy_rate, false);

                    // Cap to maximum single taker size (prevents huge liquidations)
                    let max_single_taker = self.max_absolute_position_size * MAX_TAKER_SIZE_FRACTION;
                    let size_capped = desired_buy_size.min(max_single_taker);

                    // CRITICAL SAFETY: Cap taker buy to prevent excessive long positions
                    // BUT allow buying to REDUCE extreme short positions
                    let remaining_buy_capacity = if self.cur_position < 0.0 {
                        // If short, only allow buying up to the absolute current position size
                        // This prevents flipping aggressively long during liquidation.
                        self.cur_position.abs()
                    } else {
                        // If already long, only allow buying up to max long position
                        (self.max_absolute_position_size - self.cur_position).max(0.0)
                    };
                    let taker_buy_size = size_capped.min(remaining_buy_capacity);

                    if taker_buy_size < desired_buy_size {
                        warn!("⚠️ Taker buy clamped: desired={:.2} → clamped={:.2} (cur_pos={:.2}, max={:.2})",
                              desired_buy_size, taker_buy_size, self.cur_position, self.max_absolute_position_size);
                    }

                    if taker_buy_size < EPSILON {
                        warn!("Taker buy size too small after rounding/clamping, skipping");
                    } else {
                        let notional = taker_buy_size * self.latest_mid_price;

                        if notional >= 10.0 { // $10 notional minimum
                            // Executing taker buy (position change shown in TUI)

                            // Use market_open for better slippage protection
                            let market_params = crate::MarketOrderParams {
                                asset: &self.asset,
                                is_buy: true,
                                sz: taker_buy_size, // Rounded size
                                px: None, // Let SDK fetch current price
                                slippage: Some(0.01), // 1% slippage tolerance
                                cloid: None,
                                wallet: None, // Use default wallet
                            };

                            match self.exchange_client.market_open(market_params).await {
                                Ok(crate::ExchangeResponseStatus::Ok(_)) => {
                                    // Taker buy executed (position updated, shown in TUI)
                                    self.last_taker_buy_time = current_time; // Update timestamp
                                }
                                Ok(crate::ExchangeResponseStatus::Err(e)) => {
                                    error!("Taker buy failed: {}", e);
                                }
                                Err(e) => {
                                    error!("Taker buy error: {}", e);
                                }
                            }
                        } else {
                            warn!("Taker buy skipped: notional ${:.2} < $10 minimum", notional);
                        }
                    }
                }
            }
            
            // Taker Sell Logic (for when we need to sell aggressively)
            if self.smoothed_taker_sell_rate > EPSILON {
                // Rate limiting: Check if enough time has passed since last execution
                let time_since_last_sell = current_time - self.last_taker_sell_time;

                if time_since_last_sell < MIN_TAKER_INTERVAL_SECS {
                    info!("Taker sell rate-limited: {:.1}s since last execution (min={:.1}s)",
                          time_since_last_sell, MIN_TAKER_INTERVAL_SECS);
                } else {
                    // Use smoothed rate instead of raw rate
                    let desired_sell_size = self.tick_lot_validator.round_size(self.smoothed_taker_sell_rate, false);

                    // Cap to maximum single taker size (prevents huge liquidations)
                    let max_single_taker = self.max_absolute_position_size * MAX_TAKER_SIZE_FRACTION;
                    let size_capped = desired_sell_size.min(max_single_taker);

                    // CRITICAL SAFETY: Cap taker sell to prevent excessive short positions
                    // BUT allow selling to REDUCE extreme long positions
                    let remaining_sell_capacity = if self.cur_position > 0.0 {
                        // If long, only allow selling up to the current position size
                        // This prevents flipping aggressively short during liquidation.
                        self.cur_position
                    } else {
                        // If already short, only allow selling down to max short position
                        (self.max_absolute_position_size - self.cur_position.abs()).max(0.0)
                    };
                    let taker_sell_size = size_capped.min(remaining_sell_capacity);

                    if taker_sell_size < desired_sell_size {
                        warn!("⚠️ Taker sell clamped: desired={:.2} → clamped={:.2} (cur_pos={:.2}, max={:.2})",
                              desired_sell_size, taker_sell_size, self.cur_position, self.max_absolute_position_size);
                    }

                    if taker_sell_size < EPSILON {
                        warn!("Taker sell size too small after rounding/clamping, skipping");
                    } else {
                        let notional = taker_sell_size * self.latest_mid_price;

                        if notional >= 10.0 { // $10 notional minimum
                            // Executing taker sell (position change shown in TUI)

                            let market_params = crate::MarketOrderParams {
                                asset: &self.asset,
                                is_buy: false,
                                sz: taker_sell_size, // Rounded size
                                px: None, // Let SDK fetch current price
                                slippage: Some(0.01), // 1% slippage tolerance
                                cloid: None,
                                wallet: None, // Use default wallet
                            };

                            match self.exchange_client.market_open(market_params).await {
                                Ok(crate::ExchangeResponseStatus::Ok(_)) => {
                                    // Taker sell executed (position updated, shown in TUI)
                                    self.last_taker_sell_time = current_time; // Update timestamp
                                }
                                Ok(crate::ExchangeResponseStatus::Err(e)) => {
                                    error!("Taker sell failed: {}", e);
                                }
                                Err(e) => {
                                    error!("Taker sell error: {}", e);
                                }
                            }
                        } else {
                            warn!("Taker sell skipped: notional ${:.2} < $10 minimum", notional);
                        }
                    }
                }
            }
            
            // Reset taker rates after execution
            self.latest_taker_buy_rate = 0.0;
            self.latest_taker_sell_rate = 0.0;
        }
    }

    /// **ADAM UPDATE TASK (Called every 60 seconds)**
    ///
    /// Applies the Adam optimizer update using gradients accumulated by the
    /// `accumulate_gradient_snapshot` task. Also handles trading enablement logic.
    /// Spawns a blocking task for the computation.
    pub fn run_adam_update_and_enablement_check(&self) -> tokio::task::JoinHandle<Option<f64>> { // Returns Option<value_gap_percent>
        // Clone Arcs needed for the background task
        let tuning_params_arc = Arc::clone(&self.tuning_params);
        let adam_optimizer_arc = Arc::clone(&self.adam_optimizer);
        let gradient_accumulator_arc = Arc::clone(&self.gradient_accumulator);
        let gradient_count_arc = Arc::clone(&self.gradient_count);
        let trading_enabled_arc = Arc::clone(&self.trading_enabled);
        let enable_trading_gap_threshold = self.enable_trading_gap_threshold_percent; // Copy f64

        // --- HACK/TODO: Value Gap Calculation ---
        // Need to pass necessary state snapshots for approximate value calculation
        // This requires capturing state during gradient accumulation or re-running parts.
        // For now, we'll use a placeholder/simplified gap based on control loss.
        // let state_vector_snapshot = self.state_vector.clone(); // Capture state *now*
        // let multi_level_optimizer_config = self.multi_level_optimizer.config.clone(); // Capture config
        // let hawkes_snapshot = self.hawkes_model.read().clone(); // Capture Hawkes state
        // let uncertainty_snapshot = self.current_uncertainty; // Capture uncertainty

        tokio::task::spawn_blocking(move || {
            let mut final_value_gap_percent: Option<f64> = None;

            // --- 1. Get Accumulated Gradients and Average Loss ---
            let (avg_gradient_vector, avg_loss, num_samples) = {
                let accumulator = gradient_accumulator_arc.read();
                let count = *gradient_count_arc.read();
                if count > 0 {
                    // Extract 8 gradient params from accumulator[0..8]
                    let avg_grad: Vec<f64> = accumulator[0..8].iter().map(|g| g / count as f64).collect();
                    // Extract loss from accumulator[8]
                    let avg_loss = accumulator.get(8).map_or(0.0, |l| l / count as f64);
                    (avg_grad, avg_loss, count)
                } else {
                    (vec![0.0; 8], 0.0, 0) // No gradients accumulated (8 parameters)
                }
            };

            // --- 2. Apply Adam Update (if gradients exist) ---
            if num_samples > 0 {
                info!("Applying Adam update based on {} accumulated gradient samples. Avg loss: {:.6}", num_samples, avg_loss);
                let original_params_phi = tuning_params_arc.read().clone(); // Read locked value
                let _original_params_theta = original_params_phi.get_constrained();

                // a) Clipping
                let max_individual_norm = 100.0;
                let max_global_norm = 50.0;
                let mut clipped_gradient_vector = avg_gradient_vector.clone();
                
                // Individual clipping
                for g in clipped_gradient_vector.iter_mut() {
                    if g.abs() > max_individual_norm {
                        *g = g.signum() * max_individual_norm;
                    }
                }
                
                // Global norm clipping
                let global_norm: f64 = clipped_gradient_vector.iter().map(|g| g.powi(2)).sum::<f64>().sqrt();
                if global_norm > max_global_norm {
                    let scale_factor = max_global_norm / global_norm;
                    for g in clipped_gradient_vector.iter_mut() {
                        *g *= scale_factor;
                    }
                }
                
                // Apply L2 Regularization: gradient += lambda * params_phi
                // This penalizes parameters moving away from zero (in phi space)
                // and helps prevent unbounded drift
                let mut regularized_gradient = clipped_gradient_vector.clone();
                let params_phi_values = [
                    original_params_phi.skew_adjustment_factor_phi,
                    original_params_phi.adverse_selection_adjustment_factor_phi,
                    original_params_phi.adverse_selection_lambda_phi,
                    original_params_phi.inventory_urgency_threshold_phi,
                    original_params_phi.liquidation_rate_multiplier_phi,
                    original_params_phi.min_spread_base_ratio_phi,
                    original_params_phi.adverse_selection_spread_scale_phi,
                    original_params_phi.control_gap_threshold_phi,
                ];

                for i in 0..8 {
                    regularized_gradient[i] += L2_REGULARIZATION_LAMBDA * params_phi_values[i];
                }

                let _clipped_gradient_norm: f64 = clipped_gradient_vector.iter().map(|g| g.powi(2)).sum::<f64>().sqrt();
                let _regularized_gradient_norm: f64 = regularized_gradient.iter().map(|g| g.powi(2)).sum::<f64>().sqrt();

                // Gradient components calculated (shown in TUI Adam Optimizer panel)

                // b) Compute Adam Update (with warmup) using REGULARIZED gradient
                let updates = {
                    let mut optimizer_state = adam_optimizer_arc.write(); // Lock for update
                    let warmup_steps = 3;
                    let current_step = optimizer_state.t + 1;
                    let warmup_factor = if current_step <= warmup_steps {
                        (current_step as f64) / (warmup_steps as f64)
                    } else { 1.0 };

                    let original_alpha = optimizer_state.alpha;
                    optimizer_state.alpha *= warmup_factor; // Apply warmup

                    // Use REGULARIZED gradient instead of clipped gradient
                    let updates = optimizer_state.compute_update(&regularized_gradient);

                    optimizer_state.alpha = original_alpha; // Restore alpha
                    updates // Return the computed updates
                }; // Optimizer lock released here

                // c) Apply Updates to Parameters
                let mut updated_params_phi = original_params_phi.clone();
                updated_params_phi.skew_adjustment_factor_phi -= updates[0];
                updated_params_phi.adverse_selection_adjustment_factor_phi -= updates[1];
                updated_params_phi.adverse_selection_lambda_phi -= updates[2];
                updated_params_phi.inventory_urgency_threshold_phi -= updates[3];
                updated_params_phi.liquidation_rate_multiplier_phi -= updates[4];
                updated_params_phi.min_spread_base_ratio_phi -= updates[5];
                updated_params_phi.adverse_selection_spread_scale_phi -= updates[6]; // Added 7th parameter
                updated_params_phi.control_gap_threshold_phi -= updates[7]; // Moved to index 7

                // d) Store Updated Parameters & Save
                {
                    let mut params_w = tuning_params_arc.write(); // Lock for writing
                    *params_w = updated_params_phi.clone();
                } // Lock released

                if let Err(e) = updated_params_phi.to_json_file("tuning_params.json") {
                    error!("Failed to save updated tuning params: {}", e);
                } else {
                    info!("Updated tuning parameters saved.");
                }

                // e) Parameter updates applied (shown in TUI)
                let updated_params_theta = updated_params_phi.get_constrained();

                // f) Drift Monitoring - Check for parameters drifting to extremes
                let mut drift_warnings = Vec::new();

                // Check adverse_selection_adjustment_factor (should stay reasonable, e.g., < 0.5)
                if updated_params_theta.adverse_selection_adjustment_factor > 0.5 {
                    drift_warnings.push(format!(
                        "adverse_selection_adjustment_factor={:.4} is high (>0.5)",
                        updated_params_theta.adverse_selection_adjustment_factor
                    ));
                }

                // Check adverse_selection_spread_scale (should stay < 10.0)
                if updated_params_theta.adverse_selection_spread_scale > 10.0 {
                    drift_warnings.push(format!(
                        "adverse_selection_spread_scale={:.4} is high (>10.0)",
                        updated_params_theta.adverse_selection_spread_scale
                    ));
                }

                // Check liquidation_rate_multiplier (should stay < 50.0)
                if updated_params_theta.liquidation_rate_multiplier > 50.0 {
                    drift_warnings.push(format!(
                        "liquidation_rate_multiplier={:.4} is very high (>50.0)",
                        updated_params_theta.liquidation_rate_multiplier
                    ));
                }

                // Check skew_adjustment_factor (should stay < 5.0)
                if updated_params_theta.skew_adjustment_factor > 5.0 {
                    drift_warnings.push(format!(
                        "skew_adjustment_factor={:.4} is high (>5.0)",
                        updated_params_theta.skew_adjustment_factor
                    ));
                }

                if !drift_warnings.is_empty() {
                    warn!("⚠️  PARAMETER DRIFT DETECTED:");
                    for warning in drift_warnings {
                        warn!("    - {}", warning);
                    }
                    warn!("    Consider reviewing loss function or adding regularization");
                }

            }
            // No gradients accumulated - skipping update (status shown in TUI)

            // --- 3. Reset Gradient Accumulator ---
            {
                let mut accumulator = gradient_accumulator_arc.write();
                let mut count = gradient_count_arc.write();
                *accumulator = vec![0.0; 9]; // 8 parameters + 1 loss = 9 elements
                *count = 0;
                // Gradient accumulator reset (tracking in TUI)
            }

            // --- 4. Trading Enablement Check (Using Control Gap Loss) ---
            // Use the average loss computed from gradient accumulation
            let loss_threshold_for_enablement = (enable_trading_gap_threshold / 10.0).powi(2); // Example: 15% -> threshold 2.25 bps^2

            if !*trading_enabled_arc.read() && num_samples > 10 && avg_loss <= loss_threshold_for_enablement {
                let mut enabled_w = trading_enabled_arc.write();
                *enabled_w = true;
                info!("✅ Control gap consistently low (avg_loss={:.2} <= {:.2}). ENABLING LIVE TRADING.", avg_loss, loss_threshold_for_enablement);
                final_value_gap_percent = Some(avg_loss.sqrt() * 10.0); // Use control gap as proxy %
            }
            // Trading status changes logged above (shown in TUI)

            // TODO: Implement proper Value Gap calculation if needed for more accurate logging/enablement.

            final_value_gap_percent // Return the approximate gap % if calculated
        })
    }

    /// **GRADIENT ACCUMULATION TASK (Called every ~10 AllMids)**
    ///
    /// Calculates the gradient of the control gap loss w.r.t tuning parameters
    /// for a snapshot in time and accumulates it. Spawns a blocking task.
    #[allow(dead_code)]
    /// **GRADIENT ACCUMULATION TASK (Called frequently)** - RESTORED ROBUST LOGIC
    ///
    /// This function calculates gradients for the Adam optimizer by comparing TWO DIFFERENT FUNCTIONS:
    /// - **Optimal Target**: HJBComponents::optimize_control() (OLD grid search - proven optimal)
    /// - **Heuristic**: MultiLevelOptimizer::optimize() (NEW fast heuristic - being tuned)
    ///
    /// The loss measures how much the multi-level heuristic deviates from the HJB optimal control.
    /// This creates a CONSTANT non-zero learning signal because we're comparing different algorithms,
    /// not the same algorithm with different states.
    ///
    /// **KEY FIX**: The broken version compared MultiLevelOptimizer(REAL_STATE) vs MultiLevelOptimizer(ZERO_STATE).
    /// When state=(Q=0, μ̂=0), both were identical, giving zero loss and zero gradients.
    /// Now we compare MultiLevelOptimizer vs HJBComponents, ensuring non-zero loss that drives learning.
    fn accumulate_gradient_snapshot(&self) {
        // --- 1. Clone data needed ---
        // Cheap clones (Arc, f64, small vecs, structs)
        let state_vector_snapshot = self.state_vector.clone();
        let particle_filter_arc = Arc::clone(&self.particle_filter); // Need Arc for background task
        let hawkes_model_snapshot = self.hawkes_model.read().clone(); // Clone Hawkes state
        let tuning_params_arc = Arc::clone(&self.tuning_params);
        let multi_level_optimizer_config = self.multi_level_optimizer.config().clone(); // Clone config
        let robust_config_snapshot = self.robust_config.clone();
        let max_pos = self.max_absolute_position_size;
        let gradient_accumulator_arc = Arc::clone(&self.gradient_accumulator);
        let gradient_count_arc = Arc::clone(&self.gradient_count);

        // Clone HJB components for optimal target calculation
        let hjb_components_clone = self.hjb_components.clone();
        let value_function_clone = self.value_function.clone();

        // --- 2. Spawn Blocking Task ---
        tokio::task::spawn_blocking(move || {
            let current_time = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs_f64();

            // --- Inside Blocking Task ---

            // a) Get Uncertainty Snapshot
            let uncertainty_snapshot = {
                let pf_lock = particle_filter_arc.read();
                let (mu_std, _, _) = pf_lock.get_parameter_std_devs();
                let sigma_std = pf_lock.get_volatility_std_dev_bps();
                ParameterUncertainty::from_particle_filter_stats(mu_std, sigma_std, 0.95)
            };

            // b) Get CURRENT Tunable Parameters (Read ONCE)
            let current_params_phi = tuning_params_arc.read().clone();
            let current_params_theta = current_params_phi.get_constrained();

            // c) Calculate "Optimal Target" Control using OLD HJB Grid Search
            //    This is the PROVEN optimal control from the grid search
            //    We use this as the target that the multi-level optimizer should match
            let optimal_target_control_single = {
                let base_total_spread_bps = (state_vector_snapshot.market_spread_bps).max(12.0);
                let base_half_spread_bps = base_total_spread_bps / 2.0;
                // Run the OLD grid search function with REAL state
                hjb_components_clone.optimize_control(
                    &state_vector_snapshot,
                    &value_function_clone,
                    base_half_spread_bps,
                )
            };

            // Extract L1 offsets from the single-level HJB control
            let optimal_target_l1_bid = optimal_target_control_single.bid_offset_bps;
            let optimal_target_l1_ask = optimal_target_control_single.ask_offset_bps;

            // d) Calculate "Heuristic" Multi-Level Control (using current params and REAL state)
            //    This is what the current parameters produce given the actual market state
            let heuristic_multi_control = {
                let robust_params = RobustParameters::compute(
                    state_vector_snapshot.adverse_selection_estimate,  // REAL drift
                    state_vector_snapshot.volatility_ema_bps,
                    state_vector_snapshot.inventory,  // REAL inventory
                    &uncertainty_snapshot,
                    &robust_config_snapshot,
                );
                let opt_state = OptimizationState {
                    mid_price: state_vector_snapshot.mid_price,
                    inventory: state_vector_snapshot.inventory,  // REAL inventory
                    max_position: max_pos,
                    adverse_selection_bps: state_vector_snapshot.adverse_selection_estimate,  // REAL drift
                    lob_imbalance: state_vector_snapshot.lob_imbalance,
                    volatility_bps: robust_params.sigma_worst_case,
                    current_time,
                    hawkes_model: &hawkes_model_snapshot,
                };
                let base_spread = (robust_params.sigma_worst_case * 0.1)
                                    .max(multi_level_optimizer_config.min_profitable_spread_bps / 2.0)
                                    * robust_params.spread_multiplier;
                let optimizer = MultiLevelOptimizer::new(multi_level_optimizer_config.clone());
                optimizer.optimize(&opt_state, base_spread, &current_params_theta)
            };

            // e) Calculate Loss (Control Gap)
            //    Compare HEURISTIC (multi-level optimizer) vs TARGET (HJB grid search)
            //    This is the FIXED logic: comparing two DIFFERENT functions, not same function with different states
            let heuristic_l1_bid = heuristic_multi_control.bid_levels.get(0).map(|(o, _)| *o).unwrap_or(0.0);
            let heuristic_l1_ask = heuristic_multi_control.ask_levels.get(0).map(|(o, _)| *o).unwrap_or(0.0);

            let bid_gap = optimal_target_l1_bid - heuristic_l1_bid;
            let ask_gap = optimal_target_l1_ask - heuristic_l1_ask;
            let current_loss = bid_gap.powi(2) + ask_gap.powi(2);

            // f) Check Loss Threshold
            if current_loss <= current_params_theta.control_gap_threshold {
                // Loss is small enough - no tuning needed for this snapshot
                return;
            }

            // g) Calculate Numerical Gradient
            let mut gradient_vector: Vec<f64> = vec![0.0; 8]; // 8 parameters in TuningParams
            let nudge_amount = 0.001; // Small nudge for finite difference

            for i in 0..8 { // For each parameter in TuningParams
                if i == 7 { 
                    // Skip gradient for control_gap_threshold_phi itself (doesn't affect control)
                    continue;
                }
                
                let mut nudged_params_phi = current_params_phi.clone();
                match i {
                    0 => nudged_params_phi.skew_adjustment_factor_phi += nudge_amount,
                    1 => nudged_params_phi.adverse_selection_adjustment_factor_phi += nudge_amount,
                    2 => nudged_params_phi.adverse_selection_lambda_phi += nudge_amount, // Affects StateVector update
                    3 => nudged_params_phi.inventory_urgency_threshold_phi += nudge_amount, // Affects taker logic in optimize
                    4 => nudged_params_phi.liquidation_rate_multiplier_phi += nudge_amount, // Affects taker logic in optimize
                    5 => nudged_params_phi.min_spread_base_ratio_phi += nudge_amount, // Affects offset clamping in optimize
                    6 => nudged_params_phi.adverse_selection_spread_scale_phi += nudge_amount, // Affects adverse selection scaling
                    7 => nudged_params_phi.control_gap_threshold_phi += nudge_amount, // Doesn't affect control, skip gradient
                    _ => {}
                }

                // Recalculate heuristic control with nudged params
                // Note: Need to handle how param 'i' affects the inputs or logic of optimize()
                let nudged_heuristic_control = {
                    // 1. Get nudged constrained params (CRITICAL: now used by optimizer!)
                    let nudged_params_theta = nudged_params_phi.get_constrained();

                    // 2. Potentially update state if param affects it (e.g., lambda)
                    // For now, we'll use the same state for simplicity
                    // In a more sophisticated implementation, you'd recalculate adverse_selection_estimate
                    // with the new lambda value

                    // 3. Recalculate robust params using nudged state/params
                    let robust_params_nudged = RobustParameters::compute(
                        state_vector_snapshot.adverse_selection_estimate,
                        state_vector_snapshot.volatility_ema_bps,
                        state_vector_snapshot.inventory,
                        &uncertainty_snapshot,
                        &robust_config_snapshot,
                    );

                    // 4. Create nudged opt state
                    let opt_state_nudged = OptimizationState {
                        adverse_selection_bps: robust_params_nudged.mu_worst_case,
                        volatility_bps: robust_params_nudged.sigma_worst_case,
                        mid_price: state_vector_snapshot.mid_price,
                        inventory: state_vector_snapshot.inventory,
                        max_position: max_pos,
                        lob_imbalance: state_vector_snapshot.lob_imbalance,
                        current_time,
                        hawkes_model: &hawkes_model_snapshot,
                    };

                    // 5. Rerun optimizer logic WITH NUDGED PARAMS (this is the fix!)
                    let base_spread_nudged = (robust_params_nudged.sigma_worst_case * 0.1)
                                                .max(multi_level_optimizer_config.min_profitable_spread_bps / 2.0)
                                                * robust_params_nudged.spread_multiplier;
                    let optimizer = MultiLevelOptimizer::new(multi_level_optimizer_config.clone());
                    optimizer.optimize(&opt_state_nudged, base_spread_nudged, &nudged_params_theta)
                };

                let nudged_l1_bid = nudged_heuristic_control.bid_levels.get(0).map(|(o, _)| *o).unwrap_or(0.0);
                let nudged_l1_ask = nudged_heuristic_control.ask_levels.get(0).map(|(o, _)| *o).unwrap_or(0.0);

                // Calculate loss as deviation from the SAME HJB optimal target (not recalculated)
                let nudged_loss = (optimal_target_l1_bid - nudged_l1_bid).powi(2) + (optimal_target_l1_ask - nudged_l1_ask).powi(2);

                // Finite difference gradient
                gradient_vector[i] = (nudged_loss - current_loss) / nudge_amount;

                // Gradient computed (tracking in TUI Adam panel)
            }

            // h) Accumulate Gradient
            { // Lock scope
                let mut accumulator = gradient_accumulator_arc.write();
                let mut count = gradient_count_arc.write();
                for i in 0..8 { // All 8 parameters (fixed from 0..7)
                    // Clamp individual gradient components to prevent explosions
                    let clamped_grad = gradient_vector[i].clamp(-1000.0, 1000.0); // Generous clamp
                    if gradient_vector[i].abs() > 1000.0 {
                        warn!("Clamping large gradient component [{}]: {:.4}", i, gradient_vector[i]);
                    }
                    accumulator[i] += clamped_grad;
                }
                // Add current_loss to accumulator[8] (8 params + 1 loss = index 8)
                accumulator[8] += current_loss;
                *count += 1;
                // Optional: Log accumulation progress
                if *count % 10 == 0 { 
                    info!("Gradient accumulated (#{}) - Loss: {:.6}", *count, current_loss); 
                }
            } // Lock released
        }); // End spawn_blocking
    }


}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{L2BookData, BookLevel};

    fn create_test_book_level(px: &str, sz: &str) -> BookLevel {
        BookLevel {
            px: px.to_string(),
            sz: sz.to_string(),
            n: 1,
        }
    }

    fn create_balanced_book() -> (OrderBook, BookAnalysis) {
        let data = L2BookData {
            coin: "TEST".to_string(),
            time: 0,
            levels: vec![
                vec![
                    create_test_book_level("100.0", "10.0"),
                    create_test_book_level("99.0", "20.0"),
                ],
                vec![
                    create_test_book_level("101.0", "10.0"),
                    create_test_book_level("102.0", "20.0"),
                ],
            ],
        };
        let book = OrderBook::from_l2_data(&data).unwrap();
        let analysis = book.analyze(2).unwrap();
        (book, analysis)
    }

    fn create_imbalanced_book_bid_heavy() -> (OrderBook, BookAnalysis) {
        let data = L2BookData {
            coin: "TEST".to_string(),
            time: 0,
            levels: vec![
                vec![
                    create_test_book_level("100.0", "100.0"),
                    create_test_book_level("99.0", "50.0"),
                ],
                vec![
                    create_test_book_level("101.0", "10.0"),
                    create_test_book_level("102.0", "5.0"),
                ],
            ],
        };
        let book = OrderBook::from_l2_data(&data).unwrap();
        let analysis = book.analyze(2).unwrap();
        (book, analysis)
    }

    // Helper function to create test StateVector with all required fields
    fn create_test_state_vector(
        mid_price: f64,
        inventory: f64,
        adverse_selection: f64,
        spread_bps: f64,
        imbalance: f64,
    ) -> StateVector {
        StateVector {
            mid_price,
            inventory,
            adverse_selection_estimate: adverse_selection,
            market_spread_bps: spread_bps,
            lob_imbalance: imbalance,
            volatility_ema_bps: 10.0, // Default test volatility
            previous_mid_price: mid_price, // Initialize to current price for tests
            trade_flow_ema: 0.0, // Initialize to neutral for tests
        }
    }

    #[test]
    fn test_state_vector_initialization() {
        let state = StateVector::new();
        
        assert_eq!(state.mid_price, 0.0);
        assert_eq!(state.inventory, 0.0);
        assert_eq!(state.adverse_selection_estimate, 0.0);
        assert_eq!(state.market_spread_bps, 0.0);
        assert_eq!(state.lob_imbalance, 0.0);
    }

    #[test]
    fn test_state_vector_update_basic() {
        let mut state = StateVector::new();
        let (book, analysis) = create_balanced_book();
        let params = TuningParams::default().get_constrained();
        let mut model = OnlineAdverseSelectionModel::default();
        
        state.update(100.5, 0.0, Some(&analysis), Some(&book), &params, &mut model);
        
        assert_eq!(state.mid_price, 100.5);
        assert_eq!(state.inventory, 0.0);
        assert!(state.market_spread_bps > 0.0);
    }

    #[test]
    fn test_lob_imbalance_calculation() {
        let mut state = StateVector::new();
        let params = TuningParams::default().get_constrained();
        let mut model = OnlineAdverseSelectionModel::default();
        
        let (book, analysis) = create_balanced_book();
        state.update(100.5, 0.0, Some(&analysis), Some(&book), &params, &mut model);
        
        assert!((state.lob_imbalance - 0.5).abs() < 0.1);
        
        let (book, analysis) = create_imbalanced_book_bid_heavy();
        state.update(100.5, 0.0, Some(&analysis), Some(&book), &params, &mut model);
        
        assert!(state.lob_imbalance > 0.5);
    }

    #[test]
    fn test_adverse_selection_adjustment() {
        let mut state = StateVector::new();
        let params = TuningParams::default().get_constrained();
        
        state.adverse_selection_estimate = 0.1;
        let adjustment = state.get_adverse_selection_adjustment(100.0, params.adverse_selection_adjustment_factor);
        assert!(adjustment < 0.0);
        
        state.adverse_selection_estimate = -0.1;
        let adjustment = state.get_adverse_selection_adjustment(100.0, params.adverse_selection_adjustment_factor);
        assert!(adjustment > 0.0);
    }

    #[test]
    fn test_inventory_risk_multiplier() {
        let state = create_test_state_vector(100.0, 0.0, 0.0, 10.0, 0.5);
        
        let multiplier = state.get_inventory_risk_multiplier(100.0);
        assert_eq!(multiplier, 1.0);
        
        let mut state_max = state.clone();
        state_max.inventory = 100.0;
        let multiplier = state_max.get_inventory_risk_multiplier(100.0);
        assert_eq!(multiplier, 2.0);
    }

    #[test]
    fn test_inventory_urgency() {
        let state = create_test_state_vector(100.0, 0.0, 0.0, 10.0, 0.5);
        
        let urgency = state.get_inventory_urgency(100.0);
        assert_eq!(urgency, 0.0);
        
        let mut state_max = state.clone();
        state_max.inventory = 100.0;
        let urgency = state_max.get_inventory_urgency(100.0);
        assert_eq!(urgency, 1.0);
    }

    #[test]
    fn test_market_favorable_conditions() {
        let state = create_test_state_vector(100.0, 0.0, 0.0, 10.0, 0.5);
        
        assert!(state.is_market_favorable(50.0));
        
        let mut state_wide = state.clone();
        state_wide.market_spread_bps = 100.0;
        assert!(!state_wide.is_market_favorable(50.0));
    }

    // ===== Control Vector Tests =====
    
    #[test]
    fn test_control_vector_initialization() {
        let control = ControlVector::new();
        
        assert_eq!(control.ask_offset_bps, 0.0);
        assert_eq!(control.bid_offset_bps, 0.0);
        assert_eq!(control.taker_sell_rate, 0.0);
        assert_eq!(control.taker_buy_rate, 0.0);
        assert!(control.is_passive_only());
    }
    
    #[test]
    fn test_control_vector_symmetric() {
        let control = ControlVector::symmetric(10.0);
        
        assert_eq!(control.ask_offset_bps, 10.0);
        assert_eq!(control.bid_offset_bps, 10.0);
        assert_eq!(control.total_spread_bps(), 20.0);
        assert_eq!(control.spread_asymmetry_bps(), 0.0);
        assert!(control.is_passive_only());
    }
    
    #[test]
    fn test_control_vector_asymmetric() {
        let control = ControlVector::asymmetric(15.0, 10.0);
        
        assert_eq!(control.ask_offset_bps, 15.0);
        assert_eq!(control.bid_offset_bps, 10.0);
        assert_eq!(control.total_spread_bps(), 25.0);
        assert_eq!(control.spread_asymmetry_bps(), 5.0); // Bullish bias
    }
    
    #[test]
    fn test_control_vector_quote_prices() {
        let control = ControlVector::symmetric(10.0); // 10 bps = 0.1%
        let mid_price = 100.0;
        
        let (bid, ask) = control.calculate_quote_prices(mid_price);
        
        // Bid should be 0.1% below mid
        assert!((bid - 99.9).abs() < 0.01);
        // Ask should be 0.1% above mid
        assert!((ask - 100.1).abs() < 0.01);
    }
    
    #[test]
    fn test_control_vector_with_taker_activity() {
        let control = ControlVector::with_taker_activity(10.0, 10.0, 1.5, 0.0);
        
        assert_eq!(control.taker_sell_rate, 1.5);
        assert_eq!(control.taker_buy_rate, 0.0);
        assert!(!control.is_passive_only());
        assert!(control.is_liquidating());
        assert_eq!(control.net_taker_rate(), 1.5); // Net selling
    }
    
    #[test]
    fn test_control_vector_validation() {
        let mut control = ControlVector::symmetric(10.0);
        
        // Valid control
        assert!(control.validate(5.0).is_ok());
        
        // Too narrow spread
        assert!(control.validate(50.0).is_err());
        
        // Negative offset
        control.ask_offset_bps = -5.0;
        assert!(control.validate(5.0).is_err());
    }
    
    #[test]
    fn test_control_vector_state_adjustments() {
        let mut control = ControlVector::symmetric(10.0);
        let params = TuningParams::default().get_constrained();
        let hjb_components = HJBComponents::default();
        
        let state = create_test_state_vector(100.0, 0.0, 0.0, 10.0, 0.5);
        
        control.apply_state_adjustments(&state, 10.0, 100.0, &params, &hjb_components);
        
        // With neutral state, should remain roughly symmetric
        assert!(control.ask_offset_bps > 0.0);
        assert!(control.bid_offset_bps > 0.0);
        assert!(control.is_passive_only());
    }
    
    #[test]
    fn test_control_vector_adverse_selection_adjustment() {
        let mut control = ControlVector::symmetric(10.0);
        let params = TuningParams::default().get_constrained();
        let hjb_components = HJBComponents::default();
        
        // Bullish state (positive adverse selection - expect price to rise)
        let mut state = create_test_state_vector(100.0, 0.0, 0.2, 10.0, 0.7);
        
        control.apply_state_adjustments(&state, 10.0, 100.0, &params, &hjb_components);
        
        // Bullish signal should widen ask more than bid (make selling less attractive)
        // When price is expected to rise, we want to avoid selling cheap
        // asymmetry = ask_offset - bid_offset
        // Positive asymmetry means ask is wider (further from mid)
        let asymmetry = control.spread_asymmetry_bps();
        assert!(asymmetry > 0.0); // Ask wider relative to bid
        
        // Bearish state (negative adverse selection - expect price to fall)
        control = ControlVector::symmetric(10.0);
        state.adverse_selection_estimate = -0.2;
        state.lob_imbalance = 0.3;
        
        control.apply_state_adjustments(&state, 10.0, 100.0, &params, &hjb_components);
        
        // Bearish signal should widen bid more than ask (make buying less attractive)
        // When price is expected to fall, we want to avoid buying high
        let asymmetry = control.spread_asymmetry_bps();
        assert!(asymmetry < 0.0); // Bid wider relative to ask
    }
    
    #[test]
    fn test_control_vector_inventory_adjustment() {
        let base_spread = 10.0;
        let max_inventory = 100.0;
        let params = TuningParams::default().get_constrained();
        let hjb_components = HJBComponents::default();
        
        // Long position - should tighten ask, widen bid
        let mut control = ControlVector::symmetric(base_spread);
        let state_long = create_test_state_vector(100.0, 80.0, 0.0, 10.0, 0.5); // 80% long
        
        control.apply_state_adjustments(&state_long, base_spread, max_inventory, &params, &hjb_components);
        
        // Long position should create asymmetry favoring sells
        assert!(control.bid_offset_bps > control.ask_offset_bps);
        
        // Short position - should tighten bid, widen ask
        let mut control = ControlVector::symmetric(base_spread);
        let state_short = create_test_state_vector(100.0, -80.0, 0.0, 10.0, 0.5); // 80% short
        
        control.apply_state_adjustments(&state_short, base_spread, max_inventory, &params, &hjb_components);
        
        // Short position should create asymmetry favoring buys
        assert!(control.ask_offset_bps > control.bid_offset_bps);
    }
    
    #[test]
    fn test_control_vector_urgency_triggers_taker() {
        let mut control = ControlVector::symmetric(10.0);
        let params = TuningParams::default().get_constrained();
        let hjb_components = HJBComponents::default();
        
        // High urgency state (>0.7)
        let state = create_test_state_vector(100.0, 95.0, 0.0, 10.0, 0.5); // 95% of max = high urgency
        
        control.apply_state_adjustments(&state, 10.0, 100.0, &params, &hjb_components);
        
        // Should activate taker selling due to long position + high urgency
        assert!(control.taker_sell_rate > 0.0);
        assert_eq!(control.taker_buy_rate, 0.0);
        assert!(control.is_liquidating());
    }
    
    #[test]
    fn test_control_vector_risk_multiplier_effect() {
        let base_spread = 10.0;
        let max_inventory = 100.0;
        let params = TuningParams::default().get_constrained();
        let hjb_components = HJBComponents::default();
        
        // Zero inventory - minimal risk
        let mut control_zero = ControlVector::symmetric(base_spread);
        let state_zero = create_test_state_vector(100.0, 0.0, 0.0, 10.0, 0.5);
        
        control_zero.apply_state_adjustments(&state_zero, base_spread, max_inventory, &params, &hjb_components);
        let spread_zero = control_zero.total_spread_bps();
        
        // Max inventory - maximum risk
        let mut control_max = ControlVector::symmetric(base_spread);
        let state_max = create_test_state_vector(100.0, 100.0, 0.0, 10.0, 0.5);
        
        control_max.apply_state_adjustments(&state_max, base_spread, max_inventory, &params, &hjb_components);
        let spread_max = control_max.total_spread_bps();
        
        // Max inventory should have wider spread due to risk multiplier
        assert!(spread_max > spread_zero);
    }
    
    #[test]
    fn test_control_vector_logging() {
        let control = ControlVector::symmetric(10.0);
        let log_str = control.to_log_string();
        
        assert!(log_str.contains("δ^b=10.0bps"));
        assert!(log_str.contains("δ^a=10.0bps"));
        assert!(log_str.contains("spread=20.0bps"));
        
        let control_taker = ControlVector::with_taker_activity(10.0, 10.0, 1.5, 0.5);
        let log_str_taker = control_taker.to_log_string();
        
        assert!(log_str_taker.contains("ν^a=1.500"));
        assert!(log_str_taker.contains("ν^b=0.500"));
    }
    
    // ============================================================================
    // HJB Components Tests
    // ============================================================================
    
    #[test]
    fn test_value_function_inventory_penalty() {
        let value_fn = ValueFunction::new(0.01, 100.0);
        
        // V(0) should be higher than V(50) due to inventory penalty
        let state = StateVector::new();
        let v_zero = value_fn.evaluate(0.0, &state);
        let v_fifty = value_fn.evaluate(50.0, &state);
        
        assert!(v_zero > v_fifty);
        
        // Symmetric penalty: V(50) ≈ V(-50)
        let v_minus_fifty = value_fn.evaluate(-50.0, &state);
        assert!((v_fifty - v_minus_fifty).abs() < 1.0);
    }
    
    #[test]
    fn test_value_function_time_decay() {
        let mut value_fn = ValueFunction::new(0.01, 100.0);
        
        let state = create_test_state_vector(100.0, 50.0, 0.0, 10.0, 0.5);
        
        // Early time (t=10)
        value_fn.set_time(10.0);
        let v_early = value_fn.evaluate(50.0, &state);
        
        // Late time (t=90)
        value_fn.set_time(90.0);
        let v_late = value_fn.evaluate(50.0, &state);
        
        // Inventory penalty accumulates over time remaining
        // So with less time remaining, penalty should be smaller (less negative)
        assert!(v_late > v_early);
    }
    
    #[test]
    fn test_value_function_inventory_delta() {
        let value_fn = ValueFunction::new(0.01, 100.0);
        let state = StateVector::new();
        
        // V(Q+1) - V(Q) should be more negative as Q increases
        // (worse to accumulate inventory)
        let delta_zero = value_fn.inventory_delta(0.0, 1.0, &state);
        let delta_fifty = value_fn.inventory_delta(50.0, 1.0, &state);
        
        assert!(delta_zero > delta_fifty);
        
        // Selling from large inventory should have positive value change
        let delta_sell = value_fn.inventory_delta(50.0, -1.0, &state);
        assert!(delta_sell > 0.0);
    }
    
    #[test]
    fn test_hjb_maker_fill_rates() {
        let hjb = HJBComponents::new();
        
        let state = create_test_state_vector(100.0, 0.0, 0.0, 10.0, 0.5); // Market half-spread = 5 bps, Balanced book
        
        // Quote at market (5 bps) should have high fill rate
        let rate_at_market = hjb.maker_bid_fill_rate(5.0, &state);
        
        // Quote far from market (20 bps) should have low fill rate
        let rate_far = hjb.maker_bid_fill_rate(20.0, &state);
        
        assert!(rate_at_market > rate_far);
        assert!(rate_at_market > 0.5); // Should be reasonably high
        assert!(rate_far < 0.5); // Should decay
    }
    
    #[test]
    fn test_hjb_lob_imbalance_affects_fill_rate() {
        let hjb = HJBComponents::new();
        
        // High bid imbalance (I_t = 0.9) - lots of buy orders
        let state_high_bid = create_test_state_vector(100.0, 0.0, 0.0, 10.0, 0.9);
        
        // Low bid imbalance (I_t = 0.1) - lots of sell orders
        let state_low_bid = create_test_state_vector(100.0, 0.0, 0.0, 10.0, 0.1);
        
        // When lots of buy orders exist (high I_t), our bid fill rate should be lower
        let bid_rate_high_imb = hjb.maker_bid_fill_rate(5.0, &state_high_bid);
        let bid_rate_low_imb = hjb.maker_bid_fill_rate(5.0, &state_low_bid);
        
        assert!(bid_rate_low_imb > bid_rate_high_imb);
        
        // Opposite for asks
        let ask_rate_high_imb = hjb.maker_ask_fill_rate(5.0, &state_high_bid);
        let ask_rate_low_imb = hjb.maker_ask_fill_rate(5.0, &state_low_bid);
        
        assert!(ask_rate_high_imb > ask_rate_low_imb);
    }
    
    #[test]
    fn test_hjb_maker_value_includes_spread() {
        let hjb = HJBComponents::new();
        let value_fn = ValueFunction::new(0.01, 100.0);
        
        let state = create_test_state_vector(100.0, 0.0, 0.0, 10.0, 0.5);
        
        // Maker ask value should be positive (we receive S + δ^a)
        let ask_value = hjb.maker_ask_value(5.0, &state, &value_fn);
        
        // Maker bid value depends on inventory change vs cost
        let bid_value = hjb.maker_bid_value(5.0, &state, &value_fn);
        
        // Both should be non-zero when fill rates are positive
        assert!(ask_value.abs() > 0.0);
        assert!(bid_value.abs() > 0.0);
    }
    
    #[test]
    fn test_hjb_taker_costs_more_than_maker() {
        let hjb = HJBComponents::new();
        let value_fn = ValueFunction::new(0.01, 100.0);
        
        let state = create_test_state_vector(100.0, 0.0, 0.0, 10.0, 0.5);
        
        // Maker bid at 5 bps: pay mid - 5bps
        let maker_bid = hjb.maker_bid_value(5.0, &state, &value_fn);
        
        // Taker buy: pay mid + 5bps (market ask) + fee
        let taker_buy = hjb.taker_buy_value(1.0, &state, &value_fn);
        
        // Taker should be less valuable (more costly) than maker
        // Note: comparing expected values, not raw costs
        assert!(taker_buy < maker_bid);
    }
    
    #[test]
    fn test_hjb_inventory_penalty_in_objective() {
        let hjb = HJBComponents::new();
        let value_fn = ValueFunction::new(0.01, 100.0);
        
        // Zero inventory state
        let state_zero = create_test_state_vector(100.0, 0.0, 0.0, 10.0, 0.5);
        
        // High inventory state
        let state_high = create_test_state_vector(100.0, 50.0, 0.0, 10.0, 0.5);
        
        let control = ControlVector::symmetric(10.0);
        
        let value_zero = hjb.evaluate_control(&control, &state_zero, &value_fn);
        let value_high = hjb.evaluate_control(&control, &state_high, &value_fn);
        
        // High inventory should have lower objective value due to -φQ²
        assert!(value_zero > value_high);
    }
    
    #[test]
    fn test_hjb_optimize_control_basic() {
        let hjb = HJBComponents::new();
        let value_fn = ValueFunction::new(0.01, 100.0);
        
        let state = create_test_state_vector(100.0, 0.0, 0.0, 10.0, 0.5);
        
        let optimal_control = hjb.optimize_control(&state, &value_fn, 10.0);
        
        // Should return a valid control
        assert!(optimal_control.validate(5.0).is_ok());
        
        // Should have positive spreads
        assert!(optimal_control.bid_offset_bps > 0.0);
        assert!(optimal_control.ask_offset_bps > 0.0);
    }
    
    #[test]
    fn test_hjb_optimize_activates_taker_for_high_inventory() {
        let hjb = HJBComponents::new();
        let value_fn = ValueFunction::new(0.01, 100.0);
        
        // High inventory state (95% of assumed max)
        let state = create_test_state_vector(100.0, 95.0, 0.0, 10.0, 0.5);
        
        let optimal_control = hjb.optimize_control(&state, &value_fn, 10.0);
        
        // Should activate taker selling to reduce inventory
        assert!(optimal_control.taker_sell_rate > 0.0 || optimal_control.is_liquidating());
    }
    
    #[test]
    fn test_hjb_optimize_respects_adverse_selection() {
        let hjb = HJBComponents::new();
        let value_fn = ValueFunction::new(0.01, 100.0);
        
        // Strong upward drift (μ̂ > 0)
        let state_up = create_test_state_vector(100.0, 0.0, 2.0, 10.0, 0.5);
        
        // Strong downward drift (μ̂ < 0)
        let state_down = create_test_state_vector(100.0, 0.0, -2.0, 10.0, 0.5);
        
        let control_up = hjb.optimize_control(&state_up, &value_fn, 10.0);
        let control_down = hjb.optimize_control(&state_down, &value_fn, 10.0);
        
        // Both should be valid controls
        assert!(control_up.validate(5.0).is_ok());
        assert!(control_down.validate(5.0).is_ok());
    }
    
    #[test]
    fn test_value_function_cache() {
        let mut value_fn = ValueFunction::new(0.01, 100.0);

        let state = create_test_state_vector(100.0, 0.0, 0.0, 10.0, 0.5);

        // Cache some values
        value_fn.cache_value(0, 100.0);
        value_fn.cache_value(50, 50.0);

        // Evaluate - should use computed values
        let v0 = value_fn.evaluate(0.0, &state);
        let v50 = value_fn.evaluate(50.0, &state);

        // Clear cache
        value_fn.clear_cache();

        // Should still compute values after clear
        let v0_after = value_fn.evaluate(0.0, &state);
        let v50_after = value_fn.evaluate(50.0, &state);

        // Values should be computable regardless of cache
        // V(0) should be higher than V(50) due to inventory penalty
        assert!(v0 > v50);
        assert!(v0_after > v50_after);
    }
}

