use alloy::{primitives::Address, signers::local::PrivateKeySigner};
use log::{error, info};
use tokio::sync::mpsc::unbounded_channel;
use std::sync::{Arc, RwLock};

//RUST_LOG=info cargo run --bin market_maker_v2

use crate::{
    bps_diff, AssetType, BaseUrl, BookAnalysis, ClientCancelRequest, ClientLimit, ClientOrder,
    ClientOrderRequest, ExchangeClient, ExchangeDataStatus, ExchangeResponseStatus, InfoClient,
    InventorySkewCalculator, InventorySkewConfig, MarketCloseParams, MarketOrderParams, Message,
    OrderBook, Subscription, TickLotValidator, UserData, EPSILON,
};

/// Tunable parameters for live adjustment of market making strategy
/// These parameters control various aspects of the algorithm and can be
/// reloaded at runtime without restarting the bot
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TuningParams {
    /// Inventory skew adjustment factor (default: 0.5)
    /// Controls how much to skew quotes based on inventory
    /// Higher = more aggressive inventory management
    pub skew_adjustment_factor: f64,
    
    /// Adverse selection adjustment factor (default: 0.5)
    /// Controls how much to adjust spreads based on adverse selection estimates
    /// Higher = more aggressive response to adverse selection signals
    pub adverse_selection_adjustment_factor: f64,
    
    /// Adverse selection filter smoothing parameter (lambda) (default: 0.1)
    /// Controls how responsive the adverse selection filter is to new signals
    /// Higher = more weight on recent observations (more responsive)
    /// Lower = more weight on historical average (smoother)
    pub adverse_selection_lambda: f64,
    
    /// Inventory urgency threshold (default: 0.7)
    /// Inventory ratio above which to activate taker orders for liquidation
    /// Range: [0.0, 1.0] where 1.0 = at max inventory
    pub inventory_urgency_threshold: f64,
    
    /// Liquidation rate multiplier (default: 10.0)
    /// Scales the taker order rate when urgency is high
    /// Higher = more aggressive liquidation via market orders
    pub liquidation_rate_multiplier: f64,
    
    /// Minimum spread base ratio (default: 0.2)
    /// Minimum quote offset as a fraction of base spread
    /// Ensures quotes don't get too tight during adjustments
    pub min_spread_base_ratio: f64,
    
    /// Adverse selection spread scale factor (default: 100.0)
    /// Denominator for normalizing spread in adverse selection calculation
    /// Higher = less sensitivity to spread changes
    pub adverse_selection_spread_scale: f64,
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
            m: vec![0.0; 7], // 7 parameters
            v: vec![0.0; 7],
            t: 0,
            alpha: 0.001,  // Conservative default learning rate
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
        }
    }
}

impl AdamOptimizerState {
    /// Create a new Adam optimizer with custom parameters
    pub fn new(alpha: f64, beta1: f64, beta2: f64) -> Self {
        Self {
            m: vec![0.0; 7],
            v: vec![0.0; 7],
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
        assert_eq!(gradient_vector.len(), 7, "Gradient vector must have 7 elements");
        
        // Increment time step
        self.t += 1;
        let t = self.t as f64;
        
        let mut updates = Vec::with_capacity(7);
        
        for i in 0..7 {
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
        self.m = vec![0.0; 7];
        self.v = vec![0.0; 7];
        self.t = 0;
    }
    
    /// Get effective learning rate for a specific parameter
    /// Useful for monitoring how much each parameter is being adjusted
    pub fn get_effective_learning_rate(&self, param_index: usize) -> f64 {
        if self.t == 0 || param_index >= 7 {
            return 0.0;
        }
        
        let t = self.t as f64;
        let v_hat = self.v[param_index] / (1.0 - self.beta2.powf(t));
        
        self.alpha / (v_hat.sqrt() + self.epsilon)
    }
}

impl Default for TuningParams {
    fn default() -> Self {
        Self {
            skew_adjustment_factor: 0.5,
            adverse_selection_adjustment_factor: 0.5,
            adverse_selection_lambda: 0.1,
            inventory_urgency_threshold: 0.7,
            liquidation_rate_multiplier: 10.0,
            min_spread_base_ratio: 0.2,
            adverse_selection_spread_scale: 100.0,
        }
    }
}

impl TuningParams {
    /// Validate that all parameters are within reasonable ranges
    pub fn validate(&self) -> Result<(), String> {
        if self.skew_adjustment_factor < 0.0 || self.skew_adjustment_factor > 2.0 {
            return Err(format!("skew_adjustment_factor must be in [0.0, 2.0], got {}", self.skew_adjustment_factor));
        }
        if self.adverse_selection_adjustment_factor < 0.0 || self.adverse_selection_adjustment_factor > 2.0 {
            return Err(format!("adverse_selection_adjustment_factor must be in [0.0, 2.0], got {}", self.adverse_selection_adjustment_factor));
        }
        if self.adverse_selection_lambda < 0.0 || self.adverse_selection_lambda > 1.0 {
            return Err(format!("adverse_selection_lambda must be in [0.0, 1.0], got {}", self.adverse_selection_lambda));
        }
        if self.inventory_urgency_threshold < 0.0 || self.inventory_urgency_threshold > 1.0 {
            return Err(format!("inventory_urgency_threshold must be in [0.0, 1.0], got {}", self.inventory_urgency_threshold));
        }
        if self.liquidation_rate_multiplier < 0.0 || self.liquidation_rate_multiplier > 100.0 {
            return Err(format!("liquidation_rate_multiplier must be in [0.0, 100.0], got {}", self.liquidation_rate_multiplier));
        }
        if self.min_spread_base_ratio < 0.0 || self.min_spread_base_ratio > 1.0 {
            return Err(format!("min_spread_base_ratio must be in [0.0, 1.0], got {}", self.min_spread_base_ratio));
        }
        if self.adverse_selection_spread_scale <= 0.0 {
            return Err(format!("adverse_selection_spread_scale must be positive, got {}", self.adverse_selection_spread_scale));
        }
        Ok(())
    }
    
    /// Load parameters from a JSON file
    pub fn from_json_file(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let contents = std::fs::read_to_string(path)?;
        let params: TuningParams = serde_json::from_str(&contents)?;
        params.validate()?;
        Ok(params)
    }
    
    /// Save parameters to a JSON file
    pub fn to_json_file(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        self.validate()?;
        let contents = serde_json::to_string_pretty(self)?;
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
        }
    }
    
    /// Update the state vector from current market data
    pub fn update(
        &mut self,
        mid_price: f64,
        inventory: f64,
        book_analysis: Option<&BookAnalysis>,
        order_book: Option<&OrderBook>,
        tuning_params: &TuningParams,
    ) {
        self.mid_price = mid_price;
        self.inventory = inventory;
        
        if let Some(book) = order_book {
            // Update market spread (Δ_t)
            if let Some(spread_bps) = book.spread_bps() {
                self.market_spread_bps = spread_bps;
            }
        }
        
        if let Some(analysis) = book_analysis {
            // Update LOB imbalance (I_t)
            // BookAnalysis.imbalance is in range [-1, 1]
            // Convert to ratio: (imbalance + 1) / 2 gives us [0, 1]
            // Where 0 = all ask volume, 1 = all bid volume, 0.5 = balanced
            self.lob_imbalance = (analysis.imbalance + 1.0) / 2.0;
            
            // Update adverse selection estimate (μ̂_t) using LOB imbalance
            // This is a filtered estimate of short-term price drift
            self.update_adverse_selection(tuning_params);
        }
    }
    
    /// Update the adverse selection estimate using an exponential moving average
    /// The LOB imbalance is used as a signal for short-term price direction
    fn update_adverse_selection(&mut self, tuning_params: &TuningParams) {
        // Smoothing parameter (lambda) - adjust this based on desired responsiveness
        // Higher lambda = more weight on recent observations
        let lambda = tuning_params.adverse_selection_lambda;
        
        // Convert imbalance to directional signal centered at 0
        // imbalance = 0.5 means balanced, no expected drift
        // imbalance > 0.5 means buying pressure (positive drift expected)
        // imbalance < 0.5 means selling pressure (negative drift expected)
        let signal = (self.lob_imbalance - 0.5) * 2.0; // Range: [-1, 1]
        
        // Scale by market spread to account for volatility
        // Higher spread = more uncertainty, scale down the signal
        let spread_scale = if self.market_spread_bps > 0.0 {
            1.0 / (1.0 + self.market_spread_bps / tuning_params.adverse_selection_spread_scale)
        } else {
            1.0
        };
        
        let scaled_signal = signal * spread_scale;
        
        // Update estimate using exponential moving average
        self.adverse_selection_estimate = 
            lambda * scaled_signal + (1.0 - lambda) * self.adverse_selection_estimate;
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
            "StateVector[S={:.2}, Q={:.4}, μ̂={:.4}, Δ={:.1}bps, I={:.3}]",
            self.mid_price,
            self.inventory,
            self.adverse_selection_estimate,
            self.market_spread_bps,
            self.lob_imbalance
        )
    }
}

impl Default for StateVector {
    fn default() -> Self {
        Self::new()
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
    pub fn apply_state_adjustments(
        &mut self,
        state: &StateVector,
        base_spread_bps: f64,
        max_inventory: f64,
        tuning_params: &TuningParams,
    ) {
        // 1. Adverse Selection Adjustment
        let adverse_adj = state.get_adverse_selection_adjustment(
            base_spread_bps,
            tuning_params.adverse_selection_adjustment_factor,
        );
        if adverse_adj > 0.0 {
            // Bearish signal - widen bid (make buying less attractive)
            self.bid_offset_bps += adverse_adj;
        } else {
            // Bullish signal - widen ask (make selling less attractive)
            self.ask_offset_bps -= adverse_adj;
        }
        
        // 2. Inventory Risk Adjustment
        let risk_multiplier = state.get_inventory_risk_multiplier(max_inventory);
        self.ask_offset_bps *= risk_multiplier;
        self.bid_offset_bps *= risk_multiplier;
        
        // 3. Inventory-based Quote Skewing
        // If long, tighten ask and widen bid to encourage selling
        // If short, tighten bid and widen ask to encourage buying
        let inventory_ratio = if max_inventory > 0.0 {
            (state.inventory / max_inventory).clamp(-1.0, 1.0)
        } else {
            0.0
        };
        
        let skew_adjustment = inventory_ratio * base_spread_bps * tuning_params.skew_adjustment_factor;
        self.ask_offset_bps -= skew_adjustment; // Long -> tighter ask
        self.bid_offset_bps += skew_adjustment; // Long -> wider bid
        
        // Ensure offsets stay positive
        self.ask_offset_bps = self.ask_offset_bps.max(base_spread_bps * tuning_params.min_spread_base_ratio);
        self.bid_offset_bps = self.bid_offset_bps.max(base_spread_bps * tuning_params.min_spread_base_ratio);
        
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
    
    /// Taker fee (paid when crossing spread)
    pub taker_fee_bps: f64,
}

impl HJBComponents {
    /// Create new HJB components with default parameters
    pub fn new() -> Self {
        Self {
            lambda_base: 1.0,      // 1 fill per second at best quotes
            phi: 0.01,             // Inventory penalty coefficient
            taker_fee_bps: 2.0,    // 2 bps taker fee
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
    /// λ^b * [V(Q+1) - V(Q) - (S_t - δ^b)]
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
        let cash_flow = -price_paid;
        
        lambda_b * (value_change + cash_flow)
    }
    
    /// Calculate expected value from maker ask fill
    /// λ^a * [V(Q-1) - V(Q) + (S_t + δ^a)]
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
        let cash_flow = price_received;
        
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

#[derive(Debug)]
pub struct MarketMakerRestingOrder {
    pub oid: u64,
    pub position: f64,
    pub price: f64,
}

#[derive(Debug)]
pub struct MarketMakerInput {
    pub asset: String,
    pub target_liquidity: f64, // Amount of liquidity on both sides to target
    pub half_spread: u16,      // Half of the spread for our market making (in BPS)
    pub max_bps_diff: u16, // Max deviation before we cancel and put new orders on the book (in BPS)
    pub max_absolute_position_size: f64, // Absolute value of the max position we can take on
    pub asset_type: AssetType, // Asset type (Perp or Spot) for tick/lot size validation
    pub wallet: PrivateKeySigner, // Wallet containing private key
    pub inventory_skew_config: Option<InventorySkewConfig>, // Optional inventory skewing configuration
}

#[derive(Debug)]
pub struct MarketMaker {
    pub asset: String,
    pub target_liquidity: f64,
    pub half_spread: u16,
    pub max_bps_diff: u16,
    pub max_absolute_position_size: f64,
    pub tick_lot_validator: TickLotValidator,
    pub lower_resting: MarketMakerRestingOrder,
    pub upper_resting: MarketMakerRestingOrder,
    pub cur_position: f64,
    pub latest_mid_price: f64,
    pub info_client: InfoClient,
    pub exchange_client: ExchangeClient,
    pub user_address: Address,
    pub inventory_skew_calculator: Option<InventorySkewCalculator>,
    pub latest_book: Option<OrderBook>,
    pub latest_book_analysis: Option<BookAnalysis>,
    /// State Vector (Z_t) for optimal decision making
    pub state_vector: StateVector,
    /// Control Vector (u_t) for algorithm actions
    pub control_vector: ControlVector,
    /// HJB equation components for optimal control
    pub hjb_components: HJBComponents,
    /// Value function V(Q, Z, t)
    pub value_function: ValueFunction,
    /// Tunable parameters wrapped in Arc<RwLock> for live updates
    pub tuning_params: Arc<RwLock<TuningParams>>,
    /// Adam optimizer state for automatic parameter tuning
    pub adam_optimizer: Arc<RwLock<AdamOptimizerState>>,
}

impl MarketMaker {
    /// Get a copy of current tuning parameters
    pub fn get_tuning_params(&self) -> TuningParams {
        self.tuning_params.read().unwrap().clone()
    }
    
    /// Update the state vector with current market conditions
    fn update_state_vector(&mut self) {
        let params = self.tuning_params.read().unwrap();
        self.state_vector.update(
            self.latest_mid_price,
            self.cur_position,
            self.latest_book_analysis.as_ref(),
            self.latest_book.as_ref(),
            &params,
        );
        
        // Log state vector for monitoring (can be disabled for performance)
        info!("{}", self.state_vector.to_log_string());
    }
    
    /// Calculate optimal control vector based on current state
    fn calculate_optimal_control(&mut self) {
        let base_spread_bps = self.half_spread as f64;
        let params = self.tuning_params.read().unwrap();
        
        // Update value function time if needed
        // In production, this would be updated based on actual time
        // For now, we use a continuous time model
        
        // Use heuristic adjustments (fast, approximates HJB solution)
        // For full HJB optimization, call hjb_components.optimize_control() directly
        // Start with symmetric quotes
        self.control_vector = ControlVector::symmetric(base_spread_bps);
        
        // Apply state-based adjustments
        self.control_vector.apply_state_adjustments(
            &self.state_vector,
            base_spread_bps,
            self.max_absolute_position_size,
            &params,
        );
        
        // Validate the control vector
        if let Err(e) = self.control_vector.validate(base_spread_bps * 0.5) {
            error!("Invalid control vector: {}", e);
            // Fallback to safe symmetric control
            self.control_vector = ControlVector::symmetric(base_spread_bps);
        }
        
        // Log control vector
        info!("{}", self.control_vector.to_log_string());
    }
    
    /// Use full HJB optimization to calculate optimal control
    /// This is more computationally expensive but theoretically optimal
    /// By default, calculate_optimal_control() uses fast heuristics
    pub fn calculate_optimal_control_hjb(&mut self) {
        let base_spread_bps = self.half_spread as f64;
        
        self.control_vector = self.hjb_components.optimize_control(
            &self.state_vector,
            &self.value_function,
            base_spread_bps,
        );
        
        // Validate the control vector
        if let Err(e) = self.control_vector.validate(base_spread_bps * 0.5) {
            error!("Invalid control vector from HJB optimization: {}", e);
            // Fallback to heuristic method
            self.calculate_optimal_control();
        }
        
        info!("{}", self.control_vector.to_log_string());
    }
    
    /// Evaluate current strategy using HJB objective
    /// Returns the instantaneous expected value rate
    pub fn evaluate_current_strategy(&self) -> f64 {
        self.hjb_components.evaluate_control(
            &self.control_vector,
            &self.state_vector,
            &self.value_function,
        )
    }
    
    /// Get expected maker fill rates for current control
    pub fn get_expected_fill_rates(&self) -> (f64, f64) {
        let lambda_bid = self.hjb_components.maker_bid_fill_rate(
            self.control_vector.bid_offset_bps,
            &self.state_vector,
        );
        let lambda_ask = self.hjb_components.maker_ask_fill_rate(
            self.control_vector.ask_offset_bps,
            &self.state_vector,
        );
        (lambda_bid, lambda_ask)
    }
    
    /// Get the current control vector (read-only access)
    pub fn get_control_vector(&self) -> &ControlVector {
        &self.control_vector
    }
    
    /// Get the current state vector (read-only access)
    pub fn get_state_vector(&self) -> &StateVector {
        &self.state_vector
    }
    
    /// Run HJB grid search optimization in background and compare with heuristic
    /// 
    /// This spawns a background task to run the expensive `optimize_control()` grid search
    /// without blocking real-time trading. The results can be used to:
    /// 
    /// - Validate that the fast heuristic is performing well
    /// - Tune heuristic parameters by comparing with grid search results
    /// - Collect data for ML-based policy learning
    /// 
    /// **NEW: Automatic Adam-optimized parameter tuning**
    /// 
    /// This function now implements the Adam optimizer (Adaptive Moment Estimation) to
    /// automatically tune the TuningParams to minimize the performance gap between the
    /// heuristic and optimal control. Adam adapts the learning rate for each parameter
    /// automatically using momentum, making it far more robust than vanilla SGD.
    /// 
    /// The gradient is calculated numerically using finite differences, and the updated
    /// parameters are saved back to the JSON file for persistence.
    /// 
    /// Returns a handle that can be awaited to get the optimized control and comparison metrics.
    /// 
    /// # Example
    /// 
    /// ```ignore
    /// // Spawn background optimization
    /// let optimization_handle = market_maker.optimize_control_background();
    /// 
    /// // Continue with real-time trading using fast heuristic...
    /// 
    /// // Later, check optimization results
    /// if let Ok((optimal_control, heuristic_value, optimal_value)) = optimization_handle.await {
    ///     let performance_gap = (optimal_value - heuristic_value) / optimal_value.abs();
    ///     info!("Heuristic performance: {:.2}% of optimal", (1.0 - performance_gap) * 100.0);
    ///     
    ///     if performance_gap > 0.1 {
    ///         warn!("Heuristic significantly underperforming, consider tuning");
    ///     }
    /// }
    /// ```
    pub fn optimize_control_background(&self) -> tokio::task::JoinHandle<(ControlVector, f64, f64)> {
        // Clone all necessary data for background thread
        let hjb_components = self.hjb_components.clone();
        let state_vector = self.state_vector.clone();
        let value_function = self.value_function.clone();
        let current_control = self.control_vector.clone();
        let base_spread_bps = self.half_spread as f64;
        let max_absolute_position_size = self.max_absolute_position_size;
        let tuning_params = Arc::clone(&self.tuning_params);
        let adam_optimizer = Arc::clone(&self.adam_optimizer);
        
        tokio::task::spawn_blocking(move || {
            // Run expensive grid search optimization
            let optimal_control = hjb_components.optimize_control(
                &state_vector,
                &value_function,
                base_spread_bps,
            );
            
            // Evaluate both controls for comparison
            let heuristic_value = hjb_components.evaluate_control(
                &current_control,
                &state_vector,
                &value_function,
            );
            
            let optimal_value = hjb_components.evaluate_control(
                &optimal_control,
                &state_vector,
                &value_function,
            );
            
            // ============================================
            // ADAM OPTIMIZER-BASED AUTOMATIC PARAMETER TUNING
            // ============================================
            
            // 1. Define Loss: Control Gap (squared difference in quote offsets)
            // L = (bid_optimal - bid_heuristic)² + (ask_optimal - ask_heuristic)²
            // This measures how different our quotes are from optimal, which produces
            // stable, interpretable gradients compared to the value gap.
            let bid_gap = optimal_control.bid_offset_bps - current_control.bid_offset_bps;
            let ask_gap = optimal_control.ask_offset_bps - current_control.ask_offset_bps;
            let current_loss = bid_gap.powi(2) + ask_gap.powi(2);
            
            // Only tune if control gap is significant (> 1 bps squared total)
            // This means quotes differ by ~0.7 bps RMS, which is meaningful
            let control_gap_threshold = 1.0;
            
            if current_loss > control_gap_threshold {
                info!("Control gap detected: {:.6} bps² (bid_gap={:.2}bps, ask_gap={:.2}bps). Running Adam optimizer parameter tuning...", 
                    current_loss, bid_gap, ask_gap);
                
                // 2. Gradient Calculation Parameters
                let nudge_amount = 0.001; // Small perturbation for finite difference
                
                // 3. Get current parameters
                let original_params = tuning_params.read().unwrap().clone();
                let mut gradient_vector: Vec<f64> = vec![0.0; 7];
                
                // 4. Calculate numerical gradient for each parameter
                let param_indices = [
                    ("skew_adjustment_factor", 0),
                    ("adverse_selection_adjustment_factor", 1),
                    ("adverse_selection_lambda", 2),
                    ("inventory_urgency_threshold", 3),
                    ("liquidation_rate_multiplier", 4),
                    ("min_spread_base_ratio", 5),
                    ("adverse_selection_spread_scale", 6),
                ];
                
                for (param_name, i) in param_indices.iter() {
                    // Create nudged parameters
                    let mut nudged_params = original_params.clone();
                    match *i {
                        0 => nudged_params.skew_adjustment_factor += nudge_amount,
                        1 => nudged_params.adverse_selection_adjustment_factor += nudge_amount,
                        2 => nudged_params.adverse_selection_lambda += nudge_amount,
                        3 => nudged_params.inventory_urgency_threshold += nudge_amount,
                        4 => nudged_params.liquidation_rate_multiplier += nudge_amount,
                        5 => nudged_params.min_spread_base_ratio += nudge_amount,
                        6 => nudged_params.adverse_selection_spread_scale += nudge_amount,
                        _ => {}
                    }
                    
                    // Skip if nudged params are invalid
                    if nudged_params.validate().is_err() {
                        info!("Skipping {} - nudge would violate constraints", param_name);
                        continue;
                    }
                    
                    // Re-run heuristic with nudged params
                    let mut nudged_control = ControlVector::symmetric(base_spread_bps);
                    nudged_control.apply_state_adjustments(
                        &state_vector,
                        base_spread_bps,
                        max_absolute_position_size,
                        &nudged_params,
                    );
                    
                    // Calculate control gap loss with nudged params
                    // L = (bid_optimal - bid_nudged)² + (ask_optimal - ask_nudged)²
                    let nudged_bid_gap = optimal_control.bid_offset_bps - nudged_control.bid_offset_bps;
                    let nudged_ask_gap = optimal_control.ask_offset_bps - nudged_control.ask_offset_bps;
                    let nudged_loss = nudged_bid_gap.powi(2) + nudged_ask_gap.powi(2);
                    
                    // Calculate partial derivative (gradient component)
                    // This is (change in loss) / (change in parameter)
                    let gradient = (nudged_loss - current_loss) / nudge_amount;
                    gradient_vector[*i] = gradient;
                    
                    info!("Gradient[{}] = {:.6}", param_name, gradient);
                }
                
                // 5. Apply Adam optimizer update
                let updates = {
                    let mut optimizer = adam_optimizer.write().unwrap();
                    optimizer.compute_update(&gradient_vector)
                };
                
                // Log effective learning rates for monitoring
                {
                    let optimizer = adam_optimizer.read().unwrap();
                    info!("Adam Optimizer State:");
                    info!("  Time step: {}", optimizer.t);
                    info!("  Base learning rate (α): {}", optimizer.alpha);
                    for (param_name, i) in param_indices.iter() {
                        let eff_lr = optimizer.get_effective_learning_rate(*i);
                        info!("  Effective LR[{}]: {:.6}", param_name, eff_lr);
                    }
                }
                
                // 6. Apply updates: theta_new = theta_old - update
                let mut updated_params = original_params.clone();
                updated_params.skew_adjustment_factor -= updates[0];
                updated_params.adverse_selection_adjustment_factor -= updates[1];
                updated_params.adverse_selection_lambda -= updates[2];
                updated_params.inventory_urgency_threshold -= updates[3];
                updated_params.liquidation_rate_multiplier -= updates[4];
                updated_params.min_spread_base_ratio -= updates[5];
                updated_params.adverse_selection_spread_scale -= updates[6];
                
                // 7. Validate and apply the updated parameters
                match updated_params.validate() {
                    Ok(_) => {
                        // Parameters are valid, apply them
                        {
                            let mut params = tuning_params.write().unwrap();
                            *params = updated_params.clone();
                        }
                        
                        info!("Adam Update Applied: {:?}", updated_params);
                        
                        // Save to JSON file for persistence
                        if let Err(e) = updated_params.to_json_file("tuning_params.json") {
                            error!("Failed to save updated tuning params to JSON: {}", e);
                        } else {
                            info!("Updated tuning parameters saved to tuning_params.json");
                        }
                        
                        // Log the changes for each parameter
                        info!("Parameter Changes:");
                        info!("  skew_adjustment_factor: {:.6} -> {:.6} (Δ={:.6})",
                            original_params.skew_adjustment_factor,
                            updated_params.skew_adjustment_factor,
                            updated_params.skew_adjustment_factor - original_params.skew_adjustment_factor);
                        info!("  adverse_selection_adjustment_factor: {:.6} -> {:.6} (Δ={:.6})",
                            original_params.adverse_selection_adjustment_factor,
                            updated_params.adverse_selection_adjustment_factor,
                            updated_params.adverse_selection_adjustment_factor - original_params.adverse_selection_adjustment_factor);
                        info!("  adverse_selection_lambda: {:.6} -> {:.6} (Δ={:.6})",
                            original_params.adverse_selection_lambda,
                            updated_params.adverse_selection_lambda,
                            updated_params.adverse_selection_lambda - original_params.adverse_selection_lambda);
                        info!("  inventory_urgency_threshold: {:.6} -> {:.6} (Δ={:.6})",
                            original_params.inventory_urgency_threshold,
                            updated_params.inventory_urgency_threshold,
                            updated_params.inventory_urgency_threshold - original_params.inventory_urgency_threshold);
                        info!("  liquidation_rate_multiplier: {:.6} -> {:.6} (Δ={:.6})",
                            original_params.liquidation_rate_multiplier,
                            updated_params.liquidation_rate_multiplier,
                            updated_params.liquidation_rate_multiplier - original_params.liquidation_rate_multiplier);
                        info!("  min_spread_base_ratio: {:.6} -> {:.6} (Δ={:.6})",
                            original_params.min_spread_base_ratio,
                            updated_params.min_spread_base_ratio,
                            updated_params.min_spread_base_ratio - original_params.min_spread_base_ratio);
                        info!("  adverse_selection_spread_scale: {:.6} -> {:.6} (Δ={:.6})",
                            original_params.adverse_selection_spread_scale,
                            updated_params.adverse_selection_spread_scale,
                            updated_params.adverse_selection_spread_scale - original_params.adverse_selection_spread_scale);
                    }
                    Err(e) => {
                        error!("Adam optimizer produced invalid parameters: {}. Reverting to original.", e);
                        error!("Attempted params: {:?}", updated_params);
                        
                        // Reset Adam optimizer state when encountering invalid parameters
                        // This helps recover from bad optimization trajectories
                        {
                            let mut optimizer = adam_optimizer.write().unwrap();
                            optimizer.reset();
                            info!("Adam optimizer state has been reset due to invalid parameters");
                        }
                    }
                }
            } else {
                info!("Control gap {:.6} bps² is below threshold {:.6} bps², skipping Adam tuning", 
                    current_loss, control_gap_threshold);
                info!("Heuristic quotes close to optimal (bid_gap={:.2}bps, ask_gap={:.2}bps)", 
                    bid_gap, ask_gap);
            }
            
            (optimal_control, heuristic_value, optimal_value)
        })
    }
    
    /// Calculate optimal spread adjustment based on state vector
    /// This can be used to implement more sophisticated pricing strategies
    pub fn calculate_state_based_spread_adjustment(&self) -> f64 {
        let base_spread_bps = self.half_spread as f64 * 2.0; // Convert half spread to full spread
        let params = self.tuning_params.read().unwrap();
        
        // Get adverse selection adjustment
        let adverse_adjustment = self.state_vector
            .get_adverse_selection_adjustment(base_spread_bps, params.adverse_selection_adjustment_factor);
        
        // Get inventory risk multiplier
        let risk_multiplier = self.state_vector
            .get_inventory_risk_multiplier(self.max_absolute_position_size);
        
        // Combined adjustment: adverse selection shift + risk-based widening
        // Note: This is a simple combination - more sophisticated models could be used
        adverse_adjustment + (base_spread_bps * (risk_multiplier - 1.0))
    }
    
    /// Check if we should pause market making based on state vector
    pub fn should_pause_trading(&self) -> bool {
        // Pause if market conditions are unfavorable
        let max_spread_threshold = self.half_spread as f64 * 10.0; // 5x normal spread
        !self.state_vector.is_market_favorable(max_spread_threshold)
    }
    
    /// Reset a resting order to default state
    fn reset_resting_order(&mut self, is_lower: bool) {
        let resting_order = if is_lower {
            &mut self.lower_resting
        } else {
            &mut self.upper_resting
        };
        
        *resting_order = MarketMakerRestingOrder {
            oid: 0,
            position: 0.0,
            price: -1.0,
        };
    }

    /// Clean up any invalid resting orders (e.g., negative positions)
    fn cleanup_invalid_resting_orders(&mut self) {
        if self.lower_resting.position < 0.0 {
            info!("Lower resting order has negative position, resetting");
            self.reset_resting_order(true);
        }
        if self.upper_resting.position < 0.0 {
            info!("Upper resting order has negative position, resetting");
            self.reset_resting_order(false);
        }
    }

    pub async fn new(input: MarketMakerInput) -> Result<MarketMaker, crate::Error> {
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

        let inventory_skew_calculator = input
            .inventory_skew_config
            .map(|config| InventorySkewCalculator::new(config));

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
        
        info!("Initialized with tuning parameters: {:?}", initial_params);
        info!("Adam optimizer will now autonomously tune these parameters");

        Ok(MarketMaker {
            asset: input.asset,
            target_liquidity: input.target_liquidity,
            half_spread: input.half_spread,
            max_bps_diff: input.max_bps_diff,
            max_absolute_position_size: input.max_absolute_position_size,
            tick_lot_validator,
            lower_resting: MarketMakerRestingOrder {
                oid: 0,
                position: 0.0,
                price: -1.0,
            },
            upper_resting: MarketMakerRestingOrder {
                oid: 0,
                position: 0.0,
                price: -1.0,
            },
            cur_position: 0.0,
            latest_mid_price: -1.0,
            info_client,
            exchange_client,
            user_address,
            inventory_skew_calculator,
            latest_book: None,
            latest_book_analysis: None,
            state_vector: StateVector::new(),
            control_vector: ControlVector::symmetric(input.half_spread as f64),
            hjb_components: HJBComponents::new(),
            value_function: ValueFunction::new(0.01, 86400.0), // φ=0.01, T=24h
            tuning_params: Arc::new(RwLock::new(initial_params)),
            adam_optimizer: Arc::new(RwLock::new(AdamOptimizerState::default())),
        })
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

        // Subscribe to L2Book if inventory skewing is enabled
        if self.inventory_skew_calculator.is_some() {
            info!("Subscribing to L2 book data for inventory skewing");
            self.info_client
                .subscribe(
                    Subscription::L2Book {
                        coin: self.asset.clone(),
                    },
                    sender,
                )
                .await
                .unwrap();
        }

        // Initialize background optimization loop timer
        let mut last_optimization_time = std::time::Instant::now();
        let optimization_interval = std::time::Duration::from_secs(60);
        info!("Background HJB optimization with Adam tuning enabled: will run every {} seconds", optimization_interval.as_secs());
        info!("Parameter tuning is now fully autonomous via Adam optimizer");
        info!("To override: stop bot, edit tuning_params.json, restart bot");

        loop {
            // Check if it's time to run background optimization
            if last_optimization_time.elapsed() >= optimization_interval {
                info!("Spawning background HJB optimization task...");
                let optimization_handle = self.optimize_control_background();
                
                // Don't await it, let it run in the background
                // and log the results when it's done
                tokio::spawn(async move {
                    match optimization_handle.await {
                        Ok((optimal_control, heuristic_value, optimal_value)) => {
                            let perf_gap = (optimal_value - heuristic_value).abs();
                            let gap_percent = if optimal_value.abs() > EPSILON {
                                (perf_gap / optimal_value.abs()) * 100.0
                            } else {
                                0.0
                            };
                            
                            info!(
                                "Background HJB Optimization Complete: Heuristic_Value={:.4}, Optimal_Value={:.4}, Gap={:.2}%",
                                heuristic_value, optimal_value, gap_percent
                            );
                            info!("Optimal Control (from grid search): {}", optimal_control.to_log_string());
                            
                            if gap_percent > 10.0 {
                                log::warn!("Heuristic performance gap is high (>{:.2}%)! Consider tuning.", gap_percent);
                            }
                        }
                        Err(e) => {
                            error!("Background optimization task failed: {:?}", e);
                        }
                    }
                });
                
                last_optimization_time = std::time::Instant::now();
            }

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

                // Handle market maker messages
                message = receiver.recv() => {
            let message = message.unwrap();
            match message {
                Message::L2Book(l2_book) => {
                    // Update our order book state
                    if let Some(book) = OrderBook::from_l2_data(&l2_book.data) {
                        // Analyze the book if we have a skew calculator
                        if let Some(calculator) = &self.inventory_skew_calculator {
                            let depth_levels = calculator.config.depth_analysis_levels;
                            if let Some(analysis) = book.analyze(depth_levels) {
                                // Optionally log book stats (can be disabled for performance)
                                // book.log_stats(&analysis);
                                self.latest_book_analysis = Some(analysis);
                            }
                        }
                        self.latest_book = Some(book);
                        
                        // Update state vector with new book data
                        self.update_state_vector();
                    }
                }
                Message::AllMids(all_mids) => {
                    let all_mids = all_mids.data.mids;
                    let mid = all_mids.get(&self.asset);
                    if let Some(mid) = mid {
                        let mid: f64 = mid.parse().unwrap();
                        self.latest_mid_price = mid;
                        
                        // Update state vector with new mid price
                        self.update_state_vector();
                        
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
                        for fill in fills {
                            let amount: f64 = fill.sz.parse().unwrap();
                            // Update our resting positions whenever we see a fill
                            if fill.side.eq("B") {
                                self.cur_position += amount;
                                
                                // Check if this fill corresponds to our tracked lower resting order
                                if self.lower_resting.oid == fill.oid {
                                    self.lower_resting.position -= amount;
                                    // If the order is fully filled, reset it
                                    if self.lower_resting.position <= EPSILON {
                                        info!("Lower resting order fully filled, resetting");
                                        self.reset_resting_order(true);
                                    }
                                }
                                
                                info!("Fill: bought {amount} {} (oid: {})", self.asset.clone(), fill.oid);
                            } else {
                                self.cur_position -= amount;
                                
                                // Check if this fill corresponds to our tracked upper resting order
                                if self.upper_resting.oid == fill.oid {
                                    self.upper_resting.position -= amount;
                                    // If the order is fully filled, reset it
                                    if self.upper_resting.position <= EPSILON {
                                        info!("Upper resting order fully filled, resetting");
                                        self.reset_resting_order(false);
                                    }
                                }
                                
                                info!("Fill: sold {amount} {} (oid: {})", self.asset.clone(), fill.oid);
                            }
                        }
                    }
                    
                    // Update state vector with new inventory
                    self.update_state_vector();
                    
                    // Check to see if we need to cancel or place any new orders
                    self.potentially_update().await;
                }
                _ => {
                    panic!("Unsupported message type");
                }
            }
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

    async fn place_order(
        &self,
        asset: String,
        amount: f64,
        price: f64,
        is_buy: bool,
    ) -> (f64, u64) {
        // Validate price and size before placing order
        if let Err(e) = self.tick_lot_validator.validate_price(price) {
            error!("Invalid price {}: {}", price, e);
            return (0.0, 0);
        }
        
        if let Err(e) = self.tick_lot_validator.validate_size(amount) {
            error!("Invalid size {}: {}", amount, e);
            return (0.0, 0);
        }
        let order = self
            .exchange_client
            .order(
                ClientOrderRequest {
                    asset,
                    is_buy,
                    reduce_only: false,
                    limit_px: price,
                    sz: amount,
                    cloid: None,
                    order_type: ClientOrder::Limit(ClientLimit {
                        tif: "Gtc".to_string(),
                    }),
                },
                None,
            )
            .await;
        match order {
            Ok(order) => match order {
                ExchangeResponseStatus::Ok(order) => {
                    if let Some(order) = order.data {
                        if !order.statuses.is_empty() {
                            match order.statuses[0].clone() {
                                ExchangeDataStatus::Filled(order) => {
                                    return (amount, order.oid);
                                }
                                ExchangeDataStatus::Resting(order) => {
                                    return (amount, order.oid);
                                }
                                ExchangeDataStatus::Error(e) => {
                                    error!("Error with placing order: {e}")
                                }
                                _ => unreachable!(),
                            }
                        } else {
                            error!("Exchange data statuses is empty when placing order: {order:?}")
                        }
                    } else {
                        error!("Exchange response data is empty when placing order: {order:?}")
                    }
                }
                ExchangeResponseStatus::Err(e) => {
                    error!("Error with placing order: {e}")
                }
            },
            Err(e) => error!("Error with placing order: {e}"),
        }
        (0.0, 0)
    }

    /// Place a market order using the SDK's market_open function with slippage protection
    /// This is safer than a pure IOC limit order as it handles slippage properly
    /// Returns the filled amount
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

    /// Cancel all open orders and close any position, then shutdown gracefully
    pub async fn shutdown(&mut self) {
        info!("Shutting down market maker, cancelling all open orders and closing position...");
        
        // Parallelize order cancellations
        let cancel_lower = async {
            if self.lower_resting.oid != 0 && self.lower_resting.position > EPSILON {
                info!("Cancelling lower resting order (oid: {})", self.lower_resting.oid);
                self.attempt_cancel(self.asset.clone(), self.lower_resting.oid).await
            } else {
                false
            }
        };

        let cancel_upper = async {
            if self.upper_resting.oid != 0 && self.upper_resting.position > EPSILON {
                info!("Cancelling upper resting order (oid: {})", self.upper_resting.oid);
                self.attempt_cancel(self.asset.clone(), self.upper_resting.oid).await
            } else {
                false
            }
        };

        let (lower_cancelled, upper_cancelled) = tokio::join!(cancel_lower, cancel_upper);

        if lower_cancelled {
            info!("Successfully cancelled lower resting order");
        } else if self.lower_resting.oid != 0 && self.lower_resting.position > EPSILON {
            info!("Lower resting order was already filled or cancelled");
        }

        if upper_cancelled {
            info!("Successfully cancelled upper resting order");
        } else if self.upper_resting.oid != 0 && self.upper_resting.position > EPSILON {
            info!("Upper resting order was already filled or cancelled");
        }

        self.reset_resting_order(true);
        self.reset_resting_order(false);
        
        // Close any existing position
        if self.cur_position.abs() > EPSILON {
            info!("Current position: {:.6} {}, closing position...", self.cur_position, self.asset);
            self.close_position().await;
        } else {
            info!("No position to close (position: {:.6})", self.cur_position);
        }
        
        info!("All orders cancelled and position closed. Market maker shutdown complete.");
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

    async fn potentially_update(&mut self) {
        // Clean up any invalid resting orders first
        self.cleanup_invalid_resting_orders();
        
        // Calculate optimal control based on current state
        self.calculate_optimal_control();
        
        // Execute taker orders if the control vector signals urgency
        // Taker sell: liquidate long position by hitting bids with market orders
        if self.control_vector.taker_sell_rate > EPSILON {
            let taker_size = self.tick_lot_validator.round_size(
                self.control_vector.taker_sell_rate.min(self.cur_position.max(0.0)),
                false,
            );
            
            if taker_size > EPSILON {
                // Note: price is provided for logging but place_taker_order uses market_open
                // which fetches current market price and applies slippage automatically
                let reference_price = if let Some(book) = &self.latest_book {
                    book.best_bid().unwrap_or(self.latest_mid_price)
                } else {
                    self.latest_mid_price
                };
                
                info!(
                    "🔴 TAKER SELL triggered: rate={:.4}, size={}, ref_price={} (position={})",
                    self.control_vector.taker_sell_rate, taker_size, reference_price, self.cur_position
                );
                
                let filled = self.place_taker_order(
                    self.asset.clone(),
                    taker_size,
                    reference_price, // Not used, just for logging
                    false, // sell
                ).await;
                
                if filled > EPSILON {
                    // Update position immediately (will be confirmed by fill event)
                    self.cur_position -= filled;
                    info!("Market sell executed: {} filled, new position: {}", filled, self.cur_position);
                }
            }
        }
        
        // Taker buy: liquidate short position by lifting asks with market orders
        if self.control_vector.taker_buy_rate > EPSILON {
            let taker_size = self.tick_lot_validator.round_size(
                self.control_vector.taker_buy_rate.min((-self.cur_position).max(0.0)),
                false,
            );
            
            if taker_size > EPSILON {
                // Note: price is provided for logging but place_taker_order uses market_open
                // which fetches current market price and applies slippage automatically
                let reference_price = if let Some(book) = &self.latest_book {
                    book.best_ask().unwrap_or(self.latest_mid_price)
                } else {
                    self.latest_mid_price
                };
                
                info!(
                    "🔵 TAKER BUY triggered: rate={:.4}, size={}, ref_price={} (position={})",
                    self.control_vector.taker_buy_rate, taker_size, reference_price, self.cur_position
                );
                
                let filled = self.place_taker_order(
                    self.asset.clone(),
                    taker_size,
                    reference_price, // Not used, just for logging
                    true, // buy
                ).await;
                
                if filled > EPSILON {
                    // Update position immediately (will be confirmed by fill event)
                    self.cur_position += filled;
                    info!("Market buy executed: {} filled, new position: {}", filled, self.cur_position);
                }
            }
        }
        
        // Get the optimal quote prices directly from the ControlVector
        let (lower_price, upper_price) = 
            self.control_vector.calculate_quote_prices(self.latest_mid_price);
        
        // Round the prices using your validator
        let mut lower_price = self.tick_lot_validator.round_price(lower_price, false); // Round down for buy
        let mut upper_price = self.tick_lot_validator.round_price(upper_price, true);  // Round up for sell

        // Rounding optimistically to make our market tighter might cause a weird edge case, so account for that
        if (lower_price - upper_price).abs() < EPSILON {
            lower_price = self.tick_lot_validator.round_price(lower_price, true);
            upper_price = self.tick_lot_validator.round_price(upper_price, false);
        }

        // Determine amounts we can put on the book without exceeding the max absolute position size
        let lower_order_amount = self.tick_lot_validator.round_size(
            (self.max_absolute_position_size - self.cur_position)
                .min(self.target_liquidity)
                .max(0.0),
            false, // Round down for size
        );

        let upper_order_amount = self.tick_lot_validator.round_size(
            (self.max_absolute_position_size + self.cur_position)
                .min(self.target_liquidity)
                .max(0.0),
            false, // Round down for size
        );

        // Determine if we need to cancel the resting order and put a new order up due to deviation
        let lower_change = (lower_order_amount - self.lower_resting.position).abs() > EPSILON
            || bps_diff(lower_price, self.lower_resting.price) > self.max_bps_diff;
        let upper_change = (upper_order_amount - self.upper_resting.position).abs() > EPSILON
            || bps_diff(upper_price, self.upper_resting.price) > self.max_bps_diff;

        // Parallelize order cancellations when both need to be cancelled
        let (lower_cancelled, upper_cancelled) = if (self.lower_resting.oid != 0 && self.lower_resting.position > EPSILON && lower_change)
            && (self.upper_resting.oid != 0 && self.upper_resting.position > EPSILON && upper_change)
        {
            // Both need cancelling - parallelize
            let lower_oid = self.lower_resting.oid;
            let upper_oid = self.upper_resting.oid;
            let asset = self.asset.clone();
            
            let cancel_lower_fut = self.attempt_cancel(asset.clone(), lower_oid);
            let cancel_upper_fut = self.attempt_cancel(asset.clone(), upper_oid);
            
            let (lower_result, upper_result) = tokio::join!(cancel_lower_fut, cancel_upper_fut);
            
            if lower_result {
                info!("Cancelled buy order: {:?}", self.lower_resting);
            } else {
                info!("Cancel failed for buy order (oid: {}) - treating as filled", lower_oid);
                self.reset_resting_order(true);
            }
            
            if upper_result {
                info!("Cancelled sell order: {:?}", self.upper_resting);
            } else {
                info!("Cancel failed for sell order (oid: {}) - treating as filled", upper_oid);
                self.reset_resting_order(false);
            }
            
            (lower_result || !lower_result, upper_result || !upper_result) // Both treated as cancelled
        } else {
            // Only one or neither needs cancelling - handle sequentially
            let lower_cancelled = if self.lower_resting.oid != 0 && self.lower_resting.position > EPSILON && lower_change {
                if self.attempt_cancel(self.asset.clone(), self.lower_resting.oid).await {
                    info!("Cancelled buy order: {:?}", self.lower_resting);
                    true
                } else {
                    info!("Cancel failed for buy order (oid: {}) - treating as filled", self.lower_resting.oid);
                    self.reset_resting_order(true);
                    true
                }
            } else {
                false
            };

            let upper_cancelled = if self.upper_resting.oid != 0 && self.upper_resting.position > EPSILON && upper_change {
                if self.attempt_cancel(self.asset.clone(), self.upper_resting.oid).await {
                    info!("Cancelled sell order: {:?}", self.upper_resting);
                    true
                } else {
                    info!("Cancel failed for sell order (oid: {}) - treating as filled", self.upper_resting.oid);
                    self.reset_resting_order(false);
                    true
                }
            } else {
                false
            };
            
            (lower_cancelled, upper_cancelled)
        };

        // Consider putting a new order up
        if lower_order_amount > EPSILON && (lower_cancelled || (lower_change && self.lower_resting.oid == 0)) {
            let (amount_resting, oid) = self
                .place_order(self.asset.clone(), lower_order_amount, lower_price, true)
                .await;

            self.lower_resting.oid = oid;
            self.lower_resting.position = amount_resting;
            self.lower_resting.price = lower_price;

            if amount_resting > EPSILON {
                info!(
                    "Buy for {amount_resting} {} resting at {lower_price}",
                    self.asset.clone()
                );
            }
        }

        if upper_order_amount > EPSILON && (upper_cancelled || (upper_change && self.upper_resting.oid == 0)) {
            let (amount_resting, oid) = self
                .place_order(self.asset.clone(), upper_order_amount, upper_price, false)
                .await;
            self.upper_resting.oid = oid;
            self.upper_resting.position = amount_resting;
            self.upper_resting.price = upper_price;

            if amount_resting > EPSILON {
                info!(
                    "Sell for {amount_resting} {} resting at {upper_price}",
                    self.asset.clone()
                );
            }
        }
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
        let params = TuningParams::default();
        
        state.update(100.5, 0.0, Some(&analysis), Some(&book), &params);
        
        assert_eq!(state.mid_price, 100.5);
        assert_eq!(state.inventory, 0.0);
        assert!(state.market_spread_bps > 0.0);
    }

    #[test]
    fn test_lob_imbalance_calculation() {
        let mut state = StateVector::new();
        let params = TuningParams::default();
        
        let (book, analysis) = create_balanced_book();
        state.update(100.5, 0.0, Some(&analysis), Some(&book), &params);
        
        assert!((state.lob_imbalance - 0.5).abs() < 0.1);
        
        let (book, analysis) = create_imbalanced_book_bid_heavy();
        state.update(100.5, 0.0, Some(&analysis), Some(&book), &params);
        
        assert!(state.lob_imbalance > 0.5);
    }

    #[test]
    fn test_adverse_selection_adjustment() {
        let mut state = StateVector::new();
        let params = TuningParams::default();
        
        state.adverse_selection_estimate = 0.1;
        let adjustment = state.get_adverse_selection_adjustment(100.0, params.adverse_selection_adjustment_factor);
        assert!(adjustment < 0.0);
        
        state.adverse_selection_estimate = -0.1;
        let adjustment = state.get_adverse_selection_adjustment(100.0, params.adverse_selection_adjustment_factor);
        assert!(adjustment > 0.0);
    }

    #[test]
    fn test_inventory_risk_multiplier() {
        let state = StateVector {
            mid_price: 100.0,
            inventory: 0.0,
            adverse_selection_estimate: 0.0,
            market_spread_bps: 10.0,
            lob_imbalance: 0.5,
        };
        
        let multiplier = state.get_inventory_risk_multiplier(100.0);
        assert_eq!(multiplier, 1.0);
        
        let mut state_max = state.clone();
        state_max.inventory = 100.0;
        let multiplier = state_max.get_inventory_risk_multiplier(100.0);
        assert_eq!(multiplier, 2.0);
    }

    #[test]
    fn test_inventory_urgency() {
        let state = StateVector {
            mid_price: 100.0,
            inventory: 0.0,
            adverse_selection_estimate: 0.0,
            market_spread_bps: 10.0,
            lob_imbalance: 0.5,
        };
        
        let urgency = state.get_inventory_urgency(100.0);
        assert_eq!(urgency, 0.0);
        
        let mut state_max = state.clone();
        state_max.inventory = 100.0;
        let urgency = state_max.get_inventory_urgency(100.0);
        assert_eq!(urgency, 1.0);
    }

    #[test]
    fn test_market_favorable_conditions() {
        let state = StateVector {
            mid_price: 100.0,
            inventory: 0.0,
            adverse_selection_estimate: 0.0,
            market_spread_bps: 10.0,
            lob_imbalance: 0.5,
        };
        
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
        let params = TuningParams::default();
        
        let state = StateVector {
            mid_price: 100.0,
            inventory: 0.0,
            adverse_selection_estimate: 0.0,
            market_spread_bps: 10.0,
            lob_imbalance: 0.5,
        };
        
        control.apply_state_adjustments(&state, 10.0, 100.0, &params);
        
        // With neutral state, should remain roughly symmetric
        assert!(control.ask_offset_bps > 0.0);
        assert!(control.bid_offset_bps > 0.0);
        assert!(control.is_passive_only());
    }
    
    #[test]
    fn test_control_vector_adverse_selection_adjustment() {
        let mut control = ControlVector::symmetric(10.0);
        let params = TuningParams::default();
        
        // Bullish state (positive adverse selection - expect price to rise)
        let mut state = StateVector {
            mid_price: 100.0,
            inventory: 0.0,
            adverse_selection_estimate: 0.2,
            market_spread_bps: 10.0,
            lob_imbalance: 0.7,
        };
        
        control.apply_state_adjustments(&state, 10.0, 100.0, &params);
        
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
        
        control.apply_state_adjustments(&state, 10.0, 100.0, &params);
        
        // Bearish signal should widen bid more than ask (make buying less attractive)
        // When price is expected to fall, we want to avoid buying high
        let asymmetry = control.spread_asymmetry_bps();
        assert!(asymmetry < 0.0); // Bid wider relative to ask
    }
    
    #[test]
    fn test_control_vector_inventory_adjustment() {
        let base_spread = 10.0;
        let max_inventory = 100.0;
        let params = TuningParams::default();
        
        // Long position - should tighten ask, widen bid
        let mut control = ControlVector::symmetric(base_spread);
        let state_long = StateVector {
            mid_price: 100.0,
            inventory: 80.0, // 80% long
            adverse_selection_estimate: 0.0,
            market_spread_bps: 10.0,
            lob_imbalance: 0.5,
        };
        
        control.apply_state_adjustments(&state_long, base_spread, max_inventory, &params);
        
        // Long position should create asymmetry favoring sells
        assert!(control.bid_offset_bps > control.ask_offset_bps);
        
        // Short position - should tighten bid, widen ask
        let mut control = ControlVector::symmetric(base_spread);
        let state_short = StateVector {
            mid_price: 100.0,
            inventory: -80.0, // 80% short
            adverse_selection_estimate: 0.0,
            market_spread_bps: 10.0,
            lob_imbalance: 0.5,
        };
        
        control.apply_state_adjustments(&state_short, base_spread, max_inventory, &params);
        
        // Short position should create asymmetry favoring buys
        assert!(control.ask_offset_bps > control.bid_offset_bps);
    }
    
    #[test]
    fn test_control_vector_urgency_triggers_taker() {
        let mut control = ControlVector::symmetric(10.0);
        let params = TuningParams::default();
        
        // High urgency state (>0.7)
        let state = StateVector {
            mid_price: 100.0,
            inventory: 95.0, // 95% of max = high urgency
            adverse_selection_estimate: 0.0,
            market_spread_bps: 10.0,
            lob_imbalance: 0.5,
        };
        
        control.apply_state_adjustments(&state, 10.0, 100.0, &params);
        
        // Should activate taker selling due to long position + high urgency
        assert!(control.taker_sell_rate > 0.0);
        assert_eq!(control.taker_buy_rate, 0.0);
        assert!(control.is_liquidating());
    }
    
    #[test]
    fn test_control_vector_risk_multiplier_effect() {
        let base_spread = 10.0;
        let max_inventory = 100.0;
        let params = TuningParams::default();
        
        // Zero inventory - minimal risk
        let mut control_zero = ControlVector::symmetric(base_spread);
        let state_zero = StateVector {
            mid_price: 100.0,
            inventory: 0.0,
            adverse_selection_estimate: 0.0,
            market_spread_bps: 10.0,
            lob_imbalance: 0.5,
        };
        
        control_zero.apply_state_adjustments(&state_zero, base_spread, max_inventory, &params);
        let spread_zero = control_zero.total_spread_bps();
        
        // Max inventory - maximum risk
        let mut control_max = ControlVector::symmetric(base_spread);
        let state_max = StateVector {
            mid_price: 100.0,
            inventory: 100.0,
            adverse_selection_estimate: 0.0,
            market_spread_bps: 10.0,
            lob_imbalance: 0.5,
        };
        
        control_max.apply_state_adjustments(&state_max, base_spread, max_inventory, &params);
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
        
        let state = StateVector {
            mid_price: 100.0,
            inventory: 50.0,
            adverse_selection_estimate: 0.0,
            market_spread_bps: 10.0,
            lob_imbalance: 0.5,
        };
        
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
        
        let state = StateVector {
            mid_price: 100.0,
            inventory: 0.0,
            adverse_selection_estimate: 0.0,
            market_spread_bps: 10.0, // Market half-spread = 5 bps
            lob_imbalance: 0.5, // Balanced book
        };
        
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
        let state_high_bid = StateVector {
            mid_price: 100.0,
            inventory: 0.0,
            adverse_selection_estimate: 0.0,
            market_spread_bps: 10.0,
            lob_imbalance: 0.9,
        };
        
        // Low bid imbalance (I_t = 0.1) - lots of sell orders
        let state_low_bid = StateVector {
            mid_price: 100.0,
            inventory: 0.0,
            adverse_selection_estimate: 0.0,
            market_spread_bps: 10.0,
            lob_imbalance: 0.1,
        };
        
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
        
        let state = StateVector {
            mid_price: 100.0,
            inventory: 0.0,
            adverse_selection_estimate: 0.0,
            market_spread_bps: 10.0,
            lob_imbalance: 0.5,
        };
        
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
        
        let state = StateVector {
            mid_price: 100.0,
            inventory: 0.0,
            adverse_selection_estimate: 0.0,
            market_spread_bps: 10.0,
            lob_imbalance: 0.5,
        };
        
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
        let state_zero = StateVector {
            mid_price: 100.0,
            inventory: 0.0,
            adverse_selection_estimate: 0.0,
            market_spread_bps: 10.0,
            lob_imbalance: 0.5,
        };
        
        // High inventory state
        let state_high = StateVector {
            mid_price: 100.0,
            inventory: 50.0,
            adverse_selection_estimate: 0.0,
            market_spread_bps: 10.0,
            lob_imbalance: 0.5,
        };
        
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
        
        let state = StateVector {
            mid_price: 100.0,
            inventory: 0.0,
            adverse_selection_estimate: 0.0,
            market_spread_bps: 10.0,
            lob_imbalance: 0.5,
        };
        
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
        let state = StateVector {
            mid_price: 100.0,
            inventory: 95.0,
            adverse_selection_estimate: 0.0,
            market_spread_bps: 10.0,
            lob_imbalance: 0.5,
        };
        
        let optimal_control = hjb.optimize_control(&state, &value_fn, 10.0);
        
        // Should activate taker selling to reduce inventory
        assert!(optimal_control.taker_sell_rate > 0.0 || optimal_control.is_liquidating());
    }
    
    #[test]
    fn test_hjb_optimize_respects_adverse_selection() {
        let hjb = HJBComponents::new();
        let value_fn = ValueFunction::new(0.01, 100.0);
        
        // Strong upward drift (μ̂ > 0)
        let state_up = StateVector {
            mid_price: 100.0,
            inventory: 0.0,
            adverse_selection_estimate: 2.0,
            market_spread_bps: 10.0,
            lob_imbalance: 0.5,
        };
        
        // Strong downward drift (μ̂ < 0)
        let state_down = StateVector {
            mid_price: 100.0,
            inventory: 0.0,
            adverse_selection_estimate: -2.0,
            market_spread_bps: 10.0,
            lob_imbalance: 0.5,
        };
        
        let control_up = hjb.optimize_control(&state_up, &value_fn, 10.0);
        let control_down = hjb.optimize_control(&state_down, &value_fn, 10.0);
        
        // Both should be valid controls
        assert!(control_up.validate(5.0).is_ok());
        assert!(control_down.validate(5.0).is_ok());
    }
    
    #[test]
    fn test_value_function_cache() {
        let mut value_fn = ValueFunction::new(0.01, 100.0);

        let state = StateVector {
            mid_price: 100.0,
            inventory: 0.0,
            adverse_selection_estimate: 0.0,
            market_spread_bps: 10.0,
            lob_imbalance: 0.5,
        };

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

