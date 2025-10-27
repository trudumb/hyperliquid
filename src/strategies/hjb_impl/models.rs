//! Contains core HJB models and components for the strategy.
//!
//! This module includes:
//! - OnlineAdverseSelectionModel: Online learning for adverse selection estimation
//! - ControlVector: HJB optimal controls (quote offsets and taker rates)
//! - ValueFunction: Value function approximation for inventory management
//! - HJBComponents: Core HJB equation components and optimization

use super::state::StateVector;
use crate::consts::EPSILON;
use log::info;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};

/// Online Linear Regression Model for Adverse Selection Estimation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OnlineAdverseSelectionModel {
    pub weights: Vec<f64>,
    pub learning_rate: f64,
    pub lookback_ticks: usize,
    pub observation_buffer: VecDeque<(Vec<f64>, f64)>,
    pub buffer_capacity: usize,
    pub enable_learning: bool,
    pub update_count: usize,
    pub mean_absolute_error: f64,
    pub mae_decay: f64,
    pub feature_stats: Vec<(f64, f64, f64)>,
}

impl Default for OnlineAdverseSelectionModel {
    fn default() -> Self {
        Self {
            weights: vec![0.0, 0.4, 0.1, -0.05, 0.02],
            learning_rate: 0.001,
            lookback_ticks: 10,
            observation_buffer: VecDeque::with_capacity(100),
            buffer_capacity: 100,
            enable_learning: true,
            update_count: 0,
            mean_absolute_error: 0.0,
            mae_decay: 0.99,
            feature_stats: vec![(0.0, 0.0, 0.0); 4],
        }
    }
}

impl OnlineAdverseSelectionModel {
    pub fn update_feature_stats(&mut self, state: &StateVector) {
        let raw_features = vec![
            state.trade_flow_ema,
            state.lob_imbalance - 0.5,
            state.market_spread_bps,
            state.volatility_ema_bps,
        ];

        for i in 0..raw_features.len() {
            let x = raw_features[i];
            let (count, mean, m2) = &mut self.feature_stats[i];
            *count += 1.0;
            let delta = x - *mean;
            *mean += delta / *count;
            let delta2 = x - *mean;
            *m2 += delta * delta2;
        }
    }

    fn get_normalized_features(&self, state: &StateVector) -> Vec<f64> {
        let raw_features = vec![
            state.trade_flow_ema,
            state.lob_imbalance - 0.5,
            state.market_spread_bps,
            state.volatility_ema_bps,
        ];
        let mut normalized_features = vec![1.0];
        for i in 0..raw_features.len() {
            let x = raw_features[i];
            let (count, mean, m2) = &self.feature_stats[i];
            let (mean, std_dev) = if *count < 2.0 {
                (0.0, 1.0)
            } else {
                let variance = *m2 / (*count - 1.0);
                let std_dev = variance.sqrt().max(1e-6);
                (*mean, std_dev)
            };
            normalized_features.push((x - mean) / std_dev);
        }
        normalized_features
    }

    pub fn predict(&self, state: &StateVector) -> f64 {
        let features = self.get_normalized_features(state);
        self.weights
            .iter()
            .zip(features.iter())
            .map(|(w, x)| w * x)
            .sum()
    }

    pub fn record_observation(&mut self, state: &StateVector, mid_price: f64) {
        let features = self.get_normalized_features(state);
        self.observation_buffer.push_back((features, mid_price));
        if self.observation_buffer.len() > self.buffer_capacity {
            self.observation_buffer.pop_front();
        }
    }

    pub fn update(&mut self, current_mid_price: f64) {
        if !self.enable_learning {
            return;
        }
        if self.observation_buffer.len() <= self.lookback_ticks {
            return;
        }
        let lookback_idx = self.observation_buffer.len() - self.lookback_ticks - 1;
        if let Some((features, old_mid_price)) = self.observation_buffer.get(lookback_idx) {
            let actual_change_bps = if *old_mid_price > 0.0 {
                ((current_mid_price - old_mid_price) / old_mid_price) * 10000.0
            } else {
                0.0
            };
            let predicted_change_bps: f64 = self
                .weights
                .iter()
                .zip(features.iter())
                .map(|(w, x)| w * x)
                .sum();
            let error = predicted_change_bps - actual_change_bps;
            let abs_error = error.abs();
            if self.update_count == 0 {
                self.mean_absolute_error = abs_error;
            } else {
                self.mean_absolute_error = self.mae_decay * self.mean_absolute_error
                    + (1.0 - self.mae_decay) * abs_error;
            }
            for i in 0..self.weights.len() {
                self.weights[i] -= self.learning_rate * error * features[i];
            }
            self.update_count += 1;
            if self.update_count % 100 == 0 {
                info!(
                    "Online Adverse Selection Model Update #{}: MAE={:.4}bps, Weights={:?}",
                    self.update_count, self.mean_absolute_error, self.weights
                );
            }
        }
    }

    pub fn get_stats(&self) -> String {
        format!(
            "OnlineModel[updates={}, MAE={:.4}bps, lr={:.6}, enabled={}]",
            self.update_count,
            self.mean_absolute_error,
            self.learning_rate,
            self.enable_learning
        )
    }

    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

// =============================================================================
// Control Vector: HJB Optimal Controls
// =============================================================================

/// Control Vector α_t = (δ^a_t, δ^b_t, ν^a_t, ν^b_t)
///
/// Represents the optimal control decision for the HJB equation:
/// - δ^a_t: ask quote offset (bps from mid)
/// - δ^b_t: bid quote offset (bps from mid)
/// - ν^a_t: taker sell rate (aggressive liquidation)
/// - ν^b_t: taker buy rate (aggressive accumulation)
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
}

impl Default for ControlVector {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Value Function: Inventory Penalty & Expected Value
// =============================================================================

/// Value Function V(Q, Z, t)
///
/// Approximates the value of being in state (Q, Z) at time t.
/// Used in HJB equation to evaluate control decisions.
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
    value_cache: HashMap<i32, f64>,
}

impl ValueFunction {
    /// Create a new value function with given parameters
    pub fn new(phi: f64, terminal_time: f64) -> Self {
        Self {
            phi,
            terminal_time,
            current_time: 0.0,
            value_cache: HashMap::new(),
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

// =============================================================================
// HJB Components: Core Equation & Optimization
// =============================================================================

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
