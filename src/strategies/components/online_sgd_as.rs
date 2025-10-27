// ============================================================================
// Online SGD Adverse Selection Model - Adaptive Learning Implementation
// ============================================================================
//
// This component uses stochastic gradient descent (SGD) to learn a linear
// model that predicts short-term price drift from microstructure features.
//
// # Model
//
// The model learns weights W = [w_bias, w_trade_flow, w_lob_imb, w_spread, w_vol]
// to predict short-term price drift:
//
// μ̂_t = W · X_t
//
// where X_t = [1, trade_flow, lob_imbalance - 0.5, market_spread, volatility]
//
// # Learning Algorithm
//
// Uses online SGD with delayed labels:
// 1. Observe features X_t at time t
// 2. Make prediction μ̂_t
// 3. Wait `lookback_ticks` ticks
// 4. Compute actual price change Δ_actual = (S_{t+k} - S_t) / S_t
// 5. Update weights: W ← W - lr * (μ̂_t - Δ_actual) * X_t
//
// # Key Features
//
// - **Online Learning**: Continuously adapts to market regime changes
// - **Feature Normalization**: Uses Welford's algorithm for online z-score normalization
// - **Delayed Labels**: Waits for actual price movement before updating
// - **Monitoring**: Tracks MAE and update count for diagnostics
//
// # References
//
// - Bottou, L. (2010). "Large-Scale Machine Learning with Stochastic Gradient Descent"
// - Welford, B. P. (1962). "Note on a Method for Calculating Corrected Sums of Squares and Products"

use crate::{
    strategy::MarketUpdate,
    strategies::components::AdverseSelectionModel,
};
use log::info;
use std::collections::VecDeque;

/// Internal state representation extracted from market updates
#[derive(Clone, Debug)]
struct InternalState {
    mid_price: f64,
    trade_flow_ema: f64,
    lob_imbalance: f64,
    market_spread_bps: f64,
    volatility_ema_bps: f64,
    previous_mid_price: f64,
}

impl Default for InternalState {
    fn default() -> Self {
        Self {
            mid_price: 0.0,
            trade_flow_ema: 0.0,
            lob_imbalance: 0.5, // Neutral
            market_spread_bps: 0.0,
            volatility_ema_bps: 0.0,
            previous_mid_price: 0.0,
        }
    }
}

/// Online SGD-based adverse selection model
pub struct OnlineSgdAsModel {
    /// Current internal state
    state: InternalState,

    /// Regression weights: W = [w_bias, w_trade_flow, w_lob_imb, w_spread, w_vol]
    weights: Vec<f64>,

    /// Learning rate for SGD updates
    learning_rate: f64,

    /// Number of ticks to wait before computing actual price change
    lookback_ticks: usize,

    /// Observation buffer: stores (features, mid_price) for delayed label computation
    observation_buffer: VecDeque<(Vec<f64>, f64)>,

    /// Maximum buffer capacity
    buffer_capacity: usize,

    /// Enable/disable online learning
    enable_learning: bool,

    /// Count of SGD updates performed
    update_count: usize,

    /// Running average of prediction error (MAE)
    mean_absolute_error: f64,

    /// Decay factor for MAE averaging
    mae_decay: f64,

    /// Welford's algorithm for online mean/variance
    /// [(count, mean, M2), ...] for each feature
    feature_stats: Vec<(f64, f64, f64)>,

    /// EMA alpha for trade flow and volatility
    ema_alpha: f64,

    /// Volatility calculation window (for realized vol estimation)
    volatility_window_secs: f64,
    last_update_time: Option<std::time::Instant>,
}

impl Default for OnlineSgdAsModel {
    fn default() -> Self {
        Self::new()
    }
}

impl OnlineSgdAsModel {
    /// Create a new online SGD adverse selection model
    pub fn new() -> Self {
        Self {
            state: InternalState::default(),
            // Initialize weights based on 80/20 heuristic knowledge
            weights: vec![0.0, 0.4, 0.1, -0.05, 0.02],
            learning_rate: 0.001,
            lookback_ticks: 10, // Predict 10 ticks ahead
            observation_buffer: VecDeque::with_capacity(100),
            buffer_capacity: 100,
            enable_learning: true,
            update_count: 0,
            mean_absolute_error: 0.0,
            mae_decay: 0.99,
            feature_stats: vec![(0.0, 0.0, 0.0); 4], // 4 features (bias not normalized)
            ema_alpha: 0.1,
            volatility_window_secs: 60.0,
            last_update_time: None,
        }
    }

    /// Create new model (alias for new())
    pub fn new_default() -> Self {
        Self::new()
    }

    /// Create with custom parameters
    pub fn with_params(
        learning_rate: f64,
        lookback_ticks: usize,
        buffer_capacity: usize,
    ) -> Self {
        Self {
            learning_rate,
            lookback_ticks,
            buffer_capacity,
            observation_buffer: VecDeque::with_capacity(buffer_capacity),
            ..Self::new()
        }
    }

    /// Extract features from current state
    fn get_raw_features(&self) -> Vec<f64> {
        vec![
            self.state.trade_flow_ema,
            self.state.lob_imbalance - 0.5, // Center around 0
            self.state.market_spread_bps,
            self.state.volatility_ema_bps,
        ]
    }

    /// Update running feature statistics using Welford's online algorithm
    fn update_feature_stats(&mut self) {
        let raw_features = self.get_raw_features();

        for i in 0..raw_features.len() {
            let x = raw_features[i];

            // Welford's online algorithm for mean and variance
            let (count, mean, m2) = &mut self.feature_stats[i];
            *count += 1.0;
            let delta = x - *mean;
            *mean += delta / *count;
            let delta2 = x - *mean; // New mean
            *m2 += delta * delta2;
        }
    }

    /// Get normalized features using current statistics
    fn get_normalized_features(&self) -> Vec<f64> {
        let raw_features = self.get_raw_features();
        let mut normalized_features = vec![1.0]; // Start with bias term

        for i in 0..raw_features.len() {
            let x = raw_features[i];
            let (count, mean, m2) = &self.feature_stats[i];

            // Get variance and std_dev from current stats
            let (mean, std_dev) = if *count < 2.0 {
                (0.0, 1.0) // Not enough data, pass through
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

    /// Predict short-term price drift
    fn predict(&self) -> f64 {
        let features = self.get_normalized_features();

        // Dot product: prediction = sum(w_i * x_i)
        self.weights.iter()
            .zip(features.iter())
            .map(|(w, x)| w * x)
            .sum()
    }

    /// Record observation for delayed SGD update
    fn record_observation(&mut self) {
        let features = self.get_normalized_features();
        let mid_price = self.state.mid_price;

        // Add to buffer
        self.observation_buffer.push_back((features, mid_price));

        // Maintain buffer capacity
        if self.observation_buffer.len() > self.buffer_capacity {
            self.observation_buffer.pop_front();
        }
    }

    /// Perform SGD update with delayed labels
    fn sgd_update(&mut self) {
        if !self.enable_learning {
            return;
        }

        // Need at least lookback_ticks observations
        if self.observation_buffer.len() <= self.lookback_ticks {
            return;
        }

        let current_mid_price = self.state.mid_price;

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
            for i in 0..self.weights.len() {
                self.weights[i] -= self.learning_rate * error * features[i];
            }

            self.update_count += 1;

            // Log update every 100 iterations
            if self.update_count % 100 == 0 {
                info!(
                    "Online AS Model Update #{}: MAE={:.4}bps, Weights={:?}",
                    self.update_count, self.mean_absolute_error, self.weights
                );
            }
        }
    }

    /// Update internal state from market update
    fn update_state(&mut self, update: &MarketUpdate) {
        let now = std::time::Instant::now();

        // Update mid-price
        if let Some(mid_price) = update.mid_price {
            // Update previous before setting new
            if self.state.mid_price > 0.0 {
                self.state.previous_mid_price = self.state.mid_price;
            }
            self.state.mid_price = mid_price;

            // Update volatility (realized vol from price changes)
            if self.state.previous_mid_price > 0.0 && self.state.mid_price > 0.0 {
                if let Some(last_time) = self.last_update_time {
                    let dt = now.duration_since(last_time).as_secs_f64();
                    if dt > 0.0 {
                        let log_return = (self.state.mid_price / self.state.previous_mid_price).ln();
                        let realized_vol = log_return.abs() / dt.sqrt();
                        let realized_vol_bps = realized_vol * 10000.0;

                        // EMA update
                        if self.state.volatility_ema_bps > 0.0 {
                            self.state.volatility_ema_bps =
                                self.ema_alpha * realized_vol_bps +
                                (1.0 - self.ema_alpha) * self.state.volatility_ema_bps;
                        } else {
                            self.state.volatility_ema_bps = realized_vol_bps;
                        }
                    }
                }
            }
        }

        // Update LOB imbalance from L2 book
        if let Some(ref book) = update.l2_book {
            // L2BookData has levels[0] = bids, levels[1] = asks
            if book.levels.len() >= 2 {
                let bids = &book.levels[0];
                let asks = &book.levels[1];

                if !bids.is_empty() && !asks.is_empty() {
                    // Parse string values to f64
                    if let (Ok(bid_volume), Ok(ask_volume)) = (
                        bids[0].sz.parse::<f64>(),
                        asks[0].sz.parse::<f64>(),
                    ) {
                        let total_volume = bid_volume + ask_volume;

                        if total_volume > 0.0 {
                            self.state.lob_imbalance = bid_volume / total_volume;
                        }
                    }

                    // Calculate market spread
                    if let (Ok(best_bid), Ok(best_ask)) = (
                        bids[0].px.parse::<f64>(),
                        asks[0].px.parse::<f64>(),
                    ) {
                        if best_bid > 0.0 && best_ask > best_bid {
                            let mid = (best_bid + best_ask) / 2.0;
                            self.state.market_spread_bps = ((best_ask - best_bid) / mid) * 10000.0;
                        }
                    }
                }
            }
        }

        // Update trade flow from trades
        if !update.trades.is_empty() {
            for trade in &update.trades {
                // +1 for taker buy (bullish), -1 for taker sell (bearish)
                let flow_signal = if trade.side == "buy" { 1.0 } else { -1.0 };

                // EMA update
                if self.update_count > 0 {
                    self.state.trade_flow_ema =
                        self.ema_alpha * flow_signal +
                        (1.0 - self.ema_alpha) * self.state.trade_flow_ema;
                } else {
                    self.state.trade_flow_ema = flow_signal;
                }
            }
        }

        self.last_update_time = Some(now);
    }

    /// Get model statistics for logging/monitoring
    pub fn get_stats(&self) -> String {
        format!(
            "OnlineModel[updates={}, MAE={:.4}bps, lr={:.6}, enabled={}]",
            self.update_count, self.mean_absolute_error, self.learning_rate, self.enable_learning
        )
    }
}

impl AdverseSelectionModel for OnlineSgdAsModel {
    fn on_market_update(&mut self, update: &MarketUpdate) {
        // Update internal state from market update
        self.update_state(update);

        // Update feature statistics
        self.update_feature_stats();

        // Record current observation for delayed learning
        self.record_observation();

        // Perform SGD update with delayed labels
        self.sgd_update();
    }

    fn get_adverse_selection_bps(&self) -> f64 {
        // Return current prediction
        let prediction = self.predict();

        // Clamp to reasonable bounds to prevent extreme values
        prediction.clamp(-100.0, 100.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_initialization() {
        let model = OnlineSgdAsModel::new();
        assert_eq!(model.weights.len(), 5); // bias + 4 features
        assert_eq!(model.feature_stats.len(), 4);
        assert_eq!(model.update_count, 0);
    }

    #[test]
    fn test_feature_normalization() {
        let mut model = OnlineSgdAsModel::new();

        // Set some state
        model.state.trade_flow_ema = 0.5;
        model.state.lob_imbalance = 0.7;
        model.state.market_spread_bps = 10.0;
        model.state.volatility_ema_bps = 100.0;

        // Update stats
        model.update_feature_stats();

        // Get normalized features
        let features = model.get_normalized_features();
        assert_eq!(features.len(), 5); // bias + 4 features
        assert_eq!(features[0], 1.0); // Bias term
    }

    #[test]
    fn test_prediction() {
        let mut model = OnlineSgdAsModel::new();

        // Set some state
        model.state.trade_flow_ema = 0.5;
        model.state.lob_imbalance = 0.7;
        model.state.market_spread_bps = 10.0;
        model.state.volatility_ema_bps = 100.0;

        let prediction = model.get_adverse_selection_bps();

        // Should be bounded
        assert!(prediction.abs() <= 100.0);
    }
}
