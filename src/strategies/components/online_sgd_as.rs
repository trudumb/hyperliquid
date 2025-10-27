// ============================================================================
// Online SGD Adverse Selection Model - Data-Driven Drift Estimation
// ============================================================================
//
// This component implements the AdverseSelectionModel trait using online
// linear regression with stochastic gradient descent (SGD). It learns to
// predict short-term price drift from LOB features.
//
// # Algorithm
//
// The model learns weights W = [w_bias, w_trade_flow, w_lob_imb, w_spread, w_vol]
// via SGD to predict price changes:
//
//   Δprice_bps = W · X_t
//
// Where X_t = [1, trade_flow, lob_imbalance - 0.5, spread_bps, vol_bps]
//
// Training:
// 1. Record observation (features, mid_price) at time t
// 2. Wait lookback_ticks (e.g., 10 ticks)
// 3. Compute actual price change: Δprice_actual = price_{t+10} - price_t
// 4. Compute prediction error: error = predicted - actual
// 5. Update weights: W = W - learning_rate * error * X_t
//
// # Features
//
// - **Trade Flow**: EMA of recent taker direction (+1 = buy pressure, -1 = sell)
// - **LOB Imbalance**: Bid volume / (bid + ask volume) - 0.5 (centered at 0)
// - **Market Spread**: Current BBO spread in basis points
// - **Volatility**: Current volatility estimate in basis points
//
// # Online Learning
//
// The model adapts to changing market conditions by continuously updating
// its weights. This is crucial because:
// - Market microstructure changes over time
// - Different regimes have different adverse selection patterns
// - Static models become stale and underperform
//
// # Example
//
// ```rust
// use strategies::components::{AdverseSelectionModel, OnlineSgdAsModel};
//
// let mut as_model = OnlineSgdAsModel::new_default();
//
// // On each market update
// as_model.on_market_update(&market_update);
//
// // Get current estimate
// let drift_bps = as_model.get_adverse_selection_bps();
// ```

use std::sync::Arc;
use parking_lot::RwLock;
use serde_json::Value;

use crate::strategy::MarketUpdate;
use crate::market_maker_v2::OnlineAdverseSelectionModel;
use super::adverse_selection::AdverseSelectionModel;

/// Online SGD-based adverse selection model implementation.
///
/// This component wraps an OnlineAdverseSelectionModel from market_maker_v2
/// and provides the AdverseSelectionModel interface for use in modular strategies.
pub struct OnlineSgdAsModel {
    /// The underlying online model
    model: Arc<RwLock<OnlineAdverseSelectionModel>>,

    /// Cached adverse selection estimate
    cached_estimate_bps: f64,
}

impl OnlineSgdAsModel {
    /// Create a new online SGD adverse selection model with default parameters.
    ///
    /// Default configuration:
    /// - Initial weights: [0.0, 0.4, 0.1, -0.05, 0.02]
    /// - Learning rate: 0.001
    /// - Lookback ticks: 10
    /// - Buffer capacity: 100
    /// - Learning enabled: true
    pub fn new_default() -> Self {
        let model = Arc::new(RwLock::new(OnlineAdverseSelectionModel::default()));

        Self {
            model,
            cached_estimate_bps: 0.0,
        }
    }

    /// Create a new online SGD adverse selection model from JSON config.
    ///
    /// Expected JSON structure:
    /// ```json
    /// {
    ///   "learning_rate": 0.001,
    ///   "lookback_ticks": 10,
    ///   "buffer_capacity": 100,
    ///   "enable_learning": true,
    ///   "initial_weights": [0.0, 0.4, 0.1, -0.05, 0.02]
    /// }
    /// ```
    pub fn from_json(_config: &Value) -> Self {
        // TODO: Parse config and create custom model
        // For now, just use defaults
        Self::new_default()
    }

    /// Get reference to the underlying model (for advanced usage).
    pub fn model(&self) -> Arc<RwLock<OnlineAdverseSelectionModel>> {
        Arc::clone(&self.model)
    }

    /// Get model statistics for logging/monitoring.
    pub fn get_stats(&self) -> String {
        let model = self.model.read();
        model.get_stats()
    }
}

impl AdverseSelectionModel for OnlineSgdAsModel {
    fn on_market_update(&mut self, update: &MarketUpdate) {
        // Update model if we have a mid-price
        if let Some(mid_price) = update.mid_price {
            let mut model = self.model.write();
            model.update(mid_price);

            // For now, we can't call predict() here because we don't have StateVector
            // The cached estimate will be updated by the strategy layer
            // TODO: Refactor to make the model more self-contained
        }
    }

    fn get_adverse_selection_bps(&self) -> f64 {
        self.cached_estimate_bps
    }
}
