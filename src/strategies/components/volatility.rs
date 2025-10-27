// ============================================================================
// Volatility Model Trait - Swappable Volatility Estimation Component
// ============================================================================
//
// This trait defines the interface for any component that can estimate
// market volatility. Different implementations can use different approaches:
// - Particle filters for stochastic volatility
// - GARCH models
// - Simple EMA of returns
// - Implied volatility from options
//
// # Design Philosophy
//
// Volatility models are **stateful** components that:
// - Process incremental market updates (mid-price changes, trades)
// - Maintain internal state (particles, GARCH parameters, etc.)
// - Provide point estimates and uncertainty quantification
//
// # Example Implementation
//
// ```rust
// struct SimpleRealizedVolModel {
//     ema_vol: f64,
//     prev_price: f64,
// }
//
// impl VolatilityModel for SimpleRealizedVolModel {
//     fn on_market_update(&mut self, update: &MarketUpdate) {
//         if let Some(mid_price) = update.mid_price {
//             let ret = (mid_price / self.prev_price).ln().abs() * 10000.0;
//             self.ema_vol = 0.9 * self.ema_vol + 0.1 * ret;
//             self.prev_price = mid_price;
//         }
//     }
//
//     fn get_volatility_bps(&self) -> f64 {
//         self.ema_vol
//     }
//
//     fn get_uncertainty_bps(&self) -> f64 {
//         self.ema_vol * 0.1  // 10% uncertainty
//     }
// }
// ```

use crate::strategy::MarketUpdate;

/// A swappable component for volatility modeling.
///
/// Volatility models process market data and provide estimates of:
/// 1. Current volatility (point estimate in basis points)
/// 2. Uncertainty about that estimate (std dev in basis points)
///
/// The uncertainty quantification is crucial for robust control strategies
/// that need to know "how confident are we in this volatility estimate?"
pub trait VolatilityModel: Send {
    /// Update the model with new market data.
    ///
    /// This method is called on every market update (L2 book, trades, mid-price).
    /// The model should extract relevant information (e.g., mid-price changes)
    /// and update its internal state accordingly.
    ///
    /// # Arguments
    /// - `update`: Market data update containing L2 book, trades, or mid-price
    ///
    /// # Notes
    /// - Models should handle missing data gracefully (e.g., None for mid_price)
    /// - Updates should be fast (hot path) - expensive computation should be async
    fn on_market_update(&mut self, update: &MarketUpdate);

    /// Get the current volatility estimate in basis points (annualized).
    ///
    /// Returns the model's best estimate of current market volatility.
    /// For example, 100 bps = 1% volatility.
    ///
    /// # Returns
    /// Current volatility estimate in basis points (positive value)
    ///
    /// # Notes
    /// - Should always return a positive value
    /// - If model is uninitialized, return a reasonable default (e.g., 100 bps)
    fn get_volatility_bps(&self) -> f64;

    /// Get the uncertainty (std dev) of the volatility estimate in basis points.
    ///
    /// This quantifies "how confident are we in our volatility estimate?"
    /// Used by robust control to widen spreads when uncertainty is high.
    ///
    /// # Returns
    /// Standard deviation of volatility estimate in basis points
    ///
    /// # Example
    /// ```text
    /// If get_volatility_bps() = 100 and get_uncertainty_bps() = 10,
    /// then we're ~95% confident volatility is in [80, 120] bps.
    /// ```
    ///
    /// # Notes
    /// - Should always return a non-negative value
    /// - If uncertainty cannot be quantified, return a conservative estimate
    fn get_uncertainty_bps(&self) -> f64;
}
