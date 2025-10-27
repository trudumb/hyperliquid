// ============================================================================
// Fill Model Trait - Swappable Fill Rate Estimation Component
// ============================================================================
//
// This trait defines the interface for any component that can model fill
// probabilities and rates. Different implementations can use:
// - Hawkes processes (self-exciting point processes)
// - Poisson processes with fixed rates
// - Machine learning models trained on historical fills
// - Queue position models based on LOB depth
//
// # Design Philosophy
//
// Fill models are **stateful** components that:
// - Learn from observed fills (user trade confirmations)
// - Estimate fill rates at different price levels
// - Provide state to the optimizer for decision-making
//
// # Example Implementation
//
// ```rust
// struct SimplePoissonFillModel {
//     base_rate: f64,  // fills per second at L1
//     level_decay: f64,  // decay factor for deeper levels
// }
//
// impl FillModel for SimplePoissonFillModel {
//     fn on_fills(&mut self, fills: &[TradeInfo], _current_time_sec: f64) {
//         // Could use fill data to update base_rate estimate
//         self.base_rate = fills.len() as f64 / 60.0;  // fills per second
//     }
//
//     fn get_hawkes_model(&self) -> &crate::HawkesFillModel {
//         // For compatibility, return a reference to internal Hawkes model
//         // (or implement a more generic interface in the future)
//         &self.hawkes_model
//     }
// }
// ```

use crate::TradeInfo;
use crate::HawkesFillModel;

/// A swappable component for modeling fill rates.
///
/// Fill models are responsible for:
/// 1. Learning from observed fills (on_fills)
/// 2. Estimating fill probabilities at different price levels
/// 3. Providing state to the optimizer for optimal quote calculation
///
/// The current interface is designed around the HawkesFillModel, but could
/// be generalized in the future to support other fill modeling approaches.
pub trait FillModel: Send {
    /// Update the model with new user fills.
    ///
    /// This method is called whenever the bot gets filled (maker orders executed).
    /// The model should use this information to update its internal parameters.
    ///
    /// # Arguments
    /// - `fills`: List of trade confirmations from the exchange
    /// - `current_time_sec`: Current time in seconds (Unix timestamp)
    ///
    /// # Notes
    /// - Fills are reported by the exchange after execution
    /// - The model should extract level information if available
    /// - For multi-level models, tracking which level got filled is crucial
    ///
    /// # Example
    /// ```text
    /// If we get filled on a bid at level 0 (L1), the Hawkes model
    /// records this event and updates its intensity estimate for L1 bids.
    /// ```
    fn on_fills(&mut self, fills: &[TradeInfo], current_time_sec: f64);

    /// Provides a reference to the internal Hawkes model for the optimizer.
    ///
    /// **Note**: This is a temporary interface to maintain compatibility with
    /// the existing HjbMultiLevelOptimizer. In a fully generalized design, we
    /// would define a more abstract interface like:
    ///
    /// ```rust
    /// fn get_fill_probability(&self, level: usize, offset_bps: f64, side: Side) -> f64;
    /// ```
    ///
    /// For now, we provide direct access to the HawkesFillModel so the
    /// optimizer can query fill rates at different levels.
    ///
    /// # Returns
    /// Reference to the internal HawkesFillModel
    ///
    /// # Future Work
    /// - Define a generic FillRateProvider trait
    /// - Allow optimizers to query fill rates without knowing the model type
    fn get_hawkes_model(&self) -> &HawkesFillModel;
}
