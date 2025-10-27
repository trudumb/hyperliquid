// ============================================================================
// Adverse Selection Model Trait - Swappable Adverse Selection Estimation
// ============================================================================
//
// This trait defines the interface for any component that can estimate
// adverse selection (short-term predictable price drift). Different approaches:
// - Online SGD learning from LOB features
// - Fixed 80/20 heuristic (imbalance-based)
// - Machine learning models (gradient boosting, neural nets)
// - Microstructure models (Kyle's lambda, PIN)
//
// # What is Adverse Selection?
//
// Adverse selection occurs when informed traders systematically trade against
// market makers, causing prices to move against the MM's position immediately
// after being filled.
//
// For example:
// - If we sell at the ask and price immediately goes up, we suffered adverse selection
// - If we buy at the bid and price immediately goes down, we suffered adverse selection
//
// Estimating adverse selection helps us:
// - Widen spreads when informed trading is likely
// - Skew quotes away from the direction of predicted drift
// - Adjust inventory management to minimize adverse PnL
//
// # Design Philosophy
//
// Adverse selection models are **stateful** components that:
// - Process incremental market updates (price changes, LOB imbalance, trade flow)
// - Learn patterns that predict short-term price drift
// - Provide point estimates of expected drift (in basis points)
//
// # Example Implementation
//
// ```rust
// struct SimpleImbalanceModel {
//     last_imbalance: f64,
//     sensitivity: f64,
// }
//
// impl AdverseSelectionModel for SimpleImbalanceModel {
//     fn on_market_update(&mut self, update: &MarketUpdate) {
//         // Extract LOB imbalance from book
//         if let Some(book) = update.l2_book {
//             self.last_imbalance = calculate_imbalance(&book);
//         }
//     }
//
//     fn get_adverse_selection_bps(&self) -> f64 {
//         // Simple heuristic: imbalance > 0.8 = bullish, < 0.2 = bearish
//         self.sensitivity * (self.last_imbalance - 0.5)
//     }
// }
// ```

use crate::strategy::MarketUpdate;

/// A swappable component for estimating adverse selection.
///
/// Adverse selection models process market data and provide an estimate of
/// short-term predictable price drift (μ̂_t in the HJB equation).
///
/// The estimate is in **basis points**, where:
/// - Positive values = bullish drift (price expected to go up)
/// - Negative values = bearish drift (price expected to go down)
/// - Zero = no predictable drift (random walk)
pub trait AdverseSelectionModel: Send {
    /// Update the model with new market data.
    ///
    /// This method is called on every market update (L2 book, trades, mid-price).
    /// The model should extract relevant features (LOB imbalance, trade flow,
    /// spread, etc.) and update its internal state.
    ///
    /// # Arguments
    /// - `update`: Market data update containing L2 book, trades, or mid-price
    ///
    /// # Notes
    /// - Models should handle missing data gracefully
    /// - Updates should be fast (hot path)
    /// - Learning (SGD updates) can happen asynchronously or with delay
    ///
    /// # Example
    /// ```text
    /// On mid-price update:
    /// 1. Observe price change since last update
    /// 2. Compute prediction error (predicted vs. actual)
    /// 3. Update model weights via SGD
    /// 4. Store new observation for next update
    /// ```
    fn on_market_update(&mut self, update: &MarketUpdate);

    /// Get the current adverse selection estimate in basis points.
    ///
    /// Returns the model's best estimate of short-term predictable price drift.
    ///
    /// # Returns
    /// Expected price drift in basis points:
    /// - `+10.0` = expect price to go up by 10 bps
    /// - `-5.0` = expect price to go down by 5 bps
    /// - `0.0` = no predictable drift
    ///
    /// # Notes
    /// - Estimates should be bounded (e.g., [-100, +100] bps)
    /// - If model is uninitialized, return 0.0 (neutral)
    ///
    /// # Usage in Strategy
    /// ```text
    /// If adverse_selection = +10 bps (bullish):
    /// - Widen ask quotes more than bid quotes
    /// - Reduce ask size, increase bid size
    /// - Bias inventory toward long
    /// ```
    fn get_adverse_selection_bps(&self) -> f64;
}
