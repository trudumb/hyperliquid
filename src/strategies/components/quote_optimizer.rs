// ============================================================================
// Quote Optimizer Trait - Swappable Quote Calculation Component
// ============================================================================
//
// This trait defines the interface for any component that can calculate
// optimal quotes given market state and model outputs. Different approaches:
// - HJB-based optimization with Hawkes fill rates
// - Simple spread-based quoting (symmetric or skewed)
// - Reinforcement learning policies
// - Model predictive control (MPC)
//
// # Design Philosophy
//
// Quote optimizers are the "brain" of the strategy. They:
// - Take all model outputs as input (volatility, adverse selection, fill rates)
// - Consider current state (inventory, position limits, market conditions)
// - Solve an optimization problem to find optimal quotes
// - Return target bids and asks (price, size pairs)
//
// The optimizer is **stateless** - all state is passed in via OptimizerInputs
// and CurrentState. This makes it easier to test and swap implementations.
//
// # Example Implementation
//
// ```rust
// struct SimpleSpreadOptimizer {
//     target_spread_bps: f64,
//     order_size: f64,
// }
//
// impl QuoteOptimizer for SimpleSpreadOptimizer {
//     fn calculate_target_quotes(
//         &self,
//         inputs: &OptimizerInputs,
//         state: &CurrentState,
//         _fill_model: &HawkesFillModel,
//     ) -> (Vec<(f64, f64)>, Vec<(f64, f64)>) {
//         let mid = state.l2_mid_price;
//         let half_spread = self.target_spread_bps / 2.0 / 10000.0;
//
//         // Apply inventory skew
//         let skew = state.position / state.max_position_size * 0.5;
//
//         let bid_price = mid * (1.0 - half_spread - skew);
//         let ask_price = mid * (1.0 + half_spread - skew);
//
//         let bids = vec![(bid_price, self.order_size)];
//         let asks = vec![(ask_price, self.order_size)];
//
//         (bids, asks)
//     }
// }
// ```

use crate::strategy::CurrentState;
use crate::HawkesFillModel;

/// Input struct for the optimizer.
///
/// This struct bundles all the model outputs and market state needed
/// by the optimizer to calculate optimal quotes.
///
/// # Fields
/// - `current_time_sec`: Current time for time-dependent calculations
/// - `volatility_bps`: Point estimate of volatility from VolatilityModel
/// - `vol_uncertainty_bps`: Uncertainty (std dev) of volatility estimate
/// - `adverse_selection_bps`: Estimated short-term drift from AdverseSelectionModel
/// - `lob_imbalance`: LOB imbalance (0.0 = all asks, 1.0 = all bids)
///
/// # Design Note
///
/// We use a struct instead of individual parameters to:
/// - Make it easy to add new inputs without changing function signatures
/// - Group related data together
/// - Allow for easy serialization/logging
#[derive(Debug, Clone)]
pub struct OptimizerInputs {
    /// Current time in seconds (Unix timestamp)
    pub current_time_sec: f64,

    /// Volatility estimate in basis points (from VolatilityModel)
    pub volatility_bps: f64,

    /// Volatility uncertainty in basis points (from VolatilityModel)
    pub vol_uncertainty_bps: f64,

    /// Adverse selection estimate in basis points (from AdverseSelectionModel)
    pub adverse_selection_bps: f64,

    /// LOB imbalance (bid_volume / (bid_volume + ask_volume))
    /// Range: [0.0, 1.0] where 0.5 = balanced
    pub lob_imbalance: f64,
}

/// A swappable component for calculating optimal quotes.
///
/// Quote optimizers take all model outputs and current state, then solve
/// an optimization problem to determine the best bid and ask quotes.
///
/// The optimizer returns a list of (price, size) pairs for each side,
/// supporting both single-level and multi-level quoting.
pub trait QuoteOptimizer: Send {
    /// Calculate the target bid and ask quotes (price, size).
    ///
    /// This is the core method where the optimization happens. Given:
    /// - Model outputs (volatility, adverse selection)
    /// - Current state (inventory, position limits, market data)
    /// - Fill model (to estimate execution probabilities)
    ///
    /// The optimizer calculates the optimal quotes that maximize expected utility.
    ///
    /// # Arguments
    /// - `inputs`: Model outputs and market state (volatility, adverse selection, etc.)
    /// - `state`: Current bot state (inventory, orders, market data)
    /// - `fill_model`: Fill rate model for estimating execution probabilities
    ///
    /// # Returns
    /// A tuple of (target_bids, target_asks) where:
    /// - `target_bids`: Vec of (price, size) for bid orders, sorted descending by price
    /// - `target_asks`: Vec of (price, size) for ask orders, sorted ascending by price
    ///
    /// # Notes
    /// - Prices should be in the same units as state.l2_mid_price (absolute USD, not BPS)
    /// - Sizes should be in native asset units (e.g., 10.0 = 10 units of HYPE)
    /// - Empty vectors are valid (e.g., if inventory limit hit on one side)
    /// - Prices should NOT be rounded to tick size - the strategy layer handles that
    ///
    /// # Example
    /// ```text
    /// For a single-level strategy with mid = 100.0:
    /// Returns:
    ///   bids = [(99.90, 10.0)]  // 10 bps below mid, size 10
    ///   asks = [(100.10, 10.0)] // 10 bps above mid, size 10
    ///
    /// For a multi-level strategy:
    /// Returns:
    ///   bids = [(99.90, 5.0), (99.80, 5.0), (99.70, 5.0)]  // L1, L2, L3
    ///   asks = [(100.10, 5.0), (100.20, 5.0), (100.30, 5.0)]  // L1, L2, L3
    /// ```
    fn calculate_target_quotes(
        &self,
        inputs: &OptimizerInputs,
        state: &CurrentState,
        fill_model: &HawkesFillModel,
    ) -> (Vec<(f64, f64)>, Vec<(f64, f64)>);
}
