// ============================================================================
// Inventory Skew Trait - Swappable Position Management Component
// ============================================================================
//
// This trait defines the interface for any component that can calculate
// inventory-based price adjustments. Different implementations can use:
// - Simple linear skewing based on position ratio
// - Book imbalance-aware skewing
// - Time-weighted position skewing
// - Risk-adjusted inventory management
//
// # Design Philosophy
//
// Inventory skew components are **stateless calculators** that:
// - Take current state (position, book analysis) as input
// - Calculate optimal price adjustments to manage inventory risk
// - Return skew adjustments in basis points
//
// The component is stateless - all state is passed in via parameters.
// This makes it easier to test and swap implementations.
//
// # Example Implementation
//
// ```rust
// struct SimpleLinearSkew {
//     skew_factor: f64,
// }
//
// impl InventorySkewModel for SimpleLinearSkew {
//     fn calculate_skew(
//         &self,
//         current_position: f64,
//         max_position: f64,
//         _book_analysis: Option<&BookAnalysis>,
//         base_half_spread_bps: f64,
//     ) -> SkewResult {
//         let position_ratio = current_position / max_position;
//         let skew_bps = position_ratio * self.skew_factor * base_half_spread_bps;
//
//         SkewResult {
//             skew_bps,
//             position_component_bps: skew_bps,
//             book_component_bps: 0.0,
//             position_ratio,
//         }
//     }
// }
// ```

use super::book_analyzer::BookAnalysis;

/// Result of inventory skew calculation
#[derive(Debug, Clone)]
pub struct SkewResult {
    /// The total skew in basis points (positive = shift quotes up, negative = shift down)
    pub skew_bps: f64,

    /// Contribution from position (basis points)
    pub position_component_bps: f64,

    /// Contribution from book imbalance (basis points)
    pub book_component_bps: f64,

    /// Current position ratio (-1.0 to 1.0)
    pub position_ratio: f64,
}

/// A swappable component for inventory-based price skewing.
///
/// Inventory skew models calculate optimal price adjustments to:
/// 1. Reduce inventory risk by making it easier to exit positions
/// 2. React to order book imbalances
/// 3. Balance between profit maximization and risk management
///
/// The skew is returned in basis points and should be applied symmetrically
/// to both bid and ask quotes (shifting the entire quote schedule).
pub trait InventorySkewModel: Send {
    /// Calculate the price skew based on position and book conditions.
    ///
    /// This method computes how much to shift quotes (in bps) to encourage
    /// position reduction and react to market conditions.
    ///
    /// # Arguments
    /// - `current_position`: Current inventory position (positive = long, negative = short)
    /// - `max_position`: Maximum allowed absolute position size
    /// - `book_analysis`: Optional order book analysis for imbalance detection
    /// - `base_half_spread_bps`: Base half spread in basis points (for scaling)
    ///
    /// # Returns
    /// A SkewResult containing:
    /// - Total skew in basis points
    /// - Position component (from inventory risk)
    /// - Book component (from order book imbalance)
    /// - Position ratio (current_position / max_position)
    ///
    /// # Skew Logic
    /// - **Positive skew**: Shift quotes UP (make selling more attractive)
    ///   - Used when LONG to encourage selling
    ///   - Used when order book has bid pressure
    /// - **Negative skew**: Shift quotes DOWN (make buying more attractive)
    ///   - Used when SHORT to encourage buying
    ///   - Used when order book has ask pressure
    ///
    /// # Example
    /// ```text
    /// If position = +40, max_position = 50, position_ratio = 0.8:
    /// - Position component might be +8 bps (shift quotes up to sell)
    ///
    /// If order book has 60% bids, 40% asks (imbalance = 0.2):
    /// - Book component might be +2 bps (shift quotes up, market wants to buy)
    ///
    /// Total skew = +10 bps
    /// ```
    fn calculate_skew(
        &self,
        current_position: f64,
        max_position: f64,
        book_analysis: Option<&BookAnalysis>,
        base_half_spread_bps: f64,
    ) -> SkewResult;
}
