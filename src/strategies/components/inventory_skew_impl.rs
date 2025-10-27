// ============================================================================
// Standard Inventory Skew Implementation
// ============================================================================
//
// This component implements the InventorySkewModel trait using a combination
// of position-based skewing and order book imbalance analysis.
//
// # Algorithm
//
// The skew is calculated as:
//
// 1. **Position Component**:
//    ```
//    position_ratio = current_position / max_position
//    position_component = position_ratio * inventory_skew_factor * base_spread
//    ```
//    - When long (+), this is positive → shifts quotes UP to encourage selling
//    - When short (-), this is negative → shifts quotes DOWN to encourage buying
//
// 2. **Book Imbalance Component**:
//    ```
//    imbalance = (bid_depth - ask_depth) / (bid_depth + ask_depth)
//    book_component = imbalance * book_imbalance_factor * base_spread
//    ```
//    - Positive imbalance (more bids) → shifts quotes UP
//    - Negative imbalance (more asks) → shifts quotes DOWN
//
// 3. **Total Skew**:
//    ```
//    total_skew = position_component + book_component
//    ```
//
// # Configuration
//
// - `inventory_skew_factor`: Controls position-based skewing (0.0 to 1.0)
// - `book_imbalance_factor`: Controls book-based skewing (0.0 to 1.0)
// - `depth_analysis_levels`: Number of LOB levels to analyze
//
// # Example
//
// ```rust
// use strategies::components::{InventorySkewModel, StandardInventorySkew};
//
// let skew_model = StandardInventorySkew::new_default();
//
// let result = skew_model.calculate_skew(
//     40.0,  // current position (long 40)
//     50.0,  // max position
//     Some(&book_analysis),
//     10.0,  // base half spread (10 bps)
// );
//
// // Result: skew_bps might be +8 bps (shift quotes up to sell)
// ```

use crate::book_analyzer::BookAnalysis;
use super::inventory_skew::{InventorySkewModel, SkewResult};

/// Configuration for standard inventory skew component
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct InventorySkewConfig {
    /// How aggressively to manage inventory (0.0 to 1.0)
    /// Higher values mean more aggressive position management
    pub inventory_skew_factor: f64,

    /// How much to react to order book imbalance (0.0 to 1.0)
    /// Higher values mean more reaction to book conditions
    pub book_imbalance_factor: f64,

    /// Number of book levels to analyze for depth
    pub depth_analysis_levels: usize,
}

impl Default for InventorySkewConfig {
    fn default() -> Self {
        Self {
            inventory_skew_factor: 0.5,
            book_imbalance_factor: 0.3,
            depth_analysis_levels: 5,
        }
    }
}

impl InventorySkewConfig {
    /// Create a new config with validation
    pub fn new(
        inventory_skew_factor: f64,
        book_imbalance_factor: f64,
        depth_analysis_levels: usize,
    ) -> Result<Self, String> {
        if !(0.0..=1.0).contains(&inventory_skew_factor) {
            return Err("inventory_skew_factor must be between 0.0 and 1.0".to_string());
        }
        if !(0.0..=1.0).contains(&book_imbalance_factor) {
            return Err("book_imbalance_factor must be between 0.0 and 1.0".to_string());
        }
        if depth_analysis_levels == 0 {
            return Err("depth_analysis_levels must be greater than 0".to_string());
        }

        Ok(Self {
            inventory_skew_factor,
            book_imbalance_factor,
            depth_analysis_levels,
        })
    }
}

/// Standard inventory skew implementation.
///
/// This component uses a combination of position ratio and order book
/// imbalance to calculate optimal price skews.
#[derive(Debug)]
pub struct StandardInventorySkew {
    pub config: InventorySkewConfig,
}

impl StandardInventorySkew {
    /// Create a new standard inventory skew component with default config.
    pub fn new_default() -> Self {
        Self {
            config: InventorySkewConfig::default(),
        }
    }

    /// Create a new standard inventory skew component with custom config.
    pub fn new(config: InventorySkewConfig) -> Self {
        Self { config }
    }
}

impl InventorySkewModel for StandardInventorySkew {
    fn calculate_skew(
        &self,
        current_position: f64,
        max_position: f64,
        book_analysis: Option<&BookAnalysis>,
        base_half_spread_bps: f64,
    ) -> SkewResult {
        // Validate inputs
        if max_position <= 0.0 {
            return SkewResult {
                skew_bps: 0.0,
                position_component_bps: 0.0,
                book_component_bps: 0.0,
                position_ratio: 0.0,
            };
        }

        // Calculate position ratio (-1.0 to 1.0)
        let position_ratio = (current_position / max_position).clamp(-1.0, 1.0);

        // Position component: shifts quotes to encourage position reduction
        // Positive position (long) → positive skew (shift quotes up to sell)
        // Negative position (short) → negative skew (shift quotes down to buy)
        let position_component_bps =
            position_ratio * self.config.inventory_skew_factor * base_half_spread_bps;

        // Book imbalance component: shifts quotes based on LOB pressure
        let book_component_bps = if let Some(analysis) = book_analysis {
            // Use pre-computed imbalance from BookAnalysis
            // imbalance: -1.0 (all asks) to +1.0 (all bids)
            // Positive imbalance (more bids) → shift quotes UP (market wants to buy)
            // Negative imbalance (more asks) → shift quotes DOWN (market wants to sell)
            analysis.imbalance * self.config.book_imbalance_factor * base_half_spread_bps
        } else {
            0.0
        };

        // Combine components
        let skew_bps = position_component_bps + book_component_bps;

        SkewResult {
            skew_bps,
            position_component_bps,
            book_component_bps,
            position_ratio,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_position_skew_when_long() {
        let skew = StandardInventorySkew::new_default();

        // Long 40 out of 50 max
        let result = skew.calculate_skew(40.0, 50.0, None, 10.0);

        // position_ratio = 0.8
        // position_component = 0.8 * 0.5 * 10.0 = 4.0 bps
        assert!((result.position_component_bps - 4.0).abs() < 0.01);
        assert!(result.skew_bps > 0.0); // Should shift quotes up to encourage selling
    }

    #[test]
    fn test_position_skew_when_short() {
        let skew = StandardInventorySkew::new_default();

        // Short -30 out of 50 max
        let result = skew.calculate_skew(-30.0, 50.0, None, 10.0);

        // position_ratio = -0.6
        // position_component = -0.6 * 0.5 * 10.0 = -3.0 bps
        assert!((result.position_component_bps + 3.0).abs() < 0.01);
        assert!(result.skew_bps < 0.0); // Should shift quotes down to encourage buying
    }

    #[test]
    fn test_neutral_position() {
        let skew = StandardInventorySkew::new_default();

        let result = skew.calculate_skew(0.0, 50.0, None, 10.0);

        assert!((result.position_component_bps).abs() < 0.01);
        assert!((result.skew_bps).abs() < 0.01);
    }

    #[test]
    fn test_config_validation() {
        // Valid config
        assert!(InventorySkewConfig::new(0.5, 0.3, 5).is_ok());

        // Invalid inventory_skew_factor
        assert!(InventorySkewConfig::new(1.5, 0.3, 5).is_err());
        assert!(InventorySkewConfig::new(-0.1, 0.3, 5).is_err());

        // Invalid book_imbalance_factor
        assert!(InventorySkewConfig::new(0.5, 1.5, 5).is_err());

        // Invalid depth_analysis_levels
        assert!(InventorySkewConfig::new(0.5, 0.3, 0).is_err());
    }
}
