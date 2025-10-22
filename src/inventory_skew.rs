use crate::book_analyzer::BookAnalysis;
use log::info;

/// Configuration for inventory skewing
#[derive(Debug, Clone)]
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

/// Result of skew calculation
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

/// Calculator for inventory-based price skewing
#[derive(Debug)]
pub struct InventorySkewCalculator {
    pub config: InventorySkewConfig,
}

impl InventorySkewCalculator {
    pub fn new(config: InventorySkewConfig) -> Self {
        Self { config }
    }

    /// Calculate the price skew based on position and book imbalance
    /// 
    /// # Arguments
    /// * `current_position` - Current inventory position
    /// * `max_position` - Maximum allowed absolute position
    /// * `book_analysis` - Optional order book analysis
    /// * `base_half_spread_bps` - Base half spread in basis points
    /// 
    /// # Returns
    /// A SkewResult containing the total skew and its components
    /// 
    /// # Logic
    /// - Position Signal: (position / max_position) * inventory_skew_factor * base_spread
    ///   * When long, this is positive, pushing quotes UP
    ///   * When short, this is negative, pushing quotes DOWN
    ///   * This makes it easier to exit your position
    /// 
    /// - Book Signal: book_imbalance * book_imbalance_factor * base_spread
    ///   * Positive imbalance (more bids) -> shift quotes UP (make selling more attractive)
    ///   * Negative imbalance (more asks) -> shift quotes DOWN (make buying more attractive)
    pub fn calculate_skew(
        &self,
        current_position: f64,
        max_position: f64,
        book_analysis: Option<&BookAnalysis>,
        base_half_spread_bps: f64,
    ) -> SkewResult {
        // Calculate position ratio (-1.0 to 1.0)
        let position_ratio = if max_position > 0.0 {
            (current_position / max_position).clamp(-1.0, 1.0)
        } else {
            0.0
        };

        // Position component: push quotes away from where we want to exit
        // If we're long (+), we want to make selling easier, so shift quotes DOWN (negative)
        // If we're short (-), we want to make buying easier, so shift quotes UP (positive)
        // So we NEGATE the position ratio
        let position_component_bps = -position_ratio 
            * self.config.inventory_skew_factor 
            * base_half_spread_bps;

        // Book imbalance component
        let book_component_bps = if let Some(analysis) = book_analysis {
            // If there's more bid liquidity (+imbalance), shift quotes UP to encourage selling
            // If there's more ask liquidity (-imbalance), shift quotes DOWN to encourage buying
            analysis.imbalance 
                * self.config.book_imbalance_factor 
                * base_half_spread_bps
        } else {
            0.0
        };

        let total_skew_bps = position_component_bps + book_component_bps;

        SkewResult {
            skew_bps: total_skew_bps,
            position_component_bps,
            book_component_bps,
            position_ratio,
        }
    }

    /// Apply the skew to a price
    pub fn apply_skew_to_price(&self, base_price: f64, skew_bps: f64) -> f64 {
        base_price * (1.0 + skew_bps / 10000.0)
    }

    /// Log skew information
    pub fn log_skew(&self, skew: &SkewResult) {
        info!(
            "Inventory Skew - Total: {:.2} bps (Position: {:.2} bps [{:.1}%], Book: {:.2} bps)",
            skew.skew_bps,
            skew.position_component_bps,
            skew.position_ratio * 100.0,
            skew.book_component_bps
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::book_analyzer::BookAnalysis;

    #[test]
    fn test_position_skew_long() {
        let config = InventorySkewConfig::new(0.5, 0.0, 5).unwrap();
        let calculator = InventorySkewCalculator::new(config);

        // Long 3 HYPE out of max 5 HYPE = 60% position
        let skew = calculator.calculate_skew(3.0, 5.0, None, 10.0);

        // Position ratio: 0.6
        // Position component: -0.6 * 0.5 * 10.0 = -3.0 bps
        // (Negative because we want to shift down to encourage selling)
        assert!((skew.position_ratio - 0.6).abs() < 0.01);
        assert!((skew.position_component_bps - (-3.0)).abs() < 0.01);
        assert!((skew.skew_bps - (-3.0)).abs() < 0.01);
    }

    #[test]
    fn test_position_skew_short() {
        let config = InventorySkewConfig::new(0.5, 0.0, 5).unwrap();
        let calculator = InventorySkewCalculator::new(config);

        // Short 2 HYPE out of max 5 HYPE = -40% position
        let skew = calculator.calculate_skew(-2.0, 5.0, None, 10.0);

        // Position ratio: -0.4
        // Position component: -(-0.4) * 0.5 * 10.0 = +2.0 bps
        // (Positive because we want to shift up to encourage buying)
        assert!((skew.position_ratio - (-0.4)).abs() < 0.01);
        assert!((skew.position_component_bps - 2.0).abs() < 0.01);
        assert!((skew.skew_bps - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_book_imbalance_skew() {
        let config = InventorySkewConfig::new(0.0, 0.3, 5).unwrap();
        let calculator = InventorySkewCalculator::new(config);

        let book_analysis = BookAnalysis {
            bid_depth: 100.0,
            ask_depth: 50.0,
            weighted_bid_price: 100.0,
            weighted_ask_price: 101.0,
            imbalance: 0.333, // More bids
        };

        let skew = calculator.calculate_skew(0.0, 5.0, Some(&book_analysis), 10.0);

        // Book component: 0.333 * 0.3 * 10.0 ≈ 1.0 bps
        assert!((skew.book_component_bps - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_combined_skew() {
        let config = InventorySkewConfig::new(0.5, 0.3, 5).unwrap();
        let calculator = InventorySkewCalculator::new(config);

        let book_analysis = BookAnalysis {
            bid_depth: 100.0,
            ask_depth: 50.0,
            weighted_bid_price: 100.0,
            weighted_ask_price: 101.0,
            imbalance: 0.333,
        };

        // Long 3 HYPE with positive book imbalance
        let skew = calculator.calculate_skew(3.0, 5.0, Some(&book_analysis), 10.0);

        // Position: -3.0 bps (want to sell, shift down)
        // Book: +1.0 bps (more bids, shift up)
        // Total: -2.0 bps
        assert!((skew.skew_bps - (-2.0)).abs() < 0.1);
    }

    #[test]
    fn test_config_validation() {
        assert!(InventorySkewConfig::new(1.5, 0.3, 5).is_err());
        assert!(InventorySkewConfig::new(-0.1, 0.3, 5).is_err());
        assert!(InventorySkewConfig::new(0.5, 1.5, 5).is_err());
        assert!(InventorySkewConfig::new(0.5, 0.3, 0).is_err());
        assert!(InventorySkewConfig::new(0.5, 0.3, 5).is_ok());
    }

    #[test]
    fn test_apply_skew_to_price() {
        let config = InventorySkewConfig::default();
        let calculator = InventorySkewCalculator::new(config);

        // 100.0 price with 100 bps (1%) positive skew
        let skewed = calculator.apply_skew_to_price(100.0, 100.0);
        assert!((skewed - 101.0).abs() < 0.01);

        // 100.0 price with -50 bps (0.5%) negative skew
        let skewed = calculator.apply_skew_to_price(100.0, -50.0);
        assert!((skewed - 99.5).abs() < 0.01);
    }
}
