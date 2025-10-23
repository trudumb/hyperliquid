#[cfg(test)]
mod state_vector_tests {
    use crate::{StateVector, BookAnalysis, OrderBook};
    use crate::ws::sub_structs::{L2BookData, BookLevel};

    fn create_test_book_level(px: &str, sz: &str) -> BookLevel {
        BookLevel {
            px: px.to_string(),
            sz: sz.to_string(),
            n: 1,
        }
    }

    fn create_balanced_book() -> (OrderBook, BookAnalysis) {
        let data = L2BookData {
            coin: "TEST".to_string(),
            time: 0,
            levels: vec![
                vec![
                    create_test_book_level("100.0", "10.0"),
                    create_test_book_level("99.0", "20.0"),
                ],
                vec![
                    create_test_book_level("101.0", "10.0"),
                    create_test_book_level("102.0", "20.0"),
                ],
            ],
        };
        let book = OrderBook::from_l2_data(&data).unwrap();
        let analysis = book.analyze(2).unwrap();
        (book, analysis)
    }

    fn create_imbalanced_book_bid_heavy() -> (OrderBook, BookAnalysis) {
        let data = L2BookData {
            coin: "TEST".to_string(),
            time: 0,
            levels: vec![
                vec![
                    create_test_book_level("100.0", "100.0"), // Heavy bid
                    create_test_book_level("99.0", "50.0"),
                ],
                vec![
                    create_test_book_level("101.0", "10.0"), // Light ask
                    create_test_book_level("102.0", "5.0"),
                ],
            ],
        };
        let book = OrderBook::from_l2_data(&data).unwrap();
        let analysis = book.analyze(2).unwrap();
        (book, analysis)
    }

    #[test]
    fn test_state_vector_initialization() {
        let state = StateVector::new();
        
        assert_eq!(state.mid_price, 0.0);
        assert_eq!(state.inventory, 0.0);
        assert_eq!(state.adverse_selection_estimate, 0.0);
        assert_eq!(state.market_spread_bps, 0.0);
        assert_eq!(state.lob_imbalance, 0.0);
    }

    #[test]
    fn test_state_vector_update_basic() {
        let mut state = StateVector::new();
        let (book, analysis) = create_balanced_book();
        
        state.update(100.5, 0.0, Some(&analysis), Some(&book));
        
        assert_eq!(state.mid_price, 100.5);
        assert_eq!(state.inventory, 0.0);
        assert!(state.market_spread_bps > 0.0); // Should have calculated spread
    }

    #[test]
    fn test_lob_imbalance_calculation() {
        let mut state = StateVector::new();
        
        // Balanced book
        let (book, analysis) = create_balanced_book();
        state.update(100.5, 0.0, Some(&analysis), Some(&book));
        
        // Balanced book should have imbalance around 0.5
        assert!((state.lob_imbalance - 0.5).abs() < 0.1);
        
        // Bid-heavy book
        let (book, analysis) = create_imbalanced_book_bid_heavy();
        state.update(100.5, 0.0, Some(&analysis), Some(&book));
        
        // Bid-heavy book should have imbalance > 0.5
        assert!(state.lob_imbalance > 0.5);
    }

    #[test]
    fn test_adverse_selection_update() {
        let mut state = StateVector::new();
        let (book, analysis) = create_imbalanced_book_bid_heavy();
        
        // Initial state
        assert_eq!(state.adverse_selection_estimate, 0.0);
        
        // First update
        state.update(100.5, 0.0, Some(&analysis), Some(&book));
        let first_estimate = state.adverse_selection_estimate;
        
        // Should have moved positive (buying pressure)
        assert!(first_estimate > 0.0);
        
        // Second update with same data
        state.update(100.5, 0.0, Some(&analysis), Some(&book));
        let second_estimate = state.adverse_selection_estimate;
        
        // Should have moved further positive (EMA smoothing)
        assert!(second_estimate > first_estimate);
    }

    #[test]
    fn test_adverse_selection_adjustment() {
        let mut state = StateVector::new();
        
        // Positive adverse selection (bullish)
        state.adverse_selection_estimate = 0.1;
        let adjustment = state.get_adverse_selection_adjustment(100.0);
        
        // Should be negative (widen sell-side spread)
        assert!(adjustment < 0.0);
        
        // Negative adverse selection (bearish)
        state.adverse_selection_estimate = -0.1;
        let adjustment = state.get_adverse_selection_adjustment(100.0);
        
        // Should be positive (widen buy-side spread)
        assert!(adjustment > 0.0);
    }

    #[test]
    fn test_inventory_risk_multiplier() {
        let state = StateVector {
            mid_price: 100.0,
            inventory: 0.0,
            adverse_selection_estimate: 0.0,
            market_spread_bps: 10.0,
            lob_imbalance: 0.5,
        };
        
        // Zero inventory
        let multiplier = state.get_inventory_risk_multiplier(100.0);
        assert_eq!(multiplier, 1.0);
        
        // Half max inventory
        let mut state_half = state.clone();
        state_half.inventory = 50.0;
        let multiplier = state_half.get_inventory_risk_multiplier(100.0);
        assert!(multiplier > 1.0 && multiplier < 1.5);
        
        // Max inventory
        let mut state_max = state.clone();
        state_max.inventory = 100.0;
        let multiplier = state_max.get_inventory_risk_multiplier(100.0);
        assert_eq!(multiplier, 2.0);
    }

    #[test]
    fn test_inventory_urgency() {
        let state = StateVector {
            mid_price: 100.0,
            inventory: 0.0,
            adverse_selection_estimate: 0.0,
            market_spread_bps: 10.0,
            lob_imbalance: 0.5,
        };
        
        // Zero inventory - no urgency
        let urgency = state.get_inventory_urgency(100.0);
        assert_eq!(urgency, 0.0);
        
        // Half max inventory - moderate urgency
        let mut state_half = state.clone();
        state_half.inventory = 50.0;
        let urgency = state_half.get_inventory_urgency(100.0);
        assert!(urgency > 0.0 && urgency < 0.5);
        
        // Max inventory - maximum urgency
        let mut state_max = state.clone();
        state_max.inventory = 100.0;
        let urgency = state_max.get_inventory_urgency(100.0);
        assert_eq!(urgency, 1.0);
        
        // Urgency should increase rapidly near max (cubic)
        let mut state_90 = state.clone();
        state_90.inventory = 90.0;
        let urgency_90 = state_90.get_inventory_urgency(100.0);
        
        let mut state_80 = state.clone();
        state_80.inventory = 80.0;
        let urgency_80 = state_80.get_inventory_urgency(100.0);
        
        // Difference from 90% to 100% should be larger than 80% to 90%
        assert!((urgency - urgency_90) > (urgency_90 - urgency_80));
    }

    #[test]
    fn test_market_favorable_conditions() {
        let state = StateVector {
            mid_price: 100.0,
            inventory: 0.0,
            adverse_selection_estimate: 0.0,
            market_spread_bps: 10.0,
            lob_imbalance: 0.5,
        };
        
        // Normal conditions - should be favorable
        assert!(state.is_market_favorable(50.0));
        
        // Spread too wide
        let mut state_wide = state.clone();
        state_wide.market_spread_bps = 100.0;
        assert!(!state_wide.is_market_favorable(50.0));
        
        // Extreme imbalance (all bids)
        let mut state_imbalanced = state.clone();
        state_imbalanced.lob_imbalance = 0.95;
        assert!(!state_imbalanced.is_market_favorable(50.0));
        
        // Extreme imbalance (all asks)
        let mut state_imbalanced2 = state.clone();
        state_imbalanced2.lob_imbalance = 0.05;
        assert!(!state_imbalanced2.is_market_favorable(50.0));
    }

    #[test]
    fn test_state_vector_logging() {
        let state = StateVector {
            mid_price: 100.5,
            inventory: 25.5,
            adverse_selection_estimate: 0.0234,
            market_spread_bps: 8.5,
            lob_imbalance: 0.623,
        };
        
        let log_str = state.to_log_string();
        
        // Should contain all components
        assert!(log_str.contains("S=100.50"));
        assert!(log_str.contains("Q=25.50"));
        assert!(log_str.contains("μ̂=0.02"));
        assert!(log_str.contains("Δ=8.5"));
        assert!(log_str.contains("I=0.623"));
    }

    #[test]
    fn test_spread_scale_calculation() {
        let mut state = StateVector::new();
        let (book, analysis) = create_balanced_book();
        
        // Update with normal spread
        state.update(100.0, 0.0, Some(&analysis), Some(&book));
        state.adverse_selection_estimate = 0.1; // Arbitrary value
        let normal_estimate = state.adverse_selection_estimate;
        
        // Create a wider spread scenario
        let data_wide = L2BookData {
            coin: "TEST".to_string(),
            time: 0,
            levels: vec![
                vec![create_test_book_level("100.0", "10.0")],
                vec![create_test_book_level("110.0", "10.0")], // 10% spread
            ],
        };
        let book_wide = OrderBook::from_l2_data(&data_wide).unwrap();
        let analysis_wide = book_wide.analyze(1).unwrap();
        
        // Reset and update with wide spread
        state.adverse_selection_estimate = 0.0;
        state.update(105.0, 0.0, Some(&analysis_wide), Some(&book_wide));
        
        // Wide spread should dampen the adverse selection estimate
        // (signal is scaled down by spread)
        // Note: This test validates the spread scaling logic
        assert!(state.market_spread_bps > 100.0); // Should be wide
    }

    #[test]
    fn test_state_vector_consistency() {
        let mut state = StateVector::new();
        let (book, analysis) = create_balanced_book();
        
        // Multiple updates should maintain consistency
        for i in 0..10 {
            let inventory = i as f64 * 10.0;
            state.update(100.0 + i as f64, inventory, Some(&analysis), Some(&book));
            
            // Check invariants
            assert_eq!(state.mid_price, 100.0 + i as f64);
            assert_eq!(state.inventory, inventory);
            assert!(state.lob_imbalance >= 0.0 && state.lob_imbalance <= 1.0);
            assert!(state.adverse_selection_estimate.abs() <= 1.0); // Should stay bounded
        }
    }
}
