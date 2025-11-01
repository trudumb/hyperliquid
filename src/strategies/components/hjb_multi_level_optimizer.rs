// ============================================================================
// HJB Multi-Level Optimizer Component - Optimal Quote Calculation
// ============================================================================
//
// This component implements the QuoteOptimizer trait using the HJB
// (Hamilton-Jacobi-Bellman) optimal control framework with:
// - Multi-level order book optimization
// - Hawkes process fill rate modeling
// - Robust control under parameter uncertainty
//
// # Algorithm
//
// The optimizer solves for optimal quotes by:
// 1. Computing robust parameters (worst-case volatility, adverse selection)
// 2. Creating an OptimizationState with current market conditions
// 3. Calling the MultiLevelOptimizer to distribute liquidity across levels
// 4. Converting offset BPS to absolute prices and sizes
//
// # Robust Control
//
// The robust control framework handles parameter uncertainty by:
// - Expanding the uncertainty set around point estimates
// - Solving for worst-case parameters (max volatility, max adverse selection)
// - Applying a spread multiplier to account for uncertainty
//
// This makes the strategy more conservative when model confidence is low.
//
// # Multi-Level Optimization
//
// The optimizer distributes liquidity across L1, L2, L3 levels by:
// - Balancing fill probability (higher at L1) vs. adverse selection cost
// - Using Hawkes fill rates to estimate execution probabilities
// - Optimizing total expected utility across all levels
//
// # Example
//
// ```rust
// use strategies::components::{QuoteOptimizer, HjbMultiLevelOptimizer, OptimizerInputs};
//
// let optimizer = HjbMultiLevelOptimizer::new_default();
//
// let inputs = OptimizerInputs {
//     current_time_sec: 1234567890.0,
//     volatility_bps: 100.0,
//     vol_uncertainty_bps: 10.0,
//     adverse_selection_bps: 5.0,
//     lob_imbalance: 0.55,
// };
//
// let (bids, asks) = optimizer.calculate_target_quotes(&inputs, &state, &fill_model);
// ```

use log::{debug, warn};
use serde_json::Value;

use crate::strategy::CurrentState;
use crate::{
    HawkesFillModel, MultiLevelConfig, MultiLevelOptimizer,
    OptimizationState, TickLotValidator, EPSILON,
};
use super::quote_optimizer::{QuoteOptimizer, OptimizerInputs};
use super::robust_control::RobustControlModel;
use super::robust_control_impl::{StandardRobustControl, RobustConfig};
use super::inventory_skew::{InventorySkewModel};
use super::inventory_skew_impl::{StandardInventorySkew, InventorySkewConfig};
use super::parameter_transforms::StrategyConstrainedParams;
use crate::ConstrainedTuningParams;

/// Extended optimizer output with metadata.
///
/// In addition to the quotes, this includes:
/// - Taker rates for urgent inventory management
/// - Liquidation flag indicating extreme risk mode
#[derive(Debug, Clone)]
pub struct OptimizerOutput {
    /// Target bid quotes: (price, size)
    pub target_bids: Vec<(f64, f64)>,
    
    /// Target ask quotes: (price, size)
    pub target_asks: Vec<(f64, f64)>,
    
    /// Taker buy rate (for reducing long inventory)
    pub taker_buy_rate: f64,
    
    /// Taker sell rate (for reducing short inventory)
    pub taker_sell_rate: f64,
    
    /// Liquidation mode flag
    pub liquidate: bool,
}

/// HJB-based multi-level quote optimizer implementation.
///
/// This component uses the HJB optimal control framework with robust control
/// and multi-level optimization to calculate optimal quotes.
pub struct HjbMultiLevelOptimizer {
    /// Multi-level optimizer (contains Hawkes model, level logic)
    multi_level_optimizer: MultiLevelOptimizer,

    /// Robust control component
    robust_control: StandardRobustControl,

    /// Inventory skew component
    inventory_skew: StandardInventorySkew,

    /// Tick/lot size validator for price/size rounding
    tick_lot_validator: TickLotValidator,

    /// Maximum absolute position size
    max_position_size: f64,

    /// Tuning parameters (constrained, i.e., theta space)
    tuning_params: StrategyConstrainedParams,
}

impl HjbMultiLevelOptimizer {
    /// Convert new StrategyConstrainedParams to legacy ConstrainedTuningParams.
    ///
    /// This is a temporary compatibility layer while we migrate from the old
    /// parameter system to the new one. The old system has fewer parameters
    /// and different semantics, so we use reasonable defaults for unmapped fields.
    pub fn to_legacy_params(_params: &StrategyConstrainedParams) -> ConstrainedTuningParams {
        // Use default legacy parameters since the new param structure
        // doesn't have direct equivalents for all old fields
        ConstrainedTuningParams {
            skew_adjustment_factor: 0.5,
            adverse_selection_adjustment_factor: 0.5,
            adverse_selection_lambda: 1.0,
            inventory_urgency_threshold: 0.85,
            liquidation_rate_multiplier: 0.5,
            min_spread_base_ratio: 0.7,
            adverse_selection_spread_scale: 0.5,
            control_gap_threshold: 0.3,
        }
    }

    /// Create a new HJB multi-level optimizer with default parameters.
    ///
    /// Default configuration:
    /// - Standard multi-level config (3 levels, 10 bps spacing)
    /// - Standard robust config (robustness level 0.5)
    /// - Standard inventory skew config
    /// - Asset: "HYPE"
    /// - Max position: 50.0
    /// - Default tuning parameters
    pub fn new_default() -> Self {
        use super::parameter_transforms::StrategyTuningParams;
        Self::new(
            MultiLevelConfig::default(),
            RobustConfig::default(),
            InventorySkewConfig::default(),
            "HYPE".to_string(),
            50.0,
            StrategyTuningParams::default().get_constrained(),
        )
    }

    /// Create a new HJB multi-level optimizer with custom parameters.
    ///
    /// # Arguments
    /// - `multi_level_config`: Configuration for multi-level optimization
    /// - `robust_config`: Configuration for robust control
    /// - `inventory_skew_config`: Configuration for inventory skew
    /// - `asset`: Asset symbol (for tick/lot validation)
    /// - `max_position_size`: Maximum absolute position size
    /// - `tuning_params`: Constrained tuning parameters (theta space)
    pub fn new(
        multi_level_config: MultiLevelConfig,
        robust_config: RobustConfig,
        inventory_skew_config: InventorySkewConfig,
        asset: String,
        max_position_size: f64,
        tuning_params: StrategyConstrainedParams,
    ) -> Self {
        let multi_level_optimizer = MultiLevelOptimizer::new(multi_level_config);
        let robust_control = StandardRobustControl::new(robust_config);
        let inventory_skew = StandardInventorySkew::new(inventory_skew_config);

        // Initialize tick/lot validator (default to 3 sz_decimals for most assets)
        let tick_lot_validator = TickLotValidator::new(
            asset.clone(),
            crate::AssetType::Perp,
            3,
        );

        Self {
            multi_level_optimizer,
            robust_control,
            inventory_skew,
            tick_lot_validator,
            max_position_size,
            tuning_params,
        }
    }
    
    /// Update tuning parameters (for online tuning)
    pub fn set_tuning_params(&mut self, tuning_params: StrategyConstrainedParams) {
        self.tuning_params = tuning_params;
    }

    /// Get current tuning parameters
    pub fn get_tuning_params(&self) -> &StrategyConstrainedParams {
        &self.tuning_params
    }

    /// Create a new HJB multi-level optimizer from JSON config.
    ///
    /// Expected JSON structure:
    /// ```json
    /// {
    ///   "asset": "HYPE",
    ///   "max_position_size": 50.0,
    ///   "multi_level_config": { ... },
    ///   "robust_config": { ... },
    ///   "inventory_skew_config": { ... },
    ///   "tuning_params": { ... }
    /// }
    /// ```
    pub fn from_json(config: &Value) -> Self {
        use super::parameter_transforms::StrategyTuningParams;

        let asset = config.get("asset")
            .and_then(|v| v.as_str())
            .unwrap_or("HYPE")
            .to_string();

        let max_position_size = config.get("max_position_size")
            .and_then(|v| v.as_f64())
            .unwrap_or(50.0);

        let multi_level_config = config.get("multi_level_config")
            .and_then(|v| serde_json::from_value(v.clone()).ok())
            .unwrap_or_else(|| MultiLevelConfig::default());

        let robust_config = config.get("robust_config")
            .and_then(|v| serde_json::from_value(v.clone()).ok())
            .unwrap_or_else(|| RobustConfig::default());

        let inventory_skew_config = config.get("inventory_skew_config")
            .and_then(|v| serde_json::from_value(v.clone()).ok())
            .unwrap_or_else(|| InventorySkewConfig::default());

        let tuning_params = config.get("tuning_params")
            .and_then(|v| serde_json::from_value(v.clone()).ok())
            .map(|tp: StrategyTuningParams| tp.get_constrained())
            .unwrap_or_else(|| StrategyTuningParams::default().get_constrained());

        Self::new(multi_level_config, robust_config, inventory_skew_config, asset, max_position_size, tuning_params)
    }
}

impl QuoteOptimizer for HjbMultiLevelOptimizer {
    fn calculate_target_quotes(
        &self,
        inputs: &OptimizerInputs,
        state: &CurrentState,
        fill_model: &HawkesFillModel,
    ) -> (Vec<(f64, f64)>, Vec<(f64, f64)>) {
        // Use a default maker fee of 1.5 bps for the trait method
        let output = self.calculate_target_quotes_with_metadata(inputs, state, fill_model, 1.5);
        (output.target_bids, output.target_asks)
    }

    fn apply_tuning_params(&mut self, params: &super::StrategyConstrainedParams) {
        use log::info;

        info!("[HJB Multi-Level Optimizer] Applying new tuning parameters");
        info!("  Core params - phi: {:.4}, lambda: {:.2}, max_pos: {:.1}",
            params.phi, params.lambda_base, params.max_absolute_position_size);
        info!("  Multi-level - num_levels: {}, spacing: {:.1} bps, min_spread: {:.1} bps",
            params.num_levels, params.level_spacing_bps, params.min_profitable_spread_bps);
        info!("  Fees - maker: {:.1} bps, taker: {:.1} bps",
            params.maker_fee_bps, params.taker_fee_bps);

        // Update stored tuning parameters
        self.tuning_params = params.clone();

        // Update max position size
        self.max_position_size = params.max_absolute_position_size;

        // Update PositionManager with new max position
        self.multi_level_optimizer.update_max_position(params.max_absolute_position_size);

        // Update multi-level config parameters
        let ml_config = self.multi_level_optimizer.config_mut();
        ml_config.max_levels = params.num_levels;
        ml_config.level_spacing_bps = params.level_spacing_bps;
        ml_config.min_profitable_spread_bps = params.min_profitable_spread_bps;
        ml_config.volatility_to_spread_factor = params.volatility_to_spread_factor;

        // Note: The following parameters from StrategyConstrainedParams are not directly
        // mapped to MultiLevelConfig fields and would require changes to the optimizer logic:
        // - base_maker_size, maker_aggression_decay, taker_size_multiplier, min_taker_rate_threshold
        // These could be added to MultiLevelConfig in the future if needed for auto-tuning.

        // Update robust control enabled flag
        // Note: The detailed robust control parameters (robustness_level, min_epsilon_mu, etc.)
        // are not part of the auto-tunable parameters and remain fixed from the config
        let robust_config = self.robust_control.config_mut();
        robust_config.enabled = params.enable_robust_control;

        // Note: Inventory skew parameters are not part of StrategyConstrainedParams yet.
        // They could be added in the future if needed for auto-tuning.

        info!("[HJB Multi-Level Optimizer] Parameters successfully applied");
    }
}

impl HjbMultiLevelOptimizer {
    /// Calculate target quotes with robust spread-crossing prevention.
    ///
    /// This extended method returns all the information from the multi-level optimizer,
    /// not just the quotes. Use this when you need taker rates or liquidation signals.
    ///
    /// # Robustness Features
    /// - Invalid mid-price handling (returns empty quotes if mid <= 0)
    /// - BBO cross-spread prevention with tolerance
    /// - Minimum size step validation
    /// - Price validity checks
    /// - Minimum notional value enforcement
    /// - Comprehensive logging for debugging
    pub fn calculate_target_quotes_with_metadata(
        &self,
        inputs: &OptimizerInputs,
        state: &CurrentState,
        fill_model: &HawkesFillModel,
        maker_fee_bps: f64,
    ) -> OptimizerOutput {
        // --- Steps 1-5: Calculate Robust Params, Base Spread, Skew, Opt State ---

        // 1. Compute robust parameters using the robust control component
        let robust_params = self.robust_control.compute_robust_parameters(
            inputs.volatility_bps,
            inputs.vol_uncertainty_bps,
            inputs.adverse_selection_bps,
            0.0,  // as_uncertainty_bps (not used currently, could be added to OptimizerInputs)
        );

        // 2. Calculate base half-spread from volatility using configurable factor
        let min_profitable_half_spread = self.multi_level_optimizer.config().min_profitable_spread_bps / 2.0;
        let vol_to_spread_factor = self.multi_level_optimizer.config().volatility_to_spread_factor;
        let base_half_spread = (robust_params.sigma_worst_case * vol_to_spread_factor)
            .max(min_profitable_half_spread);

        debug!(
            "[SPREAD CALC] Vol: {:.2} bps, Vol Uncertainty: {:.2} bps, Worst-case Vol: {:.2} bps, Factor: {:.4}, Base Half-Spread: {:.2} bps",
            inputs.volatility_bps, inputs.vol_uncertainty_bps, robust_params.sigma_worst_case, vol_to_spread_factor, base_half_spread
        );

        // 3. Calculate inventory skew adjustment using the inventory skew component
        // Analyze the order book if available
        let book_analysis = state.order_book.as_ref()
            .and_then(|book| book.analyze(5)); // Analyze top 5 levels

        let skew_result = self.inventory_skew.calculate_skew(
            state.position,
            self.max_position_size,
            book_analysis.as_ref(),
            base_half_spread,
        );

        // 4. Apply spread multiplier from robust control
        let robust_base_half_spread = base_half_spread * robust_params.spread_multiplier;

        debug!(
            "[SPREAD CALC] Spread Multiplier: {:.4}, Robust Base Half-Spread: {:.2} bps (Full Spread: {:.2} bps)",
            robust_params.spread_multiplier, robust_base_half_spread, robust_base_half_spread * 2.0
        );

        // 5. Prepare optimization state
        let opt_state = OptimizationState {
            mid_price: state.l2_mid_price,
            inventory: state.position,
            max_position: self.max_position_size,
            adverse_selection_bps: robust_params.mu_worst_case,
            lob_imbalance: inputs.lob_imbalance,
            volatility_bps: robust_params.sigma_worst_case,
            current_time: inputs.current_time_sec,
            hawkes_model: fill_model,
            open_bids: &state.open_bids,
            open_asks: &state.open_asks,
        };

        // 6. Run multi-level optimization
        // Convert new params to legacy format for backward compatibility
        let legacy_params = Self::to_legacy_params(&self.tuning_params);
        let multi_level_control = self.multi_level_optimizer.optimize(
            &opt_state,
            robust_base_half_spread,
            &legacy_params,
            maker_fee_bps,
        );

        // --- Step 7: Convert offsets to prices, apply skew robustly, check crossing ---
        let mut target_bids = Vec::new();
        let mut target_asks = Vec::new();

        let true_mid = state.l2_mid_price;
        if true_mid <= 0.0 {
            warn!("Invalid true_mid price ({:.5}) during quote calculation. Skipping.", true_mid);
            // Return empty quotes immediately if mid-price is invalid
            return OptimizerOutput {
                target_bids,
                target_asks,
                taker_buy_rate: 0.0,
                taker_sell_rate: 0.0,
                liquidate: false,
            };
        }

        let best_bid_opt = state.order_book.as_ref().and_then(|b| b.best_bid());
        let best_ask_opt = state.order_book.as_ref().and_then(|b| b.best_ask());
        let min_price_increment = self.tick_lot_validator.min_price_step();
        let min_size_increment = self.tick_lot_validator.min_size_step();
        // Use a small fraction of the minimum price increment as tolerance
        let cross_tolerance = min_price_increment * 0.1;

        debug!(
            "Quote Calc: Mid={:.5}, Skew={:.2}bps, BBO=[{:?}, {:?}]",
            true_mid, skew_result.skew_bps, best_bid_opt, best_ask_opt
        );

        // --- Process Bid Levels ---
        for (level_index, (offset_bps, size_raw)) in multi_level_control.bid_levels.iter().enumerate() {
            if *size_raw < EPSILON {
                continue;
            }

            let size = self.tick_lot_validator.round_size(*size_raw, false); // Round size down
            if size < min_size_increment {
                debug!("Level {} bid size {:.8} rounded below min step {:.8}. Skipping.",
                       level_index + 1, *size_raw, min_size_increment);
                continue;
            }

            // **1. Calculate price from true mid and offset**
            let price_before_skew = true_mid * (1.0 - offset_bps / 10000.0);

            // **2. Apply skew multiplicatively**
            let price_with_skew = price_before_skew * (1.0 + skew_result.skew_bps / 10000.0);

            // **3. Round final price (bids DOWN)**
            let final_bid_price = self.tick_lot_validator.round_price(price_with_skew, false);

            // **4. Robust Cross-Spread Check**
            if let Some(best_ask) = best_ask_opt {
                // Check if the final bid is strictly GREATER than the best ask (allowing matching within tolerance)
                if final_bid_price > best_ask + cross_tolerance {
                    warn!(
                        "L{} Bid {:.5} would CROSS Ask {:.5}! Skipping. (Offset={:.2}bps, Skew={:.2}bps, Mid={:.5})",
                        level_index + 1, final_bid_price, best_ask, offset_bps, skew_result.skew_bps, true_mid
                    );
                    continue; // Skip this level
                }
            } else if final_bid_price > true_mid * (1.0 + min_profitable_half_spread / 10000.0) {
                // Fallback: If no ask BBO, prevent bids significantly above mid
                warn!("L{} Bid {:.5} > Mid {:.5} without Ask BBO. Skipping.",
                      level_index + 1, final_bid_price, true_mid);
                continue;
            }

            // Price validity and Notional check
            if final_bid_price <= 0.0 {
                warn!("L{} Bid price calculated <= 0 ({:.5}). Skipping.", level_index + 1, final_bid_price);
                continue;
            }
            if (size * final_bid_price) < 10.0 { // Minimum notional value (e.g., $10)
                debug!("L{} Bid notional {:.2} < $10. Skipping.", level_index + 1, size * final_bid_price);
                continue;
            }

            target_bids.push((final_bid_price, size));
        }

        // --- Process Ask Levels ---
        for (level_index, (offset_bps, size_raw)) in multi_level_control.ask_levels.iter().enumerate() {
            if *size_raw < EPSILON {
                continue;
            }

            let size = self.tick_lot_validator.round_size(*size_raw, false); // Round size down
            if size < min_size_increment {
                debug!("Level {} ask size {:.8} rounded below min step {:.8}. Skipping.",
                       level_index + 1, *size_raw, min_size_increment);
                continue;
            }

            // **1. Calculate price from true mid and offset**
            let price_before_skew = true_mid * (1.0 + offset_bps / 10000.0);

            // **2. Apply skew multiplicatively**
            let price_with_skew = price_before_skew * (1.0 + skew_result.skew_bps / 10000.0);

            // **3. Round final price (asks UP)**
            let final_ask_price = self.tick_lot_validator.round_price(price_with_skew, true);

            // **4. Robust Cross-Spread Check**
            if let Some(best_bid) = best_bid_opt {
                // Check if the final ask is strictly LESS than the best bid (allowing matching within tolerance)
                if final_ask_price < best_bid - cross_tolerance {
                    warn!(
                        "L{} Ask {:.5} would CROSS Bid {:.5}! Skipping. (Offset={:.2}bps, Skew={:.2}bps, Mid={:.5})",
                        level_index + 1, final_ask_price, best_bid, offset_bps, skew_result.skew_bps, true_mid
                    );
                    continue; // Skip this level
                }
            } else if final_ask_price < true_mid * (1.0 - min_profitable_half_spread / 10000.0) {
                // Fallback: If no bid BBO, prevent asks significantly below mid
                warn!("L{} Ask {:.5} < Mid {:.5} without Bid BBO. Skipping.",
                      level_index + 1, final_ask_price, true_mid);
                continue;
            }

            // Price validity and Notional check
            if final_ask_price <= 0.0 {
                warn!("L{} Ask price calculated <= 0 ({:.5}). Skipping.", level_index + 1, final_ask_price);
                continue;
            }
            if (size * final_ask_price) < 10.0 { // Minimum notional value (e.g., $10)
                debug!("L{} Ask notional {:.2} < $10. Skipping.", level_index + 1, size * final_ask_price);
                continue;
            }

            target_asks.push((final_ask_price, size));
        }

        // --- Step 8: Sort bids descending, asks ascending ---
        target_bids.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        target_asks.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        // --- Step 9: Return result ---
        debug!(
            "Final Quotes: Bids({}), Asks({}) | Taker Rates: Buy={:.4}, Sell={:.4}",
            target_bids.len(), target_asks.len(),
            multi_level_control.taker_buy_rate, multi_level_control.taker_sell_rate
        );

        OptimizerOutput {
            target_bids,
            target_asks,
            taker_buy_rate: multi_level_control.taker_buy_rate,
            taker_sell_rate: multi_level_control.taker_sell_rate,
            liquidate: multi_level_control.liquidate,
        }
    }
}
