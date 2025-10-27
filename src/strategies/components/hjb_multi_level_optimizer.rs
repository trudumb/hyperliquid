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

use serde_json::Value;

use crate::strategy::CurrentState;
use crate::{
    HawkesFillModel, MultiLevelConfig, MultiLevelOptimizer, OptimizationState,
    ParameterUncertainty, RobustConfig, RobustParameters, TickLotValidator, EPSILON,
};
use crate::market_maker_v2::ConstrainedTuningParams;
use super::quote_optimizer::{QuoteOptimizer, OptimizerInputs};

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

    /// Robust control configuration
    robust_config: RobustConfig,

    /// Tick/lot size validator for price/size rounding
    tick_lot_validator: TickLotValidator,

    /// Maximum absolute position size
    max_position_size: f64,
    
    /// Tuning parameters (constrained, i.e., theta space)
    tuning_params: ConstrainedTuningParams,
}

impl HjbMultiLevelOptimizer {
    /// Create a new HJB multi-level optimizer with default parameters.
    ///
    /// Default configuration:
    /// - Standard multi-level config (3 levels, 10 bps spacing)
    /// - Standard robust config (robustness level 0.5)
    /// - Asset: "HYPE"
    /// - Max position: 50.0
    /// - Default tuning parameters
    pub fn new_default() -> Self {
        use crate::market_maker_v2::TuningParams;
        Self::new(
            MultiLevelConfig::default(),
            RobustConfig::default(),
            "HYPE".to_string(),
            50.0,
            TuningParams::default().get_constrained(),
        )
    }

    /// Create a new HJB multi-level optimizer with custom parameters.
    ///
    /// # Arguments
    /// - `multi_level_config`: Configuration for multi-level optimization
    /// - `robust_config`: Configuration for robust control
    /// - `asset`: Asset symbol (for tick/lot validation)
    /// - `max_position_size`: Maximum absolute position size
    /// - `tuning_params`: Constrained tuning parameters (theta space)
    pub fn new(
        multi_level_config: MultiLevelConfig,
        robust_config: RobustConfig,
        asset: String,
        max_position_size: f64,
        tuning_params: ConstrainedTuningParams,
    ) -> Self {
        let multi_level_optimizer = MultiLevelOptimizer::new(multi_level_config);

        // Initialize tick/lot validator (default to 3 sz_decimals for most assets)
        let tick_lot_validator = TickLotValidator::new(
            asset.clone(),
            crate::AssetType::Perp,
            3,
        );

        Self {
            multi_level_optimizer,
            robust_config,
            tick_lot_validator,
            max_position_size,
            tuning_params,
        }
    }
    
    /// Update tuning parameters (for online tuning)
    pub fn set_tuning_params(&mut self, tuning_params: ConstrainedTuningParams) {
        self.tuning_params = tuning_params;
    }
    
    /// Get current tuning parameters
    pub fn get_tuning_params(&self) -> &ConstrainedTuningParams {
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
    ///   "tuning_params": { ... }
    /// }
    /// ```
    pub fn from_json(config: &Value) -> Self {
        use crate::market_maker_v2::TuningParams;
        
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
        
        let tuning_params = config.get("tuning_params")
            .and_then(|v| serde_json::from_value(v.clone()).ok())
            .map(|tp: TuningParams| tp.get_constrained())
            .unwrap_or_else(|| TuningParams::default().get_constrained());

        Self::new(multi_level_config, robust_config, asset, max_position_size, tuning_params)
    }
}

impl QuoteOptimizer for HjbMultiLevelOptimizer {
    fn calculate_target_quotes(
        &self,
        inputs: &OptimizerInputs,
        state: &CurrentState,
        fill_model: &HawkesFillModel,
    ) -> (Vec<(f64, f64)>, Vec<(f64, f64)>) {
        let output = self.calculate_target_quotes_with_metadata(inputs, state, fill_model);
        (output.target_bids, output.target_asks)
    }
}

impl HjbMultiLevelOptimizer {
    /// Calculate target quotes with additional metadata (taker rates, liquidation flag).
    ///
    /// This extended method returns all the information from the multi-level optimizer,
    /// not just the quotes. Use this when you need taker rates or liquidation signals.
    pub fn calculate_target_quotes_with_metadata(
        &self,
        inputs: &OptimizerInputs,
        state: &CurrentState,
        fill_model: &HawkesFillModel,
    ) -> OptimizerOutput {
        // 1. Compute parameter uncertainty from volatility uncertainty
        let current_uncertainty = ParameterUncertainty::from_particle_filter_stats(
            0.0,  // mu_std (not used currently)
            inputs.vol_uncertainty_bps,
            0.95,  // confidence level
        );

        // 2. Compute robust parameters (worst-case volatility and adverse selection)
        let robust_params = RobustParameters::compute(
            inputs.adverse_selection_bps,
            inputs.volatility_bps,
            state.position,
            &current_uncertainty,
            &self.robust_config,
        );

        // 3. Prepare optimization state
        let opt_state = OptimizationState {
            mid_price: state.l2_mid_price,
            inventory: state.position,
            max_position: self.max_position_size,
            adverse_selection_bps: robust_params.mu_worst_case,
            lob_imbalance: inputs.lob_imbalance,
            volatility_bps: robust_params.sigma_worst_case,
            current_time: inputs.current_time_sec,
            hawkes_model: fill_model,
        };

        // 4. Calculate robust base half-spread
        let min_profitable_half_spread = self.multi_level_optimizer.config().min_profitable_spread_bps / 2.0;
        let robust_base_half_spread = (robust_params.sigma_worst_case * 0.1)
            .max(min_profitable_half_spread)
            * robust_params.spread_multiplier;

        // 5. Run multi-level optimization
        // Use the stored tuning params instead of defaults
        let multi_level_control = self.multi_level_optimizer.optimize(
            &opt_state,
            robust_base_half_spread,
            &self.tuning_params,
        );

        // 6. Convert offsets to prices
        let mut target_bids = Vec::new();
        let mut target_asks = Vec::new();

        for (offset_bps, size_raw) in multi_level_control.bid_levels {
            if size_raw < EPSILON {
                continue;
            }

            let size = self.tick_lot_validator.round_size(size_raw, false);
            if size < EPSILON {
                continue;
            }

            let price_raw = state.l2_mid_price * (1.0 - offset_bps / 10000.0);
            let price = self.tick_lot_validator.round_price(price_raw, false);

            // Ensure minimum notional value ($10)
            if price > 0.0 && (size * price) >= 10.0 {
                target_bids.push((price, size));
            }
        }

        for (offset_bps, size_raw) in multi_level_control.ask_levels {
            if size_raw < EPSILON {
                continue;
            }

            let size = self.tick_lot_validator.round_size(size_raw, false);
            if size < EPSILON {
                continue;
            }

            let price_raw = state.l2_mid_price * (1.0 + offset_bps / 10000.0);
            let price = self.tick_lot_validator.round_price(price_raw, true);

            // Ensure minimum notional value ($10)
            if price > 0.0 && (size * price) >= 10.0 {
                target_asks.push((price, size));
            }
        }

        // 7. Sort bids descending, asks ascending
        target_bids.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        target_asks.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        OptimizerOutput {
            target_bids,
            target_asks,
            taker_buy_rate: multi_level_control.taker_buy_rate,
            taker_sell_rate: multi_level_control.taker_sell_rate,
            liquidate: multi_level_control.liquidate,
        }
    }
}
