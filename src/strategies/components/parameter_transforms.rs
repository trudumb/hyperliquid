// ============================================================================
// Parameter Transformation System for Multi-Objective Auto-Tuning
// ============================================================================
//
// This module handles the transformation between unconstrained optimization
// space (φ) and constrained parameter space (θ) for all 36 tunable strategy
// parameters.
//
// # Transformation Types
//
// 1. **Sigmoid Transform** (bounded parameters):
//    θ = θ_min + (θ_max - θ_min) * σ(φ)
//    where σ(φ) = 1 / (1 + exp(-φ))
//
// 2. **Exponential Transform** (positive parameters):
//    θ = exp(φ)
//
// 3. **Direct Transform** (already bounded by construction):
//    θ = φ (for boolean/integer parameters after rounding)
//
// # Parameter Categories
//
// The 36 parameters are organized into 5 config sections:
// - Core HJB Strategy (10 params)
// - Multi-Level Config (8 params)
// - EWMA Volatility Model (6 params)
// - Particle Filter Config (4 params)
// - Hybrid Grounding Config (8 params)

use serde::{Deserialize, Serialize};

// ============================================================================
// Unconstrained Parameters (φ space) - Used by Optimizer
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyTuningParams {
    // === Core HJB Strategy (10 params) ===
    pub phi_phi: f64,                      // risk_aversion φ ∈ [0.001, 0.1]
    pub lambda_base_phi: f64,              // fill_intensity λ ∈ [0.1, 10.0]
    pub max_position_phi: f64,             // max_position ∈ [1.0, 10.0]
    pub maker_fee_phi: f64,                // maker_fee_bps ∈ [0.0, 5.0]
    pub taker_fee_phi: f64,                // taker_fee_bps ∈ [0.0, 10.0]
    pub leverage_phi: f64,                 // leverage ∈ [1, 10] (integer)
    pub max_leverage_phi: f64,             // max_leverage ∈ [10, 50] (integer)
    pub margin_safety_phi: f64,            // margin_safety_buffer ∈ [0.1, 0.5]
    pub enable_multi_level_phi: f64,       // boolean (use sigmoid > 0.5)
    pub enable_robust_control_phi: f64,    // boolean (use sigmoid > 0.5)

    // === Multi-Level Config (8 params) ===
    pub num_levels_phi: f64,               // num_levels ∈ [1, 5] (integer)
    pub level_spacing_phi: f64,            // level_spacing_bps ∈ [5.0, 50.0]
    pub min_spread_phi: f64,               // min_profitable_spread_bps ∈ [1.0, 10.0]
    pub vol_to_spread_factor_phi: f64,     // volatility_to_spread_factor ∈ [0.001, 0.02]
    pub base_maker_size_phi: f64,          // base_maker_size ∈ [0.1, 5.0]
    pub maker_aggression_decay_phi: f64,   // maker_aggression_decay ∈ [0.1, 0.9]
    pub taker_size_multiplier_phi: f64,    // taker_size_multiplier ∈ [0.1, 1.0]
    pub min_taker_rate_phi: f64,           // min_taker_rate_threshold ∈ [0.01, 0.5]

    // === EWMA Volatility Model (6 params) ===
    pub ewma_half_life_phi: f64,           // half_life_seconds ∈ [10.0, 600.0]
    pub ewma_alpha_phi: f64,               // alpha ∈ [0.01, 0.3]
    pub ewma_outlier_thresh_phi: f64,      // outlier_threshold ∈ [2.0, 8.0]
    pub ewma_min_vol_phi: f64,             // min_volatility_bps ∈ [0.1, 2.0]
    pub ewma_max_vol_phi: f64,             // max_volatility_bps ∈ [20.0, 100.0]
    pub ewma_tick_freq_phi: f64,           // expected_tick_frequency_hz ∈ [0.1, 100.0]

    // === Particle Filter Config (4 params) ===
    pub pf_mu_phi: f64,                    // pf_mu ∈ [-20.0, -10.0]
    pub pf_phi_phi: f64,                   // pf_phi ∈ [0.8, 0.99]
    pub pf_sigma_eta_phi: f64,             // pf_sigma_eta ∈ [0.1, 2.0]
    pub pf_update_interval_phi: f64,       // pf_update_interval_ticks ∈ [5, 50] (integer)

    // === Hybrid Grounding Config (8 params) ===
    pub grounding_base_phi: f64,           // grounding_strength_base ∈ [0.05, 0.5]
    pub grounding_sensitivity_phi: f64,    // grounding_sensitivity ∈ [0.1, 1.0]
    pub min_grounding_phi: f64,            // min_grounding ∈ [0.01, 0.1]
    pub max_grounding_phi: f64,            // max_grounding ∈ [0.3, 0.8]
    pub ba_tracking_window_phi: f64,       // bid_ask_tracking_window_size ∈ [10, 200] (integer)
    pub ba_ewma_alpha_phi: f64,            // bid_ask_ewma_alpha ∈ [0.01, 0.3]
    pub ba_vol_window_phi: f64,            // bid_ask_vol_window_size ∈ [5, 50] (integer)
    pub ba_rate_scale_phi: f64,            // bid_ask_rate_scale ∈ [0.1, 10.0]
}

// ============================================================================
// Constrained Parameters (θ space) - Actual Config Values
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyConstrainedParams {
    // === Core HJB Strategy ===
    pub phi: f64,
    pub lambda_base: f64,
    pub max_absolute_position_size: f64,
    pub maker_fee_bps: f64,
    pub taker_fee_bps: f64,
    pub leverage: i32,
    pub max_leverage: i32,
    pub margin_safety_buffer: f64,
    pub enable_multi_level: bool,
    pub enable_robust_control: bool,

    // === Multi-Level Config ===
    pub num_levels: usize,
    pub level_spacing_bps: f64,
    pub min_profitable_spread_bps: f64,
    pub volatility_to_spread_factor: f64,
    pub base_maker_size: f64,
    pub maker_aggression_decay: f64,
    pub taker_size_multiplier: f64,
    pub min_taker_rate_threshold: f64,

    // === EWMA Volatility Model ===
    pub ewma_half_life_seconds: f64,
    pub ewma_alpha: f64,
    pub ewma_outlier_threshold: f64,
    pub ewma_min_volatility_bps: f64,
    pub ewma_max_volatility_bps: f64,
    pub ewma_expected_tick_frequency_hz: f64,

    // === Particle Filter Config ===
    pub pf_mu: f64,
    pub pf_phi: f64,
    pub pf_sigma_eta: f64,
    pub pf_update_interval_ticks: usize,

    // === Hybrid Grounding Config ===
    pub grounding_strength_base: f64,
    pub grounding_sensitivity: f64,
    pub min_grounding: f64,
    pub max_grounding: f64,
    pub bid_ask_tracking_window_size: usize,
    pub bid_ask_ewma_alpha: f64,
    pub bid_ask_vol_window_size: usize,
    pub bid_ask_rate_scale: f64,
}

// ============================================================================
// Transformation Functions
// ============================================================================

impl StrategyTuningParams {
    /// Create default unconstrained parameters (centered in φ space)
    pub fn default() -> Self {
        Self {
            // Core HJB Strategy
            phi_phi: inverse_sigmoid(0.01, 0.001, 0.1),
            lambda_base_phi: inverse_sigmoid(1.0, 0.1, 10.0),
            max_position_phi: inverse_sigmoid(3.0, 1.0, 10.0),
            maker_fee_phi: inverse_sigmoid(1.5, 0.0, 5.0),
            taker_fee_phi: inverse_sigmoid(4.5, 0.0, 10.0),
            leverage_phi: inverse_sigmoid(3.0, 1.0, 10.0),
            max_leverage_phi: inverse_sigmoid(50.0, 10.0, 50.0),
            margin_safety_phi: inverse_sigmoid(0.2, 0.1, 0.5),
            enable_multi_level_phi: 2.0,  // sigmoid(2.0) ≈ 0.88 > 0.5 → true
            enable_robust_control_phi: 2.0,

            // Multi-Level Config
            num_levels_phi: inverse_sigmoid(3.0, 1.0, 5.0),
            level_spacing_phi: inverse_sigmoid(10.0, 5.0, 50.0),
            min_spread_phi: inverse_sigmoid(3.0, 1.0, 10.0),
            vol_to_spread_factor_phi: inverse_sigmoid(0.008, 0.001, 0.02),
            base_maker_size_phi: inverse_sigmoid(1.0, 0.1, 5.0),
            maker_aggression_decay_phi: inverse_sigmoid(0.5, 0.1, 0.9),
            taker_size_multiplier_phi: inverse_sigmoid(0.3, 0.1, 1.0),
            min_taker_rate_phi: inverse_sigmoid(0.1, 0.01, 0.5),

            // EWMA Volatility Model
            ewma_half_life_phi: inverse_sigmoid(60.0, 10.0, 600.0),
            ewma_alpha_phi: inverse_sigmoid(0.1, 0.01, 0.3),
            ewma_outlier_thresh_phi: inverse_sigmoid(4.0, 2.0, 8.0),
            ewma_min_vol_phi: inverse_sigmoid(0.5, 0.1, 2.0),
            ewma_max_vol_phi: inverse_sigmoid(50.0, 20.0, 100.0),
            ewma_tick_freq_phi: inverse_sigmoid(10.0, 0.1, 100.0),

            // Particle Filter Config
            pf_mu_phi: inverse_sigmoid(-18.0, -20.0, -10.0),
            pf_phi_phi: inverse_sigmoid(0.95, 0.8, 0.99),
            pf_sigma_eta_phi: inverse_sigmoid(0.5, 0.1, 2.0),
            pf_update_interval_phi: inverse_sigmoid(10.0, 5.0, 50.0),

            // Hybrid Grounding Config
            grounding_base_phi: inverse_sigmoid(0.2, 0.05, 0.5),
            grounding_sensitivity_phi: inverse_sigmoid(0.5, 0.1, 1.0),
            min_grounding_phi: inverse_sigmoid(0.05, 0.01, 0.1),
            max_grounding_phi: inverse_sigmoid(0.5, 0.3, 0.8),
            ba_tracking_window_phi: inverse_sigmoid(50.0, 10.0, 200.0),
            ba_ewma_alpha_phi: inverse_sigmoid(0.1, 0.01, 0.3),
            ba_vol_window_phi: inverse_sigmoid(20.0, 5.0, 50.0),
            ba_rate_scale_phi: inverse_sigmoid(1.0, 0.1, 10.0),
        }
    }

    /// Convert to 36-element vector for optimizer
    pub fn to_vec(&self) -> Vec<f64> {
        vec![
            // Core HJB Strategy (10)
            self.phi_phi,
            self.lambda_base_phi,
            self.max_position_phi,
            self.maker_fee_phi,
            self.taker_fee_phi,
            self.leverage_phi,
            self.max_leverage_phi,
            self.margin_safety_phi,
            self.enable_multi_level_phi,
            self.enable_robust_control_phi,
            // Multi-Level Config (8)
            self.num_levels_phi,
            self.level_spacing_phi,
            self.min_spread_phi,
            self.vol_to_spread_factor_phi,
            self.base_maker_size_phi,
            self.maker_aggression_decay_phi,
            self.taker_size_multiplier_phi,
            self.min_taker_rate_phi,
            // EWMA Volatility Model (6)
            self.ewma_half_life_phi,
            self.ewma_alpha_phi,
            self.ewma_outlier_thresh_phi,
            self.ewma_min_vol_phi,
            self.ewma_max_vol_phi,
            self.ewma_tick_freq_phi,
            // Particle Filter Config (4)
            self.pf_mu_phi,
            self.pf_phi_phi,
            self.pf_sigma_eta_phi,
            self.pf_update_interval_phi,
            // Hybrid Grounding Config (8)
            self.grounding_base_phi,
            self.grounding_sensitivity_phi,
            self.min_grounding_phi,
            self.max_grounding_phi,
            self.ba_tracking_window_phi,
            self.ba_ewma_alpha_phi,
            self.ba_vol_window_phi,
            self.ba_rate_scale_phi,
        ]
    }

    /// Create from 36-element vector
    pub fn from_vec(vec: &[f64]) -> Self {
        assert_eq!(vec.len(), 36, "Expected 36 parameters");
        Self {
            // Core HJB Strategy (10)
            phi_phi: vec[0],
            lambda_base_phi: vec[1],
            max_position_phi: vec[2],
            maker_fee_phi: vec[3],
            taker_fee_phi: vec[4],
            leverage_phi: vec[5],
            max_leverage_phi: vec[6],
            margin_safety_phi: vec[7],
            enable_multi_level_phi: vec[8],
            enable_robust_control_phi: vec[9],
            // Multi-Level Config (8)
            num_levels_phi: vec[10],
            level_spacing_phi: vec[11],
            min_spread_phi: vec[12],
            vol_to_spread_factor_phi: vec[13],
            base_maker_size_phi: vec[14],
            maker_aggression_decay_phi: vec[15],
            taker_size_multiplier_phi: vec[16],
            min_taker_rate_phi: vec[17],
            // EWMA Volatility Model (6)
            ewma_half_life_phi: vec[18],
            ewma_alpha_phi: vec[19],
            ewma_outlier_thresh_phi: vec[20],
            ewma_min_vol_phi: vec[21],
            ewma_max_vol_phi: vec[22],
            ewma_tick_freq_phi: vec[23],
            // Particle Filter Config (4)
            pf_mu_phi: vec[24],
            pf_phi_phi: vec[25],
            pf_sigma_eta_phi: vec[26],
            pf_update_interval_phi: vec[27],
            // Hybrid Grounding Config (8)
            grounding_base_phi: vec[28],
            grounding_sensitivity_phi: vec[29],
            min_grounding_phi: vec[30],
            max_grounding_phi: vec[31],
            ba_tracking_window_phi: vec[32],
            ba_ewma_alpha_phi: vec[33],
            ba_vol_window_phi: vec[34],
            ba_rate_scale_phi: vec[35],
        }
    }

    /// Transform to constrained parameter space
    pub fn get_constrained(&self) -> StrategyConstrainedParams {
        StrategyConstrainedParams {
            // Core HJB Strategy
            phi: apply_sigmoid(self.phi_phi, 0.001, 0.1),
            lambda_base: apply_sigmoid(self.lambda_base_phi, 0.1, 10.0),
            max_absolute_position_size: apply_sigmoid(self.max_position_phi, 1.0, 10.0),
            maker_fee_bps: apply_sigmoid(self.maker_fee_phi, 0.0, 5.0),
            taker_fee_bps: apply_sigmoid(self.taker_fee_phi, 0.0, 10.0),
            leverage: apply_sigmoid(self.leverage_phi, 1.0, 10.0).round() as i32,
            max_leverage: apply_sigmoid(self.max_leverage_phi, 10.0, 50.0).round() as i32,
            margin_safety_buffer: apply_sigmoid(self.margin_safety_phi, 0.1, 0.5),
            enable_multi_level: sigmoid(self.enable_multi_level_phi) > 0.5,
            enable_robust_control: sigmoid(self.enable_robust_control_phi) > 0.5,

            // Multi-Level Config
            num_levels: apply_sigmoid(self.num_levels_phi, 1.0, 5.0).round() as usize,
            level_spacing_bps: apply_sigmoid(self.level_spacing_phi, 5.0, 50.0),
            min_profitable_spread_bps: apply_sigmoid(self.min_spread_phi, 1.0, 10.0),
            volatility_to_spread_factor: apply_sigmoid(self.vol_to_spread_factor_phi, 0.001, 0.02),
            base_maker_size: apply_sigmoid(self.base_maker_size_phi, 0.1, 5.0),
            maker_aggression_decay: apply_sigmoid(self.maker_aggression_decay_phi, 0.1, 0.9),
            taker_size_multiplier: apply_sigmoid(self.taker_size_multiplier_phi, 0.1, 1.0),
            min_taker_rate_threshold: apply_sigmoid(self.min_taker_rate_phi, 0.01, 0.5),

            // EWMA Volatility Model
            ewma_half_life_seconds: apply_sigmoid(self.ewma_half_life_phi, 10.0, 600.0),
            ewma_alpha: apply_sigmoid(self.ewma_alpha_phi, 0.01, 0.3),
            ewma_outlier_threshold: apply_sigmoid(self.ewma_outlier_thresh_phi, 2.0, 8.0),
            ewma_min_volatility_bps: apply_sigmoid(self.ewma_min_vol_phi, 0.1, 2.0),
            ewma_max_volatility_bps: apply_sigmoid(self.ewma_max_vol_phi, 20.0, 100.0),
            ewma_expected_tick_frequency_hz: apply_sigmoid(self.ewma_tick_freq_phi, 0.1, 100.0),

            // Particle Filter Config
            pf_mu: apply_sigmoid(self.pf_mu_phi, -20.0, -10.0),
            pf_phi: apply_sigmoid(self.pf_phi_phi, 0.8, 0.99),
            pf_sigma_eta: apply_sigmoid(self.pf_sigma_eta_phi, 0.1, 2.0),
            pf_update_interval_ticks: apply_sigmoid(self.pf_update_interval_phi, 5.0, 50.0).round() as usize,

            // Hybrid Grounding Config
            grounding_strength_base: apply_sigmoid(self.grounding_base_phi, 0.05, 0.5),
            grounding_sensitivity: apply_sigmoid(self.grounding_sensitivity_phi, 0.1, 1.0),
            min_grounding: apply_sigmoid(self.min_grounding_phi, 0.01, 0.1),
            max_grounding: apply_sigmoid(self.max_grounding_phi, 0.3, 0.8),
            bid_ask_tracking_window_size: apply_sigmoid(self.ba_tracking_window_phi, 10.0, 200.0).round() as usize,
            bid_ask_ewma_alpha: apply_sigmoid(self.ba_ewma_alpha_phi, 0.01, 0.3),
            bid_ask_vol_window_size: apply_sigmoid(self.ba_vol_window_phi, 5.0, 50.0).round() as usize,
            bid_ask_rate_scale: apply_sigmoid(self.ba_rate_scale_phi, 0.1, 10.0),
        }
    }

    /// Get parameter names (for logging/debugging)
    pub fn get_param_names() -> Vec<&'static str> {
        vec![
            // Core HJB Strategy
            "phi", "lambda_base", "max_position", "maker_fee_bps", "taker_fee_bps",
            "leverage", "max_leverage", "margin_safety", "enable_multi_level", "enable_robust_control",
            // Multi-Level Config
            "num_levels", "level_spacing_bps", "min_spread_bps", "vol_to_spread_factor",
            "base_maker_size", "maker_aggression_decay", "taker_size_multiplier", "min_taker_rate",
            // EWMA Volatility Model
            "ewma_half_life", "ewma_alpha", "ewma_outlier_thresh", "ewma_min_vol",
            "ewma_max_vol", "ewma_tick_freq",
            // Particle Filter Config
            "pf_mu", "pf_phi", "pf_sigma_eta", "pf_update_interval",
            // Hybrid Grounding Config
            "grounding_base", "grounding_sensitivity", "min_grounding", "max_grounding",
            "ba_tracking_window", "ba_ewma_alpha", "ba_vol_window", "ba_rate_scale",
        ]
    }

    /// Add random noise for exploration (used by SPSA)
    pub fn add_noise(&self, noise: &[f64]) -> Self {
        assert_eq!(noise.len(), 36);
        let mut vec = self.to_vec();
        for i in 0..36 {
            vec[i] += noise[i];
        }
        Self::from_vec(&vec)
    }
}

// ============================================================================
// Helper Functions: Sigmoid Transforms
// ============================================================================

/// Sigmoid function: σ(x) = 1 / (1 + exp(-x))
#[inline]
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Apply sigmoid transform to map φ ∈ (-∞, +∞) to θ ∈ [min, max]
#[inline]
fn apply_sigmoid(phi: f64, min: f64, max: f64) -> f64 {
    min + (max - min) * sigmoid(phi)
}

/// Inverse sigmoid: φ = log(θ / (1 - θ)) where θ = (x - min) / (max - min)
#[inline]
fn inverse_sigmoid(x: f64, min: f64, max: f64) -> f64 {
    let theta = (x - min) / (max - min);
    let theta_clamped = theta.clamp(0.001, 0.999); // Avoid log(0)
    (theta_clamped / (1.0 - theta_clamped)).ln()
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sigmoid_roundtrip() {
        let x = 5.5;
        let phi = inverse_sigmoid(x, 1.0, 10.0);
        let x_recovered = apply_sigmoid(phi, 1.0, 10.0);
        assert!((x - x_recovered).abs() < 1e-6);
    }

    #[test]
    fn test_vec_roundtrip() {
        let params = StrategyTuningParams::default();
        let vec = params.to_vec();
        assert_eq!(vec.len(), 36);
        let params_recovered = StrategyTuningParams::from_vec(&vec);
        let vec_recovered = params_recovered.to_vec();
        for i in 0..36 {
            assert!((vec[i] - vec_recovered[i]).abs() < 1e-10);
        }
    }

    #[test]
    fn test_constrained_bounds() {
        let params = StrategyTuningParams::default();
        let constrained = params.get_constrained();

        // Check all bounds
        assert!(constrained.phi >= 0.001 && constrained.phi <= 0.1);
        assert!(constrained.lambda_base >= 0.1 && constrained.lambda_base <= 10.0);
        assert!(constrained.leverage >= 1 && constrained.leverage <= 10);
        assert!(constrained.num_levels >= 1 && constrained.num_levels <= 5);
        assert!(constrained.ewma_alpha >= 0.01 && constrained.ewma_alpha <= 0.3);
        assert!(constrained.pf_mu >= -20.0 && constrained.pf_mu <= -10.0);
        assert!(constrained.grounding_strength_base >= 0.05 && constrained.grounding_strength_base <= 0.5);
    }

    #[test]
    fn test_add_noise() {
        let params = StrategyTuningParams::default();
        let noise: Vec<f64> = (0..36).map(|i| (i as f64) * 0.01).collect();
        let noisy_params = params.add_noise(&noise);

        let vec1 = params.to_vec();
        let vec2 = noisy_params.to_vec();

        for i in 0..36 {
            assert!((vec2[i] - vec1[i] - noise[i]).abs() < 1e-10);
        }
    }
}
