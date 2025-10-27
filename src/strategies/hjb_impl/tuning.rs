//! Contains structs and logic for autonomous parameter tuning (Adam optimizer).

use super::utils::*;
use serde::{Deserialize, Serialize};

/// Tunable parameters (UNCONSTRAINED)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TuningParams {
    pub skew_adjustment_factor_phi: f64,
    pub adverse_selection_adjustment_factor_phi: f64,
    pub adverse_selection_lambda_phi: f64,
    pub inventory_urgency_threshold_phi: f64,
    pub liquidation_rate_multiplier_phi: f64,
    pub min_spread_base_ratio_phi: f64,
    pub adverse_selection_spread_scale_phi: f64,
    pub control_gap_threshold_phi: f64,
}

/// Tunable parameters (CONSTRAINED)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstrainedTuningParams {
    pub skew_adjustment_factor: f64,
    pub adverse_selection_adjustment_factor: f64,
    pub adverse_selection_lambda: f64,
    pub inventory_urgency_threshold: f64,
    pub liquidation_rate_multiplier: f64,
    pub min_spread_base_ratio: f64,
    pub adverse_selection_spread_scale: f64,
    pub control_gap_threshold: f64,
}

/// Adam Optimizer State
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdamOptimizerState {
    pub m: Vec<f64>,
    pub v: Vec<f64>,
    pub t: usize,
    pub alpha: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub epsilon: f64,
}

impl Default for AdamOptimizerState {
    fn default() -> Self {
        Self {
            m: vec![0.0; 8],
            v: vec![0.0; 8],
            t: 0,
            alpha: 0.1,
            beta1: 0.9,
            beta2: 0.99,
            epsilon: 1e-8,
        }
    }
}

impl AdamOptimizerState {
    pub fn new(alpha: f64, beta1: f64, beta2: f64) -> Self {
        Self {
            m: vec![0.0; 8],
            v: vec![0.0; 8],
            t: 0,
            alpha,
            beta1,
            beta2,
            epsilon: 1e-8,
        }
    }

    pub fn compute_update(&mut self, gradient_vector: &[f64]) -> Vec<f64> {
        assert_eq!(gradient_vector.len(), 8, "Gradient vector must have 8 elements");
        self.t += 1;
        let t = self.t as f64;
        let mut updates = Vec::with_capacity(8);
        for i in 0..8 {
            let g_t = gradient_vector[i];
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * g_t;
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * g_t.powi(2);
            let m_hat = self.m[i] / (1.0 - self.beta1.powf(t));
            let v_hat = self.v[i] / (1.0 - self.beta2.powf(t));
            let update = self.alpha * m_hat / (v_hat.sqrt() + self.epsilon);
            updates.push(update);
        }
        updates
    }

    pub fn reset(&mut self) {
        self.m = vec![0.0; 8];
        self.v = vec![0.0; 8];
        self.t = 0;
    }

    pub fn get_effective_learning_rate(&self, param_index: usize) -> f64 {
        if self.t == 0 || param_index >= 8 {
            return 0.0;
        }
        let t = self.t as f64;
        let v_hat = self.v[param_index] / (1.0 - self.beta2.powf(t));
        self.alpha / (v_hat.sqrt() + self.epsilon)
    }
}

impl Default for TuningParams {
    fn default() -> Self {
        Self {
            skew_adjustment_factor_phi: inv_scaled_sigmoid(0.5, 0.0, 2.0),
            adverse_selection_adjustment_factor_phi: inv_scaled_sigmoid(0.5, 0.0, 2.0),
            adverse_selection_lambda_phi: inv_scaled_sigmoid(0.1, 0.0, 1.0),
            inventory_urgency_threshold_phi: inv_scaled_sigmoid(0.7, 0.0, 1.0),
            liquidation_rate_multiplier_phi: inv_scaled_sigmoid(10.0, 0.0, 100.0),
            min_spread_base_ratio_phi: inv_scaled_sigmoid(0.2, 0.0, 1.0),
            adverse_selection_spread_scale_phi: inv_exp_transform(100.0),
            control_gap_threshold_phi: inv_exp_transform(0.1),
        }
    }
}

impl TuningParams {
    pub fn get_constrained(&self) -> ConstrainedTuningParams {
        ConstrainedTuningParams {
            skew_adjustment_factor: scaled_sigmoid(self.skew_adjustment_factor_phi, 0.0, 2.0),
            adverse_selection_adjustment_factor: scaled_sigmoid(
                self.adverse_selection_adjustment_factor_phi,
                0.0,
                2.0,
            ),
            adverse_selection_lambda: scaled_sigmoid(self.adverse_selection_lambda_phi, 0.0, 1.0),
            inventory_urgency_threshold: scaled_sigmoid(
                self.inventory_urgency_threshold_phi,
                0.0,
                1.0,
            ),
            liquidation_rate_multiplier: scaled_sigmoid(
                self.liquidation_rate_multiplier_phi,
                0.0,
                100.0,
            ),
            min_spread_base_ratio: scaled_sigmoid(self.min_spread_base_ratio_phi, 0.0, 1.0),
            adverse_selection_spread_scale: exp_transform(
                self.adverse_selection_spread_scale_phi,
            ),
            control_gap_threshold: exp_transform(self.control_gap_threshold_phi),
        }
    }

    fn from_constrained(constrained: &ConstrainedTuningParams) -> Self {
        Self {
            skew_adjustment_factor_phi: inv_scaled_sigmoid(
                constrained.skew_adjustment_factor,
                0.0,
                2.0,
            ),
            adverse_selection_adjustment_factor_phi: inv_scaled_sigmoid(
                constrained.adverse_selection_adjustment_factor,
                0.0,
                2.0,
            ),
            adverse_selection_lambda_phi: inv_scaled_sigmoid(
                constrained.adverse_selection_lambda,
                0.0,
                1.0,
            ),
            inventory_urgency_threshold_phi: inv_scaled_sigmoid(
                constrained.inventory_urgency_threshold,
                0.0,
                1.0,
            ),
            liquidation_rate_multiplier_phi: inv_scaled_sigmoid(
                constrained.liquidation_rate_multiplier,
                0.0,
                100.0,
            ),
            min_spread_base_ratio_phi: inv_scaled_sigmoid(
                constrained.min_spread_base_ratio,
                0.0,
                1.0,
            ),
            adverse_selection_spread_scale_phi: inv_exp_transform(
                constrained.adverse_selection_spread_scale,
            ),
            control_gap_threshold_phi: inv_exp_transform(constrained.control_gap_threshold),
        }
    }

    pub fn from_json_file(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let contents = std::fs::read_to_string(path)?;
        let constrained_params: ConstrainedTuningParams = serde_json::from_str(&contents)?;
        Ok(Self::from_constrained(&constrained_params))
    }

    pub fn to_json_file(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let constrained_params = self.get_constrained();
        let contents = serde_json::to_string_pretty(&constrained_params)?;
        std::fs::write(path, contents)?;
        Ok(())
    }
}
