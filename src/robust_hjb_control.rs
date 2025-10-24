//! Robust Control with Parameter Uncertainty
//!
//! This module extends the HJB framework with robust optimization that accounts for
//! parameter uncertainty from the particle filter. It implements worst-case optimization
//! to protect against estimation errors.

use serde::{Deserialize, Serialize};

/// Parameter uncertainty bounds from particle filter
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ParameterUncertainty {
    /// Adverse selection uncertainty (±bps)
    pub epsilon_mu: f64,
    
    /// Volatility uncertainty (±bps)
    pub epsilon_sigma: f64,
    
    /// Confidence level (e.g., 0.95 for 95% CI)
    pub confidence: f64,
}

impl Default for ParameterUncertainty {
    fn default() -> Self {
        Self {
            epsilon_mu: 0.5,
            epsilon_sigma: 5.0,
            confidence: 0.95,
        }
    }
}

impl ParameterUncertainty {
    /// Extract from particle filter statistics
    /// This should be called with your ParticleFilterState
    pub fn from_particle_filter_stats(
        mu_std: f64,
        sigma_std: f64,
        confidence: f64,
    ) -> Self {
        // Convert std dev to confidence interval
        // For 95% CI: ±1.96 * std_dev
        // For 99% CI: ±2.58 * std_dev
        let z_score = match confidence {
            x if x >= 0.99 => 2.58,
            x if x >= 0.95 => 1.96,
            x if x >= 0.90 => 1.645,
            _ => 1.96,
        };
        
        Self {
            epsilon_mu: z_score * mu_std,
            epsilon_sigma: z_score * sigma_std,
            confidence,
        }
    }
    
    /// Check if uncertainty is high (should trade more conservatively)
    pub fn is_high(&self) -> bool {
        self.epsilon_mu > 1.0 || self.epsilon_sigma > 10.0
    }
}

/// Robust optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RobustConfig {
    /// Enable robust optimization
    pub enabled: bool,
    
    /// Robustness scaling (0.0 = nominal, 1.0 = full worst-case)
    pub robustness_level: f64,
    
    /// Minimum uncertainty to apply robustness
    pub min_epsilon_mu: f64,
    pub min_epsilon_sigma: f64,
}

impl Default for RobustConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            robustness_level: 0.7, // 70% toward worst-case
            min_epsilon_mu: 0.2,
            min_epsilon_sigma: 2.0,
        }
    }
}

/// Robust parameter adjustments
pub struct RobustParameters {
    /// Worst-case adverse selection (for inventory management)
    pub mu_worst_case: f64,
    
    /// Worst-case volatility (for spread widening)
    pub sigma_worst_case: f64,
    
    /// Spread adjustment multiplier
    pub spread_multiplier: f64,
    
    /// Inventory penalty multiplier
    pub inventory_penalty_multiplier: f64,
}

impl RobustParameters {
    /// Compute robust parameters based on uncertainty
    pub fn compute(
        nominal_mu: f64,
        nominal_sigma: f64,
        inventory: f64,
        uncertainty: &ParameterUncertainty,
        config: &RobustConfig,
    ) -> Self {
        if !config.enabled {
            return Self {
                mu_worst_case: nominal_mu,
                sigma_worst_case: nominal_sigma,
                spread_multiplier: 1.0,
                inventory_penalty_multiplier: 1.0,
            };
        }
        
        // Only apply robustness if uncertainty is significant
        let apply_mu_robust = uncertainty.epsilon_mu > config.min_epsilon_mu;
        let apply_sigma_robust = uncertainty.epsilon_sigma > config.min_epsilon_sigma;
        
        // Worst-case drift depends on inventory position
        let mu_adjustment = if apply_mu_robust {
            if inventory > 0.0 {
                // Long position: worst case is price drops
                -uncertainty.epsilon_mu * config.robustness_level
            } else if inventory < 0.0 {
                // Short position: worst case is price rises
                uncertainty.epsilon_mu * config.robustness_level
            } else {
                0.0 // No position, no directional risk
            }
        } else {
            0.0
        };
        
        let mu_worst_case = nominal_mu + mu_adjustment;
        
        // Worst-case volatility is always higher (more uncertainty = wider spreads)
        let sigma_worst_case = if apply_sigma_robust {
            nominal_sigma + uncertainty.epsilon_sigma * config.robustness_level
        } else {
            nominal_sigma
        };
        
        // Spread multiplier: increase spread with volatility uncertainty
        let spread_multiplier = if apply_sigma_robust {
            1.0 + (uncertainty.epsilon_sigma / nominal_sigma.max(1.0)) * 0.5 * config.robustness_level
        } else {
            1.0
        };
        
        // Inventory penalty: increase penalty with drift uncertainty
        let inventory_penalty_multiplier = if apply_mu_robust {
            1.0 + (uncertainty.epsilon_mu / 2.0) * config.robustness_level
        } else {
            1.0
        };
        
        Self {
            mu_worst_case,
            sigma_worst_case,
            spread_multiplier,
            inventory_penalty_multiplier,
        }
    }
    
    /// Apply robust adjustments to base spread
    pub fn adjust_spread(&self, base_spread_bps: f64) -> f64 {
        base_spread_bps * self.spread_multiplier
    }
    
    /// Apply robust adjustments to adverse selection offset
    pub fn adjust_adverse_selection(&self, base_offset_bps: f64, inventory: f64) -> f64 {
        // Shift quotes away from adverse direction
        if inventory > 0.0 {
            // Long: worst case is downward drift, so widen asks less, bids more
            base_offset_bps + self.mu_worst_case
        } else if inventory < 0.0 {
            // Short: worst case is upward drift
            base_offset_bps - self.mu_worst_case
        } else {
            base_offset_bps
        }
    }
}

// Helper function for computing standard deviation (currently unused but available for future use)
#[allow(dead_code)]
fn compute_std(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    
    let mean: f64 = values.iter().sum::<f64>() / values.len() as f64;
    let variance: f64 = values.iter()
        .map(|v| (v - mean).powi(2))
        .sum::<f64>() / values.len() as f64;
    
    variance.sqrt()
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_parameter_uncertainty() {
        let uncertainty = ParameterUncertainty::from_particle_filter_stats(
            0.5,  // mu_std
            5.0,  // sigma_std
            0.95,
        );
        
        assert!((uncertainty.epsilon_mu - 0.5 * 1.96).abs() < 0.01);
        assert!((uncertainty.epsilon_sigma - 5.0 * 1.96).abs() < 0.1);
    }
    
    #[test]
    fn test_robust_parameters_long() {
        let uncertainty = ParameterUncertainty {
            epsilon_mu: 1.0,
            epsilon_sigma: 10.0,
            confidence: 0.95,
        };
        
        let config = RobustConfig::default();
        
        // Long position: worst case is price drops
        let params = RobustParameters::compute(
            2.0,   // nominal_mu (bullish)
            100.0, // nominal_sigma
            50.0,  // inventory (long)
            &uncertainty,
            &config,
        );
        
        // Worst case should be less bullish (or bearish)
        assert!(params.mu_worst_case < 2.0);
        
        // Volatility should be higher
        assert!(params.sigma_worst_case > 100.0);
        
        // Spread should be wider
        assert!(params.spread_multiplier > 1.0);
    }
    
    #[test]
    fn test_robust_parameters_short() {
        let uncertainty = ParameterUncertainty {
            epsilon_mu: 1.0,
            epsilon_sigma: 10.0,
            confidence: 0.95,
        };
        
        let config = RobustConfig::default();
        
        // Short position: worst case is price rises
        let params = RobustParameters::compute(
            -2.0,   // nominal_mu (bearish)
            100.0,
            -50.0,  // inventory (short)
            &uncertainty,
            &config,
        );
        
        // Worst case should be less bearish (or bullish)
        assert!(params.mu_worst_case > -2.0);
    }
    
    #[test]
    fn test_robust_disabled() {
        let uncertainty = ParameterUncertainty::default();
        let config = RobustConfig {
            enabled: false,
            ..Default::default()
        };
        
        let params = RobustParameters::compute(
            2.0,
            100.0,
            50.0,
            &uncertainty,
            &config,
        );
        
        // Should be unchanged
        assert_eq!(params.mu_worst_case, 2.0);
        assert_eq!(params.sigma_worst_case, 100.0);
        assert_eq!(params.spread_multiplier, 1.0);
    }
}