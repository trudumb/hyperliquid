// ============================================================================
// Standard Robust Control Implementation
// ============================================================================
//
// This component implements the RobustControlModel trait using worst-case
// optimization under parameter uncertainty. It follows the min-max framework
// where we optimize for the worst-case parameter values within an uncertainty
// set.
//
// # Algorithm
//
// Given point estimates (μ, σ) and uncertainties (ε_μ, ε_σ), compute:
//
// 1. **Worst-Case Parameters**:
//    ```
//    μ_worst = μ + robustness_level * ε_μ
//    σ_worst = σ + robustness_level * ε_σ
//    ```
//
// 2. **Spread Multiplier**:
//    ```
//    spread_mult = 1.0 + robustness_level * (ε_σ / σ)
//    ```
//    - Widens spreads when volatility uncertainty is high
//    - Protects against underestimating volatility
//
// 3. **Inventory Penalty Multiplier**:
//    ```
//    inv_mult = 1.0 + robustness_level * (ε_μ / |μ|)
//    ```
//    - Increases inventory penalty when adverse selection uncertainty is high
//    - Makes the strategy more risk-averse with positions
//
// # Configuration
//
// - `enabled`: Whether to apply robust adjustments
// - `robustness_level`: How conservative to be (0.0 to 1.0)
//   - 0.0 = nominal (no adjustments)
//   - 1.0 = full worst-case
// - `min_epsilon_mu`, `min_epsilon_sigma`: Thresholds below which robustness isn't applied
//
// # Example
//
// ```rust
// use strategies::components::{RobustControlModel, StandardRobustControl};
//
// let robust = StandardRobustControl::new_default();
//
// let params = robust.compute_robust_parameters(
//     100.0,  // volatility = 100 bps
//     10.0,   // vol uncertainty = 10 bps
//     5.0,    // adverse selection = 5 bps
//     0.5,    // as uncertainty = 0.5 bps
// );
//
// // Result: sigma_worst_case = 107 bps (100 + 0.7 * 10)
// //         spread_multiplier = 1.07 (widen spreads by 7%)
// ```

use super::robust_control::{RobustControlModel, RobustParameters};
use serde::{Deserialize, Serialize};

/// Configuration for standard robust control component
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

impl RobustConfig {
    /// Create a new config with validation
    pub fn new(
        enabled: bool,
        robustness_level: f64,
        min_epsilon_mu: f64,
        min_epsilon_sigma: f64,
    ) -> Result<Self, String> {
        if !(0.0..=1.0).contains(&robustness_level) {
            return Err("robustness_level must be between 0.0 and 1.0".to_string());
        }
        if min_epsilon_mu < 0.0 {
            return Err("min_epsilon_mu must be non-negative".to_string());
        }
        if min_epsilon_sigma < 0.0 {
            return Err("min_epsilon_sigma must be non-negative".to_string());
        }

        Ok(Self {
            enabled,
            robustness_level,
            min_epsilon_mu,
            min_epsilon_sigma,
        })
    }
}

/// Standard robust control implementation.
///
/// This component uses worst-case optimization to protect against
/// parameter estimation errors.
#[derive(Debug, Clone)]
pub struct StandardRobustControl {
    pub config: RobustConfig,
}

impl StandardRobustControl {
    /// Create a new standard robust control component with default config.
    pub fn new_default() -> Self {
        Self {
            config: RobustConfig::default(),
        }
    }

    /// Create a new standard robust control component with custom config.
    pub fn new(config: RobustConfig) -> Self {
        Self { config }
    }
}

impl RobustControlModel for StandardRobustControl {
    fn compute_robust_parameters(
        &self,
        volatility_bps: f64,
        vol_uncertainty_bps: f64,
        adverse_selection_bps: f64,
        as_uncertainty_bps: f64,
    ) -> RobustParameters {
        // If disabled, return nominal parameters (no adjustments)
        if !self.config.enabled {
            return RobustParameters {
                mu_worst_case: adverse_selection_bps,
                sigma_worst_case: volatility_bps,
                spread_multiplier: 1.0,
                inventory_penalty_multiplier: 1.0,
            };
        }

        // Apply minimum thresholds - don't apply robustness for tiny uncertainties
        let effective_epsilon_mu = if as_uncertainty_bps >= self.config.min_epsilon_mu {
            as_uncertainty_bps
        } else {
            0.0
        };

        let effective_epsilon_sigma = if vol_uncertainty_bps >= self.config.min_epsilon_sigma {
            vol_uncertainty_bps
        } else {
            0.0
        };

        // Compute worst-case parameters
        // For volatility: always increase (worst case is higher vol)
        let sigma_worst_case = volatility_bps + self.config.robustness_level * effective_epsilon_sigma;

        // For adverse selection: increase in absolute value
        // (worst case is stronger adverse selection)
        let mu_worst_case = if adverse_selection_bps >= 0.0 {
            adverse_selection_bps + self.config.robustness_level * effective_epsilon_mu
        } else {
            adverse_selection_bps - self.config.robustness_level * effective_epsilon_mu
        };

        // Spread multiplier: widen spreads when volatility uncertainty is high
        let spread_multiplier = if volatility_bps > 0.0 && effective_epsilon_sigma > 0.0 {
            1.0 + self.config.robustness_level * (effective_epsilon_sigma / volatility_bps)
        } else {
            1.0
        };

        // Inventory penalty multiplier: increase when adverse selection uncertainty is high
        let inventory_penalty_multiplier = if adverse_selection_bps.abs() > 0.0 && effective_epsilon_mu > 0.0 {
            1.0 + self.config.robustness_level * (effective_epsilon_mu / adverse_selection_bps.abs())
        } else {
            1.0
        };

        // Ensure multipliers are at least 1.0
        let spread_multiplier = spread_multiplier.max(1.0);
        let inventory_penalty_multiplier = inventory_penalty_multiplier.max(1.0);

        RobustParameters {
            mu_worst_case,
            sigma_worst_case,
            spread_multiplier,
            inventory_penalty_multiplier,
        }
    }

    fn is_enabled(&self) -> bool {
        self.config.enabled
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_robust_parameters_basic() {
        let robust = StandardRobustControl::new_default();

        let params = robust.compute_robust_parameters(
            100.0, // volatility
            10.0,  // vol uncertainty
            5.0,   // adverse selection
            0.5,   // as uncertainty
        );

        // With robustness_level = 0.7:
        // sigma_worst = 100 + 0.7 * 10 = 107
        assert!((params.sigma_worst_case - 107.0).abs() < 0.01);

        // spread_mult = 1.0 + 0.7 * (10 / 100) = 1.07
        assert!((params.spread_multiplier - 1.07).abs() < 0.01);

        // Should be conservative
        assert!(params.spread_multiplier >= 1.0);
        assert!(params.inventory_penalty_multiplier >= 1.0);
    }

    #[test]
    fn test_disabled_robustness() {
        let mut config = RobustConfig::default();
        config.enabled = false;

        let robust = StandardRobustControl::new(config);

        let params = robust.compute_robust_parameters(100.0, 10.0, 5.0, 0.5);

        // Should return nominal parameters (no adjustments)
        assert_eq!(params.sigma_worst_case, 100.0);
        assert_eq!(params.mu_worst_case, 5.0);
        assert_eq!(params.spread_multiplier, 1.0);
        assert_eq!(params.inventory_penalty_multiplier, 1.0);
    }

    #[test]
    fn test_minimum_thresholds() {
        let robust = StandardRobustControl::new_default();

        // Uncertainty below thresholds (min_epsilon_sigma = 2.0)
        let params = robust.compute_robust_parameters(
            100.0, // volatility
            1.0,   // vol uncertainty (< 2.0 threshold)
            5.0,   // adverse selection
            0.1,   // as uncertainty (< 0.2 threshold)
        );

        // Should not apply robustness adjustments (or minimal)
        assert!((params.sigma_worst_case - 100.0).abs() < 0.01);
        assert!((params.spread_multiplier - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_high_uncertainty() {
        let robust = StandardRobustControl::new_default();

        // High uncertainty scenario
        let params = robust.compute_robust_parameters(
            100.0, // volatility
            50.0,  // vol uncertainty (50% of vol)
            5.0,   // adverse selection
            2.0,   // as uncertainty (40% of as)
        );

        // Should apply significant adjustments
        assert!(params.sigma_worst_case > 130.0); // 100 + 0.7 * 50 = 135
        assert!(params.spread_multiplier > 1.3);  // 1.0 + 0.7 * (50/100) = 1.35
    }

    #[test]
    fn test_config_validation() {
        // Valid config
        assert!(RobustConfig::new(true, 0.5, 0.1, 1.0).is_ok());

        // Invalid robustness_level
        assert!(RobustConfig::new(true, 1.5, 0.1, 1.0).is_err());
        assert!(RobustConfig::new(true, -0.1, 0.1, 1.0).is_err());

        // Invalid min_epsilon
        assert!(RobustConfig::new(true, 0.5, -0.1, 1.0).is_err());
        assert!(RobustConfig::new(true, 0.5, 0.1, -1.0).is_err());
    }
}
