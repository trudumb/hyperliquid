// ============================================================================
// Robust Control Trait - Swappable Uncertainty Management Component
// ============================================================================
//
// This trait defines the interface for any component that can handle
// parameter uncertainty in the optimization. Different implementations can use:
// - Worst-case optimization (min-max)
// - Bayesian robust optimization
// - Adaptive robustness based on uncertainty levels
// - Confidence interval-based adjustments
//
// # Design Philosophy
//
// Robust control components are **stateless calculators** that:
// - Take parameter estimates and their uncertainties as input
// - Compute worst-case or risk-adjusted parameters
// - Return adjustment multipliers for spreads and inventory penalties
//
// The component is stateless - all state is passed in via parameters.
// This makes it easier to test and swap implementations.
//
// # Example Implementation
//
// ```rust
// struct SimpleWorstCaseRobust {
//     robustness_level: f64,
// }
//
// impl RobustControlModel for SimpleWorstCaseRobust {
//     fn compute_robust_parameters(
//         &self,
//         volatility_bps: f64,
//         vol_uncertainty_bps: f64,
//         adverse_selection_bps: f64,
//         _as_uncertainty_bps: f64,
//     ) -> RobustParameters {
//         // Simple worst-case: add uncertainty to parameters
//         let sigma_worst = volatility_bps + self.robustness_level * vol_uncertainty_bps;
//         let spread_mult = 1.0 + self.robustness_level * (vol_uncertainty_bps / volatility_bps);
//
//         RobustParameters {
//             mu_worst_case: adverse_selection_bps,
//             sigma_worst_case: sigma_worst,
//             spread_multiplier: spread_mult,
//             inventory_penalty_multiplier: 1.0,
//         }
//     }
// }
// ```

/// Parameter uncertainty quantification
#[derive(Debug, Clone, Copy)]
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
    /// Create from particle filter statistics.
    ///
    /// Converts standard deviations to confidence intervals using z-scores.
    ///
    /// # Arguments
    /// - `mu_std`: Standard deviation of adverse selection estimate
    /// - `sigma_std`: Standard deviation of volatility estimate
    /// - `confidence`: Desired confidence level (0.90, 0.95, 0.99)
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

/// Robust parameter adjustments
#[derive(Debug, Clone)]
pub struct RobustParameters {
    /// Worst-case adverse selection (for inventory management)
    pub mu_worst_case: f64,

    /// Worst-case volatility (for spread widening)
    pub sigma_worst_case: f64,

    /// Spread adjustment multiplier (>= 1.0)
    /// Applied to base spreads to account for uncertainty
    pub spread_multiplier: f64,

    /// Inventory penalty multiplier (>= 1.0)
    /// Applied to inventory penalties to be more risk-averse
    pub inventory_penalty_multiplier: f64,
}

/// A swappable component for robust control under parameter uncertainty.
///
/// Robust control models take parameter estimates and their uncertainties,
/// then compute worst-case or risk-adjusted parameters and adjustment
/// multipliers for the optimizer.
///
/// The goal is to make the strategy more conservative when model confidence
/// is low, protecting against estimation errors.
pub trait RobustControlModel: Send {
    /// Compute robust parameters given parameter estimates and uncertainties.
    ///
    /// This method takes point estimates (volatility, adverse selection) and
    /// their uncertainties, then computes worst-case parameters and adjustment
    /// multipliers.
    ///
    /// # Arguments
    /// - `volatility_bps`: Point estimate of volatility (from VolatilityModel)
    /// - `vol_uncertainty_bps`: Uncertainty of volatility estimate (std dev)
    /// - `adverse_selection_bps`: Point estimate of adverse selection (from AdverseSelectionModel)
    /// - `as_uncertainty_bps`: Uncertainty of adverse selection estimate (std dev)
    ///
    /// # Returns
    /// RobustParameters containing:
    /// - Worst-case adverse selection (typically point + robustness * uncertainty)
    /// - Worst-case volatility (typically point + robustness * uncertainty)
    /// - Spread multiplier (>= 1.0, higher when uncertainty is high)
    /// - Inventory penalty multiplier (>= 1.0, higher when uncertainty is high)
    ///
    /// # Usage by Optimizer
    /// The optimizer uses these parameters as:
    /// ```text
    /// adjusted_spread = base_spread * spread_multiplier
    /// adjusted_inventory_penalty = base_penalty * inventory_penalty_multiplier
    /// volatility_for_computation = sigma_worst_case
    /// adverse_selection_for_computation = mu_worst_case
    /// ```
    ///
    /// # Example
    /// ```text
    /// If volatility = 100 bps with uncertainty = 10 bps:
    /// - sigma_worst_case might be 107 bps (100 + 0.7 * 10)
    /// - spread_multiplier might be 1.05 (widen spreads by 5%)
    ///
    /// This protects against underestimating volatility.
    /// ```
    fn compute_robust_parameters(
        &self,
        volatility_bps: f64,
        vol_uncertainty_bps: f64,
        adverse_selection_bps: f64,
        as_uncertainty_bps: f64,
    ) -> RobustParameters;

    /// Check if the component is enabled.
    ///
    /// Some implementations may allow disabling robust control entirely,
    /// returning nominal parameters (no adjustments).
    fn is_enabled(&self) -> bool;
}
