// ============================================================================
// Auto-Tuner Integration for HJB Strategy
// ============================================================================
//
// This module provides integration hooks between the MultiObjectiveTuner
// and the HjbStrategy, enabling online parameter optimization during live trading.
//
// # Integration Architecture
//
// 1. **Episode-Based Evaluation**: Each "episode" = N market updates
// 2. **Candidate Generation**: Tuner generates φ+ and φ- candidates
// 3. **Performance Tracking**: PerformanceTracker records all trading events
// 4. **Score Computation**: Multi-objective score after each episode
// 5. **Parameter Update**: SPSA gradient → Adam update → new parameters
//
// # Usage Pattern
//
// ```rust
// // In HjbStrategy::new()
// let tuner_integration = TunerIntegration::new(config, initial_params, seed);
//
// // In HjbStrategy::on_tick()
// if let Some(new_params) = tuner_integration.on_tick() {
//     self.apply_new_parameters(new_params);
// }
//
// // In HjbStrategy::on_market_update()
// tuner_integration.on_market_update(&update);
//
// // In HjbStrategy::on_user_update()
// tuner_integration.on_user_fill(pnl, ...);
// ```

use super::components::{
    MultiObjectiveTuner, TunerConfig, PerformanceTracker,
    StrategyTuningParams, StrategyConstrainedParams,
};
use log::{info, debug};

// ============================================================================
// Tuner Integration State
// ============================================================================

pub struct TunerIntegration {
    /// The multi-objective tuner (None if disabled)
    tuner: Option<MultiObjectiveTuner>,

    /// Episode tracking
    updates_in_episode: usize,
    updates_per_episode: usize,

    /// Current evaluation phase
    phase: EvaluationPhase,

    /// Pending parameter application
    pending_params: Option<StrategyConstrainedParams>,
}

#[derive(Debug, Clone, PartialEq)]
enum EvaluationPhase {
    /// Using current best parameters (not evaluating)
    Normal,

    /// Evaluating φ+ candidate
    EvaluatingPlus,

    /// Evaluating φ- candidate
    EvaluatingMinus,

    /// Ready to update parameters
    #[allow(dead_code)]
    ReadyToUpdate,
}

impl TunerIntegration {
    /// Create new tuner integration
    pub fn new(
        config: TunerConfig,
        initial_params: StrategyTuningParams,
        updates_per_episode: usize,
        seed: u64,
    ) -> Self {
        let tuner = if config.enabled {
            Some(MultiObjectiveTuner::new(config, initial_params, seed))
        } else {
            None
        };

        Self {
            tuner,
            updates_in_episode: 0,
            updates_per_episode,
            phase: EvaluationPhase::Normal,
            pending_params: None,
        }
    }

    /// Check if tuning is enabled
    pub fn is_enabled(&self) -> bool {
        self.tuner.is_some()
    }

    /// Called on every market update (tick)
    pub fn on_market_update(&mut self) {
        self.updates_in_episode += 1;
    }

    /// Called on every tick - returns new parameters if update is ready
    pub fn on_tick(&mut self) -> Option<StrategyConstrainedParams> {
        let tuner = match self.tuner.as_mut() {
            Some(t) => t,
            None => return None,
        };

        // Check if episode is complete
        if self.updates_in_episode < self.updates_per_episode {
            return None;
        }

        // Reset episode counter
        self.updates_in_episode = 0;

        // Handle different phases
        match self.phase {
            EvaluationPhase::Normal => {
                // Check if we should start evaluation
                if tuner.should_update() {
                    // Generate candidates
                    let (params_plus, _params_minus) = tuner.generate_candidates();

                    info!("[TUNER] Starting candidate evaluation");
                    self.phase = EvaluationPhase::EvaluatingPlus;
                    self.pending_params = Some(params_plus.clone());

                    // Return φ+ for evaluation
                    return Some(params_plus);
                }
            }

            EvaluationPhase::EvaluatingPlus => {
                // Compute score for φ+
                let weights = tuner.config.objective_weights.clone();
                let score = tuner.performance_tracker().compute_multi_objective_score(&weights);
                tuner.record_candidate_score(true, score);

                debug!("[TUNER] Plus candidate score: {:.6}", score);

                // Move to φ- evaluation
                let (_, params_minus) = tuner.generate_candidates();
                self.phase = EvaluationPhase::EvaluatingMinus;
                self.pending_params = Some(params_minus.clone());

                // Return φ- for evaluation
                return Some(params_minus);
            }

            EvaluationPhase::EvaluatingMinus => {
                // Compute score for φ-
                let weights = tuner.config.objective_weights.clone();
                let score = tuner.performance_tracker().compute_multi_objective_score(&weights);
                tuner.record_candidate_score(false, score);

                debug!("[TUNER] Minus candidate score: {:.6}", score);

                // Update parameters using SPSA + Adam
                let new_params = tuner.update_parameters();

                info!("[TUNER] Parameters updated");
                self.phase = EvaluationPhase::Normal;
                self.pending_params = Some(new_params.clone());

                // Return new optimal parameters
                return Some(new_params);
            }

            EvaluationPhase::ReadyToUpdate => {
                // Should not reach here
                self.phase = EvaluationPhase::Normal;
            }
        }

        None
    }

    /// Get performance tracker for recording events
    pub fn performance_tracker(&mut self) -> Option<&mut PerformanceTracker> {
        self.tuner.as_mut().map(|t| t.performance_tracker())
    }

    /// Get current evaluation phase (for logging/debugging)
    pub fn current_phase(&self) -> &str {
        match self.phase {
            EvaluationPhase::Normal => "normal",
            EvaluationPhase::EvaluatingPlus => "eval_plus",
            EvaluationPhase::EvaluatingMinus => "eval_minus",
            EvaluationPhase::ReadyToUpdate => "ready_update",
        }
    }

    /// Export tuning history
    pub fn export_history(&self) -> Option<String> {
        self.tuner.as_ref().map(|t| {
            serde_json::to_string_pretty(&t.export_history()).unwrap_or_default()
        })
    }

    /// Get best parameters found so far
    pub fn get_best_params(&self) -> Option<StrategyConstrainedParams> {
        self.tuner.as_ref().map(|t| t.get_best_params())
    }
}

// ============================================================================
// Helper: Apply Parameters to HjbStrategy
// ============================================================================

/// Helper struct to apply new parameters to HjbStrategy configuration
pub struct ParameterApplicator;

impl ParameterApplicator {
    /// Apply constrained parameters to strategy config
    pub fn apply_to_hjb_strategy(
        params: &StrategyConstrainedParams,
        config: &mut super::hjb_strategy::HjbStrategyConfig,
    ) {
        // Core HJB Strategy
        config.phi = params.phi;
        config.lambda_base = params.lambda_base;
        config.max_absolute_position_size = params.max_absolute_position_size;
        config.maker_fee_bps = params.maker_fee_bps;
        config.taker_fee_bps = params.taker_fee_bps;
        config.leverage = params.leverage as usize;
        config.max_leverage = params.max_leverage as usize;
        config.margin_safety_buffer = params.margin_safety_buffer;
        config.enable_multi_level = params.enable_multi_level;
        config.enable_robust_control = params.enable_robust_control;

        // Multi-Level Config
        if let Some(ref mut ml_config) = config.multi_level_config {
            ml_config.max_levels = params.num_levels;
            ml_config.level_spacing_bps = params.level_spacing_bps;
            ml_config.min_profitable_spread_bps = params.min_profitable_spread_bps;
            // Note: Some params (volatility_to_spread_factor, base_maker_size, etc.)
            // are not direct fields of MultiLevelConfig but are computed dynamically.
            // These would need to be applied via the optimizer, not the config struct.
        }

        // Note: Volatility model parameters require rebuilding the model,
        // which is more complex. For now, we skip live updates to volatility params.
        // Future work: Add hot-swappable volatility model reconfiguration.

        info!("[TUNER] Applied new parameters: phi={:.4}, lambda={:.2}, max_pos={:.1}",
            params.phi, params.lambda_base, params.max_absolute_position_size);
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tuner_integration_disabled() {
        let mut config = TunerConfig::default();
        config.enabled = false;

        let params = StrategyTuningParams::default();
        let mut integration = TunerIntegration::new(config, params, 100, 42);

        assert!(!integration.is_enabled());
        assert!(integration.on_tick().is_none());
    }

    #[test]
    fn test_episode_tracking() {
        let mut config = TunerConfig::default();
        config.enabled = true;
        config.mode = "continuous".to_string();
        config.episodes_per_update = 2;

        let params = StrategyTuningParams::default();
        let mut integration = TunerIntegration::new(config, params, 100, 42);

        assert!(integration.is_enabled());

        // Simulate updates
        for _ in 0..99 {
            integration.on_market_update();
        }

        // Should not trigger yet
        assert!(integration.on_tick().is_none());
    }
}
