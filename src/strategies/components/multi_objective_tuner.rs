// ============================================================================
// Multi-Objective Auto-Tuner with SPSA + Adam
// ============================================================================
//
// This module implements online parameter tuning for the entire strategy
// configuration using:
// - Multi-objective optimization (profitability, risk, efficiency, model quality)
// - SPSA (Simultaneous Perturbation Stochastic Approximation) for gradient estimation
// - Adam optimizer for parameter updates with adaptive learning rates
//
// # Algorithm Overview
//
// 1. **Multi-Objective Scoring**: Combine 4 objectives into weighted score
//    J(θ) = w₁·profit(θ) + w₂·risk(θ) + w₃·efficiency(θ) + w₄·model_quality(θ)
//
// 2. **SPSA Gradient Estimation**: Efficient O(1) gradient approximation
//    ∇J(φ) ≈ [J(φ + c·Δ) - J(φ - c·Δ)] / (2c) · Δ⁻¹
//    where Δ ~ Rademacher(-1, +1)ⁿ (random binary perturbations)
//
// 3. **Adam Optimizer**: Adaptive learning rate with momentum
//    m_t = β₁·m_{t-1} + (1-β₁)·∇J
//    v_t = β₂·v_{t-1} + (1-β₂)·(∇J)²
//    φ_{t+1} = φ_t + α·m_t / (√v_t + ε)
//
// # Parameter Space
//
// - φ ∈ ℝ³⁶ (unconstrained space for optimization)
// - θ ∈ Θ (constrained space via sigmoid/exp transforms)
//
// # Tuning Modes
//
// 1. **Continuous Online**: Update parameters every N episodes
// 2. **Scheduled**: Update at fixed intervals (e.g., every hour)
// 3. **Adaptive**: Update when performance degrades below threshold

use super::performance_tracker::{PerformanceTracker, MultiObjectiveWeights};
use super::parameter_transforms::{StrategyTuningParams, StrategyConstrainedParams};
use serde::{Deserialize, Serialize};
use rand::prelude::*;

// ============================================================================
// Configuration
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TunerConfig {
    /// Enable auto-tuning
    pub enabled: bool,

    /// Tuning mode: "continuous", "scheduled", "adaptive"
    pub mode: String,

    /// Number of episodes between parameter updates (continuous mode)
    pub episodes_per_update: usize,

    /// Time interval in seconds between updates (scheduled mode)
    pub update_interval_seconds: f64,

    /// Performance degradation threshold for adaptive mode (0.0-1.0)
    pub adaptive_threshold: f64,

    /// SPSA perturbation magnitude
    pub spsa_c: f64,

    /// SPSA perturbation decay rate (c_k = c / k^gamma)
    pub spsa_gamma: f64,

    /// Adam learning rate
    pub adam_alpha: f64,

    /// Adam beta1 (momentum)
    pub adam_beta1: f64,

    /// Adam beta2 (second moment)
    pub adam_beta2: f64,

    /// Adam epsilon (numerical stability)
    pub adam_epsilon: f64,

    /// Learning rate decay (α_k = α / k^decay)
    pub learning_rate_decay: f64,

    /// Maximum allowed parameter change per step (L∞ norm)
    pub max_param_change: f64,

    /// Minimum episodes for gradient estimation
    pub min_episodes_for_gradient: usize,

    /// Multi-objective weights
    pub objective_weights: MultiObjectiveWeights,

    /// Log tuning actions
    pub verbose: bool,
}

impl Default for TunerConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            mode: "continuous".to_string(),
            episodes_per_update: 100,
            update_interval_seconds: 3600.0,
            adaptive_threshold: 0.7,
            spsa_c: 0.1,
            spsa_gamma: 0.101,
            adam_alpha: 0.001,
            adam_beta1: 0.9,
            adam_beta2: 0.999,
            adam_epsilon: 1e-8,
            learning_rate_decay: 0.0,
            max_param_change: 1.0,
            min_episodes_for_gradient: 20,
            objective_weights: MultiObjectiveWeights::default(),
            verbose: true,
        }
    }
}

// ============================================================================
// Tuner State
// ============================================================================

pub struct MultiObjectiveTuner {
    pub config: TunerConfig,

    // Current parameters (unconstrained space)
    current_params: StrategyTuningParams,

    // Adam optimizer state
    adam_m: Vec<f64>,  // First moment (momentum)
    adam_v: Vec<f64>,  // Second moment (variance)
    adam_t: usize,     // Time step

    // SPSA state
    iteration: usize,
    rng: StdRng,

    // Episode tracking
    episodes_since_update: usize,
    last_update_time: Option<std::time::Instant>,

    // Performance tracking
    performance_tracker: PerformanceTracker,

    // Candidate evaluation
    candidate_params_plus: Option<StrategyTuningParams>,
    candidate_params_minus: Option<StrategyTuningParams>,
    plus_score: Option<f64>,
    minus_score: Option<f64>,
    perturbation: Option<Vec<f64>>,

    // History
    score_history: Vec<f64>,
    best_score: f64,
    best_params: StrategyTuningParams,
}

impl MultiObjectiveTuner {
    /// Create new tuner with initial parameters
    pub fn new(config: TunerConfig, initial_params: StrategyTuningParams, seed: u64) -> Self {
        let param_vec = initial_params.to_vec();
        let dim = param_vec.len();

        Self {
            config,
            current_params: initial_params.clone(),
            adam_m: vec![0.0; dim],
            adam_v: vec![0.0; dim],
            adam_t: 0,
            iteration: 0,
            rng: StdRng::seed_from_u64(seed),
            episodes_since_update: 0,
            last_update_time: None,
            performance_tracker: PerformanceTracker::new(super::performance_tracker::PerformanceConfig::default()),
            candidate_params_plus: None,
            candidate_params_minus: None,
            plus_score: None,
            minus_score: None,
            perturbation: None,
            score_history: Vec::new(),
            best_score: f64::NEG_INFINITY,
            best_params: initial_params,
        }
    }

    /// Check if it's time to update parameters
    pub fn should_update(&mut self) -> bool {
        if !self.config.enabled {
            return false;
        }

        match self.config.mode.as_str() {
            "continuous" => {
                self.episodes_since_update >= self.config.episodes_per_update
            }
            "scheduled" => {
                if let Some(last_update) = self.last_update_time {
                    last_update.elapsed().as_secs_f64() >= self.config.update_interval_seconds
                } else {
                    true // First update
                }
            }
            "adaptive" => {
                // Update if performance drops below threshold
                if let Some(recent_score) = self.score_history.last() {
                    *recent_score < self.config.adaptive_threshold * self.best_score
                } else {
                    false
                }
            }
            _ => false,
        }
    }

    /// Generate candidate parameters for SPSA evaluation
    pub fn generate_candidates(&mut self) -> (StrategyConstrainedParams, StrategyConstrainedParams) {
        // Generate Rademacher perturbation: Δ ~ {-1, +1}ⁿ
        let dim = 36;
        let perturbation: Vec<f64> = (0..dim)
            .map(|_| if self.rng.gen::<bool>() { 1.0 } else { -1.0 })
            .collect();

        // Compute perturbation magnitude with decay
        let c_k = self.config.spsa_c / (self.iteration as f64 + 1.0).powf(self.config.spsa_gamma);

        // Scale perturbation
        let scaled_perturbation: Vec<f64> = perturbation.iter().map(|&d| c_k * d).collect();

        // Generate φ⁺ and φ⁻
        let params_plus = self.current_params.add_noise(&scaled_perturbation);
        let params_minus = self.current_params.add_noise(
            &scaled_perturbation.iter().map(|&x| -x).collect::<Vec<f64>>()
        );

        // Store for gradient computation
        self.candidate_params_plus = Some(params_plus.clone());
        self.candidate_params_minus = Some(params_minus.clone());
        self.perturbation = Some(perturbation);

        // Transform to constrained space
        (params_plus.get_constrained(), params_minus.get_constrained())
    }

    /// Record performance score for a candidate
    pub fn record_candidate_score(&mut self, is_plus: bool, score: f64) {
        if is_plus {
            self.plus_score = Some(score);
        } else {
            self.minus_score = Some(score);
        }

        if self.config.verbose {
            log::info!("[TUNER] Recorded {} score: {:.6}",
                if is_plus { "+" } else { "-" }, score);
        }
    }

    /// Update parameters using SPSA gradient + Adam optimizer
    pub fn update_parameters(&mut self) -> StrategyConstrainedParams {
        // Check we have both scores
        let plus_score = match self.plus_score {
            Some(s) => s,
            None => {
                log::warn!("[TUNER] Missing plus score, skipping update");
                return self.current_params.get_constrained();
            }
        };
        let minus_score = match self.minus_score {
            Some(s) => s,
            None => {
                log::warn!("[TUNER] Missing minus score, skipping update");
                return self.current_params.get_constrained();
            }
        };

        // Compute SPSA gradient estimate
        let score_diff = plus_score - minus_score;
        let perturbation = self.perturbation.as_ref().unwrap();
        let c_k = self.config.spsa_c / (self.iteration as f64 + 1.0).powf(self.config.spsa_gamma);

        let gradient: Vec<f64> = perturbation.iter()
            .map(|&delta| score_diff / (2.0 * c_k * delta))
            .collect();

        if self.config.verbose {
            let grad_norm = gradient.iter().map(|g| g * g).sum::<f64>().sqrt();
            log::info!("[TUNER] SPSA gradient norm: {:.6}, score_diff: {:.6}", grad_norm, score_diff);
        }

        // Adam optimizer update
        self.adam_t += 1;
        let t = self.adam_t as f64;

        // Update biased first moment estimate
        for i in 0..gradient.len() {
            self.adam_m[i] = self.config.adam_beta1 * self.adam_m[i] +
                             (1.0 - self.config.adam_beta1) * gradient[i];
        }

        // Update biased second moment estimate
        for i in 0..gradient.len() {
            self.adam_v[i] = self.config.adam_beta2 * self.adam_v[i] +
                             (1.0 - self.config.adam_beta2) * gradient[i] * gradient[i];
        }

        // Compute bias-corrected moments
        let m_hat: Vec<f64> = self.adam_m.iter()
            .map(|&m| m / (1.0 - self.config.adam_beta1.powf(t)))
            .collect();
        let v_hat: Vec<f64> = self.adam_v.iter()
            .map(|&v| v / (1.0 - self.config.adam_beta2.powf(t)))
            .collect();

        // Compute parameter update with learning rate decay
        let alpha_k = self.config.adam_alpha / (1.0 + self.config.learning_rate_decay * t);
        let param_update: Vec<f64> = m_hat.iter()
            .zip(v_hat.iter())
            .map(|(&m, &v)| alpha_k * m / (v.sqrt() + self.config.adam_epsilon))
            .collect();

        // Apply max change constraint (gradient clipping in parameter space)
        let max_change = param_update.iter().map(|x| x.abs()).fold(0.0f64, f64::max);
        let param_update_clipped: Vec<f64> = if max_change > self.config.max_param_change {
            let scale = self.config.max_param_change / max_change;
            param_update.iter().map(|x| x * scale).collect()
        } else {
            param_update
        };

        // Update parameters
        let mut new_params_vec = self.current_params.to_vec();
        for i in 0..new_params_vec.len() {
            new_params_vec[i] += param_update_clipped[i];
        }

        self.current_params = StrategyTuningParams::from_vec(&new_params_vec);

        // Update tracking
        self.iteration += 1;
        self.episodes_since_update = 0;
        self.last_update_time = Some(std::time::Instant::now());

        // Track score (use average of plus/minus as current estimate)
        let current_score = (plus_score + minus_score) / 2.0;
        self.score_history.push(current_score);

        if current_score > self.best_score {
            self.best_score = current_score;
            self.best_params = self.current_params.clone();

            if self.config.verbose {
                log::info!("[TUNER] New best score: {:.6}", self.best_score);
            }
        }

        // Reset candidate tracking
        self.plus_score = None;
        self.minus_score = None;
        self.candidate_params_plus = None;
        self.candidate_params_minus = None;
        self.perturbation = None;

        if self.config.verbose {
            log::info!("[TUNER] Updated parameters (iteration {}), learning_rate: {:.6}, avg_score: {:.6}",
                self.iteration, alpha_k, current_score);
        }

        self.current_params.get_constrained()
    }

    /// Get current parameters (constrained space)
    pub fn get_current_params(&self) -> StrategyConstrainedParams {
        self.current_params.get_constrained()
    }

    /// Get best parameters found so far
    pub fn get_best_params(&self) -> StrategyConstrainedParams {
        self.best_params.get_constrained()
    }

    /// Increment episode counter
    pub fn on_episode_end(&mut self, score: f64) {
        self.episodes_since_update += 1;
        self.score_history.push(score);

        // Keep history bounded
        if self.score_history.len() > 1000 {
            self.score_history.remove(0);
        }
    }

    /// Get performance tracker reference
    pub fn performance_tracker(&mut self) -> &mut PerformanceTracker {
        &mut self.performance_tracker
    }

    /// Export tuning history for analysis
    pub fn export_history(&self) -> TuningHistory {
        TuningHistory {
            iterations: self.iteration,
            score_history: self.score_history.clone(),
            best_score: self.best_score,
            best_params: self.best_params.get_constrained(),
            current_params: self.current_params.get_constrained(),
        }
    }

    /// Check if currently evaluating candidates
    pub fn is_evaluating_candidates(&self) -> bool {
        self.candidate_params_plus.is_some() || self.candidate_params_minus.is_some()
    }
}

// ============================================================================
// Tuning History Export
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TuningHistory {
    pub iterations: usize,
    pub score_history: Vec<f64>,
    pub best_score: f64,
    pub best_params: StrategyConstrainedParams,
    pub current_params: StrategyConstrainedParams,
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tuner_initialization() {
        let config = TunerConfig::default();
        let params = StrategyTuningParams::default();
        let tuner = MultiObjectiveTuner::new(config, params, 42);

        assert_eq!(tuner.adam_m.len(), 36);
        assert_eq!(tuner.adam_v.len(), 36);
        assert_eq!(tuner.iteration, 0);
    }

    #[test]
    fn test_candidate_generation() {
        let mut config = TunerConfig::default();
        config.enabled = true;
        let params = StrategyTuningParams::default();
        let mut tuner = MultiObjectiveTuner::new(config, params, 42);

        let (plus, minus) = tuner.generate_candidates();

        // Check that candidates differ
        assert_ne!(plus.phi, minus.phi);
        assert_ne!(plus.lambda_base, minus.lambda_base);
    }

    #[test]
    fn test_parameter_update() {
        let mut config = TunerConfig::default();
        config.enabled = true;
        config.verbose = false;
        let params = StrategyTuningParams::default();
        let mut tuner = MultiObjectiveTuner::new(config, params.clone(), 42);

        // Generate candidates
        tuner.generate_candidates();

        // Record scores (plus is better)
        tuner.record_candidate_score(true, 0.8);
        tuner.record_candidate_score(false, 0.6);

        // Update
        let new_params = tuner.update_parameters();

        // Parameters should have changed
        let original = params.get_constrained();
        assert_ne!(new_params.phi, original.phi);
    }

    #[test]
    fn test_should_update_continuous() {
        let mut config = TunerConfig::default();
        config.enabled = true;
        config.mode = "continuous".to_string();
        config.episodes_per_update = 10;

        let params = StrategyTuningParams::default();
        let mut tuner = MultiObjectiveTuner::new(config, params, 42);

        // Should not update initially
        assert!(!tuner.should_update());

        // After 10 episodes, should update
        for _ in 0..10 {
            tuner.on_episode_end(0.5);
        }
        assert!(tuner.should_update());
    }
}
