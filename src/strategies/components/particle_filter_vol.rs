// ============================================================================
// Particle Filter Volatility Model - Stochastic Volatility Estimation
// ============================================================================
//
// This component implements the VolatilityModel trait using a particle filter
// for stochastic volatility estimation. It contains the full ParticleFilterState
// implementation ported from stochastic_volatility.rs.
//
// # Algorithm
//
// The particle filter estimates latent volatility by:
// 1. Maintaining a cloud of N particles, each representing a hypothesis about h_t (log vol²)
// 2. On each price update:
//    - Propagate particles forward via AR(1) dynamics: h_t = μ + φ(h_{t-1} - μ) + η_t
//    - Reweight particles based on likelihood of observed return
//    - Resample particles to avoid degeneracy
// 3. Extract volatility estimate and uncertainty from particle distribution
//
// # Model Equations
//
// State Equation (Log Variance):
//   h_t = mu + phi * (h_{t-1} - mu) + eta_t
//
// Measurement Equation (Log Return):
//   y_t = sqrt(exp(h_t) * dt) * epsilon_t
//
// # Configuration
//
// The model is configured with:
// - num_particles: Number of particles (default: 1000)
// - mu: Mean reversion level for log volatility (default: -9.2)
// - phi: Persistence parameter (default: 0.88, high persistence)
// - sigma_eta: Volatility of volatility (default: 1.2)
// - initial_h: Initial log volatility (default: -9.2, ~100 bps)
//
// # Example
//
// ```rust
// use strategies::components::{VolatilityModel, ParticleFilterVolModel};
//
// let mut vol_model = ParticleFilterVolModel::new_default();
//
// // On each market update
// vol_model.on_market_update(&market_update);
//
// // Get current estimates
// let vol_bps = vol_model.get_volatility_bps();
// let uncertainty = vol_model.get_uncertainty_bps();
// ```

use rand::prelude::*;
use rand_distr::{Distribution, Normal};
use std::time::Instant;
use serde_json::Value;

use crate::strategy::MarketUpdate;
use super::volatility::VolatilityModel;

const SECONDS_PER_YEAR: f64 = 365.25 * 24.0 * 60.0 * 60.0;

/// Represents a single particle in the filter.
#[derive(Debug, Clone, Copy)]
pub struct Particle {
    /// The hidden state: h_t = log(volatility²)
    pub log_vol: f64,
    
    /// Parameter: μ (long-term mean) - only used in Liu-West mode
    pub mu: f64,
    
    /// Parameter: φ (persistence) - only used in Liu-West mode
    pub phi: f64,
    
    /// Parameter: σ_η (vol-of-vol) - only used in Liu-West mode
    pub sigma_eta: f64,
    
    /// How "likely" this particle is, based on observations
    pub weight: f64,
}

/// Manages the state and parameters of the Particle Filter.
#[derive(Debug, Clone)]
pub struct ParticleFilterState {
    /// Collection of all particles representing the distribution
    pub particles: Vec<Particle>,
    
    /// Number of particles used in the filter
    num_particles: usize,

    /// Fixed parameters (used when adaptive mode is disabled)
    fixed_mu: f64,
    fixed_phi: f64,
    fixed_sigma_eta: f64,

    /// Random number generator for noise simulation and resampling
    rng: StdRng,
    
    /// Timestamp of the last update. Used to calculate dt.
    last_update_time: Option<Instant>,
    
    /// Previous mid-price observed. Needed to calculate log returns.
    prev_mid: Option<f64>,
    
    /// Effective Sample Size (ESS). Used to monitor particle degeneracy.
    effective_sample_size: f64,
    
    /// Threshold for ESS below which resampling is triggered
    resampling_threshold: f64,
    
    /// Observation counter
    observation_count: usize,
}

impl ParticleFilterState {
    /// Creates a new Particle Filter with FIXED parameters
    ///
    /// # Arguments
    /// * `num_particles` - The number of particles (e.g., 5000)
    /// * `mu` - Long-term mean of log-variance (FIXED)
    /// * `phi` - Persistence parameter (FIXED)
    /// * `sigma_eta` - Standard deviation of process noise (FIXED)
    /// * `initial_h` - Initial guess for starting log-variance
    /// * `initial_h_std_dev` - Uncertainty around initial guess
    /// * `seed` - Random seed for reproducibility
    pub fn new(
        num_particles: usize,
        mu: f64,
        phi: f64,
        sigma_eta: f64,
        initial_h: f64,
        initial_h_std_dev: f64,
        seed: u64,
    ) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let initial_distribution = Normal::new(initial_h, initial_h_std_dev).unwrap();
        let initial_weight = 1.0 / (num_particles as f64);

        // In fixed mode, all particles share the same parameters
        let particles: Vec<Particle> = (0..num_particles)
            .map(|_| Particle {
                log_vol: initial_distribution.sample(&mut rng),
                mu,       // Fixed
                phi,      // Fixed
                sigma_eta, // Fixed
                weight: initial_weight,
            })
            .collect();

        Self {
            particles,
            num_particles,
            fixed_mu: mu,
            fixed_phi: phi,
            fixed_sigma_eta: sigma_eta,
            rng,
            last_update_time: None,
            prev_mid: None,
            effective_sample_size: num_particles as f64,
            resampling_threshold: (num_particles as f64) * 0.4,
            observation_count: 0,
        }
    }

    /// Updates the filter state with a new observation (mid-price).
    pub fn update(&mut self, current_mid: f64) -> Option<f64> {
        let now = Instant::now();

        // Compute observed return and dt
        let (y_t, dt) = match (self.prev_mid, self.last_update_time) {
            (Some(prev), Some(last_time)) if prev > 0.0 && current_mid > 0.0 => {
                let dt_duration = now.duration_since(last_time);
                let dt_years = dt_duration.as_secs_f64() / SECONDS_PER_YEAR;
                
                if dt_years <= 0.0 {
                    return None;
                }
                
                let log_return = (current_mid / prev).ln();
                if !log_return.is_finite() {
                    self.last_update_time = Some(now);
                    self.prev_mid = Some(current_mid);
                    return None;
                }
                (log_return, dt_years)
            }
            _ => {
                self.last_update_time = Some(now);
                self.prev_mid = Some(current_mid);
                return None;
            }
        };

        // Core filter steps (standard update for fixed parameters)
        self.standard_update(dt);
        self.weight_step(y_t, dt);
        self.normalize_weights();
        self.resample_if_needed();

        self.last_update_time = Some(now);
        self.prev_mid = Some(current_mid);
        self.observation_count += 1;

        Some(self.estimate_volatility_bps())
    }

    /// Standard predict step (fixed parameters)
    fn standard_update(&mut self, dt: f64) {
        if dt <= 0.0 { return; }
        
        let process_noise_std_dev = self.fixed_sigma_eta * dt.sqrt();
        let noise_dist = Normal::new(0.0, process_noise_std_dev).unwrap();

        for particle in &mut self.particles {
            let noise = noise_dist.sample(&mut self.rng);
            particle.log_vol = self.fixed_mu + self.fixed_phi * (particle.log_vol - self.fixed_mu) + noise;
        }
    }

    /// Weight step (measurement update) - LOG-SPACE for numerical stability
    fn weight_step(&mut self, y_t: f64, dt: f64) {
        if dt <= 0.0 { return; }
        
        let mut log_weights: Vec<f64> = Vec::with_capacity(self.num_particles);
        let mut max_log_weight = f64::NEG_INFINITY;

        for particle in &self.particles {
            let variance = particle.log_vol.exp() * dt;
            
            if variance <= 0.0 || !variance.is_finite() {
                log_weights.push(f64::NEG_INFINITY);
                continue;
            }
            
            let expected_std_dev = variance.sqrt();
            
            if expected_std_dev < 1e-12 {
                log_weights.push(if y_t.abs() < 1e-12 { 0.0 } else { f64::NEG_INFINITY });
            } else {
                // Log-likelihood: -0.5 * [(y/σ)² + ln(2π) + 2ln(σ)]
                let z = y_t / expected_std_dev;
                let log_likelihood = -0.5 * (
                    z * z + 
                    (2.0 * std::f64::consts::PI).ln() + 
                    2.0 * expected_std_dev.ln()
                );
                
                let current_log_weight = if particle.weight > 0.0 {
                    particle.weight.ln()
                } else {
                    f64::NEG_INFINITY
                };
                
                let log_weight = current_log_weight + log_likelihood;
                log_weights.push(log_weight);
                
                if log_weight.is_finite() {
                    max_log_weight = max_log_weight.max(log_weight);
                }
            }
        }

        // Normalize using log-sum-exp trick
        let log_sum_exp = max_log_weight + log_weights.iter()
            .filter(|&&lw| lw.is_finite())
            .map(|&lw| (lw - max_log_weight).exp())
            .sum::<f64>()
            .ln();

        // Update particle weights
        let mut any_valid = false;
        for (i, particle) in self.particles.iter_mut().enumerate() {
            let log_weight = log_weights[i];
            if log_weight.is_finite() && log_sum_exp.is_finite() {
                particle.weight = (log_weight - log_sum_exp).exp();
                any_valid = any_valid || particle.weight > 1e-300;
            } else {
                particle.weight = 0.0;
            }
        }

        // Fallback if all weights collapsed
        if !any_valid {
            let uniform_weight = 1.0 / (self.num_particles as f64);
            for p in &mut self.particles {
                p.weight = uniform_weight;
            }
            self.effective_sample_size = self.num_particles as f64;
        }
    }

    /// Normalize particle weights so they sum to 1.0 and compute ESS
    fn normalize_weights(&mut self) {
        let total_weight: f64 = self.particles.iter().map(|p| p.weight).sum();

        if total_weight <= f64::EPSILON {
            let uniform_weight = 1.0 / (self.num_particles as f64);
            for p in &mut self.particles {
                p.weight = uniform_weight;
            }
            self.effective_sample_size = self.num_particles as f64;
            return;
        }

        // Normalize
        for p in &mut self.particles {
            p.weight /= total_weight;
        }

        // Compute ESS = 1 / sum(w_i²)
        let sum_sq_weights: f64 = self.particles.iter()
            .map(|p| p.weight * p.weight)
            .sum();
        
        self.effective_sample_size = if sum_sq_weights > 0.0 {
            1.0 / sum_sq_weights
        } else {
            0.0
        };
    }

    /// Resample if ESS falls below threshold (Systematic Resampling)
    fn resample_if_needed(&mut self) {
        if self.effective_sample_size < self.resampling_threshold {
            self.systematic_resample();
            
            // Reset weights to uniform
            let uniform_weight = 1.0 / (self.num_particles as f64);
            for p in &mut self.particles {
                p.weight = uniform_weight;
            }
            
            self.effective_sample_size = self.num_particles as f64;
        }
    }

    /// Systematic resampling (low-variance)
    fn systematic_resample(&mut self) {
        let n = self.num_particles;
        let mut new_particles: Vec<Particle> = Vec::with_capacity(n);
        
        // Cumulative sum of weights
        let mut cumulative_weights = Vec::with_capacity(n);
        let mut sum = 0.0;
        for p in &self.particles {
            sum += p.weight;
            cumulative_weights.push(sum);
        }
        
        // Systematic resampling
        let u: f64 = self.rng.gen_range(0.0..1.0 / n as f64);
        let mut i = 0;
        
        for j in 0..n {
            let threshold = u + (j as f64) / (n as f64);
            
            while i < n - 1 && cumulative_weights[i] < threshold {
                i += 1;
            }
            
            new_particles.push(self.particles[i]);
        }
        
        self.particles = new_particles;
    }

    /// Estimate volatility in basis points (weighted mean)
    pub fn estimate_volatility_bps(&self) -> f64 {
        let weighted_mean_log_vol: f64 = self.particles.iter()
            .map(|p| p.log_vol * p.weight)
            .sum();
        
        let volatility_annualized = weighted_mean_log_vol.exp().sqrt();
        volatility_annualized * 10000.0
    }

    /// Get standard deviation of current volatility estimate in BPS
    pub fn get_volatility_std_dev_bps(&self) -> f64 {
        let h_particles: Vec<f64> = self.particles.iter()
            .map(|p| p.log_vol)
            .collect();

        if h_particles.len() < 2 { 
            return 0.0; 
        }
        
        let h_mean: f64 = h_particles.iter().sum::<f64>() / h_particles.len() as f64;
        let h_variance: f64 = h_particles.iter()
            .map(|&h| (h - h_mean).powi(2))
            .sum::<f64>() / (h_particles.len() - 1) as f64;
        let h_std = h_variance.sqrt();

        // Delta method approximation
        let sigma_mean_approx = (h_mean / 2.0).exp();
        let sigma_std_approx = (0.5 * sigma_mean_approx) * h_std;

        sigma_std_approx * 10000.0
    }
}

/// Particle filter-based volatility model implementation.
///
/// This component wraps a ParticleFilterState and provides the VolatilityModel
/// interface for use in modular strategies.
pub struct ParticleFilterVolModel {
    /// The underlying particle filter state
    pf_state: ParticleFilterState,
}

impl ParticleFilterVolModel {
    /// Create a new particle filter volatility model with default parameters.
    ///
    /// Default configuration:
    /// - 1000 particles
    /// - mu = -9.2 (mean reversion level)
    /// - phi = 0.88 (high persistence)
    /// - sigma_eta = 1.2 (volatility of volatility)
    /// - initial_h = -9.2 (log vol² ≈ 100 bps)
    /// - seed = 12345 (for reproducibility)
    pub fn new_default() -> Self {
        Self::new(1000, -9.2, 0.88, 1.2, -9.2, 0.5, 12345)
    }

    /// Create a new particle filter volatility model with custom parameters.
    pub fn new(
        num_particles: usize,
        mu: f64,
        phi: f64,
        sigma_eta: f64,
        initial_h: f64,
        initial_h_std_dev: f64,
        seed: u64,
    ) -> Self {
        let pf_state = ParticleFilterState::new(
            num_particles,
            mu,
            phi,
            sigma_eta,
            initial_h,
            initial_h_std_dev,
            seed,
        );

        Self { pf_state }
    }

    /// Create from JSON config
    pub fn from_json(config: &Value) -> Self {
        let num_particles = config.get("num_particles")
            .and_then(|v| v.as_u64())
            .unwrap_or(1000) as usize;

        let mu = config.get("mu")
            .and_then(|v| v.as_f64())
            .unwrap_or(-9.2);

        let phi = config.get("phi")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.88);

        let sigma_eta = config.get("sigma_eta")
            .and_then(|v| v.as_f64())
            .unwrap_or(1.2);

        let initial_h = config.get("initial_h")
            .and_then(|v| v.as_f64())
            .unwrap_or(-9.2);

        let initial_h_std_dev = config.get("initial_h_std_dev")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.5);

        let seed = config.get("seed")
            .and_then(|v| v.as_u64())
            .unwrap_or(12345);

        Self::new(num_particles, mu, phi, sigma_eta, initial_h, initial_h_std_dev, seed)
    }
}

impl VolatilityModel for ParticleFilterVolModel {
    fn on_market_update(&mut self, update: &MarketUpdate) {
        // Update particle filter if we have a mid-price
        if let Some(mid_price) = update.mid_price {
            self.pf_state.update(mid_price);
        }
    }

    fn get_volatility_bps(&self) -> f64 {
        self.pf_state.estimate_volatility_bps()
    }

    fn get_uncertainty_bps(&self) -> f64 {
        self.pf_state.get_volatility_std_dev_bps()
    }
}
