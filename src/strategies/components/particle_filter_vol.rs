// ============================================================================
// Particle Filter Volatility Model - Stochastic Volatility Estimation
// ============================================================================
//
// This component implements the VolatilityModel trait using a particle filter
// for stochastic volatility estimation. It wraps the ParticleFilterState from
// the stochastic_volatility module.
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
// # Configuration
//
// The model is configured with:
// - num_particles: Number of particles (default: 1000)
// - mu: Mean reversion level for log volatility (default: 0.0)
// - phi: Persistence parameter (default: 0.98, high persistence)
// - sigma_eta: Volatility of volatility (default: 0.1)
// - initial_h: Initial log volatility (default: -5.0, ~100 bps)
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

use std::sync::Arc;
use parking_lot::RwLock;
use serde_json::Value;

use crate::strategy::MarketUpdate;
use crate::ParticleFilterState;
use super::volatility::VolatilityModel;

/// Particle filter-based volatility model implementation.
///
/// This component wraps a ParticleFilterState and provides the VolatilityModel
/// interface for use in modular strategies.
pub struct ParticleFilterVolModel {
    /// The underlying particle filter state
    pf_state: Arc<RwLock<ParticleFilterState>>,

    /// Cached volatility estimate (updated on each market update)
    cached_volatility_bps: f64,

    /// Cached uncertainty estimate (std dev of particle distribution)
    cached_uncertainty_bps: f64,
}

impl ParticleFilterVolModel {
    /// Create a new particle filter volatility model with default parameters.
    ///
    /// Default configuration:
    /// - 1000 particles
    /// - mu = 0.0 (mean reversion level)
    /// - phi = 0.98 (high persistence)
    /// - sigma_eta = 0.1 (volatility of volatility)
    /// - initial_h = -5.0 (log vol² ≈ 100 bps)
    /// - seed = 12345 (for reproducibility)
    pub fn new_default() -> Self {
        Self::new(1000, 0.0, 0.98, 0.1, -5.0, 0.5, 12345)
    }

    /// Create a new particle filter volatility model with custom parameters.
    ///
    /// # Arguments
    /// - `num_particles`: Number of particles (more = more accurate but slower)
    /// - `mu`: Mean reversion level for log volatility
    /// - `phi`: Persistence parameter (0 = no persistence, 1 = random walk)
    /// - `sigma_eta`: Volatility of volatility
    /// - `initial_h`: Initial log volatility squared
    /// - `initial_h_std_dev`: Initial std dev of particle distribution
    /// - `seed`: Random seed for reproducibility
    pub fn new(
        num_particles: usize,
        mu: f64,
        phi: f64,
        sigma_eta: f64,
        initial_h: f64,
        initial_h_std_dev: f64,
        seed: u64,
    ) -> Self {
        let pf_state = Arc::new(RwLock::new(ParticleFilterState::new(
            num_particles,
            mu,
            phi,
            sigma_eta,
            initial_h,
            initial_h_std_dev,
            seed,
        )));

        Self {
            pf_state,
            cached_volatility_bps: 100.0,  // Default to 100 bps
            cached_uncertainty_bps: 10.0,   // Default to 10 bps uncertainty
        }
    }

    /// Create a new particle filter volatility model from JSON config.
    ///
    /// Expected JSON structure:
    /// ```json
    /// {
    ///   "num_particles": 1000,
    ///   "mu": 0.0,
    ///   "phi": 0.98,
    ///   "sigma_eta": 0.1,
    ///   "initial_h": -5.0,
    ///   "initial_h_std_dev": 0.5,
    ///   "seed": 12345
    /// }
    /// ```
    pub fn from_json(config: &Value) -> Self {
        let num_particles = config.get("num_particles")
            .and_then(|v| v.as_u64())
            .unwrap_or(1000) as usize;

        let mu = config.get("mu")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);

        let phi = config.get("phi")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.98);

        let sigma_eta = config.get("sigma_eta")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.1);

        let initial_h = config.get("initial_h")
            .and_then(|v| v.as_f64())
            .unwrap_or(-5.0);

        let initial_h_std_dev = config.get("initial_h_std_dev")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.5);

        let seed = config.get("seed")
            .and_then(|v| v.as_u64())
            .unwrap_or(12345);

        Self::new(num_particles, mu, phi, sigma_eta, initial_h, initial_h_std_dev, seed)
    }

    /// Get reference to the underlying particle filter (for advanced usage).
    pub fn pf_state(&self) -> Arc<RwLock<ParticleFilterState>> {
        Arc::clone(&self.pf_state)
    }

    /// Update cached estimates from particle filter statistics.
    ///
    /// This method extracts the current volatility estimate and uncertainty
    /// from the particle distribution and caches them for fast access.
    fn update_cached_estimates(&mut self) {
        let pf = self.pf_state.read();

        // Get particle statistics
        let particles = &pf.particles;

        if particles.is_empty() {
            // Particle filter not initialized yet, keep defaults
            return;
        }

        // Compute weighted mean of exp(h_t/2) = σ_t
        let mut weighted_sum = 0.0;
        let mut weight_sum = 0.0;

        for particle in particles.iter() {
            // h is log(σ²), so σ = exp(h/2)
            let h = particle.log_vol;
            let w = particle.weight;
            let sigma = (h / 2.0).exp();
            weighted_sum += sigma * w;
            weight_sum += w;
        }

        let mean_sigma = if weight_sum > 0.0 {
            weighted_sum / weight_sum
        } else {
            0.01  // 1% = 100 bps
        };

        // Convert to basis points (1% = 100 bps)
        self.cached_volatility_bps = mean_sigma * 10000.0;

        // Compute weighted std dev for uncertainty estimate
        let mut weighted_var = 0.0;
        for particle in particles.iter() {
            let h = particle.log_vol;
            let w = particle.weight;
            let sigma = (h / 2.0).exp();
            let diff = sigma - mean_sigma;
            weighted_var += diff * diff * w;
        }

        let std_sigma = if weight_sum > 0.0 {
            (weighted_var / weight_sum).sqrt()
        } else {
            0.001  // 0.1% = 10 bps
        };

        self.cached_uncertainty_bps = std_sigma * 10000.0;
    }
}

impl VolatilityModel for ParticleFilterVolModel {
    fn on_market_update(&mut self, update: &MarketUpdate) {
        // Update particle filter if we have a mid-price
        if let Some(mid_price) = update.mid_price {
            let mut pf = self.pf_state.write();
            pf.update(mid_price);
            drop(pf);

            // Update cached estimates
            self.update_cached_estimates();
        }
    }

    fn get_volatility_bps(&self) -> f64 {
        self.cached_volatility_bps
    }

    fn get_uncertainty_bps(&self) -> f64 {
        self.cached_uncertainty_bps
    }
}
