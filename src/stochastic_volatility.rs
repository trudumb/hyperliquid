//! Stochastic Volatility (SV) Particle Filter Implementation
//!
//! Estimates latent volatility using a sequential Monte Carlo method (Particle Filter).
//! Based on the standard SV model:
//!   h_t = mu + phi * (h_{t-1} - mu) + eta_t    (State Equation - Log Variance)
//!   y_t = sqrt(exp(h_t) * dt) * epsilon_t      (Measurement Equation - Log Return)

use rand::prelude::*;
use rand_distr::{Distribution, Normal, Uniform};
use statrs::distribution::Normal as StatrsNormal; // Using statrs for PDF calculation
use std::time::{Duration, Instant};

const SECONDS_PER_YEAR: f64 = 365.25 * 24.0 * 60.0 * 60.0;

/// Represents a single particle in the filter.
/// Each particle is a hypothesis about the current latent log-volatility state.
#[derive(Debug, Clone, Copy)]
pub struct Particle {
    /// The hidden state: h_t = log(volatility²)
    pub log_vol: f64,
    /// How "likely" this particle is, based on observations.
    pub weight: f64,
}

/// Manages the state and parameters of the Particle Filter.
#[derive(Debug, Clone)]
pub struct ParticleFilterState {
    /// Collection of all particles representing the distribution of the latent state.
    pub particles: Vec<Particle>,
    /// Number of particles used in the filter.
    num_particles: usize,

    // --- Model Parameters (State Equation) ---
    /// mu: Long-term average log-variance E[h_t].
    mu: f64,
    /// phi: Persistence parameter (0 < phi < 1). Controls mean reversion speed.
    phi: f64,
    /// sigma_eta: Standard deviation of the process noise (eta_t). Controls volatility of volatility.
    sigma_eta: f64,

    // --- Filter State ---
    /// Random number generator for noise simulation and resampling.
    rng: StdRng,
    /// Timestamp of the last update. Used to calculate dt.
    last_update_time: Option<Instant>,
    /// Previous mid-price observed. Needed to calculate log returns.
    prev_mid: Option<f64>,
    /// Effective Sample Size (ESS). Used to monitor particle degeneracy.
    effective_sample_size: f64,
    /// Threshold for ESS below which resampling is triggered (e.g., N/2).
    resampling_threshold: f64,
}

impl ParticleFilterState {
    /// Creates a new Particle Filter instance.
    ///
    /// # Arguments
    /// * `num_particles` - The number of particles (e.g., 5000). More particles = more accuracy, but slower.
    /// * `mu` - Long-term mean of log-variance (h_t).
    /// * `phi` - Persistence parameter (e.g., 0.98).
    /// * `sigma_eta` - Standard deviation of the process noise (e.g., 0.15).
    /// * `initial_h` - The initial guess for the starting log-variance.
    /// * `initial_h_std_dev` - The standard deviation around the initial guess for particle initialization.
    /// * `seed` - Seed for the random number generator for reproducibility.
    pub fn new(
        num_particles: usize,
        mu: f64,
        phi: f64,
        sigma_eta: f64,
        initial_h: f64, // Provide an initial estimate for h_0
        initial_h_std_dev: f64, // Uncertainty around h_0
        seed: u64,
    ) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let initial_distribution = Normal::new(initial_h, initial_h_std_dev).unwrap();
        let initial_weight = 1.0 / (num_particles as f64);

        let particles: Vec<Particle> = (0..num_particles)
            .map(|_| Particle {
                log_vol: initial_distribution.sample(&mut rng),
                weight: initial_weight,
            })
            .collect();

        Self {
            particles,
            num_particles,
            mu,
            phi,
            sigma_eta,
            rng,
            last_update_time: None,
            prev_mid: None,
            effective_sample_size: num_particles as f64, // Initially all particles are equally likely
            resampling_threshold: (num_particles as f64) / 2.0,
        }
    }

    /// Updates the filter state with a new observation (mid-price).
    /// Performs predict, weight, normalize, and resample steps.
    /// Returns the new volatility estimate in BPS, or None if update couldn't be performed.
    pub fn update(&mut self, current_mid: f64) -> Option<f64> {
        let now = Instant::now();

        // --- Step 3.1: Compute Observed Return and dt ---
        let (y_t, dt) = match (self.prev_mid, self.last_update_time) {
            (Some(prev), Some(last_time)) if prev > 0.0 && current_mid > 0.0 => {
                let dt_duration = now.duration_since(last_time);
                // dt in years
                let dt_years = dt_duration.as_secs_f64() / SECONDS_PER_YEAR;

                if dt_years <= 0.0 {
                    // Avoid division by zero or sqrt of negative if time hasn't passed
                    return None; // Skip update if dt is invalid
                }

                // Log return: y_t = ln(P_t / P_{t-1})
                let log_return = (current_mid / prev).ln();

                // Ensure log_return is finite (can be NaN/inf if prices are identical or zero)
                if !log_return.is_finite() {
                    log::debug!(
                        "Skipping SV filter update: Non-finite log return (prev={}, cur={})",
                        prev,
                        current_mid
                    );
                    // Update time and price, but skip the filter steps
                    self.last_update_time = Some(now);
                    self.prev_mid = Some(current_mid);
                    return None;
                }
                (log_return, dt_years)
            }
            _ => {
                // First observation or invalid previous price
                self.last_update_time = Some(now);
                self.prev_mid = Some(current_mid);
                return None; // Cannot compute return yet
            }
        };

        // --- Core Filter Steps ---
        self.predict_step(dt);
        self.weight_step(y_t, dt);
        self.normalize_weights();
        self.resample_if_needed();

        // --- Update state for next iteration ---
        self.last_update_time = Some(now);
        self.prev_mid = Some(current_mid);

        // --- Step 4.1: Extract Volatility Estimate ---
        Some(self.estimate_volatility_bps())
    }

    /// Predict step (Time Update): Move particles forward using the state equation.
    fn predict_step(&mut self, dt: f64) {
        if dt <= 0.0 { return; } // Should not happen due to check in update, but safety first
        let process_noise_std_dev = self.sigma_eta * dt.sqrt();
        let noise_dist = Normal::new(0.0, process_noise_std_dev).unwrap();

        for particle in &mut self.particles {
            let noise = noise_dist.sample(&mut self.rng);
            // State Equation: h_t = mu + phi * (h_{t-1} - mu) + eta_t
            particle.log_vol = self.mu + self.phi * (particle.log_vol - self.mu) + noise;
        }
    }

    /// Weight step (Measurement Update): Update particle weights based on the observation y_t.
    fn weight_step(&mut self, y_t: f64, dt: f64) {
        if dt <= 0.0 { return; }
        let mut total_likelihood = 0.0; // Keep track for numerical stability

        for particle in &mut self.particles {
            // Calculate expected standard deviation for this particle's hypothesized volatility
            // expected_std_dev = sqrt(Var(y_t | h_t)) = sqrt(exp(h_t) * dt)
            let variance = particle.log_vol.exp() * dt;
            if variance < 0.0 {
                // Safety check for numerical issues
                particle.weight = 0.0;
                continue;
            }
            let expected_std_dev = variance.sqrt();

            if expected_std_dev < 1e-12 {
                // Avoid division by zero in PDF if std dev is essentially zero
                // Assign very low likelihood unless observation is also zero
                 if y_t.abs() < 1e-12 {
                    // Particle predicted zero vol, observation was zero - high likelihood (assign 1.0 for simplicity)
                     particle.weight *= 1.0; // Or a very large number relative to others
                 } else {
                     particle.weight = 0.0; // Particle is very unlikely
                 }
            } else {
                // Calculate likelihood using Gaussian PDF
                // likelihood = P(y_t | h_t) = NormalPDF(y_t, mean=0, std=expected_std_dev)
                let normal_dist = StatrsNormal::new(0.0, expected_std_dev).unwrap();
                let likelihood = normal_dist.pdf(y_t);

                // Update weight (multiply by likelihood)
                // Add small epsilon to prevent weights becoming exactly zero due to underflow
                particle.weight = particle.weight * likelihood + f64::EPSILON;
            }
            total_likelihood += particle.weight;
        }

        // Handle case where all likelihoods might underflow to near zero
        if total_likelihood < 1e-100 {
            log::warn!("SV Filter: Total likelihood near zero. Resetting weights uniformly.");
            let uniform_weight = 1.0 / (self.num_particles as f64);
            for p in &mut self.particles {
                p.weight = uniform_weight;
            }
            self.effective_sample_size = self.num_particles as f64;
        }
    }


    /// Normalize particle weights so they sum to 1.0. Also calculates ESS.
    fn normalize_weights(&mut self) {
        let total_weight: f64 = self.particles.iter().map(|p| p.weight).sum();

        if total_weight <= f64::EPSILON {
            // If total weight is effectively zero (e.g., all likelihoods were tiny),
            // reset to uniform weights to prevent division by zero and allow recovery.
            log::warn!("SV Filter: Total weight is zero or negative. Resetting weights uniformly.");
            let uniform_weight = 1.0 / (self.num_particles as f64);
            let mut sum_sq_weights = 0.0;
            for p in &mut self.particles {
                p.weight = uniform_weight;
                sum_sq_weights += uniform_weight * uniform_weight;
            }
             self.effective_sample_size = if sum_sq_weights > f64::EPSILON { 1.0 / sum_sq_weights } else { 0.0 };

        } else {
            let mut sum_sq_weights = 0.0;
            for particle in &mut self.particles {
                particle.weight /= total_weight;
                sum_sq_weights += particle.weight * particle.weight;
            }
            // Calculate Effective Sample Size (ESS)
            self.effective_sample_size = if sum_sq_weights > f64::EPSILON { 1.0 / sum_sq_weights } else { 0.0 };
        }
    }


    /// Resamples particles if ESS drops below the threshold, using systematic resampling.
    fn resample_if_needed(&mut self) {
        // --- Step 3.5: Check for Degeneracy and Resample ---
        if self.effective_sample_size < self.resampling_threshold {
            log::debug!(
                "Resampling triggered: ESS = {:.2} < Threshold = {:.2}",
                self.effective_sample_size,
                self.resampling_threshold
            );
            self.systematic_resample();
        }
    }

    /// Performs systematic resampling.
    fn systematic_resample(&mut self) {
        let n = self.num_particles as f64;
        let uniform_weight = 1.0 / n;
        let mut new_particles = Vec::with_capacity(self.num_particles);
        let mut cumulative_weight = 0.0;

        // Compute cumulative weights
        let mut cumulative_weights: Vec<f64> = Vec::with_capacity(self.num_particles);
        for particle in &self.particles {
            cumulative_weight += particle.weight;
            cumulative_weights.push(cumulative_weight);
        }
        // Ensure the last element is exactly 1.0 due to potential floating point inaccuracies
        if let Some(last_weight) = cumulative_weights.last_mut() {
            *last_weight = 1.0;
        }


        // Generate starting point for systematic sampling
        let start = self.rng.gen::<f64>() / n;
        let mut current_cumulative_weight_idx = 0;

        // Iterate through the N points
        for i in 0..self.num_particles {
            let target_weight = start + (i as f64) / n;

            // Find the particle corresponding to the target weight
            while cumulative_weights[current_cumulative_weight_idx] < target_weight {
                current_cumulative_weight_idx += 1;
                // Boundary check (should theoretically not be needed if last weight is 1.0)
                 if current_cumulative_weight_idx >= self.num_particles {
                     current_cumulative_weight_idx = self.num_particles - 1;
                     break;
                 }
            }

            // Add the selected particle to the new set
            new_particles.push(Particle {
                log_vol: self.particles[current_cumulative_weight_idx].log_vol,
                weight: uniform_weight, // Reset weight after resampling
            });
        }

        self.particles = new_particles;
        self.effective_sample_size = n; // ESS is reset to N after resampling
    }

    /// Estimates the current annualized volatility in basis points (BPS).
    /// Calculates the weighted average of particle log-volatilities.
    pub fn estimate_volatility_bps(&self) -> f64 {
        // --- Step 4.1: Compute Weighted Average ---
        let h_estimate: f64 = self
            .particles
            .iter()
            .map(|p| p.log_vol * p.weight)
            .sum();

        // Convert log-variance (h_t) to annualized standard deviation (volatility)
        // volatility = sqrt(variance) = sqrt(exp(h_t))
        let volatility_annualized = h_estimate.exp().sqrt();

        // Convert annualized volatility to basis points
        volatility_annualized * 10000.0
    }

     /// Computes the volatility estimate for the q-th percentile (e.g., q=0.05 for 5th percentile).
     pub fn estimate_volatility_percentile_bps(&self, q: f64) -> f64 {
        if !(0.0..=1.0).contains(&q) {
            panic!("Percentile q must be between 0.0 and 1.0");
        }

        // Sort particles by log_vol - need a mutable copy or sort indices
        let mut indexed_particles: Vec<(usize, &Particle)> = self.particles.iter().enumerate().collect();
        indexed_particles.sort_by(|a, b| a.1.log_vol.partial_cmp(&b.1.log_vol).unwrap_or(std::cmp::Ordering::Equal));

        let mut cumulative_weight = 0.0;
        let mut log_vol_at_percentile = self.particles[0].log_vol; // Default to first particle

        for (_, particle) in indexed_particles {
            cumulative_weight += particle.weight;
            if cumulative_weight >= q {
                log_vol_at_percentile = particle.log_vol;
                break;
            }
        }

        let volatility_annualized = log_vol_at_percentile.exp().sqrt();
        volatility_annualized * 10000.0
    }

    /// Gets the current Effective Sample Size (ESS).
    pub fn get_ess(&self) -> f64 {
        self.effective_sample_size
    }
}


// --- Helper Functions ---

/// Gaussian Probability Density Function (PDF)
/// This version handles potential zero std dev more gracefully.
#[allow(dead_code)] // Keep for potential direct use, although statrs is preferred
fn pdf_normal(x: f64, mean: f64, std_dev: f64) -> f64 {
    if std_dev <= f64::EPSILON {
        // If std dev is effectively zero, PDF is infinite at the mean, zero elsewhere
        if (x - mean).abs() < f64::EPSILON {
            // Represent "infinity" with a very large number
            // Note: This isn't strictly correct mathematically but useful for likelihood updates
             1.0 / (f64::EPSILON * (2.0 * std::f64::consts::PI).sqrt()) // Avoid true infinity
        } else {
            0.0
        }
    } else {
        let variance = std_dev * std_dev;
        let exponent = -((x - mean) * (x - mean)) / (2.0 * variance);
        let coefficient = 1.0 / (std_dev * (2.0 * std::f64::consts::PI).sqrt());
        coefficient * exponent.exp()
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq; // For floating point comparisons

    // Basic test parameters
    const TEST_NUM_PARTICLES: usize = 1000;
    const TEST_MU: f64 = -9.2; // ~ln( (0.01)^2 ), annualized vol of 100 bps
    const TEST_PHI: f64 = 0.95;
    const TEST_SIGMA_ETA: f64 = 0.1;
    const TEST_INITIAL_H: f64 = -9.2;
    const TEST_INITIAL_H_STD_DEV: f64 = 0.2;
    const TEST_SEED: u64 = 42;

    fn create_test_filter() -> ParticleFilterState {
        ParticleFilterState::new(
            TEST_NUM_PARTICLES,
            TEST_MU,
            TEST_PHI,
            TEST_SIGMA_ETA,
            TEST_INITIAL_H,
            TEST_INITIAL_H_STD_DEV,
            TEST_SEED,
        )
    }

    #[test]
    fn test_filter_initialization() {
        let filter = create_test_filter();
        assert_eq!(filter.particles.len(), TEST_NUM_PARTICLES);
        assert_relative_eq!(filter.get_ess(), TEST_NUM_PARTICLES as f64, epsilon = 1e-9);

        // Check initial weights are uniform
        let expected_weight = 1.0 / (TEST_NUM_PARTICLES as f64);
        for p in &filter.particles {
            assert_relative_eq!(p.weight, expected_weight, epsilon = 1e-9);
        }

        // Check initial log_vols are roughly centered around initial_h
        let mean_log_vol: f64 = filter.particles.iter().map(|p| p.log_vol).sum::<f64>() / (TEST_NUM_PARTICLES as f64);
        assert_relative_eq!(mean_log_vol, TEST_INITIAL_H, epsilon = 0.1); // Allow some deviation due to sampling
    }

     #[test]
    fn test_first_update_no_return() {
        let mut filter = create_test_filter();
        let estimate = filter.update(100.0); // First price
        assert!(estimate.is_none()); // Cannot estimate on first update
        assert!(filter.prev_mid.is_some());
        assert_eq!(filter.prev_mid.unwrap(), 100.0);
        assert!(filter.last_update_time.is_some());
    }

    #[test]
    fn test_second_update_calculates_estimate() {
        let mut filter = create_test_filter();
        filter.update(100.0); // Set initial price and time
        std::thread::sleep(Duration::from_millis(10)); // Ensure dt > 0
        let estimate = filter.update(100.1); // Second price
        assert!(estimate.is_some()); // Should calculate an estimate now
        let vol_bps = estimate.unwrap();
        assert!(vol_bps > 0.0); // Volatility should be positive
        println!("Test Vol Estimate (BPS): {}", vol_bps); // Print for inspection

        // Check ESS might have decreased slightly but shouldn't trigger resampling yet
        assert!(filter.get_ess() <= TEST_NUM_PARTICLES as f64);
        assert!(filter.get_ess() > filter.resampling_threshold); // Assuming small price move
    }

    #[test]
    fn test_predict_step_mean_reversion() {
        let mut filter = create_test_filter();
        // Set all particles far above the mean
        for p in &mut filter.particles {
            p.log_vol = TEST_MU + 2.0;
        }

        // Predict with a small dt
        let dt = 1.0 / SECONDS_PER_YEAR; // 1 second in year fraction
        filter.predict_step(dt);

        // Check if particles moved towards the mean mu
        let mean_log_vol_after: f64 = filter.particles.iter().map(|p| p.log_vol).sum::<f64>() / (TEST_NUM_PARTICLES as f64);
        // Expected value after one step: mu + phi*(start_log_vol - mu)
        let expected_mean = TEST_MU + TEST_PHI * ( (TEST_MU + 2.0) - TEST_MU);
        assert!(mean_log_vol_after < TEST_MU + 2.0); // Should have decreased
        assert_relative_eq!(mean_log_vol_after, expected_mean, epsilon = 0.1); // Check it's close to expected mean reversion
    }


    #[test]
    fn test_weight_step_likelihood() {
         let mut filter = create_test_filter();
         // Create two particles: one likely, one unlikely given the observation
         filter.particles = vec![
             Particle { log_vol: -9.2, weight: 0.5 }, // Corresponds to ~100bps annual vol
             Particle { log_vol: -13.8, weight: 0.5 } // Corresponds to ~1bps annual vol
         ];
         filter.num_particles = 2;

         // Simulate a large return (e.g., 5 bps over 1 second)
         let y_t = 0.0005; // 5 bps log return
         let dt = 1.0 / SECONDS_PER_YEAR;

         filter.weight_step(y_t, dt);

         // The particle with higher volatility (-9.2) should have a much higher weight now
         let weight_high_vol = filter.particles[0].weight;
         let weight_low_vol = filter.particles[1].weight;

         println!("Weight High Vol: {}, Weight Low Vol: {}", weight_high_vol, weight_low_vol);
         assert!(weight_high_vol > weight_low_vol * 10.0); // Expect high vol particle to be significantly more likely
    }


    #[test]
    fn test_normalization_and_ess() {
        let mut filter = create_test_filter();
        // Assign arbitrary weights
        filter.particles[0].weight = 0.8;
        filter.particles[1].weight = 0.1;
        for i in 2..filter.num_particles {
            filter.particles[i].weight = 0.1 / (filter.num_particles - 2) as f64;
        }

        filter.normalize_weights();

        // Check weights sum to 1
        let total_weight: f64 = filter.particles.iter().map(|p| p.weight).sum();
        assert_relative_eq!(total_weight, 1.0, epsilon = 1e-9);

        // Check ESS calculation (should be low due to concentrated weight)
        // ESS = 1 / sum(w_i^2)
        let sum_sq_weights: f64 = filter.particles.iter().map(|p| p.weight.powi(2)).sum();
        let expected_ess = 1.0 / sum_sq_weights;
        assert_relative_eq!(filter.get_ess(), expected_ess, epsilon = 1e-9);
        assert!(filter.get_ess() < filter.num_particles as f64);
        println!("Test ESS: {}", filter.get_ess());
    }

    #[test]
    fn test_resampling_trigger_and_reset() {
        let mut filter = create_test_filter();
        filter.resampling_threshold = (TEST_NUM_PARTICLES as f64) * 0.8; // Set low threshold for testing

        // Artificially concentrate weights to trigger resampling
        let high_weight = 0.99;
        let low_weight = (1.0 - high_weight) / ((TEST_NUM_PARTICLES - 1) as f64);
        filter.particles[0].weight = high_weight;
        for i in 1..TEST_NUM_PARTICLES {
            filter.particles[i].weight = low_weight;
        }

        // Manually calculate ESS and check if it's below threshold
        let sum_sq_weights: f64 = filter.particles.iter().map(|p| p.weight.powi(2)).sum();
        filter.effective_sample_size = 1.0 / sum_sq_weights;
        assert!(filter.effective_sample_size < filter.resampling_threshold);

        // Perform resampling
        filter.resample_if_needed();

        // Check ESS is reset to N
        assert_relative_eq!(filter.get_ess(), TEST_NUM_PARTICLES as f64, epsilon = 1e-9);

        // Check weights are uniform again
        let expected_weight = 1.0 / (TEST_NUM_PARTICLES as f64);
        for p in &filter.particles {
            assert_relative_eq!(p.weight, expected_weight, epsilon = 1e-9);
        }

        // Check that the high-weight particle (or copies of it) dominates the new set
        let original_high_vol = filter.particles[0].log_vol; // Log vol before resampling
        let count_similar = filter.particles.iter().filter(|p| (p.log_vol - original_high_vol).abs() < 1e-6 ).count();
        println!("Count similar after resampling: {}", count_similar);
        // Expect most particles to be copies of the original high-weight particle
        assert!(count_similar > (TEST_NUM_PARTICLES as f64 * 0.9) as usize);
    }

    #[test]
    fn test_estimate_volatility() {
        let filter = create_test_filter();
        // In initialization, particles are around TEST_INITIAL_H = -9.2
        let h_estimate: f64 = filter.particles.iter().map(|p| p.log_vol * p.weight).sum();
        let expected_vol_annualized = h_estimate.exp().sqrt();
        let expected_vol_bps = expected_vol_annualized * 10000.0;

        let estimated_bps = filter.estimate_volatility_bps();

        assert_relative_eq!(estimated_bps, expected_vol_bps, epsilon = 1e-6);
        // Expect ~100bps since mu = ln(0.01^2)
        assert_relative_eq!(estimated_bps, 100.0, epsilon = 20.0); // Allow reasonable deviation
    }

     #[test]
    fn test_percentiles() {
        let mut filter = create_test_filter();
        // Ensure weights are normalized after potential test modifications
        filter.normalize_weights();

        let p5 = filter.estimate_volatility_percentile_bps(0.05);
        let p50 = filter.estimate_volatility_percentile_bps(0.50); // Median
        let p95 = filter.estimate_volatility_percentile_bps(0.95);
        let mean_vol = filter.estimate_volatility_bps(); // Mean

        println!("P5: {:.2}bps, P50: {:.2}bps, Mean: {:.2}bps, P95: {:.2}bps", p5, p50, mean_vol, p95);

        // Check ordering
        assert!(p5 <= p50);
        assert!(p50 <= p95);
        // Mean can be different from median for skewed distributions
        assert!(p5 <= mean_vol);
        assert!(mean_vol <= p95);
    }

     #[test]
     fn test_update_with_zero_dt() {
         let mut filter = create_test_filter();
         filter.update(100.0); // Initial price
         // No sleep, call update immediately
         let estimate = filter.update(100.1);
         assert!(estimate.is_none()); // Should skip update if dt is zero
     }

     #[test]
     fn test_update_with_zero_price_change() {
         let mut filter = create_test_filter();
         filter.update(100.0);
         std::thread::sleep(Duration::from_millis(10));
         // Price doesn't change
         let estimate = filter.update(100.0);
         assert!(estimate.is_some()); // Should still update, log_return is 0.0
         let ess_before = filter.get_ess();

         std::thread::sleep(Duration::from_millis(10));
         let estimate2 = filter.update(100.0); // Still no change
         assert!(estimate2.is_some());
         let ess_after = filter.get_ess();

         // ESS should potentially decrease as particles predicting non-zero vol get lower weights
         assert!(ess_after <= ess_before);
     }
}