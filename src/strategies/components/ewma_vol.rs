// ============================================================================
// EWMA Volatility Model - Tick-Aware Exponentially Weighted Moving Average
// ============================================================================
//
// This component implements a simple, robust volatility estimator designed
// specifically for HIGH-FREQUENCY tick-by-tick market data.
//
// # Why EWMA Instead of Particle Filter for Ticks?
//
// - **Computational efficiency**: O(1) update vs O(N) particles
// - **Numerical stability**: No log-variance transformations or tiny dt values
// - **Interpretability**: Direct volatility estimate, not latent state
// - **Calibration**: Easy to tune half-life for tick frequency
//
// # Algorithm
//
// 1. On each mid-price update:
//    - Calculate log return: r_t = ln(P_t / P_{t-1})
//    - Compute squared return (variance proxy): r_t²
//    - Update EWMA variance: σ²_t = α * r_t² + (1-α) * σ²_{t-1}
//    - Output volatility: σ_t = sqrt(σ²_t)
//
// 2. Outlier filtering:
//    - Detect extreme returns using z-score or MAD
//    - Clip or ignore outliers to prevent flash-crash contamination
//
// 3. Time-scaling:
//    - Estimate tick frequency from update timestamps
//    - Scale volatility to per-tick or per-second as needed
//
// # Configuration
//
// - `alpha`: EWMA smoothing parameter (0.01 = slow, 0.5 = fast)
// - `half_life_seconds`: Half-life for exponential decay (default: 300s = 5min)
// - `outlier_threshold`: Z-score threshold for outlier detection (default: 5.0)
// - `min_volatility_bps`: Floor to prevent division by zero (default: 0.5 bps)
// - `max_volatility_bps`: Ceiling to prevent runaway estimates (default: 100 bps)
//
// # Example
//
// ```rust
// use strategies::components::{VolatilityModel, EwmaVolatilityModel};
//
// let mut vol_model = EwmaVolatilityModel::with_half_life(300.0); // 5-minute half-life
//
// // On each market update
// vol_model.on_market_update(&market_update);
//
// // Get current estimates
// let vol_bps = vol_model.get_volatility_bps();
// let uncertainty = vol_model.get_uncertainty_bps();
// ```

use std::collections::VecDeque;
use std::time::Instant;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use log::{debug, warn};

use crate::strategy::MarketUpdate;
use super::volatility::VolatilityModel;

/// Configuration for EWMA volatility model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EwmaVolConfig {
    /// EWMA smoothing parameter (0.0-1.0)
    /// alpha = 2 / (N + 1) where N is the lookback window
    /// Alternatively: alpha = 1 - exp(-ln(2) / half_life_ticks)
    #[serde(default = "default_alpha")]
    pub alpha: f64,

    /// Half-life in seconds for EWMA decay
    /// If provided, this will override alpha calculation
    #[serde(default)]
    pub half_life_seconds: Option<f64>,

    /// Z-score threshold for outlier detection
    /// Returns exceeding this threshold are clipped or ignored
    #[serde(default = "default_outlier_threshold")]
    pub outlier_threshold: f64,

    /// Minimum volatility floor in basis points
    #[serde(default = "default_min_vol")]
    pub min_volatility_bps: f64,

    /// Maximum volatility ceiling in basis points
    #[serde(default = "default_max_vol")]
    pub max_volatility_bps: f64,

    /// Number of recent returns to keep for uncertainty estimation
    #[serde(default = "default_history_size")]
    pub history_size: usize,

    /// Whether to use absolute returns (robust to direction)
    #[serde(default = "default_use_absolute")]
    pub use_absolute_returns: bool,

    /// Expected tick frequency in Hz (for initialization)
    #[serde(default = "default_tick_freq")]
    pub expected_tick_frequency_hz: f64,
}

fn default_alpha() -> f64 { 0.05 } // ~20 tick lookback
fn default_outlier_threshold() -> f64 { 5.0 }
fn default_min_vol() -> f64 { 0.5 }
fn default_max_vol() -> f64 { 100.0 }
fn default_history_size() -> usize { 100 }
fn default_use_absolute() -> bool { false }
fn default_tick_freq() -> f64 { 1.0 } // 1 Hz default (conservative)

impl Default for EwmaVolConfig {
    fn default() -> Self {
        Self {
            alpha: default_alpha(),
            half_life_seconds: None,
            outlier_threshold: default_outlier_threshold(),
            min_volatility_bps: default_min_vol(),
            max_volatility_bps: default_max_vol(),
            history_size: default_history_size(),
            use_absolute_returns: default_use_absolute(),
            expected_tick_frequency_hz: default_tick_freq(),
        }
    }
}

impl EwmaVolConfig {
    /// Create config with specific half-life in seconds
    pub fn with_half_life(half_life_seconds: f64) -> Self {
        Self {
            half_life_seconds: Some(half_life_seconds),
            ..Default::default()
        }
    }

    /// Create config for high-frequency ticks (fast adaptation)
    pub fn high_frequency() -> Self {
        Self {
            alpha: 0.1, // Fast adaptation
            half_life_seconds: Some(60.0), // 1-minute half-life
            outlier_threshold: 4.0, // Stricter outlier detection
            expected_tick_frequency_hz: 10.0, // 10 Hz
            ..Default::default()
        }
    }

    /// Create config for low-frequency ticks (slow adaptation)
    pub fn low_frequency() -> Self {
        Self {
            alpha: 0.01, // Slow adaptation
            half_life_seconds: Some(600.0), // 10-minute half-life
            outlier_threshold: 6.0, // More lenient outlier detection
            expected_tick_frequency_hz: 0.1, // 0.1 Hz (one tick per 10 seconds)
            ..Default::default()
        }
    }
}

/// EWMA-based volatility estimator for tick data
pub struct EwmaVolatilityModel {
    /// Configuration
    config: EwmaVolConfig,

    /// EWMA of squared returns (variance estimate)
    ewma_variance: f64,

    /// Previous mid-price for return calculation
    prev_mid: Option<f64>,

    /// Timestamp of previous update
    prev_time: Option<Instant>,

    /// Recent returns for uncertainty estimation
    recent_returns: VecDeque<f64>,

    /// Estimated tick frequency (adaptive)
    estimated_tick_freq_hz: f64,

    /// Running mean of absolute returns (for outlier detection)
    mean_abs_return: f64,

    /// Update counter
    update_count: usize,

    /// Effective alpha (computed from half-life if needed)
    effective_alpha: f64,
}

impl EwmaVolatilityModel {
    /// Create a new EWMA volatility model with custom config
    pub fn new(config: EwmaVolConfig) -> Self {
        // Calculate effective alpha from half-life if provided
        let effective_alpha = if let Some(half_life_sec) = config.half_life_seconds {
            // Estimate alpha from half-life and expected tick frequency
            // alpha ≈ 1 - exp(-ln(2) / (half_life_sec * tick_freq))
            let half_life_ticks = half_life_sec * config.expected_tick_frequency_hz;
            let alpha = 1.0 - (-std::f64::consts::LN_2 / half_life_ticks).exp();
            alpha.clamp(0.001, 0.5)
        } else {
            config.alpha
        };

        Self {
            config: config.clone(),
            ewma_variance: 0.0,
            prev_mid: None,
            prev_time: None,
            recent_returns: VecDeque::with_capacity(config.history_size),
            estimated_tick_freq_hz: config.expected_tick_frequency_hz,
            mean_abs_return: 0.0,
            update_count: 0,
            effective_alpha,
        }
    }

    /// Create with default configuration
    pub fn new_default() -> Self {
        Self::new(EwmaVolConfig::default())
    }

    /// Create with specific half-life
    pub fn with_half_life(half_life_seconds: f64) -> Self {
        Self::new(EwmaVolConfig::with_half_life(half_life_seconds))
    }

    /// Create from JSON config
    pub fn from_json(config: &Value) -> Self {
        let vol_config: EwmaVolConfig = serde_json::from_value(config.clone())
            .unwrap_or_else(|e| {
                warn!("Failed to parse EWMA vol config: {}. Using defaults.", e);
                EwmaVolConfig::default()
            });
        Self::new(vol_config)
    }

    /// Update volatility estimate with new mid-price
    fn update(&mut self, current_mid: f64) {
        let now = Instant::now();

        // Check if we have a previous price to calculate return
        if let Some(prev_mid) = self.prev_mid {
            if prev_mid <= 0.0 || current_mid <= 0.0 {
                // Skip invalid prices
                self.prev_mid = Some(current_mid);
                self.prev_time = Some(now);
                return;
            }

            // Calculate log return
            let log_return = (current_mid / prev_mid).ln();

            // Check if return is finite
            if !log_return.is_finite() {
                warn!("[EWMA VOL] Non-finite return: prev={}, curr={}", prev_mid, current_mid);
                self.prev_mid = Some(current_mid);
                self.prev_time = Some(now);
                return;
            }

            // Update tick frequency estimate (adaptive)
            if let Some(prev_time) = self.prev_time {
                let dt_sec = now.duration_since(prev_time).as_secs_f64();
                if dt_sec > 0.0 && dt_sec < 3600.0 { // Sanity check: less than 1 hour
                    let freq = 1.0 / dt_sec;
                    // Smooth tick frequency estimate
                    let freq_alpha = 0.05;
                    self.estimated_tick_freq_hz = freq_alpha * freq + (1.0 - freq_alpha) * self.estimated_tick_freq_hz;
                }
            }

            // Outlier detection and filtering
            let filtered_return = if self.update_count > 10 && self.mean_abs_return > 1e-8 {
                let z_score = log_return.abs() / self.mean_abs_return.max(1e-8);

                if z_score > self.config.outlier_threshold {
                    // Outlier detected - clip to threshold
                    let clipped_return = self.mean_abs_return * self.config.outlier_threshold * log_return.signum();
                    debug!(
                        "[EWMA VOL] Outlier detected: return={:.6}, z_score={:.2}, clipped to {:.6}",
                        log_return, z_score, clipped_return
                    );
                    clipped_return
                } else {
                    log_return
                }
            } else {
                log_return
            };

            // Update mean absolute return (for outlier detection)
            let abs_return = filtered_return.abs();
            if self.update_count == 0 {
                self.mean_abs_return = abs_return;
            } else {
                // Use same alpha for mean absolute return
                self.mean_abs_return = self.effective_alpha * abs_return + (1.0 - self.effective_alpha) * self.mean_abs_return;
            }

            // Calculate variance proxy (squared return)
            let return_to_use = if self.config.use_absolute_returns {
                abs_return
            } else {
                filtered_return
            };
            let squared_return = return_to_use * return_to_use;

            // Update EWMA variance
            if self.update_count == 0 {
                // Initialize with first observation
                self.ewma_variance = squared_return;
            } else {
                self.ewma_variance = self.effective_alpha * squared_return + (1.0 - self.effective_alpha) * self.ewma_variance;
            }

            // Store return for uncertainty estimation
            self.recent_returns.push_back(filtered_return);
            if self.recent_returns.len() > self.config.history_size {
                self.recent_returns.pop_front();
            }

            self.update_count += 1;

            debug!(
                "[EWMA VOL] Updated: return={:.6}, var={:.8}, vol={:.2}bps, freq={:.2}Hz",
                filtered_return,
                self.ewma_variance,
                self.get_volatility_bps(),
                self.estimated_tick_freq_hz
            );
        }

        // Update state
        self.prev_mid = Some(current_mid);
        self.prev_time = Some(now);
    }

    /// Get current volatility estimate in basis points
    /// This returns INSTANTANEOUS (per-tick) volatility, not annualized
    fn estimate_volatility_bps(&self) -> f64 {
        if self.ewma_variance <= 0.0 {
            return self.config.min_volatility_bps;
        }

        // Volatility = sqrt(variance)
        let vol_per_tick = self.ewma_variance.sqrt();

        // Convert to basis points (already per-tick, no time scaling needed)
        let vol_bps = vol_per_tick * 10000.0;

        // Apply bounds
        vol_bps.clamp(self.config.min_volatility_bps, self.config.max_volatility_bps)
    }

    /// Estimate uncertainty in volatility estimate
    /// Returns standard deviation of recent volatility estimates
    fn estimate_uncertainty_bps(&self) -> f64 {
        if self.recent_returns.len() < 2 {
            // Not enough data - return conservative uncertainty
            return self.estimate_volatility_bps() * 0.5;
        }

        // Calculate variance of recent squared returns
        let recent_squared_returns: Vec<f64> = self.recent_returns.iter()
            .map(|r| r * r)
            .collect();

        let mean_sq: f64 = recent_squared_returns.iter().sum::<f64>() / recent_squared_returns.len() as f64;

        let variance_of_variance: f64 = recent_squared_returns.iter()
            .map(|sq| {
                let dev = sq - mean_sq;
                dev * dev
            })
            .sum::<f64>() / (recent_squared_returns.len() - 1) as f64;

        // Standard deviation of variance estimate
        let std_variance = variance_of_variance.sqrt();

        // Convert to volatility space using delta method: Var(σ) ≈ Var(σ²) / (4σ²)
        let current_vol = self.estimate_volatility_bps() / 10000.0; // Convert back to decimal
        let std_vol = if current_vol > 1e-8 {
            std_variance / (4.0 * current_vol * current_vol)
        } else {
            std_variance.sqrt() // Fallback
        };

        // Convert to basis points
        let uncertainty_bps = std_vol * 10000.0;

        // Bound uncertainty (shouldn't be larger than the estimate itself)
        uncertainty_bps.clamp(0.0, self.estimate_volatility_bps() * 0.5)
    }

    /// Get diagnostic information
    pub fn diagnostics(&self) -> String {
        format!(
            "EWMA Vol: {:.2}bps ± {:.2}bps | Updates: {} | Freq: {:.2}Hz | Alpha: {:.4} | Var: {:.8}",
            self.estimate_volatility_bps(),
            self.estimate_uncertainty_bps(),
            self.update_count,
            self.estimated_tick_freq_hz,
            self.effective_alpha,
            self.ewma_variance
        )
    }

    /// Reset the model (clear all history)
    pub fn reset(&mut self) {
        self.ewma_variance = 0.0;
        self.prev_mid = None;
        self.prev_time = None;
        self.recent_returns.clear();
        self.mean_abs_return = 0.0;
        self.update_count = 0;
    }
}

impl VolatilityModel for EwmaVolatilityModel {
    fn on_market_update(&mut self, update: &MarketUpdate) {
        // Update volatility if we have a mid-price
        if let Some(mid_price) = update.mid_price {
            self.update(mid_price);
        }
    }

    fn get_volatility_bps(&self) -> f64 {
        self.estimate_volatility_bps()
    }

    fn get_uncertainty_bps(&self) -> f64 {
        self.estimate_uncertainty_bps()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ewma_vol_initialization() {
        let model = EwmaVolatilityModel::new_default();

        // Should return minimum volatility when uninitialized
        assert!(model.get_volatility_bps() >= 0.5);
        assert_eq!(model.update_count, 0);
    }

    #[test]
    fn test_ewma_vol_basic_update() {
        let mut model = EwmaVolatilityModel::with_half_life(60.0);

        // Create a simple market update
        let update1 = MarketUpdate::from_mid_price("TEST".to_string(), 100.0);
        let update2 = MarketUpdate::from_mid_price("TEST".to_string(), 100.5);

        model.on_market_update(&update1);
        assert_eq!(model.update_count, 0); // First update just sets prev_mid

        model.on_market_update(&update2);
        assert_eq!(model.update_count, 1); // Second update calculates first return

        let vol = model.get_volatility_bps();
        assert!(vol > 0.0);
        assert!(vol < 100.0); // Should be reasonable
    }

    #[test]
    fn test_ewma_vol_outlier_filtering() {
        let mut model = EwmaVolatilityModel::new(EwmaVolConfig {
            outlier_threshold: 3.0,
            ..Default::default()
        });

        // Normal updates
        let mut price = 100.0;
        for _ in 0..20 {
            let update = MarketUpdate::from_mid_price("TEST".to_string(), price);
            model.on_market_update(&update);
            price += 0.01; // Small moves
        }

        let vol_before = model.get_volatility_bps();

        // Flash crash (huge move)
        let crash_update = MarketUpdate::from_mid_price("TEST".to_string(), price * 0.8); // 20% drop
        model.on_market_update(&crash_update);

        let vol_after = model.get_volatility_bps();

        // Volatility should increase but not explode (outlier was clipped)
        assert!(vol_after > vol_before);
        assert!(vol_after < vol_before * 10.0); // Shouldn't be 10x higher
    }

    #[test]
    fn test_half_life_calculation() {
        let model = EwmaVolatilityModel::with_half_life(300.0);

        // Alpha should be calculated from half-life
        assert!(model.effective_alpha > 0.0);
        assert!(model.effective_alpha < 1.0);
    }
}
