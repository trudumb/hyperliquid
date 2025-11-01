// ============================================================================
// Hybrid Volatility Model - EWMA Grounded by Particle Filter
// ============================================================================
//
// This component combines the best of both worlds:
// 1. **EWMA**: Fast, tick-aware, numerically stable short-term tracking
// 2. **Particle Filter**: Stochastic baseline, uncertainty quantification
// 3. **Bid-Ask Rate Dynamics**: Inform volatility-of-volatility
//
// # Design Philosophy
//
// The hybrid model uses the particle filter to provide a "gravity well" that
// prevents the EWMA from drifting too far from the stochastic baseline. Think
// of it as:
// - **EWMA**: A fast-moving satellite tracking tick-to-tick changes
// - **Particle Filter**: A gravitational center providing long-term context
// - **Bid-Ask Rates**: Measure of market maker competition (affects realized vol)
//
// # Algorithm
//
// 1. **Fast Layer (EWMA)**:
//    - Update on every tick with exponential weighting
//    - Provides low-latency volatility estimate
//
// 2. **Slow Layer (Particle Filter)**:
//    - Update periodically (e.g., every 10 ticks or 1 second)
//    - Provides stochastic baseline and uncertainty bounds
//
// 3. **Grounding Mechanism**:
//    - Pull EWMA toward PF baseline: `σ_hybrid = (1-λ) * σ_EWMA + λ * σ_PF`
//    - Grounding strength `λ` increases with PF confidence (low uncertainty)
//    - Bid-ask rate volatility modulates the grounding strength
//
// 4. **Bid-Ask Rate Tracking**:
//    - Track fill rates on bid and ask sides
//    - High bid-ask rate volatility → weaken grounding (fast market)
//    - Low bid-ask rate volatility → strengthen grounding (stable market)
//
// # Configuration
//
// - `ewma_alpha`: EWMA smoothing for tick updates (default: 0.1)
// - `pf_update_interval_ticks`: Update particle filter every N ticks (default: 10)
// - `grounding_strength_base`: Base grounding strength λ (default: 0.2)
// - `grounding_sensitivity`: How much bid-ask rate affects grounding (default: 0.5)
// - `min_grounding`: Minimum grounding strength (default: 0.05)
// - `max_grounding`: Maximum grounding strength (default: 0.5)
//
// # Example
//
// ```rust
// use strategies::components::{VolatilityModel, HybridVolatilityModel};
//
// let mut vol_model = HybridVolatilityModel::new_default();
//
// // On each market update
// vol_model.on_market_update(&market_update);
//
// // Get hybrid estimate
// let vol_bps = vol_model.get_volatility_bps();
// let uncertainty = vol_model.get_uncertainty_bps(); // From particle filter
// ```

use std::collections::VecDeque;
use std::time::Instant;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use log::{debug, warn};

use crate::strategy::MarketUpdate;
use super::volatility::VolatilityModel;
use super::ewma_vol::{EwmaVolatilityModel, EwmaVolConfig};
use super::particle_filter_vol::ParticleFilterState;

/// Configuration for hybrid volatility model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HybridVolConfig {
    /// EWMA configuration for fast layer
    #[serde(default)]
    pub ewma_config: EwmaVolConfig,

    /// Number of particles for stochastic layer
    #[serde(default = "default_num_particles")]
    pub num_particles: usize,

    /// Particle filter mu (recalibrated for tick-level)
    #[serde(default = "default_pf_mu")]
    pub pf_mu: f64,

    /// Particle filter phi (persistence)
    #[serde(default = "default_pf_phi")]
    pub pf_phi: f64,

    /// Particle filter sigma_eta (vol-of-vol)
    #[serde(default = "default_pf_sigma_eta")]
    pub pf_sigma_eta: f64,

    /// Update particle filter every N ticks
    #[serde(default = "default_pf_update_interval")]
    pub pf_update_interval_ticks: usize,

    /// Base grounding strength (0.0 = pure EWMA, 1.0 = pure PF)
    #[serde(default = "default_grounding_strength")]
    pub grounding_strength_base: f64,

    /// Sensitivity to bid-ask rate dynamics (0.0 = ignore, 1.0 = full effect)
    #[serde(default = "default_grounding_sensitivity")]
    pub grounding_sensitivity: f64,

    /// Minimum grounding strength
    #[serde(default = "default_min_grounding")]
    pub min_grounding: f64,

    /// Maximum grounding strength
    #[serde(default = "default_max_grounding")]
    pub max_grounding: f64,

    /// Enable bid-ask rate tracking
    #[serde(default = "default_enable_ba_tracking")]
    pub enable_bid_ask_tracking: bool,
}

fn default_num_particles() -> usize { 1000 }
fn default_pf_mu() -> f64 { -18.0 }
fn default_pf_phi() -> f64 { 0.95 }
fn default_pf_sigma_eta() -> f64 { 0.5 }
fn default_pf_update_interval() -> usize { 10 }
fn default_grounding_strength() -> f64 { 0.2 }
fn default_grounding_sensitivity() -> f64 { 0.5 }
fn default_min_grounding() -> f64 { 0.05 }
fn default_max_grounding() -> f64 { 0.5 }
fn default_enable_ba_tracking() -> bool { true }

impl Default for HybridVolConfig {
    fn default() -> Self {
        Self {
            ewma_config: EwmaVolConfig::high_frequency(),
            num_particles: default_num_particles(),
            pf_mu: default_pf_mu(),
            pf_phi: default_pf_phi(),
            pf_sigma_eta: default_pf_sigma_eta(),
            pf_update_interval_ticks: default_pf_update_interval(),
            grounding_strength_base: default_grounding_strength(),
            grounding_sensitivity: default_grounding_sensitivity(),
            min_grounding: default_min_grounding(),
            max_grounding: default_max_grounding(),
            enable_bid_ask_tracking: default_enable_ba_tracking(),
        }
    }
}

impl HybridVolConfig {
    /// Create config optimized for high-frequency tick data
    pub fn high_frequency() -> Self {
        Self {
            ewma_config: EwmaVolConfig::high_frequency(),
            pf_update_interval_ticks: 20, // Update PF every 20 ticks
            grounding_strength_base: 0.15, // Lighter grounding for fast markets
            ..Default::default()
        }
    }

    /// Create config optimized for low-frequency tick data
    pub fn low_frequency() -> Self {
        Self {
            ewma_config: EwmaVolConfig::low_frequency(),
            pf_update_interval_ticks: 5, // Update PF every 5 ticks
            grounding_strength_base: 0.3, // Stronger grounding for slow markets
            ..Default::default()
        }
    }
}

/// Bid-Ask rate tracker for volatility-of-volatility estimation
#[derive(Debug, Clone)]
struct BidAskRateTracker {
    /// Recent bid fill rates (fills per second)
    bid_rates: VecDeque<f64>,

    /// Recent ask fill rates (fills per second)
    ask_rates: VecDeque<f64>,

    /// Maximum history size
    max_history: usize,

    /// Last update time
    last_update: Option<Instant>,

    /// Cumulative bid fills since last rate calculation
    bid_fills_count: u64,

    /// Cumulative ask fills since last rate calculation
    ask_fills_count: u64,
}

impl BidAskRateTracker {
    fn new(max_history: usize) -> Self {
        Self {
            bid_rates: VecDeque::with_capacity(max_history),
            ask_rates: VecDeque::with_capacity(max_history),
            max_history,
            last_update: None,
            bid_fills_count: 0,
            ask_fills_count: 0,
        }
    }

    /// Record a fill event
    fn record_fill(&mut self, is_bid: bool) {
        if is_bid {
            self.bid_fills_count += 1;
        } else {
            self.ask_fills_count += 1;
        }
    }

    /// Update rates (call periodically, e.g., every second)
    fn update_rates(&mut self) {
        let now = Instant::now();

        if let Some(last) = self.last_update {
            let dt = now.duration_since(last).as_secs_f64();

            if dt > 0.0 {
                // Calculate rates (fills per second)
                let bid_rate = self.bid_fills_count as f64 / dt;
                let ask_rate = self.ask_fills_count as f64 / dt;

                // Store rates
                self.bid_rates.push_back(bid_rate);
                self.ask_rates.push_back(ask_rate);

                // Limit history
                if self.bid_rates.len() > self.max_history {
                    self.bid_rates.pop_front();
                }
                if self.ask_rates.len() > self.max_history {
                    self.ask_rates.pop_front();
                }

                // Reset counters
                self.bid_fills_count = 0;
                self.ask_fills_count = 0;
            }
        }

        self.last_update = Some(now);
    }

    /// Get volatility of bid-ask rates (std dev of combined rates)
    fn get_rate_volatility(&self) -> f64 {
        if self.bid_rates.len() < 2 {
            return 0.0;
        }

        // Combine bid and ask rates
        let combined: Vec<f64> = self.bid_rates.iter()
            .zip(self.ask_rates.iter())
            .map(|(b, a)| b + a)
            .collect();

        let mean: f64 = combined.iter().sum::<f64>() / combined.len() as f64;
        let variance: f64 = combined.iter()
            .map(|r| (r - mean).powi(2))
            .sum::<f64>() / (combined.len() - 1) as f64;

        variance.sqrt()
    }

    /// Get mean bid-ask rate (for normalization)
    fn get_mean_rate(&self) -> f64 {
        if self.bid_rates.is_empty() {
            return 0.0;
        }

        let combined: Vec<f64> = self.bid_rates.iter()
            .zip(self.ask_rates.iter())
            .map(|(b, a)| b + a)
            .collect();

        combined.iter().sum::<f64>() / combined.len() as f64
    }
}

/// Hybrid EWMA-ParticleFilter volatility model
pub struct HybridVolatilityModel {
    /// Configuration
    config: HybridVolConfig,

    /// Fast layer: EWMA model
    ewma_model: EwmaVolatilityModel,

    /// Slow layer: Particle filter state
    pf_state: ParticleFilterState,

    /// Bid-ask rate tracker
    ba_tracker: BidAskRateTracker,

    /// Tick counter for PF update scheduling
    tick_count: usize,

    /// Current grounding strength (adaptive)
    current_grounding: f64,

    /// Hybrid volatility estimate (cached)
    hybrid_volatility_bps: f64,

    /// Last PF update time
    last_pf_update: Option<Instant>,
}

impl HybridVolatilityModel {
    /// Create a new hybrid volatility model
    pub fn new(config: HybridVolConfig) -> Self {
        let ewma_model = EwmaVolatilityModel::new(config.ewma_config.clone());

        let pf_state = ParticleFilterState::new(
            config.num_particles,
            config.pf_mu,
            config.pf_phi,
            config.pf_sigma_eta,
            config.pf_mu - 0.5, // initial_h slightly below mu
            1.0,                // initial_h_std_dev
            42,                 // seed (different from standalone PF)
        );

        let ba_tracker = BidAskRateTracker::new(50); // Keep 50 rate samples

        Self {
            config,
            ewma_model,
            pf_state,
            ba_tracker,
            tick_count: 0,
            current_grounding: default_grounding_strength(),
            hybrid_volatility_bps: 5.0, // Default starting value
            last_pf_update: None,
        }
    }

    /// Create with default configuration
    pub fn new_default() -> Self {
        Self::new(HybridVolConfig::default())
    }

    /// Create from JSON config
    pub fn from_json(config: &Value) -> Self {
        let hybrid_config: HybridVolConfig = serde_json::from_value(config.clone())
            .unwrap_or_else(|e| {
                warn!("Failed to parse hybrid vol config: {}. Using defaults.", e);
                HybridVolConfig::default()
            });
        Self::new(hybrid_config)
    }

    /// Record a fill event (for bid-ask rate tracking)
    pub fn record_fill(&mut self, is_bid: bool) {
        if self.config.enable_bid_ask_tracking {
            self.ba_tracker.record_fill(is_bid);
        }
    }

    /// Update the model with new mid-price
    fn update(&mut self, current_mid: f64) {
        self.tick_count += 1;

        // 1. Always update EWMA (fast layer)
        let market_update = crate::strategy::MarketUpdate {
            asset: "HYBRID".to_string(),
            mid_price: Some(current_mid),
            l2_book: None,
            trades: vec![],
            timestamp: chrono::Utc::now().timestamp_millis() as u64,
        };
        self.ewma_model.on_market_update(&market_update);

        // 2. Update particle filter periodically (slow layer)
        let should_update_pf = self.tick_count % self.config.pf_update_interval_ticks == 0;

        if should_update_pf {
            self.pf_state.update(current_mid);
            self.last_pf_update = Some(Instant::now());

            // 3. Update bid-ask rate tracking (also periodic)
            if self.config.enable_bid_ask_tracking {
                self.ba_tracker.update_rates();
            }

            // 4. Calculate adaptive grounding strength
            self.update_grounding_strength();

            debug!(
                "[HYBRID VOL] PF updated at tick {}: grounding={:.3}, ba_rate_vol={:.3}",
                self.tick_count,
                self.current_grounding,
                self.ba_tracker.get_rate_volatility()
            );
        }

        // 5. Compute hybrid volatility estimate
        self.compute_hybrid_estimate();
    }

    /// Update adaptive grounding strength based on PF uncertainty and bid-ask dynamics
    fn update_grounding_strength(&mut self) {
        let base_grounding = self.config.grounding_strength_base;

        // Factor 1: Particle filter confidence
        // Higher PF uncertainty → lower grounding (trust EWMA more)
        let pf_vol_bps = self.pf_state.estimate_volatility_bps();
        let pf_uncertainty_bps = self.pf_state.get_volatility_std_dev_bps();

        let pf_confidence = if pf_vol_bps > 0.0 {
            // Confidence = 1 - (uncertainty / volatility), clamped to [0, 1]
            (1.0 - (pf_uncertainty_bps / pf_vol_bps)).clamp(0.0, 1.0)
        } else {
            0.5 // Neutral if no PF estimate
        };

        // Factor 2: Bid-ask rate volatility
        // High rate volatility → lower grounding (fast-moving market, trust EWMA)
        // Low rate volatility → higher grounding (stable market, trust PF baseline)
        let rate_volatility = self.ba_tracker.get_rate_volatility();
        let mean_rate = self.ba_tracker.get_mean_rate().max(0.1); // Avoid division by zero

        let normalized_rate_vol = (rate_volatility / mean_rate).clamp(0.0, 2.0);

        // Rate adjustment: high vol → reduce grounding
        let rate_adjustment = if self.config.enable_bid_ask_tracking {
            1.0 - (normalized_rate_vol * self.config.grounding_sensitivity).min(0.8)
        } else {
            1.0 // No adjustment if tracking disabled
        };

        // Combine factors
        let adaptive_grounding = base_grounding * pf_confidence * rate_adjustment;

        // Apply bounds
        self.current_grounding = adaptive_grounding.clamp(
            self.config.min_grounding,
            self.config.max_grounding,
        );

        debug!(
            "[HYBRID VOL GROUNDING] pf_conf={:.3}, rate_adj={:.3} → λ={:.3}",
            pf_confidence, rate_adjustment, self.current_grounding
        );
    }

    /// Compute hybrid volatility estimate by blending EWMA and PF
    fn compute_hybrid_estimate(&mut self) {
        let ewma_vol = self.ewma_model.get_volatility_bps();
        let pf_vol = self.pf_state.estimate_volatility_bps();

        // Grounded EWMA: σ_hybrid = (1-λ) * σ_EWMA + λ * σ_PF
        let lambda = self.current_grounding;
        self.hybrid_volatility_bps = (1.0 - lambda) * ewma_vol + lambda * pf_vol;

        debug!(
            "[HYBRID VOL] EWMA={:.2}bps, PF={:.2}bps, λ={:.3} → Hybrid={:.2}bps",
            ewma_vol, pf_vol, lambda, self.hybrid_volatility_bps
        );
    }

    /// Get diagnostics string
    pub fn diagnostics(&self) -> String {
        format!(
            "Hybrid Vol: {:.2}bps | EWMA: {:.2}bps | PF: {:.2}bps | Grounding: {:.3} | BA Rate Vol: {:.3} | Ticks: {}",
            self.hybrid_volatility_bps,
            self.ewma_model.get_volatility_bps(),
            self.pf_state.estimate_volatility_bps(),
            self.current_grounding,
            self.ba_tracker.get_rate_volatility(),
            self.tick_count
        )
    }

    /// Get particle filter state (for advanced diagnostics)
    pub fn get_pf_state(&self) -> &ParticleFilterState {
        &self.pf_state
    }

    /// Get EWMA model (for advanced diagnostics)
    pub fn get_ewma_model(&self) -> &EwmaVolatilityModel {
        &self.ewma_model
    }
}

impl VolatilityModel for HybridVolatilityModel {
    fn on_market_update(&mut self, update: &MarketUpdate) {
        // Update volatility if we have a mid-price
        if let Some(mid_price) = update.mid_price {
            self.update(mid_price);
        }
    }

    fn get_volatility_bps(&self) -> f64 {
        self.hybrid_volatility_bps
    }

    fn get_uncertainty_bps(&self) -> f64 {
        // Use particle filter uncertainty (one of the key benefits of hybrid approach)
        self.pf_state.get_volatility_std_dev_bps()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hybrid_vol_initialization() {
        let model = HybridVolatilityModel::new_default();

        // Should return reasonable default
        let vol = model.get_volatility_bps();
        assert!(vol > 0.0);
        assert!(vol < 100.0);
    }

    #[test]
    fn test_hybrid_vol_grounding() {
        let mut model = HybridVolatilityModel::new_default();

        // Simulate price movements
        let mut price = 100.0;
        for i in 0..50 {
            price += if i % 2 == 0 { 0.01 } else { -0.01 };

            let update = MarketUpdate {
                asset: "TEST".to_string(),
                mid_price: Some(price),
                l2_book: None,
                trades: vec![],
                timestamp: chrono::Utc::now().timestamp_millis() as u64,
            };

            model.on_market_update(&update);
        }

        // After 50 updates, we should have:
        // - EWMA estimate
        // - PF estimate (updated ~5 times with default interval)
        // - Hybrid estimate between them
        let ewma_vol = model.ewma_model.get_volatility_bps();
        let pf_vol = model.pf_state.estimate_volatility_bps();
        let hybrid_vol = model.get_volatility_bps();

        println!("EWMA: {:.2}, PF: {:.2}, Hybrid: {:.2}", ewma_vol, pf_vol, hybrid_vol);

        // Hybrid should be between EWMA and PF (or close to one if grounding is weak/strong)
        assert!(hybrid_vol > 0.0);
        assert!(hybrid_vol < 100.0);
    }

    #[test]
    fn test_bid_ask_tracking() {
        let mut tracker = BidAskRateTracker::new(10);

        // Initialize the timestamp
        tracker.update_rates();

        // Record some fills
        for _ in 0..5 {
            tracker.record_fill(true); // Bid fill
        }
        for _ in 0..3 {
            tracker.record_fill(false); // Ask fill
        }

        // Wait a bit and update rates (second call will record the rates)
        std::thread::sleep(std::time::Duration::from_millis(100));
        tracker.update_rates();

        // Should have recorded rates
        assert!(!tracker.bid_rates.is_empty());
        assert!(!tracker.ask_rates.is_empty());
    }
}
