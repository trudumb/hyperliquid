// ============================================================================
// Hawkes Fill Model Component - Self-Exciting Point Process Fill Estimation
// ============================================================================
//
// This component implements the FillModel trait using Hawkes processes
// (self-exciting point processes) to model fill rates at multiple price levels.
//
// Ported from hawkes_multi_level.rs HawkesFillModel to be a pluggable component.
//
// # Algorithm
//
// Hawkes processes model event rates that increase after each event occurs:
//   λ_t = μ + α * Σ exp(-β * (t - t_i))
//
// Where:
// - μ = baseline intensity (fills per second)
// - α = excitation parameter (how much each fill increases the rate)
// - β = decay parameter (how fast the excitation decays)
// - t_i = times of previous fills
//
// This is a good model for order book fills because:
// - Fill rates depend on recent fill history
// - Fills tend to cluster (self-exciting behavior)
// - Different levels have different base rates
//
// # Multi-Level Modeling
//
// The model tracks separate Hawkes processes for:
// - L1 bid, L1 ask
// - L2 bid, L2 ask
// - L3 bid, L3 ask
//
// Each level has its own intensity estimate based on observed fills.

use std::collections::VecDeque;
use serde::{Deserialize, Serialize};

use crate::TradeInfo;
use super::fill_model::FillModel;

/// Epsilon for numerical stability
const EPSILON: f64 = 1e-10;

/// Hawkes process parameters for modeling self-exciting fill behavior
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HawkesParams {
    /// Base fill intensity (fills per second at best quotes)
    pub lambda_base: f64,
    
    /// Price sensitivity parameter (GLFT decay rate)
    pub kappa: f64,
    
    /// Queue position decay between levels
    pub rho: f64,
    
    /// LOB imbalance sensitivity
    pub beta_imbalance: f64,
    
    /// Self-excitation strength (0.3-0.8 typical)
    pub alpha: f64,
    
    /// Memory decay rate (seconds^-1, 0.5-2.0 typical)
    pub beta_decay: f64,
    
    /// Maximum memory window (seconds)
    pub memory_window: f64,
}

impl Default for HawkesParams {
    fn default() -> Self {
        Self {
            lambda_base: 1.0,
            kappa: 0.15,
            rho: 0.8,
            beta_imbalance: 0.5,
            alpha: 0.5,
            beta_decay: 1.0,
            memory_window: 10.0,
        }
    }
}

/// Tracks fill events for Hawkes self-excitation calculation
#[derive(Debug, Clone)]
pub struct FillHistory {
    /// Recent fill timestamps for each level
    fills: Vec<VecDeque<f64>>,
    
    /// Memory window (only keep fills within this time)
    memory_window: f64,
}

impl FillHistory {
    pub fn new(num_levels: usize, memory_window: f64) -> Self {
        Self {
            fills: vec![VecDeque::new(); num_levels],
            memory_window,
        }
    }
    
    /// Record a fill event at a given level
    pub fn record(&mut self, level: usize, timestamp: f64) {
        if level >= self.fills.len() {
            return; // Safety check
        }
        
        self.fills[level].push_back(timestamp);
        
        // Remove old fills outside memory window
        while let Some(&oldest) = self.fills[level].front() {
            if timestamp - oldest > self.memory_window {
                self.fills[level].pop_front();
            } else {
                break;
            }
        }
    }
    
    /// Compute Hawkes self-excitation term for a level
    /// ∑_i α * exp(-β * (t - t_i))
    pub fn excitation(&self, level: usize, current_time: f64, params: &HawkesParams) -> f64 {
        if level >= self.fills.len() {
            return 0.0;
        }
        
        self.fills[level]
            .iter()
            .map(|&fill_time| {
                let tau = current_time - fill_time;
                params.alpha * (-params.beta_decay * tau).exp()
            })
            .sum()
    }
    
    /// Get number of recent fills at a level
    pub fn recent_fill_count(&self, level: usize) -> usize {
        if level >= self.fills.len() {
            0
        } else {
            self.fills[level].len()
        }
    }
    
    /// Clear all history (useful for testing or resets)
    pub fn clear(&mut self) {
        for fills in &mut self.fills {
            fills.clear();
        }
    }
}

/// Complete Hawkes process fill rate model
/// PORTED FROM: hawkes_multi_level.rs HawkesFillModel
/// 
/// Note: This is the component-local version. The crate root still exports
/// the version from hawkes_multi_level.rs for backward compatibility.
#[derive(Debug, Clone)]
pub struct HawkesFillModelCore {
    /// Bid side parameters
    pub bid_params: HawkesParams,
    
    /// Ask side parameters
    pub ask_params: HawkesParams,
    
    /// Fill history for bids (per level)
    bid_history: FillHistory,
    
    /// Fill history for asks (per level)
    ask_history: FillHistory,
}

impl HawkesFillModelCore {
    /// Create new Hawkes fill model
    pub fn new(num_levels: usize) -> Self {
        let params = HawkesParams::default();
        Self {
            bid_params: params.clone(),
            ask_params: params.clone(),
            bid_history: FillHistory::new(num_levels, params.memory_window),
            ask_history: FillHistory::new(num_levels, params.memory_window),
        }
    }
    
    /// Create with custom parameters
    pub fn with_params(num_levels: usize, bid_params: HawkesParams, ask_params: HawkesParams) -> Self {
        Self {
            bid_history: FillHistory::new(num_levels, bid_params.memory_window),
            ask_history: FillHistory::new(num_levels, ask_params.memory_window),
            bid_params,
            ask_params,
        }
    }
    
    /// Record a fill event
    pub fn record_fill(&mut self, level: usize, is_bid: bool, timestamp: f64) {
        if is_bid {
            self.bid_history.record(level, timestamp);
        } else {
            self.ask_history.record(level, timestamp);
        }
    }
    
    /// Compute bid fill intensity at a level
    /// λ^b(t) = ν^b(δ, I) + excitation
    /// where ν^b = Λ^b * (1 - β*I) * exp(-κ*δ) * ρ^(i-1)
    pub fn bid_fill_intensity(
        &self,
        level: usize,
        offset_bps: f64,
        lob_imbalance: f64,
        current_time: f64,
    ) -> f64 {
        // Base intensity (GLFT model)
        let delta = offset_bps / 10000.0;
        let lob_factor = (1.0 - self.bid_params.beta_imbalance * lob_imbalance).max(EPSILON);
        let price_decay = (-self.bid_params.kappa * delta).exp();
        let queue_penalty = self.bid_params.rho.powi(level.saturating_sub(1) as i32);
        
        let base_intensity = self.bid_params.lambda_base 
            * lob_factor 
            * price_decay 
            * queue_penalty;
        
        // Self-excitation term (Hawkes)
        let excitation = self.bid_history.excitation(level, current_time, &self.bid_params);
        
        base_intensity + excitation
    }
    
    /// Compute ask fill intensity at a level
    pub fn ask_fill_intensity(
        &self,
        level: usize,
        offset_bps: f64,
        lob_imbalance: f64,
        current_time: f64,
    ) -> f64 {
        let delta = offset_bps / 10000.0;
        let lob_factor = (self.ask_params.beta_imbalance * lob_imbalance).max(EPSILON);
        let price_decay = (-self.ask_params.kappa * delta).exp();
        let queue_penalty = self.ask_params.rho.powi(level.saturating_sub(1) as i32);
        
        let base_intensity = self.ask_params.lambda_base 
            * lob_factor 
            * price_decay 
            * queue_penalty;
        
        let excitation = self.ask_history.excitation(level, current_time, &self.ask_params);
        
        base_intensity + excitation
    }
    
    /// Get excitation multiplier (1.0 = no excitation, >1.0 = momentum)
    pub fn excitation_multiplier(&self, level: usize, is_bid: bool, current_time: f64) -> f64 {
        let params = if is_bid { &self.bid_params } else { &self.ask_params };
        let history = if is_bid { &self.bid_history } else { &self.ask_history };
        
        let excitation = history.excitation(level, current_time, params);
        let base = params.lambda_base;
        
        1.0 + (excitation / base.max(EPSILON))
    }
    
    /// Check if momentum is detected (high excitation)
    pub fn has_momentum(&self, level: usize, is_bid: bool, current_time: f64, threshold: f64) -> bool {
        self.excitation_multiplier(level, is_bid, current_time) > threshold
    }
    
    /// Get recent fill count for diagnostics
    pub fn recent_fills(&self, level: usize, is_bid: bool) -> usize {
        if is_bid {
            self.bid_history.recent_fill_count(level)
        } else {
            self.ask_history.recent_fill_count(level)
        }
    }
    
    /// Clear all fill history
    pub fn clear_history(&mut self) {
        self.bid_history.clear();
        self.ask_history.clear();
    }
}

/// Hawkes process-based fill model implementation (component wrapper).
///
/// This component wraps a HawkesFillModelCore and provides the FillModel
/// interface for use in modular strategies.
pub struct HawkesFillModelImpl {
    /// The underlying Hawkes fill model
    model: HawkesFillModelCore,
}

impl HawkesFillModelImpl {
    /// Create a new Hawkes fill model with default parameters (3 levels).
    pub fn new_default() -> Self {
        Self::new(3)
    }

    /// Create a new Hawkes fill model with custom number of levels.
    pub fn new(num_levels: usize) -> Self {
        Self {
            model: HawkesFillModelCore::new(num_levels),
        }
    }

    /// Create with custom parameters
    pub fn with_params(num_levels: usize, bid_params: HawkesParams, ask_params: HawkesParams) -> Self {
        Self {
            model: HawkesFillModelCore::with_params(num_levels, bid_params, ask_params),
        }
    }

    /// Get reference to the underlying model core
    pub fn model(&self) -> &HawkesFillModelCore {
        &self.model
    }

    /// Get mutable reference to the underlying model core
    pub fn model_mut(&mut self) -> &mut HawkesFillModelCore {
        &mut self.model
    }
}

impl FillModel for HawkesFillModelImpl {
    fn on_fills(&mut self, fills: &[TradeInfo], current_time_sec: f64) {
        for fill in fills {
            // Determine if this was a bid fill or ask fill
            // Convention from market_maker_v2.rs:
            // "B" = buy side = we got filled on our bid
            // "A" = sell side = we got filled on our ask
            let is_bid_fill = fill.side == "B";

            // For now, assume all fills are at L1 (level 0)
            // TODO: Extract level information from fill metadata
            let level = 0;

            self.model.record_fill(level, is_bid_fill, current_time_sec);
        }
    }

    fn get_hawkes_model(&self) -> &crate::HawkesFillModel {
        // SAFETY: This is a temporary workaround to bridge between the component
        // version (HawkesFillModelCore) and the crate-level HawkesFillModel from
        // hawkes_multi_level.rs.
        //
        // The proper solution is to update all code to use the component version,
        // but for now we cast between them since they have the same memory layout.
        //
        // TODO: Remove this once hawkes_multi_level.rs is fully deprecated
        unsafe {
            // Both types have identical structure, so we can transmute
            std::mem::transmute::<&HawkesFillModelCore, &crate::HawkesFillModel>(&self.model)
        }
    }
}
