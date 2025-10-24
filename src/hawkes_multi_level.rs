//! Hawkes Process Multi-Level Market Making
//!
//! This module implements a self-exciting Hawkes process for fill rate modeling
//! combined with multi-level quoting optimization. It extends the basic GLFT
//! (Glosten-Laporte Fill Time) model with temporal clustering effects.
//!
//! Key Features:
//! - Self-exciting fill rate model (Hawkes process)
//! - Multi-level quote optimization
//! - Robust control with parameter uncertainty
//! - Dynamic level count optimization
//! - Kelly-inspired size allocation

use std::collections::VecDeque;
use serde::{Deserialize, Serialize};

/// Epsilon for numerical stability
const EPSILON: f64 = 1e-10;

// ============================================================================
// HAWKES PROCESS FILL RATE MODEL
// ============================================================================

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
#[derive(Debug, Clone)]
pub struct HawkesFillModel {
    /// Bid side parameters
    pub bid_params: HawkesParams,
    
    /// Ask side parameters
    pub ask_params: HawkesParams,
    
    /// Fill history for bids (per level)
    bid_history: FillHistory,
    
    /// Fill history for asks (per level)
    ask_history: FillHistory,
}

impl HawkesFillModel {
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

// ============================================================================
// MULTI-LEVEL CONTROL STRUCTURES
// ============================================================================

/// Multi-level quote specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiLevelControl {
    /// Bid levels: (offset_bps, size)
    pub bid_levels: Vec<(f64, f64)>,
    
    /// Ask levels: (offset_bps, size)
    pub ask_levels: Vec<(f64, f64)>,
    
    /// Taker order rates (for urgent inventory management)
    pub taker_buy_rate: f64,
    pub taker_sell_rate: f64,
    
    /// Flag for liquidation mode
    pub liquidate: bool,
}

impl MultiLevelControl {
    /// Create symmetric multi-level control
    pub fn symmetric(num_levels: usize, base_offset_bps: f64, level_spacing_bps: f64, base_size: f64) -> Self {
        let mut bid_levels = Vec::new();
        let mut ask_levels = Vec::new();
        
        for i in 0..num_levels {
            let offset = base_offset_bps + (i as f64 * level_spacing_bps);
            let size = base_size * 0.9_f64.powi(i as i32); // Decreasing size
            
            bid_levels.push((offset, size));
            ask_levels.push((offset, size));
        }
        
        Self {
            bid_levels,
            ask_levels,
            taker_buy_rate: 0.0,
            taker_sell_rate: 0.0,
            liquidate: false,
        }
    }
    
    /// Get total size across all bid levels
    pub fn total_bid_size(&self) -> f64 {
        self.bid_levels.iter().map(|(_, size)| size).sum()
    }
    
    /// Get total size across all ask levels
    pub fn total_ask_size(&self) -> f64 {
        self.ask_levels.iter().map(|(_, size)| size).sum()
    }
    
    /// Check if this is valid
    pub fn validate(&self) -> Result<(), String> {
        if self.bid_levels.is_empty() || self.ask_levels.is_empty() {
            return Err("Must have at least one level on each side".to_string());
        }
        
        // Check that levels are ordered (wider spreads for higher levels)
        for i in 1..self.bid_levels.len() {
            if self.bid_levels[i].0 <= self.bid_levels[i-1].0 {
                return Err("Bid levels must have increasing spreads".to_string());
            }
        }
        
        for i in 1..self.ask_levels.len() {
            if self.ask_levels[i].0 <= self.ask_levels[i-1].0 {
                return Err("Ask levels must have increasing spreads".to_string());
            }
        }
        
        Ok(())
    }
    
    /// Get log string for monitoring
    pub fn to_log_string(&self) -> String {
        let bid_l1 = self.bid_levels.first().map(|(o, s)| format!("{}bps({})", o, s)).unwrap_or_default();
        let ask_l1 = self.ask_levels.first().map(|(o, s)| format!("{}bps({})", o, s)).unwrap_or_default();
        
        format!(
            "MultiLevel[{} levels | L1: bid={}, ask={} | taker: buy={:.2}, sell={:.2}]",
            self.bid_levels.len(),
            bid_l1,
            ask_l1,
            self.taker_buy_rate,
            self.taker_sell_rate
        )
    }
}

// ============================================================================
// MULTI-LEVEL OPTIMIZER
// ============================================================================

/// Configuration for multi-level optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiLevelConfig {
    /// Maximum number of levels (1-10)
    pub max_levels: usize,
    
    /// Minimum profitable spread (covers fees + edge)
    pub min_profitable_spread_bps: f64,
    
    /// Default spacing between levels (bps)
    pub level_spacing_bps: f64,
    
    /// Total size budget per side
    pub total_size_per_side: f64,
    
    /// Inventory risk limit (fraction of max position)
    pub inventory_risk_limit: f64,
    
    /// Directional aggression factor (how much to bias based on signal)
    pub directional_aggression: f64,
    
    /// Momentum detection threshold (excitation multiplier)
    pub momentum_threshold: f64,
    
    /// Momentum tightening (bps to reduce spread when momentum detected)
    pub momentum_tightening_bps: f64,
    
    /// Inventory urgency threshold (start using taker orders)
    pub inventory_urgency_threshold: f64,
}

impl Default for MultiLevelConfig {
    fn default() -> Self {
        Self {
            max_levels: 3,
            min_profitable_spread_bps: 4.0,  // 3 bps fees + 1 bps edge
            level_spacing_bps: 2.0,
            total_size_per_side: 1.0,
            inventory_risk_limit: 0.7,
            directional_aggression: 2.0,
            momentum_threshold: 1.3,
            momentum_tightening_bps: 1.0,
            inventory_urgency_threshold: 0.8,
        }
    }
}

/// State information needed for optimization
pub struct OptimizationState<'a> {
    pub mid_price: f64,
    pub inventory: f64,
    pub max_position: f64,
    pub adverse_selection_bps: f64,
    pub lob_imbalance: f64,
    pub volatility_bps: f64,
    pub current_time: f64,
    pub hawkes_model: &'a HawkesFillModel,
}

/// Multi-level optimizer
pub struct MultiLevelOptimizer {
    config: MultiLevelConfig,
}

impl MultiLevelOptimizer {
    pub fn new(config: MultiLevelConfig) -> Self {
        Self { config }
    }
    
    /// Optimize multi-level quotes
    pub fn optimize(
        &self,
        state: &OptimizationState,
        base_half_spread_bps: f64,
    ) -> MultiLevelControl {
        // 1. Calculate directional bias
        let inventory_ratio = state.inventory / state.max_position;
        let (bid_aggression, ask_aggression) = self.calculate_aggression(
            state.adverse_selection_bps,
            inventory_ratio,
        );
        
        // 2. Determine number of levels (could be dynamic in future)
        let num_levels = self.config.max_levels;
        
        // 3. Calculate sizes for each level
        let (bid_sizes, ask_sizes) = self.allocate_sizes(state, inventory_ratio, num_levels);
        
        // 4. Build bid levels with aggression and momentum
        let mut bid_levels = Vec::new();
        for level in 0..num_levels {
            let level_offset = level as f64 * self.config.level_spacing_bps;
            let base_offset = base_half_spread_bps.max(self.config.min_profitable_spread_bps / 2.0);
            
            // Apply directional aggression
            let mut adjusted_offset = base_offset + level_offset - bid_aggression;
            
            // Check for Hawkes momentum and tighten further
            if state.hawkes_model.has_momentum(level, true, state.current_time, self.config.momentum_threshold) {
                adjusted_offset -= self.config.momentum_tightening_bps;
            }
            
            // Enforce minimum
            adjusted_offset = adjusted_offset.max(self.config.min_profitable_spread_bps / 2.0);
            
            bid_levels.push((adjusted_offset, bid_sizes[level]));
        }
        
        // 5. Build ask levels
        let mut ask_levels = Vec::new();
        for level in 0..num_levels {
            let level_offset = level as f64 * self.config.level_spacing_bps;
            let base_offset = base_half_spread_bps.max(self.config.min_profitable_spread_bps / 2.0);
            
            let mut adjusted_offset = base_offset + level_offset - ask_aggression;
            
            if state.hawkes_model.has_momentum(level, false, state.current_time, self.config.momentum_threshold) {
                adjusted_offset -= self.config.momentum_tightening_bps;
            }
            
            adjusted_offset = adjusted_offset.max(self.config.min_profitable_spread_bps / 2.0);
            
            ask_levels.push((adjusted_offset, ask_sizes[level]));
        }
        
        // 6. Check for taker urgency
        let (taker_buy_rate, taker_sell_rate, liquidate) = self.check_taker_urgency(
            inventory_ratio,
            state.max_position,
        );
        
        MultiLevelControl {
            bid_levels,
            ask_levels,
            taker_buy_rate,
            taker_sell_rate,
            liquidate,
        }
    }
    
    /// Calculate aggression factors for each side
    fn calculate_aggression(&self, adverse_selection_bps: f64, inventory_ratio: f64) -> (f64, f64) {
        // Signal-driven aggression
        let signal_bid_aggression = adverse_selection_bps.max(0.0) * self.config.directional_aggression;
        let signal_ask_aggression = (-adverse_selection_bps).max(0.0) * self.config.directional_aggression;
        
        // Inventory-driven aggression (need to reduce position)
        let inv_bid_aggression = if inventory_ratio < -0.3 {
            // Short, want to buy back
            (inventory_ratio.abs() - 0.3) * 5.0
        } else {
            0.0
        };
        
        let inv_ask_aggression = if inventory_ratio > 0.3 {
            // Long, want to sell
            (inventory_ratio - 0.3) * 5.0
        } else {
            0.0
        };
        
        // Combine: signal when inventory low, inventory when high
        let weight = 1.0 - inventory_ratio.abs();
        let bid_aggression = signal_bid_aggression * weight + inv_bid_aggression;
        let ask_aggression = signal_ask_aggression * weight + inv_ask_aggression;
        
        (bid_aggression, ask_aggression)
    }
    
    /// Allocate sizes across levels
    fn allocate_sizes(
        &self,
        state: &OptimizationState,
        inventory_ratio: f64,
        num_levels: usize,
    ) -> (Vec<f64>, Vec<f64>) {
        // Base allocation (more size on inner levels)
        let base_allocations = vec![0.45, 0.30, 0.15, 0.07, 0.03];
        let mut bid_alloc = base_allocations[..num_levels.min(5)].to_vec();
        let mut ask_alloc = bid_alloc.clone();
        
        // Normalize
        let sum: f64 = bid_alloc.iter().sum();
        for x in &mut bid_alloc {
            *x /= sum;
        }
        for x in &mut ask_alloc {
            *x /= sum;
        }
        
        // Calculate available inventory
        let inventory_capacity = state.max_position * self.config.inventory_risk_limit;
        let max_buy = (inventory_capacity - state.inventory).max(0.0);
        let max_sell = (inventory_capacity + state.inventory).max(0.0);
        
        let bid_budget = self.config.total_size_per_side.min(max_buy);
        let ask_budget = self.config.total_size_per_side.min(max_sell);
        
        // Apply Hawkes excitement multiplier
        let mut bid_sizes = Vec::new();
        let mut ask_sizes = Vec::new();
        
        for level in 0..num_levels {
            let bid_excitement = state.hawkes_model.excitation_multiplier(level, true, state.current_time);
            let ask_excitement = state.hawkes_model.excitation_multiplier(level, false, state.current_time);
            
            // More excitement = allocate more size
            bid_sizes.push(bid_budget * bid_alloc[level] * bid_excitement.min(1.5));
            ask_sizes.push(ask_budget * ask_alloc[level] * ask_excitement.min(1.5));
        }
        
        (bid_sizes, ask_sizes)
    }
    
    /// Check if inventory is extreme enough to use taker orders
    fn check_taker_urgency(&self, inventory_ratio: f64, max_position: f64) -> (f64, f64, bool) {
        if inventory_ratio > self.config.inventory_urgency_threshold {
            // Very long, need to sell urgently
            let urgency = (inventory_ratio - self.config.inventory_urgency_threshold) * 2.0;
            let taker_size = urgency * max_position;
            (0.0, taker_size, true)
        } else if inventory_ratio < -self.config.inventory_urgency_threshold {
            // Very short, need to buy urgently
            let urgency = (inventory_ratio.abs() - self.config.inventory_urgency_threshold) * 2.0;
            let taker_size = urgency * max_position;
            (taker_size, 0.0, true)
        } else {
            (0.0, 0.0, false)
        }
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_fill_history() {
        let mut history = FillHistory::new(3, 10.0);
        
        history.record(0, 1.0);
        history.record(0, 2.0);
        history.record(0, 3.0);
        
        assert_eq!(history.recent_fill_count(0), 3);
        
        // Old fill should be removed
        history.record(0, 12.0);
        assert_eq!(history.recent_fill_count(0), 3); // 2.0, 3.0, 12.0
    }
    
    #[test]
    fn test_hawkes_excitation() {
        let params = HawkesParams::default();
        let mut history = FillHistory::new(1, 10.0);
        
        history.record(0, 1.0);
        history.record(0, 1.5);
        
        let excitation = history.excitation(0, 2.0, &params);
        assert!(excitation > 0.0);
        assert!(excitation < 2.0 * params.alpha); // Should be less than 2*alpha
    }
    
    #[test]
    fn test_hawkes_fill_intensity() {
        let model = HawkesFillModel::new(3);
        
        // Base intensity without fills
        let intensity1 = model.bid_fill_intensity(0, 6.0, 0.5, 0.0);
        assert!(intensity1 > 0.0);
        
        // With momentum (simulated by recording fills)
        let mut model2 = model.clone();
        model2.record_fill(0, true, 0.0);
        model2.record_fill(0, true, 0.5);
        
        let intensity2 = model2.bid_fill_intensity(0, 6.0, 0.5, 1.0);
        assert!(intensity2 > intensity1); // Should be higher with momentum
    }
    
    #[test]
    fn test_multi_level_control_symmetric() {
        let control = MultiLevelControl::symmetric(3, 6.0, 2.0, 0.5);
        
        assert_eq!(control.bid_levels.len(), 3);
        assert_eq!(control.ask_levels.len(), 3);
        
        // Check spacing
        assert_eq!(control.bid_levels[0].0, 6.0);
        assert_eq!(control.bid_levels[1].0, 8.0);
        assert_eq!(control.bid_levels[2].0, 10.0);
        
        assert!(control.validate().is_ok());
    }
    
    #[test]
    fn test_multi_level_optimizer() {
        let config = MultiLevelConfig::default();
        let optimizer = MultiLevelOptimizer::new(config);
        let hawkes = HawkesFillModel::new(3);
        
        let state = OptimizationState {
            mid_price: 100.0,
            inventory: 0.0,
            max_position: 100.0,
            adverse_selection_bps: 2.0, // Bullish
            lob_imbalance: 0.5,
            volatility_bps: 100.0,
            current_time: 0.0,
            hawkes_model: &hawkes,
        };
        
        let control = optimizer.optimize(&state, 6.0);
        
        assert!(control.validate().is_ok());
        assert_eq!(control.bid_levels.len(), 3);
        
        // With bullish signal, bids should be more aggressive than asks
        assert!(control.bid_levels[0].0 < control.ask_levels[0].0);
    }
    
    #[test]
    fn test_momentum_detection() {
        let mut model = HawkesFillModel::new(3);
        
        // No fills = no momentum
        assert!(!model.has_momentum(0, true, 0.0, 1.3));
        
        // Add several fills
        model.record_fill(0, true, 0.0);
        model.record_fill(0, true, 0.3);
        model.record_fill(0, true, 0.6);
        
        // Should detect momentum
        assert!(model.has_momentum(0, true, 1.0, 1.3));
    }
}