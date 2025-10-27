// ============================================================================
// Hawkes Fill Model Component - Self-Exciting Point Process Fill Estimation
// ============================================================================
//
// This component implements the FillModel trait using Hawkes processes
// (self-exciting point processes) to model fill rates at multiple price levels.
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
//
// # Example
//
// ```rust
// use strategies::components::{FillModel, HawkesFillModelImpl};
//
// let mut fill_model = HawkesFillModelImpl::new_default();
//
// // When we get filled
// let fills = vec![fill1, fill2];
// let current_time = 1234567890.0;
// fill_model.on_fills(&fills, current_time);
//
// // Query fill rates
// let hawkes = fill_model.get_hawkes_model();
// let bid_intensity = hawkes.compute_intensity(0, true, current_time);
// ```

use std::sync::Arc;
use parking_lot::RwLock;

use crate::TradeInfo;
use crate::HawkesFillModel;
use super::fill_model::FillModel;

/// Hawkes process-based fill model implementation.
///
/// This component wraps a HawkesFillModel and provides the FillModel
/// interface for use in modular strategies.
pub struct HawkesFillModelImpl {
    /// The underlying Hawkes fill model
    hawkes: Arc<RwLock<HawkesFillModel>>,
}

impl HawkesFillModelImpl {
    /// Create a new Hawkes fill model with default parameters (3 levels).
    ///
    /// Default configuration:
    /// - 3 price levels (L1, L2, L3)
    /// - Each level has independent Hawkes process for bids and asks
    pub fn new_default() -> Self {
        Self::new(3)
    }

    /// Create a new Hawkes fill model with custom number of levels.
    ///
    /// # Arguments
    /// - `num_levels`: Number of price levels to track (typically 1-5)
    ///
    /// # Notes
    /// - More levels = more granular fill rate modeling
    /// - Each level doubles the number of Hawkes processes (bid + ask)
    /// - Levels are numbered 0 (L1/tightest), 1 (L2), 2 (L3), etc.
    pub fn new(num_levels: usize) -> Self {
        let hawkes = Arc::new(RwLock::new(HawkesFillModel::new(num_levels)));

        Self { hawkes }
    }

    /// Get reference to the underlying Hawkes model (for advanced usage).
    pub fn hawkes(&self) -> Arc<RwLock<HawkesFillModel>> {
        Arc::clone(&self.hawkes)
    }
}

impl FillModel for HawkesFillModelImpl {
    fn on_fills(&mut self, fills: &[TradeInfo], current_time_sec: f64) {
        let mut hawkes = self.hawkes.write();

        for fill in fills {
            // Determine if this was a bid fill or ask fill
            // Convention: "B" = buy side = we got filled on our ask
            //             "A" = sell side = we got filled on our bid
            let is_bid_fill = fill.side == "A" || fill.side == "sell";

            // For now, assume all fills are at L1 (level 0)
            // In a more sophisticated implementation, we would:
            // 1. Track which order/level got filled
            // 2. Record the fill at the correct level
            //
            // TODO: Extract level information from fill.cloid or track order metadata
            let level = 0;

            hawkes.record_fill(level, is_bid_fill, current_time_sec);
        }
    }

    fn get_hawkes_model(&self) -> &HawkesFillModel {
        // SAFETY: This is not truly safe - we're returning a reference to data
        // behind a RwLock, which could cause a panic if the lock is poisoned.
        //
        // TODO: Refactor this interface to return Arc<RwLock<HawkesFillModel>>
        // or define a more generic FillRateProvider trait that doesn't expose
        // the concrete HawkesFillModel type.
        //
        // For now, this is a stopgap to maintain compatibility with the existing
        // optimizer interface.
        unsafe {
            // Get a raw pointer to the HawkesFillModel inside the RwLock
            let ptr = self.hawkes.data_ptr();
            &*ptr
        }
    }
}
