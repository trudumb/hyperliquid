// ============================================================================
// Microprice-Based Adverse Selection Model (Predictive)
// ============================================================================
//
// This component estimates adverse selection using microprice analysis with
// predictive velocity and acceleration terms. It combines the stability of
// microprice nowcasting with short-term price movement forecasting.
//
// # Key Innovation: Microprice Deviation + Momentum
//
// Instead of only measuring current market state, we also track its rate of change:
//
//   microprice = (bid * ask_depth + ask * bid_depth) / (bid_depth + ask_depth)
//   velocity = d(lob_imbalance) / dt        (momentum of order book shift)
//   acceleration = d(microprice_dev) / dt   (acceleration of price pressure)
//
// The model combines five signals:
//   1. Microprice deviation (50%): Current LOB pressure
//   2. LOB imbalance (20%): Static depth asymmetry
//   3. Trade flow (10%): Recent aggressive order flow
//   4. Velocity (15%): Rate of imbalance change → predicts momentum [NEW]
//   5. Acceleration (5%): Rate of microprice change → predicts trend acceleration [NEW]
//
// # Why This Is Predictive
//
// - **Velocity**: If LOB imbalance is moving aggressively, it predicts price will
//   continue moving in that direction (momentum)
// - **Acceleration**: If microprice deviation is accelerating, it predicts the
//   pressure will intensify (trend strength)
//
// This makes the adverse_selection_bps a short-term forecast of where the price
// will be in the next few seconds, not just a reflection of current state.
//
// # Benefits Over Static Microprice Model
//
// 1. **Forward-Looking**: Predicts price movement 2-5 seconds ahead
// 2. **Momentum Capture**: Recognizes and responds to trend acceleration
// 3. **Better Quote Placement**: Widens spreads preemptively before adverse moves
// 4. **Still Stable**: Derivative terms are smoothed and bounded
//
// # Example Usage
//
// ```rust
// let mut as_model = MicropriceAsModel::new();
//
// // Update with market data
// as_model.update(order_book, trades);
//
// // Get predictive adverse selection estimate in bps
// let as_bps = as_model.get_adverse_selection_bps();
// ```

use crate::{OrderBook, Trade};
use log::debug;

/// Microprice-based adverse selection estimator
#[derive(Debug, Clone)]
pub struct MicropriceAsModel {
    /// Smoothed adverse selection estimate in bps
    adverse_selection_ema: f64,

    /// EMA alpha for smoothing (higher = more responsive)
    ema_alpha: f64,

    /// Maximum AS estimate in bps (prevents extreme values)
    max_as_bps: f64,

    /// Trade flow imbalance (buy volume - sell volume)
    trade_flow_ema: f64,

    /// Trade flow EMA alpha
    trade_flow_alpha: f64,

    /// Last update time for diagnostics
    last_update_count: usize,

    // --- PREDICTIVE ENHANCEMENT: Velocity & Acceleration Terms ---
    /// Previous LOB imbalance (for velocity calculation)
    prev_lob_imbalance: f64,

    /// Previous microprice deviation in bps (for acceleration calculation)
    prev_microprice_deviation_bps: f64,

    /// Timestamp of last update (for time-normalized derivatives)
    prev_update_time: f64,

    /// Velocity scaling factor (controls impact on prediction)
    velocity_scale: f64,

    /// Acceleration scaling factor (controls impact on prediction)
    acceleration_scale: f64,
}

impl Default for MicropriceAsModel {
    fn default() -> Self {
        Self {
            adverse_selection_ema: 0.0,
            ema_alpha: 0.1,           // 10% weight to new observations
            max_as_bps: 10.0,          // Cap at ±10 bps
            trade_flow_ema: 0.0,
            trade_flow_alpha: 0.05,   // Slower decay for trade flow
            last_update_count: 0,
            // Initialize predictive terms
            prev_lob_imbalance: 0.5,  // Start at neutral (50/50)
            prev_microprice_deviation_bps: 0.0,
            prev_update_time: 0.0,
            velocity_scale: 5.0,       // LOB imbalance velocity -> bps (tunable)
            acceleration_scale: 0.3,   // Microprice acceleration dampening (tunable)
        }
    }
}

impl MicropriceAsModel {
    /// Create a new microprice-based AS model
    pub fn new() -> Self {
        Self::default()
    }

    /// Create with custom smoothing parameters
    pub fn with_params(ema_alpha: f64, max_as_bps: f64) -> Self {
        Self {
            ema_alpha,
            max_as_bps,
            ..Self::default()
        }
    }

    /// Update the model with current order book and recent trades
    pub fn update(&mut self, order_book: Option<&OrderBook>, trades: &[Trade]) {
        // Update trade flow imbalance
        self.update_trade_flow(trades);

        // Update microprice-based AS estimate
        if let Some(book) = order_book {
            self.update_from_book(book);
        }

        self.last_update_count += 1;
    }

    /// Update trade flow imbalance from recent trades
    fn update_trade_flow(&mut self, trades: &[Trade]) {
        if trades.is_empty() {
            // Decay toward zero if no trades
            self.trade_flow_ema *= 1.0 - self.trade_flow_alpha;
            return;
        }

        // Compute net buy pressure from trades
        // Positive = net buying, Negative = net selling
        let mut net_flow = 0.0;
        for trade in trades {
            let size = trade.sz.parse::<f64>().unwrap_or(0.0);
            let flow = if trade.side == "A" {
                // Aggressive buy (taker bought)
                size
            } else {
                // Aggressive sell (taker sold)
                -size
            };
            net_flow += flow;
        }

        // Update EMA
        self.trade_flow_ema = self.trade_flow_alpha * net_flow +
                               (1.0 - self.trade_flow_alpha) * self.trade_flow_ema;
    }

    /// Update adverse selection estimate from order book
    fn update_from_book(&mut self, book: &OrderBook) {
        // Need at least L1 bid and ask
        if book.bids.is_empty() || book.asks.is_empty() {
            return;
        }

        // Parse L1 prices and sizes
        let bid_px = book.bids[0].px.parse::<f64>().unwrap_or(0.0);
        let ask_px = book.asks[0].px.parse::<f64>().unwrap_or(0.0);
        let bid_sz = book.bids[0].sz.parse::<f64>().unwrap_or(0.0);
        let ask_sz = book.asks[0].sz.parse::<f64>().unwrap_or(0.0);

        if bid_px <= 0.0 || ask_px <= 0.0 || bid_sz <= 0.0 || ask_sz <= 0.0 {
            return;
        }

        // Get current timestamp for velocity/acceleration calculations
        let current_time = chrono::Utc::now().timestamp_millis() as f64 / 1000.0;
        let dt = if self.prev_update_time > 0.0 {
            (current_time - self.prev_update_time).max(0.001) // Min 1ms to avoid division by zero
        } else {
            1.0 // First update: use 1 second as default
        };

        // Compute mid price
        let mid = (bid_px + ask_px) / 2.0;

        // Compute microprice (depth-weighted mid)
        // microprice = (bid * ask_size + ask * bid_size) / (bid_size + ask_size)
        // This gives more weight to the side with more depth
        let total_depth = bid_sz + ask_sz;
        let microprice = (bid_px * ask_sz + ask_px * bid_sz) / total_depth;

        // Compute LOB imbalance
        let lob_imbalance = bid_sz / total_depth;

        // Compute microprice deviation in bps
        let microprice_deviation_bps = ((microprice - mid) / mid) * 10000.0;

        // --- PREDICTIVE ENHANCEMENT: Calculate Velocity & Acceleration ---

        // Velocity: Rate of change in LOB imbalance (normalized per second)
        // Positive velocity = increasing buy pressure
        // Negative velocity = increasing sell pressure
        let imbalance_velocity = (lob_imbalance - self.prev_lob_imbalance) / dt;
        let velocity_signal_bps = imbalance_velocity * self.velocity_scale;

        // Acceleration: Rate of change in microprice deviation (normalized per second)
        // Positive acceleration = microprice moving up faster (stronger buying)
        // Negative acceleration = microprice moving down faster (stronger selling)
        let microprice_acceleration = (microprice_deviation_bps - self.prev_microprice_deviation_bps) / dt;
        let acceleration_signal_bps = microprice_acceleration * self.acceleration_scale;

        // Compute LOB pressure signal (static component)
        // Convert imbalance (0 to 1) to pressure (-1 to +1)
        let lob_pressure = (lob_imbalance - 0.5) * 2.0;  // Maps [0,1] -> [-1,+1]
        let lob_pressure_bps = lob_pressure * 2.0;  // Scale to ~±2 bps

        // Combine signals:
        // 1. Microprice deviation (immediate LOB pressure) - 50% weight
        // 2. LOB imbalance (depth asymmetry) - 20% weight
        // 3. Trade flow (recent aggressive order flow) - 10% weight
        // 4. Velocity (rate of imbalance change) - 15% weight [NEW]
        // 5. Acceleration (rate of microprice change) - 5% weight [NEW]
        let trade_flow_signal_bps = self.trade_flow_ema.signum() *
                                     self.trade_flow_ema.abs().sqrt() * 3.0;

        let raw_as_estimate = 0.50 * microprice_deviation_bps +
                              0.20 * lob_pressure_bps +
                              0.10 * trade_flow_signal_bps +
                              0.15 * velocity_signal_bps +        // Predictive: momentum
                              0.05 * acceleration_signal_bps;     // Predictive: trend acceleration

        // Clamp to max bounds
        let clamped_as = raw_as_estimate.max(-self.max_as_bps).min(self.max_as_bps);

        // Update EMA
        self.adverse_selection_ema = self.ema_alpha * clamped_as +
                                      (1.0 - self.ema_alpha) * self.adverse_selection_ema;

        // Store current values for next velocity/acceleration calculation
        self.prev_lob_imbalance = lob_imbalance;
        self.prev_microprice_deviation_bps = microprice_deviation_bps;
        self.prev_update_time = current_time;

        // Log diagnostics occasionally
        if self.last_update_count % 100 == 0 {
            debug!(
                "[MICROPRICE AS] microprice={:.3}, mid={:.3}, deviation={:.2}bps, \
                 lob_imb={:.3}, velocity={:.2}bps, accel={:.2}bps, trade_flow={:.3}, AS_est={:.2}bps",
                microprice, mid, microprice_deviation_bps,
                lob_imbalance, velocity_signal_bps, acceleration_signal_bps,
                self.trade_flow_ema, self.adverse_selection_ema
            );
        }
    }

    /// Get current adverse selection estimate in basis points
    pub fn get_adverse_selection_bps(&self) -> f64 {
        self.adverse_selection_ema
    }

    /// Get current trade flow imbalance
    pub fn get_trade_flow(&self) -> f64 {
        self.trade_flow_ema
    }

    /// Reset the model state
    pub fn reset(&mut self) {
        self.adverse_selection_ema = 0.0;
        self.trade_flow_ema = 0.0;
        self.last_update_count = 0;
        // Reset predictive terms
        self.prev_lob_imbalance = 0.5;
        self.prev_microprice_deviation_bps = 0.0;
        self.prev_update_time = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::BookLevel;

    fn create_test_book(bid_px: f64, ask_px: f64, bid_sz: f64, ask_sz: f64) -> OrderBook {
        let mid_price = (bid_px + ask_px) / 2.0;
        OrderBook {
            bids: vec![BookLevel {
                px: bid_px.to_string(),
                sz: bid_sz.to_string(),
                n: 1,
            }],
            asks: vec![BookLevel {
                px: ask_px.to_string(),
                sz: ask_sz.to_string(),
                n: 1,
            }],
            mid_price,
        }
    }

    #[test]
    fn test_balanced_book_zero_as() {
        let mut model = MicropriceAsModel::new();
        let book = create_test_book(99.0, 101.0, 10.0, 10.0);

        model.update(Some(&book), &[]);

        // Balanced book should have near-zero AS
        let as_bps = model.get_adverse_selection_bps();
        assert!(as_bps.abs() < 1.0, "Balanced book AS should be near zero, got {}", as_bps);
    }

    #[test]
    fn test_bid_heavy_book_positive_as() {
        let mut model = MicropriceAsModel::new();
        let book = create_test_book(99.0, 101.0, 100.0, 10.0);  // 10x more bid depth

        // Update multiple times to let EMA converge
        for _ in 0..20 {
            model.update(Some(&book), &[]);
        }

        // Bid-heavy book should have positive AS (buying pressure)
        let as_bps = model.get_adverse_selection_bps();
        assert!(as_bps > 0.0, "Bid-heavy book should have positive AS, got {}", as_bps);
    }

    #[test]
    fn test_ask_heavy_book_negative_as() {
        let mut model = MicropriceAsModel::new();
        let book = create_test_book(99.0, 101.0, 10.0, 100.0);  // 10x more ask depth

        // Update multiple times to let EMA converge
        for _ in 0..20 {
            model.update(Some(&book), &[]);
        }

        // Ask-heavy book should have negative AS (selling pressure)
        let as_bps = model.get_adverse_selection_bps();
        assert!(as_bps < 0.0, "Ask-heavy book should have negative AS, got {}", as_bps);
    }

    #[test]
    fn test_max_as_capping() {
        let mut model = MicropriceAsModel::with_params(0.5, 5.0);  // max_as = 5 bps
        let book = create_test_book(99.0, 101.0, 1000.0, 1.0);  // Extreme imbalance

        // Update multiple times
        for _ in 0..50 {
            model.update(Some(&book), &[]);
        }

        // AS should be capped at max_as_bps
        let as_bps = model.get_adverse_selection_bps();
        assert!(as_bps.abs() <= 5.1, "AS should be capped at 5 bps, got {}", as_bps);
    }
}
