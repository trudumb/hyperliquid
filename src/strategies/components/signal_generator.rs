// ============================================================================
// HJB Signal Generator - Pure Signal Generation
// ============================================================================
//
// Generates theoretical quotes based on HJB optimization without any
// order management or risk adjustments. This is a pure calculation layer
// that takes market state and returns optimal quotes.

use std::sync::Arc;
use parking_lot::RwLock;
use log::debug;

use crate::{HawkesFillModel, MultiLevelConfig};
use super::{
    HjbMultiLevelOptimizer, OptimizerInputs, OptimizerOutput,
    VolatilityModel, MicropriceAsModel,
};
use super::trading_state_store::MarketData;

// ============================================================================
// Quote Signal
// ============================================================================

/// Pure quote signal from the optimizer (before risk adjustments)
#[derive(Debug, Clone)]
pub struct QuoteSignal {
    /// Bid levels (price, size, urgency)
    pub bid_levels: Vec<QuoteLevel>,

    /// Ask levels (price, size, urgency)
    pub ask_levels: Vec<QuoteLevel>,

    /// Overall urgency score (0-1, higher = more urgent to trade)
    pub urgency: f64,

    /// Recommended taker buy rate (for aggressive inventory reduction)
    pub taker_buy_rate: f64,

    /// Recommended taker sell rate
    pub taker_sell_rate: f64,

    /// Signal timestamp
    pub timestamp: f64,

    /// Signal metadata for debugging
    pub metadata: SignalMetadata,
}

/// A single quote level
#[derive(Debug, Clone)]
pub struct QuoteLevel {
    /// Price offset from mid in bps
    pub offset_bps: f64,

    /// Size at this level
    pub size: f64,

    /// Urgency (0-1, higher = more important to have this quote)
    pub urgency: f64,

    /// Whether this is a bid (true) or ask (false)
    pub is_bid: bool,
}

/// Metadata about the signal generation
#[derive(Debug, Clone)]
pub struct SignalMetadata {
    /// Mid price used
    pub mid_price: f64,

    /// Volatility estimate used (bps)
    pub volatility_bps: f64,

    /// Adverse selection estimate used
    pub adverse_selection: f64,

    /// Current inventory
    pub inventory: f64,

    /// Optimizer execution time (microseconds)
    pub optimizer_time_us: u64,

    /// Whether signal was generated or cached
    pub was_cached: bool,
}

// ============================================================================
// HJB Signal Generator
// ============================================================================

/// Pure signal generation using HJB optimization
pub struct HjbSignalGenerator {
    /// Quote optimizer
    quote_optimizer: HjbMultiLevelOptimizer,

    /// Volatility model
    volatility_model: Box<dyn VolatilityModel>,

    /// Adverse selection model
    microprice_as_model: MicropriceAsModel,

    /// Hawkes fill rate model
    hawkes_model: Arc<RwLock<HawkesFillModel>>,

    /// Base fill rate (lambda_base)
    lambda_base: f64,

    /// Inventory aversion (phi)
    phi: f64,

    /// Maker fee (bps, negative = rebate)
    maker_fee_bps: f64,

    /// Taker fee (bps)
    taker_fee_bps: f64,

    /// Cached signal (to avoid redundant calculations)
    cached_signal: Option<CachedSignal>,
}

/// Cached signal with invalidation key
#[derive(Debug, Clone)]
struct CachedSignal {
    signal: QuoteSignal,
    cache_key: CacheKey,
}

/// Key for cache invalidation
#[derive(Debug, Clone, PartialEq)]
struct CacheKey {
    mid_price_rounded: i64, // Rounded to avoid floating point issues
    inventory_rounded: i64,
    volatility_rounded: i64,
}

impl CacheKey {
    fn from_state(mid_price: f64, inventory: f64, volatility_bps: f64) -> Self {
        Self {
            mid_price_rounded: (mid_price * 1000.0) as i64,
            inventory_rounded: (inventory * 1000.0) as i64,
            volatility_rounded: (volatility_bps * 100.0) as i64,
        }
    }
}

impl HjbSignalGenerator {
    /// Create a new signal generator
    pub fn new(
        quote_optimizer: HjbMultiLevelOptimizer,
        volatility_model: Box<dyn VolatilityModel>,
        microprice_as_model: MicropriceAsModel,
        hawkes_model: Arc<RwLock<HawkesFillModel>>,
        lambda_base: f64,
        phi: f64,
        maker_fee_bps: f64,
        taker_fee_bps: f64,
    ) -> Self {
        Self {
            quote_optimizer,
            volatility_model,
            microprice_as_model,
            hawkes_model,
            lambda_base,
            phi,
            maker_fee_bps,
            taker_fee_bps,
            cached_signal: None,
        }
    }

    /// Generate pure quote signals from market state
    pub fn generate_quotes(
        &mut self,
        market_data: &MarketData,
        inventory: f64,
    ) -> QuoteSignal {
        let start_time = std::time::Instant::now();

        // Check cache
        let cache_key = CacheKey::from_state(
            market_data.mid_price,
            inventory,
            market_data.volatility_bps,
        );

        if let Some(cached) = &self.cached_signal {
            if cached.cache_key == cache_key {
                debug!("[SIGNAL GEN] Using cached signal");
                let mut signal = cached.signal.clone();
                signal.timestamp = market_data.timestamp;
                signal.metadata.was_cached = true;
                return signal;
            }
        }

        // Build optimizer inputs
        let optimizer_inputs = OptimizerInputs {
            mid_price: market_data.mid_price,
            inventory,
            volatility_bps: market_data.volatility_bps,
            imbalance: market_data.imbalance,
            adverse_selection: market_data.adverse_selection,
            spread_bps: market_data.spread_bps,
            timestamp: market_data.timestamp,
        };

        // Run optimizer
        let output = self.quote_optimizer.optimize(optimizer_inputs);

        // Convert optimizer output to quote signal
        let signal = self.convert_output_to_signal(output, market_data, inventory);

        // Update cache
        self.cached_signal = Some(CachedSignal {
            signal: signal.clone(),
            cache_key,
        });

        let elapsed = start_time.elapsed();
        debug!("[SIGNAL GEN] Generated new signal in {:?}", elapsed);

        signal
    }

    /// Convert optimizer output to quote signal
    fn convert_output_to_signal(
        &self,
        output: OptimizerOutput,
        market_data: &MarketData,
        inventory: f64,
    ) -> QuoteSignal {
        let mut bid_levels = Vec::new();
        let mut ask_levels = Vec::new();

        // Convert bid levels
        for level in output.bid_levels {
            bid_levels.push(QuoteLevel {
                offset_bps: level.offset_bps,
                size: level.size,
                urgency: level.urgency,
                is_bid: true,
            });
        }

        // Convert ask levels
        for level in output.ask_levels {
            ask_levels.push(QuoteLevel {
                offset_bps: level.offset_bps,
                size: level.size,
                urgency: level.urgency,
                is_bid: false,
            });
        }

        // Calculate overall urgency
        let urgency = self.calculate_urgency(inventory, market_data.volatility_bps);

        QuoteSignal {
            bid_levels,
            ask_levels,
            urgency,
            taker_buy_rate: output.taker_buy_rate,
            taker_sell_rate: output.taker_sell_rate,
            timestamp: market_data.timestamp,
            metadata: SignalMetadata {
                mid_price: market_data.mid_price,
                volatility_bps: market_data.volatility_bps,
                adverse_selection: market_data.adverse_selection,
                inventory,
                optimizer_time_us: output.computation_time_us,
                was_cached: false,
            },
        }
    }

    /// Calculate overall urgency to trade based on inventory and volatility
    fn calculate_urgency(&self, inventory: f64, volatility_bps: f64) -> f64 {
        // Higher inventory = higher urgency
        let inventory_urgency = (inventory.abs() / 50.0).min(1.0);

        // Higher volatility = higher urgency
        let vol_urgency = (volatility_bps / 100.0).min(1.0);

        // Combined urgency
        (inventory_urgency * 0.7 + vol_urgency * 0.3).min(1.0)
    }

    /// Update volatility model with market data
    pub fn update_volatility_model(&mut self, _market_data: &MarketData) {
        // This would be called on each market update
        // The volatility model maintains its own state
    }

    /// Update adverse selection model
    pub fn update_adverse_selection_model(&mut self, _market_data: &MarketData) {
        // Update the microprice AS model with new market data
    }

    /// Update Hawkes model with fill
    pub fn update_hawkes_model(&mut self, is_buy: bool, level: usize, timestamp: f64) {
        let mut hawkes = self.hawkes_model.write();
        hawkes.record_fill(is_buy, level, timestamp);
    }

    /// Get current volatility estimate
    pub fn get_volatility_bps(&self) -> f64 {
        self.volatility_model.get_volatility_bps()
    }

    /// Get volatility uncertainty
    pub fn get_volatility_uncertainty_bps(&self) -> f64 {
        self.volatility_model.get_uncertainty_bps()
    }

    /// Clear cache (force recomputation on next signal generation)
    pub fn clear_cache(&mut self) {
        self.cached_signal = None;
    }

    /// Get cache hit rate (for monitoring)
    pub fn get_cache_stats(&self) -> (u64, u64) {
        // Returns (hits, misses)
        // TODO: Add proper tracking
        (0, 0)
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

impl QuoteSignal {
    /// Check if this signal is empty (no quotes)
    pub fn is_empty(&self) -> bool {
        self.bid_levels.is_empty() && self.ask_levels.is_empty()
    }

    /// Get the best bid offset
    pub fn best_bid_offset_bps(&self) -> Option<f64> {
        self.bid_levels.first().map(|l| l.offset_bps)
    }

    /// Get the best ask offset
    pub fn best_ask_offset_bps(&self) -> Option<f64> {
        self.ask_levels.first().map(|l| l.offset_bps)
    }

    /// Get total bid size
    pub fn total_bid_size(&self) -> f64 {
        self.bid_levels.iter().map(|l| l.size).sum()
    }

    /// Get total ask size
    pub fn total_ask_size(&self) -> f64 {
        self.ask_levels.iter().map(|l| l.size).sum()
    }

    /// Calculate theoretical spread (best bid to best ask)
    pub fn theoretical_spread_bps(&self) -> f64 {
        if let (Some(bid), Some(ask)) = (self.best_bid_offset_bps(), self.best_ask_offset_bps()) {
            ask - bid
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quote_signal_helpers() {
        let signal = QuoteSignal {
            bid_levels: vec![
                QuoteLevel {
                    offset_bps: -5.0,
                    size: 10.0,
                    urgency: 0.8,
                    is_bid: true,
                },
                QuoteLevel {
                    offset_bps: -10.0,
                    size: 5.0,
                    urgency: 0.5,
                    is_bid: true,
                },
            ],
            ask_levels: vec![
                QuoteLevel {
                    offset_bps: 5.0,
                    size: 10.0,
                    urgency: 0.8,
                    is_bid: false,
                },
            ],
            urgency: 0.6,
            taker_buy_rate: 0.0,
            taker_sell_rate: 0.0,
            timestamp: 0.0,
            metadata: SignalMetadata {
                mid_price: 100.0,
                volatility_bps: 20.0,
                adverse_selection: 0.5,
                inventory: 0.0,
                optimizer_time_us: 100,
                was_cached: false,
            },
        };

        assert_eq!(signal.best_bid_offset_bps(), Some(-5.0));
        assert_eq!(signal.best_ask_offset_bps(), Some(5.0));
        assert_eq!(signal.total_bid_size(), 15.0);
        assert_eq!(signal.total_ask_size(), 10.0);
        assert_eq!(signal.theoretical_spread_bps(), 10.0);
    }

    #[test]
    fn test_cache_key() {
        let key1 = CacheKey::from_state(100.0, 10.0, 20.0);
        let key2 = CacheKey::from_state(100.0, 10.0, 20.0);
        let key3 = CacheKey::from_state(100.1, 10.0, 20.0);

        assert_eq!(key1, key2);
        assert_ne!(key1, key3);
    }
}
