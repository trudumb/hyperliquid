// ============================================================================
// Market Data Pipeline - Reactive Market Data Processing
// ============================================================================
//
// Processes market updates through a chain of specialized processors.
// Each processor updates one aspect of market data (volatility, imbalance, etc.)

use log::debug;

use crate::strategy::MarketUpdate;
use crate::L2BookData;

use super::trading_state_store::MarketData;
use super::VolatilityModel;

// ============================================================================
// Market Data Processor Trait
// ============================================================================

/// Trait for market data processors
pub trait MarketDataProcessor: Send + Sync {
    /// Process market data and update relevant fields
    fn process(&mut self, data: &mut ProcessedMarketData);

    /// Get processor name
    fn name(&self) -> &str;
}

// ============================================================================
// Processed Market Data
// ============================================================================

/// Market data at various stages of processing
#[derive(Debug, Clone)]
pub struct ProcessedMarketData {
    /// Raw market update
    pub raw_update: MarketUpdate,

    /// Parsed L2 book data
    pub book_data: Option<L2BookData>,

    /// Mid price
    pub mid_price: f64,

    /// Best bid
    pub best_bid: f64,

    /// Best ask
    pub best_ask: f64,

    /// Spread in bps
    pub spread_bps: f64,

    /// Volatility estimate (bps)
    pub volatility_bps: f64,

    /// Volatility uncertainty (bps)
    pub vol_uncertainty_bps: f64,

    /// Order book imbalance (-1 to +1)
    pub imbalance: f64,

    /// Adverse selection estimate
    pub adverse_selection: f64,

    /// Timestamp
    pub timestamp: f64,
}

impl ProcessedMarketData {
    /// Create from raw market update
    pub fn from_update(update: MarketUpdate) -> Self {
        let book_data = if let MarketUpdate::L2Book(ref book) = update {
            Some(book.clone())
        } else {
            None
        };

        let (mid_price, best_bid, best_ask, spread_bps) = if let Some(ref book) = book_data {
            // book.levels is Vec<Vec<BookLevel>> where [0] is bids, [1] is asks
            let bid = book.levels.get(0)
                .and_then(|levels| levels.first())
                .and_then(|l| l.px.parse::<f64>().ok())
                .unwrap_or(0.0);
            let ask = book.levels.get(1)
                .and_then(|levels| levels.first())
                .and_then(|l| l.px.parse::<f64>().ok())
                .unwrap_or(0.0);
            let mid = (bid + ask) / 2.0;
            let spread = if mid > 0.0 {
                ((ask - bid) / mid) * 10000.0
            } else {
                0.0
            };
            (mid, bid, ask, spread)
        } else {
            (0.0, 0.0, 0.0, 0.0)
        };

        Self {
            raw_update: update,
            book_data,
            mid_price,
            best_bid,
            best_ask,
            spread_bps,
            volatility_bps: 10.0, // Default
            vol_uncertainty_bps: 5.0, // Default
            imbalance: 0.0,
            adverse_selection: 0.0,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs_f64(),
        }
    }

    /// Convert to MarketData for state store
    pub fn to_market_data(&self) -> MarketData {
        MarketData {
            mid_price: self.mid_price,
            best_bid: self.best_bid,
            best_ask: self.best_ask,
            bid_size: self.book_data.as_ref()
                .and_then(|b| b.levels.get(0))
                .and_then(|l| l.first())
                .and_then(|l| l.sz.parse::<f64>().ok())
                .unwrap_or(0.0),
            ask_size: self.book_data.as_ref()
                .and_then(|b| b.levels.get(1))
                .and_then(|l| l.first())
                .and_then(|l| l.sz.parse::<f64>().ok())
                .unwrap_or(0.0),
            spread_bps: self.spread_bps,
            imbalance: self.imbalance,
            volatility_bps: self.volatility_bps,
            vol_uncertainty_bps: self.vol_uncertainty_bps,
            adverse_selection: self.adverse_selection,
            timestamp: self.timestamp,
        }
    }
}

// ============================================================================
// Market Data Pipeline
// ============================================================================

/// Pipeline for processing market data through multiple stages
pub struct MarketDataPipeline {
    /// Ordered list of processors
    processors: Vec<Box<dyn MarketDataProcessor>>,
}

impl MarketDataPipeline {
    /// Create a new pipeline
    pub fn new() -> Self {
        Self {
            processors: Vec::new(),
        }
    }

    /// Add a processor to the pipeline
    pub fn add_processor(&mut self, processor: Box<dyn MarketDataProcessor>) {
        debug!("[MARKET DATA PIPELINE] Adding processor: {}", processor.name());
        self.processors.push(processor);
    }

    /// Process a market update through the pipeline
    pub fn process(&mut self, update: MarketUpdate) -> ProcessedMarketData {
        let mut data = ProcessedMarketData::from_update(update);

        for processor in &mut self.processors {
            processor.process(&mut data);
        }

        data
    }

    /// Get number of processors
    pub fn processor_count(&self) -> usize {
        self.processors.len()
    }
}

impl Default for MarketDataPipeline {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Example Processors
// ============================================================================

/// Processor for calculating order book imbalance
pub struct ImbalanceProcessor;

impl ImbalanceProcessor {
    pub fn new() -> Self {
        Self
    }
}

impl MarketDataProcessor for ImbalanceProcessor {
    fn process(&mut self, data: &mut ProcessedMarketData) {
        if let Some(ref book) = data.book_data {
            // Calculate imbalance from top 3 levels
            let mut bid_vol = 0.0;
            let mut ask_vol = 0.0;

            // book.levels has structure: [[bid_levels], [ask_levels]]
            if let Some(bid_levels) = book.levels.get(0) {
                for level in bid_levels.iter().take(3) {
                    if let Ok(size) = level.sz.parse::<f64>() {
                        bid_vol += size;
                    }
                }
            }

            if let Some(ask_levels) = book.levels.get(1) {
                for level in ask_levels.iter().take(3) {
                    if let Ok(size) = level.sz.parse::<f64>() {
                        ask_vol += size;
                    }
                }
            }

            if bid_vol + ask_vol > 0.0 {
                data.imbalance = (bid_vol - ask_vol) / (bid_vol + ask_vol);
            }
        }
    }

    fn name(&self) -> &str {
        "ImbalanceProcessor"
    }
}

/// Processor for volatility estimation
pub struct VolatilityProcessor {
    model: Box<dyn VolatilityModel>,
}

impl VolatilityProcessor {
    pub fn new(model: Box<dyn VolatilityModel>) -> Self {
        Self { model }
    }
}

impl MarketDataProcessor for VolatilityProcessor {
    fn process(&mut self, data: &mut ProcessedMarketData) {
        // Update model with new market data
        self.model.on_market_update(&data.raw_update);

        // Get estimates
        data.volatility_bps = self.model.get_volatility_bps();
        data.vol_uncertainty_bps = self.model.get_uncertainty_bps();
    }

    fn name(&self) -> &str {
        "VolatilityProcessor"
    }
}

/// Processor for adverse selection
pub struct AdverseSelectionProcessor {
    // Simplified - in reality would contain microprice AS model
}

impl AdverseSelectionProcessor {
    pub fn new() -> Self {
        Self {}
    }
}

impl MarketDataProcessor for AdverseSelectionProcessor {
    fn process(&mut self, data: &mut ProcessedMarketData) {
        // Simplified adverse selection estimate
        // In reality, use MicropriceAsModel
        data.adverse_selection = data.imbalance * 0.5;
    }

    fn name(&self) -> &str {
        "AdverseSelectionProcessor"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipeline_basic() {
        let mut pipeline = MarketDataPipeline::new();
        pipeline.add_processor(Box::new(ImbalanceProcessor::new()));
        pipeline.add_processor(Box::new(AdverseSelectionProcessor::new()));

        assert_eq!(pipeline.processor_count(), 2);
    }
}
