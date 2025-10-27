use crate::{BookLevel, L2BookData};
use log::info;

/// Represents a sorted and analyzed order book
#[derive(Debug, Clone)]
pub struct OrderBook {
    pub bids: Vec<BookLevel>,
    pub asks: Vec<BookLevel>,
    pub mid_price: f64,
}

/// Analysis results from the order book
#[derive(Debug, Clone)]
pub struct BookAnalysis {
    pub bid_depth: f64,
    pub ask_depth: f64,
    pub weighted_bid_price: f64,
    pub weighted_ask_price: f64,
    pub imbalance: f64, // Positive means more bid liquidity, negative means more ask liquidity
}

impl OrderBook {
    /// Create a new OrderBook from L2BookData
    pub fn from_l2_data(data: &L2BookData) -> Option<Self> {
        if data.levels.len() < 2 {
            return None;
        }

        let bids = data.levels[0].clone();
        let asks = data.levels[1].clone();

        if bids.is_empty() || asks.is_empty() {
            return None;
        }

        // Calculate mid price
        let best_bid: f64 = bids[0].px.parse().ok()?;
        let best_ask: f64 = asks[0].px.parse().ok()?;
        let mid_price = (best_bid + best_ask) / 2.0;

        Some(OrderBook {
            bids,
            asks,
            mid_price,
        })
    }

    /// Analyze the order book depth and imbalance
    pub fn analyze(&self, depth_levels: usize) -> Option<BookAnalysis> {
        let levels_to_analyze = depth_levels.min(self.bids.len()).min(self.asks.len());

        if levels_to_analyze == 0 {
            return None;
        }

        let mut bid_depth = 0.0;
        let mut ask_depth = 0.0;
        let mut weighted_bid_sum = 0.0;
        let mut weighted_ask_sum = 0.0;

        // Calculate depth-weighted prices for bids
        for level in self.bids.iter().take(levels_to_analyze) {
            let price: f64 = level.px.parse().ok()?;
            let size: f64 = level.sz.parse().ok()?;
            
            bid_depth += size;
            weighted_bid_sum += price * size;
        }

        // Calculate depth-weighted prices for asks
        for level in self.asks.iter().take(levels_to_analyze) {
            let price: f64 = level.px.parse().ok()?;
            let size: f64 = level.sz.parse().ok()?;
            
            ask_depth += size;
            weighted_ask_sum += price * size;
        }

        let weighted_bid_price = if bid_depth > 0.0 {
            weighted_bid_sum / bid_depth
        } else {
            0.0
        };

        let weighted_ask_price = if ask_depth > 0.0 {
            weighted_ask_sum / ask_depth
        } else {
            0.0
        };

        // Calculate imbalance: (bid_depth - ask_depth) / (bid_depth + ask_depth)
        // Range: -1 to 1
        // Positive = more bid liquidity (buying pressure)
        // Negative = more ask liquidity (selling pressure)
        let total_depth = bid_depth + ask_depth;
        let imbalance = if total_depth > 0.0 {
            (bid_depth - ask_depth) / total_depth
        } else {
            0.0
        };

        Some(BookAnalysis {
            bid_depth,
            ask_depth,
            weighted_bid_price,
            weighted_ask_price,
            imbalance,
        })
    }

    /// Get the best bid price
    pub fn best_bid(&self) -> Option<f64> {
        self.bids.first()?.px.parse().ok()
    }

    /// Get the best ask price
    pub fn best_ask(&self) -> Option<f64> {
        self.asks.first()?.px.parse().ok()
    }

    /// Get the spread in basis points
    pub fn spread_bps(&self) -> Option<f64> {
        let best_bid = self.best_bid()?;
        let best_ask = self.best_ask()?;
        let spread = best_ask - best_bid;
        Some((spread / self.mid_price) * 10000.0)
    }

    /// Log book statistics
    pub fn log_stats(&self, analysis: &BookAnalysis) {
        info!(
            "Book Stats - Mid: {:.2}, Bid Depth: {:.2}, Ask Depth: {:.2}, Imbalance: {:.3}, Spread: {:.1} bps",
            self.mid_price,
            analysis.bid_depth,
            analysis.ask_depth,
            analysis.imbalance,
            self.spread_bps().unwrap_or(0.0)
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_book_level(px: &str, sz: &str) -> BookLevel {
        BookLevel {
            px: px.to_string(),
            sz: sz.to_string(),
            n: 1,
        }
    }

    #[test]
    fn test_order_book_creation() {
        let data = L2BookData {
            coin: "HYPE".to_string(),
            time: 0,
            levels: vec![
                vec![
                    create_test_book_level("100.0", "10.0"),
                    create_test_book_level("99.0", "20.0"),
                ],
                vec![
                    create_test_book_level("101.0", "15.0"),
                    create_test_book_level("102.0", "25.0"),
                ],
            ],
        };

        let book = OrderBook::from_l2_data(&data).unwrap();
        assert_eq!(book.mid_price, 100.5);
        assert_eq!(book.bids.len(), 2);
        assert_eq!(book.asks.len(), 2);
    }

    #[test]
    fn test_book_analysis() {
        let data = L2BookData {
            coin: "HYPE".to_string(),
            time: 0,
            levels: vec![
                vec![
                    create_test_book_level("100.0", "10.0"),
                    create_test_book_level("99.0", "20.0"),
                ],
                vec![
                    create_test_book_level("101.0", "15.0"),
                    create_test_book_level("102.0", "5.0"),
                ],
            ],
        };

        let book = OrderBook::from_l2_data(&data).unwrap();
        let analysis = book.analyze(2).unwrap();

        assert_eq!(analysis.bid_depth, 30.0);
        assert_eq!(analysis.ask_depth, 20.0);
        assert!(analysis.imbalance > 0.0); // More bid liquidity
    }

    #[test]
    fn test_spread_calculation() {
        let data = L2BookData {
            coin: "HYPE".to_string(),
            time: 0,
            levels: vec![
                vec![create_test_book_level("100.0", "10.0")],
                vec![create_test_book_level("101.0", "10.0")],
            ],
        };

        let book = OrderBook::from_l2_data(&data).unwrap();
        let spread_bps = book.spread_bps().unwrap();
        assert!((spread_bps - 99.5).abs() < 0.1); // ~1% spread = ~100 bps
    }
}
