use hyperliquid_rust_sdk::{BaseUrl, InfoClient, Message, Subscription};
use log::info;
use tokio::{
    spawn,
    sync::mpsc::unbounded_channel,
    time::{sleep, Duration},
};

//RUST_LOG=info cargo run --example order_book_analysis

#[derive(Debug, Clone)]
pub struct OrderBookAnalyzer {
    bid_depth: Vec<(f64, f64)>, // (price, size)
    ask_depth: Vec<(f64, f64)>,
}

impl OrderBookAnalyzer {
    pub fn new() -> Self {
        Self {
            bid_depth: Vec::new(),
            ask_depth: Vec::new(),
        }
    }

    /// Update the order book with new L2 data
    pub fn update_order_book(&mut self, l2_data: &hyperliquid_rust_sdk::L2Book) {
        self.bid_depth.clear();
        self.ask_depth.clear();

        // L2Book levels: [0] = bids (sorted high to low), [1] = asks (sorted low to high)
        if l2_data.data.levels.len() >= 2 {
            // Process bids (index 0) - these should be sorted from highest to lowest price
            for level in &l2_data.data.levels[0] {
                if let (Ok(price), Ok(size)) = (level.px.parse::<f64>(), level.sz.parse::<f64>()) {
                    if size > 0.0 {
                        self.bid_depth.push((price, size));
                    }
                }
            }

            // Process asks (index 1) - these should be sorted from lowest to highest price
            for level in &l2_data.data.levels[1] {
                if let (Ok(price), Ok(size)) = (level.px.parse::<f64>(), level.sz.parse::<f64>()) {
                    if size > 0.0 {
                        self.ask_depth.push((price, size));
                    }
                }
            }
        }

        // Ensure proper sorting
        self.bid_depth.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap()); // Highest to lowest
        self.ask_depth.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap()); // Lowest to highest
    }

    /// Estimate fill probability based on queue position
    pub fn estimate_fill_probability(&self, our_price: f64, is_buy: bool) -> f64 {
        let levels = if is_buy { &self.bid_depth } else { &self.ask_depth };
        
        let total_volume_ahead: f64 = levels
            .iter()
            .take_while(|(price, _)| {
                if is_buy { *price > our_price } else { *price < our_price }
            })
            .map(|(_, size)| size)
            .sum();
        
        // Exponential decay model - more volume ahead = lower fill probability
        (-0.001 * total_volume_ahead).exp()
    }
    
    /// Calculate order book imbalance (positive = more bids, negative = more asks)
    pub fn order_book_imbalance(&self) -> f64 {
        let bid_vol: f64 = self.bid_depth.iter().take(5).map(|(_, s)| s).sum();
        let ask_vol: f64 = self.ask_depth.iter().take(5).map(|(_, s)| s).sum();
        
        if bid_vol + ask_vol == 0.0 {
            return 0.0;
        }
        
        (bid_vol - ask_vol) / (bid_vol + ask_vol)
    }

    /// Get the current bid-ask spread
    pub fn get_spread(&self) -> Option<f64> {
        let best_bid = self.bid_depth.first()?.0;
        let best_ask = self.ask_depth.first()?.0;
        Some(best_ask - best_bid)
    }

    /// Get the current mid price
    pub fn get_mid_price(&self) -> Option<f64> {
        let best_bid = self.bid_depth.first()?.0;
        let best_ask = self.ask_depth.first()?.0;
        Some((best_bid + best_ask) / 2.0)
    }

    /// Detect large orders (potential toxicity) based on size relative to market depth
    pub fn detect_large_orders(&self, threshold_percentile: f64) -> (Vec<(f64, f64)>, Vec<(f64, f64)>) {
        let total_bid_volume: f64 = self.bid_depth.iter().map(|(_, size)| size).sum();
        let total_ask_volume: f64 = self.ask_depth.iter().map(|(_, size)| size).sum();

        let bid_threshold = total_bid_volume * threshold_percentile;
        let ask_threshold = total_ask_volume * threshold_percentile;

        let large_bids: Vec<(f64, f64)> = self
            .bid_depth
            .iter()
            .filter(|(_, size)| *size >= bid_threshold)
            .cloned()
            .collect();

        let large_asks: Vec<(f64, f64)> = self
            .ask_depth
            .iter()
            .filter(|(_, size)| *size >= ask_threshold)
            .cloned()
            .collect();

        (large_bids, large_asks)
    }

    /// Get market depth at different price levels
    pub fn get_market_depth(&self, levels: usize) -> (Vec<(f64, f64)>, Vec<(f64, f64)>) {
        let bids = self.bid_depth.iter().take(levels).cloned().collect();
        let asks = self.ask_depth.iter().take(levels).cloned().collect();
        (bids, asks)
    }

    /// Calculate weighted average price for a given size
    pub fn get_weighted_avg_price(&self, size: f64, is_buy: bool) -> Option<f64> {
        let levels = if is_buy { &self.ask_depth } else { &self.bid_depth };
        
        let mut remaining_size = size;
        let mut total_cost = 0.0;
        
        for (price, available_size) in levels {
            if remaining_size <= 0.0 {
                break;
            }
            
            let size_to_take = remaining_size.min(*available_size);
            total_cost += size_to_take * price;
            remaining_size -= size_to_take;
        }
        
        if remaining_size > 0.0 {
            None // Not enough liquidity
        } else {
            Some(total_cost / size)
        }
    }

    /// Optimize quote placement based on order book analysis
    pub fn optimize_quote_placement(&self, target_spread_bps: u16, base_mid_price: f64) -> Option<(f64, f64)> {
        let mid_price = self.get_mid_price().unwrap_or(base_mid_price);
        let imbalance = self.order_book_imbalance();
        
        // Adjust spread based on imbalance and market conditions
        let spread_adjustment = imbalance * 0.5; // Tighten spread on the heavy side
        let base_half_spread = (mid_price * target_spread_bps as f64) / 20000.0; // Convert BPS to half spread
        
        let bid_adjustment = if imbalance > 0.1 { -spread_adjustment } else { 0.0 };
        let ask_adjustment = if imbalance < -0.1 { spread_adjustment } else { 0.0 };
        
        let optimal_bid = mid_price - base_half_spread + bid_adjustment;
        let optimal_ask = mid_price + base_half_spread + ask_adjustment;
        
        Some((optimal_bid, optimal_ask))
    }
}

#[tokio::main]
async fn main() {
    env_logger::init();

    let mut info_client = InfoClient::new(None, Some(BaseUrl::Testnet)).await.unwrap();
    let mut order_book_analyzer = OrderBookAnalyzer::new();

    let (sender, mut receiver) = unbounded_channel();
    let subscription_id = info_client
        .subscribe(
            Subscription::L2Book {
                coin: "ETH".to_string(),
            },
            sender,
        )
        .await
        .unwrap();

    spawn(async move {
        sleep(Duration::from_secs(30)).await;
        info!("Unsubscribing from l2 book data");
        info_client.unsubscribe(subscription_id).await.unwrap()
    });

    let mut update_count = 0;

    // This loop ends when we unsubscribe
    while let Some(Message::L2Book(l2_book)) = receiver.recv().await {
        // Update the order book analyzer
        order_book_analyzer.update_order_book(&l2_book);
        
        update_count += 1;
        
        // Log detailed analysis every 10 updates to avoid spam
        if update_count % 10 == 0 {
            info!("=== Order Book Analysis (Update #{}) ===", update_count);
            
            // Basic market data
            if let Some(mid_price) = order_book_analyzer.get_mid_price() {
                info!("Mid Price: ${:.4}", mid_price);
            }
            
            if let Some(spread) = order_book_analyzer.get_spread() {
                info!("Bid-Ask Spread: ${:.4}", spread);
            }
            
            // Market depth analysis
            let (top_bids, top_asks) = order_book_analyzer.get_market_depth(5);
            info!("Top 5 Bids: {:?}", top_bids);
            info!("Top 5 Asks: {:?}", top_asks);
            
            // Imbalance analysis
            let imbalance = order_book_analyzer.order_book_imbalance();
            info!("Order Book Imbalance: {:.4} (positive = more bids)", imbalance);
            
            // Fill probability estimation
            if let Some(mid_price) = order_book_analyzer.get_mid_price() {
                let bid_price = mid_price * 0.999; // 0.1% below mid
                let ask_price = mid_price * 1.001; // 0.1% above mid
                
                let bid_fill_prob = order_book_analyzer.estimate_fill_probability(bid_price, true);
                let ask_fill_prob = order_book_analyzer.estimate_fill_probability(ask_price, false);
                
                info!("Fill Probability - Bid @${:.4}: {:.2}%", bid_price, bid_fill_prob * 100.0);
                info!("Fill Probability - Ask @${:.4}: {:.2}%", ask_price, ask_fill_prob * 100.0);
            }
            
            // Large order detection (orders > 10% of total volume)
            let (large_bids, large_asks) = order_book_analyzer.detect_large_orders(0.1);
            if !large_bids.is_empty() || !large_asks.is_empty() {
                info!("Large Orders Detected - Bids: {:?}, Asks: {:?}", large_bids, large_asks);
            }
            
            // Weighted average price for different trade sizes
            if let Some(mid_price) = order_book_analyzer.get_mid_price() {
                let trade_size = mid_price * 0.1; // Example: 0.1 ETH worth
                
                if let Some(buy_wap) = order_book_analyzer.get_weighted_avg_price(trade_size, true) {
                    info!("Weighted Avg Price for buying ${:.2}: ${:.4}", trade_size, buy_wap);
                }
                
                if let Some(sell_wap) = order_book_analyzer.get_weighted_avg_price(trade_size, false) {
                    info!("Weighted Avg Price for selling ${:.2}: ${:.4}", trade_size, sell_wap);
                }
            }
            
            // Optimized quote placement
            if let Some(mid_price) = order_book_analyzer.get_mid_price() {
                if let Some((optimal_bid, optimal_ask)) = order_book_analyzer.optimize_quote_placement(50, mid_price) {
                    info!("Optimized Quotes - Bid: ${:.4}, Ask: ${:.4}", optimal_bid, optimal_ask);
                }
            }
            
            info!("=====================================");
        } else {
            // Log basic info for other updates
            if let (Some(mid_price), Some(spread)) = (order_book_analyzer.get_mid_price(), order_book_analyzer.get_spread()) {
                info!("Update #{}: Mid=${:.4}, Spread=${:.4}, Imbalance={:.3}", 
                      update_count, mid_price, spread, order_book_analyzer.order_book_imbalance());
            }
        }
    }
}