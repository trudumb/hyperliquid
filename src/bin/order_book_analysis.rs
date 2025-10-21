use hyperliquid_rust_sdk::{BaseUrl, InfoClient, Message, Subscription, OrderBookAnalyzer};
use log::info;
use tokio::{
    spawn,
    sync::mpsc::unbounded_channel,
    time::{sleep, Duration},
};

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
            
            // Market depth analysis
            let (top_bids, top_asks) = order_book_analyzer.get_market_depth(3);
            info!("Top 3 Bids: {:?}", top_bids);
            info!("Top 3 Asks: {:?}", top_asks);
            
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
            
            // Large order detection
            let has_large_orders = order_book_analyzer.has_large_orders(0.1);
            if has_large_orders {
                info!("⚠️  Large orders detected (>10% of volume)");
            }
            
            // Optimized quote placement
            if let Some(mid_price) = order_book_analyzer.get_mid_price() {
                let (optimal_bid, optimal_ask) = order_book_analyzer.optimize_quote_placement(50, mid_price);
                info!("Optimized Quotes - Bid: ${:.4}, Ask: ${:.4}", optimal_bid, optimal_ask);
            }
            
            info!("=====================================");
        } else {
            // Log basic info for other updates
            if let Some(mid_price) = order_book_analyzer.get_mid_price() {
                let imbalance = order_book_analyzer.order_book_imbalance();
                info!("Update #{}: Mid=${:.4}, Imbalance={:.3}", 
                      update_count, mid_price, imbalance);
            }
        }
    }

    info!("Order book analysis completed!");
}