use alloy::{primitives::Address, signers::local::PrivateKeySigner};
use log::{error, info};
use tokio::sync::mpsc::unbounded_channel;

//RUST_LOG=info cargo run --bin market_maker

use crate::{
    bps_diff, AssetType, BaseUrl, ClientCancelRequest, ClientLimit, ClientOrder,
    ClientOrderRequest, ExchangeClient, ExchangeDataStatus, ExchangeResponseStatus, InfoClient,
    Message, Subscription, TickLotValidator, UserData, EPSILON,
};
#[derive(Debug)]
pub struct MarketMakerRestingOrder {
    pub oid: u64,
    pub position: f64,
    pub price: f64,
}

#[derive(Debug)]
pub struct MarketMakerInput {
    pub asset: String,
    pub target_liquidity: f64, // Amount of liquidity on both sides to target
    pub half_spread: u16,      // Half of the spread for our market making (in BPS)
    pub max_bps_diff: u16, // Max deviation before we cancel and put new orders on the book (in BPS)
    pub max_absolute_position_size: f64, // Absolute value of the max position we can take on
    pub asset_type: AssetType, // Asset type (Perp or Spot) for tick/lot size validation
    pub wallet: PrivateKeySigner, // Wallet containing private key
}

#[derive(Debug)]
pub struct MarketMaker {
    pub asset: String,
    pub target_liquidity: f64,
    pub half_spread: u16,
    pub max_bps_diff: u16,
    pub max_absolute_position_size: f64,
    pub tick_lot_validator: TickLotValidator,
    pub lower_resting: MarketMakerRestingOrder,
    pub upper_resting: MarketMakerRestingOrder,
    pub cur_position: f64,
    pub latest_mid_price: f64,
    pub info_client: InfoClient,
    pub exchange_client: ExchangeClient,
    pub user_address: Address,
}

impl MarketMaker {
    /// Reset a resting order to default state
    fn reset_resting_order(&mut self, is_lower: bool) {
        let resting_order = if is_lower {
            &mut self.lower_resting
        } else {
            &mut self.upper_resting
        };
        
        *resting_order = MarketMakerRestingOrder {
            oid: 0,
            position: 0.0,
            price: -1.0,
        };
    }

    /// Clean up any invalid resting orders (e.g., negative positions)
    fn cleanup_invalid_resting_orders(&mut self) {
        if self.lower_resting.position < 0.0 {
            info!("Lower resting order has negative position, resetting");
            self.reset_resting_order(true);
        }
        if self.upper_resting.position < 0.0 {
            info!("Upper resting order has negative position, resetting");
            self.reset_resting_order(false);
        }
    }

    pub async fn new(input: MarketMakerInput) -> Result<MarketMaker, crate::Error> {
        let user_address = input.wallet.address();

        let info_client = InfoClient::new(None, Some(BaseUrl::Mainnet)).await?;
        let exchange_client =
            ExchangeClient::new(None, input.wallet, Some(BaseUrl::Mainnet), None, None)
                .await?;

        // Fetch asset metadata to get sz_decimals
        let sz_decimals = match input.asset_type {
            AssetType::Perp => {
                let meta = info_client.meta().await?;
                meta.universe
                    .iter()
                    .find(|asset_meta| asset_meta.name == input.asset)
                    .map(|asset_meta| asset_meta.sz_decimals)
                    .ok_or_else(|| crate::Error::InvalidInput(format!(
                        "Asset {} not found in metadata", input.asset
                    )))?
            }
            AssetType::Spot => {
                let _spot_meta = info_client.spot_meta().await?;
                // For spot assets, we need to look up the token info
                // For now, use a default of 6 decimals - this should be improved
                // to properly lookup the spot asset metadata
                6 // This is a placeholder - should be improved
            }
        };

        let tick_lot_validator = TickLotValidator::new(
            input.asset.clone(),
            input.asset_type,
            sz_decimals,
        );

        Ok(MarketMaker {
            asset: input.asset,
            target_liquidity: input.target_liquidity,
            half_spread: input.half_spread,
            max_bps_diff: input.max_bps_diff,
            max_absolute_position_size: input.max_absolute_position_size,
            tick_lot_validator,
            lower_resting: MarketMakerRestingOrder {
                oid: 0,
                position: 0.0,
                price: -1.0,
            },
            upper_resting: MarketMakerRestingOrder {
                oid: 0,
                position: 0.0,
                price: -1.0,
            },
            cur_position: 0.0,
            latest_mid_price: -1.0,
            info_client,
            exchange_client,
            user_address,
        })
    }

    pub async fn start(&mut self) {
        self.start_with_shutdown_signal(None).await;
    }

    pub async fn start_with_shutdown_signal(&mut self, mut shutdown_rx: Option<tokio::sync::oneshot::Receiver<()>>) {
        let (sender, mut receiver) = unbounded_channel();

        // Subscribe to UserEvents for fills
        self.info_client
            .subscribe(
                Subscription::UserEvents {
                    user: self.user_address,
                },
                sender.clone(),
            )
            .await
            .unwrap();

        // Subscribe to AllMids so we can market make around the mid price
        self.info_client
            .subscribe(Subscription::AllMids, sender)
            .await
            .unwrap();

        loop {
            tokio::select! {
                // Check for shutdown signal
                _ = async {
                    if let Some(ref mut rx) = shutdown_rx {
                        rx.await.ok();
                    } else {
                        std::future::pending::<()>().await
                    }
                } => {
                    info!("Shutdown signal received, cancelling orders and exiting...");
                    self.shutdown().await;
                    break;
                }
                // Handle market maker messages
                message = receiver.recv() => {
            let message = message.unwrap();
            match message {
                Message::AllMids(all_mids) => {
                    let all_mids = all_mids.data.mids;
                    let mid = all_mids.get(&self.asset);
                    if let Some(mid) = mid {
                        let mid: f64 = mid.parse().unwrap();
                        self.latest_mid_price = mid;
                        // Check to see if we need to cancel or place any new orders
                        self.potentially_update().await;
                    } else {
                        error!(
                            "could not get mid for asset {}: {all_mids:?}",
                            self.asset.clone()
                        );
                    }
                }
                Message::User(user_events) => {
                    // We haven't seen the first mid price event yet, so just continue
                    if self.latest_mid_price < 0.0 {
                        continue;
                    }
                    let user_events = user_events.data;
                    if let UserData::Fills(fills) = user_events {
                        for fill in fills {
                            let amount: f64 = fill.sz.parse().unwrap();
                            // Update our resting positions whenever we see a fill
                            if fill.side.eq("B") {
                                self.cur_position += amount;
                                
                                // Check if this fill corresponds to our tracked lower resting order
                                if self.lower_resting.oid == fill.oid {
                                    self.lower_resting.position -= amount;
                                    // If the order is fully filled, reset it
                                    if self.lower_resting.position <= EPSILON {
                                        info!("Lower resting order fully filled, resetting");
                                        self.reset_resting_order(true);
                                    }
                                }
                                
                                info!("Fill: bought {amount} {} (oid: {})", self.asset.clone(), fill.oid);
                            } else {
                                self.cur_position -= amount;
                                
                                // Check if this fill corresponds to our tracked upper resting order
                                if self.upper_resting.oid == fill.oid {
                                    self.upper_resting.position -= amount;
                                    // If the order is fully filled, reset it
                                    if self.upper_resting.position <= EPSILON {
                                        info!("Upper resting order fully filled, resetting");
                                        self.reset_resting_order(false);
                                    }
                                }
                                
                                info!("Fill: sold {amount} {} (oid: {})", self.asset.clone(), fill.oid);
                            }
                        }
                    }
                    // Check to see if we need to cancel or place any new orders
                    self.potentially_update().await;
                }
                _ => {
                    panic!("Unsupported message type");
                }
            }
                }
            }
        }
    }

    async fn attempt_cancel(&self, asset: String, oid: u64) -> bool {
        let cancel = self
            .exchange_client
            .cancel(ClientCancelRequest { asset, oid }, None)
            .await;

        match cancel {
            Ok(cancel) => match cancel {
                ExchangeResponseStatus::Ok(cancel) => {
                    if let Some(cancel) = cancel.data {
                        if !cancel.statuses.is_empty() {
                            match cancel.statuses[0].clone() {
                                ExchangeDataStatus::Success => {
                                    return true;
                                }
                                ExchangeDataStatus::Error(e) => {
                                    // Check if it's an expected "already gone" error
                                    if e.contains("already canceled") || e.contains("filled") || e.contains("never placed") {
                                        // Don't log as error - this is expected during fast markets
                                        return false;
                                    }
                                    error!("Error with cancelling: {e}") // Only log real errors
                                }
                                _ => {
                                    error!("Unexpected response status when cancelling: {:?}", cancel.statuses[0]);
                                }
                            }
                        } else {
                            error!("Exchange data statuses is empty when cancelling: {cancel:?}")
                        }
                    } else {
                        error!("Exchange response data is empty when cancelling: {cancel:?}")
                    }
                }
                ExchangeResponseStatus::Err(e) => error!("Error with cancelling: {e}"),
            },
            Err(e) => error!("Error with cancelling: {e}"),
        }
        false
    }

    async fn place_order(
        &self,
        asset: String,
        amount: f64,
        price: f64,
        is_buy: bool,
    ) -> (f64, u64) {
        // Validate price and size before placing order
        if let Err(e) = self.tick_lot_validator.validate_price(price) {
            error!("Invalid price {}: {}", price, e);
            return (0.0, 0);
        }
        
        if let Err(e) = self.tick_lot_validator.validate_size(amount) {
            error!("Invalid size {}: {}", amount, e);
            return (0.0, 0);
        }
        let order = self
            .exchange_client
            .order(
                ClientOrderRequest {
                    asset,
                    is_buy,
                    reduce_only: false,
                    limit_px: price,
                    sz: amount,
                    cloid: None,
                    order_type: ClientOrder::Limit(ClientLimit {
                        tif: "Gtc".to_string(),
                    }),
                },
                None,
            )
            .await;
        match order {
            Ok(order) => match order {
                ExchangeResponseStatus::Ok(order) => {
                    if let Some(order) = order.data {
                        if !order.statuses.is_empty() {
                            match order.statuses[0].clone() {
                                ExchangeDataStatus::Filled(order) => {
                                    return (amount, order.oid);
                                }
                                ExchangeDataStatus::Resting(order) => {
                                    return (amount, order.oid);
                                }
                                ExchangeDataStatus::Error(e) => {
                                    error!("Error with placing order: {e}")
                                }
                                _ => unreachable!(),
                            }
                        } else {
                            error!("Exchange data statuses is empty when placing order: {order:?}")
                        }
                    } else {
                        error!("Exchange response data is empty when placing order: {order:?}")
                    }
                }
                ExchangeResponseStatus::Err(e) => {
                    error!("Error with placing order: {e}")
                }
            },
            Err(e) => error!("Error with placing order: {e}"),
        }
        (0.0, 0)
    }

    /// Cancel all open orders and close any position, then shutdown gracefully
    pub async fn shutdown(&mut self) {
        info!("Shutting down market maker, cancelling all open orders and closing position...");
        
        // Cancel lower resting order if it exists
        if self.lower_resting.oid != 0 && self.lower_resting.position > EPSILON {
            info!("Cancelling lower resting order (oid: {})", self.lower_resting.oid);
            let cancelled = self.attempt_cancel(self.asset.clone(), self.lower_resting.oid).await;
            if cancelled {
                info!("Successfully cancelled lower resting order");
            } else {
                info!("Lower resting order was already filled or cancelled");
            }
            self.reset_resting_order(true);
        }
        
        // Cancel upper resting order if it exists
        if self.upper_resting.oid != 0 && self.upper_resting.position > EPSILON {
            info!("Cancelling upper resting order (oid: {})", self.upper_resting.oid);
            let cancelled = self.attempt_cancel(self.asset.clone(), self.upper_resting.oid).await;
            if cancelled {
                info!("Successfully cancelled upper resting order");
            } else {
                info!("Upper resting order was already filled or cancelled");
            }
            self.reset_resting_order(false);
        }
        
        // Close any existing position
        if self.cur_position.abs() > EPSILON {
            info!("Current position: {:.6} {}, closing position...", self.cur_position, self.asset);
            self.close_position().await;
        } else {
            info!("No position to close (position: {:.6})", self.cur_position);
        }
        
        info!("All orders cancelled and position closed. Market maker shutdown complete.");
    }

    /// Close the current position using a market order
    async fn close_position(&mut self) {
        let position_size = self.cur_position.abs();
        let is_sell = self.cur_position > 0.0; // If we're long, we need to sell to close
        
        // Validate the position size
        if let Err(e) = self.tick_lot_validator.validate_size(position_size) {
            error!("Invalid position size {}: {}", position_size, e);
            return;
        }
        
        info!(
            "Placing market order to {} {:.6} {} to close position",
            if is_sell { "sell" } else { "buy" },
            position_size,
            self.asset
        );
        
        // Use aggressive pricing based on current mid price to ensure fill
        let market_price = if is_sell {
            // For selling, use a price well below mid to ensure immediate fill
            self.latest_mid_price * 0.95  // 5% below mid
        } else {
            // For buying, use a price well above mid to ensure immediate fill  
            self.latest_mid_price * 1.05  // 5% above mid
        };
        
        let rounded_price = self.tick_lot_validator.round_price(market_price, !is_sell);
        
        let order = self
            .exchange_client
            .order(
                ClientOrderRequest {
                    asset: self.asset.clone(),
                    is_buy: !is_sell,
                    reduce_only: true, // This ensures we're only closing the position
                    limit_px: rounded_price,
                    sz: position_size,
                    cloid: None,
                    order_type: ClientOrder::Limit(ClientLimit {
                        tif: "Ioc".to_string(), // Immediate or Cancel for market-like behavior
                    }),
                },
                None,
            )
            .await;
            
        match order {
            Ok(order) => match order {
                ExchangeResponseStatus::Ok(order) => {
                    if let Some(order) = order.data {
                        if !order.statuses.is_empty() {
                            match order.statuses[0].clone() {
                                ExchangeDataStatus::Filled(_) => {
                                    info!("Position successfully closed with market order");
                                    self.cur_position = 0.0; // Reset position
                                }
                                ExchangeDataStatus::Resting(order) => {
                                    info!("Close order resting (oid: {}), attempting to cancel...", order.oid);
                                    // If it's still resting, cancel it and try again with more aggressive pricing
                                    self.attempt_cancel(self.asset.clone(), order.oid).await;
                                }
                                ExchangeDataStatus::Error(e) => {
                                    error!("Error closing position: {}", e);
                                }
                                _ => {
                                    error!("Unexpected order status when closing position: {:?}", order.statuses[0]);
                                }
                            }
                        } else {
                            error!("Empty order statuses when closing position");
                        }
                    } else {
                        error!("No order data when closing position");
                    }
                }
                ExchangeResponseStatus::Err(e) => {
                    error!("Error closing position: {}", e);
                }
            },
            Err(e) => {
                error!("Error placing close order: {}", e);
            }
        }
    }

    async fn potentially_update(&mut self) {
        // Clean up any invalid resting orders first
        self.cleanup_invalid_resting_orders();
        
        let half_spread = (self.latest_mid_price * self.half_spread as f64) / 10000.0;
        // Determine prices to target from the half spread
        let (lower_price, upper_price) = (
            self.latest_mid_price - half_spread,
            self.latest_mid_price + half_spread,
        );
        let (mut lower_price, mut upper_price) = (
            self.tick_lot_validator.round_price(lower_price, false), // Round down for buy orders
            self.tick_lot_validator.round_price(upper_price, true),  // Round up for sell orders
        );

        // Rounding optimistically to make our market tighter might cause a weird edge case, so account for that
        if (lower_price - upper_price).abs() < EPSILON {
            lower_price = self.tick_lot_validator.round_price(lower_price, true);
            upper_price = self.tick_lot_validator.round_price(upper_price, false);
        }

        // Determine amounts we can put on the book without exceeding the max absolute position size
        let lower_order_amount = self.tick_lot_validator.round_size(
            (self.max_absolute_position_size - self.cur_position)
                .min(self.target_liquidity)
                .max(0.0),
            false, // Round down for size
        );

        let upper_order_amount = self.tick_lot_validator.round_size(
            (self.max_absolute_position_size + self.cur_position)
                .min(self.target_liquidity)
                .max(0.0),
            false, // Round down for size
        );

        // Determine if we need to cancel the resting order and put a new order up due to deviation
        let lower_change = (lower_order_amount - self.lower_resting.position).abs() > EPSILON
            || bps_diff(lower_price, self.lower_resting.price) > self.max_bps_diff;
        let upper_change = (upper_order_amount - self.upper_resting.position).abs() > EPSILON
            || bps_diff(upper_price, self.upper_resting.price) > self.max_bps_diff;

        // Consider cancelling
        // TODO: Don't block on cancels
        let mut lower_cancelled = false;
        if self.lower_resting.oid != 0 && self.lower_resting.position > EPSILON && lower_change {
            let cancel = self
                .attempt_cancel(self.asset.clone(), self.lower_resting.oid)
                .await;
            if cancel {
                info!("Cancelled buy order: {:?}", self.lower_resting);
                lower_cancelled = true;
            } else {
                // Cancel failed - likely because order was already filled or cancelled
                // Reset the resting order state since it's no longer valid
                info!(
                    "Cancel failed for buy order (oid: {}, pos: {}) - treating as filled and resetting", 
                    self.lower_resting.oid, 
                    self.lower_resting.position
                );
                self.reset_resting_order(true);
                lower_cancelled = true; // Treat as cancelled for placement logic
            }
        }

        let mut upper_cancelled = false;
        if self.upper_resting.oid != 0 && self.upper_resting.position > EPSILON && upper_change {
            let cancel = self
                .attempt_cancel(self.asset.clone(), self.upper_resting.oid)
                .await;
            if cancel {
                info!("Cancelled sell order: {:?}", self.upper_resting);
                upper_cancelled = true;
            } else {
                // Cancel failed - likely because order was already filled or cancelled
                // Reset the resting order state since it's no longer valid
                info!(
                    "Cancel failed for sell order (oid: {}, pos: {}) - treating as filled and resetting", 
                    self.upper_resting.oid, 
                    self.upper_resting.position
                );
                self.reset_resting_order(false);
                upper_cancelled = true; // Treat as cancelled for placement logic
            }
        }

        // Consider putting a new order up
        if lower_order_amount > EPSILON && (lower_cancelled || (lower_change && self.lower_resting.oid == 0)) {
            let (amount_resting, oid) = self
                .place_order(self.asset.clone(), lower_order_amount, lower_price, true)
                .await;

            self.lower_resting.oid = oid;
            self.lower_resting.position = amount_resting;
            self.lower_resting.price = lower_price;

            if amount_resting > EPSILON {
                info!(
                    "Buy for {amount_resting} {} resting at {lower_price}",
                    self.asset.clone()
                );
            }
        }

        if upper_order_amount > EPSILON && (upper_cancelled || (upper_change && self.upper_resting.oid == 0)) {
            let (amount_resting, oid) = self
                .place_order(self.asset.clone(), upper_order_amount, upper_price, false)
                .await;
            self.upper_resting.oid = oid;
            self.upper_resting.position = amount_resting;
            self.upper_resting.price = upper_price;

            if amount_resting > EPSILON {
                info!(
                    "Sell for {amount_resting} {} resting at {upper_price}",
                    self.asset.clone()
                );
            }
        }
    }
}