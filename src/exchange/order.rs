use std::collections::HashMap;

use alloy::signers::local::PrivateKeySigner;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::{
    errors::Error,
    helpers::{float_to_string_for_hashing, float_to_string_with_decimals, uuid_to_hex_string},
    meta::AssetMeta,
    prelude::*,
};

#[derive(Deserialize, Serialize, Clone, Debug)]
pub struct Limit {
    pub tif: String,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct Trigger {
    pub is_market: bool,
    pub trigger_px: String,
    pub tpsl: String,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub enum Order {
    Limit(Limit),
    Trigger(Trigger),
}

#[derive(Deserialize, Serialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct OrderRequest {
    #[serde(rename = "a", alias = "asset")]
    pub asset: u32,
    #[serde(rename = "b", alias = "isBuy")]
    pub is_buy: bool,
    #[serde(rename = "p", alias = "limitPx")]
    pub limit_px: String,
    #[serde(rename = "s", alias = "sz")]
    pub sz: String,
    #[serde(rename = "r", alias = "reduceOnly", default)]
    pub reduce_only: bool,
    #[serde(rename = "t", alias = "orderType")]
    pub order_type: Order,
    #[serde(rename = "c", alias = "cloid", skip_serializing_if = "Option::is_none")]
    pub cloid: Option<String>,
}

#[derive(Debug, Clone)]
pub struct ClientLimit {
    pub tif: String,
}

#[derive(Debug, Clone)]
pub struct ClientTrigger {
    pub is_market: bool,
    pub trigger_px: f64,
    pub tpsl: String,
}

#[derive(Debug)]
pub struct MarketOrderParams<'a> {
    pub asset: &'a str,
    pub is_buy: bool,
    pub sz: f64,
    pub px: Option<f64>,
    pub slippage: Option<f64>,
    pub cloid: Option<Uuid>,
    pub wallet: Option<&'a PrivateKeySigner>,
}

#[derive(Debug)]
pub struct MarketCloseParams<'a> {
    pub asset: &'a str,
    pub sz: Option<f64>,
    pub px: Option<f64>,
    pub slippage: Option<f64>,
    pub cloid: Option<Uuid>,
    pub wallet: Option<&'a PrivateKeySigner>,
}

#[derive(Debug, Clone)]
pub enum ClientOrder {
    Limit(ClientLimit),
    Trigger(ClientTrigger),
}

#[derive(Debug, Clone)]
pub struct ClientOrderRequest {
    pub asset: String,
    pub is_buy: bool,
    pub reduce_only: bool,
    pub limit_px: f64,
    pub sz: f64,
    pub cloid: Option<Uuid>,
    pub order_type: ClientOrder,
}

impl ClientOrderRequest {
    pub(crate) fn convert(
        self,
        coin_to_asset: &HashMap<String, u32>,
        asset_metas: &[AssetMeta],
    ) -> Result<OrderRequest> {
        let order_type = match self.order_type {
            ClientOrder::Limit(limit) => Order::Limit(Limit { tif: limit.tif }),
            ClientOrder::Trigger(trigger) => Order::Trigger(Trigger {
                trigger_px: float_to_string_for_hashing(trigger.trigger_px),
                is_market: trigger.is_market,
                tpsl: trigger.tpsl,
            }),
        };
        let &asset = coin_to_asset.get(&self.asset).ok_or(Error::AssetNotFound)?;

        let cloid = self.cloid.map(uuid_to_hex_string);

        // Get asset metadata for decimal formatting
        // For spot assets (>= 10000), we can't look them up in asset_metas, so use default formatting
        // For perp assets, look up szDecimals to format correctly
        let (sz_str, px_str) = if asset < 10000 {
            let asset_meta = asset_metas.get(asset as usize)
                .ok_or(Error::AssetNotFound)?;
            let sz_decimals = asset_meta.sz_decimals;
            // For perps: MAX_DECIMALS_PERP = 6
            const MAX_DECIMALS_PERP: u32 = 6;
            let price_decimals = MAX_DECIMALS_PERP.saturating_sub(sz_decimals);

            (
                float_to_string_with_decimals(self.sz, sz_decimals),
                float_to_string_with_decimals(self.limit_px, price_decimals),
            )
        } else {
            // Spot assets - use default formatting
            (
                float_to_string_for_hashing(self.sz),
                float_to_string_for_hashing(self.limit_px),
            )
        };

        Ok(OrderRequest {
            asset,
            is_buy: self.is_buy,
            reduce_only: self.reduce_only,
            limit_px: px_str,
            sz: sz_str,
            order_type,
            cloid,
        })
    }
}
