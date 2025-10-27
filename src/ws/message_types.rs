use alloy::primitives::Address;
use serde::{Deserialize, Serialize};

use crate::ws::sub_structs::*;

#[derive(Deserialize, Clone, Debug)]
pub struct Trades {
    pub data: Vec<Trade>,
}

#[derive(Deserialize, Clone, Debug)]
pub struct L2Book {
    pub data: L2BookData,
}

#[derive(Deserialize, Clone, Debug)]
pub struct AllMids {
    pub data: AllMidsData,
}

#[derive(Deserialize, Clone, Debug)]
pub struct User {
    pub data: UserData,
}

#[derive(Deserialize, Clone, Debug)]
pub struct UserFills {
    pub data: UserFillsData,
}

#[derive(Deserialize, Clone, Debug)]
pub struct Candle {
    pub data: CandleData,
}

#[derive(Deserialize, Clone, Debug)]
pub struct OrderUpdates {
    pub data: Vec<OrderUpdate>,
}

#[derive(Deserialize, Clone, Debug)]
pub struct UserFundings {
    pub data: UserFundingsData,
}

#[derive(Deserialize, Clone, Debug)]
pub struct UserNonFundingLedgerUpdates {
    pub data: UserNonFundingLedgerUpdatesData,
}

#[derive(Deserialize, Clone, Debug)]
pub struct Notification {
    pub data: NotificationData,
}

#[derive(Deserialize, Clone, Debug)]
pub struct WebData2 {
    pub data: WebData2Data,
}

#[derive(Deserialize, Clone, Debug)]
pub struct ActiveAssetCtx {
    pub data: ActiveAssetCtxData,
}

#[derive(Deserialize, Clone, Debug)]
pub struct ActiveSpotAssetCtx {
    pub data: ActiveSpotAssetCtxData,
}

#[derive(Deserialize, Clone, Debug)]
pub struct ActiveAssetData {
    pub data: ActiveAssetDataData,
}

#[derive(Deserialize, Clone, Debug)]
pub struct Bbo {
    pub data: BboData,
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(tag = "type")]
#[serde(rename_all = "camelCase")]
pub enum Subscription {
    AllMids,
    Notification { user: Address },
    WebData2 { user: Address },
    Candle { coin: String, interval: String },
    L2Book { coin: String },
    Trades { coin: String },
    OrderUpdates { user: Address },
    UserEvents { user: Address },
    UserFills { user: Address },
    UserFundings { user: Address },
    UserNonFundingLedgerUpdates { user: Address },
    ActiveAssetCtx { coin: String },
    ActiveAssetData { user: Address, coin: String },
    Bbo { coin: String },
}

#[derive(Deserialize, Clone, Debug)]
#[serde(tag = "channel")]
#[serde(rename_all = "camelCase")]
pub enum Message {
    NoData,
    HyperliquidError(String),
    AllMids(AllMids),
    Trades(Trades),
    L2Book(L2Book),
    User(User),
    UserFills(UserFills),
    Candle(Candle),
    SubscriptionResponse,
    OrderUpdates(OrderUpdates),
    UserFundings(UserFundings),
    UserNonFundingLedgerUpdates(UserNonFundingLedgerUpdates),
    Notification(Notification),
    WebData2(WebData2),
    ActiveAssetCtx(ActiveAssetCtx),
    ActiveAssetData(ActiveAssetData),
    ActiveSpotAssetCtx(ActiveSpotAssetCtx),
    Bbo(Bbo),
    Pong,
}
