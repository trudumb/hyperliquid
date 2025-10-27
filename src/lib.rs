#![deny(unreachable_pub)]
mod book_analyzer;
mod consts;
mod eip712;
mod errors;
mod exchange;
mod hawkes_multi_level;
mod helpers;
mod info;
mod inventory_skew;
pub mod market_maker_v2; // Re-added for shared types only
mod meta;
pub mod strategy;
pub mod strategies;
mod prelude;
mod req;
mod robust_hjb_control;
mod signature;
mod stochastic_volatility;
mod tick_lot_size;
pub mod tui;
mod ws;
pub use book_analyzer::{BookAnalysis, OrderBook};
pub use consts::{EPSILON, LOCAL_API_URL, MAINNET_API_URL, TESTNET_API_URL};
pub use eip712::Eip712;
pub use errors::Error;
pub use exchange::*;
pub use hawkes_multi_level::{
    FillHistory, HawkesFillModel, HawkesParams, MultiLevelConfig, 
    MultiLevelControl, MultiLevelOptimizer, OptimizationState,
};
pub use helpers::{bps_diff, truncate_float, BaseUrl};
pub use info::{info_client::*, *};
pub use inventory_skew::{InventorySkewCalculator, InventorySkewConfig, SkewResult};
// Re-export shared types from market_maker_v2 for backward compatibility
// Note: The MarketMaker struct itself has been replaced by the modular v3 architecture
pub use market_maker_v2::{
    StateVector, ControlVector, ValueFunction, HJBComponents,
    TuningParams, ConstrainedTuningParams, AdamOptimizerState,
    OnlineAdverseSelectionModel, CachedVolatilityEstimate,
};
pub use meta::{AssetContext, AssetMeta, Meta, MetaAndAssetCtxs, SpotAssetMeta, SpotMeta};
pub use strategy::{
    CurrentState, MarketUpdate, RestingOrder, Strategy, StrategyAction, UserUpdate,
};
pub use robust_hjb_control::{ParameterUncertainty, RobustConfig, RobustParameters};
pub use stochastic_volatility::{AdaptiveConfig, Particle, ParticleFilterState};
pub use tick_lot_size::{AssetType, TickLotValidator};
pub use ws::*;
