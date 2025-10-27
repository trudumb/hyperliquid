#![deny(unreachable_pub)]
mod consts;
mod eip712;
mod errors;
mod exchange;
mod helpers;
mod info;
mod meta;
pub mod rate_limiter;
pub mod strategy;
pub mod strategies;
mod prelude;
mod req;
mod signature;
mod tick_lot_size;
pub mod tui;
mod ws;

// ============================================================================
// Core SDK Exports
// ============================================================================
pub use consts::{EPSILON, LOCAL_API_URL, MAINNET_API_URL, TESTNET_API_URL};
pub use eip712::Eip712;
pub use errors::Error;
pub use exchange::*;
pub use helpers::{bps_diff, truncate_float, BaseUrl};
pub use info::{info_client::*, *};
pub use meta::{AssetContext, AssetMeta, Meta, MetaAndAssetCtxs, SpotAssetMeta, SpotMeta};
pub use strategy::{
    CurrentState, MarketUpdate, RestingOrder, Strategy, StrategyAction, StrategyTuiMetrics, UserUpdate,
};
pub use tick_lot_size::{AssetType, TickLotValidator};
pub use ws::*;

// ============================================================================
// Strategy Framework Exports
// ============================================================================

// Re-export core HJB types from the strategies module
pub use strategies::{
    ConstrainedTuningParams, ControlVector, FillHistory, HawkesFillModel, HawkesParams,
    HJBComponents, MultiLevelConfig, MultiLevelControl, MultiLevelOptimizer,
    OnlineAdverseSelectionModel, OptimizationState, StateVector, TuningParams, ValueFunction,
};

// Re-export component-based strategy building blocks (NEW - recommended for new code)
pub use strategies::components::{
    // Traits
    AdverseSelectionModel, FillModel, InventorySkewModel, QuoteOptimizer,
    RobustControlModel, VolatilityModel,
    // Implementations
    HawkesFillModelImpl, HjbMultiLevelOptimizer, InventorySkewCalculator, OnlineSgdAsModel,
    ParticleFilterVolModel, StandardInventorySkew, StandardRobustControl,
    // Supporting types
    OptimizerInputs, OptimizerOutput, ParameterUncertainty, RobustParameters, SkewResult,
    // Order book types
    BookAnalysis, OrderBook,
};

// ============================================================================
// Legacy Module Exports (Deprecated - use strategies::components instead)
// ============================================================================
//
// These exports are kept for backward compatibility with existing code.
// New code should use the component-based architecture via strategies::components.
//
// The old modules (hawkes_multi_level, inventory_skew, robust_hjb_control,
// stochastic_volatility) provide the underlying implementations but are being
// wrapped in a more modular component system.
//
// Migration path:
// - Old: HawkesFillModel (direct use)
// - New: HawkesFillModelImpl (component wrapper) implementing FillModel trait
//
// - Old: InventorySkewCalculator (direct use)
// - New: StandardInventorySkew (component) implementing InventorySkewModel trait
//
// - Old: RobustConfig + RobustParameters::compute (static method)
// - New: StandardRobustControl (component) implementing RobustControlModel trait
//
// - Old: ParticleFilterState (direct use)
// - New: ParticleFilterVolModel (component wrapper) implementing VolatilityModel trait

// Note: All legacy types are now in strategies::components
// These exports maintain backward compatibility for existing code
pub use strategies::components::InventorySkewConfig as LegacyInventorySkewConfig;
pub use strategies::components::{
    RobustConfig as LegacyRobustConfig,
    RobustParameters as LegacyRobustParameters,
    ParameterUncertainty as LegacyParameterUncertainty,
};

// Re-export particle filter types from the new component location
pub use strategies::components::particle_filter_vol::{AdaptiveConfig, Particle, ParticleFilterState};
