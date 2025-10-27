// ============================================================================
// Strategy Components Module - Pluggable Strategy Building Blocks
// ============================================================================
//
// This module provides a set of traits and implementations for modular
// strategy components. Each component handles a specific aspect of the
// trading strategy (volatility modeling, fill rate estimation, etc.).
//
// # Architecture
//
// The component-based design allows strategies to:
// - Mix and match different implementations (e.g., swap volatility models)
// - Test components in isolation
// - Share components across multiple strategies
// - Iterate on individual components without modifying core strategy logic
//
// # Component Traits
//
// - `VolatilityModel`: Estimates market volatility
// - `FillModel`: Models fill probabilities and rates
// - `AdverseSelectionModel`: Estimates adverse selection
// - `QuoteOptimizer`: Calculates optimal quotes given model inputs
//
// # Example Usage
//
// ```rust
// use strategies::components::{
//     VolatilityModel, FillModel, AdverseSelectionModel, QuoteOptimizer,
// };
//
// struct MyStrategy {
//     vol_model: Box<dyn VolatilityModel>,
//     fill_model: Box<dyn FillModel>,
//     as_model: Box<dyn AdverseSelectionModel>,
//     optimizer: Box<dyn QuoteOptimizer>,
// }
// ```

// Trait definitions
pub mod volatility;
pub mod fill_model;
pub mod adverse_selection;
pub mod quote_optimizer;
pub mod inventory_skew;
pub mod robust_control;

// Concrete implementations
pub mod particle_filter_vol;
pub mod hawkes_fill_model;
pub mod online_sgd_as;
pub mod hjb_multi_level_optimizer;
pub mod inventory_skew_impl;
pub mod robust_control_impl;

// Order book analysis (supporting utilities)
pub mod book_analyzer;

// Re-export traits for convenience
pub use volatility::VolatilityModel;
pub use fill_model::FillModel;
pub use adverse_selection::AdverseSelectionModel;
pub use quote_optimizer::{QuoteOptimizer, OptimizerInputs};
pub use inventory_skew::{InventorySkewModel, SkewResult};
pub use robust_control::{RobustControlModel, RobustParameters, ParameterUncertainty};

// Re-export concrete implementations
pub use particle_filter_vol::ParticleFilterVolModel;
pub use hawkes_fill_model::HawkesFillModelImpl;
pub use online_sgd_as::OnlineSgdAsModel;
pub use hjb_multi_level_optimizer::{HjbMultiLevelOptimizer, OptimizerOutput};
pub use inventory_skew_impl::{StandardInventorySkew, InventorySkewConfig, InventorySkewCalculator};
pub use robust_control_impl::{StandardRobustControl, RobustConfig};

// Re-export book analyzer types
pub use book_analyzer::{BookAnalysis, OrderBook};
