//! Private implementation module for the HJB strategy.
//!
//! This module contains the core structs (StateVector, TuningParams, etc.)
//! that were originally in `market_maker_v2.rs`. They are now encapsulated
//! here as an internal detail of the `hjb_strategy`.

// Suppress unreachable_pub warnings for items that are pub for easier unit testing
#![allow(unreachable_pub)]

// Declare the internal modules
mod cache;
mod models;
mod state;
mod tuning;
mod utils;
pub mod hawkes;
pub mod multi_level;

// Publicly re-export the structs needed by hjb_strategy.rs
pub use cache::CachedVolatilityEstimate;
pub use models::{ControlVector, HJBComponents, OnlineAdverseSelectionModel, ValueFunction};
pub use state::StateVector;
pub use tuning::{ConstrainedTuningParams, TuningParams};

// Re-export hawkes and multi-level types for convenience
pub use hawkes::{FillHistory, HawkesFillModel, HawkesParams};
pub use multi_level::{MultiLevelConfig, MultiLevelControl, MultiLevelOptimizer, OptimizationState};
