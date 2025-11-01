// ============================================================================
// Strategies Module - Pluggable Trading Strategy Implementations
// ============================================================================
//
// This module contains concrete implementations of the Strategy trait.
// Each strategy is a self-contained module that encapsulates all logic for
// a specific trading approach.
//
// # Available Strategies
//
// - `hjb_strategy`: HJB-based market making with Hawkes processes, robust
//   control, and multi-level optimization. This is the current production
//   strategy with advanced stochastic control features.
//
// # Adding a New Strategy
//
// 1. Create a new file: `src/strategies/my_strategy.rs`
// 2. Implement the `Strategy` trait from `crate::strategy`
// 3. Add the module declaration below: `pub mod my_strategy;`
// 4. Add the strategy to the factory in `src/bin/market_maker_v2.rs`
// 5. Update your `config.json` to use the new strategy:
//    ```json
//    {
//      "strategy_name": "my_strategy",
//      "strategy_params": { ... }
//    }
//    ```

pub mod hjb_strategy;
pub mod hjb_strategy_v2;

// Component-based architecture for strategies
pub mod components;

// Private implementation module for HJB strategy
mod hjb_impl;

// Auto-tuner integration module
pub mod tuner_integration;

// Async tuner actor for non-blocking parameter optimization
pub mod async_tuner_actor;

// Re-export strategies
pub use hjb_strategy::HjbStrategy;
pub use hjb_strategy_v2::HjbStrategyV2;

// Re-export core HJB types from the private implementation module
// These are needed by users of the hjb_strategy and other modules
pub use hjb_impl::{
    ConstrainedTuningParams, ControlVector, FillHistory, HawkesFillModel, HawkesParams,
    HJBComponents, MultiLevelConfig, MultiLevelControl, MultiLevelOptimizer,
    OnlineAdverseSelectionModel, OptimizationState, StateVector, TuningParams, ValueFunction,
};

// Future strategies can be added here:
// pub mod grid_strategy;
// pub mod momentum_strategy;
// pub mod stat_arb_strategy;
