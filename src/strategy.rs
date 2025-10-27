// ============================================================================
// Strategy Trait & Core Abstractions for Modular Market Making
// ============================================================================
//
// This module defines the Strategy trait and supporting data structures to
// enable modular, plug-and-play trading strategies. The core idea is to
// separate:
//   1. Core Services (exchange, websocket, TUI) - the "bot runner"
//   2. Strategy Logic (HJB, Grid, etc.) - the "pluggable brain"
//
// Any object implementing the Strategy trait can be dropped into the main
// binary without modifying core infrastructure code.

use serde_json::Value;

use crate::{
    ClientCancelRequest, ClientOrderRequest,
    L2BookData, OrderBook, Trade, TradeInfo,
};

// ============================================================================
// 1. ACTIONS - What strategies can DO
// ============================================================================

/// Actions a strategy can decide to take.
/// These are returned from strategy callbacks and executed by the bot runner.
#[derive(Debug, Clone)]
pub enum StrategyAction {
    /// Place a new order
    Place(ClientOrderRequest),

    /// Cancel an existing order
    Cancel(ClientCancelRequest),

    /// Batch place multiple orders (more efficient than individual placements)
    BatchPlace(Vec<ClientOrderRequest>),

    /// Batch cancel multiple orders (more efficient than individual cancels)
    BatchCancel(Vec<ClientCancelRequest>),

    /// No action - strategy is satisfied with current state
    NoOp,
}

// ============================================================================
// 2. MARKET DATA - What strategies OBSERVE
// ============================================================================

/// All market data updates from WebSocket feeds.
/// This struct consolidates L2 book updates, trade flow, and mid-price changes.
#[derive(Debug, Clone)]
pub struct MarketUpdate {
    /// Asset symbol (e.g., "HYPE", "BTC")
    pub asset: String,

    /// L2 order book snapshot (optional, only present on book updates)
    pub l2_book: Option<L2BookData>,

    /// Recent trades (empty if no new trades)
    pub trades: Vec<Trade>,

    /// Mid-price from AllMids stream (optional)
    pub mid_price: Option<f64>,

    /// Unix timestamp (milliseconds) of this update
    pub timestamp: u64,
}

impl MarketUpdate {
    /// Create a market update from an L2 book snapshot
    pub fn from_l2_book(book: L2BookData) -> Self {
        Self {
            asset: book.coin.clone(),
            l2_book: Some(book),
            trades: Vec::new(),
            mid_price: None,
            timestamp: chrono::Utc::now().timestamp_millis() as u64,
        }
    }

    /// Create a market update from trades
    pub fn from_trades(asset: String, trades: Vec<Trade>) -> Self {
        Self {
            asset,
            l2_book: None,
            trades,
            mid_price: None,
            timestamp: chrono::Utc::now().timestamp_millis() as u64,
        }
    }

    /// Create a market update from a mid-price change
    pub fn from_mid_price(asset: String, mid_price: f64) -> Self {
        Self {
            asset,
            l2_book: None,
            trades: Vec::new(),
            mid_price: Some(mid_price),
            timestamp: chrono::Utc::now().timestamp_millis() as u64,
        }
    }
}

// ============================================================================
// 3. USER DATA - What strategies REACT TO
// ============================================================================

/// All user-specific updates (fills, order status changes, etc.).
/// The bot runner updates CurrentState BEFORE passing UserUpdate to the strategy.
#[derive(Debug, Clone)]
pub struct UserUpdate {
    /// New fills that occurred since last update
    pub fills: Vec<TradeInfo>,

    /// Order IDs that were successfully placed
    pub orders_placed: Vec<u64>,

    /// Order IDs that were successfully cancelled
    pub orders_cancelled: Vec<u64>,

    /// Order IDs that failed to place (with error messages)
    pub orders_failed: Vec<(u64, String)>,
}

impl UserUpdate {
    /// Create an empty user update
    pub fn empty() -> Self {
        Self {
            fills: Vec::new(),
            orders_placed: Vec::new(),
            orders_cancelled: Vec::new(),
            orders_failed: Vec::new(),
        }
    }

    /// Create a user update from fills only
    pub fn from_fills(fills: Vec<TradeInfo>) -> Self {
        Self {
            fills,
            orders_placed: Vec::new(),
            orders_cancelled: Vec::new(),
            orders_failed: Vec::new(),
        }
    }
}

// ============================================================================
// 4. CURRENT STATE - What strategies KNOW
// ============================================================================

/// Complete snapshot of the bot's current state.
/// This is the "read-only view" that strategies use to make decisions.
/// The bot runner maintains this state and passes it to strategy callbacks.
#[derive(Debug, Clone)]
pub struct CurrentState {
    // ----- Position & PnL -----
    /// Current position (positive = long, negative = short)
    pub position: f64,

    /// Average entry price of current position
    pub avg_entry_price: f64,

    /// Cost basis of current position (USD)
    pub cost_basis: f64,

    /// Unrealized PnL (mark-to-market)
    pub unrealized_pnl: f64,

    /// Realized PnL this session
    pub realized_pnl: f64,

    /// Total fees paid this session
    pub total_fees: f64,

    // ----- Market Data -----
    /// Latest mid-price from L2 book (authoritative price for order placement)
    pub l2_mid_price: f64,

    /// Latest order book snapshot
    pub order_book: Option<OrderBook>,

    /// Market spread in basis points (ask - bid)
    pub market_spread_bps: f64,

    /// Order book imbalance (bid_volume / (bid_volume + ask_volume))
    pub lob_imbalance: f64,

    // ----- Open Orders -----
    /// Resting bid orders, sorted by level (L1 = tightest)
    pub open_bids: Vec<RestingOrder>,

    /// Resting ask orders, sorted by level (L1 = tightest)
    pub open_asks: Vec<RestingOrder>,

    // ----- Account Info -----
    /// Account equity (from margin summary)
    pub account_equity: f64,

    /// Total margin used
    pub margin_used: f64,

    /// Maximum absolute position size allowed
    pub max_position_size: f64,

    // ----- Timing -----
    /// Unix timestamp (seconds) of this state snapshot
    pub timestamp: f64,

    /// Session start time (Unix timestamp)
    pub session_start_time: f64,
}

/// Simplified representation of a resting order for strategy consumption
#[derive(Debug, Clone)]
pub struct RestingOrder {
    /// Order ID
    pub oid: u64,

    /// Order size (positive)
    pub size: f64,

    /// Order price
    pub price: f64,

    /// Order level (0 = L1/tightest, 1 = L2, etc.)
    pub level: usize,

    /// Whether a cancel has been sent but not yet confirmed
    pub pending_cancel: bool,
}

impl Default for CurrentState {
    fn default() -> Self {
        Self {
            position: 0.0,
            avg_entry_price: 0.0,
            cost_basis: 0.0,
            unrealized_pnl: 0.0,
            realized_pnl: 0.0,
            total_fees: 0.0,
            l2_mid_price: 0.0,
            order_book: None,
            market_spread_bps: 0.0,
            lob_imbalance: 0.5,
            open_bids: Vec::new(),
            open_asks: Vec::new(),
            account_equity: 0.0,
            margin_used: 0.0,
            max_position_size: 0.0,
            timestamp: 0.0,
            session_start_time: 0.0,
        }
    }
}

// ============================================================================
// 5. STRATEGY TUI METRICS - Strategy-specific data for dashboard display
// ============================================================================

/// Metrics from the strategy's internal state to display in the TUI.
/// These are read-only snapshots of strategy internals (volatility, learning models, etc.)
#[derive(Debug, Clone, Default)]
pub struct StrategyTuiMetrics {
    // ----- Volatility Estimation -----
    /// EMA of realized volatility in basis points
    pub volatility_ema_bps: f64,

    // ----- Particle Filter Stats -----
    /// Effective sample size (ESS) of particle filter
    pub pf_ess: f64,

    /// Maximum number of particles in the filter
    pub pf_max_particles: usize,

    /// 5th percentile of volatility distribution (bps)
    pub pf_vol_5th: f64,

    /// 95th percentile of volatility distribution (bps)
    pub pf_vol_95th: f64,

    /// Mean volatility from particle filter (bps)
    pub pf_volatility_bps: f64,

    // ----- Adverse Selection & Flow -----
    /// Adverse selection estimate in basis points
    pub adverse_selection_estimate: f64,

    /// Trade flow EMA (positive = buying pressure)
    pub trade_flow_ema: f64,

    // ----- Online Learning Model -----
    /// Mean absolute error of online model
    pub online_model_mae: f64,

    /// Number of model updates
    pub online_model_updates: u64,

    /// Current learning rate
    pub online_model_lr: f64,

    /// Whether online learning is enabled
    pub online_model_enabled: bool,

    // ----- Adam Optimizer (Self-Tuning) -----
    /// Number of gradient samples collected
    pub adam_gradient_samples: u64,

    /// Average loss (for convergence monitoring)
    pub adam_avg_loss: f64,

    /// Seconds since last optimizer update
    pub adam_last_update_secs: f64,

    // ----- Performance Metrics -----
    /// Sharpe ratio (annualized)
    pub sharpe_ratio: f64,
}

// ============================================================================
// 6. THE STRATEGY TRAIT - The "contract" all strategies must implement
// ============================================================================

/// The Strategy trait defines the interface that all trading strategies must implement.
///
/// # Design Philosophy
///
/// Strategies are **pure decision-making logic**. They:
/// - Receive market data and current state as input
/// - Return a list of actions (place/cancel orders) as output
/// - Do NOT perform any I/O (no exchange API calls, no WebSocket management)
/// - Do NOT maintain global state (all state is passed in via CurrentState)
///
/// The bot runner handles:
/// - WebSocket connection management
/// - Exchange API calls (order placement/cancellation)
/// - State reconciliation (tracking fills, order updates)
/// - TUI rendering
///
/// # Lifecycle
///
/// 1. `new()` - Initialize strategy with asset and config
/// 2. `on_market_update()` - Called on every market data event (L2, trades, AllMids)
/// 3. `on_user_update()` - Called after fills or order status changes
/// 4. `on_tick()` - (Optional) Called on a fixed timer (e.g., every 1 second)
///
/// # Example Implementation
///
/// ```rust
/// struct SimpleGridStrategy {
///     grid_spacing_bps: f64,
///     order_size: f64,
/// }
///
/// impl Strategy for SimpleGridStrategy {
///     fn new(asset: &str, config: &Value) -> Self {
///         Self {
///             grid_spacing_bps: config["grid_spacing_bps"].as_f64().unwrap(),
///             order_size: config["order_size"].as_f64().unwrap(),
///         }
///     }
///
///     fn on_market_update(&mut self, state: &CurrentState, update: &MarketUpdate)
///         -> Vec<StrategyAction>
///     {
///         let mut actions = Vec::new();
///
///         // Calculate ideal grid levels
///         let bid_price = state.l2_mid_price * (1.0 - self.grid_spacing_bps / 10000.0);
///         let ask_price = state.l2_mid_price * (1.0 + self.grid_spacing_bps / 10000.0);
///
///         // If no orders exist, place them
///         if state.open_bids.is_empty() {
///             actions.push(StrategyAction::Place(/* bid order */));
///         }
///         if state.open_asks.is_empty() {
///             actions.push(StrategyAction::Place(/* ask order */));
///         }
///
///         actions
///     }
///
///     fn on_user_update(&mut self, state: &CurrentState, update: &UserUpdate)
///         -> Vec<StrategyAction>
///     {
///         // Re-quote after fills
///         Vec::new()
///     }
/// }
/// ```
pub trait Strategy: Send {
    /// Initialize the strategy with asset and JSON configuration.
    ///
    /// # Arguments
    /// - `asset`: Trading pair (e.g., "HYPE", "BTC")
    /// - `config`: Strategy-specific configuration from JSON file
    ///
    /// # Example Config
    /// ```json
    /// {
    ///   "strategy_name": "hjb_v1",
    ///   "strategy_params": {
    ///     "phi": 0.01,
    ///     "lambda_base": 1.0,
    ///     "enable_multi_level": true,
    ///     "multi_level_config": { ... }
    ///   }
    /// }
    /// ```
    fn new(asset: &str, config: &Value) -> Self
    where
        Self: Sized;

    /// Called on every market data update (L2 book, trades, mid-price changes).
    ///
    /// This is the "hot path" - it should be fast and efficient.
    ///
    /// # Arguments
    /// - `state`: Current bot state (position, orders, market data)
    /// - `update`: New market data (book snapshot, trades, or mid-price)
    ///
    /// # Returns
    /// A list of actions to take (place/cancel orders, or NoOp)
    ///
    /// # Notes
    /// - The bot runner will execute all returned actions asynchronously
    /// - Order placement failures are reported via `on_user_update()`
    /// - This method should be stateless (all state in `self` is private)
    fn on_market_update(
        &mut self,
        state: &CurrentState,
        update: &MarketUpdate,
    ) -> Vec<StrategyAction>;

    /// Called after user-specific events (fills, order placements, cancellations).
    ///
    /// The bot runner updates `state` BEFORE calling this method, so:
    /// - `state.position` already reflects the fill
    /// - `state.open_bids` / `state.open_asks` already reflect order updates
    ///
    /// # Arguments
    /// - `state`: Updated bot state (after fills/order updates)
    /// - `update`: User events that just occurred
    ///
    /// # Returns
    /// A list of actions to take (typically NoOp or re-quote logic)
    ///
    /// # Common Use Cases
    /// - Re-quote immediately after a fill (inventory changed)
    /// - Cancel all orders if position limit hit
    /// - Log fill events for post-trade analysis
    fn on_user_update(
        &mut self,
        state: &CurrentState,
        update: &UserUpdate,
    ) -> Vec<StrategyAction>;

    /// (Optional) Called on a fixed timer (e.g., every 1 second).
    ///
    /// Use this for periodic tasks:
    /// - Heartbeat checks
    /// - Stale order cleanup
    /// - Parameter updates
    ///
    /// # Arguments
    /// - `state`: Current bot state
    ///
    /// # Returns
    /// A list of actions to take
    ///
    /// # Default Implementation
    /// Does nothing (returns empty vec)
    fn on_tick(&mut self, _state: &CurrentState) -> Vec<StrategyAction> {
        Vec::new()
    }

    /// (Optional) Called when the bot is shutting down.
    ///
    /// Use this for cleanup:
    /// - Save strategy state to disk
    /// - Cancel all open orders
    /// - Log final statistics
    ///
    /// # Arguments
    /// - `state`: Final bot state
    ///
    /// # Returns
    /// A list of final actions to take (e.g., cancel all orders)
    ///
    /// # Default Implementation
    /// Does nothing (returns empty vec)
    fn on_shutdown(&mut self, _state: &CurrentState) -> Vec<StrategyAction> {
        Vec::new()
    }

    /// (Optional) Return a human-readable name for this strategy.
    ///
    /// Used for logging and TUI display.
    ///
    /// # Default Implementation
    /// Returns the type name (via std::any::type_name)
    fn name(&self) -> &str {
        std::any::type_name::<Self>()
    }

    /// (Optional) Return strategy-specific metrics for TUI display.
    ///
    /// This method exposes internal strategy state (volatility, learning models, etc.)
    /// to the bot runner for dashboard visualization.
    ///
    /// # Returns
    /// A snapshot of strategy metrics (volatility, particle filter stats, online learning, etc.)
    ///
    /// # Default Implementation
    /// Returns empty/zero metrics (strategies without rich internals can skip this)
    fn get_tui_metrics(&self) -> StrategyTuiMetrics {
        StrategyTuiMetrics::default()
    }

    /// (Optional) Return the maximum absolute position size for this strategy.
    ///
    /// This is used by the bot runner to set position limits and risk controls.
    ///
    /// # Returns
    /// Maximum absolute position size (e.g., 50.0 for ±50 contracts)
    ///
    /// # Default Implementation
    /// Returns 0.0 (no limit)
    fn get_max_position_size(&self) -> f64 {
        0.0
    }
}
