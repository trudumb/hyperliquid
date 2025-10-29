// src/ipc.rs
//! Defines the data structures for communication between the State Manager and Strategy Runners.

use crate::{RestingOrder, StrategyAction};
use tokio::sync::oneshot;

/// The authoritative state broadcast by the State Manager to all runners.
/// This contains the minimum information runners need to make decisions.
#[derive(Debug, Clone)]
pub struct AuthoritativeStateUpdate {
    pub position: f64,
    pub avg_entry_price: f64,
    pub realized_pnl: f64,
    pub unrealized_pnl: f64,
    pub account_equity: f64,
    pub margin_used: f64,
    pub timestamp_ms: u64,
    pub open_orders: Vec<RestingOrder>,
}

/// A request sent from a Strategy Runner to the State Manager to execute actions.
#[derive(Debug)]
pub struct ExecuteActionsRequest {
    /// The asset this request pertains to (for logging/sharding).
    pub asset: String,
    /// The list of actions to execute.
    pub actions: Vec<StrategyAction>,
    /// A channel to send the confirmation (success/failure) back to the caller.
    pub resp: oneshot::Sender<ExecuteActionsResponse>,
}

/// The response sent from the State Manager back to the Strategy Runner.
#[derive(Debug, Clone)]
pub struct ExecuteActionsResponse {
    pub success: bool,
    pub message: String,
}
