// src/ipc.rs
//! Defines the data structures for communication between the State Manager and Strategy Runners.
//!
//! This module provides a multi-asset architecture where:
//! - AssetState: Tracks per-asset position, orders, and PnL
//! - GlobalAccountState: Tracks account-level equity and margin
//! - State updates are asset-specific to avoid cross-contamination

use crate::{RestingOrder, StrategyAction};
use tokio::sync::oneshot;

/// Per-asset state tracked by the State Manager.
/// Each asset has its own independent position, orders, and PnL tracking.
#[derive(Debug, Clone, Default)]
pub struct AssetState {
    /// The asset symbol (e.g., "BTC", "ETH", "HYPE")
    pub asset: String,
    /// Current position size (positive = long, negative = short)
    pub position: f64,
    /// Average entry price for current position
    pub avg_entry_price: f64,
    /// Realized PnL for this asset
    pub realized_pnl: f64,
    /// Unrealized PnL for this asset (updated by runners with market data)
    pub unrealized_pnl: f64,
    /// Total fees paid for this asset
    pub total_fees: f64,
    /// Cost basis of current position
    pub cost_basis: f64,
    /// Open bid orders for this asset
    pub open_bids: Vec<RestingOrder>,
    /// Open ask orders for this asset
    pub open_asks: Vec<RestingOrder>,
    /// Last update timestamp (seconds since epoch)
    pub timestamp: f64,
}

/// Global account state broadcast by the State Manager to all runners.
/// This contains account-level information that is shared across all assets.
#[derive(Debug, Clone)]
pub struct GlobalAccountState {
    /// Total account equity (in USD)
    pub account_equity: f64,
    /// Total margin used across all positions (in USD)
    pub margin_used: f64,
    /// Timestamp of this update (milliseconds since epoch)
    pub timestamp_ms: u64,
}

/// The authoritative state update broadcast by the State Manager for a specific asset.
/// Runners filter these by asset to only process updates relevant to them.
#[derive(Debug, Clone)]
pub struct AuthoritativeStateUpdate {
    /// The asset this update pertains to
    pub asset: String,
    /// Position data for this specific asset
    pub position: f64,
    /// Average entry price for this asset
    pub avg_entry_price: f64,
    /// Realized PnL for this asset
    pub realized_pnl: f64,
    /// Unrealized PnL for this asset
    pub unrealized_pnl: f64,
    /// Global account equity (shared across all assets)
    pub account_equity: f64,
    /// Global margin used (shared across all assets)
    pub margin_used: f64,
    /// Timestamp of this update
    pub timestamp_ms: u64,
    /// Open orders for this specific asset only
    pub open_orders: Vec<RestingOrder>,
}

impl AuthoritativeStateUpdate {
    /// Creates a state update from an AssetState and GlobalAccountState
    pub fn from_asset_and_global(
        asset_state: &AssetState,
        global_state: &GlobalAccountState,
    ) -> Self {
        let open_orders = asset_state
            .open_bids
            .iter()
            .cloned()
            .chain(asset_state.open_asks.iter().cloned())
            .collect();

        Self {
            asset: asset_state.asset.clone(),
            position: asset_state.position,
            avg_entry_price: asset_state.avg_entry_price,
            realized_pnl: asset_state.realized_pnl,
            unrealized_pnl: asset_state.unrealized_pnl,
            account_equity: global_state.account_equity,
            margin_used: global_state.margin_used,
            timestamp_ms: global_state.timestamp_ms,
            open_orders,
        }
    }
}

/// A request sent from a Strategy Runner to the State Manager to execute actions.
#[derive(Debug)]
pub struct ExecuteActionsRequest {
    /// The asset this request pertains to (for routing and validation)
    pub asset: String,
    /// The list of actions to execute
    pub actions: Vec<StrategyAction>,
    /// A channel to send the confirmation (success/failure) back to the caller
    pub resp: oneshot::Sender<ExecuteActionsResponse>,
}

/// The response sent from the State Manager back to the Strategy Runner.
#[derive(Debug, Clone)]
pub struct ExecuteActionsResponse {
    pub success: bool,
    pub message: String,
}
