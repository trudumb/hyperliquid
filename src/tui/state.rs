use std::collections::VecDeque;

/// Snapshot of market maker state for TUI rendering
/// Updated via watch channel and consumed by dashboard renderer
#[derive(Debug, Clone)]
pub struct DashboardState {
    // ===== Position & PnL =====
    pub cur_position: f64,
    pub avg_entry_price: f64,
    pub unrealized_pnl: f64,
    pub realized_pnl: f64,       // Session realized PnL
    pub total_fees: f64,          // Session total fees paid
    pub total_session_pnl: f64,   // realized + unrealized - fees
    pub account_equity: f64,
    pub session_start_equity: f64, // Account equity at session start
    pub sharpe_ratio: f64,

    // ===== Market Data =====
    pub l2_mid_price: f64,
    pub all_mids_price: f64,
    pub market_spread_bps: f64,
    pub lob_imbalance: f64,

    // ===== State Vector Metrics =====
    pub volatility_ema_bps: f64,
    pub adverse_selection_estimate: f64,
    pub trade_flow_ema: f64,

    // ===== Particle Filter Stats =====
    pub pf_ess: f64,
    pub pf_max_particles: usize,
    pub pf_vol_5th: f64,
    pub pf_vol_95th: f64,
    pub pf_volatility_bps: f64,

    // ===== Adverse Selection Model =====
    pub online_model_mae: f64,
    pub online_model_updates: usize,
    pub online_model_lr: f64,
    pub online_model_enabled: bool,

    // ===== Adam Optimizer State =====
    pub adam_gradient_samples: usize,
    pub adam_avg_loss: f64,
    pub adam_last_update_secs: f64,

    // ===== Open Orders =====
    pub bid_levels: Vec<OrderLevel>,
    pub ask_levels: Vec<OrderLevel>,

    // ===== Recent Fills (ring buffer) =====
    pub recent_fills: VecDeque<FillEvent>,

    // ===== System Info =====
    pub uptime_secs: f64,
    pub total_messages: u64,
}

/// Represents a single order level on the book
#[derive(Debug, Clone)]
pub struct OrderLevel {
    pub side: String,  // "BID" or "ASK"
    pub level: usize,  // L1, L2, L3, etc.
    pub price: f64,
    pub size: f64,
    pub oid: u64,
}

/// Represents a fill event for the recent fills panel
#[derive(Debug, Clone)]
pub struct FillEvent {
    pub timestamp: String,  // Formatted time: "15:23:45"
    pub side: String,       // "BOUGHT" or "SOLD"
    pub size: f64,
    pub price: f64,
    pub oid: u64,
}

impl Default for DashboardState {
    fn default() -> Self {
        Self {
            cur_position: 0.0,
            avg_entry_price: 0.0,
            unrealized_pnl: 0.0,
            realized_pnl: 0.0,
            total_fees: 0.0,
            total_session_pnl: 0.0,
            account_equity: 0.0,
            session_start_equity: 0.0,
            sharpe_ratio: 0.0,

            l2_mid_price: 0.0,
            all_mids_price: 0.0,
            market_spread_bps: 0.0,
            lob_imbalance: 0.0,

            volatility_ema_bps: 0.0,
            adverse_selection_estimate: 0.0,
            trade_flow_ema: 0.0,

            pf_ess: 0.0,
            pf_max_particles: 0,
            pf_vol_5th: 0.0,
            pf_vol_95th: 0.0,
            pf_volatility_bps: 0.0,

            online_model_mae: 0.0,
            online_model_updates: 0,
            online_model_lr: 0.0,
            online_model_enabled: false,

            adam_gradient_samples: 0,
            adam_avg_loss: 0.0,
            adam_last_update_secs: 0.0,

            bid_levels: Vec::new(),
            ask_levels: Vec::new(),

            recent_fills: VecDeque::with_capacity(20),

            uptime_secs: 0.0,
            total_messages: 0,
        }
    }
}

impl DashboardState {
    /// Add a new fill event to the recent fills buffer (max 20 items)
    pub fn add_fill(&mut self, fill: FillEvent) {
        if self.recent_fills.len() >= 20 {
            self.recent_fills.pop_front();
        }
        self.recent_fills.push_back(fill);
    }
}
