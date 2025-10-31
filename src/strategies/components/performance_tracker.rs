// ============================================================================
// Performance Tracker - Multi-Objective Metrics for Auto-Tuning
// ============================================================================
//
// This component tracks all performance metrics needed for multi-objective
// optimization of strategy parameters:
//
// 1. **Profitability**: PnL, Sharpe ratio, win rate
// 2. **Risk Control**: Drawdown, inventory volatility, margin usage
// 3. **Operational Efficiency**: Fill rates, churn, spread capture
// 4. **Model Quality**: Volatility prediction error, AS prediction error
//
// # Design
//
// The tracker maintains rolling windows of observations to compute metrics
// over recent time periods (e.g., last 500 ticks). This allows the optimizer
// to respond to recent performance changes without being affected by ancient history.
//
// # Usage
//
// ```rust
// let mut tracker = PerformanceTracker::new(PerformanceConfig::default());
//
// // On every trade
// tracker.on_trade(&trade_info, pnl);
//
// // On every quote placement
// tracker.on_quote(bid_price, ask_price, bid_size, ask_size);
//
// // On every fill
// tracker.on_fill(is_bid, fill_price, fill_size);
//
// // On every cancel
// tracker.on_cancel();
//
// // Compute metrics for optimization
// let metrics = tracker.compute_metrics();
// let objective_score = tracker.compute_multi_objective_score(&weights);
// ```

use std::collections::VecDeque;
use serde::{Deserialize, Serialize};
use log::debug;

const EPSILON: f64 = 1e-10;

/// Configuration for performance tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConfig {
    /// Rolling window size for metrics (in observations)
    pub window_size: usize,

    /// Minimum observations before computing metrics
    pub min_observations: usize,
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            window_size: 500,
            min_observations: 100,
        }
    }
}

/// Aggregated performance metrics
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    // Profitability metrics
    pub total_pnl: f64,
    pub sharpe_ratio: f64,
    pub win_rate: f64,
    pub pnl_per_trade: f64,
    pub profit_factor: f64,

    // Risk metrics
    pub max_drawdown: f64,
    pub inventory_volatility: f64,
    pub avg_margin_usage: f64,
    pub max_margin_usage: f64,

    // Efficiency metrics
    pub fill_rate: f64,
    pub churn_rate: f64,
    pub avg_spread_bps: f64,
    pub spread_capture_ratio: f64,

    // Model quality metrics
    pub volatility_prediction_error: f64,
    pub adverse_selection_error: f64,

    // Count metrics
    pub num_trades: usize,
    pub num_fills: usize,
    pub num_quotes: usize,
    pub num_cancels: usize,
}

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self {
            total_pnl: 0.0,
            sharpe_ratio: 0.0,
            win_rate: 0.5,
            pnl_per_trade: 0.0,
            profit_factor: 1.0,
            max_drawdown: 0.0,
            inventory_volatility: 0.0,
            avg_margin_usage: 0.0,
            max_margin_usage: 0.0,
            fill_rate: 0.5,
            churn_rate: 0.0,
            avg_spread_bps: 10.0,
            spread_capture_ratio: 0.5,
            volatility_prediction_error: 1.0,
            adverse_selection_error: 1.0,
            num_trades: 0,
            num_fills: 0,
            num_quotes: 0,
            num_cancels: 0,
        }
    }
}

/// Trade record for PnL tracking
#[derive(Debug, Clone)]
struct TradeRecord {
    pnl: f64,
    #[allow(dead_code)]
    timestamp_ms: u64,
}

/// Quote record for efficiency tracking
#[derive(Debug, Clone)]
struct QuoteRecord {
    bid_price: f64,
    ask_price: f64,
    #[allow(dead_code)]
    bid_size: f64,
    #[allow(dead_code)]
    ask_size: f64,
    #[allow(dead_code)]
    timestamp_ms: u64,
}

/// Fill record for fill rate tracking
#[derive(Debug, Clone)]
struct FillRecord {
    #[allow(dead_code)]
    is_bid: bool,
    #[allow(dead_code)]
    price: f64,
    #[allow(dead_code)]
    size: f64,
    #[allow(dead_code)]
    timestamp_ms: u64,
}

/// Inventory snapshot for volatility calculation
#[derive(Debug, Clone)]
struct InventorySnapshot {
    position: f64,
    #[allow(dead_code)]
    timestamp_ms: u64,
}

/// Prediction error record
#[derive(Debug, Clone)]
struct PredictionError {
    predicted: f64,
    realized: f64,
    #[allow(dead_code)]
    timestamp_ms: u64,
}

/// Main performance tracker
pub struct PerformanceTracker {
    config: PerformanceConfig,

    // PnL tracking
    trades: VecDeque<TradeRecord>,
    cumulative_pnl: f64,
    peak_pnl: f64,

    // Quote/fill tracking
    quotes: VecDeque<QuoteRecord>,
    fills: VecDeque<FillRecord>,
    cancel_count: usize,

    // Inventory tracking
    inventory_snapshots: VecDeque<InventorySnapshot>,

    // Margin tracking
    margin_usage_samples: VecDeque<f64>,

    // Model quality tracking
    vol_prediction_errors: VecDeque<PredictionError>,
    as_prediction_errors: VecDeque<PredictionError>,

    // Cached metrics (updated periodically)
    cached_metrics: Option<PerformanceMetrics>,
    last_metrics_update_tick: usize,
    current_tick: usize,
}

impl PerformanceTracker {
    pub fn new(config: PerformanceConfig) -> Self {
        Self {
            config,
            trades: VecDeque::new(),
            cumulative_pnl: 0.0,
            peak_pnl: 0.0,
            quotes: VecDeque::new(),
            fills: VecDeque::new(),
            cancel_count: 0,
            inventory_snapshots: VecDeque::new(),
            margin_usage_samples: VecDeque::new(),
            vol_prediction_errors: VecDeque::new(),
            as_prediction_errors: VecDeque::new(),
            cached_metrics: None,
            last_metrics_update_tick: 0,
            current_tick: 0,
        }
    }

    /// Record a trade
    pub fn on_trade(&mut self, pnl: f64) {
        let now_ms = chrono::Utc::now().timestamp_millis() as u64;

        self.cumulative_pnl += pnl;
        if self.cumulative_pnl > self.peak_pnl {
            self.peak_pnl = self.cumulative_pnl;
        }

        self.trades.push_back(TradeRecord {
            pnl,
            timestamp_ms: now_ms,
        });

        // Maintain window size
        while self.trades.len() > self.config.window_size {
            self.trades.pop_front();
        }
    }

    /// Record a quote placement
    pub fn on_quote(&mut self, bid_price: f64, ask_price: f64, bid_size: f64, ask_size: f64) {
        let now_ms = chrono::Utc::now().timestamp_millis() as u64;

        self.quotes.push_back(QuoteRecord {
            bid_price,
            ask_price,
            bid_size,
            ask_size,
            timestamp_ms: now_ms,
        });

        while self.quotes.len() > self.config.window_size {
            self.quotes.pop_front();
        }
    }

    /// Record a fill
    pub fn on_fill(&mut self, is_bid: bool, price: f64, size: f64) {
        let now_ms = chrono::Utc::now().timestamp_millis() as u64;

        self.fills.push_back(FillRecord {
            is_bid,
            price,
            size,
            timestamp_ms: now_ms,
        });

        while self.fills.len() > self.config.window_size {
            self.fills.pop_front();
        }
    }

    /// Record a cancel
    pub fn on_cancel(&mut self) {
        self.cancel_count += 1;
    }

    /// Record inventory snapshot
    pub fn on_inventory_update(&mut self, position: f64) {
        let now_ms = chrono::Utc::now().timestamp_millis() as u64;

        self.inventory_snapshots.push_back(InventorySnapshot {
            position,
            timestamp_ms: now_ms,
        });

        while self.inventory_snapshots.len() > self.config.window_size {
            self.inventory_snapshots.pop_front();
        }
    }

    /// Record margin usage
    pub fn on_margin_update(&mut self, margin_used: f64, margin_available: f64) {
        if margin_available > EPSILON {
            let usage_ratio = margin_used / margin_available;
            self.margin_usage_samples.push_back(usage_ratio);

            while self.margin_usage_samples.len() > self.config.window_size {
                self.margin_usage_samples.pop_front();
            }
        }
    }

    /// Record volatility prediction error
    pub fn on_volatility_prediction(&mut self, predicted: f64, realized: f64) {
        let now_ms = chrono::Utc::now().timestamp_millis() as u64;

        self.vol_prediction_errors.push_back(PredictionError {
            predicted,
            realized,
            timestamp_ms: now_ms,
        });

        while self.vol_prediction_errors.len() > self.config.window_size {
            self.vol_prediction_errors.pop_front();
        }
    }

    /// Record adverse selection prediction error
    pub fn on_adverse_selection_prediction(&mut self, predicted: f64, realized: f64) {
        let now_ms = chrono::Utc::now().timestamp_millis() as u64;

        self.as_prediction_errors.push_back(PredictionError {
            predicted,
            realized,
            timestamp_ms: now_ms,
        });

        while self.as_prediction_errors.len() > self.config.window_size {
            self.as_prediction_errors.pop_front();
        }
    }

    /// Increment tick counter (call every tick)
    pub fn tick(&mut self) {
        self.current_tick += 1;
    }

    /// Compute all performance metrics
    pub fn compute_metrics(&mut self) -> PerformanceMetrics {
        // Only recompute if enough time has passed (expensive operation)
        if self.current_tick - self.last_metrics_update_tick < 10 {
            if let Some(ref cached) = self.cached_metrics {
                return cached.clone();
            }
        }

        let metrics = self.compute_metrics_internal();
        self.cached_metrics = Some(metrics.clone());
        self.last_metrics_update_tick = self.current_tick;
        metrics
    }

    fn compute_metrics_internal(&self) -> PerformanceMetrics {
        let mut metrics = PerformanceMetrics::default();

        // Profitability metrics
        if !self.trades.is_empty() {
            metrics.num_trades = self.trades.len();
            metrics.total_pnl = self.trades.iter().map(|t| t.pnl).sum();
            metrics.pnl_per_trade = metrics.total_pnl / metrics.num_trades as f64;

            let winning_trades = self.trades.iter().filter(|t| t.pnl > 0.0).count();
            metrics.win_rate = winning_trades as f64 / metrics.num_trades as f64;

            let gross_profit: f64 = self.trades.iter().filter(|t| t.pnl > 0.0).map(|t| t.pnl).sum();
            let gross_loss: f64 = self.trades.iter().filter(|t| t.pnl < 0.0).map(|t| t.pnl.abs()).sum();
            metrics.profit_factor = if gross_loss > EPSILON {
                gross_profit / gross_loss
            } else {
                if gross_profit > EPSILON { 100.0 } else { 1.0 }
            };

            // Sharpe ratio (annualized)
            if self.trades.len() >= 2 {
                let returns: Vec<f64> = self.trades.iter().map(|t| t.pnl).collect();
                let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
                let variance = returns.iter()
                    .map(|r| (r - mean_return).powi(2))
                    .sum::<f64>() / (returns.len() - 1) as f64;
                let std_dev = variance.sqrt();

                metrics.sharpe_ratio = if std_dev > EPSILON {
                    mean_return / std_dev * (252.0_f64.sqrt()) // Annualized
                } else {
                    0.0
                };
            }
        }

        // Risk metrics
        metrics.max_drawdown = self.peak_pnl - self.cumulative_pnl;

        if self.inventory_snapshots.len() >= 2 {
            let positions: Vec<f64> = self.inventory_snapshots.iter().map(|s| s.position).collect();
            let mean_pos = positions.iter().sum::<f64>() / positions.len() as f64;
            let variance = positions.iter()
                .map(|p| (p - mean_pos).powi(2))
                .sum::<f64>() / (positions.len() - 1) as f64;
            metrics.inventory_volatility = variance.sqrt();
        }

        if !self.margin_usage_samples.is_empty() {
            metrics.avg_margin_usage = self.margin_usage_samples.iter().sum::<f64>() / self.margin_usage_samples.len() as f64;
            metrics.max_margin_usage = self.margin_usage_samples.iter().cloned().fold(0.0, f64::max);
        }

        // Efficiency metrics
        metrics.num_quotes = self.quotes.len();
        metrics.num_fills = self.fills.len();
        metrics.num_cancels = self.cancel_count;

        if metrics.num_quotes > 0 {
            metrics.fill_rate = metrics.num_fills as f64 / metrics.num_quotes as f64;
            metrics.churn_rate = metrics.num_cancels as f64 / metrics.num_quotes as f64;
        }

        if !self.quotes.is_empty() {
            let total_spread: f64 = self.quotes.iter()
                .map(|q| (q.ask_price - q.bid_price) / q.bid_price * 10000.0) // In bps
                .sum();
            metrics.avg_spread_bps = total_spread / self.quotes.len() as f64;
        }

        // Spread capture ratio (realized spread / quoted spread)
        // This requires matching fills to quotes - simplified here
        if !self.fills.is_empty() && metrics.avg_spread_bps > EPSILON {
            // Estimate: assume fills capture ~50% of spread on average
            metrics.spread_capture_ratio = 0.5; // Placeholder - needs proper calculation
        }

        // Model quality metrics
        if !self.vol_prediction_errors.is_empty() {
            let mse: f64 = self.vol_prediction_errors.iter()
                .map(|e| (e.predicted - e.realized).powi(2))
                .sum::<f64>() / self.vol_prediction_errors.len() as f64;
            metrics.volatility_prediction_error = mse.sqrt(); // RMSE
        }

        if !self.as_prediction_errors.is_empty() {
            let mse: f64 = self.as_prediction_errors.iter()
                .map(|e| (e.predicted - e.realized).powi(2))
                .sum::<f64>() / self.as_prediction_errors.len() as f64;
            metrics.adverse_selection_error = mse.sqrt(); // RMSE
        }

        metrics
    }

    /// Compute multi-objective score
    pub fn compute_multi_objective_score(&mut self, weights: &MultiObjectiveWeights) -> f64 {
        let metrics = self.compute_metrics();

        // Profitability score [0, 1]
        let profit_score = self.compute_profitability_score(&metrics, &weights.profitability);

        // Risk score [0, 1] - higher is better (lower risk)
        let risk_score = self.compute_risk_score(&metrics, &weights.risk);

        // Efficiency score [0, 1]
        let efficiency_score = self.compute_efficiency_score(&metrics, &weights.efficiency);

        // Model quality score [0, 1] - higher is better (lower error)
        let model_score = self.compute_model_quality_score(&metrics);

        debug!(
            "[PERF TRACKER] Multi-objective: profit={:.3}, risk={:.3}, eff={:.3}, model={:.3}",
            profit_score, risk_score, efficiency_score, model_score
        );

        // Weighted combination
        weights.profitability_weight * profit_score +
        weights.risk_weight * risk_score +
        weights.efficiency_weight * efficiency_score +
        weights.model_quality_weight * model_score
    }

    fn compute_profitability_score(&self, metrics: &PerformanceMetrics, weights: &ProfitabilityWeights) -> f64 {
        // Normalize each component to [0, 1]
        let sharpe_normalized = (metrics.sharpe_ratio / 3.0).clamp(0.0, 1.0); // 3.0 is excellent
        let win_rate_normalized = metrics.win_rate; // Already [0, 1]
        let pnl_normalized = (metrics.pnl_per_trade / 1.0).clamp(0.0, 1.0); // $1 per trade is good

        weights.sharpe_weight * sharpe_normalized +
        weights.win_rate_weight * win_rate_normalized +
        weights.pnl_per_trade_weight * pnl_normalized
    }

    fn compute_risk_score(&self, metrics: &PerformanceMetrics, weights: &RiskWeights) -> f64 {
        // Lower is better, so invert
        let drawdown_score = (1.0 - (metrics.max_drawdown / 100.0)).clamp(0.0, 1.0); // $100 DD = 0 score
        let inv_vol_score = (1.0 - (metrics.inventory_volatility / 10.0)).clamp(0.0, 1.0); // 10 units vol = 0 score
        let margin_score = (1.0 - metrics.avg_margin_usage).clamp(0.0, 1.0); // Full margin usage = 0 score

        weights.drawdown_weight * drawdown_score +
        weights.inventory_vol_weight * inv_vol_score +
        weights.margin_weight * margin_score
    }

    fn compute_efficiency_score(&self, metrics: &PerformanceMetrics, weights: &EfficiencyWeights) -> f64 {
        let fill_rate_score = metrics.fill_rate; // Already [0, 1]
        let churn_score = (1.0 - metrics.churn_rate).clamp(0.0, 1.0); // Lower churn is better
        let spread_score = metrics.spread_capture_ratio; // Already [0, 1]

        weights.fill_rate_weight * fill_rate_score +
        weights.churn_weight * churn_score +
        weights.spread_capture_weight * spread_score
    }

    fn compute_model_quality_score(&self, metrics: &PerformanceMetrics) -> f64 {
        // Lower error is better, invert
        let vol_error_score = (1.0 - (metrics.volatility_prediction_error / 10.0)).clamp(0.0, 1.0); // 10 bps error = 0 score
        let as_error_score = (1.0 - (metrics.adverse_selection_error / 5.0)).clamp(0.0, 1.0); // 5 bps error = 0 score

        0.5 * vol_error_score + 0.5 * as_error_score
    }

    /// Get current metrics without recomputation
    pub fn get_cached_metrics(&self) -> Option<&PerformanceMetrics> {
        self.cached_metrics.as_ref()
    }

    /// Check if we have enough data for meaningful metrics
    pub fn has_sufficient_data(&self) -> bool {
        self.trades.len() >= self.config.min_observations
    }
}

/// Weights for multi-objective optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiObjectiveWeights {
    pub profitability_weight: f64,
    pub risk_weight: f64,
    pub efficiency_weight: f64,
    pub model_quality_weight: f64,

    pub profitability: ProfitabilityWeights,
    pub risk: RiskWeights,
    pub efficiency: EfficiencyWeights,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfitabilityWeights {
    pub sharpe_weight: f64,
    pub win_rate_weight: f64,
    pub pnl_per_trade_weight: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskWeights {
    pub drawdown_weight: f64,
    pub inventory_vol_weight: f64,
    pub margin_weight: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EfficiencyWeights {
    pub fill_rate_weight: f64,
    pub churn_weight: f64,
    pub spread_capture_weight: f64,
}

impl Default for MultiObjectiveWeights {
    fn default() -> Self {
        Self {
            profitability_weight: 0.4,
            risk_weight: 0.3,
            efficiency_weight: 0.2,
            model_quality_weight: 0.1,

            profitability: ProfitabilityWeights {
                sharpe_weight: 0.5,
                win_rate_weight: 0.3,
                pnl_per_trade_weight: 0.2,
            },

            risk: RiskWeights {
                drawdown_weight: 0.4,
                inventory_vol_weight: 0.3,
                margin_weight: 0.3,
            },

            efficiency: EfficiencyWeights {
                fill_rate_weight: 0.4,
                churn_weight: 0.3,
                spread_capture_weight: 0.3,
            },
        }
    }
}
