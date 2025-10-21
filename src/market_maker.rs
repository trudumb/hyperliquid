use alloy::{primitives::Address, signers::local::PrivateKeySigner};
use log::{error, info, warn};
use tokio::sync::mpsc::unbounded_channel;
use std::collections::VecDeque;
use std::time::{SystemTime, UNIX_EPOCH};
use serde_json::{json, Value};

//RUST_LOG=info cargo run --bin market_maker

use crate::{
    bps_diff, truncate_float, BaseUrl, ClientCancelRequest, ClientLimit, ClientOrder,
    ClientOrderRequest, ExchangeClient, ExchangeDataStatus, ExchangeResponseStatus, InfoClient,
    Message, Subscription, UserData, EPSILON, L2Book,
};

// Structured logging utility for better analytics
pub(crate) struct StructuredLogger;

impl StructuredLogger {
    pub(crate) fn log_trade(trade: &Trade, position: f64, unrealized_pnl: f64) {
        info!(
            "TRADE_EVENT: asset={}, side={}, size={:.4}, price={:.4}, pnl={:.2}, position={:.4}, unrealized_pnl={:.2}",
            trade.asset, trade.side, trade.size, trade.price, trade.pnl, position, unrealized_pnl
        );
    }
    
    pub(crate) fn log_order_event(action: &str, asset: &str, size: f64, price: f64, is_buy: bool, is_simulation: bool) {
        info!(
            "ORDER_EVENT: action={}, asset={}, side={}, size={:.4}, price={:.4}, simulation={}",
            action, asset, if is_buy { "BUY" } else { "SELL" }, size, price, is_simulation
        );
    }
    
    pub(crate) fn log_risk_event(event_type: &str, data: Value) {
        info!("RISK_EVENT: type={}, data={}", event_type, data);
    }
    
    pub(crate) fn log_spread_adjustment(reason: &str, original_spread: u16, new_spread: f64, multiplier: f64) {
        info!(
            "SPREAD_ADJUSTMENT: reason='{}', original={}bps, new={:.1}bps, multiplier={:.2}x",
            reason, original_spread, new_spread, multiplier
        );
    }
    
    pub(crate) fn log_performance(metrics: &PerformanceMetrics, _risk_mgr: &RiskManager) {
        info!(
            "PERFORMANCE: total_pnl={:.2}, realized_pnl={:.2}, unrealized_pnl={:.2}, roi={:.1}%, sharpe={:.2}, max_dd={:.1}%, win_rate={:.1}%, trades={}, volume={:.2}",
            metrics.total_pnl, metrics.realized_pnl, metrics.unrealized_pnl, 
            metrics.roi * 100.0, metrics.sharpe_ratio, metrics.max_drawdown * 100.0,
            metrics.win_rate * 100.0, metrics.total_trades, metrics.total_volume
        );
    }
    
    pub(crate) fn log_market_data(mid_price: f64, volatility: f64, imbalance: f64, adverse_score: f64) {
        info!(
            "MARKET_DATA: mid_price={:.4}, volatility={:.1}%, imbalance={:.3}, adverse_score={:.3}",
            mid_price, volatility * 100.0, imbalance, adverse_score
        );
    }
    
    pub(crate) fn log_inventory_management(position: f64, max_position: f64, skew_bps: f64, action: &str) {
        info!(
            "INVENTORY: position={:.4}, max={:.4}, ratio={:.1}%, skew={:.1}bps, action={}",
            position, max_position, (position / max_position) * 100.0, skew_bps, action
        );
    }
}

#[derive(Debug, Clone)]
pub struct Trade {
    pub timestamp: u64,
    pub asset: String,
    pub side: String, // "B" for buy, "S" for sell
    pub size: f64,
    pub price: f64,
    pub pnl: f64,
    pub position_after: f64,
}

#[derive(Debug, Clone)]
pub struct PnLTracker {
    pub trades: Vec<Trade>,
    pub realized_pnl: f64,
    pub unrealized_pnl: f64,
    pub sharpe_window: VecDeque<f64>,
    pub daily_returns: VecDeque<f64>, // Track daily returns for proper Sharpe ratio calculation
    pub last_day_end_value: f64, // Portfolio value at end of last day
    pub current_day_pnl: f64, // Running PnL for current day
    pub last_day: u64, // Track current day for daily aggregation
    pub initial_capital: f64,
    pub current_position: f64,
    pub average_entry_price: f64,
    pub total_volume: f64,
    pub winning_trades: usize,
    pub losing_trades: usize,
}

impl PnLTracker {
    pub fn new(initial_capital: f64) -> Self {
        let current_day = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs() / 86400; // Convert to days
        
        Self {
            trades: Vec::new(),
            realized_pnl: 0.0,
            unrealized_pnl: 0.0,
            sharpe_window: VecDeque::with_capacity(365), // One year of daily returns (deprecated)
            daily_returns: VecDeque::with_capacity(365), // Track daily returns for Sharpe ratio
            last_day_end_value: initial_capital,
            current_day_pnl: 0.0,
            last_day: current_day,
            initial_capital,
            current_position: 0.0,
            average_entry_price: 0.0,
            total_volume: 0.0,
            winning_trades: 0,
            losing_trades: 0,
        }
    }

    pub fn add_trade(&mut self, trade: Trade) {
        let previous_position = self.current_position;
        self.current_position = trade.position_after;
        self.total_volume += trade.size;
        
        // Calculate realized PnL for this trade
        let mut realized_pnl_delta = 0.0;
        
        if previous_position != 0.0 {
            // If we're reducing or closing a position, calculate realized PnL
            let position_reduction = if (previous_position > 0.0 && trade.side == "S") ||
                                       (previous_position < 0.0 && trade.side == "B") {
                trade.size.min(previous_position.abs())
            } else {
                0.0
            };
            
            if position_reduction > 0.0 {
                if previous_position > 0.0 {
                    // Closing long position
                    realized_pnl_delta = position_reduction * (trade.price - self.average_entry_price);
                } else {
                    // Closing short position
                    realized_pnl_delta = position_reduction * (self.average_entry_price - trade.price);
                }
            }
        }
        
        // Update average entry price if increasing position
        if (previous_position >= 0.0 && trade.side == "B") ||
           (previous_position <= 0.0 && trade.side == "S") {
            let new_size = if trade.side == "B" { trade.size } else { -trade.size };
            let total_position = previous_position + new_size;
            
            if total_position.abs() > 1e-10 {
                // Handle edge case where previous position is 0.0 (first trade)
                if previous_position.abs() < 1e-10 {
                    self.average_entry_price = trade.price;
                } else {
                    self.average_entry_price = ((previous_position * self.average_entry_price) + 
                                              (new_size * trade.price)) / total_position;
                }
            } else {
                // Position is closing to zero, reset average entry price
                self.average_entry_price = 0.0;
            }
        }
        
        self.realized_pnl += realized_pnl_delta;
        
        // Track winning/losing trades
        if realized_pnl_delta > 0.0 {
            self.winning_trades += 1;
        } else if realized_pnl_delta < 0.0 {
            self.losing_trades += 1;
        }
        
        // Update current day PnL
        self.current_day_pnl += realized_pnl_delta;
        
        // Store the trade with calculated PnL
        let mut trade_record = trade;
        trade_record.pnl = realized_pnl_delta;
        self.trades.push(trade_record);
        
        // Check if we've moved to a new day and finalize daily return
        let current_day = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs() / 86400;
        
        if current_day != self.last_day {
            // Calculate daily return as percentage change in portfolio value
            let end_of_day_value = self.last_day_end_value + self.current_day_pnl + self.unrealized_pnl;
            let daily_return = if self.last_day_end_value > 1e-10 {
                self.current_day_pnl / self.last_day_end_value
            } else {
                0.0
            };
            
            self.daily_returns.push_back(daily_return);
            if self.daily_returns.len() > 252 {
                self.daily_returns.pop_front();
            }
            
            // Reset for new day
            self.last_day_end_value = end_of_day_value;
            self.current_day_pnl = 0.0;
            self.last_day = current_day;
            
            // Deprecated sharpe_window for backward compatibility
            self.sharpe_window.push_back(daily_return);
            if self.sharpe_window.len() > 252 {
                self.sharpe_window.pop_front();
            }
        }
    }
    
    pub fn update_unrealized(&mut self, position: f64, entry_price: f64, current_price: f64) {
        self.unrealized_pnl = if position > 0.0 {
            position * (current_price - entry_price)
        } else if position < 0.0 {
            position * (current_price - entry_price) // This will be negative for short positions with positive price moves
        } else {
            0.0
        };
        
        // Check if we need to finalize the day's return for Sharpe calculation
        self.check_and_finalize_daily_return();
    }
    
    /// Check if a new day has started and finalize the previous day's return
    pub fn check_and_finalize_daily_return(&mut self) {
        let current_day = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs() / 86400;
        
        if current_day != self.last_day && self.last_day > 0 {
            // Calculate daily return including unrealized PnL at end of day
            let end_of_day_value = self.last_day_end_value + self.current_day_pnl + self.unrealized_pnl;
            let daily_return = if self.last_day_end_value > 1e-10 {
                (end_of_day_value - self.last_day_end_value) / self.last_day_end_value
            } else {
                0.0
            };
            
            self.daily_returns.push_back(daily_return);
            if self.daily_returns.len() > 252 {
                self.daily_returns.pop_front();
            }
            
            // Reset for new day (realized PnL becomes part of capital base)
            self.last_day_end_value += self.current_day_pnl;
            self.current_day_pnl = 0.0;
            self.last_day = current_day;
            
            // Backward compatibility
            self.sharpe_window.push_back(daily_return);
            if self.sharpe_window.len() > 252 {
                self.sharpe_window.pop_front();
            }
        }
    }
    
    pub fn sharpe_ratio(&self) -> f64 {
        // Use daily returns for proper Sharpe ratio calculation
        if self.daily_returns.is_empty() {
            return 0.0;
        }
        
        // Calculate mean and standard deviation of daily returns
        let mean: f64 = self.daily_returns.iter().sum::<f64>() / self.daily_returns.len() as f64;
        let variance: f64 = self.daily_returns
            .iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / self.daily_returns.len() as f64;
        
        let std_dev = variance.sqrt();
        
        if std_dev < 1e-10 {
            return 0.0;
        }
        
        // Annualize: assume 252 trading days per year
        // Sharpe = (mean daily return / std dev of daily returns) * sqrt(252)
        (mean / std_dev) * (252.0_f64).sqrt()
    }
    
    pub fn max_drawdown(&self) -> f64 {
        let mut peak = f64::MIN;
        let mut max_dd: f64 = 0.0;
        let mut cumulative: f64 = 0.0;
        
        for trade in &self.trades {
            cumulative += trade.pnl;
            if cumulative > peak {
                peak = cumulative;
            }
            let dd: f64 = if peak.abs() > 1e-10 {
                (peak - cumulative) / peak.abs()
            } else {
                0.0
            };
            max_dd = max_dd.max(dd);
        }
        
        max_dd
    }
    
    pub fn total_pnl(&self) -> f64 {
        self.realized_pnl + self.unrealized_pnl
    }
    
    pub fn win_rate(&self) -> f64 {
        let total_trades = self.winning_trades + self.losing_trades;
        if total_trades == 0 {
            return 0.0;
        }
        self.winning_trades as f64 / total_trades as f64
    }
    
    pub fn profit_factor(&self) -> f64 {
        let gross_profit: f64 = self.trades.iter()
            .filter(|t| t.pnl > 0.0)
            .map(|t| t.pnl)
            .sum();
        let gross_loss: f64 = self.trades.iter()
            .filter(|t| t.pnl < 0.0)
            .map(|t| t.pnl.abs())
            .sum();
        
        if gross_loss < 1e-10 {
            return if gross_profit > 0.0 { f64::INFINITY } else { 0.0 };
        }
        
        gross_profit / gross_loss
    }
    
    pub fn average_trade_pnl(&self) -> f64 {
        if self.trades.is_empty() {
            return 0.0;
        }
        self.trades.iter().map(|t| t.pnl).sum::<f64>() / self.trades.len() as f64
    }
    
    pub fn roi(&self) -> f64 {
        if self.initial_capital < 1e-10 {
            return 0.0;
        }
        self.total_pnl() / self.initial_capital
    }
    
    pub fn get_performance_metrics(&self) -> PerformanceMetrics {
        PerformanceMetrics {
            total_pnl: self.total_pnl(),
            realized_pnl: self.realized_pnl,
            unrealized_pnl: self.unrealized_pnl,
            sharpe_ratio: self.sharpe_ratio(),
            max_drawdown: self.max_drawdown(),
            win_rate: self.win_rate(),
            profit_factor: self.profit_factor(),
            average_trade_pnl: self.average_trade_pnl(),
            roi: self.roi(),
            total_trades: self.trades.len(),
            winning_trades: self.winning_trades,
            losing_trades: self.losing_trades,
            total_volume: self.total_volume,
        }
    }
}

#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    pub total_pnl: f64,
    pub realized_pnl: f64,
    pub unrealized_pnl: f64,
    pub sharpe_ratio: f64,
    pub max_drawdown: f64,
    pub win_rate: f64,
    pub profit_factor: f64,
    pub average_trade_pnl: f64,
    pub roi: f64,
    pub total_trades: usize,
    pub winning_trades: usize,
    pub losing_trades: usize,
    pub total_volume: f64,
}

#[derive(Debug, Clone)]
pub struct Fill {
    pub fill_price: f64,
    pub current_mid: f64,
    pub is_buy: bool,
    pub size: f64,
    pub timestamp: u64,
}

#[derive(Debug, Clone)]
pub struct VolatilityEstimator {
    // High-frequency price observations for realized volatility
    price_history: VecDeque<(u64, f64)>, // (timestamp, price)
    returns: VecDeque<(u64, f64)>, // (timestamp, return) for temporal weighting
    
    // Multi-timeframe EWMA estimates for responsiveness
    fast_ewma: f64,     // Fast-reacting EWMA (alpha=0.3)
    medium_ewma: f64,   // Medium EWMA (alpha=0.1) 
    slow_ewma: f64,     // Slow baseline EWMA (alpha=0.03)
    
    // Regime detection for volatility clustering
    vol_regime_high: bool, // Are we in a high volatility regime?
    regime_threshold: f64,  // Threshold for regime switching
    
    // Intraday patterns
    intraday_multipliers: [f64; 24], // Hourly volatility multipliers
    last_calibration: u64, // Last time we calibrated intraday patterns
    
    // Configuration
    lookback_minutes: u64,
    observation_frequency_secs: u64, // Expected frequency of price updates
    autocorr_adjustment: f64, // Factor to adjust for return autocorrelation
    microstructure_noise_threshold: f64, // Filter out noise from bid-ask bounce
}

impl VolatilityEstimator {
    pub fn new(lookback_minutes: u64, observation_frequency_secs: u64) -> Self {
        Self {
            price_history: VecDeque::new(),
            returns: VecDeque::new(),
            fast_ewma: 0.0,
            medium_ewma: 0.0,
            slow_ewma: 0.0,
            vol_regime_high: false,
            regime_threshold: 2.0, // 2x normal volatility triggers high regime
            intraday_multipliers: [1.0; 24], // Start with flat intraday pattern
            last_calibration: 0,
            lookback_minutes,
            observation_frequency_secs,
            autocorr_adjustment: 0.95, // Accounts for positive autocorrelation in crypto
            microstructure_noise_threshold: 0.0001, // 1 bps minimum meaningful move
        }
    }

    /// Add a new price observation with improved volatility estimation
    pub fn add_price(&mut self, price: f64) {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        
        self.clean_old_data(timestamp);
        
        // Calculate return if we have a previous price
        if let Some((last_timestamp, last_price)) = self.price_history.back() {
            if *last_price > 1e-10 {
                let time_diff = timestamp - last_timestamp;
                
                // Skip if time difference is too small (< 1 second) to avoid noise
                if time_diff > 0 {
                    let log_return = (price / last_price).ln();
                    
                    // Filter microstructure noise
                    if log_return.abs() > self.microstructure_noise_threshold {
                        // Adjust for time interval (normalize to per-second return)
                        let normalized_return = log_return / (time_diff as f64).sqrt();
                        
                        self.returns.push_back((timestamp, normalized_return));
                        self.update_ewma_estimates(normalized_return);
                        self.update_volatility_regime();
                        
                        // Periodically recalibrate intraday patterns (every 6 hours)
                        if timestamp - self.last_calibration > 21600 {
                            self.calibrate_intraday_patterns();
                            self.last_calibration = timestamp;
                        }
                    }
                }
            }
        }
        
        self.price_history.push_back((timestamp, price));
    }

    /// Clean old data beyond lookback window
    fn clean_old_data(&mut self, current_timestamp: u64) {
        let cutoff_time = current_timestamp - (self.lookback_minutes * 60);
        
        while let Some((old_time, _)) = self.price_history.front() {
            if *old_time < cutoff_time {
                self.price_history.pop_front();
            } else {
                break;
            }
        }
        
        while let Some((old_time, _)) = self.returns.front() {
            if *old_time < cutoff_time {
                self.returns.pop_front();
            } else {
                break;
            }
        }
    }

    /// Update multi-timeframe EWMA estimates
    fn update_ewma_estimates(&mut self, normalized_return: f64) {
        let squared_return = normalized_return * normalized_return;
        
        // Fast EWMA (responds quickly to volatility spikes)
        let alpha_fast = 0.3;
        if self.fast_ewma == 0.0 {
            self.fast_ewma = squared_return;
        } else {
            self.fast_ewma = alpha_fast * squared_return + (1.0 - alpha_fast) * self.fast_ewma;
        }
        
        // Medium EWMA (balanced responsiveness)
        let alpha_medium = 0.1;
        if self.medium_ewma == 0.0 {
            self.medium_ewma = squared_return;
        } else {
            self.medium_ewma = alpha_medium * squared_return + (1.0 - alpha_medium) * self.medium_ewma;
        }
        
        // Slow EWMA (stable baseline)
        let alpha_slow = 0.03;
        if self.slow_ewma == 0.0 {
            self.slow_ewma = squared_return;
        } else {
            self.slow_ewma = alpha_slow * squared_return + (1.0 - alpha_slow) * self.slow_ewma;
        }
    }

    /// Detect volatility regime changes
    fn update_volatility_regime(&mut self) {
        if self.medium_ewma > 0.0 && self.slow_ewma > 0.0 {
            let vol_ratio = self.medium_ewma / self.slow_ewma;
            
            if vol_ratio > self.regime_threshold && !self.vol_regime_high {
                self.vol_regime_high = true;
                info!("📈 Volatility regime: HIGH (ratio: {:.2})", vol_ratio);
            } else if vol_ratio < (self.regime_threshold * 0.7) && self.vol_regime_high {
                self.vol_regime_high = false;
                info!("📉 Volatility regime: NORMAL (ratio: {:.2})", vol_ratio);
            }
        }
    }

    /// Calibrate intraday volatility patterns
    fn calibrate_intraday_patterns(&mut self) {
        if self.returns.len() < 100 {
            return;
        }

        // Calculate hourly volatility averages from historical data
        let mut hourly_vols = [0.0; 24];
        let mut hourly_counts = [0; 24];
        
        for (timestamp, return_val) in &self.returns {
            let hour = ((*timestamp % 86400) / 3600) as usize;
            if hour < 24 {
                hourly_vols[hour] += return_val.abs();
                hourly_counts[hour] += 1;
            }
        }
        
        // Calculate average volatility across all hours
        let mut total_vol = 0.0;
        let mut total_count = 0;
        for i in 0..24 {
            if hourly_counts[i] > 0 {
                hourly_vols[i] /= hourly_counts[i] as f64;
                total_vol += hourly_vols[i];
                total_count += 1;
            }
        }
        
        if total_count > 12 { // Need data for at least half the hours
            let avg_vol = total_vol / total_count as f64;
            
            // Update intraday multipliers with smoothing
            for i in 0..24 {
                if hourly_counts[i] > 5 { // Need minimum observations
                    let new_multiplier = if avg_vol > 1e-10 {
                        (hourly_vols[i] / avg_vol).max(0.5).min(3.0)
                    } else {
                        1.0
                    };
                    // Smooth update to avoid sudden changes
                    self.intraday_multipliers[i] = 0.7 * self.intraday_multipliers[i] + 0.3 * new_multiplier;
                }
            }
        }
    }

    /// Get current intraday multiplier
    fn get_intraday_multiplier(&self) -> f64 {
        let current_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let hour = ((current_time % 86400) / 3600) as usize;
        if hour < 24 {
            self.intraday_multipliers[hour]
        } else {
            1.0
        }
    }

    /// Get improved volatility estimate accounting for crypto market dynamics
    pub fn get_volatility(&self) -> f64 {
        // Use regime-adaptive volatility estimation
        let base_vol_estimate = if self.vol_regime_high {
            // In high vol regime, use fast EWMA for quick response
            self.fast_ewma.max(self.medium_ewma)
        } else {
            // In normal regime, use medium EWMA for stability
            self.medium_ewma
        };
        
        if base_vol_estimate <= 0.0 {
            return self.fallback_volatility();
        }
        
        // Convert to annualized volatility with crypto-appropriate scaling
        let base_annual_vol = base_vol_estimate.sqrt() * self.get_annualization_factor();
        
        // Apply intraday adjustment
        let intraday_adjusted_vol = base_annual_vol * self.get_intraday_multiplier();
        
        // Apply autocorrelation adjustment for crypto markets
        let autocorr_adjusted_vol = intraday_adjusted_vol * self.autocorr_adjustment;
        
        autocorr_adjusted_vol.max(0.01).min(10.0) // Reasonable bounds: 1% to 1000% annual vol
    }

    /// Calculate appropriate annualization factor based on observation frequency
    fn get_annualization_factor(&self) -> f64 {
        // Crypto trades 24/7, so use full year
        // Adjust based on actual observation frequency rather than assuming 1-minute
        let observations_per_year = 365.25 * 24.0 * 3600.0 / self.observation_frequency_secs as f64;
        observations_per_year.sqrt()
    }

    /// Fallback volatility calculation using sample standard deviation
    fn fallback_volatility(&self) -> f64 {
        if self.returns.len() < 10 {
            return 0.05; // Default 5% annual volatility
        }
        
        let returns_only: Vec<f64> = self.returns.iter().map(|(_, r)| *r).collect();
        let mean = returns_only.iter().sum::<f64>() / returns_only.len() as f64;
        
        let variance = returns_only
            .iter()
            .map(|r| (r - mean).powi(2))
            .sum::<f64>() / (returns_only.len() - 1) as f64;
        
        let vol = variance.sqrt() * self.get_annualization_factor();
        vol * self.autocorr_adjustment
    }

    /// Get volatility-based spread multiplier with regime awareness
    pub fn get_volatility_spread_multiplier(&self, base_volatility: f64) -> f64 {
        let current_vol = self.get_volatility();
        if base_volatility < 1e-6 || current_vol < 1e-6 {
            return 1.0;
        }
        
        let vol_ratio = current_vol / base_volatility;
        
        // Apply regime-specific multiplier bounds
        let (min_mult, max_mult) = if self.vol_regime_high {
            (0.8, 5.0) // Allow wider spreads in high vol regime
        } else {
            (0.5, 3.0) // Standard bounds in normal regime
        };
        
        vol_ratio.max(min_mult).min(max_mult)
    }

    /// Get comprehensive volatility metrics for monitoring and analysis
    pub fn get_metrics(&self) -> VolatilityMetrics {
        VolatilityMetrics {
            current_volatility: self.get_volatility(),
            fast_ewma_vol: if self.fast_ewma > 0.0 { 
                self.fast_ewma.sqrt() * self.get_annualization_factor()
            } else { 0.0 },
            medium_ewma_vol: if self.medium_ewma > 0.0 { 
                self.medium_ewma.sqrt() * self.get_annualization_factor() 
            } else { 0.0 },
            slow_ewma_vol: if self.slow_ewma > 0.0 { 
                self.slow_ewma.sqrt() * self.get_annualization_factor() 
            } else { 0.0 },
            vol_regime_high: self.vol_regime_high,
            intraday_multiplier: self.get_intraday_multiplier(),
            returns_count: self.returns.len(),
            price_observations: self.price_history.len(),
        }
    }

    /// Get short-term volatility for immediate risk management decisions
    pub fn get_short_term_volatility(&self) -> f64 {
        // Use fast EWMA for immediate risk decisions
        if self.fast_ewma > 0.0 {
            let short_term_vol = self.fast_ewma.sqrt() * self.get_annualization_factor();
            short_term_vol * self.get_intraday_multiplier() * self.autocorr_adjustment
        } else {
            self.get_volatility()
        }
    }
}

#[derive(Debug, Clone)]
pub struct VolatilityMetrics {
    pub current_volatility: f64,
    pub fast_ewma_vol: f64,
    pub medium_ewma_vol: f64,
    pub slow_ewma_vol: f64,
    pub vol_regime_high: bool,
    pub intraday_multiplier: f64,
    pub returns_count: usize,
    pub price_observations: usize,
}

#[derive(Debug, Clone)]
pub struct AdverseSelectionMonitor {
    fill_history: VecDeque<Fill>,
    adverse_selection_threshold: f64,
    max_history_size: usize,
    // Predictive components
    price_momentum_window: VecDeque<(u64, f64)>, // (timestamp, mid_price)
    recent_volume_profile: VecDeque<(u64, f64, bool)>, // (timestamp, size, is_aggressive)
    last_mid_price: f64,
    momentum_threshold: f64, // Price momentum that suggests informed trading
    volume_spike_threshold: f64, // Volume spike multiplier that suggests toxicity
    time_window_seconds: u64, // Look back window for predictive signals
}

impl AdverseSelectionMonitor {
    pub fn new(threshold: f64, max_history: usize) -> Self {
        Self {
            fill_history: VecDeque::with_capacity(max_history),
            adverse_selection_threshold: threshold,
            max_history_size: max_history,
            price_momentum_window: VecDeque::with_capacity(100),
            recent_volume_profile: VecDeque::with_capacity(200),
            last_mid_price: 0.0,
            momentum_threshold: 0.0005, // 5 basis points of momentum
            volume_spike_threshold: 2.0, // 2x normal volume
            time_window_seconds: 30, // 30 second lookback for predictive signals
        }
    }

    pub fn add_fill(&mut self, fill: Fill) {
        self.fill_history.push_back(fill);
        
        // Keep history size manageable
        if self.fill_history.len() > self.max_history_size {
            self.fill_history.pop_front();
        }
    }

    /// Update market data for predictive analysis
    pub fn update_market_data(&mut self, mid_price: f64, recent_trade_size: Option<f64>, is_aggressive_trade: Option<bool>) {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        
        // Clean old data
        let cutoff_time = timestamp - self.time_window_seconds;
        
        // Update price momentum
        self.price_momentum_window.push_back((timestamp, mid_price));
        while let Some((old_time, _)) = self.price_momentum_window.front() {
            if *old_time < cutoff_time {
                self.price_momentum_window.pop_front();
            } else {
                break;
            }
        }
        
        // Update volume profile
        if let (Some(size), Some(is_aggressive)) = (recent_trade_size, is_aggressive_trade) {
            self.recent_volume_profile.push_back((timestamp, size, is_aggressive));
        }
        while let Some((old_time, _, _)) = self.recent_volume_profile.front() {
            if *old_time < cutoff_time {
                self.recent_volume_profile.pop_front();
            } else {
                break;
            }
        }
        
        self.last_mid_price = mid_price;
    }

    /// Calculate predictive adverse selection risk BEFORE placing orders
    pub fn predict_adverse_selection_risk(&self) -> f64 {
        let mut risk_score = 0.0;
        
        // 1. Price momentum analysis - detect if price is trending strongly
        let momentum_risk = self.calculate_momentum_risk();
        risk_score += momentum_risk * 0.4;
        
        // 2. Volume toxicity analysis - detect unusual volume patterns
        let volume_risk = self.calculate_volume_toxicity();
        risk_score += volume_risk * 0.3;
        
        // 3. Recent fill pattern analysis - are we getting picked off?
        let pattern_risk = self.calculate_fill_pattern_risk();
        risk_score += pattern_risk * 0.3;
        
        risk_score.min(1.0).max(0.0)
    }

    /// Detect price momentum that suggests informed trading
    fn calculate_momentum_risk(&self) -> f64 {
        if self.price_momentum_window.len() < 5 {
            return 0.0;
        }
        
        let oldest_price = self.price_momentum_window.front().unwrap().1;
        let newest_price = self.price_momentum_window.back().unwrap().1;
        let price_change_pct = (newest_price - oldest_price) / oldest_price;
        
        // Calculate price acceleration (second derivative)
        let mid_idx = self.price_momentum_window.len() / 2;
        let mid_price = self.price_momentum_window[mid_idx].1;
        let first_half_change = (mid_price - oldest_price) / oldest_price;
        let second_half_change = (newest_price - mid_price) / mid_price;
        
        let acceleration = second_half_change - first_half_change;
        
        // High momentum + acceleration suggests informed trading
        let momentum_magnitude = price_change_pct.abs() / self.momentum_threshold;
        let acceleration_factor = (acceleration.abs() / self.momentum_threshold).min(2.0);
        
        (momentum_magnitude * (1.0 + acceleration_factor)).min(1.0)
    }

    /// Detect volume spikes and aggressive trading patterns
    fn calculate_volume_toxicity(&self) -> f64 {
        if self.recent_volume_profile.len() < 10 {
            return 0.0;
        }
        
        // Calculate baseline volume (excluding most recent trades)
        let baseline_count = self.recent_volume_profile.len() * 2 / 3;
        let baseline_volume: f64 = self.recent_volume_profile
            .iter()
            .take(baseline_count)
            .map(|(_, size, _)| size)
            .sum::<f64>() / baseline_count as f64;
        
        if baseline_volume < 1e-10 {
            return 0.0;
        }
        
        // Check recent volume vs baseline
        let recent_count = self.recent_volume_profile.len() - baseline_count;
        let recent_volume: f64 = self.recent_volume_profile
            .iter()
            .skip(baseline_count)
            .map(|(_, size, _)| size)
            .sum::<f64>() / recent_count as f64;
        
        let volume_spike_ratio = recent_volume / baseline_volume;
        
        // Check proportion of aggressive trades (market orders hitting our quotes)
        let aggressive_trades: usize = self.recent_volume_profile
            .iter()
            .skip(baseline_count)
            .filter(|(_, _, is_aggressive)| *is_aggressive)
            .count();
        
        let aggressive_ratio = aggressive_trades as f64 / recent_count as f64;
        
        // Combine volume spike with aggressive trading ratio
        let volume_risk = ((volume_spike_ratio - 1.0) / self.volume_spike_threshold).max(0.0);
        let aggression_risk = aggressive_ratio;
        
        (volume_risk * 0.6 + aggression_risk * 0.4).min(1.0)
    }

    /// Analyze our recent fill patterns for adverse selection
    fn calculate_fill_pattern_risk(&self) -> f64 {
        if self.fill_history.len() < 5 {
            return 0.0;
        }
        
        let recent_fills = self.fill_history.iter().rev().take(10).collect::<Vec<_>>();
        let mut pattern_risk = 0.0;
        
        // 1. Check if we're consistently getting filled on one side
        let buy_fills = recent_fills.iter().filter(|f| f.is_buy).count();
        let sell_fills = recent_fills.iter().filter(|f| !f.is_buy).count();
        let total_fills = buy_fills + sell_fills;
        
        if total_fills > 0 {
            let imbalance = (buy_fills as f64 - sell_fills as f64).abs() / total_fills as f64;
            pattern_risk += imbalance * 0.3;
        }
        
        // 2. Check if price moves against us immediately after fills
        let mut immediate_adverse_moves = 0;
        for fill in &recent_fills {
            let price_move = fill.current_mid - fill.fill_price;
            let adverse_move = if fill.is_buy {
                price_move < -0.0001 // Price dropped after we bought
            } else {
                price_move > 0.0001 // Price rose after we sold
            };
            
            if adverse_move {
                immediate_adverse_moves += 1;
            }
        }
        
        let adverse_ratio = immediate_adverse_moves as f64 / recent_fills.len() as f64;
        pattern_risk += adverse_ratio * 0.4;
        
        // 3. Check fill timing patterns (are we being hit right before moves?)
        let mut pre_move_fills = 0;
        for i in 0..(recent_fills.len() - 1) {
            let current_fill = recent_fills[i];
            let next_fill = recent_fills[i + 1];
            
            // If fills happen close together and in opposite directions
            let time_diff = current_fill.timestamp.abs_diff(next_fill.timestamp);
            if time_diff < 10 && current_fill.is_buy != next_fill.is_buy {
                pre_move_fills += 1;
            }
        }
        
        if recent_fills.len() > 1 {
            let pre_move_ratio = pre_move_fills as f64 / (recent_fills.len() - 1) as f64;
            pattern_risk += pre_move_ratio * 0.3;
        }
        
        pattern_risk.min(1.0)
    }

    /// Get predictive spread adjustment BEFORE adverse selection occurs
    pub fn predictive_spread_adjustment(&self) -> f64 {
        let risk_score = self.predict_adverse_selection_risk();
        
        if risk_score > 0.3 {
            // Exponential increase in spread based on risk
            let base_adjustment = 1.0 + (risk_score * 2.0); // Up to 3x spread widening
            base_adjustment.min(3.0).max(1.0)
        } else {
            1.0
        }
    }

    /// Legacy reactive adverse selection score (keep for backwards compatibility)
    pub fn adverse_selection_score(&self) -> f64 {
        if self.fill_history.len() < 10 {
            return 0.0;
        }
        
        let mut adverse_pnl = 0.0;
        
        for fill in &self.fill_history {
            let price_move = fill.current_mid - fill.fill_price;
            // If we bought and price went down, or sold and price went up
            let realized_adverse = if fill.is_buy {
                -price_move * fill.size
            } else {
                price_move * fill.size
            };
            adverse_pnl += realized_adverse;
        }
        
        // Normalize by number of fills and average fill size
        let avg_fill_size: f64 = self.fill_history.iter().map(|f| f.size).sum::<f64>() / self.fill_history.len() as f64;
        adverse_pnl / (self.fill_history.len() as f64 * avg_fill_size)
    }
    
    /// Reactive spread adjustment (legacy - now secondary to predictive)
    pub fn spread_adjustment(&self) -> f64 {
        let score = self.adverse_selection_score();
        if score > self.adverse_selection_threshold {
            let adjustment = 1.0 + (score / self.adverse_selection_threshold) * 0.5;
            adjustment.min(2.0) // Cap at 100% spread widening
        } else {
            1.0
        }
    }

    /// Get comprehensive metrics including predictive signals
    pub fn get_metrics(&self) -> (f64, usize, f64, f64, f64, f64) {
        let reactive_score = self.adverse_selection_score();
        let fill_count = self.fill_history.len();
        let reactive_adjustment = self.spread_adjustment();
        let predictive_risk = self.predict_adverse_selection_risk();
        let predictive_adjustment = self.predictive_spread_adjustment();
        let momentum_risk = self.calculate_momentum_risk();
        
        (reactive_score, fill_count, reactive_adjustment, predictive_risk, predictive_adjustment, momentum_risk)
    }

    /// Check if we should temporarily stop trading (now uses predictive + reactive signals)
    pub fn should_pause_trading(&self) -> bool {
        let reactive_score = self.adverse_selection_score();
        let predictive_risk = self.predict_adverse_selection_risk();
        
        // Pause if either reactive adverse selection is severe OR predictive risk is very high
        (reactive_score > self.adverse_selection_threshold * 2.0 && self.fill_history.len() >= 20) ||
        (predictive_risk > 0.8)
    }

    /// Update current mid price for all recent fills (for real-time adverse selection calculation)
    pub fn update_current_mid(&mut self, current_mid: f64) {
        let current_timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        
        // Update current_mid for recent fills (within last 60 seconds)
        for fill in self.fill_history.iter_mut() {
            if current_timestamp - fill.timestamp < 60 {
                fill.current_mid = current_mid;
            }
        }
        
        // Update market data for predictive analysis
        self.update_market_data(current_mid, None, None);
    }
}
#[derive(Debug, Clone)]
pub struct OrderBookAnalyzer {
    bid_depth: Vec<(f64, f64)>, // (price, size)
    ask_depth: Vec<(f64, f64)>,
}

impl OrderBookAnalyzer {
    pub fn new() -> Self {
        Self {
            bid_depth: Vec::new(),
            ask_depth: Vec::new(),
        }
    }

    /// Update the order book with new L2 data
    pub fn update_order_book(&mut self, l2_data: &L2Book) {
        self.bid_depth.clear();
        self.ask_depth.clear();

        // L2Book levels: [0] = bids (sorted high to low), [1] = asks (sorted low to high)
        if l2_data.data.levels.len() >= 2 {
            // Process bids (index 0)
            for level in &l2_data.data.levels[0] {
                if let (Ok(price), Ok(size)) = (level.px.parse::<f64>(), level.sz.parse::<f64>()) {
                    if size > 0.0 {
                        self.bid_depth.push((price, size));
                    }
                }
            }

            // Process asks (index 1)
            for level in &l2_data.data.levels[1] {
                if let (Ok(price), Ok(size)) = (level.px.parse::<f64>(), level.sz.parse::<f64>()) {
                    if size > 0.0 {
                        self.ask_depth.push((price, size));
                    }
                }
            }
        }

        // Ensure proper sorting
        self.bid_depth.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap()); // Highest to lowest
        self.ask_depth.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap()); // Lowest to highest
    }

    /// Estimate fill probability based on queue position
    pub fn estimate_fill_probability(&self, our_price: f64, is_buy: bool) -> f64 {
        let levels = if is_buy { &self.bid_depth } else { &self.ask_depth };
        
        let total_volume_ahead: f64 = levels
            .iter()
            .take_while(|(price, _)| {
                if is_buy { *price > our_price } else { *price < our_price }
            })
            .map(|(_, size)| size)
            .sum();
        
        // Exponential decay model
        (-0.001 * total_volume_ahead).exp()
    }
    
    /// Calculate order book imbalance (positive = more bids, negative = more asks)
    pub fn order_book_imbalance(&self) -> f64 {
        let bid_vol: f64 = self.bid_depth.iter().take(5).map(|(_, s)| s).sum();
        let ask_vol: f64 = self.ask_depth.iter().take(5).map(|(_, s)| s).sum();
        
        if bid_vol + ask_vol == 0.0 {
            return 0.0;
        }
        
        (bid_vol - ask_vol) / (bid_vol + ask_vol)
    }

    /// Detect large orders (potential toxicity) - returns true if detected
    pub fn has_large_orders(&self, threshold_percentile: f64) -> bool {
        let total_bid_volume: f64 = self.bid_depth.iter().map(|(_, size)| size).sum();
        let total_ask_volume: f64 = self.ask_depth.iter().map(|(_, size)| size).sum();

        let bid_threshold = total_bid_volume * threshold_percentile;
        let ask_threshold = total_ask_volume * threshold_percentile;

        self.bid_depth.iter().any(|(_, size)| *size >= bid_threshold) ||
        self.ask_depth.iter().any(|(_, size)| *size >= ask_threshold)
    }

    /// Get the current mid price from the order book
    pub fn get_mid_price(&self) -> Option<f64> {
        let best_bid = self.bid_depth.first()?.0;
        let best_ask = self.ask_depth.first()?.0;
        Some((best_bid + best_ask) / 2.0)
    }

    /// Get market depth at different price levels
    pub fn get_market_depth(&self, levels: usize) -> (Vec<(f64, f64)>, Vec<(f64, f64)>) {
        let bids = self.bid_depth.iter().take(levels).cloned().collect();
        let asks = self.ask_depth.iter().take(levels).cloned().collect();
        (bids, asks)
    }

    /// Optimize quote placement based on order book analysis
    pub fn optimize_quote_placement(&self, target_spread_bps: u16, fallback_mid: f64) -> (f64, f64) {
        let mid_price = self.get_mid_price().unwrap_or(fallback_mid);
        let imbalance = self.order_book_imbalance();
        
        // Adjust spread based on imbalance
        let base_half_spread = (mid_price * target_spread_bps as f64) / 20000.0;
        
        // If there's significant imbalance, adjust our quotes
        let imbalance_adjustment = imbalance * base_half_spread * 0.1; // 10% of half spread max adjustment
        
        let optimal_bid = mid_price - base_half_spread - imbalance_adjustment;
        let optimal_ask = mid_price + base_half_spread + imbalance_adjustment;
        
        (optimal_bid, optimal_ask)
    }
}

#[derive(Debug, Clone)]
pub struct RiskManager {
    pub initial_capital: f64,
    pub current_capital: f64,
    pub max_drawdown_pct: f64,
    pub max_var_95: f64, // 95% VaR limit
    pub position_value_limit_pct: f64, // Max % of capital in one position
    pub max_daily_loss_pct: f64, // Max daily loss percentage
    pub daily_pnl: f64,
    pub unrealized_pnl: f64,
    pub pnl_history: VecDeque<f64>, // For VaR calculation
    pub last_reset_day: u64,
    pub heat_limit_pct: f64, // Portfolio heat limit
    pub current_heat: f64,
}

impl RiskManager {
    pub fn new(
        initial_capital: f64,
        max_drawdown_pct: f64,
        max_var_95: f64,
        position_value_limit_pct: f64,
        max_daily_loss_pct: f64,
        heat_limit_pct: f64,
    ) -> Self {
        Self {
            initial_capital,
            current_capital: initial_capital,
            max_drawdown_pct,
            max_var_95,
            position_value_limit_pct,
            max_daily_loss_pct,
            daily_pnl: 0.0,
            unrealized_pnl: 0.0,
            pnl_history: VecDeque::with_capacity(100), // Keep 100 days of history
            last_reset_day: Self::current_day(),
            heat_limit_pct,
            current_heat: 0.0,
        }
    }

    fn current_day() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs() / 86400 // Convert to days
    }

    pub fn update_pnl(&mut self, realized_pnl_delta: f64, new_unrealized_pnl: f64) {
        let current_day = Self::current_day();
        
        // Reset daily PnL if new day
        if current_day != self.last_reset_day {
            self.pnl_history.push_back(self.daily_pnl);
            if self.pnl_history.len() > 100 {
                self.pnl_history.pop_front();
            }
            self.daily_pnl = 0.0;
            self.last_reset_day = current_day;
        }

        self.daily_pnl += realized_pnl_delta;
        self.current_capital += realized_pnl_delta;
        self.unrealized_pnl = new_unrealized_pnl;
        
        // Update portfolio heat (max adverse excursion)
        let total_pnl = self.daily_pnl + self.unrealized_pnl;
        if total_pnl < 0.0 {
            self.current_heat = self.current_heat.max(total_pnl.abs() / self.initial_capital);
        }
    }

    pub fn calculate_var_95(&self) -> f64 {
        if self.pnl_history.len() < 20 {
            return self.max_var_95; // Conservative default
        }

        let mut sorted_pnl: Vec<f64> = self.pnl_history.iter().cloned().collect();
        sorted_pnl.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let index = (sorted_pnl.len() as f64 * 0.05) as usize; // 5th percentile
        sorted_pnl.get(index).copied().unwrap_or(0.0).abs()
    }

    pub fn can_trade(&self, position_value: f64) -> RiskCheckResult {
        let total_pnl = self.daily_pnl + self.unrealized_pnl;
        
        // Check drawdown
        let current_drawdown = (self.initial_capital - self.current_capital - self.unrealized_pnl) 
            / self.initial_capital;
        if current_drawdown > self.max_drawdown_pct {
            return RiskCheckResult::Rejected("Maximum drawdown exceeded".to_string());
        }
        
        // Check daily loss limit
        if total_pnl < -(self.initial_capital * self.max_daily_loss_pct) {
            return RiskCheckResult::Rejected("Daily loss limit exceeded".to_string());
        }
        
        // Check portfolio heat
        if self.current_heat > self.heat_limit_pct {
            return RiskCheckResult::Rejected("Portfolio heat limit exceeded".to_string());
        }
        
        // Check position concentration
        if position_value.abs() > self.current_capital * self.position_value_limit_pct {
            return RiskCheckResult::Rejected("Position concentration limit exceeded".to_string());
        }
        
        // Check VaR
        let current_var = self.calculate_var_95();
        if current_var > self.max_var_95 {
            return RiskCheckResult::ReduceSize(0.5); // Reduce position size by 50%
        }
        
        RiskCheckResult::Approved
    }

    pub fn get_max_position_size(&self, asset_price: f64) -> f64 {
        let max_by_capital = (self.current_capital * self.position_value_limit_pct) / asset_price;
        let max_by_var = if self.calculate_var_95() > self.max_var_95 * 0.8 {
            max_by_capital * 0.5 // Reduce size if approaching VaR limit
        } else {
            max_by_capital
        };
        
        max_by_capital.min(max_by_var)
    }

    pub fn should_reduce_exposure(&self) -> f64 {
        let total_pnl = self.daily_pnl + self.unrealized_pnl;
        let loss_ratio = total_pnl.abs() / (self.initial_capital * self.max_daily_loss_pct);
        
        if loss_ratio > 0.8 {
            0.5 // Reduce exposure by 50%
        } else if loss_ratio > 0.6 {
            0.75 // Reduce exposure by 25%
        } else {
            1.0 // No reduction
        }
    }

    pub fn reset_heat(&mut self) {
        self.current_heat = 0.0;
    }
}

#[derive(Debug)]
pub enum RiskCheckResult {
    Approved,
    ReduceSize(f64), // Factor to multiply position size by
    Rejected(String), // Reason for rejection
}

#[derive(Debug)]
pub struct MarketMakerRestingOrder {
    pub oid: u64,
    pub position: f64,
    pub price: f64,
}

#[derive(Debug, Clone)]
pub struct VirtualOrder {
    pub id: u64,
    pub asset: String,
    pub size: f64,
    pub price: f64,
    pub is_buy: bool,
    pub timestamp: u64,
    pub filled_size: f64,
    pub status: VirtualOrderStatus,
}

#[derive(Debug, Clone)]
pub enum VirtualOrderStatus {
    Pending,
    PartiallyFilled,
    Filled,
    Cancelled,
}

#[derive(Debug)]
pub struct MarketMakerInput {
    pub asset: String,
    pub target_liquidity: f64, // Amount of liquidity on both sides to target
    pub half_spread: u16,      // Half of the spread for our market making (in BPS)
    pub max_bps_diff: u16, // Max deviation before we cancel and put new orders on the book (in BPS)
    pub max_absolute_position_size: f64, // Absolute value of the max position we can take on
    pub decimals: u32,     // Decimals to round to for pricing
    pub wallet: PrivateKeySigner, // Wallet containing private key
    // Risk management parameters
    pub initial_capital: f64,
    pub max_drawdown_pct: f64,
    pub max_var_95: f64,
    pub position_value_limit_pct: f64,
    pub max_daily_loss_pct: f64,
    pub heat_limit_pct: f64,
    // Adverse selection parameters
    pub adverse_selection_threshold: f64,
    pub adverse_selection_history_size: usize,
    // Volatility parameters
    pub base_volatility: f64, // Expected "normal" volatility for the asset
    pub volatility_lookback_minutes: u64, // How far back to look for volatility calculation  
    pub observation_frequency_secs: u64, // Expected frequency of price updates (for proper annualization)
    // Simulation parameters
    pub simulation_mode: bool, // If true, don't place real orders, just track performance
}

#[derive(Debug)]
pub struct MarketMaker {
    pub asset: String,
    pub target_liquidity: f64,
    pub half_spread: u16,
    pub max_bps_diff: u16,
    pub max_absolute_position_size: f64,
    pub decimals: u32,
    pub lower_resting: MarketMakerRestingOrder,
    pub upper_resting: MarketMakerRestingOrder,
    pub cur_position: f64,
    pub latest_mid_price: f64,
    pub info_client: InfoClient,
    pub exchange_client: ExchangeClient,
    pub user_address: Address,
    pub order_book_analyzer: OrderBookAnalyzer,
    pub risk_manager: RiskManager,
    pub adverse_selection_monitor: AdverseSelectionMonitor,
    pub pnl_tracker: PnLTracker,
    pub volatility_estimator: VolatilityEstimator,
    pub base_volatility: f64,
    pub simulation_mode: bool,
    pub virtual_orders: Vec<VirtualOrder>, // Track virtual orders in simulation mode
}

impl MarketMaker {
    pub async fn new(input: MarketMakerInput) -> MarketMaker {
        let user_address = input.wallet.address();

        let info_client = InfoClient::new(None, Some(BaseUrl::Mainnet)).await.unwrap();
        let exchange_client =
            ExchangeClient::new(None, input.wallet, Some(BaseUrl::Mainnet), None, None)
                .await
                .unwrap();

        let risk_manager = RiskManager::new(
            input.initial_capital,
            input.max_drawdown_pct,
            input.max_var_95,
            input.position_value_limit_pct,
            input.max_daily_loss_pct,
            input.heat_limit_pct,
        );

        let adverse_selection_monitor = AdverseSelectionMonitor::new(
            input.adverse_selection_threshold,
            input.adverse_selection_history_size,
        );

        let volatility_estimator = VolatilityEstimator::new(
            input.volatility_lookback_minutes,
            input.observation_frequency_secs,
        );

        MarketMaker {
            asset: input.asset,
            target_liquidity: input.target_liquidity,
            half_spread: input.half_spread,
            max_bps_diff: input.max_bps_diff,
            max_absolute_position_size: input.max_absolute_position_size,
            decimals: input.decimals,
            lower_resting: MarketMakerRestingOrder {
                oid: 0,
                position: 0.0,
                price: -1.0,
            },
            upper_resting: MarketMakerRestingOrder {
                oid: 0,
                position: 0.0,
                price: -1.0,
            },
            cur_position: 0.0,
            latest_mid_price: -1.0,
            info_client,
            exchange_client,
            user_address,
            order_book_analyzer: OrderBookAnalyzer::new(),
            risk_manager,
            adverse_selection_monitor,
            pnl_tracker: PnLTracker::new(input.initial_capital),
            volatility_estimator,
            base_volatility: input.base_volatility,
            simulation_mode: input.simulation_mode,
            virtual_orders: Vec::new(),
        }
    }

    pub async fn start(&mut self) {
        let (sender, mut receiver) = unbounded_channel();

        // Subscribe to UserEvents for fills
        self.info_client
            .subscribe(
                Subscription::UserEvents {
                    user: self.user_address,
                },
                sender.clone(),
            )
            .await
            .unwrap();

        // Subscribe to AllMids so we can market make around the mid price
        self.info_client
            .subscribe(Subscription::AllMids, sender.clone())
            .await
            .unwrap();

        // Subscribe to L2Book for order book analysis
        self.info_client
            .subscribe(
                Subscription::L2Book {
                    coin: self.asset.clone(),
                },
                sender,
            )
            .await
            .unwrap();

        loop {
            let message = receiver.recv().await.unwrap();
            match message {
                Message::AllMids(all_mids) => {
                    let all_mids = all_mids.data.mids;
                    let mid = all_mids.get(&self.asset);
                    if let Some(mid) = mid {
                        let mid: f64 = mid.parse().unwrap();
                        self.latest_mid_price = mid;
                        
                        // Update volatility estimator with new price
                        self.volatility_estimator.add_price(mid);
                        
                        // Update PnL tracker's unrealized PnL with new mid price
                        self.pnl_tracker.update_unrealized(
                            self.cur_position,
                            self.pnl_tracker.average_entry_price,
                            mid
                        );
                        
                        // Update adverse selection monitor with current mid price
                        self.adverse_selection_monitor.update_current_mid(mid);
                        
                        // Check virtual fills in simulation mode
                        if self.simulation_mode {
                            self.check_virtual_fills(mid);
                        }
                        
                        // Check to see if we need to cancel or place any new orders
                        self.potentially_update().await;
                    } else {
                        error!(
                            "could not get mid for asset {}: {all_mids:?}",
                            self.asset.clone()
                        );
                    }
                }
                Message::L2Book(l2_book) => {
                    // Update order book analysis
                    self.order_book_analyzer.update_order_book(&l2_book);
                    
                    // Log order book analysis periodically (every 100th update to avoid spam)
                    static mut UPDATE_COUNTER: usize = 0;
                    unsafe {
                        UPDATE_COUNTER += 1;
                        if UPDATE_COUNTER % 100 == 0 {
                            let imbalance = self.order_book_analyzer.order_book_imbalance();
                            let has_large_orders = self.order_book_analyzer.has_large_orders(0.3);
                            
                            info!(
                                "Order Book Analysis: Imbalance={:.3}, Large Orders={}, Mid Price={:.4}",
                                imbalance,
                                has_large_orders,
                                self.order_book_analyzer.get_mid_price().unwrap_or(0.0)
                            );
                            
                            if has_large_orders {
                                info!("⚠️  Large orders detected - potential market impact");
                            }
                        }
                    }
                }
                Message::User(user_events) => {
                    // Skip real user events in simulation mode
                    if self.simulation_mode {
                        continue;
                    }
                    
                    // We haven't seen the first mid price event yet, so just continue
                    if self.latest_mid_price < 0.0 {
                        continue;
                    }
                    let user_events = user_events.data;
                    if let UserData::Fills(fills) = user_events {
                        for fill in fills {
                            let amount: f64 = fill.sz.parse().unwrap();
                            let fill_price: f64 = fill.px.parse().unwrap_or(0.0);
                            let is_buy = fill.side.eq("B");
                            
                            // Create fill record for adverse selection monitoring
                            let fill_record = Fill {
                                fill_price,
                                current_mid: self.latest_mid_price,
                                is_buy,
                                size: amount,
                                timestamp: SystemTime::now()
                                    .duration_since(UNIX_EPOCH)
                                    .unwrap()
                                    .as_secs(),
                            };
                            
                            // Add to adverse selection monitor and update with trade data
                            self.adverse_selection_monitor.add_fill(fill_record);
                            // Update with market data - this fill is aggressive (it hit our passive quote)
                            self.adverse_selection_monitor.update_market_data(self.latest_mid_price, Some(amount), Some(true));
                            
                            // Update our resting positions whenever we see a fill
                            if is_buy {
                                self.cur_position += amount;
                                self.lower_resting.position -= amount;
                                info!("Fill: bought {amount} {} at {fill_price}", self.asset.clone());
                            } else {
                                self.cur_position -= amount;
                                self.upper_resting.position -= amount;
                                info!("Fill: sold {amount} {} at {fill_price}", self.asset.clone());
                            }
                            
                            // Create trade record for PnL tracking
                            let trade = Trade {
                                timestamp: SystemTime::now()
                                    .duration_since(UNIX_EPOCH)
                                    .unwrap()
                                    .as_secs(),
                                asset: self.asset.clone(),
                                side: if is_buy { "B".to_string() } else { "S".to_string() },
                                size: amount,
                                price: fill_price,
                                pnl: 0.0, // Will be calculated in add_trade
                                position_after: self.cur_position,
                            };
                            
                            // Add trade to PnL tracker
                            self.pnl_tracker.add_trade(trade.clone());
                            
                            // Update unrealized PnL in tracker
                            self.pnl_tracker.update_unrealized(
                                self.cur_position,
                                self.pnl_tracker.average_entry_price,
                                self.latest_mid_price
                            );
                            
                            // Update risk manager with PnL from tracker
                            let unrealized_pnl = self.pnl_tracker.unrealized_pnl;
                            let realized_pnl_delta = if let Some(last_trade) = self.pnl_tracker.trades.last() {
                                last_trade.pnl
                            } else {
                                0.0
                            };
                            self.risk_manager.update_pnl(realized_pnl_delta, unrealized_pnl);
                            
                            // Structured logging for trade
                            StructuredLogger::log_trade(&trade, self.cur_position, unrealized_pnl);
                            
                            // Log adverse selection metrics and PnL
                            let (reactive_score, _fill_count, reactive_adj, predictive_risk, predictive_adj, momentum_risk) = self.adverse_selection_monitor.get_metrics();
                            let performance_metrics = self.pnl_tracker.get_performance_metrics();
                            
                            info!("💰 PnL Update: Realized=${:.2}, Unrealized=${:.2}, Daily=${:.2}", 
                                  performance_metrics.realized_pnl, performance_metrics.unrealized_pnl, self.risk_manager.daily_pnl);
                            info!("📊 Performance: Win Rate={:.1}%, Profit Factor={:.2}, Sharpe={:.2}, Max DD={:.1}%", 
                                  performance_metrics.win_rate * 100.0, performance_metrics.profit_factor, 
                                  performance_metrics.sharpe_ratio, performance_metrics.max_drawdown * 100.0);
                            info!("� Adverse Selection: Predictive Risk={:.1}% (adj={:.2}x), Reactive Score={:.4} (adj={:.2}x), Momentum Risk={:.1}%", 
                                  predictive_risk * 100.0, predictive_adj, reactive_score, reactive_adj, momentum_risk * 100.0);
                        }
                    }
                    // Check to see if we need to cancel or place any new orders
                    self.potentially_update().await;
                }
                _ => {
                    // Handle other message types or log warning
                    info!("Received unsupported message type");
                }
            }
        }
    }

    async fn attempt_cancel(&mut self, asset: String, oid: u64) -> bool {
        if self.simulation_mode {
            return self.cancel_virtual_order(oid).await;
        }
        
        // Real cancel logic
        let cancel = self
            .exchange_client
            .cancel(ClientCancelRequest { asset, oid }, None)
            .await;

        match cancel {
            Ok(cancel) => match cancel {
                ExchangeResponseStatus::Ok(cancel) => {
                    if let Some(cancel) = cancel.data {
                        if !cancel.statuses.is_empty() {
                            match cancel.statuses[0].clone() {
                                ExchangeDataStatus::Success => {
                                    return true;
                                }
                                ExchangeDataStatus::Error(e) => {
                                    error!("Error with cancelling: {e}")
                                }
                                _ => unreachable!(),
                            }
                        } else {
                            error!("Exchange data statuses is empty when cancelling: {cancel:?}")
                        }
                    } else {
                        error!("Exchange response data is empty when cancelling: {cancel:?}")
                    }
                }
                ExchangeResponseStatus::Err(e) => error!("Error with cancelling: {e}"),
            },
            Err(e) => error!("Error with cancelling: {e}"),
        }
        false
    }

    /// Place a virtual order in simulation mode
    fn place_virtual_order(
        &mut self,
        asset: String,
        amount: f64,
        price: f64,
        is_buy: bool,
    ) -> (f64, u64) {
        let order_id = self.virtual_orders.len() as u64 + 1;
        let virtual_order = VirtualOrder {
            id: order_id,
            asset: asset.clone(),
            size: amount,
            price,
            is_buy,
            timestamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
            filled_size: 0.0,
            status: VirtualOrderStatus::Pending,
        };
        
        self.virtual_orders.push(virtual_order.clone());
        StructuredLogger::log_order_event("placed", &asset, amount, price, is_buy, true);
        info!("🎮 SIMULATION: Placed virtual {} order: {:.4} @ {:.4}", 
              if is_buy { "BUY" } else { "SELL" }, amount, price);
        
        (amount, order_id)
    }
    
    /// Cancel a virtual order in simulation mode
    async fn cancel_virtual_order(&mut self, oid: u64) -> bool {
        if let Some(order) = self.virtual_orders.iter_mut().find(|o| o.id == oid) {
            if matches!(order.status, VirtualOrderStatus::Pending | VirtualOrderStatus::PartiallyFilled) {
                order.status = VirtualOrderStatus::Cancelled;
                info!("🎮 SIMULATION: Cancelled virtual order {}", oid);
                return true;
            }
        }
        false
    }
    
    /// Check if virtual orders should be filled based on current market price
    fn check_virtual_fills(&mut self, current_mid: f64) {
        for order in self.virtual_orders.iter_mut() {
            if !matches!(order.status, VirtualOrderStatus::Pending | VirtualOrderStatus::PartiallyFilled) {
                continue;
            }
            
            let should_fill = if order.is_buy {
                current_mid <= order.price // Buy order fills when market price hits or goes below our bid
            } else {
                current_mid >= order.price // Sell order fills when market price hits or goes above our ask
            };
            
            if should_fill {
                let fill_size = order.size - order.filled_size;
                order.filled_size = order.size;
                order.status = VirtualOrderStatus::Filled;
                
                // Simulate the fill by updating our position
                if order.is_buy {
                    self.cur_position += fill_size;
                    self.lower_resting.position -= fill_size;
                } else {
                    self.cur_position -= fill_size;
                    self.upper_resting.position -= fill_size;
                }
                
                // Create trade record for PnL tracking
                let trade = Trade {
                    timestamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
                    asset: order.asset.clone(),
                    side: if order.is_buy { "B".to_string() } else { "S".to_string() },
                    size: fill_size,
                    price: order.price, // Assume we get filled at our order price
                    pnl: 0.0,
                    position_after: self.cur_position,
                };
                
                // Add trade to PnL tracker
                self.pnl_tracker.add_trade(trade);
                
                // Update unrealized PnL
                self.pnl_tracker.update_unrealized(
                    self.cur_position,
                    self.pnl_tracker.average_entry_price,
                    current_mid
                );
                
                // Update risk manager
                let unrealized_pnl = self.pnl_tracker.unrealized_pnl;
                let realized_pnl_delta = if let Some(last_trade) = self.pnl_tracker.trades.last() {
                    last_trade.pnl
                } else {
                    0.0
                };
                self.risk_manager.update_pnl(realized_pnl_delta, unrealized_pnl);
                
                info!("🎮 SIMULATION FILL: {} {:.4} {} @ {:.4}, New Position: {:.4}", 
                      if order.is_buy { "Bought" } else { "Sold" },
                      fill_size, order.asset, order.price, self.cur_position);
            }
        }
    }

    async fn place_order(
        &mut self,
        asset: String,
        amount: f64,
        price: f64,
        is_buy: bool,
    ) -> (f64, u64) {
        if self.simulation_mode {
            return self.place_virtual_order(asset, amount, price, is_buy);
        }
        
        // Real order placement logic
        let order = self
            .exchange_client
            .order(
                ClientOrderRequest {
                    asset,
                    is_buy,
                    reduce_only: false,
                    limit_px: price,
                    sz: amount,
                    cloid: None,
                    order_type: ClientOrder::Limit(ClientLimit {
                        tif: "Gtc".to_string(),
                    }),
                },
                None,
            )
            .await;
        match order {
            Ok(order) => match order {
                ExchangeResponseStatus::Ok(order) => {
                    if let Some(order) = order.data {
                        if !order.statuses.is_empty() {
                            match order.statuses[0].clone() {
                                ExchangeDataStatus::Filled(order) => {
                                    return (amount, order.oid);
                                }
                                ExchangeDataStatus::Resting(order) => {
                                    return (amount, order.oid);
                                }
                                ExchangeDataStatus::Error(e) => {
                                    error!("Error with placing order: {e}")
                                }
                                _ => unreachable!(),
                            }
                        } else {
                            error!("Exchange data statuses is empty when placing order: {order:?}")
                        }
                    } else {
                        error!("Exchange response data is empty when placing order: {order:?}")
                    }
                }
                ExchangeResponseStatus::Err(e) => {
                    error!("Error with placing order: {e}")
                }
            },
            Err(e) => error!("Error with placing order: {e}"),
        }
        (0.0, 0)
    }

    async fn potentially_update(&mut self) {
        // Update PnL tracker's unrealized PnL with current mid price
        self.pnl_tracker.update_unrealized(
            self.cur_position,
            self.pnl_tracker.average_entry_price,
            self.latest_mid_price
        );
        
        // Calculate current position value for risk checks
        let position_value = self.cur_position * self.latest_mid_price;
        let unrealized_pnl = self.pnl_tracker.unrealized_pnl;
        
        // Update risk manager with current unrealized PnL
        self.risk_manager.update_pnl(0.0, unrealized_pnl);
        
        // Periodic performance logging (every ~60 seconds to avoid spam)
        static mut LAST_PERF_LOG: u64 = 0;
        let current_time = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        unsafe {
            if LAST_PERF_LOG == 0 || current_time - LAST_PERF_LOG > 60 {
                let metrics = self.pnl_tracker.get_performance_metrics();
                if metrics.total_trades > 0 || self.cur_position.abs() > 0.001 {
                    StructuredLogger::log_performance(&metrics, &self.risk_manager);
                    info!("📊 CURRENT POSITION: {:.4} HYPE @ avg entry ${:.4}, current mid ${:.4}",
                          self.cur_position, self.pnl_tracker.average_entry_price, self.latest_mid_price);
                }
                LAST_PERF_LOG = current_time;
            }
        }
        
        // PREDICTIVE adverse selection check - prevent losses BEFORE they happen
        let predictive_as_risk = self.adverse_selection_monitor.predict_adverse_selection_risk();
        if predictive_as_risk > 0.8 {
            warn!("🔮 PREDICTIVE: High adverse selection risk detected ({:.1}%), widening spreads preemptively", predictive_as_risk * 100.0);
        }
        
        // Check adverse selection protection (both predictive and reactive)
        if self.adverse_selection_monitor.should_pause_trading() {
            let (reactive_score, _, _, predictive_risk, _, _) = self.adverse_selection_monitor.get_metrics();
            warn!("🚫 Trading paused - Reactive AS: {:.3}, Predictive Risk: {:.1}%", reactive_score, predictive_risk * 100.0);
            self.cancel_all_orders().await;
            return;
        }

        // Check if we can trade at all
        match self.risk_manager.can_trade(position_value) {
            RiskCheckResult::Rejected(reason) => {
                warn!("🚫 Trading halted: {}", reason);
                StructuredLogger::log_risk_event("trading_halted", json!({
                    "reason": reason,
                    "position_value": position_value,
                    "current_capital": self.risk_manager.current_capital,
                    "daily_pnl": self.risk_manager.daily_pnl
                }));
                // Cancel all orders and return
                self.cancel_all_orders().await;
                return;
            }
            RiskCheckResult::ReduceSize(factor) => {
                warn!("⚠️  Reducing position size by factor: {}", factor);
                StructuredLogger::log_risk_event("position_size_reduction", json!({
                    "reduction_factor": factor,
                    "position_value": position_value,
                    "var_95": self.risk_manager.calculate_var_95()
                }));
                self.target_liquidity *= factor;
            }
            RiskCheckResult::Approved => {}
        }

        // Get risk-adjusted maximum position size
        let risk_adjusted_max_size = self.risk_manager.get_max_position_size(self.latest_mid_price)
            .min(self.max_absolute_position_size);
        
        // Apply exposure reduction if needed
        let exposure_factor = self.risk_manager.should_reduce_exposure();
        let adjusted_target_liquidity = self.target_liquidity * exposure_factor;
        
        if exposure_factor < 1.0 {
            info!("📉 Reducing exposure by {}% due to risk limits", (1.0 - exposure_factor) * 100.0);
        }

        // Use PREDICTIVE adverse selection adjustment as primary signal
        let predictive_as_adjustment = self.adverse_selection_monitor.predictive_spread_adjustment();
        let reactive_as_adjustment = self.adverse_selection_monitor.spread_adjustment();
        
        // Take the maximum of predictive and reactive adjustments
        let adverse_selection_adjustment = predictive_as_adjustment.max(reactive_as_adjustment);
        
        // Unified inventory management - only through quote skewing (no spread crossing)
        self.manage_inventory().await;
        
        // Calculate consolidated spread adjustments to avoid double-counting
        let mut total_spread_multiplier = 1.0;
        let mut adjustment_reasons = Vec::new();
        
        // Apply improved volatility adjustment with regime awareness
        let volatility_multiplier = self.volatility_estimator.get_volatility_spread_multiplier(self.base_volatility);
        if (volatility_multiplier - 1.0).abs() > 0.1 {
            total_spread_multiplier *= volatility_multiplier;
            adjustment_reasons.push(format!("volatility ({:.1}x)", volatility_multiplier));
        }
        
        // Apply adverse selection adjustment (predictive + reactive)
        if adverse_selection_adjustment > 1.0 {
            total_spread_multiplier *= adverse_selection_adjustment;
            let adjustment_type = if predictive_as_adjustment > reactive_as_adjustment {
                "predictive AS"
            } else {
                "reactive AS"
            };
            adjustment_reasons.push(format!("{} ({:.1}x)", adjustment_type, adverse_selection_adjustment));
        }
        
        // Apply risk adjustment
        if exposure_factor < 1.0 {
            let risk_multiplier = 1.5;
            total_spread_multiplier *= risk_multiplier;
            adjustment_reasons.push(format!("elevated risk ({:.1}x)", risk_multiplier));
        }
        
        // Apply large order adjustment (reduced sensitivity from 10% to 30%)
        let has_large_orders = self.order_book_analyzer.has_large_orders(0.3);
        if has_large_orders {
            let large_order_multiplier = 1.2;
            total_spread_multiplier *= large_order_multiplier;
            adjustment_reasons.push(format!("large orders ({:.1}x)", large_order_multiplier));
        }
        
        // Calculate final adjusted spread
        let final_adjusted_spread = (self.half_spread as f64 * total_spread_multiplier) as u16;
        
        // Log consolidated spread adjustments
        if total_spread_multiplier > 1.1 {
            let reason = adjustment_reasons.join(", ");
            info!("🔄 Spread widening {:.2}x due to: {}", total_spread_multiplier, reason);
            StructuredLogger::log_spread_adjustment(&reason, self.half_spread, final_adjusted_spread as f64, total_spread_multiplier);
        }
        
        // Use order book analysis for optimized quote placement if available
        let (bid_price, ask_price) = if let Some(ob_mid) = self.order_book_analyzer.get_mid_price() {
            let imbalance = self.order_book_analyzer.order_book_imbalance();
            
            // Use order book optimized placement with consolidated spread adjustment
            let (opt_bid, opt_ask) = self.order_book_analyzer.optimize_quote_placement(final_adjusted_spread, self.latest_mid_price);
            
            // Apply inventory skewing to the optimized quotes
            let base_spread = ((opt_ask - opt_bid) / (2.0 * self.latest_mid_price)) * 10000.0; // Convert to bps
            let (skewed_bid, skewed_ask) = self.calculate_skewed_quotes(base_spread);
            
            // CRITICAL: Robust protection against inverted quotes
            let min_spread = self.latest_mid_price * 0.0001; // 1 basis point minimum spread
            let mut final_bid = skewed_bid;
            let mut final_ask = skewed_ask;
            
            // Ensure proper ordering with sufficient spread
            if final_bid >= final_ask - min_spread {
                warn!("🚨 Order book quotes would invert: bid={:.6} ask={:.6}, fixing...", final_bid, final_ask);
                let mid = self.latest_mid_price;
                final_bid = mid - min_spread / 2.0;
                final_ask = mid + min_spread / 2.0;
                warn!("🔧 Applied emergency symmetric quotes: bid={:.6} ask={:.6}", final_bid, final_ask);
            }
            
            // Estimate fill probabilities for our intended prices
            let bid_fill_prob = self.order_book_analyzer.estimate_fill_probability(final_bid, true);
            let ask_fill_prob = self.order_book_analyzer.estimate_fill_probability(final_ask, false);
            
            // DYNAMIC QUOTE ADJUSTMENT: Use fill probability to optimize pricing
            let bid_adjustment_factor = if bid_fill_prob < 0.05 {
                // Bid too wide - tighten it by moving up
                1.0001 // Move bid up by 1 bp
            } else if bid_fill_prob > 0.9 {
                // Bid too aggressive - widen it by moving down  
                0.9999 // Move bid down by 1 bp
            } else {
                1.0 // Keep as is
            };
            
            let ask_adjustment_factor = if ask_fill_prob < 0.05 {
                // Ask too wide - tighten it by moving down
                0.9999 // Move ask down by 1 bp
            } else if ask_fill_prob > 0.9 {
                // Ask too aggressive - widen it by moving up
                1.0001 // Move ask up by 1 bp
            } else {
                1.0 // Keep as is
            };
            
            // Apply dynamic adjustments 
            final_bid *= bid_adjustment_factor;
            final_ask *= ask_adjustment_factor;
            
            if bid_adjustment_factor != 1.0 || ask_adjustment_factor != 1.0 {
                info!("🎯 DYNAMIC ADJUSTMENT: bid={:.4}x ask={:.4}x (probs: bid={:.1}% ask={:.1}%)", 
                      bid_adjustment_factor, ask_adjustment_factor, 
                      bid_fill_prob * 100.0, ask_fill_prob * 100.0);
            }
            
            // Log unified inventory management through quote skewing when significant
            let inventory_ratio = self.cur_position / self.max_absolute_position_size;
            if inventory_ratio.abs() > 0.3 {
                info!("🎯 UNIFIED INVENTORY MGMT: ratio={:.1}%, bid skew={:.1}bps, ask skew={:.1}bps (quote skewing only)", 
                      inventory_ratio * 100.0,
                      ((final_bid - opt_bid) / self.latest_mid_price) * 10000.0,
                      ((final_ask - opt_ask) / self.latest_mid_price) * 10000.0);
            }
            
            info!(
                "📊 Order Book Analysis: Mid=${:.4} (vs AllMids=${:.4}), Imbalance={:.3}, Fill Probs: Bid={:.1}% Ask={:.1}%",
                ob_mid, self.latest_mid_price, imbalance, bid_fill_prob * 100.0, ask_fill_prob * 100.0
            );
            
            // CRITICAL VALIDATION: Log quote direction
            info!("🎯 QUOTES: BID=${:.4} < MID=${:.4} < ASK=${:.4}, Spread={:.1}bps", 
                  final_bid, self.latest_mid_price, final_ask, 
                  ((final_ask - final_bid) / self.latest_mid_price) * 10000.0);
            
            if has_large_orders {
                info!("⚠️  Large orders detected - potential market impact");
            }
            
            (final_bid, final_ask)
        } else {
            // Fallback to original logic with consolidated adjustments and inventory skewing
            let base_half_spread = (self.latest_mid_price * final_adjusted_spread as f64) / 10000.0;
            
            // Convert to BPS and apply inventory skewing
            let base_spread_bps = (base_half_spread * 2.0 / self.latest_mid_price) * 10000.0;
            let (skewed_bid, skewed_ask) = self.calculate_skewed_quotes(base_spread_bps);
            
            // CRITICAL: Robust protection against inverted quotes
            let min_spread = self.latest_mid_price * 0.0001; // 1 basis point minimum spread
            let mut final_bid = skewed_bid;
            let mut final_ask = skewed_ask;
            
            // Ensure proper ordering with sufficient spread
            if final_bid >= final_ask - min_spread {
                warn!("🚨 Fallback quotes would invert: bid={:.6} ask={:.6}, fixing...", final_bid, final_ask);
                let mid = self.latest_mid_price;
                final_bid = mid - min_spread / 2.0;
                final_ask = mid + min_spread / 2.0;
                warn!("🔧 Applied emergency symmetric quotes: bid={:.6} ask={:.6}", final_bid, final_ask);
            }
            
            // Log unified inventory management through quote skewing when significant
            let inventory_ratio = self.cur_position / self.max_absolute_position_size;
            if inventory_ratio.abs() > 0.3 {
                let base_bid = self.latest_mid_price - base_half_spread;
                let base_ask = self.latest_mid_price + base_half_spread;
                info!("🎯 UNIFIED INVENTORY MGMT: ratio={:.1}%, bid skew={:.1}bps, ask skew={:.1}bps (quote skewing only)", 
                      inventory_ratio * 100.0,
                      ((final_bid - base_bid) / self.latest_mid_price) * 10000.0,
                      ((final_ask - base_ask) / self.latest_mid_price) * 10000.0);
            }
            
            // CRITICAL VALIDATION: Log quote direction
            info!("🎯 QUOTES: BID=${:.4} < MID=${:.4} < ASK=${:.4}, Spread={:.1}bps", 
                  final_bid, self.latest_mid_price, final_ask, 
                  ((final_ask - final_bid) / self.latest_mid_price) * 10000.0);
            
            (final_bid, final_ask)
        };
        
        // Add minimum spread protection BEFORE truncation
        let min_spread_dollars = self.latest_mid_price * 0.0003; // 3 bps minimum spread
        
        // NOTE: For better price precision and fewer rounding issues, consider changing
        // decimals from 1 to 2 in src/bin/market_maker.rs (line ~23)
        let (mut bid_price, mut ask_price) = if (ask_price - bid_price) < min_spread_dollars {
            let adjustment = (min_spread_dollars - (ask_price - bid_price)) / 2.0;
            info!("📏 Applied minimum spread protection: adjusted by {:.6}", adjustment);
            (bid_price - adjustment, ask_price + adjustment)
        } else {
            (bid_price, ask_price)
        };

        // FIXED: Correct truncation logic - round bids DOWN and asks UP to maintain spread
        let pre_truncate_bid = bid_price;
        let pre_truncate_ask = ask_price;
        bid_price = truncate_float(bid_price, self.decimals, false); // Round bid DOWN (less aggressive)
        ask_price = truncate_float(ask_price, self.decimals, true);  // Round ask UP (less aggressive)
        
        // Debug: Log truncation effect to identify precision issues
        if (bid_price - pre_truncate_bid).abs() > 0.01 || (ask_price - pre_truncate_ask).abs() > 0.01 {
            info!("📐 TRUNCATION EFFECT: bid {:.4} → {:.4} ({:+.1}bps), ask {:.4} → {:.4} ({:+.1}bps)", 
                  pre_truncate_bid, bid_price, ((bid_price - pre_truncate_bid) / self.latest_mid_price) * 10000.0,
                  pre_truncate_ask, ask_price, ((ask_price - pre_truncate_ask) / self.latest_mid_price) * 10000.0);
        }

        // Emergency fallback: if still too tight after conservative rounding
        if (ask_price - bid_price) < min_spread_dollars {
            bid_price = truncate_float(bid_price - min_spread_dollars / 2.0, self.decimals, false);
            ask_price = truncate_float(ask_price + min_spread_dollars / 2.0, self.decimals, true);
            info!("🔧 Applied emergency spread widening after truncation");
        }

        // CRITICAL FINAL VALIDATION: Prevent inverted quotes at all costs
        if bid_price >= ask_price {
            error!("🚨 CRITICAL: Inverted quotes detected after truncation! bid={:.6} >= ask={:.6}", bid_price, ask_price);
            let mid = self.latest_mid_price;
            let min_spread = mid * 0.0001; // 1 basis point minimum
            bid_price = mid - min_spread / 2.0;
            ask_price = mid + min_spread / 2.0;
            error!("🔧 Fixed with emergency quotes: bid={:.6} ask={:.6}", bid_price, ask_price);
        }

        // Get inventory-adjusted limits
        let (max_buy_size, max_sell_size) = self.get_inventory_adjusted_limits();
        
        // Determine amounts we can put on the book without exceeding risk and inventory limits
        let lower_order_amount = risk_adjusted_max_size
            .min(max_buy_size)
            .min(adjusted_target_liquidity)
            .max(0.0);

        let upper_order_amount = risk_adjusted_max_size
            .min(max_sell_size)
            .min(adjusted_target_liquidity)
            .max(0.0);

        // Log predictive adverse selection warnings
        if predictive_as_risk > 0.5 {
            let momentum_risk = self.adverse_selection_monitor.calculate_momentum_risk();
            let (_, _, _, _, predictive_adj, _) = self.adverse_selection_monitor.get_metrics();
            warn!("🔮 PREDICTIVE AS WARNING: Risk={:.1}%, Momentum={:.1}%, Spread Adj={:.2}x", 
                  predictive_as_risk * 100.0, momentum_risk * 100.0, predictive_adj);
        }

        // Log risk metrics and performance periodically
        static mut RISK_LOG_COUNTER: usize = 0;
        unsafe {
            RISK_LOG_COUNTER += 1;
            if RISK_LOG_COUNTER % 50 == 0 {
                let inventory_ratio = self.cur_position / self.max_absolute_position_size;
                let (max_buy, max_sell) = self.get_inventory_adjusted_limits();
                let performance = self.pnl_tracker.get_performance_metrics();
                let vol_metrics = self.volatility_estimator.get_metrics();
                let vol_estimate = vol_metrics.current_volatility;
                let adverse_score = self.adverse_selection_monitor.adverse_selection_score();
                let imbalance = self.order_book_analyzer.order_book_imbalance();
                
                // Structured logging for comprehensive performance data
                StructuredLogger::log_performance(&performance, &self.risk_manager);
                StructuredLogger::log_market_data(self.latest_mid_price, vol_estimate, imbalance, adverse_score);
                
                // Traditional logging for human readability
                info!(
                    "💰 Risk Metrics: Capital=${:.2}, Daily PnL=${:.2}, Heat={:.1}%, VaR95=${:.2}, Exposure Factor={:.1}%",
                    self.risk_manager.current_capital,
                    self.risk_manager.daily_pnl,
                    self.risk_manager.current_heat * 100.0,
                    self.risk_manager.calculate_var_95(),
                    exposure_factor * 100.0
                );
                
                info!(
                    "📊 Volatility Metrics: Current={:.1}%, Fast={:.1}%, Medium={:.1}%, Slow={:.1}%, Base={:.1}%, Regime={}, Intraday={:.1}x, Observations={}, Multiplier={:.1}x",
                    vol_metrics.current_volatility * 100.0,
                    vol_metrics.fast_ewma_vol * 100.0,
                    vol_metrics.medium_ewma_vol * 100.0,
                    vol_metrics.slow_ewma_vol * 100.0,
                    self.base_volatility * 100.0,
                    if vol_metrics.vol_regime_high { "HIGH" } else { "NORMAL" },
                    vol_metrics.intraday_multiplier,
                    vol_metrics.returns_count,
                    self.volatility_estimator.get_volatility_spread_multiplier(self.base_volatility)
                );
                
                info!(
                    "📦 Inventory: Position={:.4}, Ratio={:.1}%, Max Buy={:.4}, Max Sell={:.4}",
                    self.cur_position,
                    inventory_ratio * 100.0,
                    max_buy,
                    max_sell
                );
                
                if performance.total_trades > 10 {
                    info!(
                        "📈 Risk Metrics: Sharpe={:.2}, Max DD={:.1}%, Profit Factor={:.2}",
                        performance.sharpe_ratio,
                        performance.max_drawdown * 100.0,
                        performance.profit_factor
                    );
                }
            }
        }

        // Determine if we need to cancel the resting order and put a new order up due to deviation
        let bid_change = (lower_order_amount - self.lower_resting.position).abs() > EPSILON
            || bps_diff(bid_price, self.lower_resting.price) > self.max_bps_diff;
        let ask_change = (upper_order_amount - self.upper_resting.position).abs() > EPSILON
            || bps_diff(ask_price, self.upper_resting.price) > self.max_bps_diff;

        // Consider cancelling
        // TODO: Don't block on cancels
        if self.lower_resting.oid != 0 && self.lower_resting.position > EPSILON && bid_change {
            let cancel = self
                .attempt_cancel(self.asset.clone(), self.lower_resting.oid)
                .await;
            // If we were unable to cancel, it means we got a fill, so wait until we receive that event to do anything
            if !cancel {
                return;
            }
            info!("Cancelled buy order: {:?}", self.lower_resting);
        }

        if self.upper_resting.oid != 0 && self.upper_resting.position > EPSILON && ask_change {
            let cancel = self
                .attempt_cancel(self.asset.clone(), self.upper_resting.oid)
                .await;
            if !cancel {
                return;
            }
            info!("Cancelled sell order: {:?}", self.upper_resting);
        }

        // Consider putting a new order up
        if lower_order_amount > EPSILON && bid_change {
            let (amount_resting, oid) = self
                .place_order(self.asset.clone(), lower_order_amount, bid_price, true)
                .await;

            self.lower_resting.oid = oid;
            self.lower_resting.position = amount_resting;
            self.lower_resting.price = bid_price;

            if amount_resting > EPSILON {
                info!(
                    "✅ BUY order placed: {amount_resting} {} resting at {bid_price}",
                    self.asset.clone()
                );
            }
        }

        if upper_order_amount > EPSILON && ask_change {
            let (amount_resting, oid) = self
                .place_order(self.asset.clone(), upper_order_amount, ask_price, false)
                .await;
            self.upper_resting.oid = oid;
            self.upper_resting.position = amount_resting;
            self.upper_resting.price = ask_price;

            if amount_resting > EPSILON {
                info!(
                    "✅ SELL order placed: {amount_resting} {} resting at {ask_price}",
                    self.asset.clone()
                );
            }
        }
    }

    async fn cancel_all_orders(&mut self) {
        if self.lower_resting.oid != 0 && self.lower_resting.position > EPSILON {
            self.attempt_cancel(self.asset.clone(), self.lower_resting.oid).await;
            self.lower_resting = MarketMakerRestingOrder {
                oid: 0,
                position: 0.0,
                price: -1.0,
            };
        }
        
        if self.upper_resting.oid != 0 && self.upper_resting.position > EPSILON {
            self.attempt_cancel(self.asset.clone(), self.upper_resting.oid).await;
            self.upper_resting = MarketMakerRestingOrder {
                oid: 0,
                position: 0.0,
                price: -1.0,
            };
        }
    }

    /// Calculate inventory-skewed quotes to manage position drift
    fn calculate_skewed_quotes(&self, base_spread: f64) -> (f64, f64) {
        let inventory_ratio = self.cur_position / self.max_absolute_position_size;
        
        // UNIFIED INVENTORY MANAGEMENT: Only quote skewing, no crossing spread
        // Make skewing MUCH more aggressive - up to 100+ bps at extreme inventory
        
        // Progressive skewing based on inventory ratio
        let skew_bps = if inventory_ratio.abs() > 0.8 {
            // Extreme inventory (80%+): Very aggressive skewing up to 150 bps
            let extreme_factor = (inventory_ratio.abs() - 0.8) / 0.2; // 0 to 1 as we go from 80% to 100%
            let base_extreme_skew = 100.0; // 100 bps base at 80%
            let max_extreme_skew = 150.0; // 150 bps max at 100%
            let extreme_skew = base_extreme_skew + (max_extreme_skew - base_extreme_skew) * extreme_factor;
            extreme_skew * inventory_ratio.signum()
        } else if inventory_ratio.abs() > 0.5 {
            // High inventory (50-80%): Aggressive skewing 30-100 bps
            let high_factor = (inventory_ratio.abs() - 0.5) / 0.3; // 0 to 1 as we go from 50% to 80%
            let base_high_skew = 30.0; // 30 bps at 50%
            let max_high_skew = 100.0; // 100 bps at 80%
            let high_skew = base_high_skew + (max_high_skew - base_high_skew) * high_factor;
            high_skew * inventory_ratio.signum()
        } else if inventory_ratio.abs() > 0.25 {
            // Moderate inventory (25-50%): Progressive skewing 5-30 bps
            let mid_factor = (inventory_ratio.abs() - 0.25) / 0.25; // 0 to 1 as we go from 25% to 50%
            let base_mid_skew = 5.0; // 5 bps at 25%
            let max_mid_skew = 30.0; // 30 bps at 50%
            let mid_skew = base_mid_skew + (max_mid_skew - base_mid_skew) * mid_factor;
            mid_skew * inventory_ratio.signum()
        } else {
            // Low inventory (<25%): Minimal skewing up to 5 bps
            5.0 * inventory_ratio // Linear skewing up to 5 bps
        };
        
        // Log inventory management when skewing is significant
        if skew_bps.abs() > 5.0 {
            StructuredLogger::log_inventory_management(
                self.cur_position, 
                self.max_absolute_position_size, 
                skew_bps, 
                &format!("aggressive_quote_skewing_{:.0}pct", inventory_ratio.abs() * 100.0)
            );
            
            info!("🎯 AGGRESSIVE INVENTORY SKEWING: position={:.4}/{:.4} ({:.1}%), skew={:.1}bps", 
                  self.cur_position, self.max_absolute_position_size, inventory_ratio * 100.0, skew_bps);
        }
        
        // CRITICAL FIX: Ensure spreads never invert
        // When we have long position (inventory_ratio > 0):
        // - We want to DISCOURAGE more buying by widening the bid (lowering bid price)
        // - We want to ENCOURAGE selling by tightening the ask (lowering ask price)
        // When we have short position (inventory_ratio < 0):
        // - We want to ENCOURAGE buying by tightening the bid (raising bid price)  
        // - We want to DISCOURAGE selling by widening the ask (raising ask price)
        
        let half_spread_bps = base_spread / 2.0;
        let min_spread_bps = 3.0; // Minimum 3 bps on each side to prevent inversion with aggressive skewing
        
        // Apply skewing with bounds checking to prevent inversion
        let bid_spread_bps = (half_spread_bps + skew_bps).max(min_spread_bps); // Widen bid when long
        let ask_spread_bps = (half_spread_bps - skew_bps).max(min_spread_bps); // Tighten ask when long, but never below minimum
        
        let bid_price = self.latest_mid_price * (1.0 - bid_spread_bps / 10000.0);
        let ask_price = self.latest_mid_price * (1.0 + ask_spread_bps / 10000.0);
        
        // CRITICAL VALIDATION: Double-check bid < ask with minimum spread enforcement
        let min_dollar_spread = self.latest_mid_price * min_spread_bps * 2.0 / 10000.0; // Total minimum spread in dollars
        
        if bid_price >= ask_price - min_dollar_spread {
            // If quotes would invert or spread is too tight, use symmetric quotes with minimum spread
            warn!("🚨 QUOTE INVERSION PREVENTED (aggressive skewing): bid={:.4} ask={:.4} skew={:.1}bps", bid_price, ask_price, skew_bps);
            let mid = self.latest_mid_price;
            let safe_half_spread = (min_dollar_spread / 2.0).max(half_spread_bps * mid / 10000.0);
            return (mid - safe_half_spread, mid + safe_half_spread);
        }
        
        // Additional safety check: ensure minimum spread
        let actual_spread = ask_price - bid_price;
        if actual_spread < min_dollar_spread {
            let adjustment = (min_dollar_spread - actual_spread) / 2.0;
            return (bid_price - adjustment, ask_price + adjustment);
        }
        
        (bid_price, ask_price)
    }
    
    /// Unified inventory management - ONLY through quote skewing, no active reduction
    async fn manage_inventory(&mut self) {
        let inventory_ratio = self.cur_position.abs() / self.max_absolute_position_size;
        
        // UNIFIED APPROACH: Only log inventory status, all management is done through quote skewing
        if inventory_ratio > 0.8 {
            info!("🎯 EXTREME INVENTORY: {:.1}% of limit - managing through aggressive quote skewing only", 
                  inventory_ratio * 100.0);
        } else if inventory_ratio > 0.5 {
            info!("📊 HIGH INVENTORY: {:.1}% of limit - managing through quote skewing", 
                  inventory_ratio * 100.0);
        }
        
        // No active orders placed - all inventory management is handled by calculate_skewed_quotes()
        // This ensures we never cross the spread, only adjust our passive quotes
    }

    /// Calculate inventory-aware position limits
    fn get_inventory_adjusted_limits(&self) -> (f64, f64) {
        let inventory_ratio = self.cur_position / self.max_absolute_position_size;
        
        // Reduce available size as we approach position limits
        let inventory_buffer = 1.0 - inventory_ratio.abs();
        
        // Long position reduces our ability to buy more
        let max_buy_size = if self.cur_position > 0.0 {
            (self.max_absolute_position_size - self.cur_position) * inventory_buffer
        } else {
            self.max_absolute_position_size * inventory_buffer
        };
        
        // Short position reduces our ability to sell more
        let max_sell_size = if self.cur_position < 0.0 {
            (self.max_absolute_position_size + self.cur_position) * inventory_buffer
        } else {
            self.max_absolute_position_size * inventory_buffer
        };
        
        (max_buy_size.max(0.0), max_sell_size.max(0.0))
    }

    /// Get comprehensive performance summary
    pub fn get_performance_summary(&self) -> PerformanceMetrics {
        self.pnl_tracker.get_performance_metrics()
    }
    
    /// Log current performance metrics on demand
    pub fn log_performance_metrics(&self) {
        let performance = self.pnl_tracker.get_performance_metrics();
        
        info!("=== 📊 PERFORMANCE REPORT ===");
        info!("💰 Total PnL: ${:.2} (ROI: {:.2}%)", performance.total_pnl, performance.roi * 100.0);
        info!("💵 Realized PnL: ${:.2}", performance.realized_pnl);
        info!("💸 Unrealized PnL: ${:.2}", performance.unrealized_pnl);
        info!("📈 Trades: {} total ({} wins, {} losses)", performance.total_trades, performance.winning_trades, performance.losing_trades);
        info!("🎯 Win Rate: {:.1}%", performance.win_rate * 100.0);
        info!("💼 Average Trade PnL: ${:.2}", performance.average_trade_pnl);
        info!("📊 Total Volume: {:.2}", performance.total_volume);
        
        if performance.total_trades > 1 {
            info!("📉 Max Drawdown: {:.1}%", performance.max_drawdown * 100.0);
            info!("⚡ Profit Factor: {:.2}", performance.profit_factor);
            
            if performance.total_trades > 10 {
                info!("📐 Sharpe Ratio: {:.2}", performance.sharpe_ratio);
            }
        }
        
        info!("=== END PERFORMANCE REPORT ===");
    }
    
    /// Reset PnL tracker (useful for backtesting or strategy restarts)
    pub fn reset_pnl_tracker(&mut self) {
        self.pnl_tracker = PnLTracker::new(self.pnl_tracker.initial_capital);
    }
    
    /// Get simulation statistics (only meaningful in simulation mode)
    pub fn get_simulation_stats(&self) -> SimulationStats {
        let total_orders = self.virtual_orders.len();
        let filled_orders = self.virtual_orders.iter()
            .filter(|o| matches!(o.status, VirtualOrderStatus::Filled))
            .count();
        let cancelled_orders = self.virtual_orders.iter()
            .filter(|o| matches!(o.status, VirtualOrderStatus::Cancelled))
            .count();
        let pending_orders = self.virtual_orders.iter()
            .filter(|o| matches!(o.status, VirtualOrderStatus::Pending | VirtualOrderStatus::PartiallyFilled))
            .count();
        
        let total_volume_traded = self.virtual_orders.iter()
            .filter(|o| matches!(o.status, VirtualOrderStatus::Filled))
            .map(|o| o.size)
            .sum();
        
        SimulationStats {
            total_orders,
            filled_orders,
            cancelled_orders,
            pending_orders,
            total_volume_traded,
            fill_rate: if total_orders > 0 { filled_orders as f64 / total_orders as f64 } else { 0.0 },
            performance_metrics: self.get_performance_summary(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct SimulationStats {
    pub total_orders: usize,
    pub filled_orders: usize,
    pub cancelled_orders: usize,
    pub pending_orders: usize,
    pub total_volume_traded: f64,
    pub fill_rate: f64,
    pub performance_metrics: PerformanceMetrics,
}


