//! Contains the StateVector (Z_t) for the HJB strategy.

use crate::{BookAnalysis, OrderBook};
use super::models::OnlineAdverseSelectionModel;
use super::tuning::ConstrainedTuningParams;

/// State Vector (Z_t)
#[derive(Debug, Clone)]
pub struct StateVector {
    pub mid_price: f64,
    pub inventory: f64,
    pub adverse_selection_estimate: f64,
    pub market_spread_bps: f64,
    pub lob_imbalance: f64,
    pub volatility_ema_bps: f64,
    pub previous_mid_price: f64,
    pub trade_flow_ema: f64,
}

impl StateVector {
    pub fn new() -> Self {
        Self {
            mid_price: 0.0,
            inventory: 0.0,
            adverse_selection_estimate: 0.0,
            market_spread_bps: 0.0,
            lob_imbalance: 0.0,
            volatility_ema_bps: 10.0,
            previous_mid_price: 0.0,
            trade_flow_ema: 0.0,
        }
    }

    pub fn update(
        &mut self,
        mid_price: f64,
        inventory: f64,
        _book_analysis: Option<&BookAnalysis>,
        order_book: Option<&OrderBook>,
        tuning_params: &ConstrainedTuningParams,
        online_model: &mut OnlineAdverseSelectionModel,
    ) {
        self.previous_mid_price = self.mid_price;
        self.mid_price = mid_price;
        self.inventory = inventory;

        if let Some(book) = order_book {
            if let Some(spread_bps) = book.spread_bps() {
                self.market_spread_bps = spread_bps;
            }
            if !book.bids.is_empty() && !book.asks.is_empty() {
                if let (Ok(bid_vol), Ok(ask_vol)) = (
                    book.bids[0].sz.parse::<f64>(),
                    book.asks[0].sz.parse::<f64>(),
                ) {
                    let total_vol = bid_vol + ask_vol;
                    if total_vol > 1e-10 {
                        self.lob_imbalance = bid_vol / total_vol;
                    } else {
                        self.lob_imbalance = 0.5;
                    }
                } else {
                    self.lob_imbalance = 0.5;
                }
            } else {
                self.lob_imbalance = 0.5;
            }
            self.update_adverse_selection(tuning_params, online_model);
        } else {
            self.lob_imbalance = 0.5;
        }
    }

    fn update_adverse_selection(
        &mut self,
        tuning_params: &ConstrainedTuningParams,
        online_model: &mut OnlineAdverseSelectionModel,
    ) {
        online_model.update_feature_stats(self);
        let raw_prediction = online_model.predict(self);
        online_model.record_observation(self, self.mid_price);
        let lambda = tuning_params.adverse_selection_lambda;
        self.adverse_selection_estimate =
            lambda * raw_prediction + (1.0 - lambda) * self.adverse_selection_estimate;
    }

    pub fn get_adverse_selection_adjustment(
        &self,
        base_spread_bps: f64,
        adjustment_factor: f64,
    ) -> f64 {
        -self.adverse_selection_estimate * base_spread_bps * adjustment_factor
    }

    pub fn get_inventory_risk_multiplier(&self, max_inventory: f64) -> f64 {
        if max_inventory <= 0.0 {
            return 1.0;
        }
        let inventory_ratio = (self.inventory.abs() / max_inventory).min(1.0);
        1.0 + inventory_ratio.powi(2)
    }

    pub fn get_inventory_urgency(&self, max_inventory: f64) -> f64 {
        if max_inventory <= 0.0 {
            return 0.0;
        }
        let inventory_ratio = (self.inventory.abs() / max_inventory).min(1.0);
        inventory_ratio.powi(3)
    }

    pub fn is_market_favorable(&self, max_spread_bps: f64) -> bool {
        self.market_spread_bps > 0.0
            && self.market_spread_bps < max_spread_bps
            && self.lob_imbalance > 0.1
            && self.lob_imbalance < 0.9
    }

    pub fn to_log_string(&self) -> String {
        format!(
            "StateVector[S={:.2}, Q={:.4}, μ̂={:.4}, Δ={:.1}bps, I={:.3}, σ̂={:.2}bps, TF_EMA={:.3}]",
            self.mid_price,
            self.inventory,
            self.adverse_selection_estimate,
            self.market_spread_bps,
            self.lob_imbalance,
            self.volatility_ema_bps,
            self.trade_flow_ema
        )
    }

    pub fn update_trade_flow_ema(
        &mut self,
        trades: &Vec<crate::Trade>,
        tuning_params: &ConstrainedTuningParams,
    ) {
        if trades.is_empty() {
            return;
        }
        let lambda = tuning_params.adverse_selection_lambda;
        let decay = 1.0 - lambda;
        for trade in trades {
            let signal = if trade.side == "A" { 1.0 } else { -1.0 };
            self.trade_flow_ema = lambda * signal + decay * self.trade_flow_ema;
        }
    }
}

impl Default for StateVector {
    fn default() -> Self {
        let mut s = Self::new();
        s.trade_flow_ema = 0.0;
        s
    }
}
