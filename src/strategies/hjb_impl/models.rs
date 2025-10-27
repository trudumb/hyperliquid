//! Contains the OnlineAdverseSelectionModel for the HJB strategy.

use super::state::StateVector;
use log::info;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// Online Linear Regression Model for Adverse Selection Estimation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OnlineAdverseSelectionModel {
    pub weights: Vec<f64>,
    pub learning_rate: f64,
    pub lookback_ticks: usize,
    pub observation_buffer: VecDeque<(Vec<f64>, f64)>,
    pub buffer_capacity: usize,
    pub enable_learning: bool,
    pub update_count: usize,
    pub mean_absolute_error: f64,
    pub mae_decay: f64,
    pub feature_stats: Vec<(f64, f64, f64)>,
}

impl Default for OnlineAdverseSelectionModel {
    fn default() -> Self {
        Self {
            weights: vec![0.0, 0.4, 0.1, -0.05, 0.02],
            learning_rate: 0.001,
            lookback_ticks: 10,
            observation_buffer: VecDeque::with_capacity(100),
            buffer_capacity: 100,
            enable_learning: true,
            update_count: 0,
            mean_absolute_error: 0.0,
            mae_decay: 0.99,
            feature_stats: vec![(0.0, 0.0, 0.0); 4],
        }
    }
}

impl OnlineAdverseSelectionModel {
    pub fn update_feature_stats(&mut self, state: &StateVector) {
        let raw_features = vec![
            state.trade_flow_ema,
            state.lob_imbalance - 0.5,
            state.market_spread_bps,
            state.volatility_ema_bps,
        ];

        for i in 0..raw_features.len() {
            let x = raw_features[i];
            let (count, mean, m2) = &mut self.feature_stats[i];
            *count += 1.0;
            let delta = x - *mean;
            *mean += delta / *count;
            let delta2 = x - *mean;
            *m2 += delta * delta2;
        }
    }

    fn get_normalized_features(&self, state: &StateVector) -> Vec<f64> {
        let raw_features = vec![
            state.trade_flow_ema,
            state.lob_imbalance - 0.5,
            state.market_spread_bps,
            state.volatility_ema_bps,
        ];
        let mut normalized_features = vec![1.0];
        for i in 0..raw_features.len() {
            let x = raw_features[i];
            let (count, mean, m2) = &self.feature_stats[i];
            let (mean, std_dev) = if *count < 2.0 {
                (0.0, 1.0)
            } else {
                let variance = *m2 / (*count - 1.0);
                let std_dev = variance.sqrt().max(1e-6);
                (*mean, std_dev)
            };
            normalized_features.push((x - mean) / std_dev);
        }
        normalized_features
    }

    pub fn predict(&self, state: &StateVector) -> f64 {
        let features = self.get_normalized_features(state);
        self.weights
            .iter()
            .zip(features.iter())
            .map(|(w, x)| w * x)
            .sum()
    }

    pub fn record_observation(&mut self, state: &StateVector, mid_price: f64) {
        let features = self.get_normalized_features(state);
        self.observation_buffer.push_back((features, mid_price));
        if self.observation_buffer.len() > self.buffer_capacity {
            self.observation_buffer.pop_front();
        }
    }

    pub fn update(&mut self, current_mid_price: f64) {
        if !self.enable_learning {
            return;
        }
        if self.observation_buffer.len() <= self.lookback_ticks {
            return;
        }
        let lookback_idx = self.observation_buffer.len() - self.lookback_ticks - 1;
        if let Some((features, old_mid_price)) = self.observation_buffer.get(lookback_idx) {
            let actual_change_bps = if *old_mid_price > 0.0 {
                ((current_mid_price - old_mid_price) / old_mid_price) * 10000.0
            } else {
                0.0
            };
            let predicted_change_bps: f64 = self
                .weights
                .iter()
                .zip(features.iter())
                .map(|(w, x)| w * x)
                .sum();
            let error = predicted_change_bps - actual_change_bps;
            let abs_error = error.abs();
            if self.update_count == 0 {
                self.mean_absolute_error = abs_error;
            } else {
                self.mean_absolute_error = self.mae_decay * self.mean_absolute_error
                    + (1.0 - self.mae_decay) * abs_error;
            }
            for i in 0..self.weights.len() {
                self.weights[i] -= self.learning_rate * error * features[i];
            }
            self.update_count += 1;
            if self.update_count % 100 == 0 {
                info!(
                    "Online Adverse Selection Model Update #{}: MAE={:.4}bps, Weights={:?}",
                    self.update_count, self.mean_absolute_error, self.weights
                );
            }
        }
    }

    pub fn get_stats(&self) -> String {
        format!(
            "OnlineModel[updates={}, MAE={:.4}bps, lr={:.6}, enabled={}]",
            self.update_count,
            self.mean_absolute_error,
            self.learning_rate,
            self.enable_learning
        )
    }

    pub fn reset(&mut self) {
        *self = Self::default();
    }
}
