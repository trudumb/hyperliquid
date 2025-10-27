//! Contains the CachedVolatilityEstimate struct for the HJB strategy.

/// Cached volatility estimate
#[derive(Debug, Clone)]
pub struct CachedVolatilityEstimate {
    pub volatility_bps: f64,
    pub vol_5th_percentile: f64,
    pub vol_95th_percentile: f64,
    pub param_std_devs: (f64, f64, f64),
    pub volatility_std_dev_bps: f64,
    pub last_update_time: f64,
}

impl Default for CachedVolatilityEstimate {
    fn default() -> Self {
        Self {
            volatility_bps: 100.0,
            vol_5th_percentile: 80.0,
            vol_95th_percentile: 120.0,
            param_std_devs: (0.1, 0.01, 0.1),
            volatility_std_dev_bps: 10.0,
            last_update_time: 0.0,
        }
    }
}
