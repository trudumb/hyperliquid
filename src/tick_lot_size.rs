use crate::{meta::AssetMeta, prelude::Result, Error};

/// Constants for price validation
const MAX_DECIMALS_PERP: u32 = 6;
const MAX_DECIMALS_SPOT: u32 = 8;
const MAX_SIGNIFICANT_FIGURES: u32 = 5;

/// Asset type for determining decimal limits
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AssetType {
    Perp,
    Spot,
}

/// Helper struct for tick and lot size validation
#[derive(Debug, Clone)]
pub struct TickLotValidator {
    pub asset: String,
    pub asset_type: AssetType,
    pub sz_decimals: u32,
}

impl TickLotValidator {
    /// Create a new validator from asset metadata
    pub fn new(asset: String, asset_type: AssetType, sz_decimals: u32) -> Self {
        Self {
            asset,
            asset_type,
            sz_decimals,
        }
    }

    /// Create a validator from AssetMeta (for perps)
    pub fn from_asset_meta(asset_meta: &AssetMeta, asset_type: AssetType) -> Self {
        Self::new(
            asset_meta.name.clone(),
            asset_type,
            asset_meta.sz_decimals,
        )
    }

    /// Get the maximum allowed decimal places for prices
    pub fn max_price_decimals(&self) -> u32 {
        let base_max = match self.asset_type {
            AssetType::Perp => MAX_DECIMALS_PERP,
            AssetType::Spot => MAX_DECIMALS_SPOT,
        };
        base_max.saturating_sub(self.sz_decimals)
    }

    /// Validate a price according to Hyperliquid's rules
    pub fn validate_price(&self, price: f64) -> Result<()> {
        if !price.is_finite() || price <= 0.0 {
            return Err(Error::InvalidInput(format!(
                "Price must be positive and finite: {}",
                price
            )));
        }

        // Check if it's an integer price (always valid regardless of significant figures)
        if price.fract() == 0.0 {
            return Ok(());
        }

        // Check significant figures
        let significant_figures = count_significant_figures(price);
        if significant_figures > MAX_SIGNIFICANT_FIGURES {
            return Err(Error::InvalidInput(format!(
                "Price {} has {} significant figures, maximum allowed is {}",
                price, significant_figures, MAX_SIGNIFICANT_FIGURES
            )));
        }

        // Check decimal places
        let decimal_places = count_decimal_places(price);
        let max_decimals = self.max_price_decimals();
        if decimal_places > max_decimals {
            return Err(Error::InvalidInput(format!(
                "Price {} has {} decimal places, maximum allowed is {} (MAX_DECIMALS {} - szDecimals {})",
                price, decimal_places, max_decimals, 
                match self.asset_type {
                    AssetType::Perp => MAX_DECIMALS_PERP,
                    AssetType::Spot => MAX_DECIMALS_SPOT,
                },
                self.sz_decimals
            )));
        }

        Ok(())
    }

    /// Validate a size according to Hyperliquid's rules
    pub fn validate_size(&self, size: f64) -> Result<()> {
        if !size.is_finite() || size <= 0.0 {
            return Err(Error::InvalidInput(format!(
                "Size must be positive and finite: {}",
                size
            )));
        }

        let decimal_places = count_decimal_places(size);
        if decimal_places > self.sz_decimals {
            return Err(Error::InvalidInput(format!(
                "Size {} has {} decimal places, maximum allowed is {} (szDecimals for {})",
                size, decimal_places, self.sz_decimals, self.asset
            )));
        }

        Ok(())
    }

    /// Round a price to be valid according to the tick size rules
    /// This ensures the price has at most MAX_SIGNIFICANT_FIGURES significant figures
    /// and at most max_price_decimals() decimal places
    pub fn round_price(&self, price: f64, round_up: bool) -> f64 {
        if !price.is_finite() || price <= 0.0 {
            return price;
        }

        // If it's already an integer, return as-is (integers are always valid)
        if price.fract() == 0.0 {
            return price;
        }

        let max_decimals = self.max_price_decimals();
        
        // First, limit by decimal places
        let price_limited_decimals = limit_decimal_places(price, max_decimals, round_up);
        
        // Then, limit by significant figures (but allow integers regardless of sig figs)
        if price_limited_decimals.fract() == 0.0 {
            price_limited_decimals
        } else {
            limit_significant_figures(price_limited_decimals, MAX_SIGNIFICANT_FIGURES, round_up)
        }
    }

    /// Round a size to be valid according to the lot size rules
    pub fn round_size(&self, size: f64, round_up: bool) -> f64 {
        if !size.is_finite() || size <= 0.0 {
            return size;
        }

        limit_decimal_places(size, self.sz_decimals, round_up)
    }
}

/// Count the number of significant figures in a number
/// This is a simplified implementation that works well for typical trading prices
fn count_significant_figures(num: f64) -> u32 {
    if num == 0.0 {
        return 1;
    }

    // Convert to string without scientific notation for typical trading ranges
    let abs_num = num.abs();
    
    // For very small numbers, count digits after leading zeros
    if abs_num < 1.0 {
        let s = format!("{:.15}", abs_num);
        return count_sig_figs_from_decimal(&s);
    }
    
    // For numbers >= 1, count all non-zero digits and zeros between them
    let s = format!("{:.10}", abs_num).trim_end_matches('0').trim_end_matches('.').to_string();
    let digits_only: String = s.chars().filter(|&c| c.is_ascii_digit()).collect();
    
    if digits_only.is_empty() {
        return 1;
    }
    
    // Remove leading zeros and count remaining digits
    let trimmed = digits_only.trim_start_matches('0');
    if trimmed.is_empty() {
        1
    } else {
        trimmed.len() as u32
    }
}

/// Helper function to count significant figures for decimal numbers < 1
fn count_sig_figs_from_decimal(s: &str) -> u32 {
    if let Some(decimal_pos) = s.find('.') {
        let after_decimal = &s[decimal_pos + 1..];
        let trimmed = after_decimal.trim_end_matches('0');
        let mut sig_figs = 0;
        let mut found_nonzero = false;
        
        for ch in trimmed.chars() {
            if ch.is_ascii_digit() {
                if ch != '0' {
                    found_nonzero = true;
                    sig_figs += 1;
                } else if found_nonzero {
                    sig_figs += 1;
                }
            }
        }
        
        sig_figs.max(1)
    } else {
        1
    }
}

/// Count decimal places in a number
/// This implementation works well for typical trading prices and sizes
fn count_decimal_places(num: f64) -> u32 {
    // Handle edge cases
    if num.fract() == 0.0 {
        return 0;
    }
    
    // For typical trading ranges, we can use string representation
    // with enough precision to capture meaningful decimal places
    let abs_num = num.abs();
    
    // Convert to string with sufficient precision for trading
    let s = if abs_num < 0.0001 {
        format!("{:.10}", num)
    } else if abs_num < 1.0 {
        format!("{:.8}", num)
    } else {
        format!("{:.6}", num)
    };
    
    if let Some(decimal_pos) = s.find('.') {
        let after_decimal = &s[decimal_pos + 1..];
        // Remove trailing zeros
        let trimmed = after_decimal.trim_end_matches('0');
        trimmed.len() as u32
    } else {
        0
    }
}

/// Limit a number to a specific number of decimal places
fn limit_decimal_places(num: f64, max_decimals: u32, round_up: bool) -> f64 {
    let multiplier = 10f64.powi(max_decimals as i32);
    let scaled = num * multiplier;
    
    if round_up {
        scaled.ceil() / multiplier
    } else {
        scaled.floor() / multiplier
    }
}

/// Limit a number to a specific number of significant figures
fn limit_significant_figures(num: f64, max_sig_figs: u32, round_up: bool) -> f64 {
    if num == 0.0 {
        return 0.0;
    }
    
    let magnitude = num.abs().log10().floor();
    let shift = max_sig_figs as f64 - 1.0 - magnitude;
    let multiplier = 10f64.powf(shift);
    let scaled = num * multiplier;
    
    let rounded = if round_up {
        scaled.ceil()
    } else {
        scaled.floor()
    };
    
    rounded / multiplier
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_count_significant_figures() {
        assert_eq!(count_significant_figures(1234.5), 5);
        assert_eq!(count_significant_figures(1234.56), 6);
        assert_eq!(count_significant_figures(0.001234), 4);
        assert_eq!(count_significant_figures(0.0012345), 5);
        assert_eq!(count_significant_figures(123456.0), 6);
        assert_eq!(count_significant_figures(0.00076), 2);
        assert_eq!(count_significant_figures(1.001), 4);
    }

    #[test]
    fn test_count_decimal_places() {
        assert_eq!(count_decimal_places(1234.5), 1);
        assert_eq!(count_decimal_places(1234.56), 2);
        assert_eq!(count_decimal_places(0.001234), 6);
        assert_eq!(count_decimal_places(0.0012345), 7);
        assert_eq!(count_decimal_places(123456.0), 0);
        assert_eq!(count_decimal_places(1.001), 3);
    }

    #[test]
    fn test_perp_price_validation() {
        let validator = TickLotValidator::new("ETH".to_string(), AssetType::Perp, 1);
        
        // Valid cases  
        assert!(validator.validate_price(1234.5).is_ok()); // 5 sig figs, 1 decimal
        assert!(validator.validate_price(123456.0).is_ok()); // Integer always valid
        assert!(validator.validate_price(1.2345).is_ok()); // 5 sig figs, 4 decimals (< 5 max)
        
        // Invalid cases
        assert!(validator.validate_price(1234.56).is_err()); // Too many sig figs (6 > 5)
        assert!(validator.validate_price(0.001234).is_err()); // Too many decimals (6 > 5 max for sz_decimals=1)
        assert!(validator.validate_price(0.012345).is_err()); // Too many decimals (6 > 5)
    }

    #[test]
    fn test_spot_price_validation() {
        let validator = TickLotValidator::new("BTC/USDC".to_string(), AssetType::Spot, 2);
        
        // Valid cases
        assert!(validator.validate_price(1234.5).is_ok()); // 5 sig figs, 1 decimal
        assert!(validator.validate_price(123456.0).is_ok()); // Integer always valid
        assert!(validator.validate_price(12.123).is_ok()); // 5 sig figs, 3 decimals
        
        // Max decimals for spot with sz_decimals=2 is 8-2=6
        assert!(validator.validate_price(12.123456).is_err()); // 6 decimals, but too many sig figs (8)
        assert!(validator.validate_price(12.1234567).is_err()); // 7 decimals > 6
    }

    #[test]
    fn test_size_validation() {
        let validator = TickLotValidator::new("ETH".to_string(), AssetType::Perp, 3);
        
        // Valid cases
        assert!(validator.validate_size(1.001).is_ok()); // 3 decimals
        assert!(validator.validate_size(10.123).is_ok()); // 3 decimals
        assert!(validator.validate_size(1.0).is_ok()); // 1 decimal
        
        // Invalid cases
        assert!(validator.validate_size(1.0001).is_err()); // 4 decimals > 3
        assert!(validator.validate_size(0.12345).is_err()); // 5 decimals > 3
    }

    #[test]
    fn test_price_rounding() {
        let validator = TickLotValidator::new("ETH".to_string(), AssetType::Perp, 1);
        
        // Test decimal place limiting (sz_decimals=1, so max_decimals=6-1=5)
        let rounded_down = validator.round_price(1.123456, false);
        let rounded_up = validator.round_price(1.123456, true);
        
        // Should be limited to 5 decimal places
        assert!(count_decimal_places(rounded_down) <= 5);
        assert!(count_decimal_places(rounded_up) <= 5);
        assert!(rounded_down <= 1.123456);
        assert!(rounded_up >= 1.123456);
        
        // Integers should remain unchanged
        assert_eq!(validator.round_price(123456.0, false), 123456.0);
    }

    #[test]
    fn test_size_rounding() {
        let validator = TickLotValidator::new("ETH".to_string(), AssetType::Perp, 3);
        
        let rounded_down = validator.round_size(1.0001, false);
        let rounded_up = validator.round_size(1.0001, true);
        
        // Should be limited to 3 decimal places
        assert!(count_decimal_places(rounded_down) <= 3);
        assert!(count_decimal_places(rounded_up) <= 3);
        assert!(rounded_down <= 1.0001);
        assert!(rounded_up >= 1.0001);
        
        // Check a size that's already valid
        let valid_size = validator.round_size(1.001, false);
        assert!(count_decimal_places(valid_size) <= 3);
    }
}