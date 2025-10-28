// src/bin/check_hype_metadata.rs
// Fixed: no .index, proper type conversion, clean compile

use hyperliquid_rust_sdk::{BaseUrl, InfoClient};
use colored::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("{}", "Hyperliquid HYPE Metadata & Validation Checker".bold().cyan());
    println!("{}", "=".repeat(60).dimmed());

    let info_client = InfoClient::new(None, Some(BaseUrl::Mainnet)).await?;
    let meta = info_client.meta().await?;

    // Find HYPE with its position in the universe array
    let (asset_index, hype) = meta
        .universe
        .iter()
        .enumerate()
        .find(|(_, a)| a.name == "HYPE")
        .ok_or("HYPE not found in universe")?;

    println!("{} Found HYPE at universe[{}]", "Found".green(), asset_index);
    println!("   {}: {}", "Name".bold(), hype.name);
    println!(
        "   {}: {} {}",
        "szDecimals".bold().yellow(),
        hype.sz_decimals,
        "← CRITICAL FOR SIZE".italic()
    );
    println!("   {}: {}", "Max Leverage".bold(), format!("{}x", hype.max_leverage).green());
    println!();

    // === SIZE VALIDATION ===
    println!("{}", "SIZE (sz) VALIDATION".bold().underline());
    println!("Sizes must have ≤ {} decimal place(s).", hype.sz_decimals);

    let valid_sizes = match hype.sz_decimals {
        0 => vec!["1", "10", "500"],
        1 => vec!["0.1", "0.5", "1.0", "2.7"],
        2 => vec!["0.01", "0.05", "1.23", "10.50"],
        3 => vec!["0.001", "0.025", "1.234", "5.678"],
        4 => vec!["0.0001", "0.0025", "1.2345"],
        _ => vec!["1.0"],
    };

    let invalid_sizes = match hype.sz_decimals {
        0 => vec!["0.5", "1.1"],
        1 => vec!["0.01", "1.23"],
        2 => vec!["0.001", "1.234"],
        3 => vec!["0.0001", "1.2345"],
        4 => vec!["0.00001", "1.23456"],
        _ => vec!["0.0001"],
    };

    println!("\n{} Valid sizes:", "Valid".green());
    for sz in &valid_sizes {
        println!("   {} {}", sz, format!("(≤ {} decimals)", hype.sz_decimals).dimmed());
    }

    println!("\n{} Invalid sizes:", "Invalid".red());
    for sz in &invalid_sizes {
        println!("   {} {} (too many decimals)", sz, "Failed".dimmed());
    }

    // === PRICE VALIDATION (Perps: MAX_DECIMALS = 6) ===
    println!("\n{}", "PRICE (px) VALIDATION (Perpetuals)".bold().underline());
    let max_decimals_in_price = 6 - hype.sz_decimals as usize; // convert u32 → usize

    println!("Rules:");
    println!("  • Max 6 significant digits");
    println!("  • Max {} decimal places (6 - szDecimals)", max_decimals_in_price);
    println!("  • Integer prices always allowed");

    struct PriceTest {
        px: String,
        valid: bool,
        reason: String,
    }

    let price_tests = vec![
        PriceTest { px: "1234.56".to_string(),   valid: true,  reason: "5 sig figs, 2 decimals".to_string() },
        PriceTest { px: "1234.567".to_string(),  valid: false, reason: "7 sig figs → invalid".to_string() },
        PriceTest { px: "0.00123".to_string(), valid: true, reason: "5 sig figs".to_string() },
        PriceTest { px: "0.001234".to_string(),  valid: true,  reason: "6 sig figs OK".to_string() },
        PriceTest { px: "0.0012345".to_string(), valid: false, reason: "7 sig figs → invalid".to_string() },
        PriceTest { px: "123456".to_string(),    valid: true,  reason: "Integer → always allowed".to_string() },
        PriceTest { px: "12.3456".to_string(),   valid: max_decimals_in_price >= 4, reason: format!("4 decimals ≤ {}?", max_decimals_in_price) },
    ];

    println!("\n{} Price examples:", "Check".blue());
    for test in price_tests {
        let icon = if test.valid { "Valid".green() } else { "Invalid".red() };
        let status = if test.valid { "VALID".green() } else { "INVALID".red() };
        println!("   {} {} → {} ({})", icon, test.px.bold(), status, test.reason.dimmed());
    }

    // === SIMULATED VALIDATION LOGIC ===
    println!("\n{}", "SIMULATED VALIDATION LOGIC".bold().underline());

    fn validate_size(sz: &str, sz_decimals: u32) -> (bool, String) {
        let parts: Vec<&str> = sz.split('.').collect();
        if parts.len() > 2 { return (false, "multiple decimal points".to_string()); }
        if parts.len() == 1 { return (true, "integer size".to_string()); }

        let decimals = parts[1].len() as u32;
        if decimals > sz_decimals {
            (false, format!("{} > {} decimals", decimals, sz_decimals))
        } else {
            (true, format!("{} ≤ {} decimals", decimals, sz_decimals))
        }
    }

    fn count_sigfigs(s: &str) -> usize {
        s.chars().filter(|c| c.is_digit(10) && *c != '0').count()
            + s.chars().filter(|c| *c == '.').count().min(1)
    }

    fn validate_price(px: &str, sz_decimals: u32) -> (bool, String) {
        let max_decimals = 6 - sz_decimals as usize;
        let parts: Vec<&str> = px.split('.').collect();

        if parts.len() == 1 {
            return (true, "integer price → always valid".to_string());
        }

        if parts.len() != 2 {
            return (false, "invalid format".to_string());
        }

        let decimal_places = parts[1].len();
        let sigfigs = count_sigfigs(px); // Now used!

        if sigfigs > 6 {
            return (false, format!("{} sigfigs > 6", sigfigs));
        }

        if decimal_places > max_decimals {
            return (false, format!("{} decimals > {} allowed", decimal_places, max_decimals));
        }

        (true, format!("{} sigfigs, {} decimals ≤ {}", sigfigs, decimal_places, max_decimals))
    }

    // Test examples
    println!("{} Simulated size check sz=1.2345:", "Tool".yellow());
    let (ok, msg) = validate_size("1.2345", hype.sz_decimals);
    println!("   sz=1.2345 → {} ({})", if ok { "PASS".green() } else { "FAIL".red() }, msg.dimmed());

    println!("{} Simulated price check px=12.3456:", "Tool".yellow());
    let (ok, msg) = validate_price("12.3456", hype.sz_decimals);
    println!("   px=12.3456 → {} ({})", if ok { "PASS".green() } else { "FAIL".red() }, msg.dimmed());

    println!();
    println!("{}", "Done! Use this to avoid order rejections.".bold().green());
    Ok(())
}