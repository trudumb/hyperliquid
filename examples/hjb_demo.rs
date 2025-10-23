/// HJB Framework Demonstration
/// 
/// This example demonstrates the Hamilton-Jacobi-Bellman (HJB) optimization framework
/// for market making. It shows how the optimal control changes based on different
/// market states and how the HJB objective evaluates different strategies.

use hyperliquid_rust_sdk::{StateVector, ControlVector, ValueFunction, HJBComponents};

fn main() {
    println!("=== HJB Framework Demonstration ===\n");
    println!("Using Market Maker V2 Binary Configuration:");
    println!("  Asset: HYPE");
    println!("  Target Liquidity: 2.0 per side");
    println!("  Half Spread: 5 bps");
    println!("  Max BPS Diff: 10 bps");
    println!("  Max Position: 3.0\n");
    
    // Initialize HJB components matching market maker v2 binary
    let hjb = HJBComponents::new();
    let value_fn = ValueFunction::new(0.01, 86400.0); // φ=0.01, T=24h
    let half_spread = 5.0; // From market_maker_v2 binary
    let max_position = 3.0; // From market_maker_v2 binary
    
    println!("HJB Configuration:");
    println!("  Inventory penalty (φ): {}", value_fn.phi);
    println!("  Terminal time (T): {}s ({}h)", value_fn.terminal_time, value_fn.terminal_time / 3600.0);
    println!("  Base fill rate (λ₀): {}/s", hjb.lambda_base);
    println!("  Taker fee: {}bps\n", hjb.taker_fee_bps);
    
    // ========================================================================
    // Scenario 1: Balanced State (No Inventory, No Drift)
    // ========================================================================
    println!("--- Scenario 1: Balanced Market State ---");
    let state_balanced = StateVector {
        mid_price: 100.0,
        inventory: 0.0,
        adverse_selection_estimate: 0.0,
        market_spread_bps: 10.0,
        lob_imbalance: 0.5,
        previous_mid_price: 100.0,
        volatility_ema_bps: 10.0,
    };
    
    println!("State: {}", state_balanced.to_log_string());
    
    // Evaluate symmetric control using configured half_spread
    let control_sym = ControlVector::symmetric(half_spread);
    let value_sym = hjb.evaluate_control(&control_sym, &state_balanced, &value_fn);
    
    println!("Control: {}", control_sym.to_log_string());
    println!("HJB Objective Value: {:.6}", value_sym);
    
    // Get expected fill rates
    let lambda_b = hjb.maker_bid_fill_rate(control_sym.bid_offset_bps, &state_balanced);
    let lambda_a = hjb.maker_ask_fill_rate(control_sym.ask_offset_bps, &state_balanced);
    println!("Expected Fill Rates: λ^b={:.3}/s, λ^a={:.3}/s\n", lambda_b, lambda_a);
    
    // ========================================================================
    // Scenario 2: Long Inventory (Need to Sell)
    // ========================================================================
    println!("--- Scenario 2: Long Inventory Position ---");
    let state_long = StateVector {
        mid_price: 100.0,
        inventory: 2.4, // 80% of max_position (3.0)
        adverse_selection_estimate: 0.0,
        market_spread_bps: 10.0,
        lob_imbalance: 0.5,
        previous_mid_price: 100.0,
        volatility_ema_bps: 10.0,
    };
    
    println!("State: {}", state_long.to_log_string());
    
    // Find optimal control using configured half_spread
    let control_long = hjb.optimize_control(&state_long, &value_fn, half_spread);
    let value_long = hjb.evaluate_control(&control_long, &state_long, &value_fn);
    
    println!("Optimal Control: {}", control_long.to_log_string());
    println!("HJB Objective Value: {:.6}", value_long);
    
    // Show asymmetry
    println!("Spread Asymmetry:");
    println!("  Ask offset: {:.2}bps (tighter to sell)", control_long.ask_offset_bps);
    println!("  Bid offset: {:.2}bps (wider to avoid buying)", control_long.bid_offset_bps);
    
    if control_long.is_liquidating() {
        println!("  ⚠️  LIQUIDATION MODE ACTIVE: ν^a={:.3}/s", control_long.taker_sell_rate);
    }
    println!();
    
    // ========================================================================
    // Scenario 3: Adverse Selection (Upward Drift)
    // ========================================================================
    println!("--- Scenario 3: Predicted Upward Price Movement ---");
    let state_upward = StateVector {
        mid_price: 100.0,
        inventory: 0.0,
        adverse_selection_estimate: 3.0, // Strong upward drift
        market_spread_bps: 10.0,
        lob_imbalance: 0.5,
        previous_mid_price: 100.0,
        volatility_ema_bps: 10.0,
    };
    
    println!("State: {}", state_upward.to_log_string());
    
    let control_upward = hjb.optimize_control(&state_upward, &value_fn, half_spread);
    let value_upward = hjb.evaluate_control(&control_upward, &state_upward, &value_fn);
    
    println!("Optimal Control: {}", control_upward.to_log_string());
    println!("HJB Objective Value: {:.6}", value_upward);
    
    println!("Adverse Selection Response:");
    println!("  Bid offset: {:.2}bps (tighter to buy before rise)", control_upward.bid_offset_bps);
    println!("  Ask offset: {:.2}bps (wider to avoid selling cheap)", control_upward.ask_offset_bps);
    println!();
    
    // ========================================================================
    // Scenario 4: LOB Imbalance (High Buy Pressure)
    // ========================================================================
    println!("--- Scenario 4: High Buy-Side Imbalance ---");
    let state_imbalance = StateVector {
        mid_price: 100.0,
        inventory: 0.0,
        adverse_selection_estimate: 0.0,
        market_spread_bps: 10.0,
        lob_imbalance: 0.9, // Lots of buy orders
        previous_mid_price: 100.0,
        volatility_ema_bps: 10.0,
    };
    
    println!("State: {}", state_imbalance.to_log_string());
    
    let _control_imbalance = hjb.optimize_control(&state_imbalance, &value_fn, half_spread);
    
    // Compare fill rates with balanced book (using configured half_spread)
    let lambda_b_imb = hjb.maker_bid_fill_rate(half_spread, &state_imbalance);
    let lambda_a_imb = hjb.maker_ask_fill_rate(half_spread, &state_imbalance);
    let lambda_b_bal = hjb.maker_bid_fill_rate(half_spread, &state_balanced);
    let lambda_a_bal = hjb.maker_ask_fill_rate(half_spread, &state_balanced);
    
    println!("Fill Rate Comparison ({}bps quotes):", half_spread);
    println!("  Balanced book: λ^b={:.3}/s, λ^a={:.3}/s", lambda_b_bal, lambda_a_bal);
    println!("  Buy imbalance: λ^b={:.3}/s, λ^a={:.3}/s", lambda_b_imb, lambda_a_imb);
    println!("  Interpretation: Bid fills harder (more competition), ask fills easier (more demand)");
    println!();
    
    // ========================================================================
    // Scenario 5: Value Function Analysis
    // ========================================================================
    println!("--- Scenario 5: Value Function Behavior ---");
    
    let test_state = StateVector {
        mid_price: 100.0,
        inventory: 0.0,
        adverse_selection_estimate: 0.0,
        market_spread_bps: 10.0,
        lob_imbalance: 0.5,
        previous_mid_price: 100.0,
        volatility_ema_bps: 10.0,
    };
    
    println!("Inventory Penalty (φQ²) for max_position={}:", max_position);
    for q in [0.0, 0.75, 1.5, 2.25, 3.0] {
        let v = value_fn.evaluate(q, &test_state);
        println!("  Q={:>5.2}: V(Q)={:>10.2}", q, v);
    }
    println!();
    
    // ========================================================================
    // Scenario 6: Cost Comparison (Maker vs Taker)
    // ========================================================================
    println!("--- Scenario 6: Maker vs Taker Economics ---");
    
    let comparison_state = StateVector {
        mid_price: 100.0,
        inventory: 0.0,
        adverse_selection_estimate: 0.0,
        market_spread_bps: 10.0,
        lob_imbalance: 0.5,
        previous_mid_price: 100.0,
        volatility_ema_bps: 10.0,
    };
    
    // Maker bid at configured half_spread
    let maker_bid_value = hjb.maker_bid_value(half_spread, &comparison_state, &value_fn);
    
    // Taker buy (cross spread)
    let taker_buy_value = hjb.taker_buy_value(1.0, &comparison_state, &value_fn);
    
    println!("Expected Values (per unit/second) at {}bps:", half_spread);
    println!("  Maker Bid ({}bps): {:.6}", half_spread, maker_bid_value);
    println!("  Taker Buy (1/s): {:.6}", taker_buy_value);
    println!("  Difference: {:.6}", maker_bid_value - taker_buy_value);
    println!("  Interpretation: Maker is more valuable (avoids paying spread + fee)");
    println!();
    
    // ========================================================================
    // Scenario 7: Grid Search Visualization
    // ========================================================================
    println!("--- Scenario 7: Objective Value Heatmap ---");
    
    let grid_state = StateVector {
        mid_price: 100.0,
        inventory: 1.5, // 50% of max_position (3.0)
        adverse_selection_estimate: 1.0, // Slight upward drift
        market_spread_bps: 10.0,
        lob_imbalance: 0.5,
        previous_mid_price: 100.0,
        volatility_ema_bps: 10.0,
    };
    
    println!("State: Q=1.5 (50% of max), μ̂=1.0, Δ=10bps");
    println!("\nObjective Value Grid (δ^bid vs δ^ask in bps):");
    println!("       δ^ask →");
    print!("δ^bid ↓ ");
    for ask_bps in [2.5, 3.75, 5.0, 6.25, 7.5] {
        print!("{:>8.2}", ask_bps);
    }
    println!();
    
    for bid_bps in [2.5, 3.75, 5.0, 6.25, 7.5] {
        print!("{:>6.2}  ", bid_bps);
        for ask_bps in [2.5, 3.75, 5.0, 6.25, 7.5] {
            let control = ControlVector::asymmetric(ask_bps, bid_bps);
            let value = hjb.evaluate_control(&control, &grid_state, &value_fn);
            print!("{:>8.3}", value);
        }
        println!();
    }
    
    println!("\nOptimal control from grid search:");
    let optimal = hjb.optimize_control(&grid_state, &value_fn, half_spread);
    println!("  {}", optimal.to_log_string());
    
    println!("\n=== Demo Complete ===");
    println!("\nKey Takeaways:");
    println!("  1. HJB framework optimizes P&L while managing inventory risk");
    println!("  2. Optimal quotes adapt to inventory, drift, and LOB state");
    println!("  3. Taker orders activate only in extreme conditions");
    println!("  4. Fill rates depend on quote competitiveness and LOB imbalance");
    println!("  5. Value function captures inventory penalty φQ²(T-t)");
}
