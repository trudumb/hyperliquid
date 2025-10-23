# Hyperliquid Rust SDK

**Advanced Market Making System with Optimal Control Theory**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Rust](https://img.shields.io/badge/rust-1.70%2B-orange.svg)](https://www.rust-lang.org/)

A production-ready, theoretically-grounded market making system for the Hyperliquid decentralized exchange. This SDK combines cutting-edge academic research in market microstructure with practical algorithmic trading implementation.

## 🌟 Key Features

- **🎓 Theoretically Sound**: Based on Hamilton-Jacobi-Bellman equations and Avellaneda-Stoikov optimal market making
- **🛡️ Adverse Selection Protection**: Real-time drift estimation to avoid being picked off by informed traders
- **📊 Intelligent Inventory Management**: Automatic position risk management via asymmetric quoting
- **🔄 LOB-Aware Decision Making**: Order book analysis and fill rate calibration
- **⚡ High Performance**: Microsecond latency for decision making
- **🧪 Comprehensive Testing**: 40+ unit tests covering all components
- **📈 Production Ready**: Structured logging, risk management, and simulation mode

## 🚀 Quick Start

### Installation

```bash
cargo add hyperliquid_rust_sdk
```

### Basic Market Maker

```rust
use hyperliquid_rust_sdk::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    // Configure market maker
    let input = MarketMakerInput {
        asset: "ETH".to_string(),
        target_liquidity: 0.1,
        half_spread: 10,  // 10 bps
        max_absolute_position_size: 10.0,
        max_drawdown_pct: 0.1,  // 10% max loss
        // ... other parameters
        simulation_mode: true,  // Start with dry-run
    };

    // Create and start
    let mut market_maker = MarketMaker::new(input).await?;
    market_maker.start().await?;
    
    Ok(())
}
```

### With Advanced Features (Inventory Skewing)

```rust
use hyperliquid_rust_sdk::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    // Configure inventory skewing
    let skew_config = InventorySkewConfig::new(
        0.5,  // inventory_skew_factor
        0.3,  // book_imbalance_factor  
        5,    // depth_analysis_levels
    )?;

    let input = MarketMakerInput {
        asset: "HYPE".to_string(),
        target_liquidity: 10.0,
        half_spread: 15,
        max_absolute_position_size: 50.0,
        inventory_skew_config: Some(skew_config),
        // ... other parameters
    };

    let mut market_maker = MarketMaker::new(input).await?;
    market_maker.start().await?;
    
    Ok(())
}
```

## 📚 Complete Documentation

For a comprehensive understanding of the system, see **[PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md)** which covers:

- 🎯 Project mission and theoretical foundation
- 🏗️ Architecture and core components
- 📐 Mathematical formulations (State Vector, Control Vector, HJB Framework)
- 🎛️ Configuration and parameter tuning
- 🔬 Testing and validation
- 📊 Performance monitoring
- 🚀 Production deployment guide
- 🎓 Educational resources and academic references

### Quick Reference Guides

- **[QUICKSTART_SKEWING.md](QUICKSTART_SKEWING.md)** - Get started with inventory management
- **[STATE_VECTOR_QUICK_REF.md](STATE_VECTOR_QUICK_REF.md)** - State vector component reference

### Deep Dive Documentation

- **[STATE_VECTOR.md](STATE_VECTOR.md)** - Market observation framework
- **[CONTROL_VECTOR.md](CONTROL_VECTOR.md)** - Trading action framework
- **[HJB_FRAMEWORK.md](HJB_FRAMEWORK.md)** - Optimal control theory implementation
- **[INVENTORY_SKEWING.md](INVENTORY_SKEWING.md)** - Position management system
- **[STATE_CONTROL_FRAMEWORK.md](STATE_CONTROL_FRAMEWORK.md)** - Complete integration guide
- **[ARCHITECTURE_DIAGRAM.md](ARCHITECTURE_DIAGRAM.md)** - System architecture diagrams

## 🎓 Theoretical Foundation

This system implements optimal market making based on:

1. **Avellaneda-Stoikov Model (2008)**: High-frequency trading in limit order books
2. **Hamilton-Jacobi-Bellman Equation**: Optimal control for continuous-time systems
3. **Market Microstructure Theory**: Order flow and adverse selection

The market maker solves:

$$\max_{\mathbf{u}_t} \mathbb{E} \left[ \int_t^T (dP\&L_s - \phi Q_s^2 ds) \right]$$

Where $\mathbf{u}_t$ is the control vector, $Q_s$ is inventory, and $\phi$ is risk aversion.

## 🏗️ System Architecture

```
Market Data → State Vector (Observation) → HJB Decision Engine → Control Vector (Action) → Order Execution
```

**Key Components**:
- **State Vector ($\mathbf{Z}_t$)**: Mid price, inventory, adverse selection estimate, spread, imbalance
- **Control Vector ($\mathbf{u}_t$)**: Bid/ask offsets, taker buy/sell rates
- **Value Function**: $V(Q, \mathbf{Z}, t)$ - Maximum expected P&L
- **HJB Optimizer**: Grid-based optimal control selection

## 📊 Examples

Run the included examples to see the system in action:

```bash
# Basic market maker
cargo run --bin market_maker

# With inventory skewing
cargo run --bin market_maker_with_skew

# State vector demonstration
cargo run --example state_vector_demo

# HJB framework scenarios
cargo run --example hjb_demo

# Tick/lot size handling
cargo run --example tick_lot_size_demo
```

See `src/bin/` and `examples/` directories for more examples.

## 🧪 Testing

Comprehensive test suite with 40+ tests:

```bash
# Run all tests
cargo test

# Run specific component tests
cargo test state_vector
cargo test hjb
cargo test inventory_skew

# Run with output
cargo test -- --nocapture
```

## 🎛️ Configuration

Key parameters for tuning:

```rust
MarketMakerInput {
    // Spread Configuration
    half_spread: 10,                    // Base half spread (bps)
    max_bps_diff: 5,                   // Min change to update (bps)
    
    // Risk Management
    max_absolute_position_size: 100.0, // Max inventory
    max_drawdown_pct: 0.1,            // Max daily loss (10%)
    position_value_limit_pct: 0.5,    // Max portfolio % (50%)
    
    // Adverse Selection
    adverse_selection_threshold: 0.05, // Spread widening trigger
    
    // Volatility
    base_volatility: 0.25,             // Expected volatility (25%)
    
    // HJB Parameters
    phi: 0.01,                         // Inventory aversion
    terminal_time: 86400.0,            // Trading horizon (24h)
    
    // Testing
    simulation_mode: true,             // Dry-run mode
}
```

See [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md) for detailed configuration guides and strategy presets.

## 📈 Performance Monitoring

The system tracks comprehensive metrics:

- **P&L**: Total, realized, unrealized, ROI, Sharpe ratio
- **Risk**: Max drawdown, VaR, portfolio heat, inventory levels
- **Operations**: Fill rates, adverse selection scores, quote uptime
- **Structured Logging**: JSON events for external monitoring

## 🔬 Academic References

1. Avellaneda, M., & Stoikov, S. (2008). "High-frequency trading in a limit order book"
2. Cartea, Á., Jaimungal, S., & Penalva, J. (2015). "Algorithmic and High-Frequency Trading"
3. Guéant, O., et al. (2013). "Dealing with the inventory risk"
4. Cont, R., et al. (2010). "A stochastic model for order book dynamics"

## 🚀 Production Deployment

**Deployment Checklist**:
- ✅ Parameter calibration from historical data
- ✅ Risk limits appropriate to capital
- ✅ 24h+ simulation testing
- ✅ Monitoring and alerting configured
- ✅ Secure key management
- ✅ Circuit breakers implemented

See [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md) for complete production deployment guide.

## 🔮 Roadmap

- [ ] Machine learning integration for drift prediction
- [ ] Multi-asset portfolio optimization
- [ ] Advanced order types (iceberg, TWAP)
- [ ] Backtesting framework
- [ ] Real-time dashboard

## 🤝 Contributing

Contributions welcome! Areas to extend:

1. Add new state vector components
2. Implement ML-based prediction models
3. Extend to multi-asset portfolios
4. Add new risk management features
5. Improve documentation

See [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md) for development guidelines.

## License

This project is licensed under the terms of the `MIT` license. See [LICENSE](LICENSE.md) for more details.

```bibtex
@misc{hyperliquid-rust-sdk,
  author = {Hyperliquid},
  title = {SDK for Hyperliquid API trading with Rust.},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/hyperliquid-dex/hyperliquid-rust-sdk}}
}
```
