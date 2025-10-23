# Hyperliquid Rust SDK: Complete Project Overview & Intentions

## 🎯 Project Mission

This project implements a **theoretically-grounded, production-ready market making system** for the Hyperliquid decentralized exchange. It combines cutting-edge academic research in market microstructure with practical algorithmic trading implementation, creating a sophisticated automated trading system that:

1. **Protects against adverse selection** using real-time drift estimation
2. **Manages inventory risk** systematically using optimal control theory  
3. **Adapts to market conditions** via order book analysis and volatility tracking
4. **Executes optimally** balancing profitability with risk management

## 📐 Theoretical Foundation

### Academic Framework

The system is built on three pillars of modern quantitative finance:

1. **Avellaneda-Stoikov Model (2008)**: High-frequency trading in limit order books
2. **Hamilton-Jacobi-Bellman Equation**: Optimal control for continuous-time stochastic systems
3. **Market Microstructure Theory**: Order flow, adverse selection, and liquidity provision

### Mathematical Core

The market maker solves a continuous-time optimization problem:

$$\max_{\mathbf{u}_t} \mathbb{E} \left[ \int_t^T (dP\&L_s - \phi Q_s^2 ds) \right]$$

Where:
- **$\mathbf{u}_t$**: Control vector (our trading actions)
- **$Q_s$**: Inventory (position)
- **$\phi$**: Inventory aversion (risk parameter)
- **$T$**: Terminal time

This balances **profit maximization** with **inventory risk management**.

## 🏗️ Architecture Overview

### The Observe-Decide-Act Loop

```
┌────────────────────────────────────────────────────────────┐
│                    MARKET DATA FEEDS                        │
│  (Price Streams, Order Book, User Events)                 │
└────────────────────┬───────────────────────────────────────┘
                     │
                     ▼
         ┌───────────────────────┐
         │   STATE VECTOR (Z_t)  │  ◄── OBSERVATION
         │   ─────────────────   │
         │   • S_t: Mid Price    │
         │   • Q_t: Inventory    │
         │   • μ̂_t: Adverse Sel  │
         │   • Δ_t: Spread       │
         │   • I_t: Imbalance    │
         └───────────┬───────────┘
                     │
                     ▼
         ┌───────────────────────┐
         │  HJB DECISION ENGINE  │  ◄── DECISION
         │   ─────────────────   │
         │  Value Function V(Q)  │
         │  Fill Rate Models λ   │
         │  Optimization Logic   │
         └───────────┬───────────┘
                     │
                     ▼
         ┌───────────────────────┐
         │  CONTROL VECTOR (u_t) │  ◄── ACTION
         │   ─────────────────   │
         │   • δ^a: Ask Offset   │
         │   • δ^b: Bid Offset   │
         │   • ν^a: Taker Sell   │
         │   • ν^b: Taker Buy    │
         └───────────┬───────────┘
                     │
                     ▼
         ┌───────────────────────┐
         │   ORDER EXECUTION     │
         │   ─────────────────   │
         │   Passive Quotes      │
         │   Active Orders       │
         └───────────────────────┘
```

## 🔑 Core Components

### 1. State Vector ($\mathbf{Z}_t$)

**Purpose**: Complete representation of market state for decision making

**Components**:
- **$S_t$ (Mid Price)**: Current market reference price
- **$Q_t$ (Inventory)**: Our position (+long/-short)
- **$\hat{\mu}_t$ (Adverse Selection Estimate)**: Expected short-term drift
- **$\Delta_t$ (Market Spread)**: BBO spread in basis points
- **$I_t$ (LOB Imbalance)**: Order book bid/ask volume ratio

**Key Innovation**: Exponential moving average filter for drift estimation:
$$\hat{\mu}_t = \lambda \cdot \text{signal}_t \cdot \text{spread\_scale}_t + (1-\lambda) \cdot \hat{\mu}_{t-1}$$

This protects against informed traders by detecting directional momentum.

**Files**: 
- Implementation: `src/market_maker_v2.rs`
- Documentation: `STATE_VECTOR.md`, `STATE_VECTOR_QUICK_REF.md`
- Demo: `examples/state_vector_demo.rs`

### 2. Control Vector ($\mathbf{u}_t$)

**Purpose**: Complete specification of trading actions

**Components**:
- **$\delta^a_t$ (Ask Offset)**: Distance from mid to sell quote (bps)
- **$\delta^b_t$ (Bid Offset)**: Distance from mid to buy quote (bps)  
- **$\nu^a_t$ (Taker Sell Rate)**: Aggressive sell rate (units/sec)
- **$\nu^b_t$ (Taker Buy Rate)**: Aggressive buy rate (units/sec)

**Key Innovation**: Unified framework for passive (market making) and active (inventory management) strategies.

**Files**:
- Implementation: `src/market_maker_v2.rs`
- Documentation: `CONTROL_VECTOR.md`

### 3. HJB Framework

**Purpose**: Theoretically optimal decision making via Hamilton-Jacobi-Bellman equation

**Core Structures**:

1. **ValueFunction**: $V(Q, \mathbf{Z}, t)$ - Maximum expected P&L from state
   - Inventory penalty: $V(Q) \approx -\phi Q^2 (T-t)$
   - Time evolution tracking
   - Value caching for performance

2. **HJBComponents**: Optimization engine
   - LOB-aware fill rate models: $\lambda(\delta, \mathbf{Z})$
   - Maker value: $\lambda^a \cdot [V(Q-1) - V(Q) + (S + \delta^a)]$
   - Taker value: $\nu^a \cdot [V(Q-1) - V(Q) + S^b]$
   - Grid-based control optimization

**Key Innovation**: Poisson fill rates calibrated from order book state:
$$\lambda(\delta, \mathbf{Z}) = \lambda_0 \cdot f_{\text{imbalance}}(I_t) \cdot e^{-\beta \cdot (\delta - \delta_{\text{market}})}$$

**Files**:
- Implementation: `src/market_maker_v2.rs` (lines 390-690)
- Documentation: `HJB_FRAMEWORK.md`, `HJB_IMPLEMENTATION_SUMMARY.md`
- Demo: `examples/hjb_demo.rs`

### 4. Inventory Skewing

**Purpose**: Automatic position risk management via asymmetric quoting

**Mechanism**:
- **Position Component**: When long, shift quotes DOWN to encourage selling
- **Book Component**: React to order book liquidity imbalances

**Formula**:
```
total_skew = position_skew + book_skew
position_skew = -(Q/Q_max) × skew_factor × base_spread
book_skew = imbalance × book_factor × base_spread
```

**Benefits**:
- ✅ Prevents toxic position buildup
- ✅ Natural inventory rebalancing
- ✅ Market-adaptive quoting
- ✅ Improved risk-adjusted returns

**Files**:
- Implementation: `src/inventory_skew.rs`, `src/book_analyzer.rs`
- Documentation: `INVENTORY_SKEWING.md`, `QUICKSTART_SKEWING.md`
- Demo: `src/bin/market_maker_with_skew.rs`

### 5. Volatility-Based Spread Scaling

**Purpose**: Dynamic spread adjustment based on market conditions

**Implementation**:
- EWMA volatility estimation from price changes
- Spread multiplier: $\text{mult} = \frac{\sigma_{\text{current}}}{\sigma_{\text{base}}}$
- Automatic widening in volatile markets

**Benefits**:
- Better protection in uncertain conditions
- Tighter spreads when safe
- Adaptive to market regimes

**Files**: `src/market_maker_v2.rs` (VolatilityEstimator)

### 6. Adverse Selection Protection

**Purpose**: Detect and protect against informed traders

**Mechanism**:
- Track post-fill price movements
- Calculate adverse selection score: $\text{score} = \text{avg}(\text{price change after fill})$
- Dynamic spread widening when score exceeds threshold
- Trading pause in severe cases

**Key Metric**: If consistently picked off, widen spreads asymmetrically based on $\hat{\mu}_t$.

**Files**: `src/market_maker_v2.rs` (AdverseSelectionMonitor)

## 🎛️ Configuration & Parameters

### Risk Parameters

```rust
pub struct MarketMakerInput {
    // Position Limits
    pub max_absolute_position_size: f64,     // Max inventory (e.g., 1000 units)
    pub position_value_limit_pct: f64,       // Max portfolio % (e.g., 0.5 = 50%)
    
    // Loss Limits
    pub max_drawdown_pct: f64,               // Max daily loss (e.g., 0.1 = 10%)
    
    // Spread Configuration
    pub half_spread: u64,                    // Base half spread (bps)
    pub max_bps_diff: u64,                   // Min price change to update (bps)
    
    // Adverse Selection
    pub adverse_selection_threshold: f64,    // When to widen spreads (e.g., 0.05)
    
    // Volatility
    pub base_volatility: f64,                // Expected "normal" volatility
    pub volatility_lookback_minutes: u64,    // Historical window
    
    // HJB Parameters
    pub phi: f64,                            // Inventory aversion (default: 0.01)
    pub terminal_time: f64,                  // Trading horizon (default: 86400s = 24h)
}
```

### Inventory Skewing Configuration

```rust
pub struct InventorySkewConfig {
    pub inventory_skew_factor: f64,     // 0.5 = moderate position management
    pub book_imbalance_factor: f64,     // 0.3 = conservative book reaction
    pub depth_analysis_levels: usize,   // 5 = balanced performance
}
```

### Strategy Presets

| Strategy | Risk Level | Spreads | Inventory Aversion | Use Case |
|----------|-----------|---------|-------------------|----------|
| **Conservative** | Low | 20 bps | φ = 0.02 | Capital preservation |
| **Balanced** | Medium | 10 bps | φ = 0.01 | Standard market making |
| **Aggressive** | High | 5 bps | φ = 0.005 | High-frequency trading |

## 🔬 Testing & Validation

### Comprehensive Test Suite

**Test Coverage**:
- ✅ 40+ unit tests across all components
- ✅ Edge cases (zero position, extreme inventory)
- ✅ Mathematical correctness (HJB equations, fill rates)
- ✅ Integration tests (state-control loop)
- ✅ Performance tests (latency, memory)

**Key Test Categories**:

1. **State Vector Tests** (`src/state_vector_tests.rs`)
   - Imbalance calculations
   - Adverse selection filtering
   - Risk multipliers
   - Market condition checks

2. **HJB Framework Tests** (`src/market_maker_v2.rs`)
   - Value function correctness
   - Fill rate calibration
   - Optimization logic
   - Maker vs taker economics

3. **Inventory Skewing Tests** (`src/inventory_skew.rs`)
   - Position skew calculations
   - Book imbalance integration
   - Parameter validation

4. **Risk Management Tests** (`src/market_maker_v2.rs`)
   - Position limits enforcement
   - Daily loss limits
   - VaR calculations
   - Portfolio heat tracking

**Running Tests**:
```bash
# All tests
cargo test

# Specific component
cargo test hjb
cargo test state_vector
cargo test inventory_skew

# With output
cargo test -- --nocapture
```

### Simulation/Dry-Run Mode

**Purpose**: Safe strategy testing without real capital

**Features**:
- Virtual order tracking
- Simulated fills based on book analysis
- Complete P&L accounting
- Performance statistics

**Usage**:
```rust
let input = MarketMakerInput {
    simulation_mode: true,  // Enable dry-run
    // ... other params
};
```

**Benefits**:
- ✅ Risk-free strategy development
- ✅ Parameter tuning
- ✅ Performance validation
- ✅ Backtesting capabilities

## 📊 Performance Monitoring

### Key Metrics

**P&L Metrics**:
- Total P&L (Realized + Unrealized)
- ROI (Return on Investment)
- Sharpe Ratio (risk-adjusted returns)
- Win Rate
- Profit Factor

**Risk Metrics**:
- Max Drawdown
- Value at Risk (VaR 95%)
- Portfolio Heat (maximum adverse excursion)
- Average Inventory Level

**Operational Metrics**:
- Fill Rate
- Adverse Selection Score
- Spread Capture Ratio
- Quote Uptime

### Structured Logging

**Format**: JSON-structured events for monitoring systems

**Event Types**:
```rust
pub enum LogEvent {
    Trade,              // Fill events
    Performance,        // P&L updates  
    Risk,              // Risk threshold breaches
    MarketData,        // State vector updates
    Order,             // Order placement/cancellation
    Inventory,         // Position changes
}
```

**Example Log**:
```json
{
  "timestamp": "2025-10-23T10:30:00Z",
  "event": "Trade",
  "asset": "HYPE",
  "side": "sell",
  "size": 10.0,
  "price": 25.50,
  "pnl": 12.50,
  "inventory": 40.0
}
```

## 🚀 Production Deployment

### Deployment Checklist

- [ ] **Parameter Tuning**: Calibrate to asset characteristics
- [ ] **Risk Limits**: Set appropriate position/loss limits  
- [ ] **Fill Rate Calibration**: Validate λ_base from historical data
- [ ] **Simulation Testing**: Run 24h+ in dry-run mode
- [ ] **Monitoring Setup**: Configure logging and alerts
- [ ] **Backup Keys**: Secure wallet private keys
- [ ] **Fail-safes**: Implement circuit breakers
- [ ] **Performance Baseline**: Establish expected metrics

### Operation Modes

**Mode 1: Heuristic (Default - Recommended)**
- Fast rule-based decision making
- Microsecond latency
- Good approximation of HJB solution
- Production-ready

**Mode 2: Full HJB Optimization (Feature Flag)**
- Numerical optimization via grid search
- ~100μs latency (acceptable for most markets)
- Theoretically optimal
- Requires parameter tuning

**Mode 3: Simulation**
- Virtual trading for testing
- No real capital at risk
- Complete performance tracking

### Monitoring in Production

**Real-time Monitoring**:
```rust
// Get performance snapshot
let perf = market_maker.get_performance_summary();
info!("P&L: ${:.2}, Sharpe: {:.2}, Inventory: {:.1}%", 
      perf.total_pnl, 
      perf.sharpe_ratio,
      perf.inventory_ratio * 100.0);

// Check risk status
let var = market_maker.risk_manager.calculate_var_95();
let heat = market_maker.risk_manager.current_heat;
warn_if!(var > risk_limit, "VaR breach: ${:.2}", var);

// Evaluate strategy
let hjb_value = market_maker.evaluate_current_strategy();
info!("HJB objective: {:.6}", hjb_value);
```

## 🎓 Educational Resources

### Academic References

1. **Avellaneda, M., & Stoikov, S. (2008)**
   - "High-frequency trading in a limit order book"
   - *Quantitative Finance*, 8(3), 217-224
   - Foundation for optimal market making

2. **Cartea, Á., Jaimungal, S., & Penalva, J. (2015)**
   - "Algorithmic and High-Frequency Trading"
   - Cambridge University Press
   - Comprehensive treatment of HJB framework

3. **Guéant, O., Lehalle, C. A., & Fernandez-Tapia, J. (2013)**
   - "Dealing with the inventory risk"
   - *Mathematics and Financial Economics*, 7(4), 477-507
   - Inventory management strategies

4. **Cont, R., Stoikov, S., & Talreja, R. (2010)**
   - "A stochastic model for order book dynamics"
   - *Operations Research*, 58(3), 549-563
   - Order book microstructure

### Documentation Structure

**Quick Start Guides**:
- `QUICKSTART_SKEWING.md`: Inventory management
- `STATE_VECTOR_QUICK_REF.md`: State vector reference
- `README.md`: SDK overview

**Deep Dives**:
- `STATE_VECTOR.md`: Complete state vector theory
- `CONTROL_VECTOR.md`: Action framework
- `HJB_FRAMEWORK.md`: Optimal control theory
- `INVENTORY_SKEWING.md`: Position management
- `STATE_CONTROL_FRAMEWORK.md`: Integration guide

**Implementation**:
- `HJB_IMPLEMENTATION_SUMMARY.md`: Technical details
- `IMPLEMENTATION_SUMMARY.md`: Overall summary
- `ENHANCEMENTS.md`: Feature improvements
- `ARCHITECTURE_DIAGRAM.md`: System diagrams

**Examples** (`examples/`):
- `state_vector_demo.rs`: State observation
- `hjb_demo.rs`: HJB optimization scenarios
- `market_maker_with_skew.rs`: Production example

## 🔮 Future Enhancements

### Planned Features

1. **Machine Learning Integration**
   - Neural network for $\hat{\mu}_t$ prediction
   - RL-based control learning
   - Feature engineering from state vector

2. **Multi-Asset Support**
   - Portfolio-level optimization
   - Cross-asset correlation tracking
   - Risk aggregation

3. **Advanced Order Types**
   - Iceberg orders
   - Time-weighted execution
   - TWAP/VWAP strategies

4. **Regime Detection**
   - Automatic parameter adaptation
   - Market condition classification
   - Dynamic strategy switching

5. **Backtesting Framework**
   - Historical simulation
   - Strategy comparison
   - Performance attribution

### Research Directions

- **Stochastic Control**: Jump-diffusion processes
- **Game Theory**: Multi-agent market making
- **High-Frequency Limits**: Infinitesimal spreads
- **Transaction Costs**: Non-linear fee structures

## 📝 Key Innovations

### What Makes This System Unique

1. **Theoretical Soundness + Practical Implementation**
   - Academic rigor (HJB equations, optimal control)
   - Production-ready code (tested, documented, performant)

2. **Complete State-Control Framework**
   - Unified observation (State Vector)
   - Unified action (Control Vector)
   - Clear separation of concerns

3. **LOB-Aware Decision Making**
   - Order book integration
   - Fill rate calibration from imbalance
   - Real-time microstructure analysis

4. **Multi-Mode Operation**
   - Heuristic (fast, good enough)
   - HJB (optimal, slower)
   - Simulation (safe testing)

5. **Comprehensive Risk Management**
   - Position limits
   - Loss limits
   - Volatility adaptation
   - Adverse selection protection

6. **Production-Grade Engineering**
   - Extensive testing
   - Structured logging
   - Performance monitoring
   - Documentation

## 🎯 Project Intentions Summary

### Primary Goals

1. **Build a sophisticated market making system** that combines:
   - Modern academic research
   - Production-quality engineering
   - Real-world risk management

2. **Provide educational value** through:
   - Clear documentation
   - Working examples
   - Theoretical foundations

3. **Enable systematic trading** via:
   - Reproducible strategies
   - Backtestable logic
   - Extensible framework

### Design Principles

1. **Separation of Concerns**
   - State (observation) separate from control (action)
   - Decision logic separate from execution
   - Configuration separate from implementation

2. **Theoretical Grounding**
   - Every feature backed by research
   - Mathematical formulation for key algorithms
   - Academic references provided

3. **Production Readiness**
   - Comprehensive testing
   - Performance optimization
   - Monitoring and logging
   - Error handling

4. **Extensibility**
   - Modular architecture
   - Clean APIs
   - Plugin points for ML, multi-asset, etc.

### Non-Goals

- ❌ Not a "get rich quick" system
- ❌ Not a black box (everything explained)
- ❌ Not over-optimized (clarity over micro-optimization)
- ❌ Not locked to one strategy (framework, not rigid system)

## 🤝 Contributing

### How to Extend

1. **Add State Components**: Extend `StateVector` with new signals
2. **Add Control Actions**: Extend `ControlVector` with new levers
3. **Customize Decision Logic**: Modify `apply_state_adjustments()`
4. **Integrate ML Models**: Use state vector as features
5. **Multi-Asset**: Extend to portfolio optimization

### Development Guidelines

- Write tests for all new features
- Document mathematical foundations
- Provide usage examples
- Maintain performance benchmarks
- Keep academic references

## 📞 Support & Resources

**Documentation**: See all `.md` files in repository root
**Examples**: See `examples/` directory
**Tests**: Run `cargo test` for validation
**Issues**: GitHub issues for bugs/questions

---

## Version History

- **v1.0** (Oct 2025): Initial implementation
  - State Vector framework
  - Control Vector framework
  - HJB optimization
  - Inventory skewing
  - Comprehensive documentation

---

**This project represents the convergence of quantitative finance theory and practical software engineering, creating a sophisticated automated trading system suitable for both production deployment and educational study.**

*For detailed information on specific components, please refer to the individual documentation files listed throughout this overview.*
