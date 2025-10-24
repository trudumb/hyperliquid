# State Vector Architecture Diagram

## System Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         MARKET DATA FEEDS                            │
├─────────────────────────────────────────────────────────────────────┤
│  AllMids Stream    │   L2Book Stream   │   UserEvents Stream        │
│  (Mid Prices)      │   (Order Book)    │   (Fills/Updates)          │
└──────┬─────────────┴─────────┬─────────┴────────────┬───────────────┘
       │                       │                       │
       │ S_t                   │ Depth, Bids/Asks     │ Position Changes
       │                       │                       │
       ▼                       ▼                       ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       STATE VECTOR (Z_t)                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐   │
│  │  S_t       │  │  Q_t       │  │  μ̂_t       │  │  Δ_t       │   │
│  │  Mid Price │  │  Inventory │  │  Adverse   │  │  Spread    │   │
│  │            │  │            │  │  Selection │  │            │   │
│  └────────────┘  └────────────┘  └────────────┘  └────────────┘   │
│                                                                       │
│  ┌────────────┐                                                     │
│  │  I_t       │       Exponential Moving Average Filter             │
│  │  LOB       │       Signal Processing & Scaling                   │
│  │  Imbalance │       Risk Calculations                             │
│  └────────────┘                                                     │
│                                                                       │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
                            │ Decision Support API
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    DECISION MAKING LAYER                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  get_adverse_selection_adjustment()  ──▶  Spread Asymmetry          │
│  get_inventory_risk_multiplier()     ──▶  Spread Widening           │
│  get_inventory_urgency()             ──▶  Exit Priority             │
│  is_market_favorable()               ──▶  Trade/Pause Decision      │
│                                                                       │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
                            │ Trading Signals
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      MARKET MAKER LOGIC                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  Calculate Optimal Quotes:                                           │
│  ┌─────────────────────────────────────────────┐                    │
│  │ Base Spread (half_spread parameter)         │                    │
│  │ + Adverse Selection Adjustment              │                    │
│  │ × Inventory Risk Multiplier                 │                    │
│  │ + Inventory Skew (existing feature)         │                    │
│  └─────────────────────────────────────────────┘                    │
│                                                                       │
│  Place Orders:                                                       │
│  • Buy @ (Mid - Adjusted Spread)                                    │
│  • Sell @ (Mid + Adjusted Spread)                                   │
│                                                                       │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
                            │ Order Placement
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      HYPERLIQUID EXCHANGE                            │
└─────────────────────────────────────────────────────────────────────┘
```

## Data Flow Diagram

```
Market Data                State Vector               Decision Making
──────────────────────────────────────────────────────────────────────

All Mids ────────┐
                 ├──▶ mid_price (S_t)
L2 Book ─────────┤
                 │
                 ├──▶ market_spread_bps (Δ_t) ──┐
                 │                               │
                 │                               ├──▶ spread_scale
                 │                               │
Book Analysis ───┼──▶ lob_imbalance (I_t) ──────┤
                 │                               │
                 │                               ├──▶ signal
                 │                               │
User Fills ──────┴──▶ inventory (Q_t)           │
                                                 │
                                                 ▼
                                    adverse_selection_estimate (μ̂_t)
                                                 │
                 ┌───────────────────────────────┴───────────────────┐
                 │                                                   │
                 ▼                                                   ▼
    get_adverse_selection_adjustment()            get_inventory_risk_multiplier()
                 │                                                   │
                 │                                                   │
                 └─────────────────┬───────────────────────────────┘
                                   │
                                   ▼
                          Calculate Final Quotes
                                   │
                                   ▼
                            Place/Adjust Orders
```

## Component Interaction

```
┌──────────────────────────────────────────────────────────────┐
│                        MarketMaker                           │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  Fields:                                                     │
│  • state_vector: StateVector                                │
│  • latest_book: Option<OrderBook>                           │
│  • latest_book_analysis: Option<BookAnalysis>               │
│  • cur_position: f64                                        │
│  • latest_mid_price: f64                                    │
│                                                              │
│  Methods:                                                    │
│  • update_state_vector()  ◄──┐                             │
│  • get_state_vector()         │                             │
│  • potentially_update()       │ Called on every event       │
│                               │                             │
│  Event Handlers:              │                             │
│  • Message::AllMids       ────┘                             │
│  • Message::L2Book        ────┐                             │
│  • Message::User          ────┤                             │
│                               │                             │
└───────────────────────────────┼─────────────────────────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │    StateVector        │
                    ├───────────────────────┤
                    │ • mid_price           │
                    │ • inventory           │
                    │ • adverse_selection_  │
                    │   estimate            │
                    │ • market_spread_bps   │
                    │ • lob_imbalance       │
                    │                       │
                    │ update() ◄────────────┼── Market Data
                    │ get_*() ──────────────┼──▶ Decisions
                    └───────────────────────┘
```

## Signal Processing Pipeline

```
Raw LOB Data
    │
    ├─▶ Calculate Imbalance (I_t)
    │       I_t = V^b / (V^b + V^a)
    │
    ├─▶ Convert to Signal
    │       signal = 2 * (I_t - 0.5)
    │       Range: [-1, 1]
    │
    ├─▶ Calculate Spread Scale
    │       spread_scale = 1 / (1 + Δ_t/100)
    │       (Dampens signal in wide spreads)
    │
    ├─▶ Scale Signal
    │       scaled_signal = signal * spread_scale
    │
    └─▶ Update Estimate (EMA)
            μ̂_t = λ * scaled_signal + (1-λ) * μ̂_{t-1}
            
            Result: Filtered drift estimate
```

## Decision Tree

```
                        New Market Data
                              │
                              ▼
                    Update State Vector
                              │
                              ▼
                ┌─────────────┴─────────────┐
                │                           │
                ▼                           ▼
    Check Market Conditions      Calculate Adjustments
    (is_market_favorable)         │
                │                 ├─▶ Adverse Selection
                │                 ├─▶ Inventory Risk
                │                 └─▶ Inventory Urgency
                │                           │
        ┌───────┴────────┐                 │
        │                │                 │
        ▼                ▼                 │
    Unfavorable      Favorable            │
    (Pause)          (Continue)           │
                         │                 │
                         └────────┬────────┘
                                  │
                                  ▼
                         Apply Adjustments
                                  │
                    ┌─────────────┼─────────────┐
                    │             │             │
                    ▼             ▼             ▼
                Buy Side      Mid Price      Sell Side
                    │             │             │
                    │             │             │
        Base - Adj  │             │             │  Base + Adj
        × Risk Mult │             │             │  × Risk Mult
                    │             │             │
                    └─────────────┴─────────────┘
                                  │
                                  ▼
                          Validate & Place Orders
```

## Adverse Selection Signal Flow

```
Order Book Imbalance (I_t)
         │
         │ [0, 1] range
         │
         ▼
    I_t - 0.5          ◄── Center at 0
         │
         │ × 2
         │
         ▼
    Signal [-1, 1]     ◄── Directional signal
         │
         │ × spread_scale
         │
         ▼
    Scaled Signal      ◄── Volatility adjusted
         │
         │ EMA (λ=0.1)
         │
         ▼
    μ̂_t [-1, 1]        ◄── Filtered estimate
         │
         ├──▶ If μ̂_t > 0: Bullish ──▶ Widen sell spread
         └──▶ If μ̂_t < 0: Bearish ──▶ Widen buy spread
```

## Inventory Risk Calculation

```
Current Position (Q_t)
         │
         │ ÷ Max Position
         │
         ▼
    Position Ratio [-1, 1]
         │
         │ abs()
         │
         ▼
    |Position Ratio| [0, 1]
         │
         │ ^2 (Square)
         │
         ▼
    Risk Component [0, 1]
         │
         │ + 1.0
         │
         ▼
    Risk Multiplier [1.0, 2.0]  ◄── Widen spreads as inventory grows
```

## State Vector Update Timeline

```
t=0     t=1     t=2     t=3     t=4     t=5
│       │       │       │       │       │
│       │       │       │       │       │
├─ L2Book       │       │       │       │
│   Update      │       │       │       │
│   • Imbalance │       │       │       │
│   • Spread    │       │       │       │
│   │           │       │       │       │
│   └─▶ State Vector Update    │       │
│       • I_t           │       │       │
│       • Δ_t           │       │       │
│       • μ̂_t (filtered)│       │       │
│                       │       │       │
├─────── AllMids ───────┤       │       │
│        • S_t          │       │       │
│        │              │       │       │
│        └─▶ State Vector Update       │
│            • S_t updated      │       │
│                       │       │       │
├───────────── Fill ────┤       │       │
│              • Q_t    │       │       │
│              │        │       │       │
│              └─▶ State Vector Update │
│                  • Q_t updated│       │
│                       │       │       │
▼                       ▼       ▼       ▼
Continue...
```

## Integration Points

```
┌─────────────────────────────────────────────────────────────┐
│                    Existing Features                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  InventorySkewCalculator  ◄──┐                             │
│  • Provides book analysis     │                             │
│  • Calculates imbalance       │ Integration                 │
│                               │                             │
│  BookAnalyzer                 ├──▶  StateVector            │
│  • Depth analysis             │     • Uses imbalance       │
│  • Imbalance calculation      │     • Uses spread          │
│                               │     • Filters signals      │
│  OrderBook                    │                             │
│  • Best bid/ask               │                             │
│  • Spread calculation     ────┘                             │
│                                                              │
│                                       │                      │
│                                       │ Provides signals     │
│                                       ▼                      │
│                                                              │
│  MarketMaker                   New Methods:                 │
│  • Order placement       ◄──── • get_state_vector()        │
│  • Position management   ◄──── • calculate_spread_adj()    │
│  • Risk management       ◄──── • should_pause_trading()    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Legend

```
Symbols Used:
─────────────
│  ▼  ▶  ◄  ▲  : Flow direction
├─────────────┤ : Box/container borders  
┌─────────────┐ : Component boundaries
───▶           : Data flow
◄────          : Return/feedback
═══▶           : Important flow
••• ▶          : Multiple inputs
```

## Key Relationships

```
S_t (Mid Price)
    └──▶ Base reference for quotes

Q_t (Inventory)
    ├──▶ Risk Multiplier (quadratic)
    └──▶ Urgency Score (cubic)

μ̂_t (Adverse Selection)
    ├──▶ Spread Adjustment
    └──▶ Asymmetric Quoting

Δ_t (Market Spread)
    ├──▶ Signal Scaling (dampening)
    └──▶ Market Condition Check

I_t (LOB Imbalance)
    └──▶ Signal Input for μ̂_t

Combined:
    Z_t = (S_t, Q_t, μ̂_t, Δ_t, I_t) ──▶ Optimal Decision
```
