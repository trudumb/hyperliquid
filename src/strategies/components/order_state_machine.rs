// ============================================================================
// Order State Machine - Proper Order Lifecycle Management
// ============================================================================
//
// Implements a proper state machine for order lifecycle to prevent stuck
// orders (especially in PendingCancel state). Includes timeout handling
// and automatic recovery.

use std::time::{Duration, Instant};
use log::{debug, error, warn};

// ============================================================================
// Order State
// ============================================================================

/// The state of an order in its lifecycle
#[derive(Debug, Clone, PartialEq)]
pub enum OrderState {
    /// Order creation requested but not yet confirmed by exchange
    Creating {
        requested_at: Instant,
        client_order_id: String,
        size: f64,
        price: f64,
        is_buy: bool,
    },

    /// Order is open on the exchange
    Open {
        confirmed_at: Instant,
        order_id: u64,
        remaining_size: f64,
        price: f64,
        is_buy: bool,
    },

    /// Cancel requested but not yet confirmed
    PendingCancel {
        requested_at: Instant,
        order_id: u64,
        original_size: f64,
        remaining_size: f64,
    },

    /// Order was successfully canceled
    Canceled {
        completed_at: Instant,
        order_id: u64,
        filled_size: f64,
    },

    /// Order was fully filled
    Filled {
        completed_at: Instant,
        order_id: u64,
        fill_size: f64,
        fill_price: f64,
    },

    /// Order partially filled and then canceled
    PartiallyFilled {
        completed_at: Instant,
        order_id: u64,
        filled_size: f64,
        canceled_size: f64,
    },

    /// Order failed or stuck in invalid state
    Failed {
        failed_at: Instant,
        order_id: Option<u64>,
        error: String,
    },
}

impl OrderState {
    /// Check if this is a terminal state (order lifecycle is complete)
    pub fn is_terminal(&self) -> bool {
        matches!(self,
            OrderState::Canceled { .. } |
            OrderState::Filled { .. } |
            OrderState::PartiallyFilled { .. } |
            OrderState::Failed { .. }
        )
    }

    /// Check if this order is still active on the exchange
    pub fn is_active(&self) -> bool {
        matches!(self, OrderState::Open { .. })
    }

    /// Get the order ID if available
    pub fn order_id(&self) -> Option<u64> {
        match self {
            OrderState::Creating { .. } => None,
            OrderState::Open { order_id, .. } => Some(*order_id),
            OrderState::PendingCancel { order_id, .. } => Some(*order_id),
            OrderState::Canceled { order_id, .. } => Some(*order_id),
            OrderState::Filled { order_id, .. } => Some(*order_id),
            OrderState::PartiallyFilled { order_id, .. } => Some(*order_id),
            OrderState::Failed { order_id, .. } => *order_id,
        }
    }

    /// Get a human-readable state name
    pub fn state_name(&self) -> &'static str {
        match self {
            OrderState::Creating { .. } => "Creating",
            OrderState::Open { .. } => "Open",
            OrderState::PendingCancel { .. } => "PendingCancel",
            OrderState::Canceled { .. } => "Canceled",
            OrderState::Filled { .. } => "Filled",
            OrderState::PartiallyFilled { .. } => "PartiallyFilled",
            OrderState::Failed { .. } => "Failed",
        }
    }
}

// ============================================================================
// Order Events
// ============================================================================

/// Events that trigger state transitions
#[derive(Debug, Clone)]
pub enum OrderEvent {
    /// Exchange confirmed order creation
    Confirmed {
        order_id: u64,
    },

    /// Order was partially filled
    PartialFill {
        filled_size: f64,
        remaining_size: f64,
        fill_price: f64,
    },

    /// Order was fully filled
    FullFill {
        filled_size: f64,
        fill_price: f64,
    },

    /// Cancel was requested
    CancelRequested,

    /// Exchange confirmed cancellation
    CancelConfirmed {
        filled_size: f64,
    },

    /// Order creation failed
    CreationFailed {
        error: String,
    },

    /// Cancel failed
    CancelFailed {
        error: String,
    },

    /// Periodic tick for timeout checking
    Tick,
}

// ============================================================================
// State Machine Errors
// ============================================================================

#[derive(Debug, Clone)]
pub enum StateError {
    /// Invalid state transition
    InvalidTransition {
        from: String,
        event: String,
    },

    /// State is terminal, no more transitions allowed
    TerminalState,

    /// Timeout occurred
    Timeout {
        state: String,
        elapsed: Duration,
    },
}

impl std::fmt::Display for StateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StateError::InvalidTransition { from, event } => {
                write!(f, "Invalid transition from {} on event {}", from, event)
            }
            StateError::TerminalState => {
                write!(f, "Cannot transition from terminal state")
            }
            StateError::Timeout { state, elapsed } => {
                write!(f, "Timeout in state {} after {:?}", state, elapsed)
            }
        }
    }
}

impl std::error::Error for StateError {}

// ============================================================================
// Order State Machine
// ============================================================================

/// State machine for managing order lifecycle
pub struct OrderStateMachine {
    /// Current state
    state: OrderState,

    /// History of transitions (for debugging)
    transitions: Vec<Transition>,

    /// Maximum number of transitions to keep in history
    max_history: usize,

    /// Timeout durations for different states
    config: StateMachineConfig,
}

/// Configuration for state machine timeouts
#[derive(Debug, Clone)]
pub struct StateMachineConfig {
    /// Timeout for order creation
    pub creation_timeout: Duration,

    /// Timeout for pending cancel
    pub cancel_timeout: Duration,

    /// Enable automatic recovery from stuck states
    pub auto_recovery: bool,
}

impl Default for StateMachineConfig {
    fn default() -> Self {
        Self {
            creation_timeout: Duration::from_secs(10),
            cancel_timeout: Duration::from_secs(5),
            auto_recovery: true,
        }
    }
}

/// A recorded state transition
#[derive(Debug, Clone)]
struct Transition {
    from_state: String,
    to_state: String,
    event: String,
    timestamp: Instant,
}

impl OrderStateMachine {
    /// Create a new state machine in Creating state
    pub fn new(
        client_order_id: String,
        size: f64,
        price: f64,
        is_buy: bool,
        config: StateMachineConfig,
    ) -> Self {
        Self {
            state: OrderState::Creating {
                requested_at: Instant::now(),
                client_order_id,
                size,
                price,
                is_buy,
            },
            transitions: Vec::new(),
            max_history: 50,
            config,
        }
    }

    /// Create a state machine from an existing open order
    pub fn from_open_order(
        order_id: u64,
        size: f64,
        price: f64,
        is_buy: bool,
        config: StateMachineConfig,
    ) -> Self {
        Self {
            state: OrderState::Open {
                confirmed_at: Instant::now(),
                order_id,
                remaining_size: size,
                price,
                is_buy,
            },
            transitions: Vec::new(),
            max_history: 50,
            config,
        }
    }

    /// Get current state
    pub fn state(&self) -> &OrderState {
        &self.state
    }

    /// Transition to a new state based on an event
    pub fn transition(&mut self, event: OrderEvent) -> Result<(), StateError> {
        // Check if terminal
        if self.state.is_terminal() {
            return Err(StateError::TerminalState);
        }

        let old_state_name = self.state.state_name().to_string();
        let event_name = format!("{:?}", event);

        // Determine new state based on current state and event
        let new_state = self.compute_next_state(&event)?;

        // Record transition
        self.transitions.push(Transition {
            from_state: old_state_name.clone(),
            to_state: new_state.state_name().to_string(),
            event: event_name.clone(),
            timestamp: Instant::now(),
        });

        // Trim history if needed
        if self.transitions.len() > self.max_history {
            self.transitions.drain(0..self.transitions.len() - self.max_history);
        }

        debug!(
            "[ORDER STATE] Transition: {} -> {} (event: {})",
            old_state_name,
            new_state.state_name(),
            event_name
        );

        self.state = new_state;
        Ok(())
    }

    /// Compute the next state based on current state and event
    fn compute_next_state(&self, event: &OrderEvent) -> Result<OrderState, StateError> {
        match (&self.state, event) {
            // Creating -> Open
            (OrderState::Creating { .. }, OrderEvent::Confirmed { order_id }) => {
                if let OrderState::Creating { size, price, is_buy, .. } = &self.state {
                    Ok(OrderState::Open {
                        confirmed_at: Instant::now(),
                        order_id: *order_id,
                        remaining_size: *size,
                        price: *price,
                        is_buy: *is_buy,
                    })
                } else {
                    unreachable!()
                }
            }

            // Creating -> Failed
            (OrderState::Creating { .. }, OrderEvent::CreationFailed { error }) => {
                Ok(OrderState::Failed {
                    failed_at: Instant::now(),
                    order_id: None,
                    error: error.clone(),
                })
            }

            // Creating -> timeout check
            (OrderState::Creating { requested_at, .. }, OrderEvent::Tick) => {
                if self.config.auto_recovery && requested_at.elapsed() > self.config.creation_timeout {
                    error!(
                        "[ORDER STATE] Creation timeout after {:?}",
                        requested_at.elapsed()
                    );
                    Ok(OrderState::Failed {
                        failed_at: Instant::now(),
                        order_id: None,
                        error: "Creation timeout".to_string(),
                    })
                } else {
                    Ok(self.state.clone())
                }
            }

            // Open -> PartialFill (stays Open with updated size)
            (OrderState::Open { order_id, price, is_buy, .. }, OrderEvent::PartialFill { remaining_size, .. }) => {
                Ok(OrderState::Open {
                    confirmed_at: Instant::now(),
                    order_id: *order_id,
                    remaining_size: *remaining_size,
                    price: *price,
                    is_buy: *is_buy,
                })
            }

            // Open -> Filled
            (OrderState::Open { order_id, .. }, OrderEvent::FullFill { filled_size, fill_price }) => {
                Ok(OrderState::Filled {
                    completed_at: Instant::now(),
                    order_id: *order_id,
                    fill_size: *filled_size,
                    fill_price: *fill_price,
                })
            }

            // Open -> PendingCancel
            (OrderState::Open { order_id, remaining_size, .. }, OrderEvent::CancelRequested) => {
                Ok(OrderState::PendingCancel {
                    requested_at: Instant::now(),
                    order_id: *order_id,
                    original_size: *remaining_size,
                    remaining_size: *remaining_size,
                })
            }

            // PendingCancel -> Canceled
            (OrderState::PendingCancel { order_id, original_size, .. }, OrderEvent::CancelConfirmed { filled_size }) => {
                if *filled_size > 0.0 && *filled_size < *original_size {
                    Ok(OrderState::PartiallyFilled {
                        completed_at: Instant::now(),
                        order_id: *order_id,
                        filled_size: *filled_size,
                        canceled_size: *original_size - *filled_size,
                    })
                } else {
                    Ok(OrderState::Canceled {
                        completed_at: Instant::now(),
                        order_id: *order_id,
                        filled_size: *filled_size,
                    })
                }
            }

            // PendingCancel -> Failed (cancel failed)
            (OrderState::PendingCancel { order_id, .. }, OrderEvent::CancelFailed { error }) => {
                warn!(
                    "[ORDER STATE] Cancel failed for order {}: {}",
                    order_id, error
                );
                Ok(OrderState::Failed {
                    failed_at: Instant::now(),
                    order_id: Some(*order_id),
                    error: format!("Cancel failed: {}", error),
                })
            }

            // PendingCancel -> timeout check
            (OrderState::PendingCancel { requested_at, order_id, .. }, OrderEvent::Tick) => {
                if self.config.auto_recovery && requested_at.elapsed() > self.config.cancel_timeout {
                    error!(
                        "[ORDER STATE] Order {} stuck in PendingCancel for {:?}, forcing cleanup",
                        order_id,
                        requested_at.elapsed()
                    );
                    Ok(OrderState::Failed {
                        failed_at: Instant::now(),
                        order_id: Some(*order_id),
                        error: "Cancel timeout".to_string(),
                    })
                } else {
                    Ok(self.state.clone())
                }
            }

            // PendingCancel -> Filled (filled while cancel was pending)
            (OrderState::PendingCancel { order_id, original_size, .. }, OrderEvent::FullFill { fill_price, .. }) => {
                Ok(OrderState::Filled {
                    completed_at: Instant::now(),
                    order_id: *order_id,
                    fill_size: *original_size,
                    fill_price: *fill_price,
                })
            }

            // Invalid transitions
            _ => Err(StateError::InvalidTransition {
                from: self.state.state_name().to_string(),
                event: format!("{:?}", event),
            }),
        }
    }

    /// Perform periodic tick to check for timeouts
    pub fn tick(&mut self) -> Result<(), StateError> {
        self.transition(OrderEvent::Tick)
    }

    /// Get transition history
    pub fn get_transitions(&self) -> &[Transition] {
        &self.transitions
    }

    /// Check if order has been stuck too long in current state
    pub fn is_stuck(&self) -> Option<Duration> {
        match &self.state {
            OrderState::Creating { requested_at, .. } => {
                let elapsed = requested_at.elapsed();
                if elapsed > self.config.creation_timeout {
                    Some(elapsed)
                } else {
                    None
                }
            }
            OrderState::PendingCancel { requested_at, .. } => {
                let elapsed = requested_at.elapsed();
                if elapsed > self.config.cancel_timeout {
                    Some(elapsed)
                } else {
                    None
                }
            }
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_order_lifecycle_success() {
        let config = StateMachineConfig::default();
        let mut sm = OrderStateMachine::new(
            "cloid123".to_string(),
            10.0,
            100.0,
            true,
            config,
        );

        // Creating -> Open
        assert!(sm.transition(OrderEvent::Confirmed { order_id: 1 }).is_ok());
        assert!(matches!(sm.state(), OrderState::Open { .. }));

        // Open -> PendingCancel
        assert!(sm.transition(OrderEvent::CancelRequested).is_ok());
        assert!(matches!(sm.state(), OrderState::PendingCancel { .. }));

        // PendingCancel -> Canceled
        assert!(sm.transition(OrderEvent::CancelConfirmed { filled_size: 0.0 }).is_ok());
        assert!(matches!(sm.state(), OrderState::Canceled { .. }));
    }

    #[test]
    fn test_partial_fill() {
        let config = StateMachineConfig::default();
        let mut sm = OrderStateMachine::new(
            "cloid123".to_string(),
            10.0,
            100.0,
            true,
            config,
        );

        // Creating -> Open
        sm.transition(OrderEvent::Confirmed { order_id: 1 }).unwrap();

        // Partial fill
        sm.transition(OrderEvent::PartialFill {
            filled_size: 5.0,
            remaining_size: 5.0,
            fill_price: 100.0,
        }).unwrap();

        if let OrderState::Open { remaining_size, .. } = sm.state() {
            assert_eq!(*remaining_size, 5.0);
        } else {
            panic!("Expected Open state");
        }

        // Cancel partially filled order
        sm.transition(OrderEvent::CancelRequested).unwrap();
        sm.transition(OrderEvent::CancelConfirmed { filled_size: 5.0 }).unwrap();

        assert!(matches!(sm.state(), OrderState::PartiallyFilled { .. }));
    }

    #[test]
    fn test_invalid_transition() {
        let config = StateMachineConfig::default();
        let mut sm = OrderStateMachine::new(
            "cloid123".to_string(),
            10.0,
            100.0,
            true,
            config,
        );

        // Can't cancel before confirming
        let result = sm.transition(OrderEvent::CancelRequested);
        assert!(result.is_err());
    }

    #[test]
    fn test_terminal_state() {
        let config = StateMachineConfig::default();
        let mut sm = OrderStateMachine::new(
            "cloid123".to_string(),
            10.0,
            100.0,
            true,
            config,
        );

        sm.transition(OrderEvent::Confirmed { order_id: 1 }).unwrap();
        sm.transition(OrderEvent::FullFill {
            filled_size: 10.0,
            fill_price: 100.0,
        }).unwrap();

        // Can't transition from terminal state
        let result = sm.transition(OrderEvent::CancelRequested);
        assert!(matches!(result, Err(StateError::TerminalState)));
    }
}
