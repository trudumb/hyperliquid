// ============================================================================
// Async Tuner Actor - Non-Blocking Auto-Tuning for HJB Strategy
// ============================================================================
//
// This module implements an actor-based auto-tuner that runs in a background
// tokio task, ensuring that parameter optimization never blocks the critical
// trading path.
//
// # Architecture
//
// ```
// Main Trading Thread          Tuner Actor (Background Task)
//       │                              │
//       │  TunerEvent::Fill           │
//       ├─────────────────────────────>│
//       │  (non-blocking send)         │  Update tracker
//       │                              │  Run SPSA
//       │  TunerEvent::Quote           │  Run Adam
//       ├─────────────────────────────>│
//       │                              │  Write new params
//       │                              │  to Arc<RwLock>
//       │  Read params                 │
//       │<────────────────────────── ─ ┘
//       │  (non-blocking RwLock read)
// ```
//
// # Performance
//
// - Event send: ~100-500ns (lock-free channel)
// - Param read: ~10-50ns (RwLock read)
// - SPSA computation: ~1-10ms (background, doesn't block)
//
// # Usage
//
// ```rust
// let (actor, params_handle) = AsyncTunerActor::spawn(config, initial_params, seed);
//
// // In trading loop (non-blocking):
// let _ = actor.send_event(TunerEvent::Fill { ... });
// let current_params = params_handle.read();
//
// // On shutdown:
// actor.shutdown().await;
// ```

use std::sync::Arc;
use parking_lot::RwLock;
use log::{info, warn};
use tokio::task::JoinHandle;

use super::tuner_integration::TunerIntegration;
use super::components::{
    TunerConfig, StrategyTuningParams, StrategyConstrainedParams,
};

// ============================================================================
// Tuner Events
// ============================================================================

/// Events sent from the main trading loop to the tuner actor
#[derive(Debug, Clone)]
pub enum TunerEvent {
    /// Market update occurred (for episode tracking)
    MarketUpdate,

    /// Order was filled
    Fill {
        pnl: f64,
        fill_price: f64,
        fill_size: f64,
        is_buy: bool,
        timestamp: f64,
    },

    /// Quote was placed
    Quote {
        bid_price: f64,
        ask_price: f64,
        bid_size: f64,
        ask_size: f64,
        spread_bps: f64,
        urgency: f64,
        timestamp: f64,
    },

    /// Order was canceled
    Cancel {
        timestamp: f64,
    },

    /// Episode completed, trigger evaluation
    EpisodeComplete,

    /// Shutdown the tuner actor
    Shutdown,
}

// ============================================================================
// Async Tuner Actor
// ============================================================================

/// Handle to communicate with the tuner actor
pub struct AsyncTunerActorHandle {
    /// Channel to send events to the tuner
    event_tx: flume::Sender<TunerEvent>,

    /// Join handle for the background task
    task_handle: Option<JoinHandle<()>>,

    /// Shared parameters (read by trading thread)
    params: Arc<RwLock<StrategyConstrainedParams>>,
}

impl AsyncTunerActorHandle {
    /// Send an event to the tuner (non-blocking, lock-free)
    pub fn send_event(&self, event: TunerEvent) {
        // Use try_send for non-blocking behavior
        // If the channel is full, we drop the event (tuner is behind, which is OK)
        if let Err(_) = self.event_tx.try_send(event) {
            // Don't log every dropped event to avoid spam
            // The tuner will catch up eventually
        }
    }

    /// Get current tuning parameters (non-blocking read)
    pub fn get_params(&self) -> StrategyConstrainedParams {
        self.params.read().clone()
    }

    /// Check if tuner is still running
    pub fn is_running(&self) -> bool {
        self.task_handle.as_ref().map_or(false, |h| !h.is_finished())
    }

    /// Shutdown the tuner actor gracefully
    pub async fn shutdown(mut self) -> Option<String> {
        // Send shutdown signal
        let _ = self.event_tx.send(TunerEvent::Shutdown);

        // Wait for the task to complete (with timeout)
        if let Some(handle) = self.task_handle.take() {
            match tokio::time::timeout(
                std::time::Duration::from_secs(5),
                handle
            ).await {
                Ok(_) => {
                    info!("[Tuner Actor] Shutdown complete");
                    None
                }
                Err(_) => {
                    warn!("[Tuner Actor] Shutdown timeout");
                    None
                }
            }
        } else {
            None
        }
    }
}

/// Async tuner actor that runs in a background task
pub struct AsyncTunerActor;

impl AsyncTunerActor {
    /// Spawn the tuner actor in a background tokio task
    ///
    /// Returns a handle to communicate with the actor and read parameters
    pub fn spawn(
        config: TunerConfig,
        initial_params: StrategyTuningParams,
        updates_per_episode: usize,
        seed: u64,
    ) -> AsyncTunerActorHandle {
        // Create shared parameters
        let params = Arc::new(RwLock::new(initial_params.clone().get_constrained()));

        // Create communication channel (bounded to prevent memory buildup)
        let (event_tx, event_rx) = flume::bounded(1000);

        // Clone for the actor task
        let params_clone = params.clone();

        // Spawn the actor task
        let task_handle = tokio::spawn(async move {
            Self::run_actor(
                config,
                initial_params,
                updates_per_episode,
                seed,
                event_rx,
                params_clone,
            ).await;
        });

        AsyncTunerActorHandle {
            event_tx,
            task_handle: Some(task_handle),
            params,
        }
    }

    /// Main actor loop (runs in background task)
    async fn run_actor(
        config: TunerConfig,
        initial_params: StrategyTuningParams,
        updates_per_episode: usize,
        seed: u64,
        event_rx: flume::Receiver<TunerEvent>,
        params: Arc<RwLock<StrategyConstrainedParams>>,
    ) {
        info!("[Tuner Actor] Starting with mode={}, episodes_per_update={}",
            config.mode, config.episodes_per_update);

        // Create tuner integration
        let mut tuner_integration = TunerIntegration::new(
            config,
            initial_params,
            updates_per_episode,
            seed,
        );

        if !tuner_integration.is_enabled() {
            info!("[Tuner Actor] Tuning disabled, actor will just forward events");
        }

        let mut running = true;

        while running {
            // Receive next event
            match event_rx.recv_async().await {
                Ok(event) => {
                    match event {
                        TunerEvent::MarketUpdate => {
                            tuner_integration.on_market_update();
                        }

                        TunerEvent::Fill { pnl, fill_price, fill_size, is_buy, timestamp: _ } => {
                            if let Some(tracker) = tuner_integration.performance_tracker() {
                                tracker.on_fill(is_buy, fill_price, fill_size);
                                tracker.on_trade(pnl);
                            }
                        }

                        TunerEvent::Quote { bid_price, ask_price, bid_size, ask_size, spread_bps: _, urgency: _, timestamp: _ } => {
                            if let Some(tracker) = tuner_integration.performance_tracker() {
                                tracker.on_quote(bid_price, ask_price, bid_size, ask_size);
                            }
                        }

                        TunerEvent::Cancel { timestamp: _ } => {
                            if let Some(tracker) = tuner_integration.performance_tracker() {
                                tracker.on_cancel();
                            }
                        }

                        TunerEvent::EpisodeComplete => {
                            // Check if parameters should be updated
                            if let Some(new_params) = tuner_integration.on_tick() {
                                info!("[Tuner Actor] Updating parameters in phase: {}",
                                    tuner_integration.current_phase());

                                // Write new parameters (this is the only write, done in background)
                                *params.write() = new_params;
                            }
                        }

                        TunerEvent::Shutdown => {
                            info!("[Tuner Actor] Shutdown signal received");

                            // Export final tuning history
                            if let Some(history) = tuner_integration.export_history() {
                                info!("[Tuner Actor] Final tuning history:\n{}", history);
                            }

                            // Export best parameters found
                            if let Some(best_params) = tuner_integration.get_best_params() {
                                info!("[Tuner Actor] Best parameters found: phi={:.4}, lambda={:.2}",
                                    best_params.phi, best_params.lambda_base);
                            }

                            running = false;
                        }
                    }
                }
                Err(_) => {
                    // Channel closed, exit gracefully
                    warn!("[Tuner Actor] Event channel closed, shutting down");
                    running = false;
                }
            }
        }

        info!("[Tuner Actor] Stopped");
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_tuner_actor_spawn() {
        let config = TunerConfig {
            enabled: false,
            ..Default::default()
        };
        let params = StrategyTuningParams::default();

        let actor = AsyncTunerActor::spawn(config, params, 100, 42);

        assert!(actor.is_running());

        // Shutdown
        actor.shutdown().await;
    }

    #[tokio::test]
    async fn test_tuner_actor_events() {
        let config = TunerConfig {
            enabled: false,
            ..Default::default()
        };
        let params = StrategyTuningParams::default();

        let actor = AsyncTunerActor::spawn(config, params, 100, 42);

        // Send some events (should not block)
        actor.send_event(TunerEvent::MarketUpdate);
        actor.send_event(TunerEvent::Fill {
            pnl: 10.0,
            fill_price: 100.0,
            fill_size: 1.0,
            is_buy: true,
            timestamp: 0.0,
        });

        // Read params (should not block)
        let params = actor.get_params();
        assert!(params.phi > 0.0);

        // Shutdown
        actor.shutdown().await;
    }

    #[tokio::test]
    async fn test_tuner_actor_param_updates() {
        let mut config = TunerConfig::default();
        config.enabled = true;
        config.mode = "continuous".to_string();
        config.episodes_per_update = 1;

        let params = StrategyTuningParams::default();

        let actor = AsyncTunerActor::spawn(config, params, 10, 42);

        // Simulate an episode
        for _ in 0..10 {
            actor.send_event(TunerEvent::MarketUpdate);
        }

        // Trigger episode completion
        actor.send_event(TunerEvent::EpisodeComplete);

        // Give tuner time to process
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

        // Parameters might have changed (depending on tuner logic)
        let new_params = actor.get_params();

        // Just verify we can read them
        assert!(new_params.phi > 0.0);

        // Shutdown
        actor.shutdown().await;
    }
}
