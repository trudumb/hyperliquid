use futures::future::join_all;
use parking_lot::Mutex;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Semaphore;
use tokio::time::timeout;

use crate::{
    exchange::{
        ClientCancelRequest, ClientCancelRequestCloid, ClientModifyRequest, ClientOrderRequest,
        ExchangeClient, ExchangeResponseStatus,
    },
    prelude::*,
    rate_limiter::{RateLimitConfig, RateLimiter, RequestWeight},
    Error,
};

/// Strategy action types for batch processing
///
/// Used to categorize different exchange operations that can be batched together
#[derive(Debug, Clone)]
pub enum StrategyAction {
    /// Place a single order
    Place(ClientOrderRequest),
    /// Place multiple orders in a batch
    BatchPlace(Vec<ClientOrderRequest>),
    /// Cancel a single order by OID
    Cancel(ClientCancelRequest),
    /// Cancel multiple orders by OID in a batch
    BatchCancel(Vec<ClientCancelRequest>),
    /// Cancel a single order by CLOID (more reliable)
    CancelByCloid(ClientCancelRequestCloid),
    /// Cancel multiple orders by CLOID in a batch (more reliable)
    BatchCancelByCloid(Vec<ClientCancelRequestCloid>),
    /// Modify a single order
    Modify(ClientModifyRequest),
    /// Modify multiple orders in a batch
    BatchModify(Vec<ClientModifyRequest>),
}

/// Result of a parallel execution batch
#[derive(Debug)]
pub struct BatchExecutionResult {
    /// Number of successful operations
    pub successful: usize,
    /// Number of failed operations
    pub failed: usize,
    /// Total execution time in milliseconds
    pub execution_time_ms: u64,
    /// Individual responses from the exchange
    pub responses: Vec<Result<ExchangeResponseStatus>>,
}

/// Configuration for the parallel order executor
#[derive(Debug, Clone)]
pub struct ExecutorConfig {
    /// Maximum number of concurrent requests to the exchange
    pub max_concurrent: usize,
    /// Time window for batching operations (microseconds)
    pub batch_window_us: u64,
    /// Maximum timeout for each request (milliseconds)
    pub request_timeout_ms: u64,
    /// Maximum timeout for entire batch (milliseconds)
    pub batch_timeout_ms: u64,
    /// Rate limiter configuration
    pub rate_limit_config: RateLimitConfig,
}

impl Default for ExecutorConfig {
    fn default() -> Self {
        Self {
            max_concurrent: 10,          // 10 concurrent requests
            batch_window_us: 100,        // 100μs batching window
            request_timeout_ms: 500,     // 500ms per request (matches HTTP timeout)
            batch_timeout_ms: 2000,      // 2s for entire batch
            rate_limit_config: RateLimitConfig::rest_api(), // Default REST API limits
        }
    }
}

/// High-performance parallel order executor with smart batching
///
/// This executor optimizes HFT operations by:
/// - Batching similar operations together
/// - Limiting concurrent requests with semaphores
/// - Executing operations in parallel
/// - Providing timeout protection
/// - Reducing API calls through intelligent grouping
///
/// # Example
/// ```no_run
/// use hyperliquid_rust_sdk::prelude::*;
/// use std::sync::Arc;
///
/// async fn example() -> Result<()> {
///     let exchange = Arc::new(ExchangeClient::new(/* ... */).await?);
///     let executor = ParallelOrderExecutor::new(exchange, 10);
///
///     let actions = vec![
///         StrategyAction::Place(order1),
///         StrategyAction::Place(order2),
///         StrategyAction::Cancel(cancel1),
///     ];
///
///     let result = executor.execute_actions_parallel(actions).await;
///     println!("Executed {} operations", result.successful + result.failed);
///     Ok(())
/// }
/// ```
pub struct ParallelOrderExecutor {
    exchange: Arc<ExchangeClient>,
    semaphore: Arc<Semaphore>,
    rate_limiter: RateLimiter,
    config: ExecutorConfig,
    /// Statistics tracking
    stats: Arc<Mutex<ExecutorStats>>,
}

#[derive(Debug, Default, Clone)]
pub struct ExecutorStats {
    pub total_actions: usize,
    pub total_batches: usize,
    pub total_successful: usize,
    pub total_failed: usize,
    pub avg_batch_size: f64,
}

impl ParallelOrderExecutor {
    /// Create a new ParallelOrderExecutor with default configuration
    ///
    /// # Arguments
    /// * `exchange` - Arc-wrapped ExchangeClient
    /// * `max_concurrent` - Maximum number of concurrent requests
    pub fn new(exchange: Arc<ExchangeClient>, max_concurrent: usize) -> Self {
        let mut config = ExecutorConfig::default();
        config.max_concurrent = max_concurrent;
        Self::with_config(exchange, config)
    }

    /// Create a ParallelOrderExecutor with custom configuration
    ///
    /// # Arguments
    /// * `exchange` - Arc-wrapped ExchangeClient
    /// * `config` - Custom executor configuration
    pub fn with_config(exchange: Arc<ExchangeClient>, config: ExecutorConfig) -> Self {
        Self {
            exchange,
            semaphore: Arc::new(Semaphore::new(config.max_concurrent)),
            rate_limiter: RateLimiter::new(config.rate_limit_config.clone()),
            config,
            stats: Arc::new(Mutex::new(ExecutorStats::default())),
        }
    }

    /// Execute multiple strategy actions in parallel with smart batching
    ///
    /// This method:
    /// 1. Separates actions by type (place, cancel, modify)
    /// 2. Batches similar actions together
    /// 3. Executes batches in parallel with semaphore limiting
    /// 4. Returns aggregated results with statistics
    ///
    /// # Arguments
    /// * `actions` - Vector of strategy actions to execute
    ///
    /// # Returns
    /// BatchExecutionResult with success/failure counts and responses
    pub async fn execute_actions_parallel(
        &self,
        actions: Vec<StrategyAction>,
    ) -> BatchExecutionResult {
        let start_time = std::time::Instant::now();

        // Separate actions by type for intelligent batching
        let mut places = Vec::new();
        let mut cancels = Vec::new();
        let mut cancels_by_cloid = Vec::new();
        let mut modifies = Vec::new();

        for action in actions {
            match action {
                StrategyAction::Place(order) => places.push(order),
                StrategyAction::BatchPlace(orders) => places.extend(orders),
                StrategyAction::Cancel(cancel) => cancels.push(cancel),
                StrategyAction::BatchCancel(cancels_batch) => cancels.extend(cancels_batch),
                StrategyAction::CancelByCloid(cancel) => cancels_by_cloid.push(cancel),
                StrategyAction::BatchCancelByCloid(cancels_batch) => cancels_by_cloid.extend(cancels_batch),
                StrategyAction::Modify(modify) => modifies.push(modify),
                StrategyAction::BatchModify(modifies_batch) => modifies.extend(modifies_batch),
            }
        }

        let total_actions = places.len() + cancels.len() + cancels_by_cloid.len() + modifies.len();

        // Execute batches in parallel
        let mut handles = Vec::new();

        // Batch place orders
        if !places.is_empty() {
            let batch_length = places.len();
            let permit = self.semaphore.clone().acquire_owned().await.ok();
            let exchange = self.exchange.clone();
            let rate_limiter = self.rate_limiter.clone();
            let timeout_ms = self.config.request_timeout_ms;

            handles.push(tokio::spawn(async move {
                // Acquire rate limit capacity for this batch
                rate_limiter.acquire(RequestWeight::Exchange { batch_length }, 0).await;

                let result = timeout(
                    Duration::from_millis(timeout_ms),
                    exchange.bulk_order(places, None),
                )
                .await;

                drop(permit); // Release semaphore permit

                match result {
                    Ok(Ok(response)) => Ok(response),
                    Ok(Err(e)) => Err(e),
                    Err(_) => Err(Error::GenericRequest("Request timeout".to_string())),
                }
            }));
        }

        // Batch cancel orders (by OID)
        if !cancels.is_empty() {
            let batch_length = cancels.len();
            let permit = self.semaphore.clone().acquire_owned().await.ok();
            let exchange = self.exchange.clone();
            let rate_limiter = self.rate_limiter.clone();
            let timeout_ms = self.config.request_timeout_ms;

            handles.push(tokio::spawn(async move {
                // Acquire rate limit capacity for this batch
                rate_limiter.acquire(RequestWeight::Exchange { batch_length }, 0).await;

                let result = timeout(
                    Duration::from_millis(timeout_ms),
                    exchange.bulk_cancel(cancels, None),
                )
                .await;

                drop(permit);

                match result {
                    Ok(Ok(response)) => Ok(response),
                    Ok(Err(e)) => Err(e),
                    Err(_) => Err(Error::GenericRequest("Request timeout".to_string())),
                }
            }));
        }

        // Batch cancel orders (by CLOID - more reliable)
        if !cancels_by_cloid.is_empty() {
            let batch_length = cancels_by_cloid.len();
            let permit = self.semaphore.clone().acquire_owned().await.ok();
            let exchange = self.exchange.clone();
            let rate_limiter = self.rate_limiter.clone();
            let timeout_ms = self.config.request_timeout_ms;

            handles.push(tokio::spawn(async move {
                // Acquire rate limit capacity for this batch
                rate_limiter.acquire(RequestWeight::Exchange { batch_length }, 0).await;

                let result = timeout(
                    Duration::from_millis(timeout_ms),
                    exchange.bulk_cancel_by_cloid(cancels_by_cloid, None),
                )
                .await;

                drop(permit);

                match result {
                    Ok(Ok(response)) => Ok(response),
                    Ok(Err(e)) => Err(e),
                    Err(_) => Err(Error::GenericRequest("Request timeout".to_string())),
                }
            }));
        }

        // Batch modify orders
        if !modifies.is_empty() {
            let batch_length = modifies.len();
            let permit = self.semaphore.clone().acquire_owned().await.ok();
            let exchange = self.exchange.clone();
            let rate_limiter = self.rate_limiter.clone();
            let timeout_ms = self.config.request_timeout_ms;

            handles.push(tokio::spawn(async move {
                // Acquire rate limit capacity for this batch
                rate_limiter.acquire(RequestWeight::Exchange { batch_length }, 0).await;

                let result = timeout(
                    Duration::from_millis(timeout_ms),
                    exchange.bulk_modify(modifies, None),
                )
                .await;

                drop(permit);

                match result {
                    Ok(Ok(response)) => Ok(response),
                    Ok(Err(e)) => Err(e),
                    Err(_) => Err(Error::GenericRequest("Request timeout".to_string())),
                }
            }));
        }

        // Wait for all operations to complete with global timeout
        let batch_timeout = Duration::from_millis(self.config.batch_timeout_ms);
        let results = timeout(batch_timeout, join_all(handles)).await;

        let responses = match results {
            Ok(join_results) => join_results
                .into_iter()
                .map(|r| match r {
                    Ok(result) => result,
                    Err(_) => Err(Error::GenericRequest("Task panicked".to_string())),
                })
                .collect::<Vec<_>>(),
            Err(_) => {
                vec![Err(Error::GenericRequest(
                    "Batch timeout exceeded".to_string(),
                ))]
            }
        };

        let successful = responses.iter().filter(|r| r.is_ok()).count();
        let failed = responses.iter().filter(|r| r.is_err()).count();

        let execution_time_ms = start_time.elapsed().as_millis() as u64;

        // Update statistics
        {
            let mut stats = self.stats.lock();
            stats.total_actions += total_actions;
            stats.total_batches += 1;
            stats.total_successful += successful;
            stats.total_failed += failed;
            stats.avg_batch_size =
                stats.total_actions as f64 / stats.total_batches.max(1) as f64;
        }

        BatchExecutionResult {
            successful,
            failed,
            execution_time_ms,
            responses,
        }
    }

    /// Execute a single action immediately without batching
    ///
    /// Useful for urgent orders that can't wait for batch window
    ///
    /// # Arguments
    /// * `action` - Single strategy action to execute
    ///
    /// # Returns
    /// Result from the exchange
    pub async fn execute_single_immediate(
        &self,
        action: StrategyAction,
    ) -> Result<ExchangeResponseStatus> {
        let _permit = self.semaphore.acquire().await.ok();

        let timeout_duration = Duration::from_millis(self.config.request_timeout_ms);

        match action {
            StrategyAction::Place(order) => {
                timeout(timeout_duration, self.exchange.order(order, None))
                    .await
                    .map_err(|_| Error::GenericRequest("Request timeout".to_string()))?
            }
            StrategyAction::BatchPlace(orders) => {
                timeout(timeout_duration, self.exchange.bulk_order(orders, None))
                    .await
                    .map_err(|_| Error::GenericRequest("Request timeout".to_string()))?
            }
            StrategyAction::Cancel(cancel) => {
                timeout(timeout_duration, self.exchange.cancel(cancel, None))
                    .await
                    .map_err(|_| Error::GenericRequest("Request timeout".to_string()))?
            }
            StrategyAction::BatchCancel(cancels) => {
                timeout(timeout_duration, self.exchange.bulk_cancel(cancels, None))
                    .await
                    .map_err(|_| Error::GenericRequest("Request timeout".to_string()))?
            }
            StrategyAction::CancelByCloid(cancel) => {
                timeout(timeout_duration, self.exchange.cancel_by_cloid(cancel, None))
                    .await
                    .map_err(|_| Error::GenericRequest("Request timeout".to_string()))?
            }
            StrategyAction::BatchCancelByCloid(cancels) => {
                timeout(timeout_duration, self.exchange.bulk_cancel_by_cloid(cancels, None))
                    .await
                    .map_err(|_| Error::GenericRequest("Request timeout".to_string()))?
            }
            StrategyAction::Modify(modify) => {
                timeout(timeout_duration, self.exchange.modify(modify, None))
                    .await
                    .map_err(|_| Error::GenericRequest("Request timeout".to_string()))?
            }
            StrategyAction::BatchModify(modifies) => {
                timeout(timeout_duration, self.exchange.bulk_modify(modifies, None))
                    .await
                    .map_err(|_| Error::GenericRequest("Request timeout".to_string()))?
            }
        }
    }

    /// Get executor statistics
    pub fn get_stats(&self) -> ExecutorStats {
        self.stats.lock().clone()
    }

    /// Reset executor statistics
    pub fn reset_stats(&self) {
        let mut stats = self.stats.lock();
        *stats = ExecutorStats::default();
    }

    /// Get current configuration
    pub fn config(&self) -> &ExecutorConfig {
        &self.config
    }

    /// Get number of available semaphore permits
    pub fn available_permits(&self) -> usize {
        self.semaphore.available_permits()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strategy_action_batching() {
        let actions = vec![
            StrategyAction::BatchPlace(vec![]),
            StrategyAction::Place(ClientOrderRequest {
                asset: "BTC".to_string(),
                is_buy: true,
                reduce_only: false,
                limit_px: 50000.0,
                sz: 1.0,
                cloid: None,
                order_type: crate::exchange::ClientOrder::Limit(
                    crate::exchange::ClientLimit {
                        tif: "Gtc".to_string(),
                    },
                ),
            }),
        ];

        assert_eq!(actions.len(), 2);
    }

    #[test]
    fn test_executor_config_defaults() {
        let config = ExecutorConfig::default();
        assert_eq!(config.max_concurrent, 10);
        assert_eq!(config.batch_window_us, 100);
        assert_eq!(config.request_timeout_ms, 500);
        assert_eq!(config.batch_timeout_ms, 2000);
    }

    #[test]
    fn test_executor_stats_tracking() {
        let stats = ExecutorStats::default();
        assert_eq!(stats.total_actions, 0);
        assert_eq!(stats.total_batches, 0);
        assert_eq!(stats.total_successful, 0);
        assert_eq!(stats.total_failed, 0);
        assert_eq!(stats.avg_batch_size, 0.0);
    }

    #[test]
    fn test_batch_execution_result() {
        let result = BatchExecutionResult {
            successful: 5,
            failed: 2,
            execution_time_ms: 150,
            responses: vec![],
        };

        assert_eq!(result.successful, 5);
        assert_eq!(result.failed, 2);
        assert_eq!(result.execution_time_ms, 150);
    }
}
