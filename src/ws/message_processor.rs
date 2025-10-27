/// Inline message processing utilities for zero-copy WebSocket handling
///
/// This module provides utilities for processing WebSocket messages without
/// unnecessary cloning, optimized for high-frequency trading applications.
///
/// # Benefits
/// - Avoids cloning message data when possible
/// - Processes messages inline with pattern matching
/// - Provides type-safe access to message data
/// - Reduces memory allocations

use crate::ws::message_types::{AllMids, L2Book, Message, Trades, User};

/// Process a message inline without cloning
///
/// This function pattern matches on the message and calls the appropriate
/// processor function with a reference to avoid cloning.
///
/// # Example
/// ```no_run
/// use hyperliquid_rust_sdk::ws::message_processor::process_message_inline;
///
/// process_message_inline(&message, |l2_book| {
///     // Process L2 book data inline
///     println!("Mid price: {}", l2_book.data.coin);
/// }, |trades| {
///     // Process trades data inline
///     println!("Trades count: {}", trades.data.len());
/// }, |all_mids| {
///     // Process all mids data inline
/// }, |user| {
///     // Process user events inline
/// });
/// ```
pub fn process_message_inline<F1, F2, F3, F4>(
    message: &Message,
    l2_book_handler: F1,
    trades_handler: F2,
    all_mids_handler: F3,
    user_handler: F4,
) where
    F1: FnOnce(&L2Book),
    F2: FnOnce(&Trades),
    F3: FnOnce(&AllMids),
    F4: FnOnce(&User),
{
    match message {
        Message::L2Book(ref l2_book) => l2_book_handler(l2_book),
        Message::Trades(ref trades) => trades_handler(trades),
        Message::AllMids(ref all_mids) => all_mids_handler(all_mids),
        Message::User(ref user) => user_handler(user),
        _ => {} // Ignore other message types
    }
}

/// Extract L2 book data without cloning
///
/// Returns None if message is not an L2Book
#[inline]
pub fn extract_l2_book(message: &Message) -> Option<&L2Book> {
    if let Message::L2Book(ref l2_book) = message {
        Some(l2_book)
    } else {
        None
    }
}

/// Extract trades data without cloning
///
/// Returns None if message is not Trades
#[inline]
pub fn extract_trades(message: &Message) -> Option<&Trades> {
    if let Message::Trades(ref trades) = message {
        Some(trades)
    } else {
        None
    }
}

/// Extract all mids data without cloning
///
/// Returns None if message is not AllMids
#[inline]
pub fn extract_all_mids(message: &Message) -> Option<&AllMids> {
    if let Message::AllMids(ref all_mids) = message {
        Some(all_mids)
    } else {
        None
    }
}

/// Extract user events without cloning
///
/// Returns None if message is not User
#[inline]
pub fn extract_user_events(message: &Message) -> Option<&User> {
    if let Message::User(ref user) = message {
        Some(user)
    } else {
        None
    }
}

/// Message processor trait for custom inline processing
///
/// Implement this trait to create custom zero-copy message handlers
///
/// # Example
/// ```no_run
/// use hyperliquid_rust_sdk::ws::message_processor::MessageProcessor;
/// use hyperliquid_rust_sdk::ws::message_types::L2Book;
///
/// struct MyProcessor {
///     l2_count: u64,
/// }
///
/// impl MessageProcessor for MyProcessor {
///     fn process_l2_book_inline(&mut self, l2_book: &L2Book) {
///         self.l2_count += 1;
///         // Process inline without cloning
///     }
///
///     fn process_trades_inline(&mut self, trades: &hyperliquid_rust_sdk::ws::message_types::Trades) {
///         // Process trades inline
///     }
/// }
/// ```
pub trait MessageProcessor {
    /// Process L2 book updates inline
    fn process_l2_book_inline(&mut self, l2_book: &L2Book) {
        let _ = l2_book;
    }

    /// Process trade updates inline
    fn process_trades_inline(&mut self, trades: &Trades) {
        let _ = trades;
    }

    /// Process all mids updates inline
    fn process_all_mids_inline(&mut self, all_mids: &AllMids) {
        let _ = all_mids;
    }

    /// Process user events inline
    fn process_user_events_inline(&mut self, user: &User) {
        let _ = user;
    }

    /// Dispatch message to appropriate inline processor
    fn dispatch_message(&mut self, message: &Message) {
        match message {
            Message::L2Book(ref l2_book) => self.process_l2_book_inline(l2_book),
            Message::Trades(ref trades) => self.process_trades_inline(trades),
            Message::AllMids(ref all_mids) => self.process_all_mids_inline(all_mids),
            Message::User(ref user) => self.process_user_events_inline(user),
            _ => {} // Ignore other message types
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    // Test with message data

    #[test]
    fn test_extract_l2_book() {
        let message = Message::Pong;
        assert!(extract_l2_book(&message).is_none());
    }

    #[test]
    fn test_extract_trades() {
        let message = Message::Pong;
        assert!(extract_trades(&message).is_none());
    }

    struct TestProcessor {
        l2_count: u32,
        trades_count: u32,
    }

    impl MessageProcessor for TestProcessor {
        fn process_l2_book_inline(&mut self, _l2_book: &L2Book) {
            self.l2_count += 1;
        }

        fn process_trades_inline(&mut self, _trades: &Trades) {
            self.trades_count += 1;
        }
    }

    #[test]
    fn test_message_processor_trait() {
        let mut processor = TestProcessor {
            l2_count: 0,
            trades_count: 0,
        };

        // Test with Pong message (should be ignored)
        processor.dispatch_message(&Message::Pong);
        assert_eq!(processor.l2_count, 0);
        assert_eq!(processor.trades_count, 0);
    }
}
