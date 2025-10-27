pub mod state;
pub mod dashboard;

use crossterm::{
    event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::CrosstermBackend,
    Terminal,
};
use std::io;
use tokio::sync::watch;

use self::state::DashboardState;
use self::dashboard::render_dashboard;

/// Main TUI event loop
/// Runs in a separate tokio task and renders the dashboard at ~10 FPS
pub async fn run_tui(mut state_rx: watch::Receiver<DashboardState>) -> io::Result<()> {
    // Setup terminal
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let result = run_app(&mut terminal, &mut state_rx).await;

    // Restore terminal
    disable_raw_mode()?;
    execute!(
        terminal.backend_mut(),
        LeaveAlternateScreen,
        DisableMouseCapture
    )?;
    terminal.show_cursor()?;

    result
}

/// Application rendering loop
async fn run_app<B: ratatui::backend::Backend>(
    terminal: &mut Terminal<B>,
    state_rx: &mut watch::Receiver<DashboardState>,
) -> io::Result<()> {
    let mut last_state = state_rx.borrow().clone();

    loop {
        // Render current state
        terminal.draw(|f| render_dashboard(f, &last_state))?;

        // Check for keyboard input (non-blocking with 100ms timeout)
        if event::poll(std::time::Duration::from_millis(100))? {
            if let Event::Key(key) = event::read()? {
                match key.code {
                    KeyCode::Char('q') | KeyCode::Esc => {
                        return Ok(());
                    }
                    KeyCode::Char('c') if key.modifiers.contains(event::KeyModifiers::CONTROL) => {
                        return Ok(());
                    }
                    _ => {}
                }
            }
        }

        // Update state if changed (non-blocking)
        if state_rx.has_changed().unwrap_or(false) {
            last_state = state_rx.borrow_and_update().clone();
        }
    }
}
