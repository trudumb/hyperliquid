use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, List, ListItem, Paragraph, Row, Table},
    Frame,
};

use super::state::DashboardState;

/// Render the entire dashboard UI
pub fn render_dashboard(frame: &mut Frame, state: &DashboardState) {
    // Main layout: Split vertically into top and bottom sections
    let main_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage(20),  // Top row (position + orders)
            Constraint::Percentage(50),  // Middle row (fills + metrics)
            Constraint::Percentage(30),  // Bottom (logs)
        ])
        .split(frame.area());

    // Top row: Position panel (left) + Open Orders (right)
    let top_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(main_chunks[0]);

    // Middle row: Recent Fills (left) + Metrics (right)
    let middle_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(40), Constraint::Percentage(60)])
        .split(main_chunks[1]);

    // Render each panel
    render_position_panel(frame, top_chunks[0], state);
    render_open_orders_panel(frame, top_chunks[1], state);
    render_recent_fills_panel(frame, middle_chunks[0], state);
    render_metrics_panel(frame, middle_chunks[1], state);
    render_logs_panel(frame, main_chunks[2], state);
}

/// Panel 1: Position Summary
fn render_position_panel(frame: &mut Frame, area: Rect, state: &DashboardState) {
    let position_color = if state.cur_position > 0.0 {
        Color::Green
    } else if state.cur_position < 0.0 {
        Color::Red
    } else {
        Color::White
    };

    let unrealized_pnl_color = if state.unrealized_pnl > 0.0 {
        Color::Green
    } else if state.unrealized_pnl < 0.0 {
        Color::Red
    } else {
        Color::White
    };

    let realized_pnl_color = if state.realized_pnl > 0.0 {
        Color::Green
    } else if state.realized_pnl < 0.0 {
        Color::Red
    } else {
        Color::White
    };

    let total_pnl_color = if state.total_session_pnl > 0.0 {
        Color::Green
    } else if state.total_session_pnl < 0.0 {
        Color::Red
    } else {
        Color::White
    };

    // Calculate session PnL % relative to starting equity
    let session_pnl_pct = if state.session_start_equity > 0.0 {
        (state.total_session_pnl / state.session_start_equity) * 100.0
    } else {
        0.0
    };

    let lines = vec![
        Line::from(vec![
            Span::raw("Position: "),
            Span::styled(
                format!("{:.4} HYPE", state.cur_position),
                Style::default().fg(position_color).add_modifier(Modifier::BOLD),
            ),
            Span::raw(" @ "),
            Span::styled(
                format!("${:.3}", state.avg_entry_price),
                Style::default().fg(Color::Cyan),
            ),
        ]),
        Line::from(vec![
            Span::raw("Unrealized PnL: "),
            Span::styled(
                format!("${:.2}", state.unrealized_pnl),
                Style::default().fg(unrealized_pnl_color).add_modifier(Modifier::BOLD),
            ),
        ]),
        Line::from(vec![
            Span::raw("Realized PnL: "),
            Span::styled(
                format!("${:.2}", state.realized_pnl),
                Style::default().fg(realized_pnl_color),
            ),
            Span::raw(" | Fees: "),
            Span::styled(
                format!("${:.2}", state.total_fees),
                Style::default().fg(Color::Red),
            ),
        ]),
        Line::from(vec![
            Span::raw("Session PnL: "),
            Span::styled(
                format!("${:.2} ({:+.2}%)", state.total_session_pnl, session_pnl_pct),
                Style::default().fg(total_pnl_color).add_modifier(Modifier::BOLD),
            ),
        ]),
        Line::from(vec![
            Span::raw("Account Equity: "),
            Span::styled(
                format!("${:.2}", state.account_equity),
                Style::default().fg(Color::Yellow),
            ),
        ]),
        Line::from(vec![
            Span::raw("Sharpe: "),
            Span::styled(
                format!("{:.2}", state.sharpe_ratio),
                Style::default().fg(
                    if state.sharpe_ratio > 2.0 {
                        Color::Green
                    } else if state.sharpe_ratio > 1.0 {
                        Color::Yellow
                    } else if state.sharpe_ratio > 0.0 {
                        Color::White
                    } else {
                        Color::Red
                    }
                ),
            ),
            Span::raw(" | L2 Mid: "),
            Span::styled(
                format!("${:.3}", state.l2_mid_price),
                Style::default().fg(Color::White),
            ),
        ]),
    ];

    let paragraph = Paragraph::new(lines)
        .block(Block::default().title("Position & PnL").borders(Borders::ALL));

    frame.render_widget(paragraph, area);
}

/// Panel 2: Open Orders Table
fn render_open_orders_panel(frame: &mut Frame, area: Rect, state: &DashboardState) {
    let mut rows: Vec<Row> = Vec::new();

    // Add bid levels (green)
    for bid in &state.bid_levels {
        let row = Row::new(vec![
            bid.side.clone(),
            format!("L{}", bid.level),
            format!("{:.3}", bid.price),
            format!("{:.4}", bid.size),
            format!("{}", bid.oid),
        ])
        .style(Style::default().fg(Color::Green));
        rows.push(row);
    }

    // Add ask levels (red)
    for ask in &state.ask_levels {
        let row = Row::new(vec![
            ask.side.clone(),
            format!("L{}", ask.level),
            format!("{:.3}", ask.price),
            format!("{:.4}", ask.size),
            format!("{}", ask.oid),
        ])
        .style(Style::default().fg(Color::Red));
        rows.push(row);
    }

    let table = Table::new(
        rows,
        [
            Constraint::Length(5),  // Side
            Constraint::Length(5),  // Level
            Constraint::Length(10), // Price
            Constraint::Length(10), // Size
            Constraint::Length(12), // OID
        ],
    )
    .header(
        Row::new(vec!["Side", "Level", "Price", "Size", "OID"])
            .style(Style::default().add_modifier(Modifier::BOLD))
            .bottom_margin(1),
    )
    .block(
        Block::default()
            .title(format!("Open Orders ({})", state.bid_levels.len() + state.ask_levels.len()))
            .borders(Borders::ALL),
    );

    frame.render_widget(table, area);
}

/// Panel 3: Recent Fills
fn render_recent_fills_panel(frame: &mut Frame, area: Rect, state: &DashboardState) {
    let items: Vec<ListItem> = state
        .recent_fills
        .iter()
        .rev()  // Most recent first
        .map(|fill| {
            let color = if fill.side == "BOUGHT" {
                Color::Green
            } else {
                Color::Red
            };

            let content = format!(
                "[{}] {} {:.4} @ {:.3} (oid: {})",
                fill.timestamp, fill.side, fill.size, fill.price, fill.oid
            );

            ListItem::new(Line::from(Span::styled(content, Style::default().fg(color))))
        })
        .collect();

    let list = List::new(items).block(
        Block::default()
            .title(format!("Recent Fills ({})", state.recent_fills.len()))
            .borders(Borders::ALL),
    );

    frame.render_widget(list, area);
}

/// Panel 4: Risk & Model Metrics
fn render_metrics_panel(frame: &mut Frame, area: Rect, state: &DashboardState) {
    let lines = vec![
        Line::from(Span::styled(
            "State Vector",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD | Modifier::UNDERLINED),
        )),
        Line::from(format!("  Mid Price: ${:.3}", state.l2_mid_price)),
        Line::from(format!(
            "  Volatility (σ̂): {:.2} bps",
            state.volatility_ema_bps
        )),
        Line::from(vec![
            Span::raw("  Drift (μ̂): "),
            Span::styled(
                format!("{:.2} bps", state.adverse_selection_estimate),
                if state.adverse_selection_estimate > 0.0 {
                    Style::default().fg(Color::Green)
                } else {
                    Style::default().fg(Color::Red)
                },
            ),
        ]),
        Line::from(format!("  LOB Imbalance (I): {:.3}", state.lob_imbalance)),
        Line::from(format!("  Spread (Δ): {:.2} bps", state.market_spread_bps)),
        Line::from(format!("  Trade Flow: {:.3}", state.trade_flow_ema)),
        Line::from(""),
        Line::from(Span::styled(
            "Particle Filter",
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD | Modifier::UNDERLINED),
        )),
        Line::from(format!(
            "  ESS: {:.1} / {}",
            state.pf_ess, state.pf_max_particles
        )),
        Line::from(format!(
            "  Vol [5th, 95th]: [{:.2}, {:.2}] bps",
            state.pf_vol_5th, state.pf_vol_95th
        )),
        Line::from(format!(
            "  Current Vol: {:.2} bps",
            state.pf_volatility_bps
        )),
        Line::from(""),
        Line::from(Span::styled(
            "Adverse Selection Model",
            Style::default()
                .fg(Color::Magenta)
                .add_modifier(Modifier::BOLD | Modifier::UNDERLINED),
        )),
        Line::from(format!(
            "  MAE: {:.4} bps (Updates: {})",
            state.online_model_mae, state.online_model_updates
        )),
        Line::from(format!("  Learning Rate: {:.6}", state.online_model_lr)),
        Line::from(format!(
            "  Status: {}",
            if state.online_model_enabled {
                "ENABLED"
            } else {
                "DISABLED"
            }
        )),
        Line::from(""),
        Line::from(Span::styled(
            "Adam Optimizer",
            Style::default()
                .fg(Color::LightBlue)
                .add_modifier(Modifier::BOLD | Modifier::UNDERLINED),
        )),
        Line::from(format!(
            "  Gradient Samples: {}",
            state.adam_gradient_samples
        )),
        Line::from(format!("  Avg Loss: {:.6}", state.adam_avg_loss)),
        Line::from(format!(
            "  Last Update: {:.1}s ago",
            state.adam_last_update_secs
        )),
        Line::from(""),
        Line::from(Span::styled(
            format!("System: Uptime {:.0}s | Messages: {}", state.uptime_secs, state.total_messages),
            Style::default().fg(Color::DarkGray),
        )),
    ];

    let paragraph = Paragraph::new(lines)
        .block(Block::default().title("Risk & Model Metrics").borders(Borders::ALL));

    frame.render_widget(paragraph, area);
}

/// Panel 5: Live Log Panel (placeholder for now)
fn render_logs_panel(frame: &mut Frame, area: Rect, _state: &DashboardState) {
    let lines = vec![
        Line::from(Span::styled(
            "Live logs disabled - check market_maker.log for details",
            Style::default().fg(Color::DarkGray),
        )),
    ];

    let paragraph = Paragraph::new(lines)
        .block(Block::default().title("System Logs").borders(Borders::ALL));

    frame.render_widget(paragraph, area);
}
