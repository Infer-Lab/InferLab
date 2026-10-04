use super::text::{display_width, ellipsize_end, ellipsize_middle, ellipsize_start};
use super::theme::Palette;
use crate::tui::{App, InputMode, MIN_HEIGHT, MIN_WIDTH, OverviewSummary, RefreshStatus, View};
use ratatui::layout::{Alignment, Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph, Tabs, Wrap};
use std::time::Duration;

pub(super) fn render_tiny(p: Palette, frame: &mut ratatui::Frame<'_>, area: Rect) {
    let lines = vec![
        Line::from(Span::styled(
            "InferLab / Terminal",
            Style::default().fg(p.accent).add_modifier(Modifier::BOLD),
        )),
        Line::from(""),
        Line::from(vec![
            Span::styled(
                format!("{MIN_WIDTH}×{MIN_HEIGHT}"),
                Style::default()
                    .fg(p.secondary)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(" required", Style::default().fg(p.muted)),
        ]),
        Line::from(vec![
            Span::styled(
                format!("{}×{}", area.width, area.height),
                Style::default().fg(p.warning).add_modifier(Modifier::BOLD),
            ),
            Span::styled(" current", Style::default().fg(p.muted)),
        ]),
        Line::from(""),
        Line::from(Span::styled(
            "Resize the terminal to continue.",
            Style::default().fg(p.secondary),
        )),
    ];
    frame.render_widget(
        Paragraph::new(lines)
            .block(
                Block::default()
                    .borders(Borders::TOP)
                    .border_style(Style::default().fg(p.muted)),
            )
            .alignment(Alignment::Center)
            .wrap(Wrap { trim: true }),
        area,
    );
}

/// The wide console's sidebar ([[ADR-0055]]): workspace identity, the four
/// views with their counts, the current view's status filter with a count per
/// status, and the display-refresh indicator.
pub(super) fn render_sidebar(
    frame: &mut ratatui::Frame<'_>,
    app: &App,
    refresh_status: RefreshStatus,
    area: Rect,
) {
    let p = app.palette;
    let surface = p
        .panel
        .map_or_else(Style::default, |tint| Style::default().bg(tint));
    frame.render_widget(Block::default().style(surface), area);
    let width = usize::from(area.width).saturating_sub(4);
    let (name, revision, dirty) = app.snapshot.as_ref().map_or(
        ("syncing workspace".to_owned(), "—".to_owned(), false),
        |snapshot| {
            let name = snapshot
                .root
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("workspace")
                .to_owned();
            let workspace = snapshot.workspace.value.as_ref();
            (
                name,
                workspace
                    .map_or("—", |value| short_revision(&value.revision))
                    .to_owned(),
                workspace.is_some_and(|value| value.dirty),
            )
        },
    );
    let mut lines = vec![
        Line::from(""),
        Line::from(vec![
            Span::styled(
                "  ◆ ",
                Style::default()
                    .fg(super::theme::BRAND)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                "InferLab",
                Style::default().fg(p.text).add_modifier(Modifier::BOLD),
            ),
        ]),
        Line::from(""),
        Line::from(Span::styled(
            format!("  {}", ellipsize_middle(&name, width)),
            Style::default().fg(p.text).add_modifier(Modifier::BOLD),
        )),
        Line::from(vec![
            Span::styled(
                format!("  {}", revision.get(..7).unwrap_or(&revision)),
                Style::default().fg(p.muted),
            ),
            Span::styled(" · ", Style::default().fg(p.faint)),
            Span::styled(
                if dirty { "dirty" } else { "clean" },
                Style::default().fg(if dirty { p.warning } else { p.success }),
            ),
        ]),
        Line::from(""),
        Line::from(""),
    ];
    let counts = app.view_counts();
    for (index, view) in View::ALL.iter().enumerate() {
        let active = *view == app.view;
        let count = counts[index].to_string();
        let label_width = width.saturating_sub(display_width(&count) + 3);
        let style = if active {
            p.selected()
        } else {
            Style::default()
        };
        lines.push(Line::from(vec![
            Span::styled(
                if active { " ▎" } else { "  " },
                style.fg(super::theme::BRAND),
            ),
            Span::styled(
                format!("{} ", index + 1),
                style
                    .fg(if active { p.accent } else { p.faint })
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                super::text::pad_right(view.title(), label_width),
                if active {
                    style.fg(p.on_surface).add_modifier(Modifier::BOLD)
                } else {
                    style.fg(p.muted)
                },
            ),
            Span::styled(format!("{count} "), style.fg(p.muted)),
        ]));
    }
    if let Some(current) = app.status_filter() {
        lines.push(Line::from(""));
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(
            "  STATUS",
            Style::default().fg(p.faint).add_modifier(Modifier::BOLD),
        )));
        for (filter, count) in app.status_counts() {
            let active = filter == current;
            let count_color = match filter {
                crate::console::StatusFilter::All => p.muted,
                crate::console::StatusFilter::Attention if count > 0 => p.critical,
                crate::console::StatusFilter::Attention => p.muted,
                crate::console::StatusFilter::Running => p.accent,
            };
            let count = count.to_string();
            let label_width = width.saturating_sub(display_width(&count) + 2);
            lines.push(Line::from(vec![
                Span::styled(
                    if active { "  ● " } else { "  ○ " },
                    Style::default().fg(if active { super::theme::BRAND } else { p.faint }),
                ),
                Span::styled(
                    super::text::pad_right(filter.label(), label_width),
                    if active {
                        Style::default().fg(p.text).add_modifier(Modifier::BOLD)
                    } else {
                        Style::default().fg(p.muted)
                    },
                ),
                Span::styled(count, Style::default().fg(count_color)),
            ]));
        }
    }
    frame.render_widget(Paragraph::new(lines), area);
    let overdue = matches!(refresh_status, RefreshStatus::Overdue { .. });
    let indicator = Rect::new(
        area.x,
        area.y.saturating_add(area.height.saturating_sub(2)),
        area.width,
        1,
    );
    frame.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled(
                "  ● ",
                Style::default().fg(match refresh_status {
                    RefreshStatus::Waiting => p.faint,
                    RefreshStatus::Healthy { .. } => p.success,
                    RefreshStatus::Overdue { .. } => p.warning,
                }),
            ),
            Span::styled(
                refresh_label(refresh_status, false, width),
                Style::default()
                    .fg(if overdue { p.warning } else { p.muted })
                    .add_modifier(if overdue {
                        Modifier::BOLD
                    } else {
                        Modifier::empty()
                    }),
            ),
        ])),
        indicator,
    );
}

pub(super) fn render_header(
    frame: &mut ratatui::Frame<'_>,
    app: &App,
    refresh_status: RefreshStatus,
    area: Rect,
) {
    let p = app.palette;
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Length(1),
        ])
        .split(area);
    let refresh_width = if area.width < 64 { 17 } else { 25 };
    let top = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Min(1), Constraint::Length(refresh_width)])
        .split(rows[0]);
    let root = app
        .snapshot
        .as_ref()
        .map_or("syncing workspace", |snapshot| {
            snapshot.root.to_str().unwrap_or("workspace")
        });
    let root_width = usize::from(top[0].width).saturating_sub(12);
    frame.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled(
                " InferLab ",
                Style::default().fg(p.accent).add_modifier(Modifier::BOLD),
            ),
            Span::styled("/ ", Style::default().fg(p.muted)),
            Span::styled(
                ellipsize_middle(root, root_width),
                Style::default().fg(p.secondary),
            ),
        ])),
        top[0],
    );
    let refresh = refresh_label(
        refresh_status,
        area.width < 64,
        usize::from(top[1].width).saturating_sub(1),
    );
    let overdue = matches!(refresh_status, RefreshStatus::Overdue { .. });
    frame.render_widget(
        Paragraph::new(Span::styled(
            format!("{refresh} "),
            Style::default()
                .fg(if overdue { p.warning } else { p.muted })
                .add_modifier(if overdue {
                    Modifier::BOLD
                } else {
                    Modifier::empty()
                }),
        ))
        .alignment(Alignment::Right),
        top[1],
    );
    render_summary_strip(frame, app, rows[1]);
    render_tabs(frame, app, rows[2]);
    frame.render_widget(
        Block::default()
            .borders(Borders::BOTTOM)
            .border_style(Style::default().fg(p.muted)),
        rows[3],
    );
}

fn render_summary_strip(frame: &mut ratatui::Frame<'_>, app: &App, area: Rect) {
    let p = app.palette;
    let (summary, revision, dirty) =
        app.snapshot
            .as_ref()
            .map_or((OverviewSummary::default(), "—", false), |snapshot| {
                let workspace = snapshot.workspace.value.as_ref();
                (
                    app.overview_summary(),
                    workspace.map_or("—", |value| value.revision.as_str()),
                    workspace.is_some_and(|value| value.dirty),
                )
            });
    let spans = if area.width >= 112 {
        wide_summary(p, summary, revision, dirty)
    } else if area.width >= 72 {
        medium_summary(p, summary, dirty)
    } else {
        compact_summary(p, summary, dirty)
    };
    frame.render_widget(Paragraph::new(Line::from(spans)), area);
}

fn wide_summary(
    p: Palette,
    summary: OverviewSummary,
    revision: &str,
    dirty: bool,
) -> Vec<Span<'static>> {
    vec![
        label(p, " Operations "),
        value(summary.ephemeral_active, p.success),
        label(p, " active · "),
        alert(p, summary.ephemeral_attention),
        label(p, " issues    Workflows "),
        value(summary.recorded_active, p.success),
        label(p, " running · "),
        alert(p, summary.recorded_attention),
        label(p, " issues · "),
        value(summary.recorded_recent, p.secondary),
        label(p, " recent    Revision "),
        Span::styled(
            short_revision(revision).to_owned(),
            Style::default().fg(p.secondary),
        ),
        dirty_span(p, dirty, " · dirty", " · clean"),
    ]
}

fn medium_summary(p: Palette, summary: OverviewSummary, dirty: bool) -> Vec<Span<'static>> {
    vec![
        label(p, " Ops "),
        value(summary.ephemeral_active, p.success),
        label(p, " active · "),
        alert(p, summary.ephemeral_attention),
        label(p, " issues   Workflows "),
        value(summary.recorded_active, p.success),
        label(p, " running · "),
        alert(p, summary.recorded_attention),
        label(p, " issues · "),
        value(summary.recorded_recent, p.secondary),
        label(p, " recent"),
        dirty_span(p, dirty, " · dirty", ""),
    ]
}

fn compact_summary(p: Palette, summary: OverviewSummary, dirty: bool) -> Vec<Span<'static>> {
    let issues = summary
        .ephemeral_attention
        .saturating_add(summary.recorded_attention);
    vec![
        label(p, " Ops "),
        value(summary.ephemeral_active, p.success),
        label(p, "  Issues "),
        alert(p, issues),
        label(p, "  Running "),
        value(summary.recorded_active, p.success),
        label(p, "  Recent "),
        value(summary.recorded_recent, p.secondary),
        dirty_span(p, dirty, "  DIRTY", ""),
    ]
}

fn label(p: Palette, text: &'static str) -> Span<'static> {
    Span::styled(text, Style::default().fg(p.muted))
}

fn value(value: usize, color: Color) -> Span<'static> {
    Span::styled(
        value.to_string(),
        Style::default().fg(color).add_modifier(Modifier::BOLD),
    )
}

fn alert(p: Palette, value: usize) -> Span<'static> {
    Span::styled(
        value.to_string(),
        Style::default()
            .fg(if value == 0 { p.muted } else { p.critical })
            .add_modifier(Modifier::BOLD),
    )
}

fn dirty_span(
    p: Palette,
    dirty: bool,
    dirty_text: &'static str,
    clean_text: &'static str,
) -> Span<'static> {
    Span::styled(
        if dirty { dirty_text } else { clean_text },
        Style::default().fg(if dirty { p.warning } else { p.muted }),
    )
}

fn render_tabs(frame: &mut ratatui::Frame<'_>, app: &App, area: Rect) {
    let p = app.palette;
    let titles = View::ALL
        .iter()
        .enumerate()
        .map(|(index, view)| {
            let title = if area.width < 64 && *view == View::Operations {
                "Ops"
            } else {
                view.title()
            };
            Line::from(vec![
                Span::styled(format!("{}", index + 1), Style::default().fg(p.accent_soft)),
                Span::raw(format!(" {title}")),
            ])
        })
        .collect::<Vec<_>>();
    let selected = View::ALL
        .iter()
        .position(|view| *view == app.view)
        .unwrap_or(0);
    frame.render_widget(
        Tabs::new(titles)
            .select(selected)
            .highlight_style(Style::default().fg(p.accent).add_modifier(Modifier::BOLD))
            .style(Style::default().fg(p.secondary)),
        area,
    );
}

pub(super) fn render_footer(frame: &mut ratatui::Frame<'_>, app: &App, area: Rect) {
    let p = app.palette;
    let spans = match app.input {
        InputMode::GlobalFind => input_prompt(p, "FIND", &app.query, area.width),
        InputMode::LocalSearch => input_prompt(
            p,
            if app.detail { "SEARCH LOG" } else { "FILTER" },
            &app.query,
            area.width,
        ),
        InputMode::Normal => normal_hints(app, area.width),
    };
    frame.render_widget(
        Paragraph::new(Line::from(spans)).block(
            Block::default()
                .borders(Borders::TOP)
                .border_style(Style::default().fg(p.muted)),
        ),
        area,
    );
}

fn normal_hints(app: &App, width: u16) -> Vec<Span<'static>> {
    let p = app.palette;
    if app.metric_selection.is_some() {
        let gap = if width < 80 { "  " } else { "   " };
        let required = [
            ("r", if width < 64 { "Sync" } else { "Refresh" }),
            ("q", "Quit"),
        ];
        let mut hints = if width < 80 {
            vec![("Esc", "Back"), ("↑↓", "Metric"), ("Pg", "Cases")]
        } else {
            vec![("Esc", "Record"), ("↑↓", "Metric"), ("PgUp/PgDn", "Cases")]
        };
        if width >= 64 {
            let find = if width < 80 {
                ("^K", "Find")
            } else {
                ("Ctrl+K", "Find")
            };
            if hint_fits_with_tail(p, &hints, find, &required, gap, width) {
                hints.push(find);
            }
        }
        fit_to_width(p, &mut hints, &required, gap, width);
        hints.extend(required);
        return hint_spans(p, &hints, gap);
    }
    let has_metrics = app.selected_record_has_metrics();
    let has_log = app.selected_has_log();
    let multiple_logs = app
        .selected_log_position()
        .is_some_and(|(_, count)| count > 1);
    let gap = if width < 80 { "  " } else { "   " };
    let required = [
        ("r", if width < 80 { "Sync" } else { "Refresh" }),
        ("q", "Quit"),
    ];
    let mut hints = Vec::new();
    if app.detail {
        hints.push(("Esc", "Back"));
        hints.push(("↑↓", "Scroll"));
        if width >= 64 {
            hints.push(("Pg", "Page"));
        }
        if multiple_logs && hint_fits_with_tail(p, &hints, ("[ ]", "Log"), &required, gap, width) {
            hints.push(("[ ]", "Log"));
        }
        if has_log && hint_fits_with_tail(p, &hints, ("/", "Find"), &required, gap, width) {
            hints.push(("/", "Find"));
        }
    } else {
        hints.push(("↑↓", "Select"));
        hints.push(("Enter", "Open"));
        hints.push(("/", "Filter"));
        if width >= 64 {
            hints.push((if width >= 80 { "Ctrl+K" } else { "^K" }, "Find"));
        }
        if hint_fits_with_tail(p, &hints, ("t", "Theme"), &required, gap, width) {
            hints.push(("t", "Theme"));
        }
    }
    fit_to_width(p, &mut hints, &required, gap, width);
    hints.extend(required);
    let mut spans = hint_spans(p, &hints, gap);
    if has_metrics && width >= 80 {
        append_if_fits(
            &mut spans,
            vec![
                Span::raw(gap.to_owned()),
                Span::styled(" m ", p.keycap()),
                Span::styled(" Metrics", Style::default().fg(p.muted)),
            ],
            width,
        );
    }
    if let Some((index, count)) = app
        .selected_log_position()
        .filter(|(_, count)| *count > 1 && width >= 80)
    {
        append_if_fits(
            &mut spans,
            vec![
                Span::styled("   │   ", Style::default().fg(p.muted)),
                Span::styled(
                    format!("Log {index}/{count}"),
                    Style::default().fg(p.secondary),
                ),
            ],
            width,
        );
    }
    if width >= 80 {
        if !app.status.is_empty() {
            append_dynamic(p, &mut spans, "   │   ", &app.status, width);
        } else if !app.query.is_empty() {
            append_dynamic(p, &mut spans, "   │   FILTER ", &app.query, width);
        }
    }
    spans
}

fn hint_spans(p: Palette, hints: &[(&str, &str)], gap: &str) -> Vec<Span<'static>> {
    let mut spans = Vec::new();
    for (index, (key, action)) in hints.iter().enumerate() {
        if index > 0 {
            spans.push(Span::raw(gap.to_owned()));
        }
        // Keys render as keycaps.
        spans.push(Span::styled(format!(" {key} "), p.keycap()));
        spans.push(Span::styled(
            format!(" {action}"),
            Style::default().fg(p.muted),
        ));
    }
    spans
}

/// Drop optional hints from the end until they and the required tail fit:
/// the required controls always stay visible.
fn fit_to_width(
    p: Palette,
    hints: &mut Vec<(&'static str, &'static str)>,
    required: &[(&'static str, &'static str)],
    gap: &str,
    width: u16,
) {
    while !hints.is_empty() {
        let mut projected = hints.clone();
        projected.extend_from_slice(required);
        if spans_width(&hint_spans(p, &projected, gap)) <= usize::from(width) {
            return;
        }
        hints.pop();
    }
}

fn hint_fits_with_tail(
    p: Palette,
    hints: &[(&'static str, &'static str)],
    candidate: (&'static str, &'static str),
    required: &[(&'static str, &'static str)],
    gap: &str,
    width: u16,
) -> bool {
    let mut projected = hints.to_vec();
    projected.push(candidate);
    projected.extend_from_slice(required);
    spans_width(&hint_spans(p, &projected, gap)) <= usize::from(width)
}

fn input_prompt(p: Palette, label: &str, query: &str, width: u16) -> Vec<Span<'static>> {
    let label = format!("{label}  ");
    let suffix = if width < 80 {
        "  Esc cancel"
    } else {
        "   Esc cancel · Enter apply"
    };
    let query_width = usize::from(width)
        .saturating_sub(display_width(&label) + display_width("▌") + display_width(suffix));
    let mut spans = vec![
        Span::styled(
            label,
            Style::default()
                .fg(p.accent_soft)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(
            ellipsize_start(query, query_width),
            Style::default().fg(Color::Reset),
        ),
        Span::styled("▌", Style::default().fg(p.accent)),
    ];
    spans.push(Span::styled(suffix, Style::default().fg(p.muted)));
    spans
}

fn append_if_fits(spans: &mut Vec<Span<'static>>, addition: Vec<Span<'static>>, width: u16) {
    let addition_width = spans_width(&addition);
    if spans_width(spans).saturating_add(addition_width) <= usize::from(width) {
        spans.extend(addition);
    }
}

fn append_dynamic(
    p: Palette,
    spans: &mut Vec<Span<'static>>,
    prefix: &'static str,
    value: &str,
    width: u16,
) {
    let remaining = usize::from(width)
        .saturating_sub(spans_width(spans))
        .saturating_sub(display_width(prefix));
    if remaining == 0 {
        return;
    }
    spans.push(Span::styled(prefix, Style::default().fg(p.muted)));
    spans.push(Span::styled(
        ellipsize_end(value, remaining),
        Style::default().fg(p.secondary),
    ));
}

fn spans_width(spans: &[Span<'_>]) -> usize {
    spans
        .iter()
        .map(|span| display_width(span.content.as_ref()))
        .sum()
}

fn refresh_label(state: RefreshStatus, compact: bool, max_width: usize) -> String {
    let label = match state {
        RefreshStatus::Waiting => "WAITING".to_owned(),
        RefreshStatus::Healthy { interval } => {
            let interval = short_duration(interval);
            if compact {
                format!("AUTO {interval}")
            } else {
                format!("AUTO · {interval}")
            }
        }
        RefreshStatus::Overdue { elapsed } => {
            let elapsed = short_duration(elapsed);
            if compact {
                format!("LAST · {elapsed}")
            } else {
                format!("LAST REFRESH · {elapsed} AGO")
            }
        }
    };
    ellipsize_end(&label, max_width)
}

fn short_duration(duration: Duration) -> String {
    if duration < Duration::from_secs(1) {
        return format!("{}ms", duration.as_millis());
    }
    if duration < Duration::from_secs(60) {
        return format!("{:.1}s", duration.as_secs_f64());
    }
    if duration < Duration::from_secs(60 * 60) {
        return format!("{:.1}m", duration.as_secs_f64() / 60.0);
    }
    format!("{:.1}h", duration.as_secs_f64() / (60.0 * 60.0))
}

fn short_revision(revision: &str) -> &str {
    revision.get(..12).unwrap_or(revision)
}
