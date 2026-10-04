mod chrome;
mod detail;
mod metric_page;
mod text;
mod theme;

pub(super) use theme::Palette;

use super::{
    App, DisplayEntry, InputMode, MIN_HEIGHT, MIN_WIDTH, RefreshStatus, State, WIDE_WIDTH, search,
};
use ratatui::layout::{Alignment, Constraint, Direction, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{List, ListItem, ListState, Paragraph, Wrap};
use std::path::Path;
use text::{display_width, ellipsize_end};

const LOADING_MARK: [&str; 6] = [
    "   ████  █████▄",
    "  ████▀ ██████▀",
    "  ███▀  ▀▀████",
    " ██████ ▄████▀",
    " █████ ▄█████▄▄▄▄▄",
    "█████ ▄██████████▀",
];
const LOADING_MARK_WIDTH: u16 = 18;
const SIDEBAR_WIDTH: u16 = 24;
/// The sidebar appears only when the two panes still get the wide width.
const SIDEBAR_LAYOUT_WIDTH: u16 = WIDE_WIDTH + SIDEBAR_WIDTH;
const LOADING_MARK_HEIGHT: u16 = 6;
const LOADING_COPY_HEIGHT: u16 = 2;
const LOADING_GAP: u16 = 1;
const BRANDED_LOADING_HEIGHT: u16 = LOADING_MARK_HEIGHT + LOADING_GAP + LOADING_COPY_HEIGHT;

pub(super) fn render(frame: &mut ratatui::Frame<'_>, app: &mut App, refresh_status: RefreshStatus) {
    let p = app.palette;
    let area = frame.area();
    if area.width < MIN_WIDTH || area.height < MIN_HEIGHT {
        chrome::render_tiny(p, frame, area);
        return;
    }
    if area.width >= SIDEBAR_LAYOUT_WIDTH {
        // A wide console carries identity, views, the status filter, and the
        // refresh indicator in a sidebar beside the two panes ([[ADR-0055]]).
        let columns = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Length(SIDEBAR_WIDTH), Constraint::Min(1)])
            .split(area);
        chrome::render_sidebar(frame, app, refresh_status, columns[0]);
        let rows = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Min(3), Constraint::Length(2)])
            .split(columns[1]);
        render_body(frame, app, rows[0]);
        chrome::render_footer(frame, app, rows[1]);
        return;
    }
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(4),
            Constraint::Min(3),
            Constraint::Length(2),
        ])
        .split(area);
    chrome::render_header(frame, app, refresh_status, rows[0]);
    render_body(frame, app, rows[1]);
    chrome::render_footer(frame, app, rows[2]);
}

fn render_body(frame: &mut ratatui::Frame<'_>, app: &mut App, area: Rect) {
    let p = app.palette;
    if app.metric_selection.is_some() {
        metric_page::render(frame, app, area);
        return;
    }
    if app.snapshot.is_none() {
        render_initial_sync(p, frame, area);
        return;
    }
    let entry_count = app.visible_len();
    let selected = app.selected.min(entry_count.saturating_sub(1));
    let presentation_unix_ms = app.presentation_unix_ms();
    let global_find = app.input == InputMode::GlobalFind;
    let (list_area, detail_area) = if global_find {
        (Some(area), None)
    } else {
        content_areas(area, app.detail)
    };
    if let Some(list_area) = list_area {
        let block = theme::panel(p, global_find || !app.detail || detail_area.is_none()).title(
            Span::styled(
                list_title(app, selected, entry_count),
                Style::default().fg(p.accent_soft),
            ),
        );
        if entry_count == 0 {
            frame.render_widget(
                Paragraph::new(vec![
                    Line::from(""),
                    Line::from(Span::styled(
                        format!("  {}", empty_message(app)),
                        Style::default().fg(p.muted),
                    )),
                ])
                .block(block),
                list_area,
            );
        } else {
            let items = (0..entry_count)
                .filter_map(|index| {
                    let entry = app.visible_entry(index)?;
                    let group = app.visible_group(index);
                    let previous_group = index
                        .checked_sub(1)
                        .and_then(|previous| app.visible_group(previous));
                    Some(ListItem::new(entry_lines(
                        p,
                        entry,
                        RowContext {
                            heading: group.filter(|_| group != previous_group),
                            has_preceding_entry: index > 0,
                            selected: index == selected,
                            width: list_area.width.saturating_sub(2),
                            presentation_unix_ms,
                            animation_frame: app.animation_frame,
                            tree: app.visible_tree_mark(index),
                        },
                    )))
                })
                .collect::<Vec<_>>();
            let mut state = ListState::default().with_selected(Some(selected));
            frame.render_stateful_widget(
                List::new(items)
                    .block(block)
                    .highlight_style(Style::default()),
                list_area,
                &mut state,
            );
        }
    }
    if let Some(detail_area) = detail_area {
        let entry = app.visible_entry(selected);
        let mut lines = entry.map_or_else(
            || {
                vec![Line::from(Span::styled(
                    empty_message(app),
                    Style::default().fg(p.muted),
                ))]
            },
            |entry| {
                detail::lines(
                    p,
                    entry,
                    presentation_unix_ms,
                    detail_area.width.saturating_sub(2),
                    app.record_metrics_of(&entry.key),
                )
            },
        );
        let mut log_search = false;
        if let (Some(entry), Some(log)) = (entry, app.loaded_log.as_ref())
            && entry.key == log.entry_key
        {
            let query = app.active_log_query(log);
            let projected = projected_log_lines(&log.text, query);
            if query.is_some() {
                lines.clear();
                log_search = true;
            }
            detail::append_log(
                p,
                &mut lines,
                &log.path,
                log.index + 1,
                log.count,
                query.filter(|query| !query.is_empty()),
                &projected,
            );
        }
        let title = entry.map_or_else(
            || " DETAIL ".to_owned(),
            |entry| {
                let prefix = if log_search { "LOG SEARCH" } else { "DETAIL" };
                let log_context = app
                    .loaded_log
                    .as_ref()
                    .filter(|log| log.entry_key == entry.key)
                    .map_or_else(String::new, |log| {
                        let name = Path::new(&log.path)
                            .file_name()
                            .and_then(|name| name.to_str())
                            .unwrap_or(&log.path);
                        format!(
                            " · log {}/{} {}",
                            log.index + 1,
                            log.count,
                            ellipsize_end(name, 12)
                        )
                    });
                let suffix = format!("{log_context} · {}/{}", selected + 1, entry_count);
                let title_width = usize::from(detail_area.width)
                    .saturating_sub(display_width(prefix) + display_width(&suffix) + 5);
                format!(
                    " {prefix} · {}{suffix} ",
                    ellipsize_end(&entry.title, title_width)
                )
            },
        );
        frame.render_widget(
            Paragraph::new(lines)
                .block(
                    theme::panel(p, app.detail || list_area.is_none())
                        .title(Span::styled(title, Style::default().fg(p.accent_soft))),
                )
                .scroll((app.detail_scroll, 0))
                .wrap(Wrap { trim: false }),
            detail_area,
        );
    }
}

fn render_initial_sync(p: Palette, frame: &mut ratatui::Frame<'_>, area: Rect) {
    if area.height < BRANDED_LOADING_HEIGHT {
        let copy_area = Rect::new(
            area.x.saturating_add(2),
            area.y.saturating_add(1),
            area.width.saturating_sub(2),
            LOADING_COPY_HEIGHT,
        );
        frame.render_widget(Paragraph::new(initial_sync_copy(p)), copy_area);
        return;
    }

    let top = area
        .y
        .saturating_add(area.height.saturating_sub(BRANDED_LOADING_HEIGHT) / 2);
    let mark_area = Rect::new(
        area.x
            .saturating_add(area.width.saturating_sub(LOADING_MARK_WIDTH) / 2),
        top,
        LOADING_MARK_WIDTH.min(area.width),
        LOADING_MARK_HEIGHT,
    );
    let mark = LOADING_MARK
        .iter()
        .map(|line| {
            Line::from(Span::styled(
                *line,
                Style::default().fg(p.accent).add_modifier(Modifier::BOLD),
            ))
        })
        .collect::<Vec<_>>();
    frame.render_widget(Paragraph::new(mark), mark_area);

    let copy_area = Rect::new(
        area.x,
        top.saturating_add(LOADING_MARK_HEIGHT + LOADING_GAP),
        area.width,
        LOADING_COPY_HEIGHT,
    );
    frame.render_widget(
        Paragraph::new(initial_sync_copy(p)).alignment(Alignment::Center),
        copy_area,
    );
}

fn initial_sync_copy(p: Palette) -> Vec<Line<'static>> {
    vec![
        Line::from(Span::styled(
            "SYNCING WORKSPACE",
            Style::default().fg(p.accent).add_modifier(Modifier::BOLD),
        )),
        Line::from(Span::styled(
            "Waiting for the first complete workspace read…",
            Style::default().fg(p.muted),
        )),
    ]
}

fn content_areas(area: Rect, detail_open: bool) -> (Option<Rect>, Option<Rect>) {
    if area.width >= WIDE_WIDTH {
        let columns = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Percentage(50),
                Constraint::Length(1),
                Constraint::Percentage(50),
            ])
            .split(area);
        (Some(columns[0]), Some(columns[2]))
    } else if detail_open {
        (None, Some(area))
    } else {
        (Some(area), None)
    }
}

/// Columns of a one-line list row ([[ADR-0055]]): status glyph, kind, name,
/// outcome or lifecycle, authority badge, and age.
const KIND_WIDTH: usize = 7;
const OUTCOME_WIDTH: usize = 10;
const BADGE_WIDTH: usize = 4;
const AGE_WIDTH: usize = 3;

struct RowContext<'a> {
    heading: Option<&'a str>,
    has_preceding_entry: bool,
    selected: bool,
    width: u16,
    presentation_unix_ms: u64,
    animation_frame: usize,
    tree: Option<crate::tui::app::TreeMark>,
}

fn entry_lines(p: Palette, entry: &DisplayEntry, row: RowContext<'_>) -> Vec<Line<'static>> {
    let width = usize::from(row.width);
    let mut lines = Vec::new();
    if let Some(group) = row.heading {
        if row.has_preceding_entry {
            lines.push(Line::from(""));
        }
        lines.push(Line::from(Span::styled(
            ellipsize_end(&format!("  {group}"), width),
            Style::default()
                .fg(p.section(group))
                .add_modifier(Modifier::BOLD),
        )));
    }
    let base = if row.selected {
        p.selected()
    } else {
        Style::default()
    };
    let (kind, name) = entry.kind_and_name();
    // The outcome column keeps the recorded lifecycle; exceptional read
    // health qualifies the row beside the name instead of replacing it.
    let outcome = entry.lifecycle.clone().unwrap_or_default();
    let outcome_color = p.tone(entry.tone);
    let age = entry.moment_unix_ms.map_or_else(String::new, |moment| {
        crate::tui::views::compact_age(row.presentation_unix_ms.saturating_sub(moment))
    });
    // A child row hangs from its parent's tree; a parent row shows whether
    // its children are expanded.
    let (branch, fold) = match row.tree {
        Some(crate::tui::app::TreeMark::Child { last }) => (if last { "╰ " } else { "├ " }, ""),
        Some(crate::tui::app::TreeMark::Parent { expanded }) => {
            ("", if expanded { " ▾" } else { " ▸" })
        }
        None => ("", ""),
    };
    let right_width = 1 + OUTCOME_WIDTH + 1 + BADGE_WIDTH + 1 + AGE_WIDTH;
    let name_width = width.saturating_sub(
        3 + display_width(branch) + KIND_WIDTH + 1 + right_width + display_width(fold),
    );
    let (summary, summary_color) = if entry.state == State::Live {
        (
            if entry.summary.is_empty() {
                String::new()
            } else {
                format!("  {}", entry.summary)
            },
            p.muted,
        )
    } else {
        (
            format!("  refresh {}", entry.state.label()),
            p.state(entry.state),
        )
    };
    let shown_name = ellipsize_end(name, name_width);
    let shown_summary = ellipsize_end(
        &summary,
        name_width.saturating_sub(display_width(&shown_name)),
    );
    let padding =
        name_width.saturating_sub(display_width(&shown_name) + display_width(&shown_summary));
    lines.push(Line::from(vec![
        Span::styled(if row.selected { "▎" } else { " " }, base.fg(theme::BRAND)),
        Span::styled(branch, base.fg(p.faint)),
        Span::styled(
            format!("{} ", row_glyph(entry, row.animation_frame)),
            base.fg(p.tone(entry.tone)).add_modifier(Modifier::BOLD),
        ),
        Span::styled(
            format!("{} ", text::pad_right(kind, KIND_WIDTH)),
            base.fg(p.muted),
        ),
        Span::styled(
            shown_name,
            base.fg(if row.selected { p.accent } else { p.text })
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(fold, base.fg(p.accent)),
        Span::styled(shown_summary, base.fg(summary_color)),
        Span::styled(" ".repeat(padding), base),
        Span::styled(
            format!(" {}", text::pad_left(&outcome, OUTCOME_WIDTH)),
            base.fg(outcome_color),
        ),
        Span::styled(
            format!(" {}", text::pad_right(entry.authority.badge(), BADGE_WIDTH)),
            base.fg(p.faint),
        ),
        Span::styled(
            format!(" {}", text::pad_left(&age, AGE_WIDTH)),
            base.fg(p.muted),
        ),
    ]));
    if let Some((index, total)) = entry.progress.filter(|(_, total)| *total > 0) {
        let bar_width = width.saturating_sub(16).min(24);
        let (filled, rest) = theme::smooth_bar(index as f64 / total as f64, bar_width);
        lines.push(Line::from(vec![
            Span::styled("     ", base),
            Span::styled(filled, base.fg(theme::BRAND)),
            Span::styled(rest, base.bg(p.chip)),
            Span::styled(format!(" {index}/{total}"), base.fg(p.muted)),
        ]));
    }
    // An exceptional read keeps its reason in view even where no detail pane
    // shows beside the list.
    if entry.state != State::Live && !entry.summary.is_empty() {
        lines.push(Line::from(Span::styled(
            ellipsize_end(&format!("    {}", entry.summary), width),
            base.fg(p.state(entry.state)),
        )));
    }
    lines
}

/// A live operation animates its glyph; every other row shows its status.
fn row_glyph(entry: &DisplayEntry, frame: usize) -> &'static str {
    if entry.kind == crate::tui::EntryKind::Operation && entry.state == State::Live {
        theme::SPINNER[frame % theme::SPINNER.len()]
    } else {
        entry.tone.glyph()
    }
}

fn list_title(app: &App, selected: usize, count: usize) -> String {
    let label = if app.input == InputMode::GlobalFind {
        "GLOBAL FIND".to_owned()
    } else {
        app.view.title().to_uppercase()
    };
    if count == 0 {
        format!(" {label} · empty ")
    } else {
        format!(" {label} · {}/{} ", selected + 1, count)
    }
}

fn empty_message(app: &App) -> String {
    if !app.query.is_empty() {
        return format!("No results for “{}”", app.query);
    }
    match app.view {
        super::View::Overview => "No overview objects are available".to_owned(),
        super::View::Operations => "No active or retained operations".to_owned(),
        super::View::Records => "No records have been written yet".to_owned(),
        super::View::Workspace => "No definitions or scratchpad entries".to_owned(),
    }
}

fn projected_log_lines(text: &str, query: Option<&str>) -> Vec<String> {
    text.lines()
        .enumerate()
        .filter(|(_, line)| {
            query
                .filter(|query| !query.is_empty())
                .is_none_or(|query| search::match_rank(query, line).is_some())
        })
        .map(|(index, line)| format!("{:>6} │ {line}", index + 1))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::render;
    use crate::tui::presentation::{EntrySource, Presentation};
    use crate::tui::{
        App, CaseView, DefinitionView, DisplayEntry, ObjectState, OperationView, OverviewSection,
        RecordView, RefreshStatus, Snapshot, State, View, WorkspaceView,
    };
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;
    use std::collections::BTreeMap;
    use std::path::PathBuf;
    use std::time::Duration;

    fn snapshot() -> Snapshot {
        Snapshot {
            root: PathBuf::from("/workspace/inferlab-vllm"),
            observed_unix_ms: 2_000,
            workspace: ObjectState {
                state: State::Live,
                value: Some(WorkspaceView {
                    revision: "0123456789abcdef".to_owned(),
                    dirty: false,
                }),
                reason: None,
                observed_unix_ms: 2_000,
                last_success_unix_ms: Some(2_000),
            },
            operations: vec![OperationView {
                key: "operation-1".to_owned(),
                state: State::Live,
                reason: None,
                command: Some("bench random-8k1k".to_owned()),
                phase: Some("measurement".to_owned()),
                item: Some("request 64/100".to_owned()),
                record_ref: Some("record-running".to_owned()),
                log_ref: None,
                started_unix_ms: Some(1_000),
                updated_unix_ms: Some(2_000),
                observed_unix_ms: 2_000,
                last_success_unix_ms: Some(2_000),
                schema_version: Some(1),
                producer: None,
                position: Some(crate::operation::OperationPosition {
                    index: 64,
                    total: 100,
                }),
                lock: None,
                readiness_failure: None,
            }],
            records: vec![RecordView {
                path: PathBuf::from(
                    "/workspace/inferlab-vllm/.inferlab/records/record-failed/record.json",
                ),
                state: State::Live,
                reason: None,
                id: Some("record-failed".to_owned()),
                kind: "bench".to_owned(),
                status: Some("failed".to_owned()),
                definition_ids: vec!["long-context".to_owned()],
                case: None,
                workflow: None,
                error: Some("deadline exceeded".to_owned()),
                started_unix_ms: Some(500),
                finished_unix_ms: Some(1_500),
                log_refs: Vec::new(),
                observed_unix_ms: 2_000,
                last_success_unix_ms: Some(2_000),
                child_refs: Vec::new(),
                topology: None,
                cases: vec![
                    CaseView {
                        id: Some("long-context".to_owned()),
                        load: crate::tui::CaseLoad::Concurrency(8),
                        status: Some("succeeded".to_owned()),
                        stdout: None,
                        stderr: None,
                        error: None,
                        metrics: BTreeMap::from([
                            ("p95_ttft_ms".to_owned(), 47.8),
                            ("request_throughput".to_owned(), 7.412500701361551),
                        ]),
                    },
                    CaseView {
                        id: Some("prefill".to_owned()),
                        load: crate::tui::CaseLoad::Concurrency(1),
                        status: Some("failed".to_owned()),
                        stdout: None,
                        stderr: None,
                        error: Some("deadline exceeded".to_owned()),
                        metrics: BTreeMap::from([("request_throughput".to_owned(), 1.25)]),
                    },
                ],
                outcome_facts: Vec::new(),
                bench_details: Vec::new(),
                artifact_refs: Vec::new(),
                process_observation: None,
            }],
            child_records: Vec::new(),
            definitions: Vec::new(),
            journal: Vec::new(),
            operations_error: None,
            records_error: None,
            definitions_error: None,
            journal_error: None,
        }
    }

    fn rendered(width: u16, height: u16, app: &mut App) -> String {
        rendered_with_refresh(
            width,
            height,
            app,
            RefreshStatus::Healthy {
                interval: Duration::from_secs(1),
            },
        )
    }

    fn rendered_with_refresh(
        width: u16,
        height: u16,
        app: &mut App,
        refresh_status: RefreshStatus,
    ) -> String {
        let backend = TestBackend::new(width, height);
        let mut terminal = match Terminal::new(backend) {
            Ok(terminal) => terminal,
            Err(error) => return format!("terminal error: {error}"),
        };
        if let Err(error) = terminal.draw(|frame| render(frame, app, refresh_status)) {
            return format!("draw error: {error}");
        }
        let buffer = terminal.backend().buffer();
        (0..height)
            .map(|y| {
                (0..width)
                    .filter_map(|x| buffer.cell((x, y)).map(|cell| cell.symbol()))
                    .collect::<String>()
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn overview_entry_matches(
        snapshot: &Snapshot,
        key: &str,
        section: OverviewSection,
        predicate: impl Fn(&DisplayEntry) -> bool,
    ) -> bool {
        let presentation = Presentation::from_snapshot(snapshot);
        let source = EntrySource::View(View::Overview);
        (0..presentation.len(source)).any(|position| {
            let item = presentation.item(source, position);
            let entry = presentation.entry(source, position);
            item.is_some_and(|item| item.section == Some(section))
                && entry.is_some_and(|entry| entry.key == key && predicate(entry))
        })
    }

    #[test]
    fn overview_has_scan_friendly_console_regions() {
        let mut app = App::default();
        app.accept(snapshot());

        let screen = rendered(120, 32, &mut app);

        assert!(screen.contains("InferLab"));
        assert!(screen.contains("Operations"));
        assert!(screen.contains("Workflows"));
        assert!(screen.contains("ATTENTION"));
        assert!(screen.contains("NOW"));
        assert!(screen.contains("recent"));
        assert!(screen.contains("WORKSPACE"));
    }

    /// Text drawn on a fill always carries a color chosen for that fill: the
    /// terminal's own foreground may be dark on a dark fill, as when the
    /// dark palette meets a light terminal that did not report its colors.
    #[test]
    fn text_on_a_fill_never_takes_the_terminal_foreground() -> Result<(), Box<dyn std::error::Error>>
    {
        use crate::tui::appearance::Appearance;
        use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
        use ratatui::style::Color;
        for light in [false, true] {
            for view in 0..4 {
                for keys in [
                    &[][..],
                    &[KeyCode::Enter][..],
                    &[KeyCode::Enter, KeyCode::Char('m')][..],
                ] {
                    let mut app = App::default();
                    app.palette = super::Palette::for_appearance(Appearance {
                        light,
                        background: None,
                    });
                    app.accept(snapshot());
                    app.select_view(view);
                    for key in keys {
                        let _ = app.handle_key(KeyEvent::new(*key, KeyModifiers::NONE));
                    }
                    let mut terminal = Terminal::new(TestBackend::new(160, 40))?;
                    terminal.draw(|frame| {
                        render(
                            frame,
                            &mut app,
                            RefreshStatus::Healthy {
                                interval: Duration::from_secs(1),
                            },
                        );
                    })?;
                    let buffer = terminal.backend().buffer();
                    for y in 0..40 {
                        for x in 0..160 {
                            let Some(cell) = buffer.cell((x, y)) else {
                                continue;
                            };
                            assert!(
                                cell.bg == Color::Reset
                                    || cell.symbol().trim().is_empty()
                                    || cell.fg != Color::Reset,
                                "light={light} view={view} keys={keys:?}: {:?} at ({x}, {y}) is on {:?} in the terminal foreground",
                                cell.symbol(),
                                cell.bg
                            );
                        }
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn eighty_column_layout_preserves_navigation_and_readable_rows() {
        let mut app = App::default();
        app.accept(snapshot());

        let screen = rendered(80, 24, &mut app);

        assert!(screen.contains("1 Overview"));
        assert!(screen.contains("bench random-8k1k"));
        assert!(screen.contains(" Ctrl+K  Find"));
        assert!(!screen.contains("DETAILS"));
    }

    #[test]
    fn minimum_supported_width_uses_compact_complete_chrome() {
        let mut app = App::default();
        app.accept(snapshot());

        let screen = rendered(50, 20, &mut app);

        assert!(screen.contains("Ops"));
        assert!(screen.contains("Issues"));
        assert!(screen.contains("Running"));
        assert!(screen.contains("Recent"));
        assert!(screen.contains("AUTO"));
        assert!(screen.contains("4 Workspace"));
        assert!(screen.contains(" r  Sync"));
        assert!(screen.contains(" q  Quit"));
    }

    #[test]
    fn minimum_width_multi_log_detail_keeps_priority_footer_controls()
    -> Result<(), Box<dyn std::error::Error>> {
        let directory = tempfile::tempdir()?;
        std::fs::write(directory.path().join("first.log"), "first\n")?;
        std::fs::write(directory.path().join("second.log"), "second\n")?;
        let mut current = snapshot();
        current.root = directory.path().to_path_buf();
        current.operations.clear();
        current.records[0].log_refs = vec!["first.log".to_owned(), "second.log".to_owned()];
        let mut app = App::default();
        app.accept(current);
        app.select_view(2);
        let _ = app.handle_key(crossterm::event::KeyEvent::new(
            crossterm::event::KeyCode::Enter,
            crossterm::event::KeyModifiers::NONE,
        ));

        let screen = rendered(50, 20, &mut app);
        let footer = screen.lines().last().unwrap_or_default();

        assert!(footer.contains(" Esc  Back"));
        assert!(footer.contains(" ↑↓  Scroll"));
        assert!(footer.contains(" r  Sync"));
        assert!(footer.contains(" q  Quit"));
        Ok(())
    }

    #[test]
    fn workspace_observation_state_does_not_replace_the_refresh_indicator() {
        let mut current = snapshot();
        current.workspace.state = State::Unavailable;
        let mut app = App::default();
        app.accept(current);

        let screen = rendered_with_refresh(
            50,
            20,
            &mut app,
            RefreshStatus::Healthy {
                interval: Duration::from_secs(1),
            },
        );
        let header = screen.lines().next().unwrap_or_default();

        assert!(header.contains("AUTO"));
        assert!(header.contains("1.0s"));
        assert!(!header.contains("UNAVAILABLE"));
    }

    #[test]
    fn healthy_refresh_label_is_stable_across_the_display_cycle() {
        let mut app = App::default();
        app.accept(snapshot());

        let just_completed = rendered_with_refresh(
            80,
            24,
            &mut app,
            RefreshStatus::Healthy {
                interval: Duration::from_secs(1),
            },
        );
        let near_next_tick = rendered_with_refresh(
            80,
            24,
            &mut app,
            RefreshStatus::Healthy {
                interval: Duration::from_secs(1),
            },
        );

        let first_header = just_completed.lines().next().unwrap_or_default();
        let second_header = near_next_tick.lines().next().unwrap_or_default();
        assert_eq!(first_header, second_header);
        assert!(first_header.contains("AUTO"));
        assert!(first_header.contains("1.0s"));
        assert!(!first_header.contains("ago"));
    }

    #[test]
    fn refresh_indicator_crosses_the_override_boundary_and_recovers() {
        let mut app = App::default();
        app.accept(snapshot());
        let interval = Duration::from_secs(2);

        let healthy = rendered_with_refresh(80, 24, &mut app, RefreshStatus::Healthy { interval });
        let overdue = rendered_with_refresh(
            80,
            24,
            &mut app,
            RefreshStatus::Overdue {
                elapsed: Duration::from_secs(4),
            },
        );
        let recovered =
            rendered_with_refresh(80, 24, &mut app, RefreshStatus::Healthy { interval });

        assert!(healthy.lines().next().unwrap_or_default().contains("AUTO"));
        assert!(healthy.lines().next().unwrap_or_default().contains("2.0s"));
        assert!(
            overdue
                .lines()
                .next()
                .unwrap_or_default()
                .contains("LAST REFRESH")
        );
        assert!(overdue.lines().next().unwrap_or_default().contains("4.0s"));
        assert_eq!(
            healthy.lines().next().unwrap_or_default(),
            recovered.lines().next().unwrap_or_default()
        );
    }

    #[test]
    fn refresh_indicator_waits_for_the_first_complete_generation() {
        let mut app = App::default();

        let screen = rendered_with_refresh(80, 24, &mut app, RefreshStatus::Waiting);

        assert!(
            screen
                .lines()
                .next()
                .unwrap_or_default()
                .contains("WAITING")
        );
    }

    #[test]
    fn spacious_initial_sync_shows_the_block_mark_and_complete_copy() {
        let mut app = App::default();

        let screen = rendered(80, 24, &mut app);

        assert!(screen.contains("████  █████▄"));
        assert!(screen.contains("SYNCING WORKSPACE"));
        assert!(screen.contains("Waiting for the first complete workspace read…"));
    }

    #[test]
    fn minimum_height_initial_sync_keeps_compact_complete_copy() {
        let mut app = App::default();

        let screen = rendered(50, 12, &mut app);

        assert!(!screen.contains("████  █████▄"));
        assert!(screen.contains("SYNCING WORKSPACE"));
        assert!(screen.contains("Waiting for the first complete workspace read…"));
    }

    #[test]
    fn first_complete_snapshot_replaces_the_loading_mark_immediately() {
        let mut app = App::default();
        let loading = rendered(80, 24, &mut app);

        app.accept(snapshot());
        let workspace = rendered(80, 24, &mut app);

        assert!(loading.contains("████  █████▄"));
        assert!(!workspace.contains("████  █████▄"));
        assert!(workspace.contains("bench random-8k1k"));
    }

    #[test]
    fn declaration_omits_normal_refresh_health_and_groups_by_kind() {
        let mut current = snapshot();
        current.definitions.push(DefinitionView {
            kind: "bench".to_owned(),
            id: "random-8k1k".to_owned(),
            relationship: "standalone".to_owned(),
            fact_sections: Vec::new(),
            state: State::Live,
            observed_unix_ms: 2_000,
            last_success_unix_ms: 2_000,
            reason: None,
        });
        let mut app = App::default();
        app.accept(current);
        app.select_view(3);

        let screen = rendered(80, 24, &mut app);

        assert!(screen.contains("BENCH"));
        let declaration = row(&screen, "random-8k1k");
        assert!(
            declaration.contains("standalone") && declaration.contains("DECL"),
            "{screen}"
        );
        assert!(!screen.contains("refresh live"));
    }

    #[test]
    fn unscheduled_declaration_keeps_and_displays_its_observation_age() {
        let mut current = snapshot();
        current.observed_unix_ms = 62_000;
        current.definitions.push(DefinitionView {
            kind: "bench".to_owned(),
            id: "qualification".to_owned(),
            relationship: "standalone".to_owned(),
            fact_sections: Vec::new(),
            state: State::Live,
            observed_unix_ms: 2_000,
            last_success_unix_ms: 2_000,
            reason: None,
        });
        let mut app = App::default();
        app.accept(current);
        app.select_view(3);

        let screen = rendered(120, 40, &mut app);

        assert!(
            screen.contains("refreshed 1970-01-01 00:00:02 UTC · 1.0 min ago"),
            "{screen}"
        );
    }

    #[test]
    fn presentation_clock_advances_source_age_without_a_new_snapshot() {
        let mut current = snapshot();
        current.observed_unix_ms = 62_000;
        current.definitions.push(DefinitionView {
            kind: "bench".to_owned(),
            id: "qualification".to_owned(),
            relationship: "standalone".to_owned(),
            fact_sections: Vec::new(),
            state: State::Live,
            observed_unix_ms: 2_000,
            last_success_unix_ms: 2_000,
            reason: None,
        });
        let mut app = App::default();
        app.accept(current);
        app.select_view(3);
        app.advance_presentation_clock(122_000);

        let screen = rendered(120, 40, &mut app);

        assert!(
            screen.contains("refreshed 1970-01-01 00:00:02 UTC · 2.0 min ago"),
            "{screen}"
        );
    }

    #[test]
    fn record_lifecycle_precedes_explicit_refresh_health() {
        let mut app = App::default();
        app.accept(snapshot());
        app.select_view(2);

        let screen = rendered(80, 24, &mut app);

        let record = row(&screen, "record-failed");
        assert!(
            record.contains("failed") && record.contains("REC"),
            "{screen}"
        );
        assert!(
            !record.contains("refresh"),
            "a current read adds no qualifier"
        );
    }

    #[test]
    fn missing_record_status_does_not_become_refresh_lifecycle() {
        let mut current = snapshot();
        current.records[0].status = None;
        current.records[0].state = State::Stale;
        current.records[0].reason = Some("record refresh failed".to_owned());
        let mut app = App::default();
        app.accept(current);
        app.select_view(2);

        let screen = rendered(120, 40, &mut app);

        assert!(screen.contains("Status  unknown"));
        assert!(screen.contains("refresh stale"));
        assert!(!screen.contains("Status        stale"));
    }

    #[test]
    fn process_liveness_keeps_observed_authority_separate_from_record_lifecycle() {
        let mut current = snapshot();
        current.records[0].kind = "server".to_owned();
        current.records[0].status = Some("running".to_owned());
        current.records[0].process_observation = Some(ObjectState {
            state: State::Live,
            value: Some(true),
            reason: None,
            observed_unix_ms: 2_000,
            last_success_unix_ms: Some(2_000),
        });
        let mut app = App::default();
        app.accept(current);
        app.select_view(2);

        let screen = rendered(160, 80, &mut app);

        let server = row(&screen, "record-failed");
        assert!(
            server.contains("running")
                && server.contains("REC")
                && server.contains("OBS process alive"),
            "{screen}"
        );
        assert!(screen.contains("PROCESS LIVENESS"));
        assert!(screen.contains("Authority     observed"));
        assert!(screen.contains("Read health   current"));
    }

    #[test]
    fn presentation_clock_advances_a_skipped_process_observation_age() {
        let mut current = snapshot();
        current.observed_unix_ms = 62_000;
        current.records[0].kind = "server".to_owned();
        current.records[0].status = Some("running".to_owned());
        current.records[0].process_observation = Some(ObjectState {
            state: State::Live,
            value: Some(true),
            reason: None,
            observed_unix_ms: 2_000,
            last_success_unix_ms: Some(2_000),
        });
        let mut app = App::default();
        app.accept(current);
        app.select_view(2);
        app.advance_presentation_clock(122_000);

        let screen = rendered(120, 80, &mut app);

        assert!(screen.contains("PROCESS LIVENESS"));
        assert!(screen.contains("Observed      2.0 min ago"));
        assert!(screen.contains("Last success  2.0 min ago"));
    }

    #[test]
    fn stale_running_record_is_classified_only_as_recorded_attention() {
        let mut snapshot = snapshot();
        snapshot.records[0].kind = "server".to_owned();
        snapshot.records[0].status = Some("running".to_owned());
        snapshot.records[0].state = State::Stale;

        let summary = snapshot.overview_summary();
        let entries = snapshot.entries(View::Overview);

        assert_eq!(summary.recorded_active, 0);
        assert_eq!(summary.recorded_attention, 1);
        assert_eq!(summary.recorded_recent, 0);
        assert_eq!(
            entries
                .iter()
                .filter(|entry| entry.key == "record-failed")
                .count(),
            1
        );
        assert!(overview_entry_matches(
            &snapshot,
            "record-failed",
            OverviewSection::Attention,
            |_| true
        ));
    }

    #[test]
    fn live_operations_do_not_hide_a_running_server_from_active() {
        let mut snapshot = snapshot();
        let operation = snapshot.operations[0].clone();
        snapshot.operations = (0..5)
            .map(|index| {
                let mut operation = operation.clone();
                operation.key = format!("operation-{index}");
                operation
            })
            .collect();
        snapshot.records[0].kind = "server".to_owned();
        snapshot.records[0].status = Some("running".to_owned());
        snapshot.records[0].state = State::Live;
        snapshot.records[0].process_observation = Some(ObjectState {
            state: State::Live,
            value: Some(true),
            reason: None,
            observed_unix_ms: 2_000,
            last_success_unix_ms: Some(2_000),
        });

        let summary = snapshot.overview_summary();

        assert_eq!(summary.ephemeral_active, 5);
        assert_eq!(summary.recorded_active, 1);
        assert!(overview_entry_matches(
            &snapshot,
            "record-failed",
            OverviewSection::Active,
            |_| true
        ));
    }

    #[test]
    fn running_top_level_workflow_is_active_without_a_process_probe() {
        let mut current = snapshot();
        current.records[0].kind = "bench".to_owned();
        current.records[0].status = Some("running".to_owned());
        current.records[0].process_observation = None;

        let summary = current.overview_summary();

        assert_eq!(summary.recorded_active, 1);
        assert_eq!(summary.recorded_attention, 0);
        assert!(overview_entry_matches(
            &current,
            "record-failed",
            OverviewSection::Active,
            |_| true
        ));
    }

    #[test]
    fn dead_recorded_running_server_is_attention_not_active() {
        let mut current = snapshot();
        current.records[0].kind = "server".to_owned();
        current.records[0].status = Some("running".to_owned());
        current.records[0].process_observation = Some(ObjectState {
            state: State::Live,
            value: Some(false),
            reason: None,
            observed_unix_ms: 2_000,
            last_success_unix_ms: Some(2_000),
        });

        let summary = current.overview_summary();

        assert_eq!(summary.recorded_active, 0);
        assert_eq!(summary.recorded_attention, 1);
        assert!(overview_entry_matches(
            &current,
            "record-failed",
            OverviewSection::Attention,
            |entry| {
                entry.lifecycle.as_deref() == Some("running")
                    && entry.summary.contains("process dead")
            }
        ));
    }

    #[test]
    fn recipe_owned_running_server_is_observed_in_overview_without_flattening_records() {
        let mut current = snapshot();
        let mut child = current.records[0].clone();
        child.id = Some("recipe-server".to_owned());
        child.kind = "server".to_owned();
        child.status = Some("running".to_owned());
        child.process_observation = Some(ObjectState {
            state: State::Live,
            value: Some(true),
            reason: None,
            observed_unix_ms: 2_000,
            last_success_unix_ms: Some(2_000),
        });
        current.child_records.push(child);

        let records = current.entries(View::Records);

        assert!(overview_entry_matches(
            &current,
            "recipe-server",
            OverviewSection::Active,
            |_| true
        ));
        assert!(!records.iter().any(|entry| entry.key == "recipe-server"));
        let summary = current.overview_summary();
        assert_eq!(summary.recorded_active, 0);
        assert_eq!(summary.recorded_attention, 1);
    }

    #[test]
    fn top_level_workflow_attention_precedes_child_server_attention() {
        let mut current = snapshot();
        let mut child = current.records[0].clone();
        child.id = Some("recipe-server".to_owned());
        child.kind = "server".to_owned();
        child.status = Some("running".to_owned());
        child.process_observation = Some(ObjectState {
            state: State::Live,
            value: Some(false),
            reason: None,
            observed_unix_ms: 2_000,
            last_success_unix_ms: Some(2_000),
        });
        current.child_records.push(child);

        let overview = current.entries(View::Overview);
        let workflow = overview
            .iter()
            .position(|entry| entry.key == "record-failed");
        let server = overview
            .iter()
            .position(|entry| entry.key == "recipe-server");

        assert!(matches!((workflow, server), (Some(workflow), Some(server)) if workflow < server));
    }

    #[test]
    fn overview_retains_child_server_visibility_after_prioritizing_top_level_failures() {
        let mut current = snapshot();
        let failed = current.records[0].clone();
        current.records = (0..5)
            .map(|index| {
                let mut record = failed.clone();
                record.id = Some(format!("failed-workflow-{index}"));
                record
            })
            .collect();
        let mut child = failed;
        child.id = Some("recipe-server".to_owned());
        child.kind = "server".to_owned();
        child.status = Some("running".to_owned());
        child.process_observation = Some(ObjectState {
            state: State::Live,
            value: Some(false),
            reason: None,
            observed_unix_ms: 2_000,
            last_success_unix_ms: Some(2_000),
        });
        current.child_records.push(child);

        let overview = current.entries(View::Overview);

        assert!(
            overview
                .iter()
                .any(|entry| entry.key == "failed-workflow-0")
        );
        assert!(overview.iter().any(|entry| entry.key == "recipe-server"));
        assert_eq!(current.overview_summary().recorded_attention, 5);
    }

    #[test]
    fn record_detail_tabulates_case_metrics_without_flattening_case_values() {
        let mut app = App::default();
        app.accept(snapshot());
        app.select_view(2);

        let screen = rendered(160, 50, &mut app);

        assert!(screen.contains("OUTCOME"));
        assert!(screen.contains("TIMING"));
        assert!(screen.contains("METRICS"));
        let header = row_in_detail(&screen, "c8");
        assert!(header.contains("c1"), "one column per case:\n{screen}");
        let throughput = row_in_detail(&screen, "Request throughput");
        assert!(
            throughput.contains("7.41") && throughput.contains("1.25"),
            "a metric row holds every case's value:\n{screen}"
        );
        assert!(
            throughput
                .chars()
                .any(|character| "▁▂▃▄▅▆▇█".contains(character)),
            "a trend sparkline follows the values:\n{screen}"
        );
        assert!(!screen.contains("long-context.p95_ttft_ms"));
        assert!(
            screen.contains(" m "),
            "the comparison surface stays reachable"
        );
    }

    #[test]
    fn detail_opens_with_status_and_authority_pills_and_condensed_health() {
        let mut app = App::default();
        app.accept(snapshot());
        app.select_view(2);

        let screen = rendered(160, 60, &mut app);
        assert!(
            row_in_detail(&screen, "FAILED").contains("REC recorded"),
            "status and authority open the detail as pills:\n{screen}"
        );
        assert!(
            row_in_detail(&screen, "current").contains("recorded"),
            "a current read condenses to one line:\n{screen}"
        );
        assert!(!screen.contains("Last success"), "{screen}");

        let mut stale = snapshot();
        stale.records[0].state = State::Stale;
        stale.records[0].reason = Some("record refresh failed".to_owned());
        let mut app = App::default();
        app.accept(stale);
        app.select_view(2);
        let screen = rendered(160, 60, &mut app);
        assert!(
            screen.contains("Last success"),
            "an exceptional read expands:\n{screen}"
        );
    }

    #[test]
    fn case_artifacts_remain_mapped_to_their_owning_case_in_technical_details() {
        let mut current = snapshot();
        current.records[0].cases[0].stdout = Some("cases/long-context/stdout.log".to_owned());
        current.records[0].cases[0].stderr = Some("cases/long-context/stderr.log".to_owned());
        let mut app = App::default();
        app.accept(current);
        app.select_view(2);

        let screen = rendered(120, 80, &mut app);

        assert!(screen.contains("CASE ARTIFACTS"));
        assert!(screen.contains("long-context"));
        assert!(screen.contains("stdout  cases/long-context/stdout.log"));
        assert!(screen.contains("stderr  cases/long-context/stderr.log"));
    }

    #[test]
    fn wide_metrics_surface_groups_the_selector_and_draws_record_local_bars() {
        let mut app = App::default();
        app.accept(snapshot());
        app.select_view(2);
        let _ = app.handle_key(crossterm::event::KeyEvent::new(
            crossterm::event::KeyCode::Char('m'),
            crossterm::event::KeyModifiers::NONE,
        ));

        let screen = rendered(120, 32, &mut app);

        assert!(screen.contains("METRIC SELECTOR"));
        assert!(screen.contains("THROUGHPUT"));
        assert!(screen.contains("Request throughput"));
        assert!(screen.contains("CONCURRENCY"));
        assert!(screen.contains("c1"));
        assert!(screen.contains("c8"));
        assert!(screen.contains("1.25 req/s"));
        assert!(screen.contains("7.413 req/s"));
        assert!(screen.contains("cases 1–2/2"));
        assert!(screen.contains('█'));
        assert!(!screen.contains("SLO"));
    }

    #[test]
    fn narrow_metrics_surface_keeps_the_chart_and_contextual_keys() {
        let mut app = App::default();
        app.accept(snapshot());
        app.select_view(2);
        let _ = app.handle_key(crossterm::event::KeyEvent::new(
            crossterm::event::KeyCode::Char('m'),
            crossterm::event::KeyModifiers::NONE,
        ));

        let screen = rendered(80, 24, &mut app);

        assert!(!screen.contains("METRIC SELECTOR"));
        assert!(screen.contains("Request throughput"));
        assert!(screen.contains("THROUGHPUT"));
        assert!(screen.contains("c1"));
        assert!(screen.contains("c8"));
        assert!(screen.contains(" ↑↓  Metric"));
        assert!(screen.contains(" PgUp/PgDn  Cases"));
    }

    #[test]
    fn sixty_four_column_metrics_keeps_refresh_and_quit_visible() {
        let mut app = App::default();
        app.accept(snapshot());
        app.select_view(2);
        let _ = app.handle_key(crossterm::event::KeyEvent::new(
            crossterm::event::KeyCode::Char('m'),
            crossterm::event::KeyModifiers::NONE,
        ));

        let screen = rendered(64, 24, &mut app);
        let footer = screen.lines().last().unwrap_or_default();

        assert!(footer.contains(" r  Refresh"));
        assert!(footer.contains(" q  Quit"));
    }

    #[test]
    fn minimum_width_metrics_explicitly_ellipsize_long_dynamic_fields() {
        let mut current = snapshot();
        current.records[0].id =
            Some("bench-record-with-a-deliberately-long-operator-facing-identity".to_owned());
        current.records[0].cases.truncate(1);
        let case = &mut current.records[0].cases[0];
        case.id = Some("case-with-a-deliberately-long-identity".to_owned());
        case.load = crate::tui::CaseLoad::Unknown;
        case.status = Some("completed-with-a-deliberately-long-state".to_owned());
        case.metrics = BTreeMap::from([(
            "metric-that-is-deliberately-too-long-for-fifty-columns".to_owned(),
            1.0,
        )]);
        let mut app = App::default();
        app.accept(current);
        app.select_view(2);
        let _ = app.handle_key(crossterm::event::KeyEvent::new(
            crossterm::event::KeyCode::Char('m'),
            crossterm::event::KeyModifiers::NONE,
        ));

        let screen = rendered(50, 24, &mut app);

        assert!(
            screen
                .lines()
                .any(|line| line.contains("METRICS") && line.contains('…'))
        );
        assert!(
            screen
                .lines()
                .any(|line| line.contains("metric-that") && line.contains('…'))
        );
        assert!(screen.contains("OTHER"));
        assert!(
            screen
                .lines()
                .any(|line| line.contains("case-with") && line.contains('…'))
        );
    }

    #[test]
    fn minimum_width_find_keeps_the_query_tail_cursor_and_controls_visible() {
        let mut app = App::default();
        app.accept(snapshot());
        app.start_global_find();
        app.query = "abcdefghijklmnopqrstuvwxyz".repeat(4);

        let screen = rendered(50, 20, &mut app);

        assert!(screen.lines().any(|line| {
            line.contains("FIND")
                && line.contains('…')
                && line.contains("uvwxyz▌")
                && line.contains("Esc cancel")
        }));
    }

    #[test]
    fn global_find_from_narrow_metrics_returns_to_a_browsable_list() {
        let mut app = App::default();
        app.accept(snapshot());
        app.select_view(2);
        let _ = app.handle_key(crossterm::event::KeyEvent::new(
            crossterm::event::KeyCode::Char('m'),
            crossterm::event::KeyModifiers::NONE,
        ));
        let _ = app.handle_key(crossterm::event::KeyEvent::new(
            crossterm::event::KeyCode::Char('k'),
            crossterm::event::KeyModifiers::CONTROL,
        ));

        let screen = rendered(80, 24, &mut app);

        assert!(screen.contains("GLOBAL FIND"));
        assert!(screen.contains("bench random-8k1k"));
        assert!(screen.contains("record-failed"));
        assert!(!screen.contains("DETAILS"));
        assert!(!screen.contains("Request throughput"));
    }

    #[test]
    fn wide_global_find_is_a_labeled_cross_view_result_surface() {
        let mut app = App::default();
        app.accept(snapshot());
        app.start_global_find();

        let screen = rendered(120, 32, &mut app);

        assert!(screen.contains("GLOBAL FIND"));
        assert!(screen.contains("OPERATIONS"));
        assert!(screen.contains("RECORDS"));
        assert!(
            row(&screen, "bench random-8k1k").contains("EPH"),
            "{screen}"
        );
        assert!(row(&screen, "record-failed").contains("REC"), "{screen}");
        assert!(!screen.contains("DETAIL ·"));
    }

    #[test]
    fn lists_distinguish_empty_filtered_and_unavailable_states() {
        let mut empty = snapshot();
        empty.operations.clear();
        let mut app = App::default();
        app.accept(empty);
        app.select_view(1);
        assert!(rendered(80, 24, &mut app).contains("No active or retained operations"));

        let _ = app.handle_key(crossterm::event::KeyEvent::new(
            crossterm::event::KeyCode::Char('/'),
            crossterm::event::KeyModifiers::NONE,
        ));
        for character in "absent".chars() {
            let _ = app.handle_key(crossterm::event::KeyEvent::new(
                crossterm::event::KeyCode::Char(character),
                crossterm::event::KeyModifiers::NONE,
            ));
        }
        assert!(rendered(80, 24, &mut app).contains("No results for “absent”"));

        let mut unavailable = snapshot();
        unavailable.operations.clear();
        unavailable.operations_error = Some("observation directory denied".to_owned());
        app = App::default();
        app.accept(unavailable);
        app.select_view(1);
        let screen = rendered(80, 24, &mut app);
        assert!(screen.contains("operation observations"));
        assert!(screen.contains("refresh unavailable"));
        assert!(screen.contains("observation directory denied"));
    }

    #[test]
    fn workspace_catalog_is_grouped_by_definition_kind() {
        let mut current = snapshot();
        current.definitions = vec![
            DefinitionView {
                kind: "model".to_owned(),
                id: "qwen".to_owned(),
                relationship: "weights".to_owned(),
                fact_sections: Vec::new(),
                state: State::Live,
                observed_unix_ms: 2_000,
                last_success_unix_ms: 2_000,
                reason: None,
            },
            DefinitionView {
                kind: "bench".to_owned(),
                id: "random-8k1k".to_owned(),
                relationship: "standalone".to_owned(),
                fact_sections: Vec::new(),
                state: State::Live,
                observed_unix_ms: 2_000,
                last_success_unix_ms: 2_000,
                reason: None,
            },
        ];
        let mut app = App::default();
        app.accept(current);
        app.select_view(3);

        let screen = rendered(80, 28, &mut app);
        let bench = screen.find("BENCH");
        let model = screen.find("MODEL");
        assert!(matches!((bench, model), (Some(bench), Some(model)) if bench < model));
    }

    #[test]
    fn detail_labels_keep_their_gap_and_wrapped_values_keep_their_column() {
        let mut current = snapshot();
        current.definitions = vec![DefinitionView {
            kind: "bench".to_owned(),
            id: "random-8k1k".to_owned(),
            relationship: "requests · random".to_owned(),
            fact_sections: vec![crate::tui::FactSection {
                title: "POPULATION",
                rows: vec![
                    ("Warmup prompts / concurrency".to_owned(), "1".to_owned()),
                    (
                        "Note".to_owned(),
                        "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike november oscar papa quebec romeo sierra tango".to_owned(),
                    ),
                ],
            }],
            state: State::Live,
            observed_unix_ms: 2_000,
            last_success_unix_ms: 2_000,
            reason: None,
        }];
        let mut app = App::default();
        app.accept(current);
        app.select_view(3);

        let screen = rendered(120, 40, &mut app);
        let lines = screen.lines().collect::<Vec<_>>();
        assert!(
            screen.contains("Warmup prompts / concurrency  1"),
            "a long label keeps a gap before its value:\n{screen}"
        );
        let note = lines
            .iter()
            .position(|line| line.contains("Note"))
            .unwrap_or(usize::MAX);
        let value_column = lines
            .get(note)
            .and_then(|line| line.find("alpha"))
            .unwrap_or(usize::MAX);
        let continuation = lines.get(note + 1).copied().unwrap_or_default();
        assert_eq!(
            continuation
                .char_indices()
                .find(|(_, character)| character.is_alphabetic())
                .map(|(index, _)| index),
            Some(value_column),
            "a wrapped value continues under its value column:\n{screen}"
        );
    }

    #[test]
    fn glyphs_carry_status_and_the_focus_ring_follows_focus() -> Result<(), String> {
        let mut current = snapshot();
        let mut succeeded = current.records[0].clone();
        succeeded.id = Some("record-succeeded".to_owned());
        succeeded.status = Some("succeeded".to_owned());
        succeeded.error = None;
        current.records.push(succeeded);
        let mut app = App::default();
        app.accept(current);
        app.select_view(2);

        let border = |app: &mut App, x: u16| {
            let backend = TestBackend::new(120, 30);
            let mut terminal = Terminal::new(backend).map_err(|error| error.to_string())?;
            terminal
                .draw(|frame| {
                    render(
                        frame,
                        app,
                        RefreshStatus::Healthy {
                            interval: Duration::from_secs(1),
                        },
                    );
                })
                .map_err(|error| error.to_string())?;
            let buffer = terminal.backend().buffer().clone();
            let cell = buffer.cell((x, 4)).ok_or("no cell")?;
            Ok::<_, String>((cell.symbol().to_owned(), cell.fg))
        };
        let screen = rendered(120, 30, &mut app);
        assert!(
            row(&screen, "record-failed").contains("× bench"),
            "{screen}"
        );
        assert!(
            row(&screen, "record-succeeded").contains("✓ bench"),
            "{screen}"
        );

        let (corner, list_focused) = border(&mut app, 0)?;
        assert_eq!(corner, "╭");
        assert_eq!(list_focused, super::theme::BRAND);
        let _ = app.handle_key(crossterm::event::KeyEvent::new(
            crossterm::event::KeyCode::Enter,
            crossterm::event::KeyModifiers::NONE,
        ));
        let (_, list_unfocused) = border(&mut app, 0)?;
        assert_eq!(list_unfocused, app.palette.rule, "focus moved to detail");
        Ok(())
    }

    /// The list pane's part of the rendered line that holds a needle, for
    /// one-line row assertions; the detail pane beside it is cut off.
    fn row<'a>(screen: &'a str, needle: &str) -> &'a str {
        screen
            .lines()
            .map(|line| {
                ["│ │", "│ ╭", "╮ ╭", "╯ ╰"]
                    .iter()
                    .filter_map(|boundary| line.find(boundary))
                    .min()
                    .map_or(line, |end| &line[..end])
            })
            .find(|list| list.contains(needle) && !list.contains("DETAIL ·"))
            .unwrap_or_default()
    }

    /// The detail pane's part of the rendered line that holds a needle.
    fn row_in_detail<'a>(screen: &'a str, needle: &str) -> &'a str {
        screen
            .lines()
            .filter_map(|line| line.rfind("│ │").map(|start| &line[start + "│ │".len()..]))
            .find(|detail| detail.contains(needle))
            .unwrap_or_default()
    }

    fn key(app: &mut App, code: crossterm::event::KeyCode) {
        let _ = app.handle_key(crossterm::event::KeyEvent::new(
            code,
            crossterm::event::KeyModifiers::NONE,
        ));
    }

    fn two_records() -> Snapshot {
        let mut current = snapshot();
        let mut succeeded = current.records[0].clone();
        succeeded.id = Some("record-succeeded".to_owned());
        succeeded.status = Some("succeeded".to_owned());
        succeeded.error = None;
        current.records.push(succeeded);
        current
    }

    #[test]
    fn a_wide_console_shows_the_sidebar_and_one_line_rows() {
        let mut app = App::default();
        app.accept(two_records());
        app.select_view(2);

        let screen = rendered(160, 30, &mut app);
        let lines = screen.lines().collect::<Vec<_>>();
        assert!(screen.contains("◆ InferLab"), "{screen}");
        assert!(screen.contains("inferlab-vllm"), "{screen}");
        assert!(
            screen.contains("AUTO"),
            "the refresh indicator lives in the sidebar"
        );
        assert!(
            lines
                .iter()
                .any(|line| line.contains("3 Records") && line.contains('2')),
            "views carry their counts:\n{screen}"
        );
        let row = lines
            .iter()
            .find(|line| line.contains("record-failed"))
            .copied()
            .unwrap_or_default();
        assert!(
            row.contains("bench") && row.contains("failed") && row.contains("REC"),
            "one row carries kind, name, outcome, and authority: {row}"
        );
        assert!(
            !screen.contains("REC · failed"),
            "the two-line metadata row is gone:\n{screen}"
        );

        let narrow = rendered(120, 30, &mut app);
        assert!(
            !narrow.contains("◆ InferLab"),
            "below the sidebar width the header returns"
        );
        assert!(narrow.contains("1 Overview"));
    }

    #[test]
    fn f_cycles_the_status_filter_and_shows_counts() {
        let mut app = App::default();
        app.accept(two_records());
        app.select_view(2);

        let all = rendered(160, 30, &mut app);
        assert!(all.contains("record-failed") && all.contains("record-succeeded"));
        assert!(
            all.lines()
                .any(|line| line.contains("issues") && line.contains('1')),
            "{all}"
        );

        key(&mut app, crossterm::event::KeyCode::Char('f'));
        let issues = rendered(160, 30, &mut app);
        assert!(issues.contains("record-failed"), "{issues}");
        assert!(!issues.contains("record-succeeded"), "{issues}");

        key(&mut app, crossterm::event::KeyCode::Char('f'));
        let running = rendered(160, 30, &mut app);
        assert!(!running.contains("record-failed"), "{running}");

        key(&mut app, crossterm::event::KeyCode::Char('f'));
        let cycled = rendered(160, 30, &mut app);
        assert!(
            cycled.contains("record-succeeded"),
            "the filter cycles back to all"
        );

        app.select_view(3);
        key(&mut app, crossterm::event::KeyCode::Char('f'));
        let workspace = rendered(160, 30, &mut app);
        assert!(
            !workspace.contains("STATUS"),
            "Workspace has no status filter:\n{workspace}"
        );
    }

    #[test]
    fn overview_shows_attention_then_now_then_recent_by_day() {
        let mut current = two_records();
        // 2026-01-02 and 2026-01-01, midday UTC.
        current.records[1].finished_unix_ms = Some(1_767_355_200_000);
        let mut older = current.records[1].clone();
        older.id = Some("record-older".to_owned());
        older.finished_unix_ms = Some(1_767_268_800_000);
        current.records.push(older);
        let mut app = App::default();
        app.accept(current);

        let screen = rendered(160, 40, &mut app);
        let at = |needle: &str| screen.find(needle).unwrap_or(usize::MAX);
        assert!(at("ATTENTION") < at("NOW"), "{screen}");
        assert!(at("NOW") < at("RECENT · Jan 2"), "{screen}");
        assert!(at("RECENT · Jan 2") < at("RECENT · Jan 1"), "{screen}");
        assert!(at("record-succeeded") < at("RECENT · Jan 1"), "{screen}");
        assert!(at("RECENT · Jan 1") < at("record-older"), "{screen}");
    }

    #[test]
    fn an_active_operation_shows_its_progress_and_animates() {
        let mut app = App::default();
        app.accept(snapshot());

        let first = rendered(160, 40, &mut app);
        let progress = first
            .lines()
            .find(|line| line.contains("64/100"))
            .unwrap_or_default();
        assert!(
            progress.contains('█'),
            "a progress bar accompanies the position:\n{first}"
        );
        assert!(app.animates(), "an active operation animates its glyph");
        let glyph = |screen: &str| {
            row(screen, "bench random-8k1k")
                .chars()
                .find(|character| "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏".contains(*character))
        };
        let before = glyph(&first);
        app.advance_animation();
        let after = glyph(&rendered(160, 40, &mut app));
        assert!(
            before.is_some() && after.is_some() && before != after,
            "{first}"
        );
    }

    #[test]
    fn records_fold_children_beneath_their_parent_by_day() {
        let mut current = snapshot();
        current.operations.clear();
        let parent = &mut current.records[0];
        parent.id = Some("parent-run".to_owned());
        parent.kind = "recipe".to_owned();
        parent.child_refs = vec![
            "child-serve".to_owned(),
            "parent-run-bench-000".to_owned(),
            "child-missing".to_owned(),
        ];
        // 2026-01-02, midday UTC.
        parent.finished_unix_ms = Some(1_767_355_200_000);
        let mut server = current.records[0].clone();
        server.id = Some("child-serve".to_owned());
        server.kind = "server".to_owned();
        server.status = Some("stopped".to_owned());
        server.child_refs = Vec::new();
        let mut bench = server.clone();
        bench.id = Some("parent-run-bench-000".to_owned());
        bench.kind = "bench".to_owned();
        bench.status = Some("succeeded".to_owned());
        current.child_records = vec![server, bench];
        let mut app = App::default();
        app.accept(current);
        app.select_view(2);

        let collapsed = rendered(160, 30, &mut app);
        assert!(
            collapsed.contains("Jan 2"),
            "records group by day:\n{collapsed}"
        );
        assert!(row(&collapsed, "parent-run").contains('▸'), "{collapsed}");
        assert!(!collapsed.contains("bench-000"), "children start collapsed");

        key(&mut app, crossterm::event::KeyCode::Right);
        let expanded = rendered(160, 30, &mut app);
        assert!(row(&expanded, "parent-run").contains('▾'), "{expanded}");
        assert!(row(&expanded, "child-serve").contains('├'), "{expanded}");
        assert!(
            row(&expanded, "child-missing").contains("refresh unavail"),
            "{expanded}"
        );
        assert!(
            expanded.contains("child record cannot be read"),
            "{expanded}"
        );
        assert!(row(&expanded, "child-missing").contains('╰'), "{expanded}");

        key(&mut app, crossterm::event::KeyCode::Down);
        key(&mut app, crossterm::event::KeyCode::Down);
        let child = rendered(160, 30, &mut app);
        assert!(child.contains("DETAIL · bench / bench-000"), "{child}");
        assert!(
            !row(&child, "bench-000").contains("parent-run-bench"),
            "a child is named relative to its parent:\n{child}"
        );

        key(&mut app, crossterm::event::KeyCode::Left);
        let folded = rendered(160, 30, &mut app);
        assert!(!folded.contains("bench-000"), "{folded}");
        assert!(folded.contains("DETAIL · recipe / parent-run"), "{folded}");
    }

    #[test]
    fn log_detail_identifies_the_selected_reference_and_search_matches()
    -> Result<(), Box<dyn std::error::Error>> {
        let directory = tempfile::tempdir()?;
        std::fs::write(directory.path().join("first.log"), "first\n")?;
        std::fs::write(
            directory.path().join("second.log"),
            "before\nneedle\nafter\n",
        )?;
        let mut current = snapshot();
        current.root = directory.path().to_path_buf();
        current.operations.clear();
        current.records[0].log_refs = vec!["first.log".to_owned(), "second.log".to_owned()];
        let mut app = App::default();
        app.accept(current);
        app.select_view(2);
        let _ = app.handle_key(crossterm::event::KeyEvent::new(
            crossterm::event::KeyCode::Enter,
            crossterm::event::KeyModifiers::NONE,
        ));
        let _ = app.handle_key(crossterm::event::KeyEvent::new(
            crossterm::event::KeyCode::Char(']'),
            crossterm::event::KeyModifiers::NONE,
        ));

        let selected = rendered(100, 30, &mut app);
        assert!(selected.contains("log 2/2 second.log"));

        let _ = app.handle_key(crossterm::event::KeyEvent::new(
            crossterm::event::KeyCode::Char('/'),
            crossterm::event::KeyModifiers::NONE,
        ));
        for character in "needle".chars() {
            let _ = app.handle_key(crossterm::event::KeyEvent::new(
                crossterm::event::KeyCode::Char(character),
                crossterm::event::KeyModifiers::NONE,
            ));
        }
        let searched = rendered(100, 30, &mut app);
        assert!(searched.contains("LOG SEARCH"));
        assert!(searched.contains("1 for “needle”"));
        assert!(searched.contains("2 │ needle"));
        assert!(!searched.contains("1 │ before"));
        Ok(())
    }

    #[test]
    fn empty_log_tail_is_distinct_from_an_empty_search_result()
    -> Result<(), Box<dyn std::error::Error>> {
        let directory = tempfile::tempdir()?;
        std::fs::write(directory.path().join("empty.log"), "")?;
        let mut current = snapshot();
        current.root = directory.path().to_path_buf();
        current.operations.clear();
        current.records[0].log_refs = vec!["empty.log".to_owned()];
        let mut app = App::default();
        app.accept(current);
        app.select_view(2);
        let _ = app.handle_key(crossterm::event::KeyEvent::new(
            crossterm::event::KeyCode::Enter,
            crossterm::event::KeyModifiers::NONE,
        ));

        let screen = rendered(100, 80, &mut app);

        assert!(screen.contains("Log tail is empty"));
        assert!(!screen.contains("No matching lines"));
        Ok(())
    }

    #[test]
    fn missing_metric_is_not_drawn_as_zero_and_keeps_the_failed_state() {
        let mut app = App::default();
        app.accept(snapshot());
        app.select_view(2);
        let _ = app.handle_key(crossterm::event::KeyEvent::new(
            crossterm::event::KeyCode::Char('m'),
            crossterm::event::KeyModifiers::NONE,
        ));
        let _ = app.handle_key(crossterm::event::KeyEvent::new(
            crossterm::event::KeyCode::Down,
            crossterm::event::KeyModifiers::NONE,
        ));

        let screen = rendered(120, 32, &mut app);

        assert!(screen.contains("P95 TTFT"));
        assert!(screen.contains("c1"));
        assert!(screen.contains("—"));
        assert!(screen.contains("failed"));
        assert!(screen.contains("47.8 ms"));
    }

    #[test]
    fn tiny_layout_reports_required_and_current_dimensions() {
        let mut app = App::default();

        let screen = rendered(40, 10, &mut app);

        assert!(screen.contains("50×12 required"));
        assert!(screen.contains("40×10 current"));
    }
}
