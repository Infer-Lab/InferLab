use super::text::{display_width, pad_left, pad_right, wrap};
use super::theme::Palette;
use crate::console::metrics::human_number;
use crate::tui::metrics::RecordMetrics;
use crate::tui::{DetailSection, DisplayEntry};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use std::path::Path;

/// The longest label a section aligns its values to; a longer one keeps its own
/// line width and pushes only its value right.
const MAX_LABEL_WIDTH: usize = 28;

pub(super) fn lines(
    p: Palette,
    entry: &DisplayEntry,
    presentation_unix_ms: u64,
    width: u16,
    metrics: Option<&RecordMetrics>,
) -> Vec<Line<'static>> {
    let (kind, name) = entry
        .title
        .split_once(" / ")
        .unwrap_or(("", entry.title.as_str()));
    let mut lines = vec![
        Line::from(vec![
            Span::styled(
                format!("{} ", entry.tone.glyph()),
                Style::default()
                    .fg(p.tone(entry.tone))
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                if kind.is_empty() {
                    String::new()
                } else {
                    format!("{kind}  ")
                },
                Style::default().fg(p.muted),
            ),
            Span::styled(
                name.to_owned(),
                Style::default().add_modifier(Modifier::BOLD),
            ),
        ]),
        Line::from(""),
        detail_context(p, entry),
        Line::from(""),
    ];
    for section in &entry.details {
        match (section.title, metrics) {
            // A current read needs one line; an exceptional one keeps every fact.
            ("SOURCE HEALTH", _) if entry.state == crate::tui::State::Live => {
                append_current_health(p, &mut lines, section, presentation_unix_ms);
            }
            ("METRICS", Some(metrics)) => {
                append_metric_table(p, &mut lines, metrics, usize::from(width));
            }
            _ => append_section(
                p,
                &mut lines,
                section,
                presentation_unix_ms,
                usize::from(width),
            ),
        }
    }
    lines
}

/// The status and authority pills that open a detail, then its summary.
fn detail_context(p: Palette, entry: &DisplayEntry) -> Line<'static> {
    let mut spans = Vec::new();
    if let Some(lifecycle) = entry.lifecycle.as_deref() {
        spans.extend(pill(
            &lifecycle.to_uppercase(),
            p.tone(entry.tone),
            p.on_fill,
            p,
        ));
        spans.push(Span::raw(" "));
    }
    spans.extend(pill(
        &format!("{} {}", entry.authority.badge(), entry.authority.label()),
        p.chip,
        p.on_surface,
        p,
    ));
    if entry.state != crate::tui::State::Live {
        spans.push(Span::raw(" "));
        spans.extend(pill(
            &format!("refresh {}", entry.state.label()),
            p.state(entry.state),
            p.on_fill,
            p,
        ));
    }
    if !entry.summary.is_empty() {
        spans.push(Span::styled(
            format!("  {}", entry.summary),
            Style::default().fg(p.muted),
        ));
    }
    Line::from(spans)
}

/// A filled label with half-block caps, so it reads as a pill.
fn pill(text: &str, fill: Color, fg: Color, p: Palette) -> [Span<'static>; 3] {
    let surface = p.panel.unwrap_or(Color::Reset);
    [
        Span::styled("▐", Style::default().fg(fill).bg(surface)),
        Span::styled(
            format!(" {text} "),
            Style::default()
                .fg(fg)
                .bg(fill)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled("▌", Style::default().fg(fill).bg(surface)),
    ]
}

/// Case labels without the prefix they all share up to a `-`, so columns name
/// what differs: `concurrency-000`, `concurrency-001` become `000`, `001`.
fn distinct_suffixes<'a>(labels: &[&'a str]) -> Vec<&'a str> {
    let Some(first) = labels.first() else {
        return Vec::new();
    };
    let shared = labels.iter().skip(1).fold(first.len(), |shared, label| {
        first
            .bytes()
            .zip(label.bytes())
            .take(shared)
            .take_while(|(left, right)| left == right)
            .count()
    });
    let cut = if labels.len() > 1 {
        first
            .get(..shared)
            .and_then(|prefix| prefix.rfind('-'))
            .map_or(0, |dash| dash + 1)
    } else {
        0
    };
    labels
        .iter()
        .map(|label| {
            label
                .get(cut..)
                .filter(|rest| !rest.is_empty())
                .unwrap_or(label)
        })
        .collect()
}

/// A current read condenses Source Health to one line.
fn append_current_health(
    p: Palette,
    lines: &mut Vec<Line<'static>>,
    section: &DetailSection,
    presentation_unix_ms: u64,
) {
    let value = |label: &str| {
        section
            .rows
            .iter()
            .find(|(row, _)| row == label)
            .map(|(_, value)| value.render(presentation_unix_ms))
            .unwrap_or_default()
    };
    lines.push(Line::from(Span::styled(
        section.title,
        Style::default()
            .fg(p.accent_soft)
            .add_modifier(Modifier::BOLD),
    )));
    lines.push(Line::from(vec![
        Span::styled("  ● ", Style::default().fg(p.success)),
        Span::styled("current", Style::default().fg(p.text)),
        Span::styled(
            format!(" · {}", value("Authority")),
            Style::default().fg(p.muted),
        ),
    ]));
    lines.push(Line::from(Span::styled(
        format!("    refreshed {}", value("Refreshed")),
        Style::default().fg(p.muted),
    )));
    lines.push(Line::from(""));
}

/// A workload record's recorded case metrics as one table: a row per metric,
/// a column per case, and a trend across the cases.
fn append_metric_table(
    p: Palette,
    lines: &mut Vec<Line<'static>>,
    metrics: &RecordMetrics,
    width: usize,
) {
    const SPARK: [&str; 8] = ["▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"];
    const LABEL: usize = 22;
    const COLUMN: usize = 10;
    let cases = metrics.points.first().map_or(0, Vec::len);
    let unit_width = 6;
    let shown = width
        .saturating_sub(2 + LABEL + unit_width + cases + 2)
        .checked_div(COLUMN)
        .unwrap_or(0)
        .clamp(1, cases.max(1));
    lines.push(Line::from(vec![
        Span::styled(
            "METRICS",
            Style::default()
                .fg(p.accent_soft)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(
            format!("  {cases} case{}", if cases == 1 { "" } else { "s" }),
            Style::default().fg(p.muted),
        ),
    ]));
    let mut header = vec![Span::raw(" ".repeat(2 + LABEL))];
    let labels = metrics
        .points
        .first()
        .map(|points| {
            points
                .iter()
                .map(|point| point.label.as_str())
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    for label in distinct_suffixes(&labels).iter().take(shown) {
        header.push(Span::styled(
            format!(" {}", pad_left(label, COLUMN - 1)),
            Style::default().fg(p.muted).add_modifier(Modifier::BOLD),
        ));
    }
    if cases > shown {
        header.push(Span::styled(
            format!("  +{} more", cases - shown),
            Style::default().fg(p.faint),
        ));
    }
    lines.push(Line::from(header));
    let mut family = None;
    for (descriptor, points) in metrics.catalog.iter().zip(&metrics.points) {
        if family != Some(descriptor.family) {
            family = Some(descriptor.family);
            lines.push(Line::from(Span::styled(
                format!("  {}", descriptor.family.label()),
                Style::default().fg(p.accent_soft),
            )));
        }
        let mut row = vec![Span::styled(
            pad_right(&format!("  {}", descriptor.label), 2 + LABEL),
            Style::default().fg(p.text),
        )];
        for point in points.iter().take(shown) {
            let value = point.value.map_or_else(
                || "—".to_owned(),
                |value| human_number(descriptor.unit.display_value(value)),
            );
            row.push(Span::styled(
                format!(" {}", pad_left(&value, COLUMN - 1)),
                Style::default().fg(p.text).add_modifier(Modifier::BOLD),
            ));
        }
        row.push(Span::styled(
            format!(" {}", pad_right(descriptor.unit.label(), unit_width - 1)),
            Style::default().fg(p.muted),
        ));
        let values = points
            .iter()
            .filter_map(|point| point.value)
            .collect::<Vec<_>>();
        let (low, high) = values
            .iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(low, high), value| {
                (low.min(*value), high.max(*value))
            });
        if values.len() > 1 && high > low {
            let trend = points
                .iter()
                .map(|point| {
                    point.value.map_or(" ", |value| {
                        let step = ((value - low) / (high - low) * 7.0).round() as usize;
                        SPARK[step.min(7)]
                    })
                })
                .collect::<String>();
            row.push(Span::styled(
                format!(" {trend}"),
                Style::default().fg(super::theme::BRAND),
            ));
        }
        lines.push(Line::from(row));
    }
    lines.push(Line::from(vec![
        Span::raw("  "),
        Span::styled(" m ", p.keycap()),
        Span::styled(
            " compare one metric across cases",
            Style::default().fg(p.muted),
        ),
    ]));
    lines.push(Line::from(""));
}

fn append_section(
    p: Palette,
    lines: &mut Vec<Line<'static>>,
    section: &DetailSection,
    presentation_unix_ms: u64,
    width: usize,
) {
    lines.push(Line::from(Span::styled(
        section.title,
        Style::default()
            .fg(p.accent_soft)
            .add_modifier(Modifier::BOLD),
    )));
    let value_style = if matches!(section.title, "PROGRESS" | "METRICS") {
        Style::default().fg(p.accent).add_modifier(Modifier::BOLD)
    } else {
        Style::default().fg(Color::Reset)
    };
    // Each section sizes its label column to its own labels, so a long label
    // never runs into its value and short ones are not pushed far right.
    let label_width = section
        .rows
        .iter()
        .map(|(label, _)| display_width(label))
        .max()
        .unwrap_or(0)
        .min(MAX_LABEL_WIDTH);
    let value_column = 2 + label_width + 2;
    let value_width = width.saturating_sub(value_column).max(16);
    for (label, value) in &section.rows {
        let value = value.render(presentation_unix_ms);
        let mut first = true;
        for paragraph in value.lines() {
            for text in wrap(paragraph, value_width) {
                let lead = if first {
                    let padding = label_width.saturating_sub(display_width(label));
                    Span::styled(
                        format!("  {label}{}  ", " ".repeat(padding)),
                        Style::default().fg(p.muted),
                    )
                } else {
                    Span::raw(" ".repeat(value_column))
                };
                first = false;
                lines.push(Line::from(vec![lead, Span::styled(text, value_style)]));
            }
        }
        if first {
            lines.push(Line::from(Span::styled(
                format!("  {label}"),
                Style::default().fg(p.muted),
            )));
        }
    }
    for body in &section.body {
        for line in body.lines() {
            lines.push(Line::from(Span::styled(
                format!("  {line}"),
                Style::default().fg(p.secondary),
            )));
        }
    }
    lines.push(Line::from(""));
}

pub(super) fn append_log(
    p: Palette,
    lines: &mut Vec<Line<'static>>,
    path: &str,
    index: usize,
    count: usize,
    query: Option<&str>,
    projected: &[String],
) {
    lines.push(Line::from(Span::styled(
        format!("LOG {index} OF {count}"),
        Style::default()
            .fg(p.accent_soft)
            .add_modifier(Modifier::BOLD),
    )));
    lines.push(Line::from(vec![
        Span::styled("  File          ", Style::default().fg(p.muted)),
        Span::styled(
            Path::new(path)
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or(path)
                .to_owned(),
            Style::default().fg(Color::Reset),
        ),
    ]));
    lines.push(Line::from(vec![
        Span::styled("  Reference     ", Style::default().fg(p.muted)),
        Span::styled(path.to_owned(), Style::default().fg(p.secondary)),
    ]));
    if let Some(query) = query {
        lines.push(Line::from(vec![
            Span::styled("  Matches       ", Style::default().fg(p.muted)),
            Span::styled(
                format!("{} for “{query}”", projected.len()),
                Style::default().fg(p.accent),
            ),
        ]));
    }
    lines.push(Line::from(""));
    if projected.is_empty() {
        lines.push(Line::from(Span::styled(
            if query.is_some() {
                "  No matching lines"
            } else {
                "  Log tail is empty"
            },
            Style::default().fg(p.muted),
        )));
    }
    lines.extend(projected.iter().map(|line| {
        Line::from(Span::styled(
            format!("  {line}"),
            Style::default().fg(p.secondary),
        ))
    }));
}

#[cfg(test)]
mod tests {
    use super::distinct_suffixes;

    #[test]
    fn case_columns_drop_only_the_prefix_every_label_shares() {
        assert_eq!(
            distinct_suffixes(&["concurrency-000", "concurrency-001"]),
            ["000", "001"]
        );
        assert_eq!(distinct_suffixes(&["c1", "c8"]), ["c1", "c8"]);
        assert_eq!(distinct_suffixes(&["prefill-8k"]), ["prefill-8k"]);
    }
}
