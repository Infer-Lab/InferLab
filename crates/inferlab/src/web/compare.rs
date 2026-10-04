//! Bench comparison across workspaces ([[RFC-0012:C-VIEWS]]): one metric of
//! the selected records, each record one series placed by its cases'
//! recorded effective load, one chart per load kind, and a table of the same
//! values. Loads without a value stay in the table; nothing is interpreted
//! against an SLO.

use super::hub::{Feed, Generation};
use super::pages::{encode, layout};
use crate::console::metrics::{self, LoadGroup, MetricDescriptor, MetricPoint, human_number};
use crate::console::{CaseLoad, RecordView};
use maud::{Markup, html};
use std::sync::Arc;

/// Series colors; the stylesheet maps `s0`..`s7` to a palette readable on
/// both color schemes.
const SERIES_COLORS: usize = 8;
const WIDTH: f64 = 640.0;
const HEIGHT: f64 = 260.0;
const LEFT: f64 = 56.0;
const RIGHT: f64 = 16.0;
const TOP: f64 = 14.0;
const BOTTOM: f64 = 34.0;

/// What the comparison URL selects: `r=<workspace>:<record>` per record, the
/// metric, and the picker's text filter.
pub(super) struct Selection {
    pub(super) records: Vec<(String, String)>,
    pub(super) metric: Option<String>,
    pub(super) filter: String,
}

impl Selection {
    pub(super) fn parse(query: &str) -> Self {
        let mut selection = Self {
            records: Vec::new(),
            metric: None,
            filter: String::new(),
        };
        for (key, value) in form_urlencoded::parse(query.as_bytes()) {
            match key.as_ref() {
                "r" => {
                    if let Some((workspace, record)) = value.split_once(':') {
                        let pair = (workspace.to_owned(), record.to_owned());
                        if !selection.records.contains(&pair) {
                            selection.records.push(pair);
                        }
                    }
                }
                "metric" if !value.is_empty() => selection.metric = Some(value.into_owned()),
                "q" => selection.filter = value.trim().to_owned(),
                _ => {}
            }
        }
        selection
    }
}

/// The comparison link that starts from one record.
pub(super) fn href(workspace_id: &str, record: &str) -> String {
    format!("/compare?r={}:{}", encode(workspace_id), encode(record))
}

/// One workspace's latest generation, if its first read has completed.
pub(super) struct Source {
    pub(super) feed: Arc<Feed>,
    /// The workspace's name among the registered ones.
    pub(super) label: String,
    pub(super) generation: Option<Arc<Generation>>,
}

struct Candidate<'a> {
    workspace_id: &'a str,
    workspace: String,
    record: &'a RecordView,
    key: &'a str,
}

/// One selected record's points for the selected metric.
struct Series {
    color: usize,
    workspace: String,
    record: String,
    label: String,
    points: Vec<MetricPoint>,
}

pub(super) fn page(sources: &[Source], selection: &Selection, now_unix_ms: u64) -> Markup {
    let candidates = candidates(sources);
    let mut missing = Vec::new();
    let mut chosen = Vec::new();
    for (workspace_id, key) in &selection.records {
        match candidates
            .iter()
            .find(|candidate| candidate.workspace_id == workspace_id && candidate.key == key)
        {
            Some(candidate) => chosen.push(candidate),
            None => missing.push(key.as_str()),
        }
    }
    let cases = chosen
        .iter()
        .flat_map(|candidate| candidate.record.cases.iter().cloned())
        .collect::<Vec<_>>();
    let catalog = metrics::catalog(&cases);
    let descriptor = selection
        .metric
        .as_deref()
        .and_then(|name| catalog.iter().find(|descriptor| descriptor.name == name))
        .or_else(|| catalog.first());
    let series = descriptor.map_or_else(Vec::new, |descriptor| {
        chosen
            .iter()
            .enumerate()
            .map(|(index, candidate)| Series {
                color: index % SERIES_COLORS,
                workspace: candidate.workspace.clone(),
                record: metrics::record_label(candidate.record),
                label: format!(
                    "{} · {}",
                    candidate.workspace,
                    metrics::record_label(candidate.record)
                ),
                points: metrics::points(&candidate.record.cases, &descriptor.name),
            })
            .collect()
    });
    let filter = selection.filter.to_lowercase();
    let body = html! {
        header.workspace-head {
            div {
                h1 { "Compare Bench records" }
                div.meta { "One metric across records, placed by each case's recorded load." }
            }
        }
        div.split.compare {
            section.panel.picker {
                form method="get" action="/compare" {
                    div.pathbar {
                        input type="search" name="q" value=(selection.filter) placeholder="Filter records" aria-label="Filter records";
                        button.quiet type="submit" { "Filter" }
                    }
                    @if !catalog.is_empty() {
                        label.metric-select {
                            span.muted { "Metric" }
                            select name="metric" onchange="this.form.submit()" {
                                @for option in &catalog {
                                    option value=(option.name) selected[descriptor.is_some_and(|descriptor| descriptor.name == option.name)] {
                                        (option.heading()) " (" (option.unit.label()) ")"
                                    }
                                }
                            }
                        }
                    }
                    @for source in sources {
                        @let listed = candidates.iter().filter(|candidate| candidate.workspace_id == source.feed.id && (filter.is_empty() || candidate.key.to_lowercase().contains(&filter) || selection.records.iter().any(|(workspace, key)| workspace == candidate.workspace_id && key == candidate.key))).collect::<Vec<_>>();
                        h2.group { (source.label.clone()) }
                        @if source.generation.is_none() {
                            p.empty { "Waiting for the first complete read of this workspace…" }
                        } @else if listed.is_empty() {
                            p.empty { "No Bench records." }
                        }
                        @for candidate in listed {
                            @let value = format!("{}:{}", candidate.workspace_id, candidate.key);
                            label.pick {
                                input type="checkbox" name="r" value=(value) checked[selection.records.iter().any(|(workspace, key)| workspace == candidate.workspace_id && key == candidate.key)];
                                span.name title=(candidate.key) { (metrics::record_label(candidate.record)) }
                                span.age { (candidate.record.started_unix_ms.map(|started| crate::console::views::compact_age(now_unix_ms.saturating_sub(started))).unwrap_or_default()) }
                                span.summary { (load_summary(candidate.record)) }
                            }
                        }
                    }
                    div.picker-actions { button type="submit" { "Compare" } }
                }
            }
            section.panel.results {
                @if !missing.is_empty() {
                    p.warning { "Not available as Bench records: " (missing.join(", ")) }
                }
                @match descriptor {
                    None => p.empty { "Select Bench records with case metrics to compare." },
                    Some(descriptor) => (results(descriptor, &series)),
                }
            }
        }
    };
    layout("Compare", body)
}

fn candidates(sources: &[Source]) -> Vec<Candidate<'_>> {
    let mut candidates = Vec::new();
    for source in sources {
        let Some(generation) = &source.generation else {
            continue;
        };
        let mut records = generation
            .snapshot
            .records
            .iter()
            .chain(&generation.snapshot.child_records)
            .filter(|record| record.kind == "bench" && !record.cases.is_empty())
            .filter_map(|record| {
                record.id.as_deref().map(|key| Candidate {
                    workspace_id: &source.feed.id,
                    workspace: source.label.clone(),
                    record,
                    key,
                })
            })
            .collect::<Vec<_>>();
        records.sort_by(|left, right| {
            right
                .record
                .started_unix_ms
                .cmp(&left.record.started_unix_ms)
                .then(left.key.cmp(right.key))
        });
        records.dedup_by(|left, right| left.key == right.key);
        candidates.extend(records);
    }
    candidates
}

/// The record's case count and the loads it swept, for choosing records.
fn load_summary(record: &RecordView) -> String {
    let span = |values: Vec<f64>, prefix: &str| {
        let low = values.iter().copied().fold(f64::INFINITY, f64::min);
        let high = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        if values.is_empty() {
            None
        } else if low == high {
            Some(format!("{prefix}{}", metrics::concise_number(low)))
        } else {
            Some(format!(
                "{prefix}{}–{prefix}{}",
                metrics::concise_number(low),
                metrics::concise_number(high)
            ))
        }
    };
    let concurrency = record
        .cases
        .iter()
        .filter_map(|case| match case.load {
            CaseLoad::Concurrency(value) => Some(f64::from(value)),
            _ => None,
        })
        .collect();
    let rate = record
        .cases
        .iter()
        .filter_map(|case| match case.load {
            CaseLoad::RequestRate(value) => Some(value),
            _ => None,
        })
        .collect();
    let loads = [span(concurrency, "c"), span(rate, "r")]
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
    let cases = record.cases.len();
    let noun = if cases == 1 { "case" } else { "cases" };
    if loads.is_empty() {
        format!("{cases} {noun}")
    } else {
        format!("{cases} {noun} · {}", loads.join(", "))
    }
}

fn results(descriptor: &MetricDescriptor, series: &[Series]) -> Markup {
    html! {
        h3 { (descriptor.heading()) span.muted { " · " (descriptor.unit.label()) } }
        ul.legend {
            @for line in series {
                li { span.swatch.(format!("s{}", line.color)) {} (line.label) }
            }
        }
        @for group in [LoadGroup::Concurrency, LoadGroup::RequestRate] {
            @if series.iter().any(|line| line.points.iter().any(|point| point.group == group)) {
                section.load-group {
                    h3 { (group.label()) }
                    (chart(descriptor, group, series))
                    (placed_table(descriptor, group, series))
                }
            }
        }
        @for group in [LoadGroup::Unbounded, LoadGroup::Case] {
            @if series.iter().any(|line| line.points.iter().any(|point| point.group == group)) {
                section.load-group {
                    h3 { (group.label()) }
                    p.muted { "These cases record no numeric load, so they are listed rather than charted." }
                    (unplaced_table(descriptor, group, series))
                }
            }
        }
        p.muted.note { "— the case recorded no value for this metric; a blank cell: the record has no case at that load." }
    }
}

/// The numeric load a point is placed at, for the charted load kinds.
fn load_value(load: &CaseLoad) -> Option<f64> {
    match *load {
        CaseLoad::Concurrency(value) => Some(f64::from(value)),
        CaseLoad::RequestRate(value) => Some(value),
        CaseLoad::UnboundedRequestRate | CaseLoad::Unknown => None,
    }
}

/// The distinct loads of one group across all series, ascending.
fn loads(group: LoadGroup, series: &[Series]) -> Vec<f64> {
    let mut loads = series
        .iter()
        .flat_map(|line| &line.points)
        .filter(|point| point.group == group)
        .filter_map(|point| load_value(&point.load))
        .collect::<Vec<_>>();
    loads.sort_by(f64::total_cmp);
    loads.dedup();
    loads
}

fn load_label(group: LoadGroup, load: f64) -> String {
    let prefix = if group == LoadGroup::Concurrency {
        "c"
    } else {
        "r"
    };
    format!("{prefix}{}", metrics::concise_number(load))
}

fn display(descriptor: &MetricDescriptor, point: &MetricPoint) -> Option<f64> {
    point
        .value
        .map(|value| descriptor.unit.display_value(value))
}

fn placed_table(descriptor: &MetricDescriptor, group: LoadGroup, series: &[Series]) -> Markup {
    let lines = series
        .iter()
        .filter(|line| line.points.iter().any(|point| point.group == group))
        .collect::<Vec<_>>();
    html! {
        table.metrics.comparison {
            thead { tr { th { "load" } @for line in &lines { th.series { span.swatch.(format!("s{}", line.color)) {} span.record { (line.record) } br; span.muted { (line.workspace) } } } } }
            tbody {
                @for load in loads(group, series) {
                    tr {
                        th { (load_label(group, load)) }
                        @for line in &lines {
                            @let values = line.points.iter().filter(|point| point.group == group && load_value(&point.load) == Some(load)).map(|point| display(descriptor, point)).collect::<Vec<_>>();
                            td { (values.iter().map(|value| value.map_or_else(|| "—".to_owned(), human_number)).collect::<Vec<_>>().join(" / ")) }
                        }
                    }
                }
            }
        }
    }
}

fn unplaced_table(descriptor: &MetricDescriptor, group: LoadGroup, series: &[Series]) -> Markup {
    html! {
        table.metrics.comparison {
            thead { tr { th { "record" } th { "case" } th { (descriptor.unit.label()) } } }
            tbody {
                @for line in series {
                    @for point in line.points.iter().filter(|point| point.group == group) {
                        tr {
                            th { span.swatch.(format!("s{}", line.color)) {} (line.label) }
                            td { (point.label) }
                            td { (display(descriptor, point).map_or_else(|| "—".to_owned(), human_number)) }
                        }
                    }
                }
            }
        }
    }
}

/// A value ceiling and grid step that read as round numbers.
fn nice_axis(high: f64) -> (f64, f64) {
    if high <= 0.0 || !high.is_finite() {
        return (1.0, 0.25);
    }
    let raw = high / 5.0;
    let magnitude = 10_f64.powf(raw.log10().floor());
    let step = [1.0, 2.0, 2.5, 5.0, 10.0]
        .into_iter()
        .map(|factor| factor * magnitude)
        .find(|step| *step >= raw)
        .unwrap_or(10.0 * magnitude);
    ((high / step).ceil() * step, step)
}

fn chart(descriptor: &MetricDescriptor, group: LoadGroup, series: &[Series]) -> Markup {
    let loads = loads(group, series);
    let (Some(&low), Some(&high)) = (loads.first(), loads.last()) else {
        return html! {};
    };
    // Sweeps double their load, so a wide positive range reads on a log axis.
    let logarithmic = low > 0.0 && high / low >= 8.0;
    let scale = |load: f64| if logarithmic { load.log2() } else { load };
    let plot_width = WIDTH - LEFT - RIGHT;
    let plot_height = HEIGHT - TOP - BOTTOM;
    let x = |load: f64| {
        if (scale(high) - scale(low)).abs() < f64::EPSILON {
            LEFT + plot_width / 2.0
        } else {
            LEFT + (scale(load) - scale(low)) / (scale(high) - scale(low)) * plot_width
        }
    };
    let ceiling = series
        .iter()
        .flat_map(|line| &line.points)
        .filter(|point| point.group == group)
        .filter_map(|point| display(descriptor, point))
        .fold(0.0_f64, f64::max);
    let (top, step) = nice_axis(ceiling);
    let y = |value: f64| TOP + plot_height - value.max(0.0) / top * plot_height;
    let grid = (0..)
        .map(|index| f64::from(index) * step)
        .take_while(|value| *value <= top + step / 2.0)
        .collect::<Vec<_>>();
    // Label at most a dozen loads so the axis stays legible.
    let label_every = loads.len().div_ceil(12).max(1);
    let lines = series
        .iter()
        .map(|line| {
            let mut placed = line
                .points
                .iter()
                .filter(|point| point.group == group)
                .filter_map(|point| load_value(&point.load).map(|load| (load, point)))
                .collect::<Vec<_>>();
            placed.sort_by(|left, right| left.0.total_cmp(&right.0));
            (line, placed)
        })
        .collect::<Vec<_>>();
    html! {
        svg.chart viewBox=(format!("0 0 {WIDTH} {HEIGHT}")) role="img" aria-label=(format!("{} by {}", descriptor.heading(), group.label().to_lowercase())) {
            @for value in &grid {
                line.grid x1=(LEFT) x2=(WIDTH - RIGHT) y1=(coordinate(y(*value))) y2=(coordinate(y(*value))) {}
                text.tick x=(LEFT - 6.0) y=(coordinate(y(*value) + 4.0)) text-anchor="end" { (human_number(*value)) }
            }
            @for (index, load) in loads.iter().enumerate() {
                @if index % label_every == 0 || index + 1 == loads.len() {
                    text.tick x=(coordinate(x(*load))) y=(HEIGHT - BOTTOM + 16.0) text-anchor="middle" { (load_label(group, *load)) }
                }
            }
            text.axis x=(LEFT + plot_width / 2.0) y=(HEIGHT - 4.0) text-anchor="middle" {
                (if group == LoadGroup::Concurrency { "concurrency" } else { "request rate (req/s)" })
                (if logarithmic { " · log scale" } else { "" })
            }
            @for (line, placed) in &lines {
                @for segment in segments(descriptor, placed) {
                    @if segment.len() > 1 {
                        polyline.line.(format!("s{}", line.color)) points=(segment.iter().map(|(load, value)| format!("{},{}", coordinate(x(*load)), coordinate(y(*value)))).collect::<Vec<_>>().join(" ")) {}
                    }
                }
                @for (load, point) in placed {
                    @if let Some(value) = display(descriptor, point) {
                        circle.dot.(format!("s{}", line.color)) cx=(coordinate(x(*load))) cy=(coordinate(y(value))) r="3.5" {
                            title { (line.label) " · " (load_label(group, *load)) " · " (human_number(value)) " " (descriptor.unit.label()) }
                        }
                    }
                }
            }
        }
    }
}

/// Runs of consecutive present values; a missing value breaks the line
/// rather than drawing it through zero.
fn segments(descriptor: &MetricDescriptor, placed: &[(f64, &MetricPoint)]) -> Vec<Vec<(f64, f64)>> {
    let mut segments = vec![Vec::new()];
    for (load, point) in placed {
        match display(descriptor, point) {
            Some(value) => {
                if let Some(current) = segments.last_mut() {
                    current.push((*load, value));
                }
            }
            None => segments.push(Vec::new()),
        }
    }
    segments
}

fn coordinate(value: f64) -> String {
    format!("{value:.1}")
}
