//! A workspace's views in the browser ([[RFC-0012:C-VIEWS]]): the TUI's
//! rows, groups, status filter, record tree, and detail, rendered from the
//! shared read model. Selection, filter, and open parents live in the URL.

use super::hub::{Feed, Generation};
use super::pages::{encode, layout};
use crate::console::metrics::{RecordMetrics, human_number};
use crate::console::presentation::EntrySource;
use crate::console::records::LogPage;
use crate::console::{DisplayEntry, DisplayTone, State, StatusFilter, View};
use maud::{Markup, html};
use std::collections::BTreeSet;
use std::time::{Duration, Instant};

/// What the URL selects within one workspace view.
pub(super) struct ViewQuery {
    pub(super) view: View,
    pub(super) status: StatusFilter,
    pub(super) open: BTreeSet<String>,
    pub(super) selected: Option<String>,
}

impl ViewQuery {
    fn href(&self, base: &str, change: impl FnOnce(&mut Query)) -> String {
        let mut query = Query {
            view: self.view,
            status: self.status,
            open: self.open.clone(),
            selected: self.selected.clone(),
        };
        change(&mut query);
        let mut params = Vec::new();
        if query.status != StatusFilter::All {
            params.push(format!("status={}", status_param(query.status)));
        }
        if !query.open.is_empty() {
            params.push(format!(
                "open={}",
                query
                    .open
                    .iter()
                    .map(|key| encode(key))
                    .collect::<Vec<_>>()
                    .join(",")
            ));
        }
        if let Some(selected) = &query.selected {
            params.push(format!("selected={}", encode(selected)));
        }
        let path = format!("{base}/{}", view_path(query.view));
        if params.is_empty() {
            path
        } else {
            format!("{path}?{}", params.join("&"))
        }
    }
}

struct Query {
    view: View,
    status: StatusFilter,
    open: BTreeSet<String>,
    selected: Option<String>,
}

pub(super) fn view_path(view: View) -> &'static str {
    match view {
        View::Overview => "overview",
        View::Operations => "operations",
        View::Records => "records",
        View::Workspace => "workspace",
    }
}

pub(super) fn status_param(filter: StatusFilter) -> &'static str {
    match filter {
        StatusFilter::All => "all",
        StatusFilter::Attention => "attention",
        StatusFilter::Running => "running",
    }
}

fn tone_class(tone: DisplayTone) -> &'static str {
    match tone {
        DisplayTone::Normal => "muted",
        DisplayTone::Success => "success",
        DisplayTone::Active => "accent",
        DisplayTone::Warning => "warning",
        DisplayTone::Critical => "critical",
    }
}

fn state_class(state: State) -> &'static str {
    match state {
        State::Live => "success",
        State::Stale => "warning",
        State::Unavailable => "critical",
        State::Incompatible => "incompatible",
    }
}

/// The display-refresh indicator of one workspace ([[RFC-0010:C-REFRESH]]):
/// the cadence while generations arrive, the elapsed time once two intervals
/// pass without one.
fn refresh_indicator(generation: Option<&Generation>, interval: Duration) -> Markup {
    let Some(generation) = generation else {
        return html! { span.refresh.muted { "○ waiting for the first refresh" } };
    };
    let elapsed = Instant::now().saturating_duration_since(generation.completed);
    if elapsed < interval * 2 {
        html! { span.refresh { span.success { "● " } "live · every " (seconds(interval)) } }
    } else {
        html! { span.refresh.warning { "◆ last refresh " (seconds(elapsed)) " ago" } }
    }
}

fn seconds(duration: Duration) -> String {
    format!("{:.1}s", duration.as_secs_f64())
}

/// Which tab of a workspace page is open.
#[derive(Clone, Copy, Eq, PartialEq)]
pub(super) enum Tab {
    View(View),
    Jobs,
}

/// A workspace page: its header with the refresh indicator, its tabs, and
/// `content`. A live page re-renders from the workspace's event stream.
pub(super) fn frame(
    feed: &Feed,
    generation: Option<&Generation>,
    interval: Duration,
    tab: Tab,
    back: &str,
    live: bool,
    content: Markup,
) -> Markup {
    let base = format!("/w/{}", feed.id);
    let name = feed.name();
    let body = html! {
        div #console data-events=[live.then(|| format!("{base}/events"))] {
            header.workspace-head {
                div {
                    h1 { (name) }
                    div.meta {
                        span.path { (feed.root.display()) }
                        @if let Some(workspace) = generation.and_then(|generation| generation.snapshot.workspace.value.as_ref()) {
                            span { " · " (workspace.revision.get(..7).unwrap_or(&workspace.revision)) " · " }
                            @if workspace.dirty { span.warning { "dirty" } } @else { span.success { "clean" } }
                        }
                    }
                }
                div.head-actions {
                    (refresh_indicator(generation, interval))
                    form method="post" action={ (base) "/refresh" } {
                        input type="hidden" name="back" value=(back);
                        button.quiet type="submit" { "Refresh" }
                    }
                }
            }
            nav.tabs {
                @for view in View::ALL {
                    @let count = generation.map_or(0, |generation| top_level(&generation.presentation, EntrySource::View(view)));
                    a.tab.active[tab == Tab::View(view)] href={ (base) "/" (view_path(view)) } {
                        (view.title()) span.count { (count) }
                    }
                }
                a.tab.active[tab == Tab::Jobs] href={ (base) "/jobs" } { "Jobs" }
            }
            (content)
        }
    };
    layout(&name, body)
}

pub(super) fn workspace_page(
    feed: &Feed,
    generation: Option<&Generation>,
    interval: Duration,
    query: &ViewQuery,
    now_unix_ms: u64,
) -> Markup {
    let base = format!("/w/{}", feed.id);
    let source = EntrySource::View(query.view);
    let content = html! {
        @match generation {
            None => { (waiting()) }
            Some(generation) => {
                (body_of(generation, &feed.id, source, query, now_unix_ms))
            }
        }
    };
    frame(
        feed,
        generation,
        interval,
        Tab::View(query.view),
        &query.href(&base, |_| {}),
        true,
        content,
    )
}

pub(super) fn waiting() -> Markup {
    html! { section.panel { p.empty { "Waiting for the first complete read of this workspace…" } } }
}

/// Each visible position with the group heading that opens before it: a
/// day or section label, emitted once where the group changes.
fn headed(
    presentation: &crate::console::presentation::Presentation,
    source: EntrySource,
    visible: &[usize],
) -> Vec<(Option<String>, usize)> {
    let mut previous = None;
    visible
        .iter()
        .map(|&position| {
            let group = presentation.item(source, position).and_then(|item| {
                item.group
                    .clone()
                    .or_else(|| item.section.map(|section| section.label().to_owned()))
            });
            let heading = (group.is_some() && group != previous)
                .then(|| group.clone())
                .flatten();
            previous = group;
            (heading, position)
        })
        .collect()
}

fn top_level(
    presentation: &crate::console::presentation::Presentation,
    source: EntrySource,
) -> usize {
    presentation
        .visible(source, None, StatusFilter::All, |_| false)
        .len()
}

fn body_of(
    generation: &Generation,
    workspace_id: &str,
    source: EntrySource,
    query: &ViewQuery,
    now_unix_ms: u64,
) -> Markup {
    let base = &format!("/w/{workspace_id}");
    let presentation = &generation.presentation;
    let visible = presentation.visible(source, None, query.status, |key| query.open.contains(key));
    let selected = query
        .selected
        .as_deref()
        .and_then(|key| {
            visible.iter().copied().find(|position| {
                presentation
                    .entry(source, *position)
                    .is_some_and(|entry| entry.key == key)
            })
        })
        .or_else(|| visible.first().copied());
    let filters = matches!(query.view, View::Overview | View::Records);
    html! {
        @if filters {
            div.filters {
                @for filter in StatusFilter::ALL {
                    @let count = presentation.visible(source, None, filter, |_| false).len();
                    a.chip.active[filter == query.status]
                        href=(query.href(base, |next| { next.status = filter; next.selected = None; })) {
                        (filter.label()) " " span.count.critical[filter == StatusFilter::Attention && count > 0] { (count) }
                    }
                }
            }
        }
        div.split {
            section.panel.list {
                @if visible.is_empty() {
                    p.empty { "Nothing here yet." }
                }
                @for (heading, position) in headed(presentation, source, &visible) {
                    @if let Some(heading) = heading {
                        h2.group { (heading) }
                    }
                    @if let (Some(entry), Some(item)) = (presentation.entry(source, position), presentation.item(source, position)) {
                        @let children = presentation.children(source, position);
                        (row(entry, item.child_of.is_some(), &children, query, base, Some(position) == selected, now_unix_ms))
                    }
                }
            }
            section.panel.detail {
                @if let Some(entry) = selected.and_then(|position| presentation.entry(source, position)) {
                    (detail(entry, presentation.record_metrics(&entry.key), workspace_id, now_unix_ms))
                } @else {
                    p.empty { "Select an object to see its detail." }
                }
            }
        }
    }
}

fn row(
    entry: &DisplayEntry,
    child: bool,
    children: &[usize],
    query: &ViewQuery,
    base: &str,
    selected: bool,
    now_unix_ms: u64,
) -> Markup {
    let (kind, name) = entry.kind_and_name();
    let key = entry.key.clone();
    let open = query.open.contains(&entry.key);
    let age = entry
        .moment_unix_ms
        .map(|moment| crate::console::views::compact_age(now_unix_ms.saturating_sub(moment)))
        .unwrap_or_default();
    html! {
        div.row.selected[selected].child[child] {
            @if !children.is_empty() {
                a.fold href=(query.href(base, |next| {
                    if open { next.open.remove(&key); } else { next.open.insert(key.clone()); }
                })) title=(if open { "Collapse" } else { "Expand" }) { (if open { "▾" } else { "▸" }) }
            } @else if child {
                span.fold.muted { "└" }
            } @else {
                span.fold {}
            }
            span.glyph.(tone_class(entry.tone)) { (entry.tone.glyph()) }
            span.kind { (kind) }
            span.label {
                a.name href=(query.href(base, |next| next.selected = Some(entry.key.clone()))) title=(name) { (name) }
                span.summary {
                    @if entry.state != State::Live {
                        span.(state_class(entry.state)) { "refresh " (entry.state.label()) " · " }
                    }
                    (entry.summary)
                }
            }
            span.lifecycle.(tone_class(entry.tone)) { (entry.lifecycle.as_deref().unwrap_or("")) }
            span.badge { (entry.authority.badge()) }
            span.age { (age) }
        }
    }
}

fn detail(
    entry: &DisplayEntry,
    metrics: Option<&RecordMetrics>,
    workspace_id: &str,
    now_unix_ms: u64,
) -> Markup {
    let base = format!("/w/{workspace_id}");
    let (kind, name) = entry.kind_and_name();
    let comparable = kind == "bench" && metrics.is_some_and(|metrics| metrics.case_count > 0);
    html! {
        div.detail-head {
            span.glyph.(tone_class(entry.tone)) { (entry.tone.glyph()) }
            span.kind { (kind) }
            h2 { (name) }
            @if comparable {
                a.button.quiet.compare-link href=(super::compare::href(workspace_id, &entry.key)) { "Compare" }
            }
        }
        div.pills {
            @if let Some(lifecycle) = &entry.lifecycle {
                span.pill.(tone_class(entry.tone)) { (lifecycle.to_uppercase()) }
            }
            span.pill.quiet { (entry.authority.badge()) " " (entry.authority.label()) }
            @if entry.state != State::Live {
                span.pill.(state_class(entry.state)) { "refresh " (entry.state.label()) }
            }
            @if !entry.summary.is_empty() { span.muted { (entry.summary) } }
        }
        @for section in &entry.details {
            @if section.title == "METRICS" && metrics.is_some() {
                @if let Some(metrics) = metrics { (metric_table(metrics)) }
            } @else {
                h3 { (section.title) }
                @if !section.rows.is_empty() {
                    dl.facts {
                        @for (label, value) in &section.rows {
                            dt { (label) }
                            dd { (value.render(now_unix_ms)) }
                        }
                    }
                }
                @for body in &section.body {
                    pre.body { (body) }
                }
            }
        }
        @if !entry.log_refs.is_empty() {
            h3 { "LOGS" }
            ul.logs {
                @for reference in &entry.log_refs {
                    li { a href={ (base) "/log?ref=" (encode(reference)) } { (reference) } }
                }
            }
        }
    }
}

/// A workload record's recorded case metrics: a row per metric, a column
/// per case.
fn metric_table(metrics: &RecordMetrics) -> Markup {
    let cases = metrics
        .points
        .first()
        .map(|points| {
            points
                .iter()
                .map(|point| point.label.clone())
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let mut previous = None;
    let rows = metrics
        .catalog
        .iter()
        .zip(&metrics.points)
        .map(|(descriptor, points)| {
            let opens_family = previous != Some(descriptor.family);
            previous = Some(descriptor.family);
            (opens_family, descriptor, points)
        })
        .collect::<Vec<_>>();
    html! {
        h3 { "METRICS" span.muted { " " (cases.len()) " cases" } }
        table.metrics {
            thead { tr { th {} @for case in &cases { th { (case) } } th {} } }
            tbody {
                @for (opens_family, descriptor, points) in rows {
                    @if opens_family {
                        tr.family { td colspan=(cases.len() + 2) { (descriptor.family.label()) } }
                    }
                    tr {
                        th { (descriptor.label) }
                        @for point in points {
                            td { (point.value.map_or_else(|| "—".to_owned(), |value| human_number(descriptor.unit.display_value(value)))) }
                        }
                        td.unit { (descriptor.unit.label()) }
                    }
                }
            }
        }
    }
}

/// One page of a log, counted from its end. `page_href` names another page.
pub(super) fn log_page(
    title: &str,
    back: (&str, &str),
    page_href: impl Fn(u64) -> String,
    page: u64,
    log: &LogPage,
) -> Markup {
    layout(
        title,
        html! {
            section.panel {
                div.panel-head {
                    h1 { "Log" }
                    a.button.quiet href=(back.0) { (back.1) }
                }
                p.path { (title) " · page " (page + 1) " from the end" }
                div.pager {
                    @if log.earlier {
                        a.button.quiet href=(page_href(page + 1)) { "Earlier" }
                    }
                    @if page > 0 {
                        a.button.quiet href=(page_href(page - 1)) { "Later" }
                    }
                }
                pre.log { (log.text) }
            }
        },
    )
}
