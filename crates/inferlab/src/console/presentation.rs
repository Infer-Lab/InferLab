use super::views::{definition_display, record_display, unavailable_entry, workspace_display};
use super::{
    Authority, DisplayEntry, EntryKind, JournalView, OverviewSection, OverviewSummary, Snapshot,
    State, metrics, search,
};
use super::{StatusFilter, View};
use std::collections::HashMap;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum EntrySource {
    View(View),
    Global,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(crate) struct EntryIdentity {
    kind: EntryKind,
    key: String,
}

impl EntryIdentity {
    pub(crate) fn new(kind: EntryKind, key: String) -> Self {
        Self { kind, key }
    }

    pub(crate) fn of(entry: &DisplayEntry) -> Self {
        Self {
            kind: entry.kind,
            key: entry.key.clone(),
        }
    }
}

#[derive(Clone)]
pub(crate) struct ViewItem {
    entry: usize,
    pub(crate) section: Option<OverviewSection>,
    pub(crate) group: Option<String>,
    /// The position of the parent this item is an explicit child reference
    /// of, in the same view.
    pub(crate) child_of: Option<usize>,
}

#[derive(Default)]
struct ViewIndex {
    items: Vec<ViewItem>,
    positions: HashMap<EntryIdentity, usize>,
}

impl ViewIndex {
    fn push(&mut self, entry: usize, display: &DisplayEntry) {
        self.push_with(entry, display, None, None);
    }

    fn push_with(
        &mut self,
        entry: usize,
        display: &DisplayEntry,
        section: Option<OverviewSection>,
        group: Option<String>,
    ) {
        let position = self.items.len();
        self.items.push(ViewItem {
            entry,
            section,
            group,
            child_of: None,
        });
        self.positions
            .entry(EntryIdentity::of(display))
            .or_insert(position);
    }

    fn len(&self) -> usize {
        self.items.len()
    }

    /// Place an explicit child reference beneath the item at `parent`.
    fn push_child(&mut self, entry: usize, display: &DisplayEntry, parent: usize) {
        let group = self.items.get(parent).and_then(|item| item.group.clone());
        self.push_with(entry, display, None, group);
        if let Some(item) = self.items.last_mut() {
            item.child_of = Some(parent);
        }
    }

    fn item(&self, position: usize) -> Option<&ViewItem> {
        self.items.get(position)
    }

    fn position(&self, identity: &EntryIdentity) -> Option<usize> {
        self.positions.get(identity).copied()
    }
}

pub(crate) struct Presentation {
    entries: Vec<DisplayEntry>,
    overview: ViewIndex,
    operations: ViewIndex,
    records: ViewIndex,
    workspace: ViewIndex,
    global: ViewIndex,
    record_metrics: HashMap<String, metrics::RecordMetrics>,
    overview_summary: OverviewSummary,
}

impl Presentation {
    pub(crate) fn from_snapshot(snapshot: &Snapshot) -> Self {
        let mut presentation = Self {
            entries: Vec::new(),
            overview: ViewIndex::default(),
            operations: ViewIndex::default(),
            records: ViewIndex::default(),
            workspace: ViewIndex::default(),
            global: ViewIndex::default(),
            record_metrics: snapshot
                .records
                .iter()
                .chain(&snapshot.child_records)
                .filter_map(metrics::presentation)
                .map(|record| (record.record_key.clone(), record))
                .collect(),
            overview_summary: OverviewSummary::default(),
        };
        let operation_error = snapshot.operations_error.as_deref().map(|reason| {
            push_display(
                &mut presentation,
                unavailable_entry(
                    EntryKind::Operation,
                    "operations-unavailable",
                    Authority::Ephemeral,
                    "operation observations",
                    reason,
                ),
            )
        });
        let record_error = snapshot.records_error.as_deref().map(|reason| {
            push_display(
                &mut presentation,
                unavailable_entry(
                    EntryKind::Record,
                    "records-unavailable",
                    Authority::Recorded,
                    "records",
                    reason,
                ),
            )
        });

        let operation_entries = snapshot
            .operations
            .iter()
            .map(|operation| push_display(&mut presentation, operation.display()))
            .collect::<Vec<_>>();
        let journal_by_record = journal_by_record(&snapshot.journal);
        let record_entries = snapshot
            .records
            .iter()
            .map(|record| {
                let notes = record
                    .id
                    .as_deref()
                    .and_then(|id| journal_by_record.get(id))
                    .map(Vec::as_slice);
                push_display(&mut presentation, record_display(record, notes))
            })
            .collect::<Vec<_>>();
        let child_entries = snapshot
            .child_records
            .iter()
            .map(|record| push_display(&mut presentation, record.display()))
            .collect::<Vec<_>>();
        let workspace_entry = push_display(&mut presentation, workspace_display(snapshot));

        let mut definition_entries = snapshot
            .definitions
            .iter()
            .map(|definition| {
                let group = definition.kind.to_uppercase();
                let display = definition_display(snapshot, definition);
                let title = display.title.clone();
                let entry = push_display(&mut presentation, display);
                (group, title, entry)
            })
            .collect::<Vec<_>>();
        definition_entries
            .sort_by(|left, right| left.0.cmp(&right.0).then_with(|| left.1.cmp(&right.1)));
        let journal_count = snapshot.journal.len();
        let journal_entries = snapshot
            .journal
            .iter()
            .enumerate()
            .map(|(index, entry)| {
                // The collector exposes the append-only journal newest-first. Recovering the
                // source ordinal keeps an existing entry's identity stable when a new line is
                // prepended to this presentation order, even when timestamps are equal.
                let source_ordinal = journal_count.saturating_sub(index + 1);
                push_display(&mut presentation, entry.display(source_ordinal))
            })
            .collect::<Vec<_>>();
        let definition_error = if snapshot.definitions.is_empty() {
            snapshot.definitions_error.as_deref()
        } else {
            None
        }
        .map(|reason| {
            push_display(
                &mut presentation,
                unavailable_entry(
                    EntryKind::Definition,
                    "definitions-unavailable",
                    Authority::Declared,
                    "workspace definitions",
                    reason,
                ),
            )
        });
        let journal_error = if snapshot.journal.is_empty() {
            snapshot.journal_error.as_deref()
        } else {
            None
        }
        .map(|reason| {
            push_display(
                &mut presentation,
                unavailable_entry(
                    EntryKind::Journal,
                    "journal-unavailable",
                    Authority::Recorded,
                    "scratchpad journal",
                    reason,
                ),
            )
        });

        let operations_order = if snapshot.operations.is_empty() {
            operation_error.into_iter().collect::<Vec<_>>()
        } else {
            operation_entries.clone()
        };
        let records_order = if snapshot.records.is_empty() {
            record_error.into_iter().collect::<Vec<_>>()
        } else {
            record_entries.clone()
        };
        for entry in &operations_order {
            presentation
                .operations
                .push(*entry, &presentation.entries[*entry]);
        }
        // Records is a day-grouped timeline; each parent carries its explicit
        // child references, available or not, beneath it.
        let children_by_id = snapshot
            .child_records
            .iter()
            .filter_map(|record| record.id.as_deref().map(|id| (id, record)))
            .collect::<HashMap<_, _>>();
        let parents = snapshot
            .records
            .iter()
            .map(Some)
            .chain(std::iter::repeat(None));
        for (entry, record) in records_order.iter().zip(parents) {
            let day = presentation.entries[*entry]
                .moment_unix_ms
                .and_then(super::views::full_day_label);
            let parent = presentation.records.len();
            presentation
                .records
                .push_with(*entry, &presentation.entries[*entry], None, day);
            for child in record.map_or(&[][..], |record| record.child_refs.as_slice()) {
                let display = children_by_id.get(child.as_str()).map_or_else(
                    || {
                        unavailable_entry(
                            EntryKind::Record,
                            child,
                            Authority::Recorded,
                            child,
                            "child record cannot be read",
                        )
                    },
                    |record| {
                        let mut display = record_display(record, None);
                        relative_to_parent(&mut display, &presentation.entries[*entry].title);
                        display
                    },
                );
                let child_entry = push_display(&mut presentation, display);
                presentation.records.push_child(
                    child_entry,
                    &presentation.entries[child_entry],
                    parent,
                );
            }
        }
        let mut workspace_order = Vec::new();
        for (group, _, entry) in definition_entries {
            presentation.workspace.push_with(
                entry,
                &presentation.entries[entry],
                None,
                Some(group),
            );
            workspace_order.push(entry);
        }
        for entry in journal_entries {
            presentation.workspace.push_with(
                entry,
                &presentation.entries[entry],
                None,
                Some("SCRATCHPAD".to_owned()),
            );
            workspace_order.push(entry);
        }
        if let Some(entry) = definition_error {
            presentation
                .workspace
                .push(entry, &presentation.entries[entry]);
            workspace_order.push(entry);
        }
        if let Some(entry) = journal_error {
            presentation
                .workspace
                .push(entry, &presentation.entries[entry]);
            workspace_order.push(entry);
        }
        for entry in operations_order
            .iter()
            .chain(&records_order)
            .chain(&workspace_order)
        {
            presentation
                .global
                .push(*entry, &presentation.entries[*entry]);
        }

        let mut attention = Vec::new();
        let mut active_operations = Vec::new();
        let mut active_records = Vec::new();
        let mut child_server_attention = Vec::new();
        let mut active_child_records = Vec::new();
        let mut recent = Vec::new();
        let mut summary = OverviewSummary {
            ephemeral_active: 0,
            ephemeral_attention: 0,
            recorded_active: 0,
            recorded_attention: 0,
            recorded_recent: 0,
        };
        if let Some(entry) = operation_error {
            summary.ephemeral_attention += 1;
            attention.push(entry);
        }
        if let Some(entry) = record_error {
            summary.recorded_attention += 1;
            attention.push(entry);
        }
        for (operation, entry) in snapshot.operations.iter().zip(&operation_entries) {
            if operation.state == State::Live {
                summary.ephemeral_active += 1;
                active_operations.push(*entry);
            } else {
                summary.ephemeral_attention += 1;
                attention.push(*entry);
            }
        }
        for (record, entry) in snapshot.records.iter().zip(&record_entries) {
            if record.needs_attention() {
                summary.recorded_attention += 1;
                attention.push(*entry);
            } else if record.is_active() {
                summary.recorded_active += 1;
                active_records.push(*entry);
            } else {
                summary.recorded_recent += 1;
                recent.push(*entry);
            }
        }
        for (server, entry) in snapshot.child_records.iter().zip(&child_entries) {
            if server.kind != "server" {
                continue;
            }
            if server.needs_attention() {
                child_server_attention.push(*entry);
            } else if server.is_active() {
                active_child_records.push(*entry);
            }
        }
        let mut overview = Vec::new();
        extend_overview_section(
            &mut overview,
            &attention,
            &child_server_attention,
            5,
            OverviewSection::Attention,
        );
        overview.extend(
            active_operations
                .into_iter()
                .take(5)
                .map(|entry| (entry, OverviewSection::Active)),
        );
        extend_overview_section(
            &mut overview,
            &active_records,
            &active_child_records,
            5,
            OverviewSection::Active,
        );
        overview.extend(
            recent
                .into_iter()
                .take(10)
                .map(|entry| (entry, OverviewSection::Recent)),
        );
        overview.push((workspace_entry, OverviewSection::Workspace));
        for (entry, section) in overview {
            // RECENT groups its records by calendar day.
            let group = (section == OverviewSection::Recent)
                .then(|| presentation.entries[entry].moment_unix_ms)
                .flatten()
                .and_then(super::views::day_label)
                .map(|day| format!("{} · {day}", section.label()));
            presentation.overview.push_with(
                entry,
                &presentation.entries[entry],
                Some(section),
                group,
            );
        }
        presentation.overview_summary = summary;
        presentation
    }

    pub(crate) fn entry(&self, source: EntrySource, position: usize) -> Option<&DisplayEntry> {
        let item = self.index(source).item(position)?;
        self.entries.get(item.entry)
    }

    /// The visible positions of a view, the one rule both surfaces apply
    /// ([[RFC-0010:C-NAVIGATION]]): top-level items matching the query and the
    /// status filter, each followed by its children when it is expanded.
    pub(crate) fn visible(
        &self,
        source: EntrySource,
        query: Option<&str>,
        filter: StatusFilter,
        expanded: impl Fn(&str) -> bool,
    ) -> Vec<usize> {
        let positions = match query {
            Some(query) => self.matching_positions(source, query),
            None => (0..self.len(source)).collect(),
        };
        positions
            .into_iter()
            .filter(|position| {
                self.item(source, *position)
                    .is_none_or(|item| item.child_of.is_none())
                    && self
                        .entry(source, *position)
                        .is_some_and(|entry| filter.admits(entry))
            })
            .flat_map(|position| {
                let open = self
                    .entry(source, position)
                    .is_some_and(|entry| expanded(&entry.key));
                std::iter::once(position).chain(if open {
                    self.children(source, position)
                } else {
                    Vec::new()
                })
            })
            .collect()
    }

    /// The explicit child references placed beneath the item at `position`.
    pub(crate) fn children(&self, source: EntrySource, position: usize) -> Vec<usize> {
        let index = self.index(source);
        index
            .items
            .iter()
            .enumerate()
            .skip(position + 1)
            .take_while(|(_, item)| item.child_of == Some(position))
            .map(|(child, _)| child)
            .collect()
    }

    pub(crate) fn item(&self, source: EntrySource, position: usize) -> Option<&ViewItem> {
        self.index(source).item(position)
    }

    pub(crate) fn len(&self, source: EntrySource) -> usize {
        self.index(source).len()
    }

    pub(crate) fn identity(&self, source: EntrySource, position: usize) -> Option<EntryIdentity> {
        self.entry(source, position).map(EntryIdentity::of)
    }

    pub(crate) fn position(&self, source: EntrySource, identity: &EntryIdentity) -> Option<usize> {
        self.index(source).position(identity)
    }

    pub(crate) fn matching_positions(&self, source: EntrySource, query: &str) -> Vec<usize> {
        let query = query.to_lowercase();
        let mut matches = (0..self.len(source))
            .filter_map(|position| {
                let entry = self.entry(source, position)?;
                search::match_rank_normalized_fields(&query, &entry.search_fields)
                    .map(|rank| (position, rank))
            })
            .collect::<Vec<_>>();
        matches.sort_by_key(|(position, rank)| (*rank, *position));
        matches.into_iter().map(|(position, _)| position).collect()
    }

    pub(crate) fn overview_summary(&self) -> OverviewSummary {
        self.overview_summary
    }

    pub(crate) fn record_metrics(&self, record_key: &str) -> Option<&metrics::RecordMetrics> {
        self.record_metrics.get(record_key)
    }

    fn normalize_search_fields(entry: &mut DisplayEntry) {
        for field in &mut entry.search_fields {
            *field = field.to_lowercase();
        }
    }

    fn index(&self, source: EntrySource) -> &ViewIndex {
        match source {
            EntrySource::View(View::Overview) => &self.overview,
            EntrySource::View(View::Operations) => &self.operations,
            EntrySource::View(View::Records) => &self.records,
            EntrySource::View(View::Workspace) => &self.workspace,
            EntrySource::Global => &self.global,
        }
    }
}

fn journal_by_record(journal: &[JournalView]) -> HashMap<&str, Vec<String>> {
    let mut by_record = HashMap::<&str, Vec<String>>::new();
    for entry in journal {
        let note = format!("{}  {}  {}", entry.timestamp, entry.author, entry.text);
        for record in &entry.records {
            by_record
                .entry(record.as_str())
                .or_default()
                .push(note.clone());
        }
    }
    by_record
}

/// A child reference reads relative to its parent: a recipe's bench
/// `…-1890795-bench-000-serving` becomes `bench-000-serving` beneath it.
fn relative_to_parent(child: &mut DisplayEntry, parent_title: &str) {
    let Some((_, parent)) = parent_title.split_once(" / ") else {
        return;
    };
    let Some((kind, name)) = child.title.split_once(" / ") else {
        return;
    };
    if let Some(relative) = name
        .find(&format!("{parent}-"))
        .and_then(|start| name.get(start + parent.len() + 1..))
        .filter(|relative| !relative.is_empty())
    {
        child.title = format!("{kind} / {relative}");
    }
}

fn push_display(presentation: &mut Presentation, mut display: DisplayEntry) -> usize {
    Presentation::normalize_search_fields(&mut display);
    let entry = presentation.entries.len();
    presentation.entries.push(display);
    entry
}

fn extend_overview_section(
    entries: &mut Vec<(usize, OverviewSection)>,
    primary: &[usize],
    child_records: &[usize],
    limit: usize,
    section: OverviewSection,
) {
    let reserved = usize::from(!child_records.is_empty());
    let start = entries.len();
    entries.extend(
        primary
            .iter()
            .copied()
            .take(limit.saturating_sub(reserved))
            .map(|entry| (entry, section)),
    );
    let remaining = limit.saturating_sub(entries.len().saturating_sub(start));
    entries.extend(
        child_records
            .iter()
            .copied()
            .take(remaining)
            .map(|entry| (entry, section)),
    );
}
