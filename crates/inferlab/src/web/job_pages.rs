//! The Jobs tab ([[RFC-0012:C-ACTIONS]]): typed action forms, the dry-run
//! preview, and each job's state, vector, and logs.

use super::actions::{Action, ActionKind, Preview};
use super::hub::{Feed, Generation};
use super::jobs::{Job, JobState};
use super::views::{Tab, frame};
use crate::console::records::LogPage;
use crate::console::views::{compact_age, timestamp_with_age};
use maud::{Markup, html};
use std::time::Duration;

/// What the Jobs tab shows on its detail side: a new action's form, or a job.
pub(super) enum JobsDetail<'a> {
    Form(ActionKind),
    Job(Option<&'a Job>, &'a [(&'static str, LogPage)]),
}

pub(super) struct JobsPage<'a> {
    pub(super) feed: &'a Feed,
    pub(super) generation: Option<&'a Generation>,
    pub(super) interval: Duration,
    pub(super) now_unix_ms: u64,
}

impl JobsPage<'_> {
    fn base(&self) -> String {
        format!("/w/{}", self.feed.id)
    }

    pub(super) fn render(
        &self,
        jobs: &[Job],
        selected: Option<&str>,
        detail: &JobsDetail<'_>,
    ) -> Markup {
        let base = self.base();
        let form_open = matches!(detail, JobsDetail::Form(_));
        let open_kind = match detail {
            JobsDetail::Form(kind) => Some(*kind),
            JobsDetail::Job(..) => None,
        };
        let back = selected.map_or_else(
            || format!("{base}/jobs"),
            |job| format!("{base}/jobs?job={job}"),
        );
        let content = html! {
            div.filters {
                span.muted.new-label { "New" }
                @for kind in ActionKind::ALL {
                    a.chip.active[open_kind == Some(kind)] href={ (base) "/jobs?new=" (kind.key()) } { (kind.title()) }
                }
            }
            div.split {
                section.panel.list {
                    @if jobs.is_empty() {
                        p.empty { "No jobs yet. Choose an action above to launch one." }
                    }
                    @for job in jobs {
                        (self.row(job, selected == Some(job.id.as_str())))
                    }
                }
                section.panel.detail {
                    @match detail {
                        JobsDetail::Form(kind) => (self.form(*kind)),
                        JobsDetail::Job(Some(job), logs) => (self.job(job, logs)),
                        JobsDetail::Job(None, _) => p.empty { "Select a job to see its vector, state, and logs." },
                    }
                }
            }
        };
        // A page with a form open does not re-render under the operator.
        frame(
            self.feed,
            self.generation,
            self.interval,
            Tab::Jobs,
            &back,
            !form_open,
            content,
        )
    }

    fn row(&self, job: &Job, selected: bool) -> Markup {
        let (glyph, tone) = state_tone(&job.state);
        let target = job
            .spec
            .as_ref()
            .and_then(|spec| spec.argv.last())
            .map_or("", String::as_str);
        let action = job.spec.as_ref().map_or("job", |spec| spec.action.as_str());
        let age = job
            .spec
            .as_ref()
            .map(|spec| compact_age(self.now_unix_ms.saturating_sub(spec.started_unix_ms)))
            .unwrap_or_default();
        html! {
            div.row.job-row.selected[selected] {
                span.glyph.(tone) { (glyph) }
                span.kind { (action) }
                span.label {
                    a.name href={ (self.base()) "/jobs?job=" (job.id) } title=(job.id) { (target) }
                }
                span.lifecycle.(tone) { (job.state.label()) }
                span.age { (age) }
            }
        }
    }

    fn job(&self, job: &Job, logs: &[(&'static str, LogPage)]) -> Markup {
        let base = self.base();
        let (glyph, tone) = state_tone(&job.state);
        html! {
            div.detail-head {
                span.glyph.(tone) { (glyph) }
                span.kind { (job.spec.as_ref().map_or("job", |spec| spec.action.as_str())) }
                h2 { (job.id) }
                @if matches!(job.state, JobState::Running) {
                    form.compare-link method="post" action={ (base) "/jobs/" (job.id) "/cancel" } {
                        button.quiet type="submit" title="Deliver the interrupt Ctrl+C delivers; cleanup stays with the CLI" { "Interrupt" }
                    }
                }
            }
            div.pills {
                span.pill.(tone) { (job.state.label().to_uppercase()) }
            }
            h3 { "JOB" }
            dl.facts {
                @if let Some(spec) = &job.spec {
                    dt { "Started" } dd { (timestamp_with_age(self.now_unix_ms, Some(spec.started_unix_ms))) }
                }
                @if let JobState::Exited(exit) = &job.state {
                    dt { "Ended" } dd { (timestamp_with_age(self.now_unix_ms, Some(exit.ended_unix_ms))) }
                }
                @if let Some(producer) = &job.producer {
                    dt { "Process" } dd { (producer.host) " · pid " (producer.pid) }
                }
                dt { "Directory" } dd.path { (job.directory.display()) }
            }
            @if let Some(spec) = &job.spec {
                h3 { "VECTOR" }
                (argv(&spec.argv))
            }
            @for (stream, log) in logs {
                h3 { (stream.to_uppercase()) " "
                    a.muted href={ (base) "/jobs/" (job.id) "/log?stream=" (stream) } { "page through" }
                }
                @if log.text.is_empty() {
                    p.empty { "Empty." }
                } @else {
                    pre.log.tail { (log.text) }
                }
            }
        }
    }

    fn form(&self, kind: ActionKind) -> Markup {
        let base = self.base();
        let snapshot = self.generation.map(|generation| &generation.snapshot);
        let definitions = |wanted: &str| {
            snapshot
                .map(|snapshot| {
                    snapshot
                        .definitions
                        .iter()
                        .filter(|definition| definition.kind == wanted)
                        .map(|definition| definition.id.clone())
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default()
        };
        let running_servers = snapshot
            .map(|snapshot| {
                snapshot
                    .records
                    .iter()
                    .filter(|record| {
                        record.kind == "server" && record.status.as_deref() == Some("running")
                    })
                    .filter_map(|record| record.id.clone())
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        let target = match kind {
            ActionKind::ServeStart => Some(("Server", definitions("server"))),
            ActionKind::RecipeRun => Some(("Recipe", definitions("recipe"))),
            ActionKind::Bench => Some(("Bench", definitions("bench"))),
            ActionKind::ServeStop => Some(("Running server", running_servers.clone())),
            ActionKind::Note => None,
        };
        let endpoint = if kind.previewed() {
            "preview"
        } else {
            "launch"
        };
        let blocked = target
            .as_ref()
            .is_some_and(|(_, options)| options.is_empty())
            || (kind == ActionKind::Bench && running_servers.is_empty());
        html! {
            div.detail-head { h2 { (kind.title()) } }
            p.muted {
                @if kind.previewed() {
                    "The dry run runs first; the launch is offered only when it succeeds."
                } @else {
                    "Launches directly as a job."
                }
            }
            @if self.generation.is_none() {
                p.empty { "Waiting for the first complete read of this workspace…" }
            } @else {
                form.action-form method="post" action={ (base) "/actions/" (endpoint) } {
                    input type="hidden" name="action" value=(kind.key());
                    @if let Some((label, options)) = &target {
                        (select(label, "target", options))
                    }
                    @match kind {
                        ActionKind::ServeStart | ActionKind::RecipeRun => {
                            (text_field("Case", "case", "the server's default case"))
                            (text_field("Placement", "placement", "the local bindings' default"))
                        }
                        ActionKind::Bench => { (select("Server record", "serve", &running_servers)) }
                        ActionKind::Note => {
                            label.field {
                                span { "Text" }
                                textarea name="text" rows="5" required {}
                            }
                            (text_field("Topic", "topic", "the common stream"))
                            (text_field("Records", "records", "record ids or last, comma-separated"))
                        }
                        ActionKind::ServeStop => {}
                    }
                    div.form-actions {
                        button type="submit" disabled[blocked] {
                            (if kind.previewed() { "Preview" } else { "Launch" })
                        }
                    }
                }
            }
        }
    }

    /// The dry run's vector, status, and output, and the launch it allows.
    pub(super) fn preview(&self, action: &Action, preview: &Preview) -> Markup {
        let base = self.base();
        let content = html! {
            section.panel.preview {
                div.detail-head {
                    h2 { "Preview · " (action.kind.title()) }
                    a.button.quiet.compare-link href={ (base) "/jobs?new=" (action.kind.key()) } { "Edit" }
                }
                h3 { "DRY RUN" }
                (argv(&action.preview_argv()))
                div.pills {
                    @if preview.success {
                        span.pill.success { "SUCCEEDED" }
                    } @else {
                        span.pill.critical { "FAILED" }
                    }
                    span.muted { (preview.status) }
                }
                @if !preview.stdout.is_empty() {
                    h3 { "STDOUT" }
                    pre.log.tail { (preview.stdout) }
                }
                @if !preview.stderr.is_empty() {
                    h3 { "STDERR" }
                    pre.log.tail { (preview.stderr) }
                }
                @if preview.success {
                    h3 { "LAUNCH" }
                    p.muted { "This vector runs as a detached job. It resolves again when it starts; the preview does not bind it." }
                    (argv(action.launch_argv()))
                    form method="post" action={ (base) "/actions/launch" } {
                        (hidden_fields(action))
                        div.form-actions { button type="submit" { "Launch" } }
                    }
                } @else {
                    p.error { "The dry run failed, so nothing can be launched. Edit the inputs and preview again." }
                }
            }
        };
        frame(
            self.feed,
            self.generation,
            self.interval,
            Tab::Jobs,
            &format!("{base}/jobs"),
            false,
            content,
        )
    }
}

fn state_tone(state: &JobState) -> (&'static str, &'static str) {
    match state {
        JobState::Running => ("●", "accent"),
        JobState::Exited(exit) if exit.exit_code == Some(0) => ("✓", "success"),
        JobState::Exited(exit) if exit.exit_code.is_some() => ("×", "critical"),
        JobState::Exited(_) => ("◆", "warning"),
        JobState::Unknown | JobState::Elsewhere => ("?", "muted"),
    }
}

fn argv(argv: &[String]) -> Markup {
    html! {
        ol.argv { @for argument in argv { li { code { (argument) } } } }
    }
}

fn select(label: &str, name: &str, options: &[String]) -> Markup {
    html! {
        label.field {
            span { (label) }
            @if options.is_empty() {
                span.empty { "None in this workspace." }
            } @else {
                select name=(name) required {
                    @for option in options { option value=(option) { (option) } }
                }
            }
        }
    }
}

fn text_field(label: &str, name: &str, default: &str) -> Markup {
    html! {
        label.field {
            span { (label) }
            input type="text" name=(name) placeholder={ "optional · " (default) };
        }
    }
}

/// The previewed fields, resubmitted so the launch rebuilds the same vector.
fn hidden_fields(action: &Action) -> Markup {
    let form = &action.form;
    let fields = [
        ("action", &form.action),
        ("target", &form.target),
        ("case", &form.case),
        ("placement", &form.placement),
        ("serve", &form.serve),
        ("text", &form.text),
        ("topic", &form.topic),
        ("records", &form.records),
    ];
    html! {
        @for (name, value) in fields {
            @if !value.is_empty() {
                input type="hidden" name=(name) value=(value);
            }
        }
    }
}
