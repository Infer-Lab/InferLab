//! Server-rendered pages ([[ADR-0056]]). maud escapes every interpolated
//! value, so directory, workspace, and record names render as text.

use maud::{DOCTYPE, Markup, PreEscaped, html};
use std::path::Path;

const STYLE: &str = include_str!("console.css");

/// One registered workspace and whether it can be read.
pub(super) struct WorkspaceSummary<'a> {
    pub(super) root: &'a Path,
    pub(super) id: String,
    pub(super) unavailable: Option<String>,
    /// A failure of one source, such as definitions that do not load; the
    /// workspace's other views still open.
    pub(super) degraded: Option<String>,
}

pub(super) struct DirectoryEntry {
    pub(super) name: String,
    pub(super) workspace: bool,
}

pub(super) fn layout(title: &str, body: Markup) -> Markup {
    html! {
        (DOCTYPE)
        html lang="en" {
            head {
                meta charset="utf-8";
                meta name="viewport" content="width=device-width, initial-scale=1";
                meta name="referrer" content="no-referrer";
                title { (title) " · InferLab" }
                style { (PreEscaped(STYLE)) }
            }
            body {
                header.topbar {
                    a.brand href="/" { span.mark { "◆" } " InferLab" }
                    nav { a href="/" { "Workspaces" } a href="/compare" { "Compare" } }
                    // Shown by the script; notifications stay off until enabled here.
                    button #notify.quiet type="button" aria-pressed="false" hidden { "Notifications off" }
                }
                main { (body) }
                script src="/assets/htmx.min.js" {}
                script src="/assets/console.js" {}
            }
        }
    }
}

pub(super) fn workspaces(registered: &[WorkspaceSummary<'_>]) -> Markup {
    layout(
        "Workspaces",
        html! {
            section.panel {
                div.panel-head {
                    h1 { "Workspaces" }
                    a.button href="/browse" { "Add workspace" }
                }
                @if registered.is_empty() {
                    p.empty { "No workspace is registered. Add one from the directory browser." }
                } @else {
                    ul.rows {
                        @for workspace in registered {
                            li.row {
                                @match &workspace.unavailable {
                                    None => {
                                        @if let Some(failure) = &workspace.degraded {
                                            span.glyph.warning { "◆" }
                                            a.name href={ "/w/" (workspace.id) } { (workspace.root.file_name().map_or_else(|| workspace.root.display().to_string(), |name| name.to_string_lossy().into_owned())) }
                                            span.path title=(failure) { (workspace.root.display()) " · " (failure) }
                                            span.status.warning { "definitions unavailable" }
                                        } @else {
                                            span.glyph.success { "●" }
                                            a.name href={ "/w/" (workspace.id) } { (workspace.root.file_name().map_or_else(|| workspace.root.display().to_string(), |name| name.to_string_lossy().into_owned())) }
                                            span.path { (workspace.root.display()) }
                                            span.status.success { "available" }
                                        }
                                    }
                                    Some(reason) => {
                                        span.glyph.critical { "×" }
                                        span.name { (workspace.root.file_name().map_or_else(|| workspace.root.display().to_string(), |name| name.to_string_lossy().into_owned())) }
                                        span.path { (workspace.root.display()) " · " (reason) }
                                        span.status.critical { "unavailable" }
                                    }
                                }
                                form method="post" action="/workspaces/remove" {
                                    input type="hidden" name="path" value=(workspace.root.display());
                                    button.quiet type="submit" title="Unregister; files are untouched" { "Remove" }
                                }
                            }
                        }
                    }
                }
            }
        },
    )
}

pub(super) fn browse(
    directory: &Path,
    entries: &[DirectoryEntry],
    hidden: bool,
    error: Option<&str>,
) -> Markup {
    let parent = directory.parent();
    layout(
        "Add workspace",
        html! {
            section.panel {
                div.panel-head {
                    h1 { "Add workspace" }
                    a.button.quiet href="/" { "Done" }
                }
                @if let Some(error) = error {
                    p.error { (error) }
                }
                form.pathbar method="get" action="/browse" {
                    input type="text" name="path" value=(directory.display()) aria-label="Directory";
                    label.toggle {
                        input type="checkbox" name="hidden" value="1" checked[hidden] onchange="this.form.submit()";
                        " Show hidden"
                    }
                    button type="submit" { "Open" }
                }
                ul.rows {
                    @if let Some(parent) = parent {
                        li.row {
                            span.glyph { "↑" }
                            a.name href={ "/browse?path=" (encode_path(parent)) @if hidden { "&hidden=1" } } { ".." }
                        }
                    }
                    @for entry in entries {
                        @let child = directory.join(&entry.name);
                        li.row data-workspace=[entry.workspace.then_some(entry.name.as_str())] {
                            @if entry.workspace {
                                span.glyph.accent { "◆" }
                            } @else {
                                span.glyph { "▸" }
                            }
                            a.name href={ "/browse?path=" (encode_path(&child)) @if hidden { "&hidden=1" } } { (entry.name) }
                            @if entry.workspace {
                                span.status.accent { "workspace" }
                                form method="post" action="/workspaces" {
                                    input type="hidden" name="path" value=(child.display());
                                    button type="submit" { "Add" }
                                }
                            }
                        }
                    }
                }
            }
        },
    )
}

pub(super) fn problem(title: &str, message: &str) -> Markup {
    layout(
        title,
        html! {
            section.panel {
                h1 { (title) }
                p.error { (message) }
                a.button href="/" { "Back to workspaces" }
            }
        },
    )
}

/// Percent-encode a path for a query parameter.
pub(super) fn encode_path(path: &Path) -> String {
    encode(&path.to_string_lossy())
}

/// Percent-encode a value for a query parameter.
pub(super) fn encode(value: &str) -> String {
    value
        .bytes()
        .map(|byte| match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' | b'/' => {
                char::from(byte).to_string()
            }
            other => format!("%{other:02X}"),
        })
        .collect()
}
