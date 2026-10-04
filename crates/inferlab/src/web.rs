//! The workspace web console ([[RFC-0012]], [[ADR-0056]]): a local,
//! token-protected service over a registry of workspaces. It writes no
//! records and performs no workflow effect itself.

mod access;
mod actions;
mod compare;
mod hub;
mod job_pages;
mod jobs;
mod pages;
mod registry;
mod views;

use crate::InferlabError;
use access::Access;
use axum::Router;
use axum::extract::{Form, Query, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Redirect, Response};
use axum::routing::{get, post};
use registry::{RegisterError, Registry};
use serde::Deserialize;
use std::io::Write;
use std::net::{IpAddr, SocketAddr};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

pub(crate) struct WebOptions {
    pub(crate) bind: IpAddr,
    pub(crate) port: u16,
    /// The display-refresh cadence of every workspace ([[RFC-0010:C-REFRESH]]).
    pub(crate) refresh_interval: Duration,
    /// The workspace that discovery or `--workspace` selected, if any.
    pub(crate) workspace: Option<PathBuf>,
}

struct Console {
    registry: Mutex<Registry>,
    hub: hub::Hub,
    start: PathBuf,
    /// The running service's executable, which every action invokes.
    executable: Option<PathBuf>,
}

impl Console {
    /// Keep one refresh worker per registered workspace.
    fn track(&self) {
        if let Ok(registry) = self.registry.lock() {
            self.hub.track(registry.roots());
        }
    }
}

type Shared = Arc<Console>;

/// The hidden supervisor of one launched job.
pub(crate) fn supervise_job(directory: &Path) -> Result<(), InferlabError> {
    jobs::supervise(directory)
}

pub(crate) fn run(options: WebOptions) -> Result<(), InferlabError> {
    inferlab_runtime::interrupt::prepare()
        .map_err(|source| InferlabError::WebInterrupt { source })?;
    let mut registry = Registry::load()?;
    if let Some(workspace) = &options.workspace {
        match registry.register(workspace) {
            Ok(()) => {}
            Err(RegisterError::Persist(error)) => return Err(error),
            // A selected workspace whose manifest disappeared is reported but
            // does not prevent observing the others.
            Err(RegisterError::Rejected(reason)) => eprintln!("warning: {reason}"),
        }
    }
    // The printed URL opens the workspace this invocation selected, when it is
    // registered ([[RFC-0012:C-WORKSPACES]]).
    let focus = options
        .workspace
        .as_deref()
        .and_then(|workspace| workspace.canonicalize().ok())
        .filter(|root| registry.roots().contains(root))
        .map(|root| format!("/w/{}/overview", hub::workspace_id(&root)));
    let access = Arc::new(
        Access::generate().map_err(|source| InferlabError::WebStart {
            step: "generate its access token",
            source,
        })?,
    );
    // Browsing starts beside the selected workspace, where its siblings live.
    let start = options
        .workspace
        .as_deref()
        .and_then(Path::parent)
        .map(Path::to_path_buf)
        .or_else(|| std::env::current_dir().ok())
        .unwrap_or_else(|| PathBuf::from("/"));
    if options.refresh_interval.is_zero() {
        return Err(InferlabError::InvalidConfig {
            message: "--refresh-interval must be greater than zero".to_owned(),
        });
    }
    let console = Arc::new(Console {
        registry: Mutex::new(registry),
        hub: hub::Hub::new(options.refresh_interval),
        start,
        executable: std::env::current_exe().ok(),
    });
    console.track();
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map_err(|source| InferlabError::WebStart {
            step: "start its runtime",
            source,
        })?;
    runtime.block_on(serve(options.bind, options.port, focus, access, console))
}

async fn serve(
    bind: IpAddr,
    port: u16,
    focus: Option<String>,
    access: Arc<Access>,
    console: Shared,
) -> Result<(), InferlabError> {
    let address = SocketAddr::new(bind, port);
    let listener = tokio::net::TcpListener::bind(address)
        .await
        .map_err(|source| InferlabError::WebListener {
            address: address.to_string(),
            source,
        })?;
    let bound = listener
        .local_addr()
        .map_err(|source| InferlabError::WebListener {
            address: address.to_string(),
            source,
        })?;
    if !bind.is_loopback() {
        eprintln!(
            "warning: the web console listens on {bound}; its traffic is not encrypted, so reach it through an SSH tunnel or a trusted network"
        );
    }
    let app = Router::new()
        .route("/", get(workspaces))
        .route("/browse", get(browse))
        .route("/workspaces", post(register))
        .route("/workspaces/remove", post(unregister))
        .route("/compare", get(compare))
        .route("/watch", get(watch))
        .route("/w/{id}", get(workspace_home))
        .route("/w/{id}/refresh", post(refresh))
        .route("/w/{id}/events", get(events))
        .route("/w/{id}/log", get(log))
        .route("/w/{id}/jobs", get(jobs_view))
        .route("/w/{id}/jobs/{job}/cancel", post(cancel_job))
        .route("/w/{id}/jobs/{job}/log", get(job_log))
        .route("/w/{id}/actions/preview", post(preview_action))
        .route("/w/{id}/actions/launch", post(launch_action))
        .route("/w/{id}/{view}", get(workspace_view))
        .route("/assets/htmx.min.js", get(htmx))
        .route("/assets/console.js", get(script))
        .with_state(console)
        .layer(axum::middleware::from_fn_with_state(
            access.clone(),
            access::guard,
        ));
    // The access URL is the only stdout line, printed once connections are
    // accepted ([[RFC-0012:C-SCOPE]]).
    let host = match bound.ip() {
        _ if bind.is_unspecified() => rustix::system::uname()
            .nodename()
            .to_string_lossy()
            .into_owned(),
        IpAddr::V6(address) => format!("[{address}]"),
        IpAddr::V4(address) => address.to_string(),
    };
    let mut stdout = std::io::stdout();
    writeln!(
        stdout,
        "http://{host}:{}{}?token={}",
        bound.port(),
        focus.as_deref().unwrap_or("/"),
        access.token()
    )
    .and_then(|()| stdout.flush())
    .map_err(|source| InferlabError::WriteOutput { source })?;
    axum::serve(listener, app)
        .with_graceful_shutdown(interrupted())
        .await
        .map_err(|source| InferlabError::WebServe { source })
}

async fn interrupted() {
    while !inferlab_runtime::interrupt::received() {
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}

async fn workspaces(State(console): State<Shared>) -> Response {
    // Checking roots touches the filesystem, which may be slow.
    let checked = tokio::task::spawn_blocking({
        let console = console.clone();
        move || {
            let roots = console
                .registry
                .lock()
                .map(|registry| registry.roots().to_vec())
                .ok()?;
            Some(
                roots
                    .into_iter()
                    .map(|root| {
                        let manifest = registry::is_workspace_root(&root);
                        (root, manifest)
                    })
                    .collect::<Vec<_>>(),
            )
        }
    })
    .await;
    let Ok(Some(checked)) = checked else {
        return internal("the workspace registry is unavailable");
    };
    let summaries = checked
        .iter()
        .map(|(root, manifest)| {
            let id = hub::workspace_id(root);
            // A root whose definitions do not load still opens; the index
            // shows the failure ([[RFC-0012:C-WORKSPACES]]).
            let failure = console.hub.feed(&id).and_then(|feed| {
                feed.latest.borrow().as_ref().and_then(|generation| {
                    generation
                        .snapshot
                        .definitions_error
                        .as_ref()
                        .map(|error| format!("definitions do not load: {error}"))
                })
            });
            pages::WorkspaceSummary {
                root,
                id,
                unavailable: (!*manifest)
                    .then(|| format!("no {} at this root", crate::workspace::WORKSPACE_FILE)),
                degraded: failure,
            }
        })
        .collect::<Vec<_>>();
    pages::workspaces(&summaries).into_response()
}

#[derive(Deserialize)]
struct BrowseQuery {
    path: Option<PathBuf>,
    /// Whether hidden (dot-prefixed) directories are listed.
    #[serde(default)]
    hidden: Option<u8>,
}

async fn browse(State(console): State<Shared>, Query(query): Query<BrowseQuery>) -> Response {
    let directory = query.path.unwrap_or_else(|| console.start.clone());
    let hidden = query.hidden.is_some_and(|hidden| hidden != 0);
    // A listing on a slow filesystem must not hold up other requests.
    let listed = tokio::task::spawn_blocking({
        let directory = directory.clone();
        move || list_directories(&directory, hidden)
    })
    .await;
    match listed {
        Ok(Ok(entries)) => pages::browse(&directory, &entries, hidden, None).into_response(),
        Ok(Err(error)) => (
            StatusCode::BAD_REQUEST,
            pages::browse(
                &directory,
                &[],
                hidden,
                Some(&format!("{}: {error}", directory.display())),
            ),
        )
            .into_response(),
        Err(_) => internal("the directory listing did not finish"),
    }
}

/// The names of the directories under `directory`, workspace roots first.
/// Hidden directories are listed only on request; file contents are never
/// read.
fn list_directories(directory: &Path, hidden: bool) -> std::io::Result<Vec<pages::DirectoryEntry>> {
    let mut entries = std::fs::read_dir(directory)?
        .filter_map(Result::ok)
        .filter(|entry| hidden || !entry.file_name().to_string_lossy().starts_with('.'))
        .filter(|entry| entry.path().is_dir())
        .map(|entry| pages::DirectoryEntry {
            workspace: registry::is_workspace_root(&entry.path()),
            name: entry.file_name().to_string_lossy().into_owned(),
        })
        .collect::<Vec<_>>();
    entries.sort_by(|left, right| {
        right
            .workspace
            .cmp(&left.workspace)
            .then_with(|| left.name.cmp(&right.name))
    });
    Ok(entries)
}

#[derive(Deserialize)]
struct PathForm {
    path: PathBuf,
}

async fn register(State(console): State<Shared>, Form(form): Form<PathForm>) -> Response {
    let result = tokio::task::spawn_blocking({
        let console = console.clone();
        move || {
            let result = console
                .registry
                .lock()
                .map(|mut registry| registry.register(&form.path))
                .ok();
            console.track();
            result
        }
    })
    .await;
    match result {
        Ok(Some(Ok(()))) => Redirect::to("/").into_response(),
        Ok(Some(Err(RegisterError::Rejected(reason)))) => (
            StatusCode::BAD_REQUEST,
            pages::problem("Not a workspace", &reason),
        )
            .into_response(),
        Ok(Some(Err(RegisterError::Persist(error)))) => internal(&error.to_string()),
        Ok(None) | Err(_) => internal("the workspace registry is unavailable"),
    }
}

async fn unregister(State(console): State<Shared>, Form(form): Form<PathForm>) -> Response {
    let result = tokio::task::spawn_blocking({
        let console = console.clone();
        move || {
            let result = console
                .registry
                .lock()
                .map(|mut registry| registry.unregister(&form.path))
                .ok();
            console.track();
            result
        }
    })
    .await;
    match result {
        Ok(Some(Ok(()))) => Redirect::to("/").into_response(),
        Ok(Some(Err(error))) => internal(&error.to_string()),
        Ok(None) | Err(_) => internal("the workspace registry is unavailable"),
    }
}

fn missing_workspace() -> Response {
    (
        StatusCode::NOT_FOUND,
        pages::problem("Unknown workspace", "This workspace is not registered."),
    )
        .into_response()
}

async fn compare(
    State(console): State<Shared>,
    axum::extract::RawQuery(query): axum::extract::RawQuery,
) -> Response {
    let selection = compare::Selection::parse(query.as_deref().unwrap_or_default());
    let feeds = console.hub.feeds();
    let labels = hub::labels(
        &feeds
            .iter()
            .map(|feed| feed.root.as_path())
            .collect::<Vec<_>>(),
    );
    let sources = feeds
        .into_iter()
        .zip(labels)
        .map(|(feed, label)| {
            let generation = feed.latest.borrow().clone();
            compare::Source {
                feed,
                label,
                generation,
            }
        })
        .collect::<Vec<_>>();
    let now = crate::record::now_unix_ms().unwrap_or_default();
    compare::page(&sources, &selection, now).into_response()
}

/// What browser notifications compare between polls ([[RFC-0012:C-VIEWS]]):
/// each workspace's running and recently ended jobs, and its running
/// servers' observed liveness.
#[derive(serde::Serialize)]
struct Watch {
    jobs: Vec<WatchedJob>,
    servers: Vec<WatchedServer>,
}

#[derive(serde::Serialize)]
struct WatchedJob {
    workspace: String,
    id: String,
    action: String,
    running: bool,
    state: String,
    href: String,
}

#[derive(serde::Serialize)]
struct WatchedServer {
    workspace: String,
    id: String,
    alive: Option<bool>,
    href: String,
}

/// How long an ended job stays in the watch, so a poll sees its transition.
const WATCH_ENDED_MS: u64 = 10 * 60 * 1000;

async fn watch(State(console): State<Shared>) -> Response {
    let feeds = console.hub.feeds();
    let labels = hub::labels(
        &feeds
            .iter()
            .map(|feed| feed.root.as_path())
            .collect::<Vec<_>>(),
    );
    let now = crate::record::now_unix_ms().unwrap_or_default();
    let mut servers = Vec::new();
    for (feed, label) in feeds.iter().zip(&labels) {
        let Some(generation) = feed.latest.borrow().clone() else {
            continue;
        };
        // Recipe-owned servers count too: Overview keeps them visible.
        let running = generation
            .snapshot
            .records
            .iter()
            .chain(&generation.snapshot.child_records)
            .filter(|record| {
                record.kind == "server" && record.status.as_deref() == Some("running")
            });
        servers.extend(running.filter_map(|record| {
            let id = record.id.clone()?;
            Some(WatchedServer {
                workspace: label.clone(),
                href: format!("/w/{}/overview?selected={}", feed.id, pages::encode(&id)),
                alive: record
                    .process_observation
                    .as_ref()
                    .and_then(|observation| observation.value),
                id,
            })
        }));
    }
    let read = tokio::task::spawn_blocking(move || {
        feeds
            .iter()
            .zip(labels)
            .flat_map(|(feed, label)| {
                jobs::list(&feed.root)
                    .into_iter()
                    .filter(|job| match &job.state {
                        jobs::JobState::Running | jobs::JobState::Unknown => true,
                        // Another host's job cannot change state here.
                        jobs::JobState::Elsewhere => false,
                        jobs::JobState::Exited(exit) => {
                            now.saturating_sub(exit.ended_unix_ms) < WATCH_ENDED_MS
                        }
                    })
                    .map(|job| WatchedJob {
                        workspace: label.clone(),
                        action: job
                            .spec
                            .as_ref()
                            .map_or_else(|| "job".to_owned(), |spec| spec.action.clone()),
                        running: matches!(job.state, jobs::JobState::Running),
                        state: job.state.label(),
                        href: format!("/w/{}/jobs?job={}", feed.id, job.id),
                        id: job.id,
                    })
                    .collect::<Vec<_>>()
            })
            .collect()
    })
    .await;
    match read {
        Ok(jobs) => axum::Json(Watch { jobs, servers }).into_response(),
        Err(_) => internal("the job read did not finish"),
    }
}

async fn workspace_home(axum::extract::Path(id): axum::extract::Path<String>) -> Response {
    Redirect::to(&format!("/w/{id}/overview")).into_response()
}

#[derive(Deserialize)]
struct ViewParams {
    status: Option<String>,
    open: Option<String>,
    selected: Option<String>,
}

async fn workspace_view(
    State(console): State<Shared>,
    axum::extract::Path((id, view)): axum::extract::Path<(String, String)>,
    Query(params): Query<ViewParams>,
) -> Response {
    let Some(feed) = console.hub.feed(&id) else {
        return missing_workspace();
    };
    let Some(view) = crate::console::View::ALL
        .into_iter()
        .find(|candidate| views::view_path(*candidate) == view)
    else {
        return (StatusCode::NOT_FOUND, pages::problem("Unknown view", &view)).into_response();
    };
    let status = crate::console::StatusFilter::ALL
        .into_iter()
        .find(|filter| params.status.as_deref() == Some(views::status_param(*filter)))
        .unwrap_or_default();
    let query = views::ViewQuery {
        view,
        status,
        open: params
            .open
            .iter()
            .flat_map(|open| open.split(','))
            .filter(|key| !key.is_empty())
            .map(str::to_owned)
            .collect(),
        selected: params.selected,
    };
    let generation = feed.latest.borrow().clone();
    let now = crate::record::now_unix_ms().unwrap_or_default();
    views::workspace_page(
        &feed,
        generation.as_deref(),
        console.hub.interval(),
        &query,
        now,
    )
    .into_response()
}

#[derive(Deserialize)]
struct RefreshForm {
    back: String,
}

async fn refresh(
    State(console): State<Shared>,
    axum::extract::Path(id): axum::extract::Path<String>,
    Form(form): Form<RefreshForm>,
) -> Response {
    let Some(feed) = console.hub.feed(&id) else {
        return missing_workspace();
    };
    feed.refresh_now();
    // Only a page of this workspace is a valid return address.
    let back = if form.back.starts_with(&format!("/w/{id}/")) {
        form.back
    } else {
        format!("/w/{id}/overview")
    };
    Redirect::to(&back).into_response()
}

#[derive(Deserialize)]
struct LogParams {
    #[serde(rename = "ref")]
    reference: String,
    page: Option<u64>,
}

async fn log(
    State(console): State<Shared>,
    axum::extract::Path(id): axum::extract::Path<String>,
    Query(params): Query<LogParams>,
) -> Response {
    let Some(feed) = console.hub.feed(&id) else {
        return missing_workspace();
    };
    // Only logs a record or an operation of this workspace explicitly
    // references are shown ([[RFC-0010:C-READ-MODEL]]).
    let referenced = feed.latest.borrow().as_ref().is_some_and(|generation| {
        let snapshot = &generation.snapshot;
        snapshot
            .records
            .iter()
            .chain(&snapshot.child_records)
            .any(|record| record.log_refs.contains(&params.reference))
            || snapshot
                .operations
                .iter()
                .any(|operation| operation.log_ref.as_ref() == Some(&params.reference))
    });
    if !referenced {
        return (
            StatusCode::NOT_FOUND,
            pages::problem(
                "Unknown log",
                "No record or operation of this workspace references this log.",
            ),
        )
            .into_response();
    }
    let page = params.page.unwrap_or(0);
    let root = feed.root.clone();
    let reference = params.reference.clone();
    let read = tokio::task::spawn_blocking(move || {
        crate::console::records::read_log_page(&root, &reference, page)
    })
    .await;
    match read {
        Ok(log) => {
            let base = format!("/w/{}", feed.id);
            let reference = &params.reference;
            views::log_page(
                reference,
                (&format!("{base}/records"), "Back to records"),
                |page| format!("{base}/log?ref={}&page={page}", pages::encode(reference)),
                page,
                &log,
            )
            .into_response()
        }
        Err(_) => internal("the log read did not finish"),
    }
}

/// Server-sent events: one `generation` event per published generation, and
/// a `tick` when none arrives for two intervals so the page can show the
/// refresh as overdue.
async fn events(
    State(console): State<Shared>,
    axum::extract::Path(id): axum::extract::Path<String>,
) -> Response {
    use axum::response::sse::{Event, KeepAlive, Sse};
    let Some(feed) = console.hub.feed(&id) else {
        return missing_workspace();
    };
    let interval = console.hub.interval();
    let (sender, receiver) =
        tokio::sync::mpsc::channel::<Result<Event, std::convert::Infallible>>(4);
    let mut latest = feed.latest.clone();
    tokio::spawn(async move {
        let number = |latest: &tokio::sync::watch::Receiver<Option<Arc<hub::Generation>>>| {
            latest
                .borrow()
                .as_ref()
                .map_or(0, |generation| generation.number)
        };
        let first = Event::default()
            .event("generation")
            .data(number(&latest).to_string());
        if sender.send(Ok(first)).await.is_err() {
            return;
        }
        loop {
            let event = tokio::select! {
                changed = latest.changed() => {
                    if changed.is_err() {
                        return;
                    }
                    Event::default().event("generation").data(number(&latest).to_string())
                }
                () = tokio::time::sleep(interval * 2) => Event::default().event("tick").data(""),
            };
            if sender.send(Ok(event)).await.is_err() {
                return;
            }
        }
    });
    Sse::new(tokio_stream::wrappers::ReceiverStream::new(receiver))
        .keep_alive(KeepAlive::default())
        .into_response()
}

async fn htmx() -> Response {
    (
        [(
            axum::http::header::CONTENT_TYPE,
            "text/javascript; charset=utf-8",
        )],
        include_str!("web/assets/htmx.min.js"),
    )
        .into_response()
}

async fn script() -> Response {
    (
        [(
            axum::http::header::CONTENT_TYPE,
            "text/javascript; charset=utf-8",
        )],
        include_str!("web/assets/console.js"),
    )
        .into_response()
}

#[derive(Deserialize)]
struct JobsParams {
    job: Option<String>,
    new: Option<String>,
}

async fn jobs_view(
    State(console): State<Shared>,
    axum::extract::Path(id): axum::extract::Path<String>,
    Query(params): Query<JobsParams>,
) -> Response {
    let Some(feed) = console.hub.feed(&id) else {
        return missing_workspace();
    };
    let form = params.new.as_deref().and_then(actions::ActionKind::parse);
    let root = feed.root.clone();
    let wanted = params.job.clone();
    let read = tokio::task::spawn_blocking(move || {
        let jobs = jobs::list(&root);
        let selected = match &wanted {
            Some(wanted) => jobs.iter().position(|job| &job.id == wanted),
            None if form.is_none() => (!jobs.is_empty()).then_some(0),
            None => None,
        };
        let logs = selected.and_then(|position| jobs.get(position)).map(|job| {
            [("stdout", jobs::STDOUT_LOG), ("stderr", jobs::STDERR_LOG)].map(|(stream, file)| {
                let path = job.directory.join(file).display().to_string();
                (
                    stream,
                    crate::console::records::read_log_page(&root, &path, 0),
                )
            })
        });
        (jobs, selected, logs)
    })
    .await;
    let Ok((jobs, selected, logs)) = read else {
        return internal("the job read did not finish");
    };
    let generation = feed.latest.borrow().clone();
    let page = job_pages::JobsPage {
        feed: &feed,
        generation: generation.as_deref(),
        interval: console.hub.interval(),
        now_unix_ms: crate::record::now_unix_ms().unwrap_or_default(),
    };
    let selected_job = selected.and_then(|position| jobs.get(position));
    let logs = logs.map(Vec::from).unwrap_or_default();
    let detail = match form {
        Some(kind) => job_pages::JobsDetail::Form(kind),
        None => job_pages::JobsDetail::Job(selected_job, &logs),
    };
    page.render(&jobs, selected_job.map(|job| job.id.as_str()), &detail)
        .into_response()
}

/// Why an action was not accepted.
enum Refusal {
    NoExecutable,
    Invalid(String),
}

impl IntoResponse for Refusal {
    fn into_response(self) -> Response {
        match self {
            Self::NoExecutable => internal("the service could not determine its own executable"),
            Self::Invalid(message) => (
                StatusCode::UNPROCESSABLE_ENTITY,
                pages::problem("Action not accepted", &message),
            )
                .into_response(),
        }
    }
}

/// Build the action from its typed fields.
fn action_of(
    console: &Console,
    feed: &hub::Feed,
    form: actions::ActionForm,
) -> Result<actions::Action, Refusal> {
    let executable = console.executable.as_ref().ok_or(Refusal::NoExecutable)?;
    actions::Action::from_form(form, executable, &feed.root).map_err(Refusal::Invalid)
}

async fn preview_action(
    State(console): State<Shared>,
    axum::extract::Path(id): axum::extract::Path<String>,
    Form(form): Form<actions::ActionForm>,
) -> Response {
    let Some(feed) = console.hub.feed(&id) else {
        return missing_workspace();
    };
    let action = match action_of(&console, &feed, form) {
        Ok(action) => action,
        Err(refusal) => return refusal.into_response(),
    };
    if !action.kind.previewed() {
        return (
            StatusCode::UNPROCESSABLE_ENTITY,
            pages::problem("No preview", "This action launches without a dry run."),
        )
            .into_response();
    }
    let preview = actions::preview(&action.preview_argv(), &feed.root).await;
    let generation = feed.latest.borrow().clone();
    job_pages::JobsPage {
        feed: &feed,
        generation: generation.as_deref(),
        interval: console.hub.interval(),
        now_unix_ms: crate::record::now_unix_ms().unwrap_or_default(),
    }
    .preview(&action, &preview)
    .into_response()
}

async fn launch_action(
    State(console): State<Shared>,
    axum::extract::Path(id): axum::extract::Path<String>,
    Form(form): Form<actions::ActionForm>,
) -> Response {
    let Some(feed) = console.hub.feed(&id) else {
        return missing_workspace();
    };
    let action = match action_of(&console, &feed, form) {
        Ok(action) => action,
        Err(refusal) => return refusal.into_response(),
    };
    let root = feed.root.clone();
    let key = action.kind.key();
    let argv = action.launch_argv().to_vec();
    let started = crate::record::now_unix_ms().unwrap_or_default();
    match tokio::task::spawn_blocking(move || jobs::launch(&root, key, argv, started)).await {
        Ok(Ok(job)) => Redirect::to(&format!("/w/{id}/jobs?job={job}")).into_response(),
        Ok(Err(message)) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            pages::problem("The job could not start", &message),
        )
            .into_response(),
        Err(_) => internal("the launch did not finish"),
    }
}

async fn cancel_job(
    State(console): State<Shared>,
    axum::extract::Path((id, job)): axum::extract::Path<(String, String)>,
) -> Response {
    let Some(feed) = console.hub.feed(&id) else {
        return missing_workspace();
    };
    let root = feed.root.clone();
    let wanted = job.clone();
    let cancelled = tokio::task::spawn_blocking(move || {
        jobs::read(&root, &wanted)
            .ok_or_else(|| "no such job".to_owned())
            .and_then(|job| jobs::cancel(&job))
    })
    .await;
    match cancelled {
        Ok(Ok(())) => Redirect::to(&format!("/w/{id}/jobs?job={job}")).into_response(),
        Ok(Err(message)) => (
            StatusCode::CONFLICT,
            pages::problem("Not interrupted", &message),
        )
            .into_response(),
        Err(_) => internal("the interrupt did not finish"),
    }
}

#[derive(Deserialize)]
struct JobLogParams {
    stream: String,
    page: Option<u64>,
}

async fn job_log(
    State(console): State<Shared>,
    axum::extract::Path((id, job)): axum::extract::Path<(String, String)>,
    Query(params): Query<JobLogParams>,
) -> Response {
    let Some(feed) = console.hub.feed(&id) else {
        return missing_workspace();
    };
    let file = match params.stream.as_str() {
        "stdout" => jobs::STDOUT_LOG,
        "stderr" => jobs::STDERR_LOG,
        _ => {
            return (
                StatusCode::NOT_FOUND,
                pages::problem("Unknown log", &params.stream),
            )
                .into_response();
        }
    };
    let page = params.page.unwrap_or(0);
    let root = feed.root.clone();
    let wanted = job.clone();
    let read = tokio::task::spawn_blocking(move || {
        jobs::read(&root, &wanted).map(|job| {
            let path = job.directory.join(file).display().to_string();
            crate::console::records::read_log_page(&root, &path, page)
        })
    })
    .await;
    match read {
        Ok(Some(log)) => {
            let jobs_href = format!("/w/{id}/jobs?job={job}");
            let stream = &params.stream;
            views::log_page(
                &format!("{job} · {stream}"),
                (&jobs_href, "Back to job"),
                |page| format!("/w/{id}/jobs/{job}/log?stream={stream}&page={page}"),
                page,
                &log,
            )
            .into_response()
        }
        Ok(None) => (
            StatusCode::NOT_FOUND,
            pages::problem("Unknown job", "This job is not in the workspace."),
        )
            .into_response(),
        Err(_) => internal("the log read did not finish"),
    }
}

fn internal(message: &str) -> Response {
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        pages::problem("Console error", message),
    )
        .into_response()
}
