//! `inferlab agent` — install, update, uninstall, and diagnose the operator
//! plugin package for supported agent runtimes
//! ([[RFC-0008:C-AGENT-PLUGIN]], rationale in [[ADR-0007]]). Install
//! defaults to the plugin package embedded in this binary at compile time,
//! at the same version as the binary; `--from-checkout` overrides the
//! source with an explicit local checkout or unpacked release tarball.
//! Distribution tooling only: this module reads no workspace, bindings, or
//! records, and the native CLI orchestration lives in the
//! `agent-plugin-installer` crate.

pub(crate) use agent_plugin_installer::AgentSelector;
use agent_plugin_installer::{
    AgentPluginError, AgentPluginOperation, AgentRuntime, BatchFailure, BatchResult,
    BatchRuntimeOutcome, BatchStatus, DEFAULT_COMMAND_TIMEOUT, DoctorStatus, FailurePolicy,
    InstallRequest, PluginRef, UninstallRequest, check_operation, doctor_many,
    install as install_plugin, uninstall_many,
};
use flate2::read::GzDecoder;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs;
use std::path::{Path, PathBuf};
use tar::Archive;

const PLUGIN: PluginRef<'static> = PluginRef {
    selector: "inferlab@inferlab",
    name: "inferlab",
};
const MARKETPLACE: &str = "inferlab";
const INFERLAB_VERSION: &str = env!("CARGO_PKG_VERSION");

/// The plugin package this binary carries, packed reproducibly by
/// `build.rs` from the canonical repo-root package (`LICENSE`,
/// `.claude-plugin/`, `.agents/`, `plugins/inferlab/`), or from its generated
/// crate staging projection. Installed
/// by default; `--from-checkout` overrides the source entirely and never
/// touches this payload ([[RFC-0008:C-AGENT-PLUGIN]]).
const EMBEDDED_PLUGIN_TAR_GZ: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/inferlab-plugin.tar.gz"));

#[derive(Debug, Serialize)]
pub(crate) struct AgentReport {
    pub rows: Vec<AgentRow>,
}

impl AgentReport {
    /// The first failed row's message, if any — the caller emits the report
    /// and then still fails loudly ([[RFC-0008:C-AGENT-PLUGIN]]).
    pub(crate) fn failure(&self) -> Option<String> {
        self.rows
            .iter()
            .find(|row| row.status == "failed")
            .map(|row| {
                format!(
                    "{} {} failed: {}",
                    row.agent,
                    row.operation,
                    row.message.as_deref().unwrap_or("unknown error")
                )
            })
    }
}

#[derive(Debug, Serialize)]
pub(crate) struct AgentRow {
    pub agent: &'static str,
    pub operation: &'static str,
    pub status: &'static str,
    pub cli: &'static str,
    pub commands: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
}

pub(crate) fn doctor(selector: AgentSelector) -> AgentReport {
    let rows = doctor_many(selector)
        .into_iter()
        .map(|mut outcome| {
            let marketplace_failure = if outcome.status == DoctorStatus::Ready {
                match inspect_marketplace(outcome.runtime) {
                    Ok(inspection) => {
                        outcome.commands.push(inspection.command);
                        marketplace_state_failure(outcome.runtime, &inspection.state)
                    }
                    Err(failure) => {
                        if let Some(command) = failure.command {
                            outcome.commands.push(command);
                        }
                        Some(failure.message)
                    }
                }
            } else {
                None
            };
            AgentRow {
                agent: outcome.runtime.id(),
                operation: "doctor",
                status: match (&outcome.status, &marketplace_failure) {
                    (DoctorStatus::Ready, None) => "ready",
                    (DoctorStatus::Missing, _) => "missing",
                    (DoctorStatus::Ready | DoctorStatus::Failed, Some(_) | None) => "failed",
                },
                cli: outcome.runtime.cli(),
                commands: outcome.commands,
                message: marketplace_failure.or(outcome.message),
            }
        })
        .collect();
    AgentReport { rows }
}

#[derive(Debug, Eq, PartialEq)]
enum MarketplaceState {
    Absent,
    Local(PathBuf),
    Other,
}

struct MarketplaceInspection {
    command: String,
    state: MarketplaceState,
}

struct MarketplaceCommandFailure {
    command: Option<String>,
    message: String,
}

struct MarketplacePreparationFailure {
    commands: Vec<String>,
    message: String,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct CodexMarketplaceList {
    marketplaces: Vec<CodexMarketplace>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct CodexMarketplace {
    name: String,
    marketplace_source: Option<CodexMarketplaceSource>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct CodexMarketplaceSource {
    source_type: String,
    source: PathBuf,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct ClaudeMarketplace {
    name: String,
    source: String,
    path: Option<PathBuf>,
}

fn inspect_marketplace(
    runtime: AgentRuntime,
) -> Result<MarketplaceInspection, MarketplaceCommandFailure> {
    let args = ["plugin", "marketplace", "list", "--json"]
        .map(str::to_owned)
        .to_vec();
    let (command, stdout) = run_marketplace_command(runtime, &args, "marketplace inspection")?;
    let state = match runtime {
        AgentRuntime::Codex => codex_marketplace_state(&stdout),
        AgentRuntime::Claude => claude_marketplace_state(&stdout),
    }
    .map_err(|message| MarketplaceCommandFailure {
        command: Some(command.clone()),
        message,
    })?;
    Ok(MarketplaceInspection { command, state })
}

fn run_marketplace_command(
    runtime: AgentRuntime,
    args: &[String],
    operation: &str,
) -> Result<(String, Vec<u8>), MarketplaceCommandFailure> {
    let argv = std::iter::once(runtime.cli().to_owned())
        .chain(args.iter().cloned())
        .collect::<Vec<_>>();
    let command = render_agent_command(&argv);
    let bound = inferlab_runtime::operation_bound::OperationBound::finite(DEFAULT_COMMAND_TIMEOUT);
    let outcome = inferlab_runtime::container::run_with_bound(&argv, &[], None, None, &bound, None);
    match outcome {
        Ok(inferlab_runtime::container::BoundedWait::Exited { status, stdout, .. })
            if status.success() =>
        {
            Ok((command, stdout))
        }
        Ok(inferlab_runtime::container::BoundedWait::Exited {
            status,
            stdout,
            stderr,
            ..
        }) => {
            let detail = if stderr.is_empty() { stdout } else { stderr };
            Err(MarketplaceCommandFailure {
                command: Some(command),
                message: format!(
                    "{} {operation} exited with {}: {}",
                    runtime.id(),
                    status,
                    String::from_utf8_lossy(&detail).trim()
                ),
            })
        }
        Ok(inferlab_runtime::container::BoundedWait::Expired { .. }) => {
            Err(MarketplaceCommandFailure {
                command: Some(command),
                message: format!(
                    "{} {operation} timed out after {} seconds",
                    runtime.id(),
                    DEFAULT_COMMAND_TIMEOUT.as_secs()
                ),
            })
        }
        Ok(inferlab_runtime::container::BoundedWait::Interrupted { .. }) => {
            Err(MarketplaceCommandFailure {
                command: Some(command),
                message: format!("{} {operation} was interrupted", runtime.id()),
            })
        }
        Err(inferlab_runtime::container::BoundedError::Launch(error)) => {
            Err(MarketplaceCommandFailure {
                command: None,
                message: format!("{} {operation} could not start: {error}", runtime.id()),
            })
        }
        Err(
            inferlab_runtime::container::BoundedError::Stdin(error)
            | inferlab_runtime::container::BoundedError::Wait(error),
        ) => Err(MarketplaceCommandFailure {
            command: Some(command),
            message: format!("{} {operation} failed: {error}", runtime.id()),
        }),
        Err(inferlab_runtime::container::BoundedError::WaitCleanup { source, .. }) => {
            Err(MarketplaceCommandFailure {
                command: Some(command),
                message: format!("{} {operation} failed: {source}", runtime.id()),
            })
        }
    }
}

fn codex_marketplace_state(stdout: &[u8]) -> Result<MarketplaceState, String> {
    let marketplaces: CodexMarketplaceList = match serde_json::from_slice(stdout) {
        Ok(marketplaces) => marketplaces,
        Err(error) => {
            return Err(format!(
                "codex marketplace inspection returned invalid JSON: {error}"
            ));
        }
    };
    let Some(marketplace) = marketplaces
        .marketplaces
        .into_iter()
        .find(|marketplace| marketplace.name == MARKETPLACE)
    else {
        return Ok(MarketplaceState::Absent);
    };
    let Some(source) = marketplace.marketplace_source else {
        return Err("codex InferLab marketplace did not report its source".to_owned());
    };
    if source.source_type == "local" {
        Ok(MarketplaceState::Local(source.source))
    } else {
        Ok(MarketplaceState::Other)
    }
}

fn claude_marketplace_state(stdout: &[u8]) -> Result<MarketplaceState, String> {
    let marketplaces: Vec<ClaudeMarketplace> = match serde_json::from_slice(stdout) {
        Ok(marketplaces) => marketplaces,
        Err(error) => {
            return Err(format!(
                "claude marketplace inspection returned invalid JSON: {error}"
            ));
        }
    };
    let Some(marketplace) = marketplaces
        .into_iter()
        .find(|marketplace| marketplace.name == MARKETPLACE)
    else {
        return Ok(MarketplaceState::Absent);
    };
    if marketplace.source != "directory" {
        return Ok(MarketplaceState::Other);
    }
    let Some(path) = marketplace.path else {
        return Err("claude InferLab marketplace did not report its directory path".to_owned());
    };
    Ok(MarketplaceState::Local(path))
}

fn marketplace_state_failure(runtime: AgentRuntime, state: &MarketplaceState) -> Option<String> {
    let MarketplaceState::Local(path) = state else {
        return None;
    };
    (!path.is_dir()).then(|| {
        format!(
            "{} InferLab marketplace source {} is not an available directory",
            runtime.id(),
            path.display()
        )
    })
}

fn render_agent_command(argv: &[String]) -> String {
    argv.iter()
        .map(|arg| {
            if arg.chars().all(|ch| {
                ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_' | '.' | '/' | ':' | '@' | '=')
            }) {
                arg.clone()
            } else {
                format!("'{}'", arg.replace('\'', "'\\''"))
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// Installs the plugin package. `checkout` overrides the source with a
/// local checkout or unpacked release tarball, operating on it identically
/// to before; when omitted, the package embedded in this binary is
/// materialized under InferLab's versioned data directory and that durable
/// directory takes the checkout's place
/// ([[RFC-0008:C-AGENT-PLUGIN]]).
pub(crate) fn install(selector: AgentSelector, checkout: Option<&Path>) -> AgentReport {
    let runtimes = selector.runtimes();

    let source = match checkout {
        Some(dir) => dir.to_path_buf(),
        None => match materialize_embedded_package(runtimes) {
            Ok(path) => path,
            Err(message) => {
                return AgentReport {
                    rows: runtimes
                        .iter()
                        .copied()
                        .map(|runtime| failed_gate_row(runtime, "install", message.clone()))
                        .collect(),
                };
            }
        },
    };
    let source = source.as_path();

    if let Some(report) = package_gate(runtimes, source, "install") {
        return report;
    }

    let source = match source.canonicalize() {
        Ok(path) => path,
        Err(error) => {
            let message = format!("cannot canonicalize checkout {}: {error}", source.display());
            return AgentReport {
                rows: runtimes
                    .iter()
                    .copied()
                    .map(|runtime| failed_gate_row(runtime, "install", message.clone()))
                    .collect(),
            };
        }
    };

    install_from_source(selector, &source)
}

fn install_from_source(selector: AgentSelector, source: &Path) -> AgentReport {
    let preflights = selector
        .runtimes()
        .iter()
        .copied()
        .map(|runtime| check_operation(runtime, AgentPluginOperation::Install))
        .collect::<Vec<_>>();
    if preflights
        .iter()
        .any(|outcome| outcome.status != DoctorStatus::Ready)
    {
        return preflight_failure(preflights);
    }

    let rows = preflights
        .into_iter()
        .map(|preflight| install_runtime_from_source(preflight, source))
        .collect::<Vec<_>>();
    AgentReport { rows }
}

fn preflight_failure(preflights: Vec<agent_plugin_installer::DoctorOutcome>) -> AgentReport {
    let rows = preflights
        .into_iter()
        .map(|outcome| {
            let ready = outcome.status == DoctorStatus::Ready;
            let message = if ready {
                "mutations not attempted: a preceding gate failed".to_owned()
            } else {
                format!(
                    "{} CLI ({}) is not ready: {}",
                    outcome.runtime.id(),
                    outcome.runtime.cli(),
                    outcome
                        .message
                        .unwrap_or_else(|| "runtime is not ready".to_owned())
                )
            };
            make_row(
                outcome.runtime,
                "install",
                if ready { "skipped" } else { "failed" },
                outcome.commands,
                Some(message),
            )
        })
        .collect();
    AgentReport { rows }
}

fn install_runtime_from_source(
    preflight: agent_plugin_installer::DoctorOutcome,
    source: &Path,
) -> AgentRow {
    let runtime = preflight.runtime;
    let mut commands = preflight.commands;
    match prepare_marketplace_registration(runtime, source) {
        Ok(preparation) => commands.extend(preparation),
        Err(failure) => {
            commands.extend(failure.commands);
            return make_row(
                runtime,
                "install",
                "failed",
                commands,
                Some(failure.message),
            );
        }
    }

    match install_plugin(runtime, InstallRequest::local(source, PLUGIN)) {
        Ok(outcome) => {
            commands.extend(outcome.commands);
            make_row(runtime, "install", "installed", commands, None)
        }
        Err(error) => {
            commands.extend(error.completed.clone());
            if let AgentPluginError::CliFailed { command, .. } = &error.error {
                commands.push(command.clone());
            }
            make_row(
                runtime,
                "install",
                "failed",
                commands,
                Some(error.to_string()),
            )
        }
    }
}

fn prepare_marketplace_registration(
    runtime: AgentRuntime,
    source: &Path,
) -> Result<Vec<String>, MarketplacePreparationFailure> {
    let inspection =
        inspect_marketplace(runtime).map_err(|failure| MarketplacePreparationFailure {
            commands: failure.command.into_iter().collect(),
            message: failure.message,
        })?;
    let mut commands = vec![inspection.command];
    let replace = match (&runtime, &inspection.state) {
        (_, MarketplaceState::Absent) => false,
        (AgentRuntime::Codex, _) => true,
        (AgentRuntime::Claude, MarketplaceState::Local(current)) => current != source,
        (AgentRuntime::Claude, MarketplaceState::Other) => true,
    };
    if !replace {
        return Ok(commands);
    }
    let args = ["plugin", "marketplace", "remove", MARKETPLACE]
        .map(str::to_owned)
        .to_vec();
    match run_marketplace_command(runtime, &args, "marketplace replacement") {
        Ok((command, _)) => {
            commands.push(command);
            Ok(commands)
        }
        Err(mut failure) => {
            if let Some(command) = failure.command.take() {
                commands.push(command);
            }
            Err(MarketplacePreparationFailure {
                commands,
                message: failure.message,
            })
        }
    }
}

/// Materialize the binary-embedded package at a content-addressed,
/// versioned path owned by InferLab. Native runtimes persist local
/// marketplace paths, so a call-scoped temporary directory is not a valid
/// installation source.
fn materialize_embedded_package(runtimes: &[AgentRuntime]) -> Result<PathBuf, String> {
    let destination = embedded_package_path()?;
    if runtimes
        .iter()
        .copied()
        .all(|runtime| validate_package(runtime, &destination).is_ok())
    {
        return Ok(destination);
    }

    let parent = destination.parent().ok_or_else(|| {
        format!(
            "embedded plugin package: installation path {} has no parent",
            destination.display()
        )
    })?;
    fs::create_dir_all(parent).map_err(|error| {
        format!(
            "embedded plugin package: cannot create data directory {}: {error}",
            parent.display()
        )
    })?;
    let staging = tempfile::Builder::new()
        .prefix(".inferlab-plugin-")
        .tempdir_in(parent)
        .map_err(|error| {
            format!(
                "embedded plugin package: cannot create staging directory in {}: {error}",
                parent.display()
            )
        })?;
    let decoder = GzDecoder::new(EMBEDDED_PLUGIN_TAR_GZ);
    Archive::new(decoder)
        .unpack(staging.path())
        .map_err(|error| {
            format!("embedded plugin package: cannot extract the binary-embedded payload: {error}")
        })?;
    for runtime in runtimes {
        validate_package(*runtime, staging.path())?;
    }

    if destination.exists() {
        if destination.is_dir() {
            fs::remove_dir_all(&destination)
        } else {
            fs::remove_file(&destination)
        }
        .map_err(|error| {
            format!(
                "embedded plugin package: cannot replace incomplete materialization {}: {error}",
                destination.display()
            )
        })?;
    }
    let staging = staging.keep();
    fs::rename(&staging, &destination).map_err(|error| {
        format!(
            "embedded plugin package: cannot publish {} as {}: {error}",
            staging.display(),
            destination.display()
        )
    })?;
    Ok(destination)
}

fn embedded_package_path() -> Result<PathBuf, String> {
    let data_home = if let Some(path) = std::env::var_os("XDG_DATA_HOME") {
        PathBuf::from(path)
    } else {
        std::env::var_os("HOME")
            .map(PathBuf::from)
            .map(|home| home.join(".local/share"))
            .ok_or_else(|| {
                "embedded plugin package: neither XDG_DATA_HOME nor HOME is set".to_owned()
            })?
    };
    let digest = format!("{:x}", Sha256::digest(EMBEDDED_PLUGIN_TAR_GZ));
    Ok(data_home
        .join("inferlab/agent-plugins")
        .join(format!("{INFERLAB_VERSION}-{digest}")))
}

/// Refresh the plugin from this binary's embedded package. Each release has a
/// new persistent local marketplace path, so native Git-marketplace update
/// commands cannot perform this transition. Reuse the validated local install
/// path to replace the registration and refresh the plugin, then expose the
/// operator-requested update semantics in the report.
pub(crate) fn update(selector: AgentSelector) -> AgentReport {
    let runtimes = selector.runtimes();
    let source = match materialize_embedded_package(runtimes) {
        Ok(path) => path,
        Err(message) => {
            return AgentReport {
                rows: runtimes
                    .iter()
                    .copied()
                    .map(|runtime| failed_gate_row(runtime, "update", message.clone()))
                    .collect(),
            };
        }
    };
    if let Some(report) = package_gate(runtimes, &source, "update") {
        return report;
    }
    let source = match source.canonicalize() {
        Ok(path) => path,
        Err(error) => {
            let message = format!(
                "cannot canonicalize embedded plugin package {}: {error}",
                source.display()
            );
            return AgentReport {
                rows: runtimes
                    .iter()
                    .copied()
                    .map(|runtime| failed_gate_row(runtime, "update", message.clone()))
                    .collect(),
            };
        }
    };

    let mut report = install_from_source(selector, &source);
    for row in &mut report.rows {
        row.operation = "update";
        if row.status == "installed" {
            row.status = "updated";
        }
    }
    report
}

pub(crate) fn uninstall(selector: AgentSelector) -> AgentReport {
    from_batch(
        uninstall_many(
            selector,
            |_| UninstallRequest::new(PLUGIN),
            FailurePolicy::Continue,
        ),
        "uninstall",
        "uninstalled",
    )
}

/// Inferlab validates its shipped package before the shared installer may
/// invoke a native CLI. One invalid runtime blocks all selected runtimes.
fn package_gate(
    runtimes: &[AgentRuntime],
    checkout: &Path,
    operation: &'static str,
) -> Option<AgentReport> {
    let failures = runtimes
        .iter()
        .copied()
        .map(|runtime| validate_package(runtime, checkout).err())
        .collect::<Vec<_>>();
    if failures.iter().all(Option::is_none) {
        return None;
    }

    let rows = runtimes
        .iter()
        .copied()
        .zip(failures)
        .map(|(runtime, failure)| match failure {
            Some(message) => failed_gate_row(runtime, operation, message),
            None => make_row(
                runtime,
                operation,
                "skipped",
                Vec::new(),
                Some("mutations not attempted: a preceding gate failed".to_owned()),
            ),
        })
        .collect();
    Some(AgentReport { rows })
}

/// The package paths one runtime needs before its native CLI may run.
fn package_requirements(runtime: AgentRuntime, checkout: &Path) -> Vec<PathBuf> {
    let marketplace = match runtime {
        AgentRuntime::Claude => ".claude-plugin/marketplace.json",
        AgentRuntime::Codex => ".agents/plugins/marketplace.json",
    };
    let manifest = match runtime {
        AgentRuntime::Claude => "plugins/inferlab/.claude-plugin/plugin.json",
        AgentRuntime::Codex => "plugins/inferlab/.codex-plugin/plugin.json",
    };
    let skill = checkout.join("plugins/inferlab/skills/inferlab");
    vec![
        checkout.join(marketplace),
        checkout.join(manifest),
        skill.join("SKILL.md"),
        skill.join("references/workspace-authoring.md"),
        skill.join("references/workspace-definition.md"),
        skill.join("references/execution-authoring.md"),
        skill.join("references/eval-authoring.md"),
        skill.join("references/bench-authoring.md"),
    ]
}

fn validate_package(runtime: AgentRuntime, checkout: &Path) -> Result<(), String> {
    if !checkout.is_dir() {
        return Err(format!(
            "plugin package for {}: checkout {} is not a directory",
            runtime.id(),
            checkout.display()
        ));
    }
    for required in package_requirements(runtime, checkout) {
        if !required.is_file() {
            return Err(format!(
                "plugin package for {} is missing {}",
                runtime.id(),
                required.display()
            ));
        }
        let contents = fs::read(&required).map_err(|error| {
            format!(
                "plugin package for {} cannot read {}: {error}",
                runtime.id(),
                required.display()
            )
        })?;
        if required
            .extension()
            .is_some_and(|extension| extension == "json")
        {
            serde_json::from_slice::<serde_json::Value>(&contents).map_err(|error| {
                format!(
                    "plugin package for {}: {} is corrupted: invalid JSON: {error}",
                    runtime.id(),
                    required.display()
                )
            })?;
        } else {
            let text = std::str::from_utf8(&contents).map_err(|error| {
                format!(
                    "plugin package for {}: {} is corrupted: invalid UTF-8: {error}",
                    runtime.id(),
                    required.display()
                )
            })?;
            if text.trim().is_empty() {
                return Err(format!(
                    "plugin package for {}: {} is corrupted: file is empty",
                    runtime.id(),
                    required.display()
                ));
            }
        }
    }
    Ok(())
}

/// Map the shared installer's complete batch result into Inferlab's stable
/// JSON envelope. Mutation-time CLI absence remains a failed operation here;
/// `missing` is reserved for the explicit doctor command.
fn from_batch(
    result: BatchResult,
    operation: &'static str,
    success_status: &'static str,
) -> AgentReport {
    let report = match result {
        Ok(report) => report,
        Err(error) => error.into_report(),
    };
    let rows = report
        .outcomes
        .into_iter()
        .map(|outcome| batch_row(outcome, operation, success_status))
        .collect();
    AgentReport { rows }
}

fn batch_row(
    outcome: BatchRuntimeOutcome,
    operation: &'static str,
    success_status: &'static str,
) -> AgentRow {
    let BatchRuntimeOutcome {
        runtime,
        status,
        commands,
        failure,
        skip_reason,
        ..
    } = outcome;
    let row_status = match status {
        BatchStatus::Succeeded => success_status,
        BatchStatus::Skipped => "skipped",
        BatchStatus::Missing | BatchStatus::Failed => "failed",
        _ => "failed",
    };
    let message = match failure {
        Some(BatchFailure::Validation(error)) => Some(error.to_string()),
        Some(BatchFailure::Preflight { message }) => Some(format!(
            "{} CLI ({}) is not ready: {message}",
            runtime.id(),
            runtime.cli()
        )),
        Some(BatchFailure::Operation(error)) => Some(error.to_string()),
        Some(failure) => Some(failure.to_string()),
        None => skip_reason.map(|_| "mutations not attempted: a preceding gate failed".to_owned()),
    };
    make_row(runtime, operation, row_status, commands, message)
}

fn failed_gate_row(runtime: AgentRuntime, operation: &'static str, message: String) -> AgentRow {
    make_row(runtime, operation, "failed", Vec::new(), Some(message))
}

fn make_row(
    runtime: AgentRuntime,
    operation: &'static str,
    status: &'static str,
    commands: Vec<String>,
    message: Option<String>,
) -> AgentRow {
    AgentRow {
        agent: runtime.id(),
        operation,
        status,
        cli: runtime.cli(),
        commands,
        message,
    }
}
