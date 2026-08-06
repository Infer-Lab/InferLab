use crate::InferlabError;
use crate::progress::{Phase, Progress};
use crate::workspace::StackDefinition;
use inferlab_runtime::interrupt;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::fs;
use std::fs::Permissions;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;

pub(crate) mod status;

const PIXI_MANIFEST: &str = "pixi.toml";
const PIXI_LOCK: &str = "pixi.lock";
pub(crate) const PIXI_ENVS_DIR: &str = ".pixi/envs";

/// The on-disk prefix Pixi installs `environment` into — the same
/// convention `adapter_environment_mounts` already assumes for the adapter
/// environment's interpreter path.
pub(crate) fn pixi_environment_prefix(root: &Path, environment: &str) -> PathBuf {
    root.join(PIXI_ENVS_DIR).join(environment)
}

/// A declared environment check resolved to its content identity
/// ([[RFC-0002:C-ENVIRONMENT-CHECKS]]): the script digest keys derived
/// artifacts, so a check edit is never invisible to reuse.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub(crate) struct PlannedEnvironmentCheck {
    pub id: String,
    pub script: PathBuf,
    pub sha256: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub repair_hint: Option<String>,
}

/// A declared image-realization postprocess step resolved to its content
/// identity ([[RFC-0002:C-ENVIRONMENT-CHECKS]]).
#[derive(Clone, Debug, Deserialize, Serialize)]
pub(crate) struct PlannedEnvironmentScript {
    pub id: String,
    pub script: PathBuf,
    pub sha256: String,
}

/// The realization a check examined: the mutable local workspace
/// environment the operator owns, or an image environment the build owns.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub(crate) enum CheckRealization {
    LocalWorkspace,
    Image,
    /// A declared external serving image: not qualified by this workspace,
    /// so no environment-check claim exists for it
    /// ([[RFC-0003:C-RUNTIME-WORKFLOWS]]).
    ExternalImage,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub(crate) enum CheckOutcome {
    Passed,
    Failed,
}

/// One executed check, recorded with the realization it examined
/// ([[RFC-0002:C-ENVIRONMENT-CHECKS]]).
#[derive(Clone, Debug, Deserialize, Serialize)]
pub(crate) struct EnvironmentCheckEvidence {
    pub id: String,
    pub realization: CheckRealization,
    /// The machine whose realization was examined; absent for the
    /// controller's own workspace environment.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub machine: Option<String>,
    pub outcome: CheckOutcome,
    /// Captured combined output for checks Inferlab ran directly; in-image
    /// checks leave their output in the referenced builder log.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub log: Option<PathBuf>,
}

/// One local check for which an exit status and complete output were observed.
#[derive(Debug)]
pub(crate) struct CompletedLocalCheck {
    pub id: String,
    pub outcome: CheckOutcome,
    pub output: String,
    pub repair_hint: Option<String>,
}

impl CompletedLocalCheck {
    pub(crate) fn into_record_evidence(self) -> EnvironmentCheckEvidence {
        EnvironmentCheckEvidence {
            id: self.id,
            realization: CheckRealization::LocalWorkspace,
            machine: None,
            outcome: self.outcome,
            output: Some(self.output),
            log: None,
        }
    }
}

/// A failed local-realization check: local failure means drift, so the
/// declared repair hint goes to the operator who owns the environment.
#[derive(Clone, Debug)]
pub(crate) struct LocalCheckFailure {
    pub id: String,
    pub repair_hint: Option<String>,
    pub output: String,
}

impl LocalCheckFailure {
    pub(crate) fn message(&self, pixi_environment: &str) -> String {
        let mut message = format!(
            "environment check {:?} failed on the local workspace realization of Pixi \
             environment {pixi_environment:?}: {}",
            self.id,
            self.output.trim()
        );
        if let Some(hint) = &self.repair_hint {
            message.push_str(&format!("; repair: {hint}"));
        }
        message
    }
}

/// A declared check attempt that did not produce an exit status.
#[derive(Debug)]
pub(crate) enum LocalCheckExecutionFailure {
    Launch {
        id: String,
        source: std::io::Error,
    },
    NoExitCode {
        id: String,
        status: std::process::ExitStatus,
        stdout: String,
        stderr: String,
    },
}

impl LocalCheckExecutionFailure {
    pub(crate) fn id(&self) -> &str {
        match self {
            Self::Launch { id, .. } | Self::NoExitCode { id, .. } => id,
        }
    }

    pub(crate) fn diagnostics(&self) -> String {
        match self {
            Self::Launch { id, source } => {
                format!("environment check {id:?} failed to launch through pixi: {source}")
            }
            Self::NoExitCode {
                id,
                status,
                stdout,
                stderr,
            } => format!(
                "environment check {id:?} produced no numeric exit code ({status}); stdout: \
                 {stdout}; stderr: {stderr}"
            ),
        }
    }

    pub(crate) fn into_inferlab_error(self) -> InferlabError {
        match self {
            Self::Launch { source, .. } => InferlabError::LaunchPixi {
                action: "environment check",
                source,
            },
            Self::NoExitCode {
                status,
                stdout,
                stderr,
                ..
            } => InferlabError::PixiExit {
                action: "environment check",
                status,
                stdout,
                stderr,
            },
        }
    }
}

#[derive(Debug)]
pub(crate) enum LocalCheckConclusion {
    Passed,
    Failed(LocalCheckFailure),
    ExecutionError(LocalCheckExecutionFailure),
}

#[derive(Debug)]
pub(crate) struct LocalCheckRun {
    pub completed: Vec<CompletedLocalCheck>,
    pub conclusion: LocalCheckConclusion,
}

/// Resolve declared checks and image postprocess steps to content
/// identities, failing when a declared script is missing.
pub(crate) fn plan_environment_checks(
    root: &Path,
    definition: &StackDefinition,
) -> Result<(Vec<PlannedEnvironmentCheck>, Vec<PlannedEnvironmentScript>), InferlabError> {
    let checks = plan_stack_checks(root, definition)?;
    let mut postprocess = Vec::with_capacity(definition.image_postprocess.len());
    for step in &definition.image_postprocess {
        postprocess.push(PlannedEnvironmentScript {
            id: step.id.clone(),
            script: step.script.clone(),
            sha256: environment_script_digest(root, &step.script)?,
        });
    }
    Ok((checks, postprocess))
}

pub(crate) fn plan_stack_checks(
    root: &Path,
    definition: &StackDefinition,
) -> Result<Vec<PlannedEnvironmentCheck>, InferlabError> {
    let mut checks = Vec::with_capacity(definition.checks.len());
    for check in &definition.checks {
        checks.push(PlannedEnvironmentCheck {
            id: check.id.clone(),
            script: check.script.clone(),
            sha256: environment_script_digest(root, &check.script)?,
            repair_hint: check.repair_hint.clone(),
        });
    }
    Ok(checks)
}

fn environment_script_digest(root: &Path, script: &Path) -> Result<String, InferlabError> {
    let path = root.join(script);
    let bytes = fs::read(&path).map_err(|source| InferlabError::Read {
        path: path.clone(),
        source,
    })?;
    Ok(sha256(&bytes))
}

/// Execute the declared checks against the local workspace realization
/// ([[RFC-0002:C-ENVIRONMENT-CHECKS]]): the environment's own interpreter
/// runs each script from the workspace root, stopping at the first failure.
/// Completed evidence covers every check that produced an exit status; a
/// launch error remains a separate conclusion. Inferlab never mutates the
/// local environment itself.
pub(crate) fn run_local_checks(
    root: &Path,
    pixi_environment: &str,
    checks: &[PlannedEnvironmentCheck],
    progress: &Progress,
    phase_name: &str,
) -> Result<LocalCheckRun, InferlabError> {
    let mut completed = Vec::new();
    for (index, check) in checks.iter().enumerate() {
        progress.phase(Phase::named(phase_name).item(&check.id, index + 1, checks.len()))?;
        let output = match Command::new("pixi")
            .current_dir(root)
            .args(["run", "--locked", "--no-install", "--executable", "-e"])
            .arg(pixi_environment)
            .arg("--")
            .arg("python")
            .arg(&check.script)
            .output()
        {
            Ok(output) => output,
            Err(source) => {
                return Ok(LocalCheckRun {
                    completed,
                    conclusion: LocalCheckConclusion::ExecutionError(
                        LocalCheckExecutionFailure::Launch {
                            id: check.id.clone(),
                            source,
                        },
                    ),
                });
            }
        };
        let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
        let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
        if output.status.code().is_none() {
            return Ok(LocalCheckRun {
                completed,
                conclusion: LocalCheckConclusion::ExecutionError(
                    LocalCheckExecutionFailure::NoExitCode {
                        id: check.id.clone(),
                        status: output.status,
                        stdout,
                        stderr,
                    },
                ),
            });
        }
        let combined = format!("{stdout}{stderr}");
        let passed = output.status.success();
        completed.push(CompletedLocalCheck {
            id: check.id.clone(),
            outcome: if passed {
                CheckOutcome::Passed
            } else {
                CheckOutcome::Failed
            },
            output: combined.clone(),
            repair_hint: if passed {
                None
            } else {
                check.repair_hint.clone()
            },
        });
        if !passed {
            return Ok(LocalCheckRun {
                completed,
                conclusion: LocalCheckConclusion::Failed(LocalCheckFailure {
                    id: check.id.clone(),
                    repair_hint: check.repair_hint.clone(),
                    output: combined,
                }),
            });
        }
    }
    Ok(LocalCheckRun {
        completed,
        conclusion: LocalCheckConclusion::Passed,
    })
}

pub(crate) fn tail(text: &str, limit: usize) -> String {
    if text.len() <= limit {
        return text.to_owned();
    }
    let start = text.len() - limit;
    let boundary = (start..text.len())
        .find(|index| text.is_char_boundary(*index))
        .unwrap_or(text.len());
    text[boundary..].to_owned()
}

#[derive(Debug, Serialize)]
pub(crate) struct LockResult {
    pub manifest: PathBuf,
    pub lock: PathBuf,
    pub manifest_sha256: String,
    pub lock_sha256: String,
    pub staged_install: bool,
}

/// The outcome of resolving one environment's usability
/// ([[RFC-0002:C-PIXI-ENVIRONMENT-LIFECYCLE]]): shared by the launch-time
/// gate and the standalone `env status` query, so the two can never observe
/// a different answer for the same environment.
pub(crate) enum EnvironmentCheck {
    Confirmed,
    NeverInstalled,
    NotUsable(String),
}

/// The launch-time usability gate ([[RFC-0002:C-PIXI-ENVIRONMENT-LIFECYCLE]]):
/// backed by the content-confirmation record a prior successful check left
/// behind. Callers that must produce no persisted evidence — ad-hoc
/// execution ([[RFC-0002:C-ADHOC-EXECUTION]]) — use
/// [`ensure_usable_without_confirmation`] instead, never this function.
pub(crate) fn ensure_usable(root: &Path, environment: &str) -> Result<(), InferlabError> {
    match check_environment(root, environment)? {
        EnvironmentCheck::Confirmed => Ok(()),
        EnvironmentCheck::NeverInstalled => Err(unavailable(
            environment,
            "the environment has not been installed".to_owned(),
        )),
        EnvironmentCheck::NotUsable(diagnostics) => Err(unavailable(environment, diagnostics)),
    }
}

/// The ad-hoc usability check ([[RFC-0002:C-ADHOC-EXECUTION]]): presence and
/// a fresh lock-freshness probe only. It never reads or writes the
/// confirmation marker [`ensure_usable`] shares with `env status`, so
/// running it neither trusts qualification evidence another workflow
/// produced nor produces evidence a later launch would trust.
pub(crate) fn ensure_usable_without_confirmation(
    root: &Path,
    environment: &str,
) -> Result<(), InferlabError> {
    if !pixi_environment_prefix(root, environment).is_dir() {
        return Err(unavailable(
            environment,
            "the environment has not been installed".to_owned(),
        ));
    }
    match probe_pixi_usable(root, environment)? {
        None => Ok(()),
        Some(diagnostics) => Err(unavailable(environment, diagnostics)),
    }
}

fn unavailable(environment: &str, diagnostics: String) -> InferlabError {
    InferlabError::PixiEnvironmentUnavailable {
        environment: environment.to_owned(),
        install_command: format!("pixi install --locked --environment {environment}"),
        diagnostics,
    }
}

/// Resolve one environment's usability, backed by the content-confirmation
/// record a prior successful check left behind
/// ([[RFC-0002:C-PIXI-ENVIRONMENT-LIFECYCLE]]).
pub(crate) fn check_environment(
    root: &Path,
    environment: &str,
) -> Result<EnvironmentCheck, InferlabError> {
    // `pixi run --no-install` does not fail when the target environment was
    // never installed: it silently executes the probe against the ambient
    // PATH instead (verified against pixi 0.72.1 on a fresh workspace —
    // `which python3` inside the probe resolved to the system interpreter,
    // not any Pixi prefix). Absence is therefore checked on disk first, so
    // this gate cannot report a never-installed environment as usable.
    if !pixi_environment_prefix(root, environment).is_dir() {
        return Ok(EnvironmentCheck::NeverInstalled);
    }
    let manifest_sha256 = crate::digest::hash_file(&root.join(PIXI_MANIFEST))?;
    let lock_sha256 = crate::digest::hash_file(&root.join(PIXI_LOCK))?;
    if let Some(marker) = read_confirmation_marker(root, environment)
        && marker.pixi_manifest_sha256 == manifest_sha256
        && marker.pixi_lock_sha256 == lock_sha256
    {
        // Confirmed against exactly this manifest and lock content by a
        // prior check; a revision change that left that content unchanged
        // does not invalidate it, so the real probe is skipped.
        return Ok(EnvironmentCheck::Confirmed);
    }
    let Some(diagnostics) = probe_pixi_usable(root, environment)? else {
        // Best-effort: a cache-write failure never invalidates a probe that
        // just succeeded, it only costs the next check its fast path.
        let _ = write_confirmation_marker(root, environment, &manifest_sha256, &lock_sha256);
        return Ok(EnvironmentCheck::Confirmed);
    };
    Ok(EnvironmentCheck::NotUsable(diagnostics))
}

/// The raw pixi usability probe, with no confirmation-marker involvement:
/// `None` on success, `Some(diagnostics)` on failure. The one place either
/// usability path actually shells out to pixi.
fn probe_pixi_usable(root: &Path, environment: &str) -> Result<Option<String>, InferlabError> {
    let output = Command::new("pixi")
        .current_dir(root)
        .args([
            "run",
            "--locked",
            "--no-install",
            "--executable",
            "-e",
            environment,
            "--",
            "true",
        ])
        .output()
        .map_err(|source| InferlabError::LaunchPixi {
            action: "environment check",
            source,
        })?;
    if output.status.success() {
        Ok(None)
    } else {
        Ok(Some(
            String::from_utf8_lossy(&output.stderr).trim().to_owned(),
        ))
    }
}

pub(crate) const CONFIRMATION_CACHE_DIR: &str = ".inferlab/cache/environments";
const CONFIRMATION_SCHEMA_VERSION: u32 = 1;

#[derive(Deserialize, Serialize)]
struct ConfirmationMarker {
    schema_version: u32,
    pixi_manifest_sha256: String,
    pixi_lock_sha256: String,
}

fn confirmation_marker_path(root: &Path, environment: &str) -> PathBuf {
    root.join(CONFIRMATION_CACHE_DIR)
        .join(environment)
        .join("confirmed.json")
}

/// A missing, malformed, or wrong-schema-version marker is indistinguishable
/// from "never confirmed" — the caller falls through to the real probe
/// exactly as it would for a workspace that has never seen this check.
fn read_confirmation_marker(root: &Path, environment: &str) -> Option<ConfirmationMarker> {
    let bytes = fs::read(confirmation_marker_path(root, environment)).ok()?;
    let marker: ConfirmationMarker = serde_json::from_slice(&bytes).ok()?;
    (marker.schema_version == CONFIRMATION_SCHEMA_VERSION).then_some(marker)
}

fn write_confirmation_marker(
    root: &Path,
    environment: &str,
    manifest_sha256: &str,
    lock_sha256: &str,
) -> Result<(), InferlabError> {
    let path = confirmation_marker_path(root, environment);
    let parent = path
        .parent()
        .ok_or_else(|| InferlabError::EnvironmentLifecycle {
            message: format!("path {} has no parent directory", path.display()),
        })?;
    fs::create_dir_all(parent).map_err(|source| InferlabError::EnvironmentIo {
        path: parent.to_path_buf(),
        operation: "create environment confirmation cache directory",
        source,
    })?;
    let marker = ConfirmationMarker {
        schema_version: CONFIRMATION_SCHEMA_VERSION,
        pixi_manifest_sha256: manifest_sha256.to_owned(),
        pixi_lock_sha256: lock_sha256.to_owned(),
    };
    let bytes = serde_json::to_vec_pretty(&marker)
        .map_err(|source| InferlabError::EncodeOutput { source })?;
    atomic_write(&path, &bytes, None)
}

pub(crate) fn lock_workspace_with_progress(
    root: &Path,
    progress: &Progress,
) -> Result<LockResult, InferlabError> {
    interrupt::prepare().map_err(|source| InferlabError::EnvironmentInterrupt { source })?;
    let mut transaction = WorkspaceFileTransaction::begin(root)?;
    let result = produce_lock(root, &mut transaction, progress);
    match result {
        Ok(result) => {
            transaction.commit();
            Ok(result)
        }
        Err(error) => {
            progress.phase(Phase::named("restoration after failure or interruption"))?;
            match transaction.restore() {
                Ok(()) => Err(error),
                Err(restoration) => Err(InferlabError::EnvironmentRestore {
                    operation: error.to_string(),
                    restoration: restoration.to_string(),
                }),
            }
        }
    }
}

fn produce_lock(
    root: &Path,
    transaction: &mut WorkspaceFileTransaction,
    progress: &Progress,
) -> Result<LockResult, InferlabError> {
    let full_text = std::str::from_utf8(&transaction.manifest_bytes).map_err(|error| {
        InferlabError::InvalidConfig {
            message: format!("{} is not UTF-8: {error}", transaction.manifest.display()),
        }
    })?;
    let full_manifest: toml::Value =
        toml::from_str(full_text).map_err(|source| InferlabError::ParseToml {
            path: transaction.manifest.clone(),
            source,
        })?;
    let (base_manifest, staged_install) = derive_base_manifest(&full_manifest);

    if staged_install {
        let base_text = toml::to_string_pretty(&base_manifest)
            .map_err(|source| InferlabError::SerializeToml { source })?;
        transaction.write_manifest(base_text.as_bytes())?;
        progress.phase(Phase::named("base-lock production"))?;
        run_pixi_lock(root, &transaction.manifest)?;
        progress.phase(Phase::named("staged base installation"))?;
        run_pixi_base_install(root, &transaction.manifest)?;
        transaction.restore_manifest()?;
    }

    progress.phase(Phase::named("authoritative full-lock production"))?;
    run_pixi_lock(root, &transaction.manifest)?;
    let lock_bytes = fs::read(&transaction.lock).map_err(|source| InferlabError::Read {
        path: transaction.lock.clone(),
        source,
    })?;
    Ok(LockResult {
        manifest: transaction.manifest.clone(),
        lock: transaction.lock.clone(),
        manifest_sha256: sha256(&transaction.manifest_bytes),
        lock_sha256: sha256(&lock_bytes),
        staged_install,
    })
}

fn derive_base_manifest(full: &toml::Value) -> (toml::Value, bool) {
    let packages = full
        .get("pypi-options")
        .and_then(|options| options.get("no-build-isolation"))
        .and_then(toml::Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(toml::Value::as_str)
        .map(str::to_owned)
        .collect::<BTreeSet<_>>();
    let mut base = full.clone();
    let removed = strip_local_packages(&mut base, &packages);
    (base, removed)
}

fn strip_local_packages(value: &mut toml::Value, packages: &BTreeSet<String>) -> bool {
    let Some(table) = value.as_table_mut() else {
        return false;
    };
    let mut removed = false;
    for key in ["pypi-dependencies", "dependency-overrides"] {
        if let Some(dependencies) = table.get_mut(key).and_then(toml::Value::as_table_mut) {
            dependencies.retain(|package, dependency| {
                let keep = !packages.contains(package) || !is_local_dependency(dependency);
                removed |= !keep;
                keep
            });
        }
    }
    for (_, child) in table.iter_mut() {
        removed |= strip_local_packages(child, packages);
    }
    removed
}

fn is_local_dependency(value: &toml::Value) -> bool {
    value
        .as_table()
        .is_some_and(|dependency| dependency.contains_key("path"))
}

fn run_pixi_lock(root: &Path, manifest: &Path) -> Result<(), InferlabError> {
    run_pixi(
        root,
        "lock",
        Command::new("pixi")
            .arg("lock")
            .arg("--manifest-path")
            .arg(manifest),
    )
}

fn run_pixi_base_install(root: &Path, manifest: &Path) -> Result<(), InferlabError> {
    run_pixi(
        root,
        "install base environment",
        Command::new("pixi")
            .arg("install")
            .arg("--all")
            .arg("--locked")
            .arg("--manifest-path")
            .arg(manifest),
    )
}

fn run_pixi(root: &Path, action: &'static str, command: &mut Command) -> Result<(), InferlabError> {
    let output = command
        .current_dir(root)
        .output()
        .map_err(|source| InferlabError::LaunchPixi { action, source })?;
    if interrupt::received() {
        return Err(InferlabError::EnvironmentLifecycle {
            message: format!("pixi {action} was interrupted"),
        });
    }
    if output.status.success() {
        Ok(())
    } else {
        Err(InferlabError::PixiExit {
            action,
            status: output.status,
            stdout: String::from_utf8_lossy(&output.stdout).trim().to_owned(),
            stderr: String::from_utf8_lossy(&output.stderr).trim().to_owned(),
        })
    }
}

struct WorkspaceFileTransaction {
    manifest: PathBuf,
    lock: PathBuf,
    manifest_bytes: Vec<u8>,
    manifest_permissions: Permissions,
    previous_lock: Option<(Vec<u8>, Permissions)>,
    finished: bool,
}

impl WorkspaceFileTransaction {
    fn begin(root: &Path) -> Result<Self, InferlabError> {
        let manifest = root.join(PIXI_MANIFEST);
        let lock = root.join(PIXI_LOCK);
        let manifest_bytes = fs::read(&manifest).map_err(|source| InferlabError::Read {
            path: manifest.clone(),
            source,
        })?;
        let manifest_permissions = fs::metadata(&manifest)
            .map_err(|source| InferlabError::Read {
                path: manifest.clone(),
                source,
            })?
            .permissions();
        let previous_lock = match fs::read(&lock) {
            Ok(bytes) => {
                let permissions = fs::metadata(&lock)
                    .map_err(|source| InferlabError::Read {
                        path: lock.clone(),
                        source,
                    })?
                    .permissions();
                Some((bytes, permissions))
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
            Err(source) => {
                return Err(InferlabError::Read { path: lock, source });
            }
        };
        Ok(Self {
            manifest,
            lock,
            manifest_bytes,
            manifest_permissions,
            previous_lock,
            finished: false,
        })
    }

    fn write_manifest(&self, bytes: &[u8]) -> Result<(), InferlabError> {
        atomic_write(&self.manifest, bytes, Some(&self.manifest_permissions))
    }

    fn restore_manifest(&self) -> Result<(), InferlabError> {
        self.write_manifest(&self.manifest_bytes)
    }

    fn restore(&mut self) -> Result<(), InferlabError> {
        self.restore_manifest()?;
        match &self.previous_lock {
            Some((bytes, permissions)) => atomic_write(&self.lock, bytes, Some(permissions))?,
            None => match fs::remove_file(&self.lock) {
                Ok(()) => {}
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                Err(source) => {
                    return Err(InferlabError::EnvironmentIo {
                        path: self.lock.clone(),
                        operation: "remove partial lock",
                        source,
                    });
                }
            },
        }
        self.finished = true;
        Ok(())
    }

    fn commit(&mut self) {
        self.finished = true;
    }
}

impl Drop for WorkspaceFileTransaction {
    fn drop(&mut self) {
        if !self.finished {
            let _ = self.restore();
        }
    }
}

fn atomic_write(
    path: &Path,
    bytes: &[u8],
    permissions: Option<&Permissions>,
) -> Result<(), InferlabError> {
    let parent = path
        .parent()
        .ok_or_else(|| InferlabError::EnvironmentLifecycle {
            message: format!("path {} has no parent directory", path.display()),
        })?;
    let mut temporary =
        tempfile::NamedTempFile::new_in(parent).map_err(|source| InferlabError::EnvironmentIo {
            path: parent.to_path_buf(),
            operation: "create temporary file",
            source,
        })?;
    temporary
        .write_all(bytes)
        .and_then(|()| temporary.flush())
        .map_err(|source| InferlabError::EnvironmentIo {
            path: temporary.path().to_path_buf(),
            operation: "write temporary file",
            source,
        })?;
    if let Some(permissions) = permissions {
        temporary
            .as_file()
            .set_permissions(permissions.clone())
            .map_err(|source| InferlabError::EnvironmentIo {
                path: temporary.path().to_path_buf(),
                operation: "preserve file permissions",
                source,
            })?;
    }
    temporary
        .persist(path)
        .map_err(|failure| InferlabError::EnvironmentIo {
            path: path.to_path_buf(),
            operation: "replace workspace file",
            source: failure.error,
        })?;
    Ok(())
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
