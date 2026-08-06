use crate::InferlabError;
use crate::progress::{Phase, Progress};
use fs2::FileExt;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::ffi::OsString;
use std::fs::{self, File, OpenOptions};
use std::path::{Path, PathBuf};
use std::process::Command;

use inferlab_runtime::operation_bound::OperationBound;

const INFERLAB_VERSION: &str = env!("CARGO_PKG_VERSION");
/// Schema version written into `complete.json` and required when reading it
/// back; the write and read gates share this one const.
const COMPLETION_SCHEMA_VERSION: u32 = 3;
const MANIFEST: &str = include_str!("../resources/eval-toolchain/pixi.toml");
const LOCK: &str = include_str!("../resources/eval-toolchain/pixi.lock");
include!(concat!(env!("OUT_DIR"), "/toolchain_python_files.rs"));

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct EvalToolchainIdentity {
    pub inferlab_version: String,
    pub platform: String,
    pub manifest_sha256: String,
    pub lock_sha256: String,
    pub runner_version: String,
    pub runner_sha256: String,
    pub lm_eval_version: String,
    pub bundled_task_closure_sha256: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct BundledEvalTask {
    pub name: String,
    pub task_identity: String,
    pub path: PathBuf,
    pub task_closure_sha256: String,
    pub task_definition_sha256: String,
    pub prompt_asset_sha256: String,
    pub dataset_asset_sha256: String,
    pub scorer_sha256: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct BenchToolchainIdentity {
    pub inferlab_version: String,
    pub platform: String,
    pub manifest_sha256: String,
    pub lock_sha256: String,
    pub runner_version: String,
    pub runner_sha256: String,
    pub aiperf_version: String,
    pub transformers_version: String,
}

pub(crate) struct InstalledEvalToolchain {
    pub identity: EvalToolchainIdentity,
    pub python: PathBuf,
    pub runner: PathBuf,
    pub python_path: PathBuf,
    pub bundled_tasks_path: PathBuf,
}

pub(crate) struct InstalledBenchToolchain {
    pub identity: BenchToolchainIdentity,
    pub python: PathBuf,
    pub runner: PathBuf,
    pub python_path: PathBuf,
}

#[derive(Debug, Serialize)]
pub(crate) struct InstallReport {
    pub state: InstallState,
    pub path: PathBuf,
    pub eval: EvalToolchainIdentity,
    pub bench: BenchToolchainIdentity,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum InstallState {
    Installed,
    AlreadyInstalled,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Completion {
    schema_version: u32,
    eval: EvalToolchainIdentity,
    bench: BenchToolchainIdentity,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct EvalHandshake {
    lm_eval_version: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct BenchHandshake {
    aiperf_version: String,
    transformers_version: String,
}

#[derive(Debug, Deserialize)]
struct PixiHostInfo {
    platform: String,
    virtual_packages: Vec<String>,
}

pub(crate) fn install_with_progress(progress: &Progress) -> Result<InstallReport, InferlabError> {
    progress.phase(Phase::named("installation-state inspection"))?;
    let platform = host_platform()?;
    let path = install_path(platform)?;
    let parent = path
        .parent()
        .ok_or_else(|| InferlabError::ToolchainVerification {
            message: format!("toolchain path {} has no parent", path.display()),
        })?;
    create_dir_all(parent)?;
    let lock_path = parent.join(format!(".{platform}.install.lock"));
    let lock = open_lock(&lock_path)?;
    progress.phase(Phase::named("writer-lock waiting").lock(&lock_path))?;
    lock.lock_exclusive()
        .map_err(|source| InferlabError::ToolchainIo {
            operation: "lock",
            path: lock_path,
            source,
        })?;

    if let Some(completion) = installed_completion(&path, platform) {
        return Ok(InstallReport {
            state: InstallState::AlreadyInstalled,
            path,
            eval: completion.eval,
            bench: completion.bench,
        });
    }

    if path.exists() {
        progress.phase(Phase::named("incomplete-installation replacement"))?;
        fs::remove_dir_all(&path).map_err(|source| removal_error(&path, source))?;
    }
    write_release_files(&path)?;
    progress.phase(Phase::named("Pixi installation"))?;
    install_locked(&path)?;
    progress.phase(Phase::named("Eval verification"))?;
    let eval = eval_identity(platform, verify_eval_runtime(&path)?);
    progress.phase(Phase::named("Bench verification"))?;
    let bench = bench_identity(platform, verify_bench_runtime(&path)?);
    write_completion(
        &path,
        &Completion {
            schema_version: COMPLETION_SCHEMA_VERSION,
            eval: eval.clone(),
            bench: bench.clone(),
        },
    )?;

    Ok(InstallReport {
        state: InstallState::Installed,
        path,
        eval,
        bench,
    })
}

pub(crate) fn require_eval() -> Result<InstalledEvalToolchain, InferlabError> {
    let platform = host_platform()?;
    let path = install_path(platform)?;
    let completion = require_completion(&path, platform)?;
    Ok(InstalledEvalToolchain {
        identity: completion.eval,
        python: eval_python_path(&path),
        runner: eval_runner_path(&path),
        python_path: path.join("runner"),
        bundled_tasks_path: path.join("runner/inferlab_eval_runner/bundled_tasks"),
    })
}

impl InstalledEvalToolchain {
    pub(crate) fn bundled_task(&self, name: &str) -> Result<BundledEvalTask, InferlabError> {
        if name != "estonia" {
            return Err(InferlabError::InvalidConfig {
                message: format!("unknown bundled Eval task {name:?}"),
            });
        }
        Ok(BundledEvalTask {
            name: name.to_owned(),
            task_identity: "inferlab_estonia".to_owned(),
            path: self.bundled_tasks_path.join("estonia/estonia.yaml"),
            task_closure_sha256: bundled_task_closure_digest(),
            task_definition_sha256: digest(ESTONIA_TASK.as_bytes()),
            prompt_asset_sha256: digest(ESTONIA_PROMPT.as_bytes()),
            dataset_asset_sha256: digest(ESTONIA_DATASET.as_bytes()),
            scorer_sha256: digest(ESTONIA_SCORER.as_bytes()),
        })
    }
}

pub(crate) fn require_bench() -> Result<InstalledBenchToolchain, InferlabError> {
    let platform = host_platform()?;
    let path = install_path(platform)?;
    let completion = require_completion(&path, platform)?;
    Ok(InstalledBenchToolchain {
        identity: completion.bench,
        python: bench_python_path(&path),
        runner: bench_runner_path(&path),
        python_path: path.join("runner"),
    })
}

fn require_completion(path: &Path, platform: &str) -> Result<Completion, InferlabError> {
    installed_completion(path, platform).ok_or_else(|| InferlabError::ToolchainUnavailable {
        version: INFERLAB_VERSION.to_owned(),
        platform: platform.to_owned(),
    })
}

fn host_platform() -> Result<&'static str, InferlabError> {
    // [[RFC-0004:C-INFERLAB-TOOLCHAIN]] The release binary is statically
    // linked with musl, so its compilation target cannot describe the host
    // ABI that the separately installed measurement environments must use.
    let uname = rustix::system::uname();
    let sysname = uname.sysname().to_string_lossy();
    let machine = uname.machine().to_string_lossy();
    let kernel_platform = resolve_kernel_platform(&sysname, &machine)?;
    let pixi = pixi_host_info()?;
    resolve_host_platform(&sysname, &machine, kernel_platform, &pixi)
}

fn resolve_kernel_platform(
    sysname: &str,
    machine: &str,
) -> Result<(&'static str, &'static str), InferlabError> {
    match (sysname, machine) {
        ("Linux", "x86_64") => Ok(("linux-x86_64", "linux-64")),
        ("Linux", "aarch64") => Ok(("linux-aarch64", "linux-aarch64")),
        _ => Err(InferlabError::UnsupportedToolchainPlatform {
            platform: format!("kernel={sysname}/{machine}"),
        }),
    }
}

fn resolve_host_platform(
    sysname: &str,
    machine: &str,
    (platform, expected_pixi_platform): (&'static str, &'static str),
    pixi: &PixiHostInfo,
) -> Result<&'static str, InferlabError> {
    if pixi.platform != expected_pixi_platform {
        return Err(InferlabError::UnsupportedToolchainPlatform {
            platform: format!(
                "kernel={sysname}/{machine}, pixi-platform={}",
                pixi.platform
            ),
        });
    }
    if !pixi
        .virtual_packages
        .iter()
        .any(|package| package == "__glibc" || package.starts_with("__glibc="))
    {
        return Err(InferlabError::UnsupportedToolchainPlatform {
            platform: format!(
                "kernel={sysname}/{machine}, pixi-platform={}, virtual-packages contain no __glibc",
                pixi.platform
            ),
        });
    }
    Ok(platform)
}

fn pixi_host_info() -> Result<PixiHostInfo, InferlabError> {
    let argv = ["pixi", "info", "--json"];
    let bound = OperationBound::unbounded();
    let output = inferlab_runtime::container::run_with_bound(
        &argv,
        Some(Path::new("/")),
        None,
        &bound,
        None,
    );
    let (status, stdout, stderr) = match output {
        Ok(inferlab_runtime::container::BoundedWait::Exited {
            status,
            stdout,
            stderr,
        }) => (status, stdout, stderr),
        Ok(inferlab_runtime::container::BoundedWait::Expired { .. }) => {
            return Err(InferlabError::ToolchainVerification {
                message: "unbounded Pixi host-platform inspection unexpectedly expired".to_owned(),
            });
        }
        Ok(inferlab_runtime::container::BoundedWait::Interrupted { kill, .. }) => {
            kill.map_err(|source| InferlabError::LaunchToolchain {
                action: "Pixi host-platform inspection cleanup",
                source,
            })?;
            return Err(InferlabError::LaunchToolchain {
                action: "Pixi host-platform inspection",
                source: std::io::Error::new(
                    std::io::ErrorKind::Interrupted,
                    "host-platform inspection was interrupted",
                ),
            });
        }
        Err(
            inferlab_runtime::container::BoundedError::Launch(source)
            | inferlab_runtime::container::BoundedError::Stdin(source)
            | inferlab_runtime::container::BoundedError::Wait(source),
        ) => {
            return Err(InferlabError::LaunchToolchain {
                action: "Pixi host-platform inspection",
                source,
            });
        }
        Err(inferlab_runtime::container::BoundedError::WaitCleanup { source, .. }) => {
            return Err(InferlabError::LaunchToolchain {
                action: "Pixi host-platform inspection",
                source,
            });
        }
    };
    if !status.success() {
        return Err(InferlabError::ToolchainExit {
            action: "Pixi host-platform inspection",
            status,
            stdout: String::from_utf8_lossy(&stdout).into_owned(),
            stderr: String::from_utf8_lossy(&stderr).into_owned(),
        });
    }
    serde_json::from_slice(&stdout).map_err(|error| InferlabError::ToolchainVerification {
        message: format!("Pixi host-platform inspection returned invalid JSON: {error}"),
    })
}

fn data_home() -> Result<PathBuf, InferlabError> {
    if let Some(path) = std::env::var_os("XDG_DATA_HOME") {
        return Ok(PathBuf::from(path));
    }
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .map(|home| home.join(".local/share"))
        .ok_or_else(|| InferlabError::ToolchainVerification {
            message: "neither XDG_DATA_HOME nor HOME is set".to_owned(),
        })
}

fn install_path(platform: &str) -> Result<PathBuf, InferlabError> {
    Ok(data_home()?
        .join("inferlab/toolchains")
        .join(INFERLAB_VERSION)
        .join(platform))
}

fn eval_identity(platform: &str, handshake: EvalHandshake) -> EvalToolchainIdentity {
    EvalToolchainIdentity {
        inferlab_version: INFERLAB_VERSION.to_owned(),
        platform: platform.to_owned(),
        manifest_sha256: digest(MANIFEST.as_bytes()),
        lock_sha256: digest(LOCK.as_bytes()),
        runner_version: INFERLAB_VERSION.to_owned(),
        runner_sha256: eval_runner_digest(),
        lm_eval_version: handshake.lm_eval_version,
        bundled_task_closure_sha256: bundled_task_closure_digest(),
    }
}

fn bench_identity(platform: &str, handshake: BenchHandshake) -> BenchToolchainIdentity {
    BenchToolchainIdentity {
        inferlab_version: INFERLAB_VERSION.to_owned(),
        platform: platform.to_owned(),
        manifest_sha256: digest(MANIFEST.as_bytes()),
        lock_sha256: digest(LOCK.as_bytes()),
        runner_version: INFERLAB_VERSION.to_owned(),
        runner_sha256: bench_runner_digest(),
        aiperf_version: handshake.aiperf_version,
        transformers_version: handshake.transformers_version,
    }
}

fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn bundled_task_closure_digest() -> String {
    let mut digest = Sha256::new();
    for (path, contents) in [
        ("estonia/dataset.json", ESTONIA_DATASET.as_bytes()),
        ("estonia/estonia.py", ESTONIA_SCORER.as_bytes()),
        ("estonia/estonia.yaml", ESTONIA_TASK.as_bytes()),
        ("estonia/prompt.txt", ESTONIA_PROMPT.as_bytes()),
    ] {
        digest.update(path.len().to_le_bytes());
        digest.update(path.as_bytes());
        digest.update(contents.len().to_le_bytes());
        digest.update(contents);
    }
    format!("{:x}", digest.finalize())
}

fn bench_runner_digest() -> String {
    runner_digest("inferlab_bench_runner/")
}

fn eval_runner_digest() -> String {
    runner_digest("inferlab_eval_runner/")
}

fn runner_digest(runner_prefix: &str) -> String {
    let mut digest = Sha256::new();
    for (path, contents) in TOOLCHAIN_PYTHON_FILES.iter().filter(|(path, _)| {
        path.starts_with(runner_prefix) || path.starts_with("inferlab_measurement_sdk/")
    }) {
        digest.update(path.len().to_le_bytes());
        digest.update(path.as_bytes());
        digest.update(contents.len().to_le_bytes());
        digest.update(contents.as_bytes());
    }
    format!("{:x}", digest.finalize())
}

fn open_lock(path: &Path) -> Result<File, InferlabError> {
    OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .truncate(false)
        .open(path)
        .map_err(|source| InferlabError::ToolchainIo {
            operation: "open lock",
            path: path.to_path_buf(),
            source,
        })
}

fn create_dir_all(path: &Path) -> Result<(), InferlabError> {
    fs::create_dir_all(path).map_err(|source| InferlabError::ToolchainIo {
        operation: "create",
        path: path.to_path_buf(),
        source,
    })
}

fn write_release_files(path: &Path) -> Result<(), InferlabError> {
    write(path.join("pixi.toml"), MANIFEST)?;
    write(path.join("pixi.lock"), LOCK)?;
    for (relative, contents) in TOOLCHAIN_PYTHON_FILES {
        write(path.join("runner").join(relative), contents)?;
    }
    Ok(())
}

fn write(path: PathBuf, contents: &str) -> Result<(), InferlabError> {
    let parent = path
        .parent()
        .ok_or_else(|| InferlabError::ToolchainVerification {
            message: format!("toolchain file path {} has no parent", path.display()),
        })?;
    create_dir_all(parent)?;
    fs::write(&path, contents).map_err(|source| InferlabError::ToolchainIo {
        operation: "write",
        path,
        source,
    })
}

fn install_locked(path: &Path) -> Result<(), InferlabError> {
    let manifest = path.join("pixi.toml");
    let argv = vec![
        OsString::from("pixi"),
        OsString::from("install"),
        OsString::from("--manifest-path"),
        manifest.into_os_string(),
        OsString::from("--all"),
        OsString::from("--locked"),
    ];
    let bound = OperationBound::unbounded();
    let (status, stdout, stderr) =
        match inferlab_runtime::container::run_with_bound(&argv, None, None, &bound, None) {
            Ok(inferlab_runtime::container::BoundedWait::Exited {
                status,
                stdout,
                stderr,
            }) => (status, stdout, stderr),
            Ok(inferlab_runtime::container::BoundedWait::Expired { .. }) => {
                return Err(InferlabError::ToolchainVerification {
                    message: "unbounded Pixi installation unexpectedly expired".to_owned(),
                });
            }
            Ok(inferlab_runtime::container::BoundedWait::Interrupted { kill, .. }) => {
                kill.map_err(|source| InferlabError::LaunchToolchain {
                    action: "Pixi install cleanup",
                    source,
                })?;
                return Err(InferlabError::LaunchToolchain {
                    action: "Pixi install",
                    source: std::io::Error::new(
                        std::io::ErrorKind::Interrupted,
                        "Pixi installation was interrupted",
                    ),
                });
            }
            Err(
                inferlab_runtime::container::BoundedError::Launch(source)
                | inferlab_runtime::container::BoundedError::Stdin(source)
                | inferlab_runtime::container::BoundedError::Wait(source),
            ) => {
                return Err(InferlabError::LaunchToolchain {
                    action: "Pixi install",
                    source,
                });
            }
            Err(inferlab_runtime::container::BoundedError::WaitCleanup { source, .. }) => {
                return Err(InferlabError::LaunchToolchain {
                    action: "Pixi install",
                    source,
                });
            }
        };
    if status.success() {
        Ok(())
    } else {
        Err(InferlabError::ToolchainExit {
            action: "Pixi install",
            status,
            stdout: String::from_utf8_lossy(&stdout).into_owned(),
            stderr: String::from_utf8_lossy(&stderr).into_owned(),
        })
    }
}

fn verify_eval_runtime(path: &Path) -> Result<EvalHandshake, InferlabError> {
    let handshake: EvalHandshake = run_handshake(
        &eval_python_path(path),
        &eval_runner_path(path),
        "Eval runner verification",
    )?;
    let expected = pinned_pypi_version("eval", "lm-eval")?;
    if handshake.lm_eval_version != expected {
        return Err(InferlabError::ToolchainVerification {
            message: format!(
                "Eval runner reported lm-eval {}, expected {expected}",
                handshake.lm_eval_version
            ),
        });
    }
    Ok(handshake)
}

fn verify_bench_runtime(path: &Path) -> Result<BenchHandshake, InferlabError> {
    let handshake: BenchHandshake = run_handshake(
        &bench_python_path(path),
        &bench_runner_path(path),
        "Bench runner verification",
    )?;
    let expected = pinned_pypi_version("bench", "aiperf")?;
    if handshake.aiperf_version != expected {
        return Err(InferlabError::ToolchainVerification {
            message: format!(
                "Bench runner reported AIPerf {}, expected {expected}",
                handshake.aiperf_version
            ),
        });
    }
    if handshake.transformers_version.trim().is_empty() {
        return Err(InferlabError::ToolchainVerification {
            message: "Bench runner reported an empty Transformers version".to_owned(),
        });
    }
    Ok(handshake)
}

fn pinned_pypi_version(feature: &str, package: &str) -> Result<String, InferlabError> {
    let manifest: toml::Value =
        toml::from_str(MANIFEST).map_err(|error| InferlabError::ToolchainVerification {
            message: format!("embedded toolchain manifest is invalid: {error}"),
        })?;
    let dependency = manifest
        .get("feature")
        .and_then(|value| value.get(feature))
        .and_then(|value| value.get("pypi-dependencies"))
        .and_then(|value| value.get(package))
        .ok_or_else(|| InferlabError::ToolchainVerification {
            message: format!(
                "embedded toolchain manifest has no version for feature {feature:?} package {package:?}"
            ),
        })?;
    let requirement = dependency
        .as_str()
        .or_else(|| dependency.get("version").and_then(toml::Value::as_str))
        .ok_or_else(|| InferlabError::ToolchainVerification {
            message: format!(
                "embedded toolchain manifest has no version for feature {feature:?} package {package:?}"
            ),
        })?;
    requirement
        .strip_prefix("==")
        .filter(|version| !version.is_empty())
        .map(str::to_owned)
        .ok_or_else(|| InferlabError::ToolchainVerification {
            message: format!(
                "embedded toolchain requirement for {package:?} is not an exact pin: {requirement:?}"
            ),
        })
}

fn run_handshake<T: DeserializeOwned>(
    python: &Path,
    runner: &Path,
    action: &'static str,
) -> Result<T, InferlabError> {
    let runner_root =
        runner
            .ancestors()
            .nth(2)
            .ok_or_else(|| InferlabError::ToolchainVerification {
                message: format!("runner path {} has no toolchain root", runner.display()),
            })?;
    let output = Command::new(python)
        .arg(runner)
        .arg("--handshake")
        .env("PYTHONPATH", runner_root)
        .env("PYTHONNOUSERSITE", "1")
        .output()
        .map_err(|source| InferlabError::LaunchToolchain { action, source })?;
    if !output.status.success() {
        return Err(InferlabError::ToolchainExit {
            action,
            status: output.status,
            stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
            stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
        });
    }
    serde_json::from_slice(&output.stdout).map_err(|error| InferlabError::ToolchainVerification {
        message: format!("{action} handshake is invalid: {error}"),
    })
}

fn write_completion(path: &Path, completion: &Completion) -> Result<(), InferlabError> {
    let destination = path.join("complete.json");
    let temporary = path.join("complete.json.tmp");
    let mut bytes = serde_json::to_vec_pretty(completion).map_err(|error| {
        InferlabError::ToolchainVerification {
            message: format!("failed to encode completion metadata: {error}"),
        }
    })?;
    bytes.push(b'\n');
    fs::write(&temporary, bytes).map_err(|source| InferlabError::ToolchainIo {
        operation: "write",
        path: temporary.clone(),
        source,
    })?;
    fs::rename(&temporary, &destination).map_err(|source| InferlabError::ToolchainIo {
        operation: "publish",
        path: destination,
        source,
    })
}

fn read_completion(path: &Path) -> Option<Completion> {
    let bytes = fs::read(path.join("complete.json")).ok()?;
    let completion: Completion = serde_json::from_slice(&bytes).ok()?;
    (completion.schema_version == COMPLETION_SCHEMA_VERSION).then_some(completion)
}

fn installed_completion(path: &Path, platform: &str) -> Option<Completion> {
    let completion = read_completion(path)?;
    (eval_identity_matches(&completion.eval, platform)
        && bench_identity_matches(&completion.bench, platform)
        && release_files_match(path)
        && eval_python_path(path).is_file()
        && bench_python_path(path).is_file())
    .then_some(completion)
}

fn eval_identity_matches(identity: &EvalToolchainIdentity, platform: &str) -> bool {
    common_identity_matches(
        &identity.inferlab_version,
        &identity.platform,
        &identity.manifest_sha256,
        &identity.lock_sha256,
        platform,
    ) && identity.runner_version == INFERLAB_VERSION
        && identity.runner_sha256 == eval_runner_digest()
        && pinned_pypi_version("eval", "lm-eval")
            .is_ok_and(|expected| identity.lm_eval_version == expected)
        && identity.bundled_task_closure_sha256 == bundled_task_closure_digest()
}

fn bench_identity_matches(identity: &BenchToolchainIdentity, platform: &str) -> bool {
    common_identity_matches(
        &identity.inferlab_version,
        &identity.platform,
        &identity.manifest_sha256,
        &identity.lock_sha256,
        platform,
    ) && identity.runner_version == INFERLAB_VERSION
        && identity.runner_sha256 == bench_runner_digest()
        && pinned_pypi_version("bench", "aiperf")
            .is_ok_and(|expected| identity.aiperf_version == expected)
}

fn common_identity_matches(
    inferlab_version: &str,
    identity_platform: &str,
    manifest_sha256: &str,
    lock_sha256: &str,
    platform: &str,
) -> bool {
    inferlab_version == INFERLAB_VERSION
        && identity_platform == platform
        && manifest_sha256 == digest(MANIFEST.as_bytes())
        && lock_sha256 == digest(LOCK.as_bytes())
}

fn release_files_match(path: &Path) -> bool {
    let manifest = fs::read(path.join("pixi.toml")).ok();
    let lock = fs::read(path.join("pixi.lock")).ok();
    let python_payload_matches = TOOLCHAIN_PYTHON_FILES.iter().all(|(relative, contents)| {
        fs::read(path.join("runner").join(relative))
            .is_ok_and(|installed| installed == contents.as_bytes())
    });
    manifest
        .as_deref()
        .is_some_and(|bytes| digest(bytes) == digest(MANIFEST.as_bytes()))
        && lock
            .as_deref()
            .is_some_and(|bytes| digest(bytes) == digest(LOCK.as_bytes()))
        && python_payload_matches
}

fn eval_python_path(path: &Path) -> PathBuf {
    path.join(".pixi/envs/eval/bin/python")
}

fn bench_python_path(path: &Path) -> PathBuf {
    path.join(".pixi/envs/bench/bin/python")
}

fn eval_runner_path(path: &Path) -> PathBuf {
    path.join("runner/inferlab_eval_runner/eval_client.py")
}

fn bench_runner_path(path: &Path) -> PathBuf {
    path.join("runner/inferlab_bench_runner/bench_client.py")
}

/// A replacement blocked by live holders identifies the holding processes
/// ([[RFC-0004:C-INFERLAB-TOOLCHAIN]]): the failure is otherwise
/// undiagnosable on network filesystems, where open handles turn removals
/// into silly-rename residue and the path stays busy until the holders
/// exit.
fn removal_error(path: &Path, source: std::io::Error) -> InferlabError {
    let holders = holding_processes(path);
    if holders.is_empty() {
        return InferlabError::ToolchainIo {
            operation: "remove incomplete",
            path: path.to_path_buf(),
            source,
        };
    }
    InferlabError::ToolchainHeld {
        path: path.to_path_buf(),
        holders: holders.join(", "),
        source,
    }
}

/// Same-user processes holding the path: executable, working directory,
/// open file descriptors, or mapped files under it. Unreadable /proc
/// entries (other users' processes) are skipped.
fn holding_processes(path: &Path) -> Vec<String> {
    const HOLDER_LIMIT: usize = 8;
    let prefix = path.to_string_lossy().into_owned();
    let mut holders = Vec::new();
    let Ok(entries) = fs::read_dir("/proc") else {
        return holders;
    };
    for entry in entries.flatten() {
        let Some(pid) = entry
            .file_name()
            .to_str()
            .and_then(|n| n.parse::<u32>().ok())
        else {
            continue;
        };
        let proc_dir = entry.path();
        if !process_holds_path(&proc_dir, &prefix) {
            continue;
        }
        let comm = fs::read_to_string(proc_dir.join("comm")).unwrap_or_default();
        holders.push(format!("{pid} ({})", comm.trim()));
        if holders.len() >= HOLDER_LIMIT {
            holders.push("and possibly more".to_owned());
            break;
        }
    }
    holders
}

fn process_holds_path(proc_dir: &Path, prefix: &str) -> bool {
    let link_holds = |name: &str| {
        fs::read_link(proc_dir.join(name))
            .map(|target| target.to_string_lossy().starts_with(prefix))
            .unwrap_or(false)
    };
    if link_holds("exe") || link_holds("cwd") {
        return true;
    }
    if let Ok(fds) = fs::read_dir(proc_dir.join("fd")) {
        for fd in fds.flatten() {
            if fs::read_link(fd.path())
                .map(|target| target.to_string_lossy().starts_with(prefix))
                .unwrap_or(false)
            {
                return true;
            }
        }
    }
    fs::read_to_string(proc_dir.join("maps"))
        .map(|maps| maps.contains(prefix))
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::{PixiHostInfo, holding_processes, resolve_host_platform, resolve_kernel_platform};
    use std::fs;
    use std::os::unix::process::CommandExt;
    use std::process::{Command, Stdio};

    #[test]
    fn runtime_host_resolution_supports_both_release_architectures() -> Result<(), String> {
        for (machine, pixi_platform, expected) in [
            ("x86_64", "linux-64", "linux-x86_64"),
            ("aarch64", "linux-aarch64", "linux-aarch64"),
        ] {
            let pixi = PixiHostInfo {
                platform: pixi_platform.to_owned(),
                virtual_packages: vec!["__glibc=2.35=0".to_owned()],
            };
            let kernel_platform =
                resolve_kernel_platform("Linux", machine).map_err(|error| error.to_string())?;
            assert_eq!(
                resolve_host_platform("Linux", machine, kernel_platform, &pixi)
                    .map_err(|error| error.to_string())?,
                expected
            );
        }
        Ok(())
    }

    #[test]
    fn runtime_host_resolution_rejects_kernel_and_pixi_architecture_drift() -> Result<(), String> {
        let pixi = PixiHostInfo {
            platform: "linux-aarch64".to_owned(),
            virtual_packages: vec!["__glibc=2.35=0".to_owned()],
        };

        let kernel_platform =
            resolve_kernel_platform("Linux", "x86_64").map_err(|error| error.to_string())?;
        let error = match resolve_host_platform("Linux", "x86_64", kernel_platform, &pixi) {
            Ok(platform) => return Err(format!("mismatched runtime facts admitted {platform}")),
            Err(error) => error,
        };

        assert!(error.to_string().contains("kernel=Linux/x86_64"));
        assert!(error.to_string().contains("pixi-platform=linux-aarch64"));
        Ok(())
    }

    #[test]
    fn runtime_host_resolution_reports_unsupported_kernel_facts() -> Result<(), String> {
        let error = match resolve_kernel_platform("Darwin", "arm64") {
            Ok((platform, _)) => return Err(format!("unsupported kernel admitted {platform}")),
            Err(error) => error,
        };

        assert!(error.to_string().contains("kernel=Darwin/arm64"));
        Ok(())
    }

    #[test]
    fn holders_are_named_by_pid_and_comm() -> Result<(), String> {
        let dir = std::env::temp_dir().join(format!("inferlab-holders-{}", std::process::id()));
        fs::create_dir_all(&dir).map_err(|error| error.to_string())?;
        let mut child = Command::new("sleep")
            .arg("60")
            .current_dir(&dir)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .process_group(0)
            .spawn()
            .map_err(|error| error.to_string())?;
        let pid = child.id();
        let holders = holding_processes(&dir);
        let _ = Command::new("kill")
            .args(["-KILL", "--", &format!("-{pid}")])
            .status();
        let _ = child.wait();
        let _ = fs::remove_dir_all(&dir);
        if !holders.iter().any(|h| h.starts_with(&format!("{pid} "))) {
            return Err(format!(
                "holder scan missed pid {pid} with cwd under the path: {holders:?}"
            ));
        }
        Ok(())
    }
}
