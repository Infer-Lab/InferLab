use super::record::{DeviceHardwareEvidence, MachineHardwareEvidence};
use crate::execution::{ProcessPlan, RemoteWorkspacePlan};
use crate::workspace::{
    WorkspaceSnapshot, git_status_flags, source_digest_script, source_pathspecs,
};
use inferlab_runtime::plan::LaunchPlan;
use inferlab_runtime::shell::{shell_quote, shell_quote_path};
use inferlab_runtime::ssh::ssh_output;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

const PREFLIGHT_MARKER: &str = "INFERLAB_PREFLIGHT\t";
const HARDWARE_MARKER: &str = "INFERLAB_HARDWARE\t";

pub(super) struct RemoteCheckRequest<'a> {
    pub target: &'a str,
    pub root: &'a Path,
    pub pixi: &'a str,
    pub pixi_environment: &'a str,
    pub checks: &'a [crate::environment::PlannedEnvironmentCheck],
    pub machine: &'a str,
    pub progress: &'a crate::progress::Progress,
}

#[derive(Debug, thiserror::Error)]
pub(super) enum RemoteCheckError {
    #[error("process {process:?} has no executable for remote environment checks")]
    MissingExecutable { process: String },
    #[error("failed to run environment check {check:?} on machine {machine:?}: {source}")]
    Ssh {
        machine: String,
        check: String,
        #[source]
        source: inferlab_runtime::ssh::SshError,
    },
}

pub(super) type RemoteCheckOutcome = Result<
    (
        Vec<crate::environment::EnvironmentCheckEvidence>,
        Option<crate::environment::LocalCheckFailure>,
    ),
    RemoteCheckError,
>;

pub(super) trait PreflightObserver {
    /// Probe the device hardware assigned on one machine through its launch
    /// path, before any serving process starts ([[RFC-0005:C-EVIDENCE]]).
    fn probe_hardware(
        &self,
        launch: &LaunchPlan,
        machine: &str,
        devices: &[u32],
    ) -> Result<MachineHardwareEvidence, HardwareProbeError>;

    fn run_remote_checks(&self, request: RemoteCheckRequest<'_>) -> RemoteCheckOutcome;
}

#[derive(Debug, thiserror::Error)]
pub(crate) enum RemotePreflightError {
    #[error("failed to probe machine {machine:?}: {source}")]
    Ssh {
        machine: String,
        #[source]
        source: inferlab_runtime::ssh::SshError,
    },
    #[error("machine {machine:?} ({target}) returned non-UTF-8 {operation} output: {source}")]
    NonUtf8 {
        machine: String,
        target: String,
        operation: &'static str,
        #[source]
        source: std::string::FromUtf8Error,
    },
    #[error(
        "machine {machine:?} ({target}) exited with {status} before returning {operation} evidence: {stderr}"
    )]
    MissingEvidence {
        machine: String,
        target: String,
        operation: &'static str,
        status: std::process::ExitStatus,
        stderr: String,
    },
    #[error(
        "machine {machine:?} ({target}) returned malformed container preflight evidence: {evidence:?}"
    )]
    MalformedContainerEvidence {
        machine: String,
        target: String,
        evidence: String,
    },
    #[error(
        "machine {machine:?} ({target}) workspace {path} does not match the controller workspace"
    )]
    WorkspaceMismatch {
        machine: String,
        target: String,
        path: PathBuf,
    },
    #[error(transparent)]
    EnvironmentUnavailable(Box<RemoteEnvironmentUnavailable>),
    #[error(
        "machine {machine:?} ({target}) does not hold external image {external_id:?} ({reference}); run on that machine: docker pull {reference}"
    )]
    ImageMissing {
        machine: String,
        target: String,
        external_id: String,
        reference: String,
    },
    #[error("remote preflight did not resolve machine {machine:?}")]
    MissingMachine { machine: String },
    #[error("process {process:?} has no executable")]
    MissingExecutable { process: String },
}

#[derive(Debug, thiserror::Error)]
#[error(
    "machine {machine:?} ({target}) does not have locked Pixi environment {environment:?} materialized in {path}; run `cd {path} && {pixi} install --locked --environment {environment}`: {stderr}",
    path = path.display(),
    pixi = pixi.display()
)]
pub(crate) struct RemoteEnvironmentUnavailable {
    pub(super) machine: String,
    pub(super) target: String,
    pub(super) environment: String,
    pub(super) path: PathBuf,
    pub(super) pixi: PathBuf,
    pub(super) stderr: String,
}

#[derive(Debug, thiserror::Error)]
pub(super) enum HardwareProbeError {
    #[error("failed to launch the hardware probe on machine {machine:?}: {source}")]
    LocalLaunch {
        machine: String,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to launch the hardware probe on machine {machine:?}: {source}")]
    Ssh {
        machine: String,
        #[source]
        source: inferlab_runtime::ssh::SshError,
    },
    #[error("hardware probe on machine {machine:?} exited with {status}: {stderr}")]
    Exit {
        machine: String,
        status: std::process::ExitStatus,
        stderr: String,
    },
    #[error("machine {machine:?} returned an unexpected probe row {row:?}")]
    UnexpectedRow { machine: String, row: String },
    #[error("machine {machine:?} probe row index {value:?}: {source}")]
    InvalidIndex {
        machine: String,
        value: String,
        #[source]
        source: std::num::ParseIntError,
    },
    #[error("machine {machine:?} probe row memory {value:?}: {source}")]
    InvalidMemory {
        machine: String,
        value: String,
        #[source]
        source: std::num::ParseIntError,
    },
    #[error("machine {machine:?} returned no probe rows for devices {devices:?}")]
    MissingRows { machine: String, devices: Vec<u32> },
    #[error(
        "machine {machine:?} probe covered devices {probed:?} but the placement assigns {assigned:?}"
    )]
    CoverageMismatch {
        machine: String,
        probed: Vec<u32>,
        assigned: Vec<u32>,
    },
}

impl PreflightObserver for inferlab_runtime::server::SystemProcessRuntime {
    fn probe_hardware(
        &self,
        launch: &LaunchPlan,
        machine: &str,
        devices: &[u32],
    ) -> Result<MachineHardwareEvidence, HardwareProbeError> {
        let script = nvidia_smi_script(devices);
        let output = match launch {
            LaunchPlan::Local => Command::new("sh")
                .args(["-c", &script])
                .stdin(Stdio::null())
                .output()
                .map_err(|source| HardwareProbeError::LocalLaunch {
                    machine: machine.to_owned(),
                    source,
                })?,
            LaunchPlan::Ssh { target } => {
                ssh_output(target, &script).map_err(|source| HardwareProbeError::Ssh {
                    machine: machine.to_owned(),
                    source,
                })?
            }
        };
        if !output.status.success() {
            return Err(HardwareProbeError::Exit {
                machine: machine.to_owned(),
                status: output.status,
                stderr: String::from_utf8_lossy(&output.stderr).trim().to_owned(),
            });
        }
        parse_hardware_output(machine, devices, &String::from_utf8_lossy(&output.stdout))
    }

    fn run_remote_checks(&self, request: RemoteCheckRequest<'_>) -> RemoteCheckOutcome {
        run_remote_checks(
            request.target,
            request.root,
            request.pixi,
            request.pixi_environment,
            request.checks,
            request.machine,
            request.progress,
        )
    }
}

/// One probe script for both launch paths: the command substitution keeps
/// nvidia-smi's exit status authoritative (a pipe would mask it), and the
/// marker prefix keeps SSH login banners out of the parsed rows.
fn nvidia_smi_script(devices: &[u32]) -> String {
    let select = if devices.is_empty() {
        String::new()
    } else {
        format!(
            " -i {}",
            devices
                .iter()
                .map(u32::to_string)
                .collect::<Vec<_>>()
                .join(",")
        )
    };
    format!(
        "set -eu; out=$(nvidia-smi{select} \
         --query-gpu=index,name,memory.total,uuid,driver_version \
         --format=csv,noheader,nounits); \
         printf '%s\\n' \"$out\" | while IFS= read -r line; \
         do printf 'INFERLAB_HARDWARE\\t%s\\n' \"$line\"; done"
    )
}

fn parse_hardware_output(
    machine: &str,
    assigned_devices: &[u32],
    stdout: &str,
) -> Result<MachineHardwareEvidence, HardwareProbeError> {
    let mut observed_devices = Vec::new();
    let mut driver_version: Option<String> = None;
    for line in stdout.lines() {
        let Some(row) = line.strip_prefix(HARDWARE_MARKER) else {
            continue;
        };
        let fields = row.split(", ").collect::<Vec<_>>();
        let [index, model, memory, uuid, driver] = fields.as_slice() else {
            return Err(HardwareProbeError::UnexpectedRow {
                machine: machine.to_owned(),
                row: row.to_owned(),
            });
        };
        let index =
            index
                .trim()
                .parse::<u32>()
                .map_err(|source| HardwareProbeError::InvalidIndex {
                    machine: machine.to_owned(),
                    value: (*index).to_owned(),
                    source,
                })?;
        let memory_total_mib =
            memory
                .trim()
                .parse::<u64>()
                .map_err(|source| HardwareProbeError::InvalidMemory {
                    machine: machine.to_owned(),
                    value: (*memory).to_owned(),
                    source,
                })?;
        driver_version.get_or_insert_with(|| driver.trim().to_owned());
        observed_devices.push(DeviceHardwareEvidence {
            index,
            model: model.trim().to_owned(),
            memory_total_mib,
            uuid: uuid.trim().to_owned(),
        });
    }
    let Some(driver_version) = driver_version else {
        return Err(HardwareProbeError::MissingRows {
            machine: machine.to_owned(),
            devices: assigned_devices.to_vec(),
        });
    };
    observed_devices.sort_by_key(|device| device.index);
    if assigned_devices.is_empty() {
        // A machine hosting only zero-device processes (a proxy-only host)
        // assigns no devices: the probe proves the driver is present, and
        // recording the machine's full inventory would over-claim it as
        // assigned ([[RFC-0005:C-EVIDENCE]]).
        observed_devices.clear();
    } else {
        let probed = observed_devices
            .iter()
            .map(|device| device.index)
            .collect::<Vec<_>>();
        let mut requested = assigned_devices.to_vec();
        requested.sort_unstable();
        requested.dedup();
        if probed != requested {
            return Err(HardwareProbeError::CoverageMismatch {
                machine: machine.to_owned(),
                probed,
                assigned: requested,
            });
        }
    }
    Ok(MachineHardwareEvidence {
        driver_version,
        devices: observed_devices,
    })
}

pub(super) fn preflight_targets(
    processes: &mut [ProcessPlan],
    workspace: &WorkspaceSnapshot,
    pixi_environment: &str,
) -> Result<BTreeMap<String, RemoteWorkspacePlan>, RemotePreflightError> {
    let mut machines = BTreeMap::new();
    for process in &*processes {
        if let LaunchPlan::Ssh { target } = &process.launch {
            machines
                .entry(process.machine.clone())
                .or_insert_with(|| (target.clone(), process.command.cwd.clone()));
        }
    }

    let mut remote_workspaces = BTreeMap::new();
    for (machine, (target, cwd)) in machines {
        let mut root = cwd;
        root.pop();
        let source_digest = source_digest_script(&workspace.source_exclusions);
        let source_pathspecs = source_pathspecs(&workspace.source_exclusions);
        let script = format!(
            "set -eu; cd {root}; pixi=$(type -P pixi); revision=$(git rev-parse HEAD); dirty=0; test -z \"$(git status {status_flags} -- {source_pathspecs})\" || dirty=1; source_digest=$({source_digest}); manifest=$(sha256sum pixi.toml | awk '{{print $1}}'); lock=$(sha256sum pixi.lock | awk '{{print $1}}'); marker={confirmation_cache_dir}/{environment}/confirmed; set +e; if test -d {pixi_envs_dir}/{environment} && test -f \"$marker\" && [ \"$(sed -n 1p \"$marker\" 2>/dev/null)\" = \"$manifest\" ] && [ \"$(sed -n 2p \"$marker\" 2>/dev/null)\" = \"$lock\" ]; then pixi_status=0; else test -d {pixi_envs_dir}/{environment} && \"$pixi\" run --locked --no-install --executable -e {environment} -- true; pixi_status=$?; if [ \"$pixi_status\" = 0 ]; then mkdir -p \"$(dirname \"$marker\")\" && printf '%s\\n%s\\n' \"$manifest\" \"$lock\" > \"$marker.tmp.$$\" && mv \"$marker.tmp.$$\" \"$marker\"; fi; fi; set -e; printf 'INFERLAB_PREFLIGHT\\t%s\\t%s\\t%s\\t%s\\t%s\\t%s\\t%s\\t%s\\t%s\\n' \"$revision\" \"$dirty\" \"$source_digest\" \"$manifest\" \"$lock\" \"$pixi\" \"$PATH\" \"$HOME\" \"$pixi_status\"; exit \"$pixi_status\"",
            root = shell_quote_path(&root),
            status_flags = git_status_flags(),
            environment = shell_quote(pixi_environment),
            pixi_envs_dir = crate::environment::PIXI_ENVS_DIR,
            confirmation_cache_dir = crate::environment::CONFIRMATION_CACHE_DIR,
        );
        let output = ssh_output(&target, &script).map_err(|source| RemotePreflightError::Ssh {
            machine: machine.clone(),
            source,
        })?;
        let stdout =
            String::from_utf8(output.stdout).map_err(|source| RemotePreflightError::NonUtf8 {
                machine: machine.clone(),
                target: target.clone(),
                operation: "workspace preflight",
                source,
            })?;
        let Some(observed) = parse_preflight_output(&stdout) else {
            return Err(RemotePreflightError::MissingEvidence {
                machine,
                target,
                operation: "workspace preflight",
                status: output.status,
                stderr: String::from_utf8_lossy(&output.stderr).trim().to_owned(),
            });
        };
        if observed.revision != workspace.revision
            || observed.dirty != workspace.dirty
            || observed.source_digest != workspace.source_digest
            || observed.pixi_manifest_sha256 != workspace.pixi_manifest_sha256
            || observed.pixi_lock_sha256 != workspace.pixi_lock_sha256
        {
            return Err(RemotePreflightError::WorkspaceMismatch {
                machine,
                target,
                path: root,
            });
        }
        if observed.pixi_status != 0 {
            return Err(RemotePreflightError::EnvironmentUnavailable(Box::new(
                RemoteEnvironmentUnavailable {
                    machine,
                    target,
                    environment: pixi_environment.to_owned(),
                    path: root,
                    pixi: observed.pixi_executable,
                    stderr: String::from_utf8_lossy(&output.stderr).trim().to_owned(),
                },
            )));
        }
        remote_workspaces.insert(
            machine,
            RemoteWorkspacePlan {
                target,
                path: root,
                revision: observed.revision,
                dirty: observed.dirty,
                source_digest: observed.source_digest,
                pixi_manifest_sha256: observed.pixi_manifest_sha256,
                pixi_lock_sha256: observed.pixi_lock_sha256,
                pixi_environment: pixi_environment.to_owned(),
                pixi_executable: observed.pixi_executable,
                environment: BTreeMap::from([
                    ("HOME".to_owned(), observed.home),
                    ("PATH".to_owned(), observed.path),
                ]),
            },
        );
    }

    for process in processes {
        if matches!(process.launch, LaunchPlan::Ssh { .. }) {
            let remote = remote_workspaces.get(&process.machine).ok_or_else(|| {
                RemotePreflightError::MissingMachine {
                    machine: process.machine.clone(),
                }
            })?;
            let executable = process.command.argv.first_mut().ok_or_else(|| {
                RemotePreflightError::MissingExecutable {
                    process: process.id.clone(),
                }
            })?;
            *executable = remote.pixi_executable.to_string_lossy().into_owned();
            process.command.env.extend(remote.environment.clone());
        }
    }
    Ok(remote_workspaces)
}

const CONTAINER_PREFLIGHT_MARKER: &str = "INFERLAB_CONTAINER_PREFLIGHT\t";

/// The remote preflight of a containerized launch
/// ([[RFC-0003:C-RUNTIME-WORKFLOWS]]): the image replaces the serving
/// environment, so no workspace realization is checked and no argv is
/// rewritten. One read-only probe per launch machine verifies the declared
/// image is present — a missing image is that machine's operator pull, never
/// Inferlab's — and gathers the machine-scoped launch facts the substitution
/// consumes: the container user identity and which declared pass-through
/// names that machine's launching environment actually holds.
pub(super) fn preflight_container_targets(
    processes: &mut [ProcessPlan],
    machines: &BTreeMap<String, crate::workspace::MachineBinding>,
    external_id: &str,
    reference: &str,
) -> Result<BTreeMap<String, crate::execution::RemoteContainerFacts>, RemotePreflightError> {
    let mut targets = BTreeMap::new();
    for process in &*processes {
        if let LaunchPlan::Ssh { target } = &process.launch {
            targets
                .entry(process.machine.clone())
                .or_insert_with(|| target.clone());
        }
    }
    let mut facts = BTreeMap::new();
    for (machine, target) in targets {
        // Pass-through names are load-validated bare identifiers, so they
        // embed into the probe script verbatim.
        let pass_env: Vec<String> = machines
            .get(&machine)
            .and_then(|binding| binding.container.as_ref())
            .map(|container| container.pass_env.clone())
            .unwrap_or_default();
        let env_probe: String = pass_env
            .iter()
            .map(|name| {
                format!("if [ -n \"${{{name}+x}}\" ]; then set_env=\"$set_env {name}\"; fi; ")
            })
            .collect();
        let script = format!(
            "set -eu; present=1; docker image inspect --format '{{{{.Id}}}}' {reference} \
             >/dev/null 2>&1 || present=0; set_env=\"\"; {env_probe}printf \
             'INFERLAB_CONTAINER_PREFLIGHT\\t%s\\t%s\\t%s\\t%s\\t%s\\t%s\\t%s\\n' \"$present\" \
             \"$(id -u)\" \"$(id -g)\" \"$(id -un)\" \"$PATH\" \"$HOME\" \"$set_env\"",
            reference = shell_quote(reference),
        );
        let output = ssh_output(&target, &script).map_err(|source| RemotePreflightError::Ssh {
            machine: machine.clone(),
            source,
        })?;
        let stdout =
            String::from_utf8(output.stdout).map_err(|source| RemotePreflightError::NonUtf8 {
                machine: machine.clone(),
                target: target.clone(),
                operation: "container preflight",
                source,
            })?;
        let Some(observed) = stdout
            .lines()
            .rev()
            .find_map(|line| line.strip_prefix(CONTAINER_PREFLIGHT_MARKER))
        else {
            return Err(RemotePreflightError::MissingEvidence {
                machine,
                target,
                operation: "container preflight",
                status: output.status,
                stderr: String::from_utf8_lossy(&output.stderr).trim().to_owned(),
            });
        };
        let mut fields = observed.split('\t');
        let (present, uid, gid, user, path, home, set_env) = (
            fields.next(),
            fields.next().and_then(|field| field.parse::<u32>().ok()),
            fields.next().and_then(|field| field.parse::<u32>().ok()),
            fields.next(),
            fields.next(),
            fields.next(),
            fields.next(),
        );
        let (
            Some(present),
            Some(uid),
            Some(gid),
            Some(user),
            Some(path),
            Some(home),
            Some(set_env),
        ) = (present, uid, gid, user, path, home, set_env)
        else {
            return Err(RemotePreflightError::MalformedContainerEvidence {
                machine,
                target,
                evidence: observed.to_owned(),
            });
        };
        if present != "1" {
            return Err(RemotePreflightError::ImageMissing {
                machine,
                target,
                external_id: external_id.to_owned(),
                reference: reference.to_owned(),
            });
        }
        facts.insert(
            machine,
            crate::execution::RemoteContainerFacts {
                target,
                user: user.to_owned(),
                uid,
                gid,
                present_pass_env: set_env.split_whitespace().map(str::to_owned).collect(),
                environment: BTreeMap::from([
                    ("HOME".to_owned(), home.to_owned()),
                    ("PATH".to_owned(), path.to_owned()),
                ]),
            },
        );
    }
    // Remote processes launch under a clean environment; the docker client
    // needs the machine's own PATH and HOME, exactly as remote host
    // processes receive them from the workspace preflight.
    for process in processes {
        if matches!(process.launch, LaunchPlan::Ssh { .. }) {
            let remote = facts.get(&process.machine).ok_or_else(|| {
                RemotePreflightError::MissingMachine {
                    machine: process.machine.clone(),
                }
            })?;
            process.command.env.extend(remote.environment.clone());
        }
    }
    Ok(facts)
}

/// Execute the declared environment checks against one remote machine's
/// workspace realization ([[RFC-0002:C-ENVIRONMENT-CHECKS]]): the remote
/// checkout carries the same committed scripts (the preflight already proved
/// revision equality), and its own Pixi environment runs them. Stops at the
/// first failure; evidence covers every check that executed.
pub(super) fn run_remote_checks(
    target: &str,
    root: &Path,
    pixi: &str,
    pixi_environment: &str,
    checks: &[crate::environment::PlannedEnvironmentCheck],
    machine: &str,
    progress: &crate::progress::Progress,
) -> Result<
    (
        Vec<crate::environment::EnvironmentCheckEvidence>,
        Option<crate::environment::LocalCheckFailure>,
    ),
    RemoteCheckError,
> {
    use crate::environment::{CheckOutcome, CheckRealization, EnvironmentCheckEvidence};
    let mut evidence = Vec::new();
    for (index, check) in checks.iter().enumerate() {
        let _ = progress.phase(
            crate::progress::Phase::named("local and remote preflight").item(
                format!("{machine}:{}", check.id),
                index + 1,
                checks.len(),
            ),
        );
        let script = format!(
            "cd {root} && {pixi} run --locked --no-install --executable -e {environment} -- \
             python {script}",
            root = shell_quote_path(root),
            pixi = shell_quote(pixi),
            environment = shell_quote(pixi_environment),
            script = shell_quote_path(&check.script),
        );
        let output = ssh_output(target, &script).map_err(|source| RemoteCheckError::Ssh {
            machine: machine.to_owned(),
            check: check.id.clone(),
            source,
        })?;
        let mut combined = String::from_utf8_lossy(&output.stdout).into_owned();
        let stderr = String::from_utf8_lossy(&output.stderr);
        if !stderr.trim().is_empty() {
            if !combined.is_empty() {
                combined.push('\n');
            }
            combined.push_str(stderr.trim_end());
        }
        let combined = crate::environment::tail(&combined, 4096);
        let passed = output.status.success();
        evidence.push(EnvironmentCheckEvidence {
            id: check.id.clone(),
            realization: CheckRealization::LocalWorkspace,
            machine: Some(machine.to_owned()),
            outcome: if passed {
                CheckOutcome::Passed
            } else {
                CheckOutcome::Failed
            },
            output: Some(combined.clone()),
            log: None,
        });
        if !passed {
            return Ok((
                evidence,
                Some(crate::environment::LocalCheckFailure {
                    id: check.id.clone(),
                    repair_hint: check.repair_hint.clone(),
                    output: combined,
                }),
            ));
        }
    }
    Ok((evidence, None))
}

struct PreflightOutput {
    revision: String,
    dirty: bool,
    source_digest: String,
    pixi_manifest_sha256: String,
    pixi_lock_sha256: String,
    pixi_executable: PathBuf,
    path: String,
    home: String,
    pixi_status: i32,
}

fn parse_preflight_output(output: &str) -> Option<PreflightOutput> {
    let result = output
        .lines()
        .rev()
        .find_map(|line| line.strip_prefix(PREFLIGHT_MARKER))?;
    let mut fields = result.split('\t');
    Some(PreflightOutput {
        revision: fields.next()?.to_owned(),
        dirty: fields.next()? == "1",
        source_digest: fields.next()?.to_owned(),
        pixi_manifest_sha256: fields.next()?.to_owned(),
        pixi_lock_sha256: fields.next()?.to_owned(),
        pixi_executable: PathBuf::from(fields.next()?),
        path: fields.next()?.to_owned(),
        home: fields.next()?.to_owned(),
        pixi_status: fields.next()?.parse().ok()?,
    })
}

#[cfg(test)]
mod tests {
    use super::parse_hardware_output;

    #[test]
    fn hardware_rows_parse_through_banner_noise_in_index_order() -> Result<(), String> {
        let stdout = "login banner\n\
                      INFERLAB_HARDWARE\t1, Fixture GPU, 97871, GPU-bbb, 580.65.06\n\
                      INFERLAB_HARDWARE\t0, Fixture GPU, 97871, GPU-aaa, 580.65.06\n";
        let evidence =
            parse_hardware_output("node-a", &[1, 0], stdout).map_err(|error| error.to_string())?;
        assert_eq!(evidence.driver_version, "580.65.06");
        let indices: Vec<u32> = evidence.devices.iter().map(|device| device.index).collect();
        assert_eq!(indices, [0, 1]);
        assert_eq!(evidence.devices[0].uuid, "GPU-aaa");
        assert_eq!(evidence.devices[0].model, "Fixture GPU");
        assert_eq!(evidence.devices[0].memory_total_mib, 97871);
        Ok(())
    }

    #[test]
    fn hardware_coverage_mismatch_and_empty_output_are_loud() {
        let one_row = "INFERLAB_HARDWARE\t0, Fixture GPU, 97871, GPU-aaa, 580.65.06\n";
        let mismatch = parse_hardware_output("node-a", &[0, 1], one_row);
        assert!(
            mismatch
                .as_ref()
                .is_err_and(|error| error.to_string().contains("assigns")),
            "{mismatch:?}"
        );
        let empty = parse_hardware_output("node-a", &[0], "login banner only\n");
        assert!(
            empty
                .as_ref()
                .is_err_and(|error| error.to_string().contains("no probe rows")),
            "{empty:?}"
        );
    }

    #[test]
    fn zero_assigned_devices_record_the_driver_without_claiming_inventory() -> Result<(), String> {
        // A proxy-only host enumerates its full inventory (no `-i`), but
        // nothing is assigned there, so no device may be recorded as assigned.
        let stdout = "INFERLAB_HARDWARE\t0, Fixture GPU, 97871, GPU-aaa, 580.65.06\n\
                      INFERLAB_HARDWARE\t1, Fixture GPU, 97871, GPU-bbb, 580.65.06\n";
        let evidence =
            parse_hardware_output("proxy-host", &[], stdout).map_err(|error| error.to_string())?;
        assert_eq!(evidence.driver_version, "580.65.06");
        assert!(evidence.devices.is_empty(), "{evidence:?}");
        Ok(())
    }
}
