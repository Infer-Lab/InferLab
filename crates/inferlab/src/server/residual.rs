//! Post-cleanup device residual probes ([[RFC-0005:C-EVIDENCE]]): a verified
//! process cleanup proves the serving processes are gone, but a dead
//! framework can leave compute memory held on its assigned devices. Each
//! assigned device is probed through the same launch path the preflight
//! hardware probe already uses, and the per-device outcome joins the cleanup
//! evidence.

use inferlab_runtime::container::{BoundedError, BoundedWait, run_cleanup_with_bound};
use inferlab_runtime::operation_bound::OperationBound;
use inferlab_runtime::plan::LaunchPlan;
use inferlab_runtime::server::{DeviceResidualEvidence, SystemProcessRuntime};
use inferlab_runtime::ssh::{SSH_ENV_REMOVE, ssh_argv};
use std::process::ExitStatus;
use std::time::Duration;

const RESIDUAL_MARKER: &str = "INFERLAB_DEVICE_RESIDUAL\t";

/// How long held devices are probed again after process cleanup: the driver
/// releases a killed process's device memory after the process has exited,
/// so a held reading right after cleanup is not yet a leak. Every probe
/// consumes this window, so a wedged driver or host cannot hang cleanup.
pub(super) const RESIDUAL_SETTLE_WINDOW: Duration = Duration::from_secs(30);
pub(super) const RESIDUAL_SETTLE_INTERVAL: Duration = Duration::from_secs(1);

pub(super) trait ResidualProbe {
    /// Residual compute memory in bytes on one assigned device, probed on its
    /// launch machine after process cleanup within the settle window.
    fn residual_bytes(
        &self,
        launch: &LaunchPlan,
        machine: &str,
        device: u32,
        bound: &OperationBound,
    ) -> Result<u64, ResidualProbeError>;
}

#[derive(Debug, thiserror::Error)]
pub(super) enum ResidualProbeError {
    #[error("failed to run the device residual probe on machine {machine:?}: {source}")]
    Command {
        machine: String,
        #[source]
        source: std::io::Error,
    },
    #[error(
        "device residual probe on machine {machine:?} did not answer within the settle window ({elapsed_ms} ms)"
    )]
    Expired { machine: String, elapsed_ms: u64 },
    #[error("device residual probe on machine {machine:?} exited with {status}: {stderr}")]
    Exit {
        machine: String,
        status: ExitStatus,
        stderr: String,
    },
    #[error("machine {machine:?} returned a non-numeric residual memory row {row:?}: {source}")]
    MalformedRow {
        machine: String,
        row: String,
        #[source]
        source: std::num::ParseIntError,
    },
}

impl ResidualProbe for SystemProcessRuntime {
    fn residual_bytes(
        &self,
        launch: &LaunchPlan,
        machine: &str,
        device: u32,
        bound: &OperationBound,
    ) -> Result<u64, ResidualProbeError> {
        let script = residual_script(device);
        let (argv, env_remove) = match launch {
            LaunchPlan::Local => (vec!["sh".to_owned(), "-c".to_owned(), script], &[][..]),
            LaunchPlan::Ssh { target } => (ssh_argv(target, &script), SSH_ENV_REMOVE),
        };
        let command_error = |source| ResidualProbeError::Command {
            machine: machine.to_owned(),
            source,
        };
        let (status, stdout, stderr) =
            match run_cleanup_with_bound(&argv, env_remove, None, None, bound, None) {
                Ok(BoundedWait::Exited {
                    status,
                    stdout,
                    stderr,
                }) => (status, stdout, stderr),
                Ok(BoundedWait::Expired {
                    operation_elapsed_ms,
                    ..
                }) => {
                    return Err(ResidualProbeError::Expired {
                        machine: machine.to_owned(),
                        elapsed_ms: operation_elapsed_ms,
                    });
                }
                Ok(BoundedWait::Interrupted { kill, .. }) => {
                    kill.map_err(command_error)?;
                    return Err(command_error(std::io::Error::other(
                        "interrupted during cleanup",
                    )));
                }
                Err(
                    BoundedError::Launch(source)
                    | BoundedError::Stdin(source)
                    | BoundedError::Wait(source)
                    | BoundedError::WaitCleanup { source, .. },
                ) => return Err(command_error(source)),
            };
        if !status.success() {
            return Err(ResidualProbeError::Exit {
                machine: machine.to_owned(),
                status,
                stderr: String::from_utf8_lossy(&stderr).trim().to_owned(),
            });
        }
        parse_residual_rows(machine, &String::from_utf8_lossy(&stdout))
    }
}

/// One probe script for both launch paths, mirroring the preflight hardware
/// probe: the command substitution keeps nvidia-smi's exit status
/// authoritative (a pipe would mask it), and the marker prefix keeps SSH
/// login banners out of the parsed rows. Per-device `-i` attribution is what
/// lets the evidence name the device that still holds memory.
fn residual_script(device: u32) -> String {
    format!(
        "set -eu; out=$(nvidia-smi -i {device} \
         --query-compute-apps=used_memory --format=csv,noheader,nounits); \
         if [ -n \"$out\" ]; then printf '%s\\n' \"$out\" | while IFS= read -r line; \
         do printf '{RESIDUAL_MARKER}%s\\n' \"$line\"; done; fi"
    )
}

/// nvidia-smi reports one used-memory row in MiB per surviving compute
/// application; no applications is empty output — the freed case.
fn parse_residual_rows(machine: &str, stdout: &str) -> Result<u64, ResidualProbeError> {
    let mut total = 0_u64;
    for line in stdout.lines() {
        let Some(row) = line.strip_prefix(RESIDUAL_MARKER) else {
            continue;
        };
        let mib = row
            .trim()
            .parse::<u64>()
            .map_err(|source| ResidualProbeError::MalformedRow {
                machine: machine.to_owned(),
                row: row.to_owned(),
                source,
            })?;
        total = total.saturating_add(mib.saturating_mul(1_048_576));
    }
    Ok(total)
}

pub(super) fn residual_held(evidence: &DeviceResidualEvidence) -> bool {
    matches!(evidence, DeviceResidualEvidence::ResidualHeld { .. })
}

/// The evidence outcome for one probed device: a probe failure records
/// probe-unavailable — it must not fail or unverify the cleanup.
pub(super) fn probe_device_residual<R: ResidualProbe>(
    runtime: &R,
    launch: &LaunchPlan,
    machine: &str,
    device: u32,
    bound: &OperationBound,
) -> DeviceResidualEvidence {
    let machine = machine.to_owned();
    match runtime.residual_bytes(launch, &machine, device, bound) {
        Ok(0) => DeviceResidualEvidence::Freed { machine, device },
        Ok(bytes) => DeviceResidualEvidence::ResidualHeld {
            machine,
            device,
            bytes,
        },
        Err(error) => DeviceResidualEvidence::ProbeUnavailable {
            machine,
            device,
            reason: error.to_string(),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::{RESIDUAL_MARKER, parse_residual_rows};

    #[test]
    fn residual_rows_sum_through_banner_noise() -> Result<(), String> {
        let stdout = format!(
            "login banner\n\
             {RESIDUAL_MARKER}512\n\
             {RESIDUAL_MARKER} 1024 \n"
        );
        let total = parse_residual_rows("node-a", &stdout).map_err(|error| error.to_string())?;
        assert_eq!(total, (512 + 1024) * 1_048_576);
        Ok(())
    }

    #[test]
    fn empty_output_is_freed_and_a_non_numeric_row_is_loud() {
        let freed = parse_residual_rows("node-a", "login banner only\n");
        assert_eq!(freed.ok(), Some(0));
        let malformed = parse_residual_rows("node-a", "login\nINFERLAB_DEVICE_RESIDUAL\tn/a\n");
        assert!(
            malformed
                .as_ref()
                .is_err_and(|error| error.to_string().contains("non-numeric")),
            "{malformed:?}"
        );
    }
}
