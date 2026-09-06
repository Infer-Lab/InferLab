//! Post-cleanup device residual probes ([[RFC-0005:C-EVIDENCE]]): a verified
//! process cleanup proves the serving processes are gone, but a dead
//! framework can leave compute memory held on its assigned devices. Each
//! assigned device is probed through the same launch path the preflight
//! hardware probe already uses, and the per-device outcome joins the cleanup
//! evidence.

use inferlab_runtime::plan::LaunchPlan;
use inferlab_runtime::server::{DeviceResidualEvidence, SystemProcessRuntime};
use inferlab_runtime::ssh::ssh_output;
use std::process::{Command, Stdio};

const RESIDUAL_MARKER: &str = "INFERLAB_DEVICE_RESIDUAL\t";

pub(super) trait ResidualProbe {
    /// Residual compute memory in bytes on one assigned device, probed on its
    /// launch machine after process cleanup.
    fn residual_bytes(
        &self,
        launch: &LaunchPlan,
        machine: &str,
        device: u32,
    ) -> Result<u64, ResidualProbeError>;
}

#[derive(Debug, thiserror::Error)]
pub(super) enum ResidualProbeError {
    #[error("failed to launch the device residual probe on machine {machine:?}: {source}")]
    LocalLaunch {
        machine: String,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to launch the device residual probe on machine {machine:?}: {source}")]
    Ssh {
        machine: String,
        #[source]
        source: inferlab_runtime::ssh::SshError,
    },
    #[error("device residual probe on machine {machine:?} exited with {status}: {stderr}")]
    Exit {
        machine: String,
        status: std::process::ExitStatus,
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
    ) -> Result<u64, ResidualProbeError> {
        let script = residual_script(device);
        let output = match launch {
            LaunchPlan::Local => Command::new("sh")
                .args(["-c", &script])
                .stdin(Stdio::null())
                .output()
                .map_err(|source| ResidualProbeError::LocalLaunch {
                    machine: machine.to_owned(),
                    source,
                })?,
            LaunchPlan::Ssh { target } => {
                ssh_output(target, &script).map_err(|source| ResidualProbeError::Ssh {
                    machine: machine.to_owned(),
                    source,
                })?
            }
        };
        if !output.status.success() {
            return Err(ResidualProbeError::Exit {
                machine: machine.to_owned(),
                status: output.status,
                stderr: String::from_utf8_lossy(&output.stderr).trim().to_owned(),
            });
        }
        parse_residual_rows(machine, &String::from_utf8_lossy(&output.stdout))
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

/// The evidence outcome for one probed device: a probe failure records
/// probe-unavailable — it must not fail or unverify the cleanup.
pub(super) fn probe_device_residual<R: ResidualProbe>(
    runtime: &R,
    launch: &LaunchPlan,
    machine: &str,
    device: u32,
) -> DeviceResidualEvidence {
    let machine = machine.to_owned();
    match runtime.residual_bytes(launch, &machine, device) {
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
