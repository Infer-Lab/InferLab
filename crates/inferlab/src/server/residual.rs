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
    #[error(
        "machine {machine:?} reports device memory held by a process it cannot identify: {row:?}"
    )]
    Unattributed { machine: String, row: String },
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
/// login banners out of the parsed rows; the leading newline keeps a banner
/// printed without one off the first row. Per-device `-i` attribution is what
/// lets the evidence name the device that still holds memory. Each row also
/// says whether its process still lives on the launch machine: only memory
/// of processes that no longer exist (or are zombies) is the stopped server's
/// residual ([[RFC-0005:C-EVIDENCE]]). Liveness comes from `kill -0`, which,
/// unlike `/proc`, also sees another user's process on a host that hides
/// them (`hidepid`): success or a permission refusal means it exists. A row
/// without a numeric PID cannot be attributed.
fn residual_script(device: u32) -> String {
    format!(
        "set -eu; printf '\\n'; out=$(nvidia-smi -i {device} \
         --query-compute-apps=pid,used_memory --format=csv,noheader,nounits); \
         if [ -n \"$out\" ]; then printf '%s\\n' \"$out\" | while IFS=', ' read -r pid mem; \
         do case $pid in ''|*[!0-9]*) live=unknown ;; \
         *) if kill -0 \"$pid\" 2>/dev/null || LC_ALL=C kill -0 \"$pid\" 2>&1 | grep -q 'not permitted'; \
         then state=$(sed 's/.*) //' \"/proc/$pid/stat\" 2>/dev/null | cut -d' ' -f1); \
         if [ \"$state\" = Z ]; then live=0; else live=1; fi; else live=0; fi ;; esac; \
         printf '{RESIDUAL_MARKER}%s\\t%s\\t%s\\n' \"$pid\" \"$mem\" \"$live\"; done; fi"
    )
}

/// nvidia-smi reports one row in MiB per compute application holding device
/// memory; the probe tags each with whether its process still lives. Only
/// rows of processes that no longer exist count: a live process is not the
/// verified-gone server's. Memory no process can be named for cannot be
/// attributed either way, so it leaves the device's outcome unavailable. No
/// applications is empty output — the freed case.
fn parse_residual_rows(machine: &str, stdout: &str) -> Result<u64, ResidualProbeError> {
    let mut total = 0_u64;
    for line in stdout.lines() {
        let Some(row) = line.strip_prefix(RESIDUAL_MARKER) else {
            continue;
        };
        let mut fields = row.split('\t');
        let (_pid, memory, live) = (fields.next(), fields.next(), fields.next());
        match live.map(str::trim) {
            Some("1") => continue,
            Some("unknown") => {
                return Err(ResidualProbeError::Unattributed {
                    machine: machine.to_owned(),
                    row: row.to_owned(),
                });
            }
            _ => {}
        }
        let mib = memory
            .unwrap_or_default()
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
    use super::{RESIDUAL_MARKER, ResidualProbeError, parse_residual_rows, residual_script};
    use std::os::unix::fs::PermissionsExt;
    use std::process::Command;

    #[test]
    fn only_rows_of_processes_that_no_longer_exist_count() -> Result<(), String> {
        let stdout = format!(
            "login banner\n\
             {RESIDUAL_MARKER}4194304\t512\t0\n\
             {RESIDUAL_MARKER}1\t4096\t1\n"
        );
        let total = parse_residual_rows("node-a", &stdout).map_err(|error| error.to_string())?;
        assert_eq!(total, 512 * 1_048_576);
        Ok(())
    }

    #[test]
    fn empty_output_is_freed_and_a_non_numeric_row_is_loud() {
        let freed = parse_residual_rows("node-a", "login banner only\n");
        assert_eq!(freed.ok(), Some(0));
        let malformed =
            parse_residual_rows("node-a", "login\nINFERLAB_DEVICE_RESIDUAL\t7\tn/a\t0\n");
        assert!(
            malformed
                .as_ref()
                .is_err_and(|error| error.to_string().contains("non-numeric")),
            "{malformed:?}"
        );
    }

    /// Run the real probe script against a fake nvidia-smi that reports the
    /// given `pid, MiB` rows, behind a login banner without a newline.
    fn probe_rows(
        rows: &[String],
    ) -> Result<Result<u64, ResidualProbeError>, Box<dyn std::error::Error>> {
        let bin = tempfile::tempdir()?;
        let fake = bin.path().join("nvidia-smi");
        std::fs::write(
            &fake,
            format!(
                "#!/bin/sh\nprintf '%s\\n' {}\n",
                rows.iter()
                    .map(|row| format!("'{row}'"))
                    .collect::<Vec<_>>()
                    .join(" ")
            ),
        )?;
        std::fs::set_permissions(&fake, std::fs::Permissions::from_mode(0o755))?;
        let path = format!("{}:{}", bin.path().display(), std::env::var("PATH")?);
        let output = Command::new("sh")
            .args([
                "-c",
                &format!("printf 'login banner'; {}", residual_script(0)),
            ])
            .env("PATH", path)
            .output()?;
        assert!(output.status.success(), "{output:?}");
        Ok(parse_residual_rows(
            "node-a",
            &String::from_utf8_lossy(&output.stdout),
        ))
    }

    #[test]
    fn the_probe_counts_gone_and_zombie_processes_but_not_live_ones()
    -> Result<(), Box<dyn std::error::Error>> {
        // A spawned child that exits and is not yet reaped is a zombie.
        let mut zombie = Command::new("sh").args(["-c", "exit 0"]).spawn()?;
        let zombie_pid = zombie.id();
        let stat = format!("/proc/{zombie_pid}/stat");
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while !std::fs::read_to_string(&stat)?.contains(") Z ") {
            if std::time::Instant::now() > deadline {
                return Err("the child never became a zombie".into());
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        let rows = [
            // PID_MAX_LIMIT: never a live process. First, so a banner printed
            // without a newline would swallow it.
            "4194304, 512".to_owned(),
            // This test process: live and visible.
            format!("{}, 4096", std::process::id()),
            // PID 1: live; owned by another user unless the tests run as root.
            "1, 2048".to_owned(),
            format!("{zombie_pid}, 256"),
        ];
        let total = probe_rows(&rows)?.map_err(|error| error.to_string())?;
        zombie.wait()?;
        assert_eq!(total, (512 + 256) * 1_048_576);
        Ok(())
    }

    #[test]
    fn held_memory_of_an_unidentified_process_is_not_attributed()
    -> Result<(), Box<dyn std::error::Error>> {
        let unattributed = probe_rows(&["[N/A], 1024".to_owned()])?;
        assert!(
            matches!(&unattributed, Err(ResidualProbeError::Unattributed { machine, .. }) if machine == "node-a"),
            "{unattributed:?}"
        );
        Ok(())
    }
}
