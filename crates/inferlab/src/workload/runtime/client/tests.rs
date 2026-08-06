use super::super::{CLIENT_HANDLE_FILE, ClientGroupHandle, ClientRun};
use super::{accept_client_result, sweep_stale_client_groups};
use crate::record::RECORDS_DIR;
use inferlab_runtime::operation_bound::OperationBound;
use inferlab_runtime::process_group::{LocalProcessGroup, process_start_time};
use serde_json::Value;
use std::fs;
use std::os::unix::process::CommandExt;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::Duration;

fn sweep_fixture(tag: &str) -> Result<(PathBuf, PathBuf), String> {
    let root = std::env::temp_dir().join(format!("inferlab-sweep-{tag}-{}", std::process::id()));
    let case_dir = root.join(RECORDS_DIR).join("run").join("cases").join("c0");
    fs::create_dir_all(&case_dir).map_err(|error| error.to_string())?;
    Ok((root, case_dir.join(CLIENT_HANDLE_FILE)))
}

fn spawn_survivor() -> Result<std::process::Child, String> {
    Command::new("sleep")
        .arg("60")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .process_group(0)
        .spawn()
        .map_err(|error| error.to_string())
}

fn write_handle(path: &PathBuf, pid: u32, ticks: u64, owner: (u32, u64)) -> Result<(), String> {
    let handle = ClientGroupHandle {
        group: LocalProcessGroup::new(pid, pid, ticks).map_err(|error| error.to_string())?,
        owner_pid: owner.0,
        owner_start_time_ticks: owner.1,
    };
    let bytes = serde_json::to_vec(&handle).map_err(|error| error.to_string())?;
    fs::write(path, bytes).map_err(|error| error.to_string())
}

/// An owner identity that can never match a live process.
const DEAD_OWNER: (u32, u64) = (u32::MAX, 1);

fn own_identity() -> Result<(u32, u64), String> {
    let pid = std::process::id();
    let ticks = process_start_time(pid)
        .map_err(|error| error.to_string())?
        .ok_or("own identity unreadable")?;
    Ok((pid, ticks))
}

fn group_alive(pid: u32) -> Result<bool, String> {
    let ticks = process_start_time(pid)
        .map_err(|error| error.to_string())?
        .unwrap_or_default();
    let group = LocalProcessGroup::new(pid, pid, ticks).map_err(|error| error.to_string())?;
    let bound = OperationBound::finite(Duration::from_secs(2));
    group
        .has_live_members(&bound)
        .map_err(|error| error.to_string())
}

#[test]
fn result_decode_cannot_accept_after_the_owner_deadline() -> Result<(), String> {
    let path = std::env::temp_dir().join(format!(
        "inferlab-late-client-result-{}.json",
        std::process::id()
    ));
    fs::write(&path, br#"{"schema_version":1}"#).map_err(|error| error.to_string())?;
    let bound = OperationBound::finite(Duration::ZERO);
    let accepted = accept_client_result::<Value>(
        &path,
        "fixture client",
        ClientRun {
            process: None,
            error: None,
            pending_cleanup: None,
            terminal_timing: None,
        },
        &bound,
    );
    let _ = fs::remove_file(path);

    if accepted.result.is_some() {
        return Err("late client result was accepted".to_owned());
    }
    if !accepted
        .decode_error
        .as_deref()
        .is_some_and(|error| error.contains("measurement-case deadline"))
    {
        return Err("late client result did not preserve deadline rejection".to_owned());
    }
    Ok(())
}

#[test]
fn termination_covers_the_whole_process_group() -> Result<(), String> {
    // A client whose group contains its own descendants: the leader
    // spawns a grandchild and both share the group created at launch.
    let mut child = Command::new("sh")
        .args(["-c", "sleep 60 & exec sleep 60"])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .process_group(0)
        .spawn()
        .map_err(|error| error.to_string())?;
    let pid = child.id();
    let group = LocalProcessGroup::capture_child(&child).map_err(|error| error.to_string())?;
    let evidence = super::terminate_client_group(
        &mut child,
        group,
        super::ClientTerminationTrigger::ResultAccepted,
    );
    let alive = group_alive(pid)?;
    let _ = child.wait();
    if !evidence.verified {
        return Err("group termination was not verified".to_owned());
    }
    if evidence.trigger != super::ClientTerminationTrigger::ResultAccepted {
        return Err("client cleanup did not record its trigger".to_owned());
    }
    if evidence.term_grace_ms != 2_000 || evidence.kill_grace_ms != 2_000 {
        return Err("client cleanup did not record its independent graces".to_owned());
    }
    if alive {
        return Err("descendants survived group termination".to_owned());
    }
    Ok(())
}

#[test]
fn sweep_skips_live_owners_clients() -> Result<(), String> {
    let (root, handle_path) = sweep_fixture("owner")?;
    let mut child = spawn_survivor()?;
    let pid = child.id();
    let ticks = process_start_time(pid)
        .map_err(|error| error.to_string())?
        .ok_or("survivor exited before recording")?;
    write_handle(&handle_path, pid, ticks, own_identity()?)?;
    sweep_stale_client_groups(&root);
    let alive = group_alive(pid)?;
    let handle_kept = handle_path.exists();
    let _ = Command::new("kill")
        .args(["-KILL", "--", &format!("-{pid}")])
        .status();
    let _ = child.wait();
    let _ = fs::remove_dir_all(&root);
    if !alive {
        return Err("sweep terminated a live concurrent run's client".to_owned());
    }
    if !handle_kept {
        return Err("sweep cleared a live concurrent run's handle".to_owned());
    }
    Ok(())
}

#[test]
fn sweep_terminates_identity_matching_survivors() -> Result<(), String> {
    let (root, handle_path) = sweep_fixture("live")?;
    let mut child = spawn_survivor()?;
    let pid = child.id();
    let ticks = process_start_time(pid)
        .map_err(|error| error.to_string())?
        .ok_or("survivor exited before recording")?;
    write_handle(&handle_path, pid, ticks, DEAD_OWNER)?;
    // Reap concurrently: the survivor is this test's child, and the sweep
    // verifies group death, which a zombie would postpone. Real
    // survivors of an unclean exit are reparented to init and reaped.
    let waiter = std::thread::spawn(move || {
        let _ = child.wait();
    });
    sweep_stale_client_groups(&root);
    waiter
        .join()
        .map_err(|_| "waiter thread panicked".to_owned())?;
    if group_alive(pid)? {
        return Err("identity-matching survivor group is still alive".to_owned());
    }
    if handle_path.exists() {
        return Err("swept handle file was not cleared".to_owned());
    }
    let _ = fs::remove_dir_all(&root);
    Ok(())
}

#[test]
fn sweep_never_signals_identity_drift() -> Result<(), String> {
    let (root, handle_path) = sweep_fixture("drift")?;
    let mut child = spawn_survivor()?;
    let pid = child.id();
    let ticks = process_start_time(pid)
        .map_err(|error| error.to_string())?
        .ok_or("survivor exited before recording")?;
    write_handle(&handle_path, pid, ticks + 1, DEAD_OWNER)?;
    sweep_stale_client_groups(&root);
    let alive = group_alive(pid)?;
    let _ = Command::new("kill")
        .args(["-KILL", "--", &format!("-{pid}")])
        .status();
    let _ = child.wait();
    if !alive {
        return Err("sweep signalled a group whose identity drifted".to_owned());
    }
    if handle_path.exists() {
        return Err("drifted handle file was not cleared".to_owned());
    }
    let _ = fs::remove_dir_all(&root);
    Ok(())
}
