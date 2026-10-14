use crate::harness::TestWorkspace;
use crate::support::{wait_for_exit, wait_until};
use serde_json::{Value, json};
use std::error::Error;
use std::fs::{self, File, OpenOptions};
use std::path::{Path, PathBuf};
use std::process::Command;
use tempfile::NamedTempFile;

/// A detached recipe waiting in its Eval: its record id and job directory.
fn detach_waiting_recipe(workspace: &TestWorkspace) -> Result<(String, PathBuf), Box<dyn Error>> {
    let output = workspace
        .command()
        .env("FIXTURE_EVAL_WAIT", "1")
        .args(["recipe", "run", "deepseek-v4-flash-qualify", "--detach"])
        .output()?;
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let detached: Value = serde_json::from_slice(&output.stdout)?;
    let record_id = detached["record_id"]
        .as_str()
        .ok_or("no record id")?
        .to_owned();
    let job_dir = PathBuf::from(detached["job_dir"].as_str().ok_or("no job dir")?);
    wait_until("the detached recipe reaching its Eval", || {
        Ok(workspace.eval_marker().exists())
    })?;
    Ok((record_id, job_dir))
}

/// Kill the detached job outright, as a lost controller host would, leaving
/// its running server behind.
fn kill_job(job_dir: &Path) -> Result<(), Box<dyn Error>> {
    let process: Value = serde_json::from_slice(&fs::read(job_dir.join("process.json"))?)?;
    let pid = process["pid"].as_u64().ok_or("job has no pid")?;
    let status = Command::new("kill")
        .args(["-KILL", "--", &format!("-{pid}")])
        .status()?;
    assert!(status.success());
    wait_until("the killed job leaving its process table entry", || {
        Ok(!Path::new(&format!("/proc/{pid}")).exists())
    })
}

fn record_dir(workspace: &TestWorkspace, id: &str) -> PathBuf {
    workspace.root().join(".inferlab/records").join(id)
}

fn hold_lock(path: &Path) -> Result<File, Box<dyn Error>> {
    let file = OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .open(path)?;
    file.lock()?;
    Ok(file)
}

/// `recipe stop` of a recipe whose controller lives asks the controller to
/// stop, waits for it, and reports the record as the controller finalized it
/// ([[RFC-0003:C-RUNTIME-WORKFLOWS]]).
#[test]
fn recipe_stop_asks_its_live_controller_and_reports_the_finalized_record()
-> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    let (record_id, _job_dir) = detach_waiting_recipe(&workspace)?;

    let output = workspace
        .command()
        .args(["recipe", "stop", &record_id])
        .output()?;

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(output.status.success(), "{stderr}");
    assert!(
        stderr.contains("stop requested of the record's controller"),
        "{stderr}"
    );
    assert!(
        stderr.contains("recipe run deepseek-v4-flash-qualify"),
        "{stderr}"
    );
    let record: Value = serde_json::from_slice(&output.stdout)?;
    assert_eq!(record["id"], record_id.as_str());
    assert_eq!(record["status"], "failed");
    assert_eq!(record["interrupted"], true);
    assert_eq!(record["operator_stop"], Value::Null);
    assert_eq!(record["cleanup"]["verified"], true);
    assert_eq!(record["server"]["status"], "stopped");
    Ok(())
}

/// When the controller is gone, `recipe stop` stops the recipe's server and
/// finalizes the record as stopped by the operator ([[RFC-0005:C-EVIDENCE]]).
#[test]
fn recipe_stop_finalizes_a_record_whose_controller_is_gone() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    let (record_id, job_dir) = detach_waiting_recipe(&workspace)?;
    kill_job(&job_dir)?;

    let output = workspace
        .command()
        .args(["recipe", "stop", &record_id])
        .output()?;

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let record: Value = serde_json::from_slice(&output.stdout)?;
    assert_eq!(record["status"], "failed");
    assert_eq!(record["operator_stop"], json!({"forced": false}));
    assert_eq!(record["cleanup"]["verified"], true);
    assert_eq!(record["server"]["status"], "stopped");
    let server = workspace.load_record(record["server"]["id"].as_str().ok_or("no server")?)?;
    assert_eq!(server["status"], "stopped");
    assert_eq!(workspace.load_record(&record_id)?, record);

    let again = workspace
        .command()
        .args(["recipe", "stop", &record_id])
        .output()?;
    assert!(again.status.success());
    assert_eq!(serde_json::from_slice::<Value>(&again.stdout)?, record);
    Ok(())
}

/// `--force` proceeds past a held controller lock, for a controller whose
/// host is gone while its lock persists, and records that it did.
#[test]
fn recipe_stop_force_proceeds_past_a_held_controller_lock() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    let (record_id, job_dir) = detach_waiting_recipe(&workspace)?;
    kill_job(&job_dir)?;
    let _stale = hold_lock(&record_dir(&workspace, &record_id).join("controller.lock"))?;

    let output = workspace
        .command()
        .args(["recipe", "stop", &record_id, "--force"])
        .output()?;

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(output.status.success(), "{stderr}");
    assert!(
        stderr.contains("stopping past the record's controller"),
        "{stderr}"
    );
    let record: Value = serde_json::from_slice(&output.stdout)?;
    assert_eq!(record["status"], "failed");
    assert_eq!(record["operator_stop"], json!({"forced": true}));
    assert_eq!(record["server"]["status"], "stopped");
    Ok(())
}

/// A launch waiting for the workspace allocation lock names the holder and
/// ends on an operator interrupt before creating any record
/// ([[RFC-0002:C-LOCAL-PLACEMENT]]).
#[test]
fn an_allocation_lock_wait_names_its_holder_and_ends_on_interrupt() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    let runtime = workspace.root().join(".inferlab/runtime");
    fs::create_dir_all(&runtime)?;
    let _held = hold_lock(&runtime.join("allocation.lock"))?;
    fs::write(
        runtime.join("allocation.json"),
        serde_json::to_vec(&json!({
            "host": "fixture-host",
            "pid": 4242,
            "started_unix_ms": 0,
            "command": "inferlab fixture holder",
        }))?,
    )?;
    let stderr = NamedTempFile::new()?;
    let mut child = workspace
        .command()
        .args(["recipe", "run", "deepseek-v4-flash-qualify"])
        .stderr(stderr.reopen()?)
        .spawn()?;
    wait_until("the launch reporting its allocation lock wait", || {
        Ok(
            fs::read_to_string(stderr.path())?
                .contains("waiting for the workspace allocation lock"),
        )
    })?;
    let status = Command::new("kill")
        .args(["-INT", &child.id().to_string()])
        .status()?;
    assert!(status.success());
    wait_for_exit(&mut child, "the interrupted lock wait did not end")?;

    assert!(!child.wait()?.success());
    let diagnostics = fs::read_to_string(stderr.path())?;
    assert!(
        diagnostics.contains("pid 4242 on fixture-host"),
        "{diagnostics}"
    );
    assert!(
        diagnostics.contains("interrupted while waiting for the workspace allocation lock"),
        "{diagnostics}"
    );
    let records = workspace.root().join(".inferlab/records");
    assert!(!records.exists() || fs::read_dir(&records)?.next().is_none());
    Ok(())
}
