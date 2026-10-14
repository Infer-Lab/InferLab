use crate::harness::TestWorkspace;
use crate::support::wait_until;
use serde_json::Value;
use std::error::Error;
use std::fs;
use std::path::Path;
use std::process::Command;

fn read_json(path: &Path) -> Option<Value> {
    fs::read(path)
        .ok()
        .and_then(|bytes| serde_json::from_slice(&bytes).ok())
}

/// A detached recipe returns once its record exists, with the job and record
/// identities, while the run continues as a job; interrupting the job runs
/// the recipe's own cleanup ([[RFC-0003:C-RUNTIME-WORKFLOWS]]).
#[test]
fn detached_recipe_returns_its_record_and_a_job_interrupt_cleans_up() -> Result<(), Box<dyn Error>>
{
    let workspace = TestWorkspace::new()?;
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
    let job_id = detached["job_id"].as_str().ok_or("no job id")?.to_owned();
    let record_id = detached["record_id"].as_str().ok_or("no record id")?;
    let job_dir = Path::new(detached["job_dir"].as_str().ok_or("no job dir")?).to_path_buf();
    assert!(record_id.contains("recipe"), "{detached}");
    assert!(Path::new(detached["stderr"].as_str().ok_or("no stderr log")?).is_file());

    wait_until("the detached recipe reaching its Eval", || {
        Ok(workspace.eval_marker().exists())
    })?;
    let interrupt = workspace
        .command()
        .args(["job", "interrupt", &job_id])
        .output()?;
    assert!(
        interrupt.status.success(),
        "{}",
        String::from_utf8_lossy(&interrupt.stderr)
    );
    let exit_file = job_dir.join("exit.json");
    let finished = wait_until("the interrupted job ending", || Ok(exit_file.is_file()));
    if finished.is_err() {
        let _ = Command::new("kill")
            .args([
                "-KILL",
                "--",
                &format!("-{}", job_pid(&job_dir).unwrap_or(0)),
            ])
            .status();
    }
    finished?;

    let record = workspace.load_record(record_id)?;
    assert_eq!(record["status"], "failed");
    assert_eq!(record["interrupted"], true);
    assert_eq!(record["cleanup"]["verified"], true);
    assert_eq!(record["server"]["status"], "stopped");
    Ok(())
}

fn job_pid(job_dir: &Path) -> Option<u64> {
    read_json(&job_dir.join("process.json")).and_then(|process| process["pid"].as_u64())
}

/// A detached run that ends before creating a record fails with the job's
/// own diagnostics.
#[test]
fn detached_run_that_fails_before_a_record_reports_the_job_error() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    let output = workspace
        .command()
        .args(["recipe", "run", "no-such-recipe", "--detach"])
        .output()?;

    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr)?;
    assert!(stderr.contains("before creating a record"), "{stderr}");
    assert!(stderr.contains("no-such-recipe"), "{stderr}");
    Ok(())
}
