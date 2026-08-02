use crate::harness::{TestWorkspace, wait_for_path};
use serde_json::Value;
use std::error::Error;
use std::fs;
use std::process::{Child, Command, Output, Stdio};
use std::thread;
use std::time::{Duration, Instant};
use tempfile::NamedTempFile;

#[test]
fn interruption_records_remaining_measurements_and_cleans_up() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    let stdout = NamedTempFile::new()?;
    let stderr = NamedTempFile::new()?;
    let mut child = workspace
        .command()
        .env("FIXTURE_EVAL_WAIT", "1")
        .env("FIXTURE_EVAL_NATIVE_CHECKPOINT", "1")
        .args(["recipe", "run", "dsv4-qualify"])
        .stdout(stdout.reopen()?)
        .stderr(stderr.reopen()?)
        .spawn()?;
    wait_for_path(workspace.eval_marker(), Duration::from_secs(5))?;
    let eval_child_pid = fs::read_to_string(workspace.eval_marker())?
        .trim()
        .parse::<u32>()?;
    let signal = Command::new("kill")
        .args(["-TERM", &child.id().to_string()])
        .status()?;
    assert!(signal.success());
    let deadline = Instant::now() + Duration::from_secs(10);
    while child.try_wait()?.is_none() && Instant::now() < deadline {
        thread::sleep(Duration::from_millis(50));
    }
    if child.try_wait()?.is_none() {
        child.kill()?;
        return Err("interrupted recipe did not finish cleanup within 10 seconds".into());
    }
    let output = read_spooled_output(child, &stdout, &stderr)?;

    assert!(!output.status.success());
    let record: Value = serde_json::from_slice(&output.stdout)?;
    assert_eq!(record["status"], "failed");
    assert_eq!(record["interrupted"], true);
    assert_eq!(record["evals"].as_array().map(Vec::len), Some(2));
    assert_eq!(record["benches"].as_array().map(Vec::len), Some(2));
    assert_eq!(record["evals"][0]["status"], "succeeded");
    assert_eq!(record["evals"][1]["status"], "failed");
    assert!(
        record["benches"]
            .as_array()
            .is_some_and(|children| children.iter().all(|child| child["status"] == "skipped"))
    );
    let interrupted_eval = workspace.load_record(
        record["evals"][1]["id"]
            .as_str()
            .ok_or("interrupted Eval has no record id")?,
    )?;
    assert_eq!(interrupted_eval["cases"][0]["process"]["interrupted"], true);
    assert_eq!(
        interrupted_eval["cases"][0]["process"]["termination"]["kill_sent"],
        true
    );
    assert_eq!(
        interrupted_eval["cases"][0]["process"]["termination"]["verified"],
        true
    );
    assert_eq!(
        interrupted_eval["cases"][0]["native_command"][0],
        "fixture-eval"
    );
    assert_eq!(
        interrupted_eval["cases"][0]["native_interrupted"],
        Value::Null
    );
    assert_eq!(
        interrupted_eval["cases"][0]["native_timed_out"],
        Value::Null
    );
    let raw_artifacts = interrupted_eval["cases"][0]["raw_artifacts"]
        .as_array()
        .ok_or("interrupted Eval has no raw artifacts")?;
    assert!(
        raw_artifacts
            .iter()
            .any(|artifact| artifact["kind"] == "directory")
    );
    assert!(
        raw_artifacts
            .iter()
            .any(|artifact| artifact["kind"] == "lm-eval-process")
    );
    wait_for_pid_exit(eval_child_pid, Duration::from_secs(5))?;
    assert_eq!(record["server"]["status"], "stopped");
    assert_eq!(record["cleanup"]["verified"], true);
    assert!(!workspace.bench_marker().exists());
    Ok(())
}

#[test]
fn interruption_during_builtin_smoke_preserves_the_interrupted_terminal_cause()
-> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    let marker = workspace.root().join("smoke-started");
    let stdout = NamedTempFile::new()?;
    let stderr = NamedTempFile::new()?;
    let mut child = workspace
        .command()
        .env("FIXTURE_SMOKE_DELAY_SECONDS", "60")
        .env("FIXTURE_SMOKE_MARKER", &marker)
        .args(["recipe", "run", "dsv4-qualify"])
        .stdout(stdout.reopen()?)
        .stderr(stderr.reopen()?)
        .spawn()?;
    wait_for_path(&marker, Duration::from_secs(5))?;
    let signal = Command::new("kill")
        .args(["-TERM", &child.id().to_string()])
        .status()?;
    assert!(signal.success());
    let deadline = Instant::now() + Duration::from_secs(10);
    while child.try_wait()?.is_none() && Instant::now() < deadline {
        thread::sleep(Duration::from_millis(50));
    }
    if child.try_wait()?.is_none() {
        child.kill()?;
        return Err("interrupted smoke recipe did not finish within 10 seconds".into());
    }
    let output = read_spooled_output(child, &stdout, &stderr)?;

    assert!(!output.status.success());
    let recipe: Value = serde_json::from_slice(&output.stdout)?;
    assert_eq!(recipe["interrupted"], true);
    let smoke = workspace.load_record(
        recipe["evals"][0]["id"]
            .as_str()
            .ok_or("interrupted smoke Eval has no record id")?,
    )?;
    assert_eq!(smoke["cases"][0]["timing"]["terminal_cause"], "interrupted");
    assert_eq!(smoke["cases"][0]["process"], Value::Null);
    assert!(
        smoke["cases"][0]["error"]
            .as_str()
            .is_some_and(|error| error.contains("OpenAI smoke interrupted"))
    );
    Ok(())
}

#[test]
fn interrupted_bench_preserves_native_evidence_and_cleans_its_group() -> Result<(), Box<dyn Error>>
{
    let workspace = TestWorkspace::new()?;
    let stdout = NamedTempFile::new()?;
    let stderr = NamedTempFile::new()?;
    let mut child = workspace
        .command()
        .env("FIXTURE_BENCH_INTERRUPT_WAIT", "1")
        .args(["recipe", "run", "dsv4-qualify"])
        .stdout(stdout.reopen()?)
        .stderr(stderr.reopen()?)
        .spawn()?;
    wait_for_path(workspace.bench_marker(), Duration::from_secs(5))?;
    let bench_child_pid = fs::read_to_string(workspace.bench_marker())?
        .trim()
        .parse::<u32>()?;
    let signal = Command::new("kill")
        .args(["-TERM", &child.id().to_string()])
        .status()?;
    assert!(signal.success());
    let deadline = Instant::now() + Duration::from_secs(10);
    while child.try_wait()?.is_none() && Instant::now() < deadline {
        thread::sleep(Duration::from_millis(50));
    }
    if child.try_wait()?.is_none() {
        child.kill()?;
        return Err("interrupted recipe did not finish Bench cleanup within 10 seconds".into());
    }
    let output = read_spooled_output(child, &stdout, &stderr)?;

    assert!(!output.status.success());
    let recipe: Value = serde_json::from_slice(&output.stdout)?;
    let bench = workspace.load_record(
        recipe["benches"][0]["id"]
            .as_str()
            .ok_or("interrupted Bench has no record id")?,
    )?;
    assert_eq!(bench["status"], "failed");
    assert_eq!(bench["cases"][0]["process"]["interrupted"], true);
    assert_eq!(
        bench["cases"][0]["process"]["termination"]["kill_sent"],
        true
    );
    assert_eq!(
        bench["cases"][0]["process"]["termination"]["verified"],
        true
    );
    assert_eq!(bench["cases"][0]["timing"]["terminal_cause"], "interrupted");
    let business_elapsed = bench["cases"][0]["timing"]["elapsed_ms"]
        .as_u64()
        .ok_or("interrupted Bench has no business elapsed time")?;
    let cleanup_elapsed = bench["cases"][0]["process"]["termination"]["elapsed_ms"]
        .as_u64()
        .ok_or("interrupted Bench has no cleanup elapsed time")?;
    assert!(business_elapsed < cleanup_elapsed);
    assert_eq!(bench["cases"][0]["native_command"][0], "fixture-bench");
    assert_eq!(bench["cases"][0]["native_exit_code"], 143);
    assert_eq!(bench["cases"][0]["raw_artifacts"][0]["name"], "partial");
    wait_for_pid_exit(bench_child_pid, Duration::from_secs(5))?;
    assert_eq!(recipe["server"]["status"], "stopped");
    assert_eq!(recipe["cleanup"]["verified"], true);
    Ok(())
}

fn read_spooled_output(
    mut child: Child,
    stdout: &NamedTempFile,
    stderr: &NamedTempFile,
) -> Result<Output, Box<dyn Error>> {
    let status = child.wait()?;
    Ok(Output {
        status,
        stdout: fs::read(stdout.path())?,
        stderr: fs::read(stderr.path())?,
    })
}

fn wait_for_pid_exit(pid: u32, timeout: Duration) -> Result<(), Box<dyn Error>> {
    let deadline = Instant::now() + timeout;
    while Instant::now() < deadline {
        let status = Command::new("kill")
            .args(["-0", &pid.to_string()])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()?;
        if !status.success() {
            return Ok(());
        }
        thread::sleep(Duration::from_millis(25));
    }
    Err(format!("client child process {pid} remained alive after cleanup").into())
}
