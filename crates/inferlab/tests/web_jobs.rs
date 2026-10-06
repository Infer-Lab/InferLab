//! Launched workflows and detached jobs in the web console
//! ([[RFC-0012:C-ACTIONS]]), exercised against the real binary: dry-run
//! previews, a launched job's directory, rediscovery after a restart, the
//! detached supervisor, and identity-checked cancel.

mod dry_run_support;
mod support;
mod web_support;

use std::error::Error;
use std::ffi::OsString;
use std::fs;
use std::io::{BufRead, BufReader};
use std::os::unix::process::{CommandExt, ExitStatusExt};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};
use web_support::{Console, identity, stat_field};

const JOBS: &str = ".inferlab/runtime/jobs";

fn workspace(path: &Path) -> Result<PathBuf, Box<dyn Error>> {
    fs::create_dir_all(path.join(".inferlab"))?;
    fs::write(
        path.join(".inferlab/workspace.toml"),
        "schema_version = 2\n",
    )?;
    Ok(path.canonicalize()?)
}

fn wait_for(path: &Path) -> Result<serde_json::Value, Box<dyn Error>> {
    let deadline = Instant::now() + support::FIXTURE_HANG_GUARD;
    while Instant::now() < deadline {
        if let Ok(bytes) = fs::read(path) {
            return Ok(serde_json::from_slice(&bytes)?);
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    Err(format!("{} never appeared", path.display()).into())
}

fn json(path: &Path) -> Result<serde_json::Value, Box<dyn Error>> {
    Ok(serde_json::from_slice(&fs::read(path)?)?)
}

fn job_spec(argv: &[&str]) -> serde_json::Value {
    serde_json::json!({
        "schema_version": 1, "action": "bench", "argv": argv, "started_unix_ms": 1,
    })
}

#[tokio::test]
async fn previews_run_the_dry_run_and_offer_launch_only_on_success() -> Result<(), Box<dyn Error>> {
    let fixture = dry_run_support::TestWorkspace::new()?;
    let state = tempfile::tempdir()?;
    let mut path = OsString::from(&fixture.adapter_bin);
    path.push(":");
    path.push(std::env::var_os("PATH").unwrap_or_default());
    let console = Console::start_with(
        fixture.root.path(),
        state.path(),
        &[
            ("PATH", path),
            ("XDG_DATA_HOME", fixture.data_home.clone().into_os_string()),
        ],
    )
    .await?;
    let base = console.workspace_path().await?;

    let preview = console
        .post(
            &format!("{base}/actions/preview"),
            &[
                ("action", "serve-start"),
                ("target", "deepseek-v4-flash-qualify"),
            ],
        )
        .await?;
    assert_eq!(preview.status(), 200);
    let preview = preview.text().await?;
    assert!(
        preview.contains("&quot;workflow&quot;:"),
        "the dry run's output is shown: {preview}"
    );
    assert!(preview.contains("<code>--dry-run</code>"));
    assert!(
        preview.contains("<code>deepseek-v4-flash-qualify</code>"),
        "the vector to launch is shown"
    );
    assert!(
        preview.contains(&format!("action=\"{base}/actions/launch\"")),
        "a successful dry run offers the launch"
    );

    let failed = console
        .post(
            &format!("{base}/actions/preview"),
            &[("action", "serve-start"), ("target", "no-such-server")],
        )
        .await?
        .text()
        .await?;
    assert!(failed.contains("no-such-server"), "{failed}");
    assert!(
        !failed.contains("actions/launch"),
        "a failed dry run offers no launch: {failed}"
    );
    assert!(
        !fixture.root.path().join(JOBS).exists(),
        "a preview creates no job"
    );
    Ok(())
}

#[tokio::test]
async fn a_launched_note_runs_as_a_job_that_survives_a_restart() -> Result<(), Box<dyn Error>> {
    let root = tempfile::tempdir()?;
    let state = tempfile::tempdir()?;
    let workspace = workspace(&root.path().join("ws"))?;
    let console = Console::start(&workspace, state.path()).await?;
    let base = console.workspace_path().await?;

    let launched = console
        .post(
            &format!("{base}/actions/launch"),
            &[
                ("action", "scratchpad-note"),
                ("text", "--reads-like-an-option"),
                ("topic", "web"),
            ],
        )
        .await?;
    assert_eq!(launched.status(), 303);
    let location = launched
        .headers()
        .get(reqwest::header::LOCATION)
        .ok_or("no location")?
        .to_str()?
        .to_owned();
    let job = location
        .split_once("job=")
        .ok_or("no job in location")?
        .1
        .to_owned();
    let directory = workspace.join(JOBS).join(&job);

    let exit = wait_for(&directory.join("exit.json"))?;
    assert_eq!(exit["exit_code"], 0, "{exit}");
    let spec = json(&directory.join("job.json"))?;
    let argv = spec["argv"].as_array().ok_or("argv")?;
    assert_eq!(
        argv[argv.len() - 2..],
        [
            serde_json::json!("--"),
            serde_json::json!("--reads-like-an-option")
        ],
        "a value is never read as an option: {spec}"
    );
    assert!(argv.contains(&serde_json::json!(format!(
        "--workspace={}",
        workspace.display()
    ))));
    assert!(spec["started_unix_ms"].as_u64().is_some());
    assert!(
        json(&directory.join("process.json"))?["pid"]
            .as_u64()
            .is_some()
    );
    assert!(directory.join("stdout.log").is_file() && directory.join("stderr.log").is_file());
    let journal = Command::new(env!("CARGO_BIN_EXE_inferlab"))
        .args(["scratchpad", "show", "--all"])
        .current_dir(&workspace)
        .output()?;
    assert!(String::from_utf8_lossy(&journal.stdout).contains("--reads-like-an-option"));

    let jobs = console.page(&location).await?;
    assert!(jobs.contains(&job) && jobs.contains("exited 0"), "{jobs}");
    assert!(
        jobs.contains(r#"id="notify""#) && jobs.contains(r#"aria-pressed="false""#),
        "notifications start off"
    );
    let watch: serde_json::Value = serde_json::from_str(&console.page("/watch").await?)?;
    let watched = watch["jobs"]
        .as_array()
        .and_then(|jobs| jobs.iter().find(|entry| entry["id"] == job.as_str()))
        .ok_or(format!("the ended job is watched: {watch}"))?;
    assert_eq!(watched["running"], false);
    assert_eq!(watched["state"], "exited 0");
    assert_eq!(watched["workspace"], "ws");

    drop(console);
    let restarted = Console::start(&workspace, state.path()).await?;
    let rediscovered = restarted.page(&format!("{base}/jobs")).await?;
    assert!(
        rediscovered.contains(&job) && rediscovered.contains("exited 0"),
        "{rediscovered}"
    );
    Ok(())
}

#[tokio::test]
async fn the_supervisor_detaches_and_cancel_interrupts_only_a_matching_job()
-> Result<(), Box<dyn Error>> {
    let root = tempfile::tempdir()?;
    let state = tempfile::tempdir()?;
    let workspace = workspace(&root.path().join("ws"))?;
    let directory = workspace.join(JOBS).join("sleeper");
    fs::create_dir_all(&directory)?;
    fs::write(
        directory.join("job.json"),
        serde_json::to_vec(&job_spec(&["sleep", "30"]))?,
    )?;

    let mut supervisor = Command::new(env!("CARGO_BIN_EXE_inferlab"))
        .args(["__internal", "job-supervise"])
        .arg(&directory)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()?;
    let mut handshake = String::new();
    BufReader::new(supervisor.stdout.take().ok_or("stdout")?).read_line(&mut handshake)?;
    assert_eq!(handshake.trim(), "started");
    let pid = json(&directory.join("process.json"))?["pid"]
        .as_u64()
        .ok_or("pid")?
        .to_string();
    assert_eq!(
        stat_field(&pid, 5)?,
        pid,
        "the job leads its own process group"
    );
    assert_ne!(
        stat_field(&pid, 6)?,
        stat_field("self", 6)?,
        "the job runs outside the launcher's session"
    );

    // A live process whose recorded identity does not match.
    let mut bystander = Command::new("sleep").arg("30").process_group(0).spawn()?;
    let forged = workspace.join(JOBS).join("forged");
    fs::create_dir_all(&forged)?;
    fs::write(
        forged.join("job.json"),
        serde_json::to_vec(&job_spec(&["sleep", "30"]))?,
    )?;
    let mut forged_identity = identity(bystander.id())?;
    forged_identity["process_start_ticks"] = serde_json::json!(
        forged_identity["process_start_ticks"]
            .as_u64()
            .ok_or("ticks")?
            + 1
    );
    fs::write(
        forged.join("process.json"),
        serde_json::to_vec(&forged_identity)?,
    )?;

    let console = Console::start(&workspace, state.path()).await?;
    let base = console.workspace_path().await?;
    let jobs = console.page(&format!("{base}/jobs?job=sleeper")).await?;
    assert!(jobs.contains("running"), "{jobs}");
    let watch: serde_json::Value = serde_json::from_str(&console.page("/watch").await?)?;
    assert!(
        watch["jobs"].as_array().is_some_and(|jobs| jobs
            .iter()
            .any(|entry| entry["id"] == "sleeper" && entry["running"] == true)),
        "{watch}"
    );
    let listed = console.page(&format!("{base}/jobs?job=forged")).await?;
    assert!(listed.contains("status unknown"), "{listed}");

    // A job whose process belongs to another host says nothing of its outcome.
    let remote = workspace.join(JOBS).join("remote");
    fs::create_dir_all(&remote)?;
    fs::write(
        remote.join("job.json"),
        serde_json::to_vec(&job_spec(&["sleep", "30"]))?,
    )?;
    let mut elsewhere = identity(std::process::id())?;
    elsewhere["host"] = serde_json::json!("example-other-host");
    fs::write(remote.join("process.json"), serde_json::to_vec(&elsewhere)?)?;
    let remote_page = console.page(&format!("{base}/jobs?job=remote")).await?;
    assert!(
        remote_page.contains("not observable from this host") && !remote_page.contains("Interrupt"),
        "{remote_page}"
    );

    let refused = console
        .post(&format!("{base}/jobs/forged/cancel"), &[])
        .await?;
    assert_eq!(refused.status(), 409);
    std::thread::sleep(Duration::from_millis(300));
    assert!(
        bystander.try_wait()?.is_none(),
        "a mismatched identity is never signalled"
    );
    bystander.kill()?;
    bystander.wait()?;

    let cancelled = console
        .post(&format!("{base}/jobs/sleeper/cancel"), &[])
        .await?;
    assert_eq!(cancelled.status(), 303);
    let exit = wait_for(&directory.join("exit.json"))?;
    assert_eq!(
        exit["signal"], 2,
        "cancel is the terminal interrupt: {exit}"
    );
    assert!(supervisor.wait()?.success());
    let ended = console.page(&format!("{base}/jobs?job=sleeper")).await?;
    assert!(ended.contains("interrupted"), "{ended}");
    Ok(())
}

#[test]
fn a_supervisor_that_cannot_start_the_job_reports_it() -> Result<(), Box<dyn Error>> {
    let root = tempfile::tempdir()?;
    let directory = root.path().join("job");
    fs::create_dir_all(&directory)?;
    fs::write(
        directory.join("job.json"),
        serde_json::to_vec(&job_spec(&["/nonexistent/inferlab"]))?,
    )?;
    let output = Command::new(env!("CARGO_BIN_EXE_inferlab"))
        .args(["__internal", "job-supervise"])
        .arg(&directory)
        .stdin(Stdio::null())
        .stderr(Stdio::null())
        .output()?;
    assert!(!output.status.success());
    assert!(output.status.signal().is_none());
    let handshake = String::from_utf8(output.stdout)?;
    assert!(handshake.starts_with("failed:"), "{handshake}");
    assert!(!directory.join("process.json").exists());
    Ok(())
}
