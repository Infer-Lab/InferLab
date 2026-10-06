//! Ad-hoc execution behavior ([[RFC-0002:C-ADHOC-EXECUTION]]): realization
//! selection, stack defaulting, activation argv, mount validation,
//! container argv composition, and exit-status propagation — all observed
//! through the binary against fake `pixi` and `docker` on PATH.

mod support;

use std::error::Error;
use std::ffi::OsString;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use tempfile::TempDir;

struct RunWorkspace {
    root: TempDir,
    bin: PathBuf,
    pixi_log: PathBuf,
    docker_log: PathBuf,
}

impl RunWorkspace {
    fn new(stacks: &[&str], external_image: bool) -> Result<Self, Box<dyn Error>> {
        let root = tempfile::tempdir()?;
        let inferlab = root.path().join(".inferlab");
        let bin = root.path().join("fixture-bin");
        fs::create_dir_all(&inferlab)?;
        fs::create_dir_all(&bin)?;

        let mut workspace = String::from("schema_version = 2\n");
        for stack in stacks {
            workspace.push_str(&format!(
                "[stacks.{stack}]\nintegration = \"vllm\"\npixi_environment = \"{stack}\"\n"
            ));
        }
        if external_image {
            workspace.push_str(&format!(
                "[external_images.base]\nreference = \"example.com/serve:v1@sha256:{}\"\n\
                 integration = \"vllm\"\n",
                "a".repeat(64)
            ));
        }
        fs::write(inferlab.join("workspace.toml"), workspace)?;

        let mut manifest = String::from(
            "[workspace]\nchannels = [\"conda-forge\"]\nplatforms = [\"linux-64\"]\n\n\
             [pypi-dependencies]\ninferlab-integration-vllm = \"==0.1.0\"\n\n\
             [environments]\n",
        );
        let mut lock = String::from("version: 6\nenvironments:\n");
        for stack in stacks {
            manifest.push_str(&format!("{stack} = []\n"));
            lock.push_str(&format!("  {stack}: {{}}\n"));
        }
        fs::write(root.path().join("pixi.toml"), manifest)?;
        fs::write(root.path().join("pixi.lock"), lock)?;
        // ensure_usable checks this prefix exists on disk before shelling
        // out to pixi at all (a fake `pixi` binary cannot fake absence).
        for stack in stacks {
            fs::create_dir_all(root.path().join(".pixi/envs").join(stack))?;
        }

        let pixi_log = root.path().join("pixi-argv.log");
        let docker_log = root.path().join("docker-argv.log");
        write_executable(
            &bin.join("pixi"),
            r#"#!/bin/sh
set -eu
printf '%s\n' "$*" >> "$FAKE_PIXI_LOG"
case "$*" in
  *"--locked --no-install"*) exit "${FAKE_PIXI_PROBE_EXIT:-0}" ;;
esac
while [ "$#" -gt 0 ] && [ "$1" != "--" ]; do shift; done
shift
exec "$@"
"#,
        )?;
        write_executable(
            &bin.join("docker"),
            r#"#!/bin/sh
set -eu
printf '%s\n' "$*" >> "$FAKE_DOCKER_LOG"
case "$1" in
  image) exit "${FAKE_DOCKER_INSPECT_EXIT:-0}" ;;
  run) exit "${FAKE_DOCKER_RUN_EXIT:-0}" ;;
esac
exit 2
"#,
        )?;

        git(root.path(), &["init", "-q"])?;
        git(root.path(), &["config", "user.email", "test@example.com"])?;
        git(root.path(), &["config", "user.name", "Inferlab Test"])?;
        git(root.path(), &["add", "."])?;
        git(root.path(), &["commit", "-qm", "fixture"])?;

        Ok(Self {
            root,
            bin,
            pixi_log,
            docker_log,
        })
    }

    fn write_image_record(&self, record_id: &str) -> Result<(), Box<dyn Error>> {
        let arch = match std::env::consts::ARCH {
            "x86_64" => "amd64",
            "aarch64" => "arm64",
            other => other,
        };
        support::write_assembled_image_record(
            self.root.path(),
            record_id,
            "vllm",
            &format!("{}/{arch}", std::env::consts::OS),
            "sha256:fixture-image",
        )
    }

    fn run(&self, args: &[&str]) -> Result<Output, Box<dyn Error>> {
        self.run_with_env(&[], args)
    }

    fn run_with_env(&self, envs: &[(&str, &str)], args: &[&str]) -> Result<Output, Box<dyn Error>> {
        let mut path = OsString::from(&self.bin);
        path.push(":");
        path.push(std::env::var_os("PATH").unwrap_or_default());
        let mut command = Command::new(env!("CARGO_BIN_EXE_inferlab"));
        command
            .current_dir(self.root.path())
            .env("PATH", path)
            .env("FAKE_PIXI_LOG", &self.pixi_log)
            .env("FAKE_DOCKER_LOG", &self.docker_log)
            // Deterministic device projection: the operator environment must
            // not leak a device selection into the fixtures.
            .env_remove("CUDA_VISIBLE_DEVICES");
        for (name, value) in envs {
            command.env(name, value);
        }
        Ok(command.arg("run").args(args).output()?)
    }

    fn pixi_argv(&self) -> String {
        fs::read_to_string(&self.pixi_log).unwrap_or_default()
    }

    fn docker_argv(&self) -> String {
        fs::read_to_string(&self.docker_log).unwrap_or_default()
    }
}

fn write_executable(path: &Path, content: &str) -> Result<(), Box<dyn Error>> {
    crate::support::write_executable(path, content)
}

fn git(root: &Path, args: &[&str]) -> Result<(), Box<dyn Error>> {
    let output = Command::new("git").current_dir(root).args(args).output()?;
    if !output.status.success() {
        return Err(format!(
            "git {args:?} failed: {}",
            String::from_utf8_lossy(&output.stderr)
        )
        .into());
    }
    Ok(())
}

fn stderr(output: &Output) -> String {
    String::from_utf8_lossy(&output.stderr).into_owned()
}

#[test]
fn usage_rejects_invalid_realization_combinations() -> Result<(), Box<dyn Error>> {
    // Pure argument-shape rejections precede workspace discovery, so no
    // fixture is needed and clap's usage exit code (2) is the contract.
    for args in [
        &["--stack", "vllm", "--image", "img-1", "--", "true"][..],
        &["--image", "a", "--external-image", "b", "--", "true"][..],
        &["--mount", "/tmp", "--", "true"][..],
        &["--devices", "0", "--", "true"][..],
        // The context options apply only to local stack execution.
        &["--image", "img-1", "--record", "--", "true"][..],
        &["--external-image", "base", "--serve", "s-1", "--", "true"][..],
        &["--image", "img-1", "--model", "m", "--", "true"][..],
    ] {
        let output = Command::new(env!("CARGO_BIN_EXE_inferlab"))
            .arg("run")
            .args(args)
            .output()?;
        assert_eq!(
            output.status.code(),
            Some(2),
            "expected a usage rejection for {args:?}: {}",
            stderr(&output)
        );
    }
    Ok(())
}

#[test]
fn local_run_activates_task_free_and_propagates_exit_status() -> Result<(), Box<dyn Error>> {
    let workspace = RunWorkspace::new(&["vllm"], false)?;
    let output = workspace.run(&["--", "sh", "-c", "exit 7"])?;
    assert_eq!(
        output.status.code(),
        Some(7),
        "the command's exit status must propagate verbatim: {}",
        stderr(&output)
    );
    // A propagated nonzero exit is the command's own report, never an
    // Inferlab diagnostic line.
    assert!(
        !stderr(&output).contains("error["),
        "no diagnostic line may accompany a propagated exit: {}",
        stderr(&output)
    );
    let argv = workspace.pixi_argv();
    let lines: Vec<&str> = argv.lines().collect();
    assert_eq!(lines.len(), 2, "usability gate then activation: {argv}");
    assert!(
        lines[0].contains("--locked --no-install --executable -e vllm"),
        "the usability gate must run against the selected environment: {argv}"
    );
    let root = workspace.root.path().canonicalize()?;
    assert!(
        lines[1].contains("-q run --as-is --executable")
            && lines[1].contains(&format!(
                "--manifest-path {}",
                root.join("pixi.toml").display()
            ))
            && lines[1].contains("-e vllm -- sh -c exit 7"),
        "activation must be task-free against the workspace manifest: {argv}"
    );
    Ok(())
}

#[test]
fn local_run_fails_before_execution_when_the_stack_is_unusable() -> Result<(), Box<dyn Error>> {
    let workspace = RunWorkspace::new(&["vllm"], false)?;
    let output = workspace.run_with_env(&[("FAKE_PIXI_PROBE_EXIT", "1")], &["--", "true"])?;
    assert_eq!(output.status.code(), Some(1));
    // The command never executed: only the two probe attempts are logged.
    assert!(
        !workspace.pixi_argv().contains("--as-is"),
        "activation must not run after a failed usability gate: {}",
        workspace.pixi_argv()
    );
    Ok(())
}

#[test]
fn local_run_neither_trusts_nor_produces_confirmation_evidence() -> Result<(), Box<dyn Error>> {
    // RFC-0002:C-ADHOC-EXECUTION: this usability check MUST NOT require or
    // produce the content-confirmation evidence stack status and the
    // serve/recipe launch-time gate share.
    let workspace = RunWorkspace::new(&["vllm"], false)?;
    let marker_path = workspace
        .root
        .path()
        .join(".inferlab/cache/environments/vllm/confirmed.json");

    let output = workspace.run(&["--", "true"])?;
    assert_eq!(output.status.code(), Some(0), "{}", stderr(&output));
    assert!(
        !marker_path.exists(),
        "ad-hoc execution must persist no confirmation evidence a later launch could trust"
    );

    // A marker a real confirmation-aware caller left behind (stack status, not
    // hand-constructed), matching current content exactly, must not be
    // trusted either: the ad-hoc probe still runs, and here it's configured
    // to fail.
    let status = Command::new(env!("CARGO_BIN_EXE_inferlab"))
        .current_dir(workspace.root.path())
        .env("PATH", {
            let mut path = OsString::from(&workspace.bin);
            path.push(":");
            path.push(std::env::var_os("PATH").unwrap_or_default());
            path
        })
        .env("FAKE_PIXI_LOG", &workspace.pixi_log)
        .args(["stack", "status"])
        .output()?;
    assert!(status.status.success(), "{}", stderr(&status));
    assert!(
        marker_path.exists(),
        "stack status must have written a real confirmation marker to seed this test"
    );

    let output = workspace.run_with_env(&[("FAKE_PIXI_PROBE_EXIT", "1")], &["--", "true"])?;
    assert_eq!(
        output.status.code(),
        Some(1),
        "an existing confirmation for identical content must not let ad-hoc execution skip its own probe: {}",
        stderr(&output)
    );
    Ok(())
}

#[test]
fn local_run_fails_before_execution_when_the_stack_was_never_installed()
-> Result<(), Box<dyn Error>> {
    // Distinct from the probe-failure case above: here the environment
    // prefix directory itself is absent, which a fake `pixi` binary cannot
    // paper over — real `pixi run --no-install` silently falls back to the
    // ambient PATH instead of failing in this exact scenario (verified
    // against pixi 0.72.1), so this must be caught before any pixi
    // invocation at all.
    let workspace = RunWorkspace::new(&["vllm"], false)?;
    fs::remove_dir_all(workspace.root.path().join(".pixi/envs/vllm"))?;
    let output = workspace.run(&["--", "true"])?;
    assert_eq!(output.status.code(), Some(1));
    assert_eq!(
        workspace.pixi_argv(),
        "",
        "absence must be caught before any pixi invocation: {}",
        workspace.pixi_argv()
    );
    Ok(())
}

#[test]
fn local_run_defaults_the_single_stack_and_requires_selection_among_several()
-> Result<(), Box<dyn Error>> {
    let workspace = RunWorkspace::new(&["sglang", "vllm"], false)?;
    let output = workspace.run(&["--", "true"])?;
    assert_eq!(output.status.code(), Some(1));

    let output = workspace.run(&["--stack", "sglang", "--", "true"])?;
    assert_eq!(
        output.status.code(),
        Some(0),
        "an explicit selection must execute: {}",
        stderr(&output)
    );
    assert!(workspace.pixi_argv().contains("-e sglang"));

    let output = workspace.run(&["--stack", "missing", "--", "true"])?;
    assert_eq!(output.status.code(), Some(1));
    Ok(())
}

#[test]
fn mount_validation_fails_before_any_container_launch() -> Result<(), Box<dyn Error>> {
    let workspace = RunWorkspace::new(&["vllm"], false)?;
    workspace.write_image_record("img-1")?;
    for mount in [
        "relative/path",
        "/definitely-not-present-here",
        "/tmp,with-comma",
    ] {
        let output = workspace.run(&["--image", "img-1", "--mount", mount, "--", "true"])?;
        assert_eq!(output.status.code(), Some(1), "mount {mount:?} must fail");
    }
    assert_eq!(
        workspace.docker_argv(),
        "",
        "no container may launch after a rejected mount"
    );
    Ok(())
}

#[test]
fn built_image_run_composes_a_bare_container_with_declared_facts_only() -> Result<(), Box<dyn Error>>
{
    let workspace = RunWorkspace::new(&["vllm"], false)?;
    workspace.write_image_record("img-1")?;
    let readable = workspace.root.path().join("probe-input");
    let writable = workspace.root.path().join("probe-output");
    fs::create_dir_all(&readable)?;
    fs::create_dir_all(&writable)?;
    let readable = readable.canonicalize()?;
    let writable = writable.canonicalize()?;
    let output = workspace.run(&[
        "--image",
        "img-1",
        "--devices",
        "0,1",
        "--mount",
        &readable.display().to_string(),
        "--mount",
        &format!("{}:rw", writable.display()),
        "--",
        "nvidia-smi",
        "-L",
    ])?;
    assert_eq!(output.status.code(), Some(0), "{}", stderr(&output));
    let argv = workspace.docker_argv();
    assert!(
        argv.contains("run --rm --interactive"),
        "bare container lifecycle: {argv}"
    );
    assert!(
        argv.contains("--gpus \"device=0,1\""),
        "an explicit device selection must pass through quoted: {argv}"
    );
    assert!(
        argv.contains(&format!(
            "--mount type=bind,source={p},target={p},readonly",
            p = readable.display()
        )),
        "declared mounts bind same-path read-only: {argv}"
    );
    assert!(
        argv.contains(&format!(
            "--mount type=bind,source={p},target={p} ",
            p = writable.display()
        )),
        "an explicit :rw mount omits readonly: {argv}"
    );
    assert!(
        argv.contains("sha256:fixture-image nvidia-smi -L"),
        "a built image executes through its own entrypoint: {argv}"
    );
    assert!(
        !argv.contains("--entrypoint"),
        "a built image's entrypoint must not be overridden: {argv}"
    );
    let mounts = argv.matches("--mount").count();
    assert_eq!(mounts, 2, "no implicit mounts may appear: {argv}");
    Ok(())
}

/// Adhoc execution follows the external image's declared entrypoint: under
/// `image` the command becomes the image entrypoint's arguments
/// ([[RFC-0002:C-ADHOC-EXECUTION]]).
#[test]
fn external_image_run_honors_a_declared_image_entrypoint() -> Result<(), Box<dyn Error>> {
    let workspace = RunWorkspace::new(&["vllm"], true)?;
    let manifest = workspace.root.path().join(".inferlab/workspace.toml");
    let mut text = fs::read_to_string(&manifest)?;
    text.push_str("entrypoint = \"image\"\n");
    fs::write(&manifest, text)?;
    let output = workspace.run(&["--external-image", "base", "--", "python3", "-V"])?;
    assert_eq!(output.status.code(), Some(0), "{}", stderr(&output));
    let argv = workspace.docker_argv();
    let run = argv.lines().last().ok_or("docker run line")?;
    assert!(
        !run.contains("--entrypoint")
            && run.contains("@sha256:")
            && run.trim_end().ends_with("python3 -V"),
        "the image entrypoint receives the command: {argv}"
    );
    Ok(())
}

#[test]
fn external_image_run_overrides_the_entrypoint_after_a_presence_probe() -> Result<(), Box<dyn Error>>
{
    let workspace = RunWorkspace::new(&["vllm"], true)?;
    let output = workspace.run(&["--external-image", "base", "--", "python3", "-V"])?;
    assert_eq!(output.status.code(), Some(0), "{}", stderr(&output));
    let argv = workspace.docker_argv();
    let lines: Vec<&str> = argv.lines().collect();
    assert_eq!(lines.len(), 2, "presence probe then run: {argv}");
    assert!(
        lines[0].starts_with("image inspect"),
        "presence is probed read-only first: {argv}"
    );
    assert!(
        lines[1].contains("--entrypoint python3")
            && lines[1].contains("example.com/serve:v1@sha256:")
            && lines[1].trim_end().ends_with("-V"),
        "an external image executes through an explicit command override: {argv}"
    );
    assert!(
        !argv.contains("--gpus"),
        "without an explicit device selection the container requests no devices: {argv}"
    );

    let output = workspace.run(&["--external-image", "unknown", "--", "true"])?;
    assert_eq!(output.status.code(), Some(1));
    Ok(())
}

fn write_local_bindings(root: &Path, local_toml: &str) -> Result<(), Box<dyn Error>> {
    fs::write(root.join(".inferlab/local.toml"), local_toml)?;
    Ok(())
}

const DEVICE_ECHO: &[&str] = &["sh", "-c", "echo \"CVD=${CUDA_VISIBLE_DEVICES-unset}\""];

#[test]
fn local_run_projects_the_default_placement_devices() -> Result<(), Box<dyn Error>> {
    let workspace = RunWorkspace::new(&["vllm"], false)?;
    write_local_bindings(
        workspace.root.path(),
        "default_placement = \"local\"\n\n\
         [machines.local]\nhost = \"127.0.0.1\"\ndevices = [4, 5]\nports = [8100]\n\n\
         [placements.local]\nmachines = [\"local\"]\n",
    )?;

    let output = workspace.run(
        &["--"]
            .iter()
            .chain(DEVICE_ECHO)
            .cloned()
            .collect::<Vec<_>>(),
    )?;

    assert_eq!(output.status.code(), Some(0), "{}", stderr(&output));
    let stdout = String::from_utf8(output.stdout)?;
    assert!(
        stdout.contains("CVD=4,5"),
        "the default placement device set must reach the command: {stdout}"
    );
    Ok(())
}

#[test]
fn operator_device_selection_wins_over_the_placement_projection() -> Result<(), Box<dyn Error>> {
    let workspace = RunWorkspace::new(&["vllm"], false)?;
    write_local_bindings(
        workspace.root.path(),
        "default_placement = \"local\"\n\n\
         [machines.local]\nhost = \"127.0.0.1\"\ndevices = [4, 5]\nports = [8100]\n\n\
         [placements.local]\nmachines = [\"local\"]\n",
    )?;

    let output = workspace.run_with_env(
        &[("CUDA_VISIBLE_DEVICES", "7")],
        &["--"]
            .iter()
            .chain(DEVICE_ECHO)
            .cloned()
            .collect::<Vec<_>>(),
    )?;

    assert_eq!(output.status.code(), Some(0), "{}", stderr(&output));
    let stdout = String::from_utf8(output.stdout)?;
    assert!(
        stdout.contains("CVD=7"),
        "the operator's explicit selection must win: {stdout}"
    );
    Ok(())
}

#[test]
fn remote_or_deviceless_default_placement_projects_nothing() -> Result<(), Box<dyn Error>> {
    let workspace = RunWorkspace::new(&["vllm"], false)?;
    write_local_bindings(
        workspace.root.path(),
        "default_placement = \"remote\"\n\n\
         [machines.remote]\nhost = \"192.0.2.10\"\ndevices = [0, 1]\nports = [8100]\n\
         workspace = \"/tmp\"\n\
         [machines.remote.launch]\nkind = \"ssh\"\ntarget = \"192.0.2.10\"\n\n\
         [placements.remote]\nmachines = [\"remote\"]\n",
    )?;

    let output = workspace.run(
        &["--"]
            .iter()
            .chain(DEVICE_ECHO)
            .cloned()
            .collect::<Vec<_>>(),
    )?;

    assert_eq!(output.status.code(), Some(0), "{}", stderr(&output));
    let stdout = String::from_utf8(output.stdout)?;
    assert!(
        stdout.contains("CVD=unset"),
        "a remote default placement must not project devices: {stdout}"
    );

    write_local_bindings(
        workspace.root.path(),
        "default_placement = \"local\"\n\n\
         [machines.local]\nhost = \"127.0.0.1\"\ndevices = []\nports = [8100]\n\n\
         [placements.local]\nmachines = [\"local\"]\n",
    )?;
    let output = workspace.run(
        &["--"]
            .iter()
            .chain(DEVICE_ECHO)
            .cloned()
            .collect::<Vec<_>>(),
    )?;
    let stdout = String::from_utf8(output.stdout)?;
    assert!(
        stdout.contains("CVD=unset"),
        "a device-less default placement must not project: {stdout}"
    );
    Ok(())
}

#[test]
fn rank_shaped_default_placement_projects_rank_devices() -> Result<(), Box<dyn Error>> {
    let workspace = RunWorkspace::new(&["vllm"], false)?;
    write_local_bindings(
        workspace.root.path(),
        "default_placement = \"local\"\n\n\
         [machines.local]\nhost = \"127.0.0.1\"\ndevices = [0, 1, 2, 3]\nports = [8100]\n\n\
         [placements.local.roles.serve]\nmachine = \"local\"\ndevices = [2, 3]\n",
    )?;

    let output = workspace.run(
        &["--"]
            .iter()
            .chain(DEVICE_ECHO)
            .cloned()
            .collect::<Vec<_>>(),
    )?;

    assert_eq!(output.status.code(), Some(0), "{}", stderr(&output));
    let stdout = String::from_utf8(output.stdout)?;
    assert!(
        stdout.contains("CVD=2,3"),
        "a rank-shaped placement projects its rank devices, not the machine inventory: {stdout}"
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// InferLab context and recorded runs ([[RFC-0002:C-ADHOC-EXECUTION]],
// [[RFC-0005:C-EVIDENCE]]).
// ---------------------------------------------------------------------------

fn run_records(root: &Path) -> Result<Vec<PathBuf>, Box<dyn Error>> {
    let records = root.join(".inferlab/records");
    if !records.exists() {
        return Ok(Vec::new());
    }
    let mut entries = fs::read_dir(records)?
        .map(|entry| entry.map(|entry| entry.path()))
        .collect::<Result<Vec<_>, _>>()?;
    entries.sort();
    Ok(entries)
}

fn load_record(dir: &Path) -> Result<serde_json::Value, Box<dyn Error>> {
    Ok(serde_json::from_slice(&fs::read(dir.join("record.json"))?)?)
}

#[test]
fn local_run_injects_the_inferlab_context_without_a_record() -> Result<(), Box<dyn Error>> {
    let workspace = RunWorkspace::new(&["vllm"], false)?;
    let output = workspace.run(&[
        "--",
        "sh",
        "-c",
        "printf '%s|%s|%s\\n' \"$INFERLAB_CONTEXT\" \"$INFERLAB_WORKSPACE_ROOT\" \"${INFERLAB_RECORD_ID-unset}\"",
    ])?;

    assert_eq!(output.status.code(), Some(0), "{}", stderr(&output));
    let root = fs::canonicalize(workspace.root.path())?;
    assert_eq!(
        String::from_utf8(output.stdout)?,
        format!("1|{}|unset\n", root.display())
    );
    assert!(run_records(workspace.root.path())?.is_empty());
    Ok(())
}

#[test]
fn recorded_run_keeps_argv_context_logs_and_exit_status() -> Result<(), Box<dyn Error>> {
    let workspace = RunWorkspace::new(&["vllm"], false)?;
    let script = "echo out-line; echo err-line >&2; \
                  echo kept > \"$INFERLAB_RECORD_ARTIFACTS/note.txt\"; exit 3";
    let output = workspace.run(&["--record", "--", "sh", "-c", script])?;

    // The command's status and streams still reach the operator.
    assert_eq!(output.status.code(), Some(3), "{}", stderr(&output));
    assert!(stderr(&output).contains("err-line"));
    assert!(String::from_utf8(output.stdout)?.contains("out-line"));

    let records = run_records(workspace.root.path())?;
    assert_eq!(records.len(), 1, "{records:?}");
    let dir = &records[0];
    let id = dir
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or("record id")?
        .to_owned();
    assert!(id.contains("-run-vllm-"), "{id}");
    let record = load_record(dir)?;
    assert_eq!(record["kind"], "run");
    assert_eq!(record["id"], id.as_str());
    assert_eq!(record["status"], "failed");
    assert_eq!(record["exit_status"]["code"], 3);
    assert_eq!(record["stack"]["id"], "vllm");
    assert_eq!(record["argv"], serde_json::json!(["sh", "-c", script]));
    assert!(
        record["workspace"]["revision"]
            .as_str()
            .is_some_and(|revision| !revision.is_empty())
    );
    let provided = &record["context"]["provided"];
    assert_eq!(provided["INFERLAB_CONTEXT"], "1");
    assert_eq!(provided["INFERLAB_RECORD_ID"], id.as_str());
    assert!(record["finished_unix_ms"].as_u64().is_some());
    let stdout_log = record["stdout"].as_str().ok_or("stdout log")?;
    assert!(fs::read_to_string(workspace.root.path().join(stdout_log))?.contains("out-line"));
    let stderr_log = record["stderr"].as_str().ok_or("stderr log")?;
    assert!(fs::read_to_string(workspace.root.path().join(stderr_log))?.contains("err-line"));
    assert_eq!(
        fs::read_to_string(dir.join("artifacts/note.txt"))?,
        "kept\n"
    );
    Ok(())
}

#[test]
fn recorded_run_resolves_and_records_the_selected_model_binding() -> Result<(), Box<dyn Error>> {
    let workspace = RunWorkspace::new(&["vllm"], false)?;
    write_local_bindings(
        workspace.root.path(),
        "[model_weights.fixture-model]\nlocator = \"/models/fixture-model\"\n",
    )?;
    let output = workspace.run(&[
        "--record",
        "--model",
        "fixture-model",
        "--",
        "sh",
        "-c",
        "echo \"$INFERLAB_MODEL_ID=$INFERLAB_MODEL_PATH\"",
    ])?;

    assert_eq!(output.status.code(), Some(0), "{}", stderr(&output));
    assert_eq!(
        String::from_utf8(output.stdout)?,
        "fixture-model=/models/fixture-model\n"
    );
    let records = run_records(workspace.root.path())?;
    let record = load_record(&records[0])?;
    assert_eq!(
        record["model"],
        serde_json::json!({"id": "fixture-model", "locator": "/models/fixture-model"})
    );

    // An undeclared binding fails before the command executes.
    let marker = workspace.root.path().join("executed");
    let output = workspace.run(&[
        "--model",
        "undeclared",
        "--",
        "touch",
        marker.to_str().ok_or("marker path")?,
    ])?;
    assert_ne!(output.status.code(), Some(0));
    assert!(
        stderr(&output).contains("undeclared"),
        "{}",
        stderr(&output)
    );
    assert!(!marker.exists());
    Ok(())
}

#[test]
fn nested_run_replaces_the_inherited_context_and_records_its_parent() -> Result<(), Box<dyn Error>>
{
    let workspace = RunWorkspace::new(&["vllm"], false)?;
    let output = workspace.run_with_env(
        &[("FIXTURE_INFERLAB_BIN", env!("CARGO_BIN_EXE_inferlab"))],
        &[
            "--record",
            "--",
            "sh",
            "-c",
            "\"$FIXTURE_INFERLAB_BIN\" run --record -- sh -c 'echo \"inner=$INFERLAB_RECORD_ID\"'",
        ],
    )?;

    assert_eq!(output.status.code(), Some(0), "{}", stderr(&output));
    let records = run_records(workspace.root.path())?
        .into_iter()
        .map(|dir| load_record(&dir))
        .collect::<Result<Vec<_>, _>>()?;
    assert_eq!(records.len(), 2);
    let (outer, inner): (Vec<_>, Vec<_>) = records
        .iter()
        .partition(|record| record["parent_record"].is_null());
    let (outer, inner) = (outer[0], inner[0]);
    assert_eq!(inner["parent_record"], outer["id"]);
    // The inner command saw its own record, not the outer one.
    assert_eq!(
        String::from_utf8(output.stdout)?,
        format!("inner={}\n", inner["id"].as_str().ok_or("inner id")?)
    );
    Ok(())
}

#[test]
fn context_variables_set_without_the_marker_fail_before_execution() -> Result<(), Box<dyn Error>> {
    let workspace = RunWorkspace::new(&["vllm"], false)?;
    let marker = workspace.root.path().join("executed");
    let output = workspace.run_with_env(
        &[("INFERLAB_RECORD_ID", "operator-set")],
        &["--", "touch", marker.to_str().ok_or("marker path")?],
    )?;

    assert_ne!(output.status.code(), Some(0));
    assert!(
        stderr(&output).contains("INFERLAB_RECORD_ID"),
        "{}",
        stderr(&output)
    );
    assert!(!marker.exists());
    Ok(())
}

#[test]
fn interrupted_recorded_run_terminates_its_process_group() -> Result<(), Box<dyn Error>> {
    let workspace = RunWorkspace::new(&["vllm"], false)?;
    let pid_file = workspace.root.path().join("sleeper.pid");
    let mut path = OsString::from(&workspace.bin);
    path.push(":");
    path.push(std::env::var_os("PATH").unwrap_or_default());
    let mut wrapper = Command::new(env!("CARGO_BIN_EXE_inferlab"))
        .current_dir(workspace.root.path())
        .env("PATH", path)
        .env("FAKE_PIXI_LOG", &workspace.pixi_log)
        .env_remove("CUDA_VISIBLE_DEVICES")
        .args([
            "run",
            "--record",
            "--",
            "sh",
            "-c",
            &format!("sleep 300 & echo $! > {}; wait", pid_file.display()),
        ])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()?;
    support::wait_for_marker(&mut wrapper, &pid_file)?;
    let sleeper: i32 = fs::read_to_string(&pid_file)?.trim().parse()?;

    Command::new("kill")
        .args(["-TERM", &wrapper.id().to_string()])
        .status()?;
    let status = wrapper.wait()?;

    let sleeper_alive = Path::new(&format!("/proc/{sleeper}")).exists();
    if sleeper_alive {
        let _ = Command::new("kill")
            .args(["-KILL", &sleeper.to_string()])
            .status();
    }
    assert!(!status.success());
    assert!(
        !sleeper_alive,
        "the command's process group must not survive"
    );
    let records = run_records(workspace.root.path())?;
    let record = load_record(&records[0])?;
    assert_eq!(record["status"], "interrupted");
    assert!(
        record["cleanup"]["verified"].as_bool().unwrap_or(false),
        "{record}"
    );
    Ok(())
}

#[test]
fn interrupting_a_nested_run_finalizes_every_record_and_its_stubborn_command()
-> Result<(), Box<dyn Error>> {
    let workspace = RunWorkspace::new(&["vllm"], false)?;
    let pid_file = workspace.root.path().join("stubborn.pid");
    let mut path = OsString::from(&workspace.bin);
    path.push(":");
    path.push(std::env::var_os("PATH").unwrap_or_default());
    // The inner command ignores SIGTERM and runs in the inner run's own
    // process group: the outer run's escalation must still reach it through
    // the process tree.
    let inner = format!(
        "trap '' TERM; echo $$ > {}; while :; do sleep 1; done",
        pid_file.display()
    );
    let mut wrapper = Command::new(env!("CARGO_BIN_EXE_inferlab"))
        .current_dir(workspace.root.path())
        .env("PATH", path)
        .env("FAKE_PIXI_LOG", &workspace.pixi_log)
        .env("FIXTURE_INFERLAB_BIN", env!("CARGO_BIN_EXE_inferlab"))
        .env("FIXTURE_INNER", &inner)
        .env_remove("CUDA_VISIBLE_DEVICES")
        .args([
            "run",
            "--record",
            "--",
            "sh",
            "-c",
            "\"$FIXTURE_INFERLAB_BIN\" run --record -- sh -c \"$FIXTURE_INNER\"",
        ])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()?;
    support::wait_for_marker(&mut wrapper, &pid_file)?;
    let stubborn: i32 = fs::read_to_string(&pid_file)?.trim().parse()?;

    Command::new("kill")
        .args(["-TERM", &wrapper.id().to_string()])
        .status()?;
    wrapper.wait()?;

    let alive = Path::new(&format!("/proc/{stubborn}")).exists();
    if alive {
        let _ = Command::new("kill")
            .args(["-KILL", &stubborn.to_string()])
            .status();
    }
    assert!(
        !alive,
        "the outer run must terminate the nested command's process group"
    );
    let records = run_records(workspace.root.path())?
        .into_iter()
        .map(|dir| load_record(&dir))
        .collect::<Result<Vec<_>, _>>()?;
    assert_eq!(records.len(), 2);
    // Both records are finalized, not just marked: the nested run is not
    // killed before it records its own cleanup ([[RFC-0005:C-EVIDENCE]]).
    for record in &records {
        assert_eq!(record["status"], "interrupted", "{record}");
        assert!(record["finished_unix_ms"].as_u64().is_some(), "{record}");
        assert_eq!(record["cleanup"]["trigger"], "interrupt", "{record}");
    }
    Ok(())
}
