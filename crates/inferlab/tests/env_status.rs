//! Content-confirmed environment reuse ([[RFC-0002:C-PIXI-ENVIRONMENT-LIFECYCLE]]):
//! a confirmation established against exact manifest and lock content
//! survives an unrelated change and lets the real pixi probe be skipped;
//! content that actually changes invalidates it. Also covers the standalone
//! `inferlab stack status` query this mechanism backs.

use serde_json::Value;
use std::error::Error;
use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use tempfile::TempDir;

struct StatusWorkspace {
    root: TempDir,
    bin: PathBuf,
    pixi_log: PathBuf,
}

impl StatusWorkspace {
    fn new(stacks: &[&str]) -> Result<Self, Box<dyn Error>> {
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
        for stack in stacks {
            fs::create_dir_all(root.path().join(".pixi/envs").join(stack))?;
        }

        let pixi_log = root.path().join("pixi-argv.log");
        write_executable(
            &bin.join("pixi"),
            r#"#!/bin/sh
set -eu
printf '%s\n' "$*" >> "$FAKE_PIXI_LOG"
case "$*" in
  *" -- true") exit "${FAKE_PIXI_PROBE_EXIT:-0}" ;;
esac
if [ -n "${FAKE_PIXI_CHECK_STDOUT:-}" ]; then
  printf '%s\n' "$FAKE_PIXI_CHECK_STDOUT"
fi
if [ -n "${FAKE_PIXI_CHECK_STDERR:-}" ]; then
  printf '%s\n' "$FAKE_PIXI_CHECK_STDERR" >&2
fi
if [ "${FAKE_PIXI_REMOVE_AFTER_CHECK:-0}" = "1" ]; then
  /bin/rm -f "$0"
fi
if [ "${FAKE_PIXI_CHECK_SIGNAL:-}" = "TERM" ]; then
  kill -TERM $$
fi
exit "${FAKE_PIXI_CHECK_EXIT:-0}"
"#,
        )?;

        Ok(Self {
            root,
            bin,
            pixi_log,
        })
    }

    fn run(&self, args: &[&str]) -> Result<Output, Box<dyn Error>> {
        self.run_with_env(&[], args)
    }

    fn run_with_env(&self, envs: &[(&str, &str)], args: &[&str]) -> Result<Output, Box<dyn Error>> {
        let mut command = Command::new(env!("CARGO_BIN_EXE_inferlab"));
        command
            .current_dir(self.root.path())
            .env("PATH", &self.bin)
            .env("FAKE_PIXI_LOG", &self.pixi_log);
        for (name, value) in envs {
            command.env(name, value);
        }
        Ok(command.args(args).output()?)
    }

    fn declare_check(
        &self,
        stack: &str,
        id: &str,
        script: &str,
        repair_hint: Option<&str>,
    ) -> Result<(), Box<dyn Error>> {
        let workspace_path = self.root.path().join(".inferlab/workspace.toml");
        let mut workspace = fs::read_to_string(&workspace_path)?;
        workspace.push_str(&format!(
            "\n[[stacks.{stack}.checks]]\nid = \"{id}\"\nscript = \"{script}\"\n"
        ));
        if let Some(repair_hint) = repair_hint {
            workspace.push_str(&format!("repair_hint = \"{repair_hint}\"\n"));
        }
        fs::write(workspace_path, workspace)?;
        let script_path = self.root.path().join(script);
        if let Some(parent) = script_path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(script_path, "print('fixture check')\n")?;
        Ok(())
    }

    fn pixi_argv(&self) -> String {
        fs::read_to_string(&self.pixi_log).unwrap_or_default()
    }

    fn probe_count(&self) -> usize {
        self.pixi_argv().lines().count()
    }
}

fn write_executable(path: &Path, content: &str) -> Result<(), Box<dyn Error>> {
    fs::write(path, content)?;
    let mut permissions = fs::metadata(path)?.permissions();
    permissions.set_mode(0o755);
    fs::set_permissions(path, permissions)?;
    Ok(())
}

fn stderr(output: &Output) -> String {
    String::from_utf8_lossy(&output.stderr).into_owned()
}

fn stdout_json(output: &Output) -> Result<Value, Box<dyn Error>> {
    Ok(serde_json::from_slice(&output.stdout)?)
}

#[test]
fn stack_status_reports_confirmed_and_exits_zero_without_local_bindings()
-> Result<(), Box<dyn Error>> {
    let workspace = StatusWorkspace::new(&["vllm"])?;
    // No .inferlab/local.toml was written: stack status must not require it.
    let output = workspace.run(&["stack", "status"])?;
    assert!(output.status.success(), "{}", stderr(&output));
    let report = stdout_json(&output)?;
    assert_eq!(report[0]["stack"], "vllm");
    assert_eq!(report[0]["pixi_environment"], "vllm");
    assert_eq!(report[0]["status"], "confirmed");
    let entry = report[0]
        .as_object()
        .ok_or("report entry must be an object")?;
    assert!(!entry.contains_key("diagnostics"));
    assert!(!entry.contains_key("install_command"));
    assert_eq!(report[0]["checks"]["state"], "not-declared");
    assert_eq!(report[0]["checks"]["evidence"], Value::Array(Vec::new()));
    let checks = report[0]["checks"]
        .as_object()
        .ok_or("checks must be an object")?;
    assert!(!checks.contains_key("error"));
    assert_eq!(report[0]["ready"], true);
    Ok(())
}

#[test]
fn stack_status_runs_declared_checks_and_reports_completed_evidence() -> Result<(), Box<dyn Error>>
{
    let workspace = StatusWorkspace::new(&["vllm"])?;
    workspace.declare_check("vllm", "native-schema", "checks/native_schema.py", None)?;

    let output = workspace.run_with_env(
        &[("FAKE_PIXI_CHECK_STDOUT", "native schema is current")],
        &["stack", "status"],
    )?;
    assert!(output.status.success(), "{}", stderr(&output));
    let report = stdout_json(&output)?;
    assert_eq!(report[0]["status"], "confirmed");
    assert_eq!(report[0]["checks"]["state"], "passed");
    assert_eq!(report[0]["checks"]["evidence"][0]["id"], "native-schema");
    assert_eq!(
        report[0]["checks"]["evidence"][0]["realization"],
        "local-workspace"
    );
    assert_eq!(report[0]["checks"]["evidence"][0]["outcome"], "passed");
    assert_eq!(
        report[0]["checks"]["evidence"][0]["output"],
        "native schema is current\n"
    );
    assert!(report[0]["checks"]["evidence"][0]["repair_hint"].is_null());
    assert!(report[0]["checks"]["error"].is_null());
    assert_eq!(report[0]["ready"], true);
    Ok(())
}

#[test]
fn stack_status_preserves_complete_check_output() -> Result<(), Box<dyn Error>> {
    let workspace = StatusWorkspace::new(&["vllm"])?;
    workspace.declare_check("vllm", "verbose", "checks/verbose.py", None)?;
    let stdout = "x".repeat(5000);
    let stderr_text = "stderr keeps trailing spaces  ";

    let output = workspace.run_with_env(
        &[
            ("FAKE_PIXI_CHECK_STDOUT", &stdout),
            ("FAKE_PIXI_CHECK_STDERR", stderr_text),
        ],
        &["stack", "status"],
    )?;
    assert!(output.status.success(), "{}", stderr(&output));
    let report = stdout_json(&output)?;
    assert_eq!(
        report[0]["checks"]["evidence"][0]["output"],
        format!("{stdout}\n{stderr_text}\n")
    );
    Ok(())
}

#[test]
fn stack_status_reruns_declared_checks_when_confirmation_is_cached() -> Result<(), Box<dyn Error>> {
    let workspace = StatusWorkspace::new(&["vllm"])?;
    workspace.declare_check("vllm", "native-schema", "checks/native_schema.py", None)?;

    let first = workspace.run(&["stack", "status"])?;
    assert!(first.status.success(), "{}", stderr(&first));
    let second = workspace.run(&["stack", "status"])?;
    assert!(second.status.success(), "{}", stderr(&second));
    assert_eq!(
        workspace
            .pixi_argv()
            .lines()
            .filter(|line| line.contains("checks/native_schema.py"))
            .count(),
        2
    );
    assert_eq!(
        workspace
            .pixi_argv()
            .lines()
            .filter(|line| line.ends_with("-- true"))
            .count(),
        1
    );
    Ok(())
}

#[test]
fn a_confirmation_survives_an_unrelated_change_and_skips_the_real_probe()
-> Result<(), Box<dyn Error>> {
    let workspace = StatusWorkspace::new(&["vllm"])?;
    let first = workspace.run(&["stack", "status"])?;
    assert!(first.status.success(), "{}", stderr(&first));
    assert_eq!(
        workspace.probe_count(),
        1,
        "the first check must run the real probe once: {}",
        workspace.pixi_argv()
    );

    // A workspace revision change that leaves manifest and lock content
    // unchanged (RFC-0002:C-PIXI-ENVIRONMENT-LIFECYCLE) — simulated here by
    // touching an unrelated file.
    fs::write(workspace.root.path().join("README.md"), "unrelated\n")?;

    let second = workspace.run(&["stack", "status"])?;
    assert!(second.status.success(), "{}", stderr(&second));
    assert_eq!(
        workspace.probe_count(),
        1,
        "a confirmation for unchanged content must skip the real probe: {}",
        workspace.pixi_argv()
    );
    let report = stdout_json(&second)?;
    assert_eq!(report[0]["status"], "confirmed");
    Ok(())
}

#[test]
fn a_manifest_change_invalidates_the_confirmation_and_reruns_the_probe()
-> Result<(), Box<dyn Error>> {
    let workspace = StatusWorkspace::new(&["vllm"])?;
    let first = workspace.run(&["stack", "status"])?;
    assert!(first.status.success(), "{}", stderr(&first));
    assert_eq!(workspace.probe_count(), 1);

    // Content actually changes: the manifest is edited (a hand-edit that
    // was never relocked is exactly the case the dual-hash design exists
    // to keep catching).
    let manifest_path = workspace.root.path().join("pixi.toml");
    let manifest = fs::read_to_string(&manifest_path)?;
    fs::write(&manifest_path, format!("{manifest}\n# edited\n"))?;

    let second = workspace.run(&["stack", "status"])?;
    assert!(second.status.success(), "{}", stderr(&second));
    assert_eq!(
        workspace.probe_count(),
        2,
        "changed manifest content must invalidate the stale confirmation and rerun the probe: {}",
        workspace.pixi_argv()
    );
    let report = stdout_json(&second)?;
    assert_eq!(report[0]["status"], "confirmed");
    Ok(())
}

#[test]
fn a_lock_change_invalidates_the_confirmation_and_reruns_the_probe() -> Result<(), Box<dyn Error>> {
    let workspace = StatusWorkspace::new(&["vllm"])?;
    let first = workspace.run(&["stack", "status"])?;
    assert!(first.status.success(), "{}", stderr(&first));
    assert_eq!(workspace.probe_count(), 1);

    let lock_path = workspace.root.path().join("pixi.lock");
    let lock = fs::read_to_string(&lock_path)?;
    fs::write(&lock_path, format!("{lock}# relocked\n"))?;

    let second = workspace.run(&["stack", "status"])?;
    assert!(second.status.success(), "{}", stderr(&second));
    assert_eq!(
        workspace.probe_count(),
        2,
        "changed lock content must invalidate the stale confirmation and rerun the probe: {}",
        workspace.pixi_argv()
    );
    Ok(())
}

#[test]
fn env_status_reports_never_installed_and_not_usable_and_exits_nonzero()
-> Result<(), Box<dyn Error>> {
    let workspace = StatusWorkspace::new(&["vllm", "sglang"])?;
    workspace.declare_check("vllm", "native-schema", "checks/native_schema.py", None)?;
    fs::remove_dir_all(workspace.root.path().join(".pixi/envs/vllm"))?;

    let output = workspace.run_with_env(&[("FAKE_PIXI_PROBE_EXIT", "1")], &["stack", "status"])?;
    assert!(!output.status.success());
    let report = stdout_json(&output)?;
    let entries = report.as_array().ok_or("report must be a JSON array")?;
    let by_stack = |id: &str| -> Result<&Value, Box<dyn Error>> {
        entries
            .iter()
            .find(|entry| entry["stack"] == id)
            .ok_or_else(|| format!("no report entry for {id}").into())
    };
    let vllm = by_stack("vllm")?;
    assert_eq!(vllm["status"], "never-installed");
    let vllm_install_command = vllm["install_command"]
        .as_str()
        .ok_or("install_command must be a string")?;
    assert!(vllm_install_command.contains("vllm"));
    assert_eq!(vllm["checks"]["state"], "skipped");
    assert_eq!(vllm["checks"]["evidence"], Value::Array(Vec::new()));
    assert_eq!(vllm["ready"], false);
    assert!(!workspace.pixi_argv().contains("checks/native_schema.py"));

    let sglang = by_stack("sglang")?;
    assert_eq!(sglang["status"], "not-usable");
    assert!(sglang["diagnostics"].is_string());
    let sglang_install_command = sglang["install_command"]
        .as_str()
        .ok_or("install_command must be a string")?;
    assert!(sglang_install_command.contains("sglang"));
    assert_eq!(sglang["checks"]["state"], "not-declared");
    assert_eq!(sglang["ready"], false);

    // No marker persists for either failure: a future check must retry
    // rather than trust a failed probe.
    assert!(
        !workspace
            .root
            .path()
            .join(".inferlab/cache/environments/sglang/confirmed.json")
            .exists()
    );
    Ok(())
}

#[test]
fn stack_status_stops_at_the_first_failed_check_and_reports_its_repair_hint()
-> Result<(), Box<dyn Error>> {
    let workspace = StatusWorkspace::new(&["vllm"])?;
    workspace.declare_check(
        "vllm",
        "native-schema",
        "checks/native_schema.py",
        Some("pixi run rebuild-native"),
    )?;
    workspace.declare_check("vllm", "later-check", "checks/later.py", None)?;

    let output = workspace.run_with_env(
        &[
            ("FAKE_PIXI_CHECK_EXIT", "9"),
            ("FAKE_PIXI_CHECK_STDOUT", "native schema is stale"),
        ],
        &["stack", "status"],
    )?;
    assert!(!output.status.success());
    let report = stdout_json(&output)?;
    assert_eq!(report[0]["status"], "confirmed");
    assert_eq!(report[0]["checks"]["state"], "failed");
    assert_eq!(
        report[0]["checks"]["evidence"].as_array().map(Vec::len),
        Some(1)
    );
    assert_eq!(report[0]["checks"]["evidence"][0]["id"], "native-schema");
    assert_eq!(report[0]["checks"]["evidence"][0]["outcome"], "failed");
    assert_eq!(
        report[0]["checks"]["evidence"][0]["repair_hint"],
        "pixi run rebuild-native"
    );
    assert!(report[0]["checks"]["error"].is_null());
    assert_eq!(report[0]["ready"], false);
    assert!(!workspace.pixi_argv().contains("checks/later.py"));
    assert!(stderr(&output).contains("not ready"));
    Ok(())
}

#[test]
fn stack_status_continues_after_an_earlier_stack_check_fails() -> Result<(), Box<dyn Error>> {
    let workspace = StatusWorkspace::new(&["a", "b"])?;
    workspace.declare_check("a", "native-schema", "checks/native_schema.py", None)?;

    let output = workspace.run_with_env(&[("FAKE_PIXI_CHECK_EXIT", "1")], &["stack", "status"])?;
    assert!(!output.status.success());
    let report = stdout_json(&output)?;
    let entries = report.as_array().ok_or("report must be a JSON array")?;
    assert_eq!(entries.len(), 2);
    assert_eq!(report[0]["stack"], "a");
    assert_eq!(report[0]["checks"]["state"], "failed");
    assert_eq!(report[1]["stack"], "b");
    assert_eq!(report[1]["status"], "confirmed");
    assert_eq!(report[1]["checks"]["state"], "not-declared");
    assert_eq!(report[1]["ready"], true);
    Ok(())
}

#[test]
fn stack_status_reports_check_launch_errors_without_losing_json_output()
-> Result<(), Box<dyn Error>> {
    let workspace = StatusWorkspace::new(&["vllm"])?;
    workspace.declare_check("vllm", "imports", "checks/imports.py", None)?;
    workspace.declare_check("vllm", "native-schema", "checks/native_schema.py", None)?;

    let output = workspace.run_with_env(
        &[
            ("FAKE_PIXI_CHECK_STDOUT", "completed before launch error"),
            ("FAKE_PIXI_REMOVE_AFTER_CHECK", "1"),
        ],
        &["stack", "status"],
    )?;
    assert!(!output.status.success());
    let report = stdout_json(&output)?;
    assert_eq!(report[0]["status"], "confirmed");
    assert_eq!(report[0]["checks"]["state"], "error");
    assert_eq!(
        report[0]["checks"]["evidence"].as_array().map(Vec::len),
        Some(1)
    );
    assert_eq!(report[0]["checks"]["evidence"][0]["id"], "imports");
    assert_eq!(report[0]["checks"]["evidence"][0]["outcome"], "passed");
    assert_eq!(report[0]["checks"]["error"]["id"], "native-schema");
    assert!(report[0]["checks"]["error"]["diagnostics"].is_string());
    assert_eq!(report[0]["ready"], false);
    Ok(())
}

#[test]
fn stack_status_reports_signaled_checks_as_errors_without_outcome_evidence()
-> Result<(), Box<dyn Error>> {
    let workspace = StatusWorkspace::new(&["vllm"])?;
    workspace.declare_check("vllm", "native-schema", "checks/native_schema.py", None)?;

    let output =
        workspace.run_with_env(&[("FAKE_PIXI_CHECK_SIGNAL", "TERM")], &["stack", "status"])?;
    assert!(!output.status.success());
    let report = stdout_json(&output)?;
    assert_eq!(report[0]["status"], "confirmed");
    assert_eq!(report[0]["checks"]["state"], "error");
    assert_eq!(report[0]["checks"]["evidence"], Value::Array(Vec::new()));
    assert_eq!(report[0]["checks"]["error"]["id"], "native-schema");
    assert!(
        report[0]["checks"]["error"]["diagnostics"]
            .as_str()
            .is_some_and(|text| text.contains("signal"))
    );
    assert_eq!(report[0]["ready"], false);
    Ok(())
}

#[test]
fn stack_status_reports_every_stack_when_pixi_confirmation_cannot_launch()
-> Result<(), Box<dyn Error>> {
    let workspace = StatusWorkspace::new(&["a", "b"])?;
    fs::remove_file(workspace.bin.join("pixi"))?;

    let output = workspace.run(&["stack", "status"])?;
    assert!(!output.status.success());
    let report = stdout_json(&output)?;
    let entries = report.as_array().ok_or("report must be a JSON array")?;
    assert_eq!(entries.len(), 2);
    for entry in entries {
        assert_eq!(entry["status"], "not-usable");
        assert!(entry["diagnostics"].is_string());
        assert!(entry["install_command"].is_string());
        assert_eq!(entry["checks"]["state"], "not-declared");
        assert_eq!(entry["ready"], false);
    }
    Ok(())
}

#[test]
fn stack_status_narrows_to_one_declared_stack() -> Result<(), Box<dyn Error>> {
    let workspace = StatusWorkspace::new(&["vllm", "sglang"])?;
    let output = workspace.run(&["stack", "status", "vllm"])?;
    assert!(output.status.success(), "{}", stderr(&output));
    let report = stdout_json(&output)?;
    let entries = report.as_array().ok_or("report must be a JSON array")?;
    assert_eq!(entries.len(), 1);
    assert_eq!(report[0]["stack"], "vllm");

    let unknown = workspace.run(&["stack", "status", "missing"])?;
    assert!(!unknown.status.success());
    Ok(())
}

// `inferlab run` deliberately does NOT share ensure_usable's
// confirmation-marker path ([[RFC-0002:C-ADHOC-EXECUTION]]) — that
// isolation, in both directions, is covered in tests/run.rs:
// local_run_neither_trusts_nor_produces_confirmation_evidence.
