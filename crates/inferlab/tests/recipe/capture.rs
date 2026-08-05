use crate::harness::{TestWorkspace, process_evidence, resolved_ranks};
use crate::support;
use serde_json::Value;
use std::error::Error;
use std::fs;

impl TestWorkspace {
    fn configure_readiness_timeout(&self, seconds: u64) -> Result<(), Box<dyn Error>> {
        let manifest = self.root().join(".inferlab/workspace.toml");
        let text = fs::read_to_string(&manifest)?.replacen(
            "readiness_timeout_seconds = 900",
            &format!("readiness_timeout_seconds = {seconds}"),
            1,
        );
        fs::write(manifest, text)?;
        Ok(())
    }

    fn configure_c8k_without_reset(&self, timeout_seconds: u64) -> Result<(), Box<dyn Error>> {
        let manifest = self.root().join(".inferlab/workspace.toml");
        let text = fs::read_to_string(&manifest)?;
        let (prefix, bench_and_rest) = text
            .split_once("[benches.c8k1k]\n")
            .ok_or("fixture has no c8k1k Bench section")?;
        let bench_and_rest = bench_and_rest
            .replacen("reset_prefix_cache = true", "reset_prefix_cache = false", 1)
            .replacen(
                "timeout_seconds = 900",
                &format!("timeout_seconds = {timeout_seconds}"),
                1,
            );
        let text = format!("{prefix}[benches.c8k1k]\n{bench_and_rest}");
        fs::write(manifest, text)?;
        Ok(())
    }

    fn configure_c8k_warmup(&self) -> Result<(), Box<dyn Error>> {
        let manifest = self.root().join(".inferlab/workspace.toml");
        let text = fs::read_to_string(&manifest)?.replacen(
            "prompts_per_concurrency = 4",
            "prompts_per_concurrency = 4\nwarmup_prompts_per_concurrency = 2",
            1,
        );
        fs::write(manifest, text)?;
        Ok(())
    }

    fn configure_capture_deadline(&self, seconds: u64) -> Result<(), Box<dyn Error>> {
        let manifest = self.root().join(".inferlab/workspace.toml");
        let text = fs::read_to_string(&manifest)?.replacen(
            "readiness_timeout_seconds = 900",
            &format!(
                "readiness_timeout_seconds = 900\ncapture_control_deadline_seconds = {seconds}"
            ),
            1,
        );
        fs::write(manifest, text)?;
        Ok(())
    }

    fn configure_capture_finalization_deadline(&self, seconds: u64) -> Result<(), Box<dyn Error>> {
        let manifest = self.root().join(".inferlab/workspace.toml");
        let text = fs::read_to_string(&manifest)?.replacen(
            "readiness_timeout_seconds = 900",
            &format!(
                "readiness_timeout_seconds = 900\ncapture_finalization_deadline_seconds = {seconds}"
            ),
            1,
        );
        fs::write(manifest, text)?;
        Ok(())
    }
}

#[test]
fn recipe_captures_one_selected_bench_and_verifies_static_ranges() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    let output = workspace
        .command()
        .args(["recipe", "run", "dsv4-qualify", "--capture", "c8k1k"])
        .output()?;

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let recipe: Value = serde_json::from_slice(&output.stdout)?;
    let bench_id = recipe["benches"][0]["id"]
        .as_str()
        .ok_or("captured Bench has no record id")?;
    let bench = workspace.load_record(bench_id)?;
    assert_eq!(bench["capture"]["status"], "succeeded");
    assert_eq!(bench["capture"]["plan"]["control"], "framework-range");
    assert_eq!(
        bench["capture"]["plan"]["deadlines"],
        serde_json::json!({
            "capture_arm_deadline_seconds": 60,
            "capture_control_deadline_seconds": 60,
            "capture_finalization_deadline_seconds": 300,
        })
    );
    assert_eq!(
        bench["capture"]["windows"].as_array().map(Vec::len),
        Some(4)
    );
    assert!(bench["capture"]["reports"].as_array().is_some_and(
        |reports| reports.len() == 4 && reports.iter().all(|report| report["verified"] == true)
    ));
    let finalization = bench["capture"]["finalization"]
        .as_array()
        .and_then(|actions| actions.first())
        .ok_or("captured Bench has no collection-finalization evidence")?;
    assert_eq!(finalization["kind"], "collection_finalization");
    assert_eq!(finalization["operation"], "finalize-collection");
    assert_eq!(finalization["outcome"], "range_end");
    assert_eq!(finalization["observed_state"], "Launched");
    assert_eq!(finalization["range_end"]["window_id"], "request-rate-001");
    assert_eq!(finalization["range_end"]["range_index"], 4);
    assert_eq!(finalization["range_end"]["expected_range_count"], 4);
    assert_eq!(
        finalization["inspection"]["operation"],
        "inspect-collection-state"
    );
    assert_eq!(finalization["stop"], Value::Null);
    assert_eq!(finalization["succeeded"], true);
    assert!(!fs::read_to_string(workspace.capture_events())?.contains("nsys_stop"));
    let server_id = recipe["server"]["id"]
        .as_str()
        .ok_or("recipe has no server record id")?;
    let server = workspace.load_record(server_id)?;
    let server_ranks = resolved_ranks(&server["resolved"]["server"])?;
    let server_evidence = process_evidence(&server, "server")?;
    assert_eq!(server_ranks[0].role_id, "serve");
    assert_eq!(server_evidence["profiler"]["executable"], "nsys");
    assert_eq!(
        server["resolved"]["server"]["capture_arm_deadline_seconds"],
        60
    );
    assert_eq!(
        server["resolved"]["server"]["capture_control_deadline_seconds"],
        60
    );
    assert_eq!(
        server["resolved"]["server"]["capture_finalization_deadline_seconds"],
        300
    );
    let control = &server_evidence["profiler"]["control"];
    let endpoint = &server_ranks[0].rank.endpoint;
    assert_eq!(control["window_control_endpoint"], "replica_entry");
    assert_eq!(control["process_id"], "server");
    assert_eq!(control["start"]["method"], "post");
    assert_eq!(
        control["start"]["body"],
        serde_json::json!({"activities": ["CUDA_PROFILER"]})
    );
    assert_eq!(
        control["start"]["effective_url"],
        format!("http://{}:{}/start_profile", endpoint.host, endpoint.port)
    );
    assert_eq!(control["stop"]["method"], "post");
    assert!(control["stop"].get("body").is_none());
    assert_eq!(
        control["stop"]["effective_url"],
        format!("http://{}:{}/stop_profile", endpoint.host, endpoint.port)
    );
    let server_finalization = &server_evidence["profiler_finalization"];
    assert_eq!(server_finalization["kind"], "collection_finalization");
    assert_eq!(server_finalization["operation"], "finalize-collection");
    assert_eq!(server_finalization["outcome"], "inactive");
    assert_eq!(server_finalization["observed_state"], "Launched");
    assert_eq!(server_finalization["range_end"], Value::Null);
    assert_eq!(server_finalization["stop"], Value::Null);
    assert_eq!(server_finalization["succeeded"], true);
    let start_action = &bench["capture"]["windows"][0]["start"][0];
    assert_eq!(start_action["method"], "post");
    assert_eq!(start_action["path"], "/start_profile");
    assert_eq!(
        start_action["body"],
        serde_json::json!({"activities": ["CUDA_PROFILER"]})
    );
    assert_eq!(start_action["status"], 200);
    assert_eq!(start_action["succeeded"], true);
    assert!(start_action.get("failure_kind").is_none());
    assert_eq!(server_evidence["profiler_cleanup"]["verified"], true);
    assert_eq!(server_evidence["profiler_cleanup"]["trigger"], "stop");
    assert_eq!(recipe["cleanup"]["verified"], true);
    // A no-escape server record carries none of the escape fields — exactly
    // the shape written before they existed — so this capture attaching to
    // it is the old-record compatibility proof
    // ([[RFC-0004:C-WORKLOAD-PROFILING]]).
    assert!(server_evidence["profiler"].get("escapes").is_none());
    assert!(
        server["resolved"]["server"]
            .get("profiler_escapes")
            .is_none()
    );
    Ok(())
}

#[test]
fn captured_bench_opens_the_window_after_warmup_and_before_profiling() -> Result<(), Box<dyn Error>>
{
    let workspace = TestWorkspace::new()?;
    workspace.configure_c8k_warmup()?;
    let output = workspace
        .command()
        .args(["recipe", "run", "dsv4-qualify", "--capture", "c8k1k"])
        .output()?;

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let events = fs::read_to_string(workspace.capture_events())?;
    let events = events.lines().collect::<Vec<_>>();
    assert_eq!(
        &events[..4],
        [
            "warmup_complete",
            "capture_open",
            "profiling_started",
            "capture_close",
        ]
    );
    Ok(())
}

#[test]
fn captured_bench_keeps_the_window_closed_when_warmup_fails() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    workspace.configure_capture_finalization_deadline(1)?;
    workspace.configure_c8k_warmup()?;
    workspace.append_manifest(
        "\n[servers.dsv4-qualify.profiler.nsys.env]\n\
         NSYS_FIXTURE = \"fallback\"\n",
    )?;
    let output = workspace
        .command()
        .env("FIXTURE_BENCH_FAIL_BEFORE_PROFILE", "1")
        .args(["recipe", "run", "dsv4-qualify", "--capture", "c8k1k"])
        .output()?;

    assert!(!output.status.success());
    let recipe: Value = serde_json::from_slice(&output.stdout)?;
    let bench = workspace.load_record(
        recipe["benches"][0]["id"]
            .as_str()
            .ok_or("captured Bench has no record id")?,
    )?;
    assert_eq!(bench["cases"][0]["status"], "failed");
    assert_eq!(
        bench["capture"]["windows"][0]["start"],
        serde_json::json!([])
    );
    assert_eq!(
        bench["capture"]["windows"][0]["stop"],
        serde_json::json!([])
    );
    assert_eq!(bench["capture"]["status"], "failed");
    let finalization = &bench["capture"]["finalization"][0];
    assert_eq!(finalization["kind"], "collection_finalization");
    assert_eq!(finalization["outcome"], "stopped");
    assert_eq!(finalization["observed_state"], "StartRange");
    assert_eq!(finalization["range_end"], Value::Null);
    assert_eq!(
        finalization["inspection"]["argv"],
        serde_json::json!([
            "env",
            "--",
            "NSYS_FIXTURE=fallback",
            "nsys",
            "sessions",
            "list",
            "--output-format=json",
        ])
    );
    assert_eq!(finalization["stop"]["operation"], "stop-collection");
    let session = finalization["session"]
        .as_str()
        .ok_or("collection finalization has no session")?;
    assert_eq!(
        finalization["stop"]["argv"],
        serde_json::json!([
            "env",
            "--",
            "NSYS_FIXTURE=fallback",
            "nsys",
            "stop",
            format!("--session={session}"),
        ])
    );
    assert_eq!(finalization["stop"]["succeeded"], true);
    assert_eq!(finalization["succeeded"], true);
    assert!(fs::read_to_string(workspace.capture_events())?.contains("nsys_stop"));
    assert_eq!(recipe["cleanup"]["verified"], true);
    Ok(())
}

#[test]
fn captured_bench_without_reset_starts_its_budget_after_the_window_opens()
-> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    workspace.configure_c8k_without_reset(1)?;
    workspace.configure_capture_deadline(30)?;
    let output = workspace
        .command()
        .env("FIXTURE_START_PROFILE_DELAY_SECONDS", "2")
        .args([
            "recipe",
            "run",
            "dsv4-qualify",
            "--capture",
            "c8k1k",
            "--set",
            "benches.c8k1k.concurrency=[1]",
        ])
        .output()?;

    assert!(
        output.status.success(),
        "profiler window latency must not consume a no-reset case budget: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let recipe: Value = serde_json::from_slice(&output.stdout)?;
    let bench = workspace.load_record(
        recipe["benches"][0]["id"]
            .as_str()
            .ok_or("captured Bench has no record id")?,
    )?;
    assert_eq!(bench["cases"][0]["status"], "succeeded");
    assert_eq!(bench["cases"][0]["process"]["timed_out"], false);
    assert_eq!(bench["capture"]["status"], "succeeded");
    Ok(())
}

/// The declared escapes splice ahead of the managed launch and start tails,
/// the dedicated trace/sampling/context-switch fields replace their managed
/// defaults, the env map leads every managed Nsight command, and the record
/// holds both the raw declaration and the effective invocations
/// ([[RFC-0004:C-WORKLOAD-PROFILING]]).
#[test]
fn capture_renders_declared_escapes_and_records_raw_and_effective() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    workspace.append_manifest(
        "\n[servers.dsv4-qualify.profiler.nsys]\n\
         launch_options = [\"--cuda-graph-trace=node\"]\n\
         start_options = [\"--nic-metrics=true\"]\n\
         trace = [\"cuda\", \"nvtx\"]\n\
         sampling = \"cpu\"\n\
         context_switch = \"process-tree\"\n\
         \n\
         [servers.dsv4-qualify.profiler.nsys.env]\n\
         NSYS_FIXTURE = \"a b\"\n",
    )?;
    let output = workspace
        .command()
        .args(["recipe", "run", "dsv4-qualify", "--capture", "c8k1k"])
        .output()?;
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let recipe: Value = serde_json::from_slice(&output.stdout)?;

    let server = workspace.load_record(
        recipe["server"]["id"]
            .as_str()
            .ok_or("recipe has no server record id")?,
    )?;
    let profiler = &process_evidence(&server, "server")?["profiler"];
    let session = profiler["session"]
        .as_str()
        .ok_or("profiler target has no session")?;
    assert_eq!(
        profiler["launch_prefix"],
        serde_json::json!([
            "env",
            "--",
            "NSYS_FIXTURE=a b",
            "nsys",
            "launch",
            "--cuda-graph-trace=node",
            "--session-new",
            session,
            "--trace=cuda,nvtx",
            "--wait=all",
        ])
    );
    assert_eq!(
        profiler["escapes"]["start_options"][0],
        "--nic-metrics=true"
    );
    assert_eq!(profiler["escapes"]["sampling"], "cpu");
    let raw = &server["resolved"]["server"]["profiler_escapes"]["common"];
    assert_eq!(raw["launch_options"][0], "--cuda-graph-trace=node");
    assert_eq!(raw["trace"], serde_json::json!(["cuda", "nvtx"]));
    assert_eq!(raw["env"]["NSYS_FIXTURE"], "a b");

    let bench = workspace.load_record(
        recipe["benches"][0]["id"]
            .as_str()
            .ok_or("captured Bench has no record id")?,
    )?;
    assert_eq!(bench["capture"]["status"], "succeeded");
    assert!(bench["capture"]["reports"].as_array().is_some_and(
        |reports| reports.len() == 4 && reports.iter().all(|report| report["verified"] == true)
    ));
    let output_base = bench["capture"]["plan"]["targets"][0]["output_base"]
        .as_str()
        .ok_or("capture plan has no output base")?;
    let start = bench["capture"]["arm"]
        .as_array()
        .and_then(|actions| {
            actions
                .iter()
                .find(|action| action["operation"] == "start-range-collection")
        })
        .ok_or("capture armed no range collection")?;
    assert_eq!(
        start["argv"],
        serde_json::json!([
            "env",
            "--",
            "NSYS_FIXTURE=a b",
            "nsys",
            "start",
            "--nic-metrics=true",
            format!("--session={session}"),
            "--sample=cpu",
            "--cpuctxsw=process-tree",
            "--force-overwrite=true",
            "--export=none",
            format!("--output={output_base}"),
            "--capture-range=cudaProfilerApi",
            "--capture-range-end=repeat:4:async",
        ])
    );
    let expected_inspection = serde_json::json!([
        "env",
        "--",
        "NSYS_FIXTURE=a b",
        "nsys",
        "sessions",
        "list",
        "--output-format=json",
    ]);
    assert_eq!(
        bench["capture"]["finalization"][0]["inspection"]["argv"],
        expected_inspection
    );
    assert_eq!(
        process_evidence(&server, "server")?["profiler_finalization"]["inspection"]["argv"],
        expected_inspection
    );
    Ok(())
}

/// Role escapes merge into common server escapes in the resolved plan: scalars
/// replace, option lists concatenate with the role's after the common values,
/// and env entries merge with the role value winning; the raw declaration
/// keeps both layers ([[RFC-0004:C-WORKLOAD-PROFILING]]).
#[test]
fn role_escapes_merge_over_common_server_escapes_in_the_resolved_plan() -> Result<(), Box<dyn Error>>
{
    let workspace = TestWorkspace::new()?;
    workspace.configure_pd("nixl")?;
    workspace.append_manifest(
        "\n[servers.dsv4-qualify.profiler.nsys]\n\
         launch_options = [\"--cuda-graph-trace=node\"]\n\
         sampling = \"cpu\"\n\
         \n\
         [servers.dsv4-qualify.profiler.nsys.env]\n\
         NSYS_SHARED = \"profile\"\n\
         NSYS_PROFILE_ONLY = \"1\"\n\
         \n\
         [servers.dsv4-qualify.roles.prefill.profiler.nsys]\n\
         launch_options = [\"--nvtx-domain-include=prefill\"]\n\
         sampling = \"process-tree\"\n\
         \n\
         [servers.dsv4-qualify.roles.prefill.profiler.nsys.env]\n\
         NSYS_SHARED = \"role\"\n",
    )?;
    let output = workspace
        .command()
        .env("FIXTURE_PD", "nixl")
        .args([
            "recipe",
            "run",
            "dsv4-qualify",
            "--capture",
            "c8k1k",
            "--dry-run",
        ])
        .output()?;
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let plan: Value = serde_json::from_slice(&output.stdout)?;
    let processes = resolved_ranks(&plan["server"])?;
    let escapes_of = |role: &str| -> Result<&support::NsysEscapesProjection, Box<dyn Error>> {
        processes
            .iter()
            .find(|process| process.role_id == role)
            .ok_or_else(|| format!("plan has no {role} process"))?
            .rank
            .capture_target
            .as_ref()
            .map(|target| &target.escapes)
            .ok_or_else(|| format!("plan has no {role} capture target").into())
    };
    let prefill = escapes_of("prefill")?;
    assert_eq!(
        prefill.launch_options,
        ["--cuda-graph-trace=node", "--nvtx-domain-include=prefill"]
    );
    assert_eq!(prefill.sampling.as_deref(), Some("process-tree"));
    assert_eq!(prefill.env["NSYS_PROFILE_ONLY"], "1");
    assert_eq!(prefill.env["NSYS_SHARED"], "role");
    let decode = escapes_of("decode")?;
    assert_eq!(decode.launch_options, ["--cuda-graph-trace=node"]);
    assert_eq!(decode.sampling.as_deref(), Some("cpu"));
    assert_eq!(decode.env["NSYS_PROFILE_ONLY"], "1");
    assert_eq!(decode.env["NSYS_SHARED"], "profile");
    let raw = &plan["server"]["profiler_escapes"];
    assert_eq!(raw["common"]["sampling"], "cpu");
    assert_eq!(raw["roles"]["prefill"]["sampling"], "process-tree");
    Ok(())
}

/// An escape option naming a managed fact is rejected when the workspace is
/// loaded, naming the escape field and the offending option
/// ([[RFC-0004:C-WORKLOAD-PROFILING]]).
#[test]
fn a_managed_launch_escape_option_is_rejected_at_workspace_load() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    workspace.append_manifest(
        "\n[servers.dsv4-qualify.profiler.nsys]\n\
         launch_options = [\"--wait=none\"]\n",
    )?;
    let output = workspace
        .command()
        .args(["recipe", "run", "dsv4-qualify", "--dry-run"])
        .output()?;
    assert!(!output.status.success());
    Ok(())
}

/// The qualified nsys parses attached short-option values (-cnone is
/// --capture-range=none), so the load gate rejects that form too
/// ([[RFC-0004:C-WORKLOAD-PROFILING]]).
#[test]
fn an_attached_managed_escape_option_is_rejected_at_workspace_load() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    workspace.append_manifest(
        "\n[servers.dsv4-qualify.profiler.nsys]\n\
         start_options = [\"-cnone\"]\n",
    )?;
    let output = workspace
        .command()
        .args(["recipe", "run", "dsv4-qualify", "--dry-run"])
        .output()?;
    assert!(!output.status.success());
    Ok(())
}

/// The qualified nsys resolves GNU-style abbreviated long options
/// (--wai=all runs as --wait), so the load gate rejects strict prefixes of
/// managed names too ([[RFC-0004:C-WORKLOAD-PROFILING]]).
#[test]
fn an_abbreviated_managed_escape_option_is_rejected_at_workspace_load() -> Result<(), Box<dyn Error>>
{
    let workspace = TestWorkspace::new()?;
    workspace.append_manifest(
        "\n[servers.dsv4-qualify.profiler.nsys]\n\
         launch_options = [\"--wai=all\"]\n",
    )?;
    let output = workspace
        .command()
        .args(["recipe", "run", "dsv4-qualify", "--dry-run"])
        .output()?;
    assert!(!output.status.success());
    Ok(())
}

/// A standalone terminator splices ahead of the managed tail and displaces
/// it into positionals of the wrapped command; the start side of the
/// qualified nsys even swallows it silently
/// ([[RFC-0004:C-WORKLOAD-PROFILING]]).
#[test]
fn a_standalone_terminator_escape_is_rejected_at_workspace_load() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    workspace.append_manifest(
        "\n[servers.dsv4-qualify.profiler.nsys]\n\
         launch_options = [\"--\"]\n",
    )?;
    let output = workspace
        .command()
        .args(["recipe", "run", "dsv4-qualify", "--dry-run"])
        .output()?;
    assert!(!output.status.success());
    Ok(())
}

/// A non-identifier env key would be parsed as an option of the environment
/// utility instead of applied as an assignment
/// ([[RFC-0004:C-WORKLOAD-PROFILING]]).
#[test]
fn a_non_identifier_escape_env_key_is_rejected_at_workspace_load() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    workspace.append_manifest(
        "\n[servers.dsv4-qualify.profiler.nsys.env]\n\
         \"--unset\" = \"NSYS_FIXTURE\"\n",
    )?;
    let output = workspace
        .command()
        .args(["recipe", "run", "dsv4-qualify", "--dry-run"])
        .output()?;
    assert!(!output.status.success());
    Ok(())
}

#[test]
fn a_managed_start_escape_option_is_rejected_at_workspace_load() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    workspace.configure_pd("nixl")?;
    workspace.append_manifest(
        "\n[servers.dsv4-qualify.roles.prefill.profiler.nsys]\n\
         start_options = [\"-c=cudaProfilerApi\"]\n",
    )?;
    let output = workspace
        .command()
        .args(["recipe", "run", "dsv4-qualify", "--dry-run"])
        .output()?;
    assert!(!output.status.success());
    Ok(())
}

/// A capture-armed server's readiness wait is unbounded, while the same slow
/// startup without capture keeps the profile budget
/// ([[RFC-0004:C-WORKLOAD-PROFILING]]).
#[test]
fn capture_armed_readiness_outlasts_the_profile_timeout() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    workspace.configure_readiness_timeout(1)?;
    let output = workspace
        .command()
        .env("FIXTURE_READY_DELAY_SECONDS", "3")
        .args(["recipe", "run", "dsv4-qualify", "--capture", "c8k1k"])
        .output()?;
    assert!(
        output.status.success(),
        "a capture-armed server must outlast the profile timeout: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let output = workspace
        .command()
        .env("FIXTURE_READY_DELAY_SECONDS", "3")
        .args(["recipe", "run", "dsv4-qualify"])
        .output()?;
    assert!(!output.status.success());
    let record: Value = serde_json::from_slice(&output.stdout)?;
    let server = workspace.load_record(
        record["server"]["id"]
            .as_str()
            .ok_or("recipe has no server record id")?,
    )?;
    assert_eq!(server["failure"]["phase"], "readiness");
    assert!(
        server["failure"]["message"]
            .as_str()
            .is_some_and(|message| message
                .contains("server did not become ready within 1 seconds")),
        "an uncaptured run keeps the bounded budget: {}",
        server["failure"]
    );
    Ok(())
}

/// The readiness probe cadence backs off instead of polling at a fixed
/// 100ms: a three-second delayed bind records an attempt count consistent
/// with the doubling schedule (~6) rather than fixed-cadence polling (~30).
#[test]
fn readiness_probing_backs_off_for_slow_starts() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    let output = workspace
        .command()
        .env("FIXTURE_READY_DELAY_SECONDS", "3")
        .args(["recipe", "run", "dsv4-qualify"])
        .output()?;
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let record: Value = serde_json::from_slice(&output.stdout)?;
    let server = workspace.load_record(
        record["server"]["id"]
            .as_str()
            .ok_or("recipe has no server record id")?,
    )?;
    let attempts = process_evidence(&server, "server")?["readiness"]["attempts"]
        .as_u64()
        .ok_or("readiness evidence has no attempt count")?;
    assert!(
        (4..=12).contains(&attempts),
        "a 3s wait must record a backed-off attempt count, got {attempts}"
    );
    Ok(())
}

/// The unbounded wait still terminates immediately when the server process
/// group dies; without that exit this test would hang rather than fail.
#[test]
fn capture_armed_readiness_fails_immediately_on_process_exit() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    let output = workspace
        .command()
        .env("FIXTURE_EXIT_BEFORE_READY", "1")
        .args(["recipe", "run", "dsv4-qualify", "--capture", "c8k1k"])
        .output()?;
    assert!(!output.status.success());
    let record: Value = serde_json::from_slice(&output.stdout)?;
    let server = workspace.load_record(
        record["server"]["id"]
            .as_str()
            .ok_or("recipe has no server record id")?,
    )?;
    assert_eq!(server["failure"]["phase"], "readiness");
    assert!(
        server["failure"]["message"]
            .as_str()
            .is_some_and(|message| message.contains("exited before readiness")),
        "process-group exit must fail the unbounded wait: {}",
        server["failure"]
    );
    Ok(())
}

/// Window-opening control keeps a deadline — the server fact
/// `capture_control_deadline_seconds` — because a lost start silently shifts
/// range identities ([[RFC-0004:C-WORKLOAD-PROFILING]]).
#[test]
fn capture_control_deadline_bounds_slow_window_starts() -> Result<(), Box<dyn Error>> {
    let slow = TestWorkspace::new()?;
    slow.configure_capture_deadline(1)?;
    slow.configure_capture_finalization_deadline(1)?;
    let output = slow
        .command()
        .env("FIXTURE_START_PROFILE_DELAY_SECONDS", "2")
        .args(["recipe", "run", "dsv4-qualify", "--capture", "c8k1k"])
        .output()?;
    assert!(!output.status.success());
    let record: Value = serde_json::from_slice(&output.stdout)?;
    let bench = slow.load_record(
        record["benches"][0]["id"]
            .as_str()
            .ok_or("captured Bench has no record id")?,
    )?;
    assert_eq!(bench["capture"]["status"], "failed");
    assert!(
        bench["capture"]["windows"][0]["start"][0]["error"]
            .as_str()
            .is_some_and(|error| error.contains("profiler control deadline expired")),
        "a window start slower than the deadline must fail the capture: {}",
        bench["capture"]
    );

    let raised = TestWorkspace::new()?;
    raised.configure_capture_deadline(30)?;
    let output = raised
        .command()
        .env("FIXTURE_START_PROFILE_DELAY_SECONDS", "2")
        .args(["recipe", "run", "dsv4-qualify", "--capture", "c8k1k"])
        .output()?;
    assert!(
        output.status.success(),
        "raising the deadline above the response delay must succeed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    Ok(())
}

/// A window-closing control failure is evidence, not a verdict: with every
/// required report verified the capture succeeds and carries the failed stop
/// actions ([[RFC-0004:C-WORKLOAD-PROFILING]]).
#[test]
fn failed_window_stop_is_adjudicated_by_report_coverage() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    let output = workspace
        .command()
        .env("FIXTURE_STOP_PROFILE_FAIL", "1")
        .args(["recipe", "run", "dsv4-qualify", "--capture", "c8k1k"])
        .output()?;
    assert!(
        output.status.success(),
        "verified reports must adjudicate a failed stop: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let recipe: Value = serde_json::from_slice(&output.stdout)?;
    let bench_id = recipe["benches"][0]["id"]
        .as_str()
        .ok_or("captured Bench has no record id")?;
    let bench = workspace.load_record(bench_id)?;
    assert_eq!(bench["capture"]["status"], "succeeded");
    assert_eq!(
        bench["capture"]["windows"][0]["stop"][0]["succeeded"],
        false
    );
    assert_eq!(bench["capture"]["windows"][0]["stop"][0]["status"], 500);
    assert!(
        bench["capture"]["reports"]
            .as_array()
            .is_some_and(|reports| reports.iter().all(|report| report["verified"] == true))
    );
    Ok(())
}

/// With a report missing, the same failed stop makes the capture fail
/// carrying both the coverage failure and the control failure as evidence.
#[test]
fn failed_window_stop_with_missing_report_fails_with_both_evidences() -> Result<(), Box<dyn Error>>
{
    let workspace = TestWorkspace::new()?;
    workspace.configure_capture_finalization_deadline(1)?;
    let output = workspace
        .command()
        .env("FIXTURE_STOP_PROFILE_FAIL", "1")
        .env("FIXTURE_STOP_PROFILE_SKIP_REPORT", "1")
        .args(["recipe", "run", "dsv4-qualify", "--capture", "c8k1k"])
        .output()?;
    assert!(!output.status.success());
    let record: Value = serde_json::from_slice(&output.stdout)?;
    let bench = workspace.load_record(
        record["benches"][0]["id"]
            .as_str()
            .ok_or("captured Bench has no record id")?,
    )?;
    assert_eq!(bench["capture"]["status"], "failed");
    let error = bench["capture"]["error"]
        .as_str()
        .ok_or("failed capture has no error")?
        .to_owned();
    assert!(
        error.contains("missing Nsight Systems report"),
        "coverage failure must surface: {error}"
    );
    assert!(
        error.contains("a window-closing control action had failed"),
        "the stop failure must ride along as evidence: {error}"
    );
    Ok(())
}
