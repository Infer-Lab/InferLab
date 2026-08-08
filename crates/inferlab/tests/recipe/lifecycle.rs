use crate::harness::{
    TestWorkspace, process_evidence, resolved_ranks, wait_for_path, write_executable,
};
use crate::support;
use serde_json::Value;
use std::error::Error;
use std::fs;
use std::process::Stdio;
use std::time::Duration;

const ENVIRONMENT_CHECK: &str = include_str!("../fixtures/bin/recipe-environment-check.py");
const PYTHON_SHIM: &str = include_str!("../fixtures/bin/recipe-python.sh");

impl TestWorkspace {
    /// Declare one realization check on the serving stack whose script
    /// exits with the given code ([[RFC-0002:C-ENVIRONMENT-CHECKS]]).
    fn declare_environment_check(&self, exit_code: i32) -> Result<(), Box<dyn Error>> {
        fs::create_dir_all(self.root().join("tools"))?;
        write_executable(
            &self.root().join("tools/fixture-check.py"),
            ENVIRONMENT_CHECK,
        )?;
        fs::write(
            self.root().join("tools/fixture-check-exit-code"),
            exit_code.to_string(),
        )?;
        // Checks run as `python <script>`; the test host may only provide
        // `python3`.
        write_executable(&self.bin().join("python"), PYTHON_SHIM)?;
        let manifest = self.root().join(".inferlab/workspace.toml");
        let mut text = fs::read_to_string(&manifest)?;
        text.push_str(
            "\n[[stacks.vllm.checks]]\n\
             id = \"fixture-guard\"\n\
             script = \"tools/fixture-check.py\"\n\
             repair_hint = \"pixi run fixture-repair\"\n",
        );
        fs::write(manifest, text)?;
        Ok(())
    }
}

#[test]
fn recipe_runs_eval_and_bench_then_stops_the_server() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    let output = workspace.run()?;

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let record: Value = serde_json::from_slice(&output.stdout)?;
    assert_eq!(record["schema_version"], 3);
    let id = record["id"].as_str().ok_or("missing recipe record id")?;
    assert_datetime_record_id(id, "recipe-dsv4-qualify-tp2")?;
    let server_id = record["server"]["id"]
        .as_str()
        .ok_or("missing server record id")?;
    assert_datetime_record_id(server_id, "serve-dsv4-qualify-tp2")?;
    assert_eq!(record["status"], "succeeded");
    assert_eq!(record["evals"].as_array().map(Vec::len), Some(2));
    assert_eq!(record["benches"].as_array().map(Vec::len), Some(2));
    assert_eq!(record["evals"][0]["id"], format!("{id}-eval-000-smoke"));
    assert_eq!(record["evals"][1]["id"], format!("{id}-eval-001-gsm8k"));
    assert_eq!(record["benches"][0]["id"], format!("{id}-bench-000-c8k1k"));
    assert_eq!(
        record["benches"][1]["id"],
        format!("{id}-bench-001-adaptive-c8k1k")
    );
    assert!(
        record["evals"]
            .as_array()
            .is_some_and(|children| children.iter().all(|child| child["status"] == "succeeded"))
    );
    assert!(
        record["benches"]
            .as_array()
            .is_some_and(|children| children.iter().all(|child| child["status"] == "succeeded"))
    );
    assert_eq!(record["server"]["status"], "stopped");
    assert_eq!(record["cleanup"]["verified"], true);
    assert_eq!(
        record["resolved"]["measurements"]["evals"][0]["execution"]["kind"],
        "native_openai_smoke"
    );
    assert_eq!(
        record["resolved"]["measurements"]["evals"][1]["execution"]["toolchain"]["lm_eval_version"],
        "0.4.12"
    );
    let matrix_id = record["benches"][0]["id"]
        .as_str()
        .ok_or("missing matrix bench record id")?;
    let matrix = workspace.load_record(matrix_id)?;
    assert_eq!(matrix["schema_version"], 14);
    assert_eq!(matrix["kind"], "bench");
    assert_eq!(matrix["passed"], true);
    assert!(
        matrix["cases"]
            .as_array()
            .is_some_and(|cases| { cases.iter().all(|case| case.get("slo").is_none()) })
    );
    assert_eq!(
        matrix["cases"][0]["cache_preparation"]["reset"]["succeeded"],
        true
    );
    assert!(matrix["cases"][0].get("eval_gate").is_none());
    assert!(matrix["cases"][0].get("eval_trial_summary").is_none());
    assert!(
        matrix["cases"][0]["cache_preparation"]["reset"]
            .get("status")
            .is_none()
    );
    let adaptive_id = record["benches"][1]["id"]
        .as_str()
        .ok_or("missing adaptive bench record id")?;
    let adaptive = workspace.load_record(adaptive_id)?;
    assert_eq!(adaptive["summary"]["policy"], "highest-feasible-rate-v1");
    assert_eq!(adaptive["summary"]["boundary_bracketed"], true);
    assert_eq!(
        adaptive["summary"]["normal_termination_reason"],
        "search_budget_exhausted"
    );
    let case_ids = adaptive["summary"]["case_ids"]
        .as_array()
        .ok_or("adaptive summary has no case_ids array")?;
    assert_eq!(
        case_ids.len(),
        adaptive["cases"].as_array().map_or(0, Vec::len)
    );
    assert!(adaptive["cases"].as_array().is_some_and(|cases| {
        cases.iter().all(|case| {
            case["status"] == "succeeded" && case["slo"]["request_slo"]["ratio_outcome"] == "passed"
        })
    }));
    assert_eq!(adaptive["summary"]["selected_rate"], 8.0);
    assert!(workspace.bench_marker().is_file());
    Ok(())
}

#[test]
fn mooncake_pd_recipe_uses_one_public_endpoint_and_one_lifecycle() -> Result<(), Box<dyn Error>> {
    run_pd_recipe("mooncake")
}

#[test]
fn nixl_pd_recipe_uses_one_public_endpoint_and_one_lifecycle() -> Result<(), Box<dyn Error>> {
    run_pd_recipe("nixl")
}

fn run_pd_recipe(transport: &str) -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    workspace.configure_pd(transport)?;
    let output = workspace
        .command()
        .env("FIXTURE_PD", transport)
        .args(["recipe", "run", "dsv4-qualify"])
        .output()?;
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let recipe: Value = serde_json::from_slice(&output.stdout)?;
    let server_id = recipe["server"]["id"]
        .as_str()
        .ok_or("recipe has no server record id")?;
    let server = workspace.load_record(server_id)?;
    let processes = resolved_ranks(&server["resolved"]["server"])?;
    let frontend = support::resolved_frontend(&server["resolved"]["server"])?;
    let process_evidence = server["process_evidence"]
        .as_object()
        .ok_or("server has no process evidence")?;
    assert_eq!(server["resolved"]["server"]["topology"], "prefill_decode");
    assert_eq!(processes.len(), 4);
    assert_eq!(process_evidence.len(), 5);
    assert_eq!(processes[0].replica_id, "prefill-000");
    assert_eq!(processes[1].replica_id, "prefill-001");
    assert_eq!(processes[2].replica_id, "decode-000");
    assert_eq!(processes[3].replica_id, "decode-001");
    assert_eq!(frontend.id, "gateway");
    assert_eq!(frontend.components, ["gateway", "pd_router"]);
    assert!(process_evidence.contains_key("gateway"));

    // The resolved plan wires the KV transfer for exactly the selected
    // transport: a mooncake break fails only the mooncake case and a nixl
    // break fails only the nixl case ([[RFC-0003:C-SERVE-TOPOLOGY]]). The
    // discriminating facts are the kv_transfer mechanism, the transport-
    // specific side link, the per-transport process port names, and the
    // concrete frontend command.
    let links = server["resolved"]["server"]["links"]
        .as_array()
        .ok_or("resolved plan has no links")?;
    let kv_mechanism = links
        .iter()
        .find(|link| link["kind"] == "kv_transfer")
        .and_then(|link| link["mechanism"].as_str());
    assert_eq!(
        kv_mechanism,
        Some(transport),
        "the kv_transfer link records the {transport} mechanism: {links:?}"
    );
    // The rendered command and port allocation live on the resolved hierarchy,
    // ordered prefill and decode, plus the independently stored frontend.
    let frontend_argv = &frontend.command.argv;
    let prefill_ports = |replica_index: usize| {
        processes[replica_index]
            .rank
            .ports
            .keys()
            .cloned()
            .collect::<Vec<_>>()
    };
    match transport {
        "mooncake" => {
            // Mooncake bootstraps prefill replicas through the P/D Router.
            assert!(
                links
                    .iter()
                    .any(|link| link["kind"] == "bootstrap" && link["target"] == "prefill"),
                "mooncake declares a bootstrap link: {links:?}"
            );
            assert!(
                !links.iter().any(|link| link["kind"] == "side_channel"),
                "mooncake does not declare a nixl side-channel link: {links:?}"
            );
            assert!(
                prefill_ports(0).contains(&"bootstrap".to_owned()),
                "a mooncake prefill replica exposes a bootstrap port: {:?}",
                prefill_ports(0)
            );
            assert!(
                frontend_argv.iter().any(|arg| arg == "vllm-mooncake"),
                "the frontend launches the mooncake implementation: {frontend_argv:?}"
            );
        }
        "nixl" => {
            // NIXL exchanges KV over a prefill/decode side channel.
            assert!(
                links.iter().any(|link| link["kind"] == "side_channel"
                    && link["source"] == "prefill"
                    && link["target"] == "decode"),
                "nixl declares a side-channel link: {links:?}"
            );
            assert!(
                !links.iter().any(|link| link["kind"] == "bootstrap"),
                "nixl does not declare a mooncake bootstrap link: {links:?}"
            );
            assert!(
                prefill_ports(0).contains(&"side_channel".to_owned()),
                "a nixl prefill replica exposes a side-channel port: {:?}",
                prefill_ports(0)
            );
            assert!(
                frontend_argv.iter().any(|arg| arg == "vllm-nixl"),
                "the frontend launches the nixl implementation: {frontend_argv:?}"
            );
        }
        other => return Err(format!("unhandled transport {other}").into()),
    }

    // configure_pd selects an uncontrolled cache start, so the matrix Bench records
    // the reset as skipped: no per-case prefix-cache reset action ran, unlike
    // the enabled path where each case carries a succeeded reset.
    let matrix_id = recipe["benches"][0]["id"]
        .as_str()
        .ok_or("recipe has no matrix Bench record id")?;
    let matrix = workspace.load_record(matrix_id)?;
    assert_eq!(
        matrix["resolved"]["client"]["effective_definition"]["cache_start"],
        "uncontrolled"
    );
    assert_eq!(
        matrix["resolved"]["client"]["prefix_cache_reset"],
        Value::Null
    );
    assert!(
        matrix["cases"]
            .as_array()
            .is_some_and(|cases| !cases.is_empty()
                && cases
                    .iter()
                    .all(|case| case["cache_preparation"] == Value::Null)),
        "with reset disabled every matrix case skips the prefix-cache reset: {}",
        matrix["cases"]
    );

    let public_port = server["resolved"]["server"]["endpoint"]["port"].clone();
    let eval_id = recipe["evals"][0]["id"]
        .as_str()
        .ok_or("recipe has no Eval record id")?;
    let eval = workspace.load_record(eval_id)?;
    assert_eq!(eval["resolved"]["endpoint"]["port"], public_port);
    assert!(process_evidence.values().all(|evidence| {
        evidence["cleanup"]
            .as_array()
            .and_then(|cleanup| cleanup.last())
            .is_some_and(|cleanup| cleanup["verified"] == true)
    }));
    assert_eq!(recipe["cleanup"]["verified"], true);
    Ok(())
}

#[test]
fn manual_bench_attaches_to_an_explicit_running_server() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    let start = workspace
        .command()
        .args(["serve", "start", "dsv4-qualify"])
        .output()?;
    assert!(
        start.status.success(),
        "{}",
        String::from_utf8_lossy(&start.stderr)
    );
    let server: Value = serde_json::from_slice(&start.stdout)?;
    let server_id = server["id"].as_str().ok_or("server record has no id")?;
    fs::remove_file(workspace.root().join(".inferlab/local.toml"))?;

    let unavailable_capture = workspace
        .command()
        .args([
            "bench",
            "c8k1k",
            "--serve",
            server_id,
            "--capture",
            "--dry-run",
        ])
        .output()?;
    assert!(!unavailable_capture.status.success());
    assert!(
        String::from_utf8_lossy(&unavailable_capture.stderr)
            .contains("was not started with profiling target preparation")
    );

    let dry_run = workspace
        .command()
        .args([
            "bench",
            "c8k1k",
            "--serve",
            server_id,
            "--set",
            "concurrency=[2]",
            "--dry-run",
        ])
        .output()?;
    assert!(
        dry_run.status.success(),
        "{}",
        String::from_utf8_lossy(&dry_run.stderr)
    );
    let plan: Value = serde_json::from_slice(&dry_run.stdout)?;
    assert_eq!(plan["dry_run"], true);
    assert_eq!(plan["target"]["server_record_id"], server_id);
    assert_eq!(
        plan["bench"]["execution"]["cases"][0]["load_shape"]["concurrency"],
        2
    );

    let bench = workspace
        .command()
        .env("FIXTURE_BENCH_WAIT", "1")
        .args(["bench", "c8k1k", "--serve", server_id])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
    wait_for_path(workspace.bench_marker(), Duration::from_secs(5))?;
    let busy_stop = workspace
        .command()
        .args(["serve", "stop", server_id])
        .output()?;
    assert!(!busy_stop.status.success());
    assert!(String::from_utf8_lossy(&busy_stop.stderr).contains("error[E4002]"));

    let bench = bench.wait_with_output()?;
    assert!(
        bench.status.success(),
        "{}",
        String::from_utf8_lossy(&bench.stderr)
    );
    let bench: Value = serde_json::from_slice(&bench.stdout)?;
    assert_eq!(bench["status"], "succeeded");
    assert_datetime_record_id(
        bench["id"].as_str().ok_or("missing Bench record id")?,
        "bench-c8k1k",
    )?;
    assert_eq!(bench["resolved"]["target"]["server_record_id"], server_id);
    assert_eq!(
        bench["resolved"]["measurement_workspace"]["source_digest"],
        server["resolved"]["workspace"]["source_digest"]
    );

    let stop = workspace
        .command()
        .args(["serve", "stop", server_id])
        .output()?;
    assert!(
        stop.status.success(),
        "{}",
        String::from_utf8_lossy(&stop.stderr)
    );
    Ok(())
}

#[test]
fn manual_bench_source_preparation_failure_leaves_target_server_running()
-> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    let manifest = workspace.root().join(".inferlab/workspace.toml");
    let mut config = fs::read_to_string(&manifest)?;
    config.push_str(
        r#"

[benches.agentx]
agentic_source = { dataset = "semianalysis_agentx_062126_256k", profile = "inferencex" }
concurrency = [1]
timeout_seconds = 60
"#,
    );
    fs::write(manifest, config)?;
    let start = workspace
        .command()
        .args(["serve", "start", "dsv4-qualify"])
        .output()?;
    assert!(
        start.status.success(),
        "{}",
        String::from_utf8_lossy(&start.stderr)
    );
    let server: Value = serde_json::from_slice(&start.stdout)?;
    let server_id = server["id"].as_str().ok_or("server record has no id")?;

    let failed = workspace
        .command()
        .env("FIXTURE_SOURCE_PREPARATION_FAIL", "1")
        .args(["bench", "agentx", "--serve", server_id])
        .output()?;
    assert!(!failed.status.success());
    let bench: Value = serde_json::from_slice(&failed.stdout)?;
    assert_eq!(bench["status"], "failed");
    assert_eq!(bench["data_assets"]["target_server_unchanged"], true);
    assert_eq!(bench["data_assets"]["attempts"][0]["state"], "failed");

    let status = workspace
        .command()
        .args(["serve", "status", server_id])
        .output()?;
    assert!(
        status.status.success(),
        "{}",
        String::from_utf8_lossy(&status.stderr)
    );
    let server_status: Value = serde_json::from_slice(&status.stdout)?;
    assert_eq!(server_status["record"]["status"], "running");
    assert_eq!(server_status["observed_alive"], true);

    let stop = workspace
        .command()
        .args(["serve", "stop", server_id])
        .output()?;
    assert!(
        stop.status.success(),
        "{}",
        String::from_utf8_lossy(&stop.stderr)
    );
    Ok(())
}

fn assert_datetime_record_id(id: &str, expected_suffix: &str) -> Result<(), Box<dyn Error>> {
    let (timestamp, suffix) = id.split_once("Z-").ok_or("record id has no UTC prefix")?;
    assert_eq!(timestamp.len(), 23);
    assert_eq!(
        timestamp
            .chars()
            .enumerate()
            .filter_map(|(index, value)| (!value.is_ascii_digit()).then_some((index, value)))
            .collect::<Vec<_>>(),
        [
            (4, '-'),
            (7, '-'),
            (10, 'T'),
            (13, '-'),
            (16, '-'),
            (19, '.')
        ]
    );
    let (stem, pid) = suffix.rsplit_once('-').ok_or("record id has no pid")?;
    assert_eq!(stem, expected_suffix);
    pid.parse::<u32>()?;
    Ok(())
}

#[test]
fn local_launch_runs_declared_checks_as_preflight() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    workspace.declare_environment_check(0)?;
    let output = workspace.run()?;
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let recipe: Value = serde_json::from_slice(&output.stdout)?;
    assert_eq!(recipe["status"], "succeeded");
    let server_id = recipe["server"]["id"].as_str().ok_or("server id")?;
    let server = workspace.load_record(server_id)?;
    assert_eq!(server["environment_checks"][0]["id"], "fixture-guard");
    assert_eq!(
        server["environment_checks"][0]["realization"],
        "local-workspace"
    );
    assert_eq!(server["environment_checks"][0]["outcome"], "passed");
    assert!(
        server["environment_checks"][0]["output"]
            .as_str()
            .is_some_and(|output| output.contains("fixture preflight ran")),
        "preflight output is captured evidence"
    );
    Ok(())
}

#[test]
fn failing_local_check_fails_before_server_launch() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    workspace.declare_environment_check(3)?;
    let output = workspace.run()?;
    let recipe: Value = serde_json::from_slice(&output.stdout)?;
    assert_eq!(
        recipe["status"],
        "failed",
        "a failed preflight check fails the recipe: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let errors = recipe["errors"].as_array().ok_or("errors")?;
    assert!(
        errors.iter().any(|error| {
            error.as_str().is_some_and(|error| {
                error.contains("fixture-guard") && error.contains("repair: pixi run fixture-repair")
            })
        }),
        "a local-realization failure presents the declared repair hint: {errors:?}"
    );
    let server_id = recipe["server"]["id"].as_str().ok_or("server id")?;
    let server = workspace.load_record(server_id)?;
    assert_eq!(server["status"], "failed");
    assert_eq!(server["failure"]["phase"], "preflight");
    assert_eq!(server["environment_checks"][0]["outcome"], "failed");
    assert_eq!(
        process_evidence(&server, "server")?["handle"],
        Value::Null,
        "no process launches after a failed preflight check"
    );
    Ok(())
}
