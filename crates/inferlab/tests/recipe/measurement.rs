use crate::harness::{TestWorkspace, WORKSPACE};
use serde_json::Value;
use std::error::Error;
use std::fs;
use std::path::Path;

impl TestWorkspace {
    fn configure_static_slo_failure(&self) -> Result<(), Box<dyn Error>> {
        let manifest = self.root().join(".inferlab/workspace.toml");
        let text = fs::read_to_string(&manifest)?.replacen(
            "[benches.c8k1k]\nkind = \"serving\"",
            "[benches.c8k1k]\nkind = \"serving\"\naggregate_slos = [{ metric = \"request_throughput\", at_least = 2.0 }]",
            1,
        );
        fs::write(manifest, text)?;
        Ok(())
    }

    fn configure_legacy_adaptive_target(&self) -> Result<(), Box<dyn Error>> {
        let manifest = self.root().join(".inferlab/workspace.toml");
        let text = fs::read_to_string(&manifest)?.replace(
            "aggregate_slos = [\n    { metric = \"request_throughput\", at_least = 1.0 },\n    { metric = \"p99_ttft_ms\", at_most = 1000.0 },\n]\nrequest_slo = { ttft_ms = 900.0, minimum_good_request_ratio = 0.99 }\nmax_search_steps = 3",
            "target_metric = \"p99_ttft_ms\"\ntarget_threshold = 1000.0\nmax_refinement_steps = 3",
        );
        fs::write(manifest, text)?;
        Ok(())
    }

    fn configure_smoke_only(&self) -> Result<(), Box<dyn Error>> {
        let config = WORKSPACE.replace(
            "evals = [\"smoke\", \"gsm8k\"]\ngate = \"gsm8k\"\nbenches = [\"c8k1k\", \"adaptive-c8k1k\"]",
            "evals = [\"smoke\"]\ngate = \"smoke\"\nbenches = []",
        );
        fs::write(self.root().join(".inferlab/workspace.toml"), config)?;
        Ok(())
    }

    fn configure_gsm8k_timeout(&self, seconds: u64) -> Result<(), Box<dyn Error>> {
        let manifest = self.root().join(".inferlab/workspace.toml");
        let text = fs::read_to_string(&manifest)?;
        let (prefix, gsm8k_and_rest) = text
            .split_once("[evals.gsm8k]\n")
            .ok_or("fixture has no gsm8k Eval section")?;
        let gsm8k_and_rest = gsm8k_and_rest.replacen(
            "timeout_seconds = 900",
            &format!("timeout_seconds = {seconds}"),
            1,
        );
        let text = format!("{prefix}[evals.gsm8k]\n{gsm8k_and_rest}");
        fs::write(manifest, text)?;
        Ok(())
    }

    fn configure_release_dataset_bench(&self) -> Result<(), Box<dyn Error>> {
        let manifest = self.root().join(".inferlab/workspace.toml");
        let text = fs::read_to_string(&manifest)?.replacen(
            "request_source = { kind = \"random\", prompt = { kind = \"server_chat\" }, input_tokens = 8192, output_tokens = 1024 }",
            "request_source = { kind = \"dataset\", dataset = \"sharegpt\", max_input_tokens = 8192 }",
            1,
        );
        fs::write(manifest, text)?;
        Ok(())
    }

    fn configure_primed_prefix_bench(&self) -> Result<(), Box<dyn Error>> {
        let manifest = self.root().join(".inferlab/workspace.toml");
        let text = fs::read_to_string(&manifest)?
            .replacen(
                "request_source = { kind = \"random\", prompt = { kind = \"server_chat\" }, input_tokens = 8192, output_tokens = 1024 }",
                "request_source = { kind = \"random\", prompt = { kind = \"flat\" }, input_tokens = 8, output_tokens = 1024, prefix_sharing = { shared_prefix_ratio = 1.0 } }",
                1,
            )
            .replacen(
                "cache = { start = \"cold\" }",
                "cache = { start = \"primed\" }\nrequest_body = { temperature = 0.0 }",
                1,
            )
            .replace(
                "benches = [\"c8k1k\", \"adaptive-c8k1k\"]",
                "benches = [\"c8k1k\"]",
            );
        fs::write(manifest, text)?;
        Ok(())
    }

    fn configure_uncontrolled_warmup_bench(&self) -> Result<(), Box<dyn Error>> {
        let manifest = self.root().join(".inferlab/workspace.toml");
        let text = fs::read_to_string(&manifest)?
            .replacen(
                "prompts_per_concurrency = 4",
                "prompts_per_concurrency = 4\nwarmup_prompts_per_concurrency = 2",
                1,
            )
            .replacen(
                "cache = { start = \"cold\" }",
                "cache = { start = \"uncontrolled\" }",
                1,
            )
            .replace(
                "benches = [\"c8k1k\", \"adaptive-c8k1k\"]",
                "benches = [\"c8k1k\"]",
            );
        fs::write(manifest, text)?;
        Ok(())
    }

    fn configure_workspace_eval(&self) -> Result<(), Box<dyn Error>> {
        let evals = self.root().join("evals");
        fs::create_dir_all(&evals)?;
        fs::write(evals.join("data.jsonl"), "{\"question\":\"cold\"}\n")?;
        fs::write(
            evals.join("custom.yaml"),
            "task: custom_eval\n\
             dataset_path: json\n\
             dataset_kwargs:\n\
               data_files: evals/data.jsonl\n\
             test_split: test\n\
             output_type: generate_until\n",
        )?;
        let manifest = self.root().join(".inferlab/workspace.toml");
        let text = fs::read_to_string(&manifest)?.replace(
            "task = \"gsm8k\"",
            "task = { yaml = \"evals/custom.yaml\" }",
        );
        fs::write(manifest, text)?;
        Ok(())
    }
}

#[test]
fn source_preparation_failure_is_durable_before_server_launch() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    let manifest = workspace.root().join(".inferlab/workspace.toml");
    let config = fs::read_to_string(&manifest)?.replace(
        "evals = [\"smoke\", \"gsm8k\"]",
        "evals = [\"smoke\", \"gsm8k\", \"second\"]",
    ) + "\n[evals.second]\nkind = \"lm-eval\"\ntask = \"arc_easy\"\nlimit = 8\nmetric = \"acc_norm\"\nthreshold = 0.5\ntimeout_seconds = 900\n";
    fs::write(manifest, config)?;
    let output = workspace
        .command()
        .env("FIXTURE_SOURCE_PREPARATION_FAIL", "1")
        .args(["recipe", "run", "dsv4-qualify"])
        .output()?;

    assert!(!output.status.success());
    let recipe: Value = serde_json::from_slice(&output.stdout)?;
    assert_eq!(recipe["schema_version"], 3);
    assert_eq!(recipe["source_preparation_completed"], false);
    assert_eq!(recipe["serving_launch_attempted"], false);
    assert_eq!(recipe["server"]["status"], Value::Null);
    assert_eq!(recipe["data_assets"][0]["state"], "failed");
    assert_eq!(
        recipe["data_assets"][0]["reproducibility"]["conclusion"],
        "not_established"
    );
    let attempts = recipe["data_assets"]
        .as_array()
        .ok_or("recipe data assets are not an array")?;
    assert!(attempts.len() > 1);
    assert!(attempts[1..].iter().all(|attempt| {
        attempt["state"] == "interrupted"
            && attempt["reproducibility"]["conclusion"] == "not_established"
            && attempt["error"]
                .as_str()
                .is_some_and(|error| error.contains("stopped after attempt"))
    }));
    let server_record_id = recipe["server"]["id"]
        .as_str()
        .ok_or("recipe omitted its planned server record id")?;
    assert!(
        !workspace
            .root()
            .join(".inferlab/records")
            .join(server_record_id)
            .exists()
    );
    Ok(())
}

#[test]
fn workspace_eval_uses_the_prepared_local_source_binding() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    workspace.configure_workspace_eval()?;
    let output = workspace
        .command()
        .env("FIXTURE_LOCAL_SNAPSHOT", "1")
        .args(["recipe", "run", "dsv4-qualify"])
        .output()?;

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let recipe: Value = serde_json::from_slice(&output.stdout)?;
    let attempt = recipe["data_assets"]
        .as_array()
        .and_then(|attempts| {
            attempts
                .iter()
                .find(|attempt| attempt["consumers"][0]["definition_id"] == "gsm8k")
        })
        .ok_or("recipe has no workspace Eval source attempt")?;
    assert_eq!(attempt["state"], "ready");
    assert_eq!(attempt["phases"][0]["phase"], "resolve");
    assert_eq!(attempt["phases"][1]["phase"], "snapshot_local");
    assert_eq!(attempt["readiness"]["kind"], "closed");
    let binding = &attempt["readiness"]["eval_binding"];

    let eval_id = recipe["evals"][1]["id"]
        .as_str()
        .ok_or("workspace Eval has no child record")?;
    let eval = workspace.load_record(eval_id)?;
    let request_path = eval["cases"][0]["request"]
        .as_str()
        .ok_or("workspace Eval case has no request evidence")?;
    let request: Value = serde_json::from_slice(&fs::read(workspace.root().join(request_path))?)?;
    assert_eq!(request["prepared_source"], *binding);
    assert!(
        Path::new(
            binding["task_path"]
                .as_str()
                .ok_or("prepared binding has no task path")?
        )
        .is_file()
    );
    Ok(())
}

#[test]
#[ignore = "manual E2E requires Hugging Face network access"]
fn release_dataset_preparation_is_cold_then_a_verified_cache_hit() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    workspace.configure_release_dataset_bench()?;
    let cache = tempfile::tempdir()?;
    let run = || {
        Ok::<_, Box<dyn Error>>(
            workspace
                .command()
                .env("XDG_CACHE_HOME", cache.path())
                .args(["recipe", "run", "dsv4-qualify"])
                .output()?,
        )
    };

    let cold = run()?;
    assert!(
        cold.status.success(),
        "{}",
        String::from_utf8_lossy(&cold.stderr)
    );
    let cold: Value = serde_json::from_slice(&cold.stdout)?;
    let warm = run()?;
    assert!(
        warm.status.success(),
        "{}",
        String::from_utf8_lossy(&warm.stderr)
    );
    let warm: Value = serde_json::from_slice(&warm.stdout)?;
    let release_attempt = |record: &Value| {
        record["data_assets"]
            .as_array()
            .and_then(|attempts| {
                attempts
                    .iter()
                    .find(|attempt| attempt["source"]["kind"] == "release_catalog")
            })
            .cloned()
    };
    let cold_attempt = release_attempt(&cold).ok_or("cold record has no release attempt")?;
    let warm_attempt = release_attempt(&warm).ok_or("warm record has no release attempt")?;

    assert_eq!(cold["source_preparation_completed"], true);
    assert_eq!(cold["serving_launch_attempted"], true);
    assert_eq!(warm["source_preparation_completed"], true);
    assert_eq!(warm["serving_launch_attempted"], true);
    assert_eq!(cold_attempt["source"], warm_attempt["source"]);
    assert_eq!(
        cold_attempt["source_key_sha256"],
        warm_attempt["source_key_sha256"]
    );
    assert_eq!(cold_attempt["state"], "ready");
    assert_eq!(warm_attempt["state"], "ready");
    assert_eq!(
        cold_attempt["phases"][0]["cache_stores"][0]["outcome"],
        "miss"
    );
    assert_eq!(
        warm_attempt["phases"][0]["cache_stores"][0]["outcome"],
        "partial_reuse"
    );
    assert_eq!(cold_attempt["phases"][1]["source_bytes"], "downloaded");
    assert_eq!(
        cold_attempt["phases"][1]["cache_stores"][0]["outcome"],
        "miss"
    );
    assert_eq!(warm_attempt["phases"][1]["source_bytes"], "reused");
    assert_eq!(
        warm_attempt["phases"][1]["cache_stores"][0]["outcome"],
        "full_hit"
    );
    assert_eq!(warm_attempt["readiness"]["kind"], "closed");
    assert_eq!(
        warm_attempt["readiness"]["verification"][0]["matched"],
        true
    );
    let bench_id = warm["benches"][0]
        .get("id")
        .and_then(Value::as_str)
        .ok_or("warm recipe omitted its dataset Bench child")?;
    let bench = workspace.load_record(bench_id)?;
    let materialization = &bench["data_asset_materialization"];
    assert_eq!(materialization["authority"], "inferlab_bench_child");
    assert_eq!(
        materialization["preparation_attempt_id"],
        warm_attempt["attempt_id"]
    );
    assert!(bench["cases"].as_array().is_some_and(|cases| {
        cases.iter().all(|case| {
            case["data_asset_materialization_identity"]
                == materialization["materialization_identity"]
                && case.get("data_asset_materialization").is_none()
        })
    }));
    Ok(())
}

#[test]
fn static_slo_failure_keeps_measurement_status_and_runs_every_case() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    workspace.configure_static_slo_failure()?;

    let output = workspace.run()?;
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let recipe: Value = serde_json::from_slice(&output.stdout)?;
    let matrix_id = recipe["benches"][0]["id"]
        .as_str()
        .ok_or("missing matrix Bench record")?;
    let matrix = workspace.load_record(matrix_id)?;

    assert_eq!(matrix["status"], "succeeded");
    assert_eq!(matrix["passed"], false);
    let cases = matrix["cases"].as_array().ok_or("missing matrix cases")?;
    assert_eq!(cases.len(), 4);
    assert!(cases.iter().all(|case| {
        case["status"] == "succeeded"
            && case["slo"]["passed"] == false
            && case["slo"]["aggregate_slos"][0]["outcome"] == "failed"
    }));
    Ok(())
}

#[test]
fn legacy_adaptive_target_fields_are_rejected_before_execution() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    workspace.configure_legacy_adaptive_target()?;

    let output = workspace
        .command()
        .args(["recipe", "run", "dsv4-qualify", "--dry-run"])
        .output()?;

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("unknown field `max_refinement_steps`"),
        "{stderr}"
    );
    assert!(!workspace.bench_marker().exists());
    Ok(())
}

#[test]
fn smoke_only_recipe_needs_no_measurement_toolchain() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    workspace.configure_smoke_only()?;
    let missing_data_home = workspace.root().join("missing-data");

    let dry_run = workspace
        .command()
        .env("XDG_DATA_HOME", &missing_data_home)
        .args(["recipe", "run", "dsv4-qualify", "--dry-run"])
        .output()?;
    assert!(
        dry_run.status.success(),
        "{}",
        String::from_utf8_lossy(&dry_run.stderr)
    );
    let plan: Value = serde_json::from_slice(&dry_run.stdout)?;
    assert_eq!(
        plan["measurements"]["evals"][0]["execution"]["kind"],
        "native_openai_smoke"
    );
    assert!(
        plan["measurements"]["evals"][0]["execution"]
            .get("toolchain")
            .is_none()
    );

    let output = workspace
        .command()
        .env("XDG_DATA_HOME", &missing_data_home)
        .args(["recipe", "run", "dsv4-qualify"])
        .output()?;
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let recipe: Value = serde_json::from_slice(&output.stdout)?;
    let eval_id = recipe["evals"][0]["id"]
        .as_str()
        .ok_or("smoke Eval has no record id")?;
    let eval = workspace.load_record(eval_id)?;
    assert_eq!(eval["schema_version"], 15);
    assert_eq!(eval["kind"], "eval");
    assert_eq!(eval["resolved"]["execution"]["kind"], "native_openai_smoke");
    assert_eq!(eval["cases"][0]["process"], Value::Null);
    assert_eq!(eval["cases"][0]["stdout"], Value::Null);
    assert_eq!(eval["cases"][0]["stderr"], Value::Null);
    assert_eq!(eval["cases"][0]["error"], Value::Null);
    assert_eq!(eval["cases"][0]["metrics"]["completed"], 1.0);
    assert_eq!(eval["cases"][0]["metrics"]["http_status"], 200.0);
    assert!(
        eval["cases"][0]["metrics"]["elapsed_ms"]
            .as_f64()
            .is_some_and(|elapsed| elapsed >= 0.0)
    );
    assert_eq!(eval["cases"][0]["metrics"]["choices_count"], 1.0);
    assert!(eval.get("request_source").is_none());
    assert!(eval.get("summary").is_none());
    assert!(eval["cases"][0].get("completed_requests").is_none());
    assert!(eval["cases"][0].get("normalization_schema").is_none());
    assert_eq!(
        eval["cases"][0]["raw_artifacts"][0]["kind"],
        "openai-response"
    );
    let request_path = eval["cases"][0]["request"]
        .as_str()
        .ok_or("smoke case has no request path")?;
    let request: Value = serde_json::from_slice(&fs::read(workspace.root().join(request_path))?)?;
    assert_eq!(request["method"], "POST");
    assert_eq!(request["body"]["model"], "deepseek-v4-flash");
    assert_eq!(request["body"]["prompt"], "San Francisco is a city in");
    assert_eq!(request["body"]["max_tokens"], 16);
    assert_eq!(request["body"]["temperature"], 0.0);
    assert_eq!(request["body"]["stream"], false);
    assert_eq!(request["body"]["n"], 1);
    let response_path = eval["cases"][0]["raw_artifacts"][0]["path"]
        .as_str()
        .ok_or("smoke case has no raw response path")?;
    let response = fs::read(response_path)?;
    assert_eq!(
        eval["cases"][0]["metrics"]["response_bytes"],
        response.len() as f64
    );
    let response: Value = serde_json::from_slice(&response)?;
    assert_eq!(response["choices"][0]["text"], " San Francisco");
    assert!(!workspace.eval_marker().exists());
    Ok(())
}

#[test]
fn smoke_rejects_an_endpoint_redirect() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    workspace.configure_smoke_only()?;
    let output = workspace
        .command()
        .env("XDG_DATA_HOME", workspace.root().join("missing-data"))
        .env("FIXTURE_SMOKE_REDIRECT", "1")
        .args(["recipe", "run", "dsv4-qualify"])
        .output()?;

    assert!(!output.status.success());
    let recipe: Value = serde_json::from_slice(&output.stdout)?;
    let eval_id = recipe["evals"][0]["id"]
        .as_str()
        .ok_or("smoke Eval has no record id")?;
    let eval = workspace.load_record(eval_id)?;
    assert_eq!(eval["cases"][0]["metrics"]["http_status"], 302.0);
    let error = eval["cases"][0]["error"]
        .as_str()
        .ok_or("redirected smoke has no error")?;
    assert!(
        error.contains("returned HTTP 302"),
        "unexpected smoke error: {error}"
    );
    assert_eq!(recipe["server"]["status"], "stopped");
    assert_eq!(recipe["cleanup"]["verified"], true);
    Ok(())
}

#[test]
fn failed_eval_gate_skips_benches_and_still_stops_the_server() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    let output = workspace
        .command()
        .env("FIXTURE_GATE_SCORE", "0.5")
        .args(["recipe", "run", "dsv4-qualify"])
        .output()?;

    assert!(!output.status.success());
    let record: Value = serde_json::from_slice(&output.stdout)?;
    assert_eq!(record["status"], "failed");
    assert!(
        record["benches"]
            .as_array()
            .is_some_and(|children| children.iter().all(|child| child["status"] == "skipped"))
    );
    let bench_id = record["benches"][0]["id"]
        .as_str()
        .ok_or("missing bench record id")?;
    let bench = workspace.load_record(bench_id)?;
    assert_eq!(bench["skip_reason"], "eval gate did not succeed");
    assert_eq!(record["cleanup"]["verified"], true);
    assert!(!workspace.bench_marker().exists());
    Ok(())
}

#[test]
fn unsupported_eval_result_envelope_version_fails_the_case() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    let output = workspace
        .command()
        .env("FIXTURE_EVAL_SCHEMA_VERSION", "99")
        .args(["recipe", "run", "dsv4-qualify"])
        .output()?;

    assert!(!output.status.success());
    let record: Value = serde_json::from_slice(&output.stdout)?;
    assert_eq!(record["status"], "failed");
    let eval_id = record["evals"][1]["id"]
        .as_str()
        .ok_or("gsm8k Eval has no record id")?;
    let eval = workspace.load_record(eval_id)?;
    assert_eq!(eval["cases"][0]["status"], "failed");
    assert_eq!(
        eval["cases"][0]["error"],
        "Eval client returned unsupported result schema version 99"
    );
    assert_eq!(record["cleanup"]["verified"], true);
    Ok(())
}

#[test]
fn successful_eval_envelope_cannot_override_client_process_failure() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    let output = workspace
        .command()
        .env("FIXTURE_EVAL_EXIT_CODE", "7")
        .args(["recipe", "run", "dsv4-qualify"])
        .output()?;

    assert!(!output.status.success());
    let recipe: Value = serde_json::from_slice(&output.stdout)?;
    let eval_id = recipe["evals"][1]["id"]
        .as_str()
        .ok_or("failed Eval has no record id")?;
    let eval = workspace.load_record(eval_id)?;
    assert_eq!(eval["cases"][0]["status"], "failed");
    assert_eq!(eval["cases"][0]["process"]["exit_code"], 7);
    assert!(
        eval["cases"][0]["error"]
            .as_str()
            .is_some_and(|error| error.contains("client exited with status"))
    );
    assert_eq!(recipe["cleanup"]["verified"], true);
    Ok(())
}

#[test]
fn eval_failure_before_native_start_does_not_claim_materialization() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    let output = workspace
        .command()
        .env("FIXTURE_EVAL_NO_RESULT", "1")
        .args(["recipe", "run", "dsv4-qualify"])
        .output()?;

    assert!(!output.status.success());
    let recipe: Value = serde_json::from_slice(&output.stdout)?;
    let eval_id = recipe["evals"][1]["id"]
        .as_str()
        .ok_or("failed Eval has no record id")?;
    let eval = workspace.load_record(eval_id)?;
    assert_eq!(eval["cases"][0]["status"], "failed");
    assert!(eval["cases"][0].get("data_asset_materialization").is_none());
    assert!(eval["cases"][0].get("native_command").is_none());
    assert_eq!(recipe["cleanup"]["verified"], true);
    Ok(())
}

#[test]
fn eval_client_deadline_rejects_a_late_result_and_cleans_up_after_timeout()
-> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    workspace.configure_gsm8k_timeout(1)?;
    let output = workspace
        .command()
        .env("FIXTURE_EVAL_WAIT", "1")
        .env("FIXTURE_EVAL_NATIVE_CHECKPOINT", "1")
        .args(["recipe", "run", "dsv4-qualify"])
        .output()?;

    assert!(!output.status.success());
    let recipe: Value = serde_json::from_slice(&output.stdout)?;
    let eval_id = recipe["evals"][1]["id"]
        .as_str()
        .ok_or("timed-out Eval has no record id")?;
    let eval = workspace.load_record(eval_id)?;
    assert_eq!(eval["cases"][0]["status"], "failed");
    assert_eq!(eval["cases"][0]["process"]["timed_out"], true);
    assert_eq!(eval["cases"][0]["process"]["interrupted"], false);
    assert_eq!(eval["cases"][0]["process"]["termination"]["verified"], true);
    assert_eq!(eval["cases"][0]["timing"]["budget"]["configured_ms"], 1_000);
    assert_eq!(eval["cases"][0]["timing"]["terminal_cause"], "timed_out");
    assert_eq!(eval["cases"][0]["native_command"][0], "fixture-eval");
    assert_eq!(eval["cases"][0]["native_timed_out"], Value::Null);
    assert_eq!(eval["cases"][0]["native_interrupted"], Value::Null);
    assert!(
        eval["cases"][0]["timing"]["elapsed_ms"]
            .as_u64()
            .is_some_and(|elapsed| elapsed <= 1_000)
    );
    assert_eq!(
        eval["cases"][0]["process"]["termination"]["status_deadline_ms"],
        0
    );
    assert_eq!(
        eval["cases"][0]["process"]["termination"]["term_grace_ms"],
        2_000
    );
    assert!(
        eval["cases"][0]["error"]
            .as_str()
            .is_some_and(|error| error.contains("measurement-case budget"))
    );
    assert_eq!(recipe["cleanup"]["verified"], true);
    Ok(())
}

#[test]
fn failed_bench_is_recorded_before_server_cleanup() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    let output = workspace
        .command()
        .env("FIXTURE_BENCH_FAIL", "1")
        .args(["recipe", "run", "dsv4-qualify"])
        .output()?;

    assert!(!output.status.success());
    let record: Value = serde_json::from_slice(&output.stdout)?;
    assert_eq!(record["status"], "failed");
    assert!(
        record["benches"]
            .as_array()
            .is_some_and(|children| children.iter().any(|child| child["status"] == "failed"))
    );
    assert_eq!(record["server"]["status"], "stopped");
    assert_eq!(record["cleanup"]["verified"], true);
    assert!(workspace.bench_marker().is_file());
    Ok(())
}

#[test]
fn synthetic_population_requires_prompt_targeting_evidence() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    let output = workspace
        .command()
        .env("FIXTURE_OMIT_PROMPT_TARGETING", "1")
        .args(["recipe", "run", "dsv4-qualify"])
        .output()?;

    assert!(!output.status.success());
    let recipe: Value = serde_json::from_slice(&output.stdout)?;
    let bench_id = recipe["benches"][0]["id"]
        .as_str()
        .ok_or("matrix bench has no record id")?;
    let bench = workspace.load_record(bench_id)?;
    assert_eq!(bench["status"], "failed");
    assert!(
        bench["error"]
            .as_str()
            .is_some_and(|error| error.contains("omitted prompt-targeting evidence"))
    );
    assert_eq!(recipe["cleanup"]["verified"], true);
    Ok(())
}

#[test]
fn partial_prefix_cache_reset_fails_the_bench_with_http_evidence() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    let output = workspace
        .command()
        .env("FIXTURE_RESET_STATUS", "206")
        .args(["recipe", "run", "dsv4-qualify"])
        .output()?;

    assert!(!output.status.success());
    let recipe: Value = serde_json::from_slice(&output.stdout)?;
    assert_eq!(recipe["status"], "failed");
    let bench_id = recipe["benches"][0]["id"]
        .as_str()
        .ok_or("matrix bench has no record id")?;
    let bench = workspace.load_record(bench_id)?;
    assert_eq!(bench["cases"][0]["status"], "failed");
    assert_eq!(
        bench["cases"][0]["cache_preparation"]["reset"]["succeeded"],
        false
    );
    assert_eq!(
        bench["cases"][0]["cache_preparation"]["reset"]["http_status"],
        206
    );
    assert_eq!(bench["cases"][0]["error"], "prefix-cache reset failed");
    assert!(bench["cases"][0].get("metrics").is_none());
    assert!(bench["cases"][0].get("completed_requests").is_none());
    assert!(bench["cases"][0].get("failed_requests").is_none());
    assert!(bench["cases"][0].get("normalization_schema").is_none());
    assert!(bench["cases"][0].get("native_command").is_none());
    assert!(bench["cases"][0].get("native_exit_code").is_none());
    assert!(bench["cases"][0].get("raw_artifacts").is_none());
    assert_eq!(recipe["cleanup"]["verified"], true);
    Ok(())
}

#[test]
fn uncontrolled_warmup_failure_never_releases_profiling() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    workspace.configure_uncontrolled_warmup_bench()?;
    let output = workspace
        .command()
        .env("FIXTURE_BENCH_FAIL_BEFORE_PROFILE", "1")
        .args(["recipe", "run", "dsv4-qualify"])
        .output()?;

    assert!(!output.status.success());
    let recipe: Value = serde_json::from_slice(&output.stdout)?;
    let bench = workspace.load_record(
        recipe["benches"][0]["id"]
            .as_str()
            .ok_or("matrix bench has no record id")?,
    )?;
    assert_eq!(bench["cases"][0]["status"], "failed");
    assert!(
        bench["cases"][0]["error"]
            .as_str()
            .is_some_and(|error| error.contains("before AIPerf reported profiling readiness"))
    );
    assert_eq!(recipe["cleanup"]["verified"], true);
    Ok(())
}

#[test]
fn invalid_profile_barrier_handshake_is_retained_as_a_failed_case() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    workspace.configure_uncontrolled_warmup_bench()?;
    let output = workspace
        .command()
        .env("FIXTURE_BENCH_INVALID_BARRIER", "1")
        .args(["recipe", "run", "dsv4-qualify"])
        .output()?;

    assert!(!output.status.success());
    let recipe: Value = serde_json::from_slice(&output.stdout)?;
    let bench = workspace.load_record(
        recipe["benches"][0]["id"]
            .as_str()
            .ok_or("matrix bench has no record id")?,
    )?;
    let cases = bench["cases"].as_array().ok_or("matrix has no cases")?;
    assert_eq!(cases.len(), 4);
    let warmup_cases = cases
        .iter()
        .filter(|case| case["population_slice"]["warmup_count"] != 0)
        .collect::<Vec<_>>();
    assert_eq!(warmup_cases.len(), 2);
    assert!(warmup_cases.iter().all(|case| case["status"] == "failed"));
    assert!(warmup_cases.iter().all(|case| {
        case["error"]
            .as_str()
            .is_some_and(|error| error.contains("invalid readiness message"))
    }));
    assert_eq!(recipe["cleanup"]["verified"], true);
    Ok(())
}

#[test]
fn primed_dry_run_projects_order_and_conditioning_values() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    workspace.configure_primed_prefix_bench()?;
    let output = workspace
        .command()
        .args(["recipe", "run", "dsv4-qualify", "--dry-run"])
        .output()?;

    assert!(output.status.success());
    let recipe: Value = serde_json::from_slice(&output.stdout)?;
    let bench = &recipe["measurements"]["benches"][0];
    assert_eq!(
        bench["execution"]["cases"][0]["preparation_order"],
        serde_json::json!(["cache_reset", "cache_conditioning", "profiling_release"])
    );
    assert_eq!(
        bench["client"]["prefix_cache_conditioning"],
        serde_json::json!({
            "route": "/v1/completions",
            "model": "deepseek-v4-flash",
            "prompt": {
                "declared": {"kind": "flat"},
                "kind": "flat",
                "request_representation": "flat_prompt",
                "route": "completions",
                "rendering_authority": "local_flat"
            },
            "request_body": {"temperature": 0.0},
            "maximum_shared_prefix_tokens": 8,
            "output_tokens": 1,
            "consumes_population_entry": false
        })
    );
    Ok(())
}

#[test]
fn primed_prefix_preparation_precedes_profiling_and_records_exact_request()
-> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    workspace.configure_primed_prefix_bench()?;
    let output = workspace
        .command()
        .env("FIXTURE_RECORD_CACHE_PREPARATION", "1")
        .env("FIXTURE_RECORD_CLIENT_START", "1")
        .args(["recipe", "run", "dsv4-qualify"])
        .output()?;

    let recipe: Value = serde_json::from_slice(&output.stdout)?;
    let bench_id = recipe["benches"][0]["id"]
        .as_str()
        .ok_or("matrix bench has no record id")?;
    let bench = workspace.load_record(bench_id)?;
    assert!(
        output.status.success(),
        "bench error: {}; case errors: {}; stderr: {}",
        bench["error"],
        bench["cases"],
        String::from_utf8_lossy(&output.stderr)
    );
    let case = &bench["cases"][0];
    assert_eq!(case["cache_preparation"]["start"], "primed");
    assert_eq!(
        case["cache_preparation"]["conditioning"]["prompt_tokens"],
        8
    );
    assert_eq!(
        case["cache_preparation"]["conditioning"]["backend_prompt_tokens"],
        8
    );
    assert_eq!(
        case["cache_preparation"]["conditioning"]["backend_cache_read_tokens"],
        0
    );
    assert_eq!(
        case["cache_preparation"]["conditioning"]["maximum_shared_prefix_tokens"],
        8
    );
    assert_eq!(
        case["cache_preparation"]["conditioning"]["prompt"]["kind"],
        "flat"
    );
    assert_eq!(
        case["cache_preparation"]["conditioning"]["request_body"],
        serde_json::json!({"temperature": 0.0})
    );
    assert_eq!(
        case["cache_preparation"]["conditioning"]["consumes_population_entry"],
        false
    );
    assert_eq!(case["metrics"]["prompt_cache_read_ratio"], 1.0);
    assert_eq!(
        case["prompt_cache_observations"][0]["cache_read_ratio"],
        1.0
    );

    let request: Value = serde_json::from_slice(&fs::read(workspace.conditioning_request())?)?;
    assert_eq!(request["model"], "deepseek-v4-flash");
    assert_eq!(request["prompt"], "canonical prefix");
    assert_eq!(request["max_tokens"], 1);
    assert_eq!(request["n"], 1);
    assert_eq!(request["stream"], false);
    assert_eq!(request["temperature"], 0.0);

    let events = fs::read_to_string(workspace.capture_events())?;
    let events = events.lines().collect::<Vec<_>>();
    assert_eq!(
        &events[..3],
        ["cache_reset", "cache_conditioning", "client_started",]
    );
    assert_eq!(recipe["cleanup"]["verified"], true);
    Ok(())
}

#[test]
fn unsupported_bench_result_envelope_version_fails_the_case() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    let output = workspace
        .command()
        .env("FIXTURE_BENCH_SCHEMA_VERSION", "99")
        .args(["recipe", "run", "dsv4-qualify"])
        .output()?;

    assert!(!output.status.success());
    let record: Value = serde_json::from_slice(&output.stdout)?;
    assert_eq!(record["status"], "failed");
    let bench_id = record["benches"][0]["id"]
        .as_str()
        .ok_or("matrix bench has no record id")?;
    let bench = workspace.load_record(bench_id)?;
    assert_eq!(bench["cases"][0]["status"], "failed");
    assert_eq!(
        bench["cases"][0]["error"],
        "Bench client returned unsupported result schema version 99"
    );
    assert_eq!(record["cleanup"]["verified"], true);
    Ok(())
}

// A genuinely evolved envelope — new version, unknown fields, none of the v1
// fields — must fail with the version-naming rejection, not die as a strict
// v1 parse error: the version gates before the DTO parse.
#[test]
fn evolved_eval_result_envelope_is_rejected_by_version() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    let output = workspace
        .command()
        .env("FIXTURE_EVAL_ENVELOPE_EVOLVED", "1")
        .args(["recipe", "run", "dsv4-qualify"])
        .output()?;

    assert!(!output.status.success());
    let record: Value = serde_json::from_slice(&output.stdout)?;
    assert_eq!(record["status"], "failed");
    let eval_id = record["evals"][1]["id"]
        .as_str()
        .ok_or("gsm8k Eval has no record id")?;
    let eval = workspace.load_record(eval_id)?;
    assert_eq!(eval["cases"][0]["status"], "failed");
    assert_eq!(
        eval["cases"][0]["error"],
        "Eval client returned unsupported result schema version 2"
    );
    assert_eq!(record["cleanup"]["verified"], true);
    Ok(())
}

#[test]
fn evolved_bench_result_envelope_is_rejected_by_version() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    let output = workspace
        .command()
        .env("FIXTURE_BENCH_ENVELOPE_EVOLVED", "1")
        .args(["recipe", "run", "dsv4-qualify"])
        .output()?;

    assert!(!output.status.success());
    let record: Value = serde_json::from_slice(&output.stdout)?;
    assert_eq!(record["status"], "failed");
    let bench_id = record["benches"][0]["id"]
        .as_str()
        .ok_or("matrix bench has no record id")?;
    let bench = workspace.load_record(bench_id)?;
    assert_eq!(bench["cases"][0]["status"], "failed");
    assert_eq!(
        bench["cases"][0]["error"],
        "Bench client returned unsupported result schema version 2"
    );
    assert_eq!(record["cleanup"]["verified"], true);
    Ok(())
}

#[test]
fn server_start_failure_skips_every_selected_measurement() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    let output = workspace
        .command()
        .env("FIXTURE_SERVER_START_FAIL", "1")
        .args(["recipe", "run", "dsv4-qualify"])
        .output()?;

    assert!(!output.status.success());
    let record: Value = serde_json::from_slice(&output.stdout)?;
    assert_eq!(record["status"], "failed");
    assert_eq!(record["server"]["status"], "failed");
    assert_eq!(record["evals"].as_array().map(Vec::len), Some(2));
    assert_eq!(record["benches"].as_array().map(Vec::len), Some(2));
    for child in record["evals"]
        .as_array()
        .into_iter()
        .flatten()
        .chain(record["benches"].as_array().into_iter().flatten())
    {
        assert_eq!(child["status"], "skipped");
        let child_record = workspace.load_record(
            child["id"]
                .as_str()
                .ok_or("measurement reference has no record id")?,
        )?;
        assert_eq!(child_record["skip_reason"], "server did not start");
    }
    assert_eq!(record["cleanup"]["verified"], true);
    assert!(!workspace.eval_marker().exists());
    assert!(!workspace.bench_marker().exists());
    Ok(())
}
