mod dry_run_support;
mod support;

use std::error::Error;
use std::fs;

use dry_run_support::*;

const AGENTIC_MEASUREMENT: &str = r#"
[benches.agentx]
agentic_source = { dataset = "semianalysis_agentx_062126_256k", profile = "inferencex" }
concurrency = [2]
timeout_seconds = 7200

[workload_suites.qualify]
benches = ["agentx"]
"#;

#[test]
fn agentic_source_dry_run_preserves_declared_boundary_and_effective_profile()
-> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    workspace.split_workspace(
        SPLIT_ROOT,
        &[
            ("serving.toml", SPLIT_SERVING),
            ("measurements.toml", AGENTIC_MEASUREMENT),
        ],
    )?;

    let canonical = workspace.run_json(&["workspace", "show", "--json"])?;
    assert_eq!(
        canonical["benches"]["agentx"]["agentic_source"],
        serde_json::json!({
            "dataset": "semianalysis_agentx_062126_256k",
            "profile": "inferencex"
        })
    );
    assert_eq!(
        canonical["benches"]["agentx"]["duration_seconds"],
        serde_json::Value::Null
    );

    let plan = workspace.run_json(&["recipe", "run", "dsv4-qualify", "--dry-run"])?;
    let bench = &plan["measurements"]["benches"][0];
    assert_eq!(bench["execution"]["cases"][0]["duration_seconds"], 1800);
    assert_eq!(
        bench["client"]["effective_definition"]["agentic_source"]["catalog"]["revision"],
        "8fecd2fc56694469f758f0afbbb6335ad3043740"
    );
    assert_eq!(
        bench["client"]["effective_definition"]["agentic_source"]["catalog"]["concurrency_semantics"],
        "root_session_tree_lanes"
    );
    assert_eq!(
        bench["client"]["effective_definition"]["agentic_source"]["catalog"]["dataset_configuration_timeout_seconds"],
        1800
    );
    assert_eq!(
        bench["client"]["effective_definition"]["prompt"]["kind"],
        "server_chat"
    );
    let asset = &plan["measurements"]["data_assets"][0];
    assert_eq!(asset["consumers"][0]["definition_id"], "agentx");
    assert_eq!(asset["source"]["kind"], "agentic");
    assert_eq!(
        asset["source"]["catalog"]["revision"],
        "8fecd2fc56694469f758f0afbbb6335ad3043740"
    );
    assert_eq!(asset["dry_run"]["state"], "local_observation");
    assert_eq!(
        asset["dry_run"]["effective_selection"]["observed_revision"],
        "8fecd2fc56694469f758f0afbbb6335ad3043740"
    );
    assert_eq!(
        asset["dry_run"]["cache_stores"][0]["authority"],
        "huggingface_hub"
    );
    assert_eq!(
        asset["dry_run"]["cache_stores"][0]["outcome"],
        "partial_reuse"
    );
    assert_eq!(
        asset["dry_run"]["observations"],
        serde_json::json!(["owning_runtime_source_observed"])
    );
    assert!(
        asset["dry_run"]["unavailable"]
            .as_array()
            .is_some_and(|facts| facts.iter().any(|fact| fact == "acquired_source"))
    );
    Ok(())
}

#[test]
fn ordinary_measurement_shorthand_is_explicit_in_workspace_and_dry_run_evidence()
-> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    workspace.split_workspace(
        SPLIT_ROOT,
        &[
            ("serving.toml", SPLIT_SERVING),
            ("measurements.toml", ORDINARY_DEFAULTED_MEASUREMENTS),
        ],
    )?;

    let canonical = workspace.run_json(&["workspace", "show", "--json"])?;
    assert_eq!(canonical["evals"]["smoke"]["prompt"], "Hello");
    assert_eq!(canonical["evals"]["smoke"]["max_tokens"], 16);
    assert_eq!(canonical["evals"]["smoke"]["timeout_seconds"], 60);
    assert_eq!(canonical["benches"]["fixed-8k1k"]["kind"], "serving");
    assert_eq!(
        canonical["benches"]["fixed-8k1k"]["request_source"]["prompt"]["kind"],
        "flat"
    );
    assert_eq!(
        canonical["benches"]["range-8k1k"]["request_source"]["input_tokens"],
        serde_json::json!({"kind": "inclusive_uniform", "min": 6553, "max": 8192})
    );
    assert_eq!(
        canonical["benches"]["range-8k1k"]["request_source"]["output_tokens"],
        serde_json::json!({"kind": "inclusive_uniform", "min": 819, "max": 1024})
    );

    let plan = workspace.run_json(&["recipe", "run", "dsv4-qualify", "--dry-run"])?;
    assert_eq!(
        plan["measurements"]["evals"][0]["definition"]["prompt"],
        "Hello"
    );
    assert_eq!(
        plan["measurements"]["benches"][0]["definition"]["kind"],
        "serving"
    );
    assert_eq!(
        plan["measurements"]["benches"][0]["definition"]["request_source"]["prompt"]["kind"],
        "flat"
    );
    assert_eq!(
        plan["measurements"]["benches"][0]["client"]["effective_definition"]["prompt"]["declared"],
        serde_json::Value::Null
    );
    assert_eq!(
        plan["measurements"]["benches"][0]["client"]["effective_definition"]["prompt"]["kind"],
        "flat"
    );
    assert_eq!(
        plan["measurements"]["benches"][1]["definition"]["request_source"]["input_tokens"],
        serde_json::json!({"kind": "inclusive_uniform", "min": 6553, "max": 8192})
    );
    assert_eq!(
        plan["measurements"]["benches"][1]["definition"]["request_source"]["output_tokens"],
        serde_json::json!({"kind": "inclusive_uniform", "min": 819, "max": 1024})
    );
    Ok(())
}

#[test]
fn recipe_measurement_overrides_preserve_declared_effective_and_ordered_values()
-> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    let plan = workspace.run_json(&[
        "recipe",
        "run",
        "dsv4-qualify",
        "--set",
        "evals.gsm8k.limit=100",
        "--set",
        "evals.gsm8k.concurrency=8",
        "--set",
        "evals.gsm8k.trials=5",
        "--set",
        "evals.gsm8k.prompt.kind='server_chat'",
        "--set",
        "evals.gsm8k.request_body.chat_template_kwargs.enable_thinking=true",
        "--set",
        "benches.c8k1k.concurrency=[1, 8]",
        "--set",
        "benches.c8k1k.warmup_prompts_per_concurrency=2",
        "--set",
        "benches.c8k1k.request_body.temperature=1.0",
        "--dry-run",
    ])?;

    assert_eq!(plan["server"]["explicit_overrides"], serde_json::json!([]));
    let gsm8k = &plan["measurements"]["evals"][1];
    assert_eq!(gsm8k["declared_definition"]["limit"], 64);
    assert!(gsm8k["declared_definition"].get("seed").is_none());
    assert_eq!(gsm8k["declared_definition"]["trials"], 1);
    assert!(gsm8k["declared_definition"].get("concurrency").is_none());
    assert_eq!(gsm8k["definition"]["limit"], 100);
    assert_eq!(gsm8k["definition"]["concurrency"], 8);
    assert_eq!(gsm8k["definition"]["trials"], 5);
    // The declared authority is flat, so the chat-only request member is legal
    // only because the override moved the effective authority to server_chat.
    assert_eq!(gsm8k["declared_definition"]["prompt"]["kind"], "flat");
    assert_eq!(gsm8k["definition"]["prompt"]["kind"], "server_chat");
    // The definitions above serialize their effective authority, so only this
    // field shows that the workspace actually declared one.
    assert_eq!(gsm8k["declared_prompt"]["kind"], "flat");
    assert_eq!(
        gsm8k["definition"]["request_body"],
        serde_json::json!({"chat_template_kwargs": {"enable_thinking": true}})
    );
    assert_eq!(
        gsm8k["overrides"],
        serde_json::json!([
            {"invocation_index": 0, "value": "evals.gsm8k.limit=100"},
            {"invocation_index": 1, "value": "evals.gsm8k.concurrency=8"},
            {"invocation_index": 2, "value": "evals.gsm8k.trials=5"},
            {"invocation_index": 3, "value": "evals.gsm8k.prompt.kind='server_chat'"},
            {
                "invocation_index": 4,
                "value": "evals.gsm8k.request_body.chat_template_kwargs.enable_thinking=true"
            },
        ])
    );
    let bench = &plan["measurements"]["benches"][0];
    assert_eq!(
        bench["declared_definition"]["concurrency"],
        serde_json::json!([1, 4])
    );
    assert_eq!(
        bench["definition"]["concurrency"],
        serde_json::json!([1, 8])
    );
    assert_eq!(bench["definition"]["warmup_prompts_per_concurrency"], 2);
    assert_eq!(bench["execution"]["cases"][0]["warmup_request_count"], 2);
    assert_eq!(bench["execution"]["cases"][1]["warmup_request_count"], 16);
    assert_eq!(bench["execution"]["cases"][2]["warmup_request_count"], 0);
    assert_eq!(
        bench["execution"]["cases"][0]["preparation_order"],
        serde_json::json!(["warmup_drain", "cache_reset", "profiling_release"])
    );
    assert_eq!(
        bench["execution"]["cases"][2]["preparation_order"],
        serde_json::json!(["cache_reset", "profiling_release"])
    );
    assert_eq!(
        bench["client"]["effective_definition"]["request_body"],
        serde_json::json!({"temperature": 1.0})
    );
    assert_eq!(
        bench["client"]["effective_definition"]["prompt"],
        serde_json::json!({
            "declared": {"kind": "server_chat"},
            "kind": "server_chat",
            "request_representation": "structured_messages",
            "route": "chat_completions",
            "rendering_authority": "server"
        })
    );
    assert_eq!(bench["client"]["tpot_applicability"], "applicable");
    assert_eq!(
        bench["overrides"],
        serde_json::json!([{
            "invocation_index": 5,
            "value": "benches.c8k1k.concurrency=[1, 8]"
        }, {
            "invocation_index": 6,
            "value": "benches.c8k1k.warmup_prompts_per_concurrency=2"
        }, {
            "invocation_index": 7,
            "value": "benches.c8k1k.request_body.temperature=1.0"
        }])
    );
    Ok(())
}

#[test]
fn concurrency_warmup_count_overflow_fails_definition_resolution() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    let output = workspace.run(&[
        "recipe",
        "run",
        "dsv4-qualify",
        "--set",
        "benches.c8k1k.concurrency=[2147483648]",
        "--set",
        "benches.c8k1k.prompts_per_concurrency=1",
        "--set",
        "benches.c8k1k.warmup_prompts_per_concurrency=2",
        "--dry-run",
    ])?;

    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("warmup request count exceeds u32"));
    Ok(())
}

#[test]
fn nested_measurement_override_rejects_traversing_a_scalar() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    let output = workspace.run(&[
        "recipe",
        "run",
        "dsv4-qualify",
        "--set",
        "evals.gsm8k.request_body.vendor=\"fixed\"",
        "--set",
        "evals.gsm8k.request_body.vendor.mode=\"fast\"",
        "--dry-run",
    ])?;

    assert!(!output.status.success());
    assert!(output.stdout.is_empty());
    let stderr = String::from_utf8(output.stderr)?;
    assert!(stderr.contains("request_body.vendor"), "{stderr}");
    assert!(stderr.contains("traverses non-table value"), "{stderr}");
    Ok(())
}

#[test]
fn bench_override_cannot_switch_the_declared_request_source_kind() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    let output = workspace.run(&[
        "recipe",
        "run",
        "dsv4-qualify",
        "--set",
        "benches.c8k1k.request_source.kind=\"dataset\"",
        "--dry-run",
    ])?;

    assert!(!output.status.success());
    assert!(output.stdout.is_empty());
    let stderr = String::from_utf8(output.stderr)?;
    assert!(
        stderr.contains("request_source.kind cannot be overridden"),
        "{stderr}"
    );
    Ok(())
}

#[test]
fn repeated_eval_rejects_zero_trials_and_a_request_body_seed() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    let zero = workspace.run(&[
        "recipe",
        "run",
        "dsv4-qualify",
        "--set",
        "evals.gsm8k.trials=0",
        "--dry-run",
    ])?;
    assert!(!zero.status.success());
    let stderr = String::from_utf8(zero.stderr)?;
    assert!(stderr.contains("trials must be positive"), "{stderr}");

    let path = workspace.root.path().join(".inferlab/workspace.toml");
    let config = format!(
        "{}\n[evals.gsm8k.request_body]\nseed = 9\n",
        fs::read_to_string(&path)?
    );
    fs::write(path, config)?;
    let seed = workspace.run(&["recipe", "run", "dsv4-qualify", "--dry-run"])?;
    assert!(!seed.status.success());
    let stderr = String::from_utf8(seed.stderr)?;
    assert!(
        stderr.contains(
            "request_body.seed conflicts with a measurement-runtime-owned request member"
        ),
        "{stderr}"
    );
    Ok(())
}

#[test]
fn workspace_lm_eval_yaml_resolves_as_the_effective_task_source() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    let task_dir = workspace.root.path().join("evals");
    fs::create_dir_all(&task_dir)?;
    fs::write(
        task_dir.join("data.jsonl"),
        "{\"prompt\":\"hello\",\"answer\":\"world\"}\n",
    )?;
    fs::write(
        task_dir.join("custom.yaml"),
        "task: custom_eval\n\
         dataset_path: json\n\
         dataset_kwargs:\n\
           data_files: evals/data.jsonl\n\
         test_split: test\n\
         output_type: generate_until\n\
         doc_to_text: '{{prompt}}'\n\
         doc_to_target: '{{answer}}'\n\
         metric_list:\n\
           - metric: exact_match\n\
             higher_is_better: true\n",
    )?;
    let path = workspace.root.path().join(".inferlab/workspace.toml");
    let config = fs::read_to_string(&path)?.replace(
        "task = \"gsm8k\"",
        "task = { yaml = \"evals/custom.yaml\" }",
    );
    fs::write(path, config)?;

    let output = workspace
        .command()
        .env("FIXTURE_LOCAL_SNAPSHOT", "1")
        .args(["recipe", "run", "dsv4-qualify", "--dry-run"])
        .output()?;
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let plan: serde_json::Value = serde_json::from_slice(&output.stdout)?;
    let eval = &plan["measurements"]["evals"][1];
    assert_eq!(
        eval["declared_definition"]["task"],
        serde_json::json!({"yaml": "evals/custom.yaml"})
    );
    assert_eq!(
        eval["definition"]["task"]["yaml"],
        workspace
            .root
            .path()
            .join("evals/custom.yaml")
            .display()
            .to_string()
    );
    let asset = plan["measurements"]["data_assets"]
        .as_array()
        .and_then(|assets| {
            assets
                .iter()
                .find(|asset| asset["consumers"][0]["definition_id"] == "gsm8k")
        })
        .ok_or("dry-run has no workspace Eval source asset")?;
    assert_eq!(asset["dry_run"]["state"], "local_observation");
    assert_eq!(
        asset["dry_run"]["observations"],
        serde_json::json!(["complete_local_closure_enumerated"])
    );
    assert_eq!(
        asset["dry_run"]["planned_external_work"],
        serde_json::json!(["immutable_local_snapshot"])
    );
    assert!(!workspace.root.path().join(".inferlab/records").exists());
    Ok(())
}

#[test]
fn workspace_lm_eval_yml_extension_uses_the_pinned_yaml_loader() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    let task_dir = workspace.root.path().join("evals");
    fs::create_dir_all(&task_dir)?;
    fs::write(
        task_dir.join("custom.yml"),
        "task: custom_eval\noutput_type: generate_until\n",
    )?;
    let path = workspace.root.path().join(".inferlab/workspace.toml");
    let config = fs::read_to_string(&path)?
        .replace("task = \"gsm8k\"", "task = { yaml = \"evals/custom.yml\" }");
    fs::write(path, config)?;

    let plan = workspace.run_json(&["recipe", "run", "dsv4-qualify", "--dry-run"])?;
    assert_eq!(
        plan["measurements"]["evals"][1]["declared_definition"]["task"],
        serde_json::json!({"yaml": "evals/custom.yml"})
    );
    Ok(())
}

#[test]
fn standalone_lm_eval_dataset_override_is_rejected_with_field_context() -> Result<(), Box<dyn Error>>
{
    let workspace = TestWorkspace::new()?;
    let path = workspace.root.path().join(".inferlab/workspace.toml");
    let config = fs::read_to_string(&path)?.replace(
        "task = \"gsm8k\"",
        "task = \"gsm8k\"\ndataset = \"ignored-before-this-fix\"",
    );
    fs::write(path, config)?;

    let output = workspace.run(&["recipe", "run", "dsv4-qualify", "--dry-run"])?;

    assert!(!output.status.success());
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("dataset"),
        "validation names the unsupported second dataset authority"
    );
    Ok(())
}

#[test]
fn recipe_measurement_override_rejects_a_definition_outside_the_selected_suite()
-> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    let output = workspace.run(&[
        "recipe",
        "run",
        "dsv4-qualify",
        "--set",
        "evals.not-selected.limit=1",
        "--dry-run",
    ])?;

    assert!(!output.status.success());
    Ok(())
}

#[test]
fn overrides_outside_the_typed_server_patch_are_rejected() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    let output = workspace.run(&[
        "recipe",
        "run",
        "dsv4-qualify",
        "--set",
        "bench.request_count=1",
        "--dry-run",
    ])?;

    assert!(!output.status.success());

    let reserved = workspace.run(&[
        "serve",
        "start",
        "dsv4-qualify",
        "--set",
        "server.model=\"other\"",
        "--dry-run",
    ])?;
    assert!(!reserved.status.success());
    Ok(())
}

#[test]
fn missing_eval_toolchain_reports_the_explicit_install_action() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    let output = workspace
        .command()
        .env("XDG_DATA_HOME", workspace.root.path().join("missing-data"))
        .args(["recipe", "run", "dsv4-qualify", "--dry-run"])
        .output()?;

    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr)?;
    assert!(stderr.contains("inferlab toolchain install"));
    assert!(!workspace.root.path().join(".inferlab/records").exists());
    Ok(())
}
