mod dry_run_support;
mod support;

use std::collections::BTreeMap;
use std::error::Error;
use std::fs;
use std::path::Path;
use std::time::{Duration, Instant};

use dry_run_support::*;
use support::{LaunchProjection, ReadinessProjection};

#[test]
fn unavailable_pixi_environment_reports_the_locked_install_action() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    let output = workspace
        .command()
        .env("FAKE_PIXI_UNAVAILABLE", "1")
        .args(["serve", "start", "dsv4-qualify", "--dry-run"])
        .output()?;

    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr)?;
    assert!(stderr.contains("Pixi environment \"vllm\" is not usable"));
    assert!(stderr.contains("pixi install --locked --environment vllm"));
    assert!(!workspace.root.path().join(".inferlab/records").exists());
    Ok(())
}

#[test]
fn serve_and_recipe_dry_run_share_the_default_case() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    let serve = workspace.run_json(&["serve", "start", "dsv4-qualify", "--dry-run"])?;
    let recipe = workspace.run_json(&["recipe", "run", "dsv4-qualify", "--dry-run"])?;

    assert_eq!(serve["workflow"], "serve-start");
    assert_eq!(recipe["workflow"], "recipe-run");
    assert!(serve.get("recipe").is_none());
    assert_eq!(recipe["recipe"]["id"], "dsv4-qualify");
    assert_eq!(serve["server"]["case"]["id"], "tp2");
    assert_eq!(serve["server"]["case"]["selection"], "default");
    assert_eq!(serve["server"], recipe["server"]);
    assert_eq!(
        serve["server"]["roles"][0]["effective_parallelism"]["outer"]["pipeline_parallel_size"],
        1
    );
    assert_eq!(
        serve["server"]["roles"][0]["declared_parallelism"]["outer"]["tensor_parallel_size"],
        2
    );
    assert_eq!(
        serve["server"]["roles"][0]["declared_settings"]["max_model_len"],
        65536
    );
    assert_eq!(
        serve["server"]["roles"][0]["declared_settings"]["trust_remote_code"],
        true
    );
    assert_eq!(
        serve["server"]["roles"][0]["effective_settings"]["trust_remote_code"],
        false
    );
    assert!(serve["server"].get("parallelism").is_none());
    assert!(serve["server"].get("settings").is_none());
    assert_eq!(serve["server"]["readiness_attempt_timeout_seconds"], 30);
    assert_eq!(serve["server"]["capture_arm_deadline_seconds"], 60);
    assert_eq!(serve["server"]["capture_control_deadline_seconds"], 60);
    assert_eq!(
        serve["server"]["capture_finalization_deadline_seconds"],
        300
    );
    assert!(
        serve["server"]["declarations"][0]["common"]
            .get("readiness_attempt_timeout_seconds")
            .is_none()
    );
    assert_eq!(
        serve["server"]["declarations"][0]["source"],
        serde_json::json!({"kind": "server", "id": "dsv4-qualify"})
    );
    assert_eq!(
        serve["server"]["declarations"][1]["source"],
        serde_json::json!({"kind": "case", "id": "tp2"})
    );
    assert_eq!(
        serve["server"]["declarations"][0]["common"]["parallelism"]["outer"]["pipeline_parallel_size"],
        1
    );
    assert_eq!(
        serve["server"]["declarations"][1]["common"]["parallelism"]["outer"]["tensor_parallel_size"],
        2
    );
    assert!(
        serve["server"]["declarations"][0]["common"]
            .get("profiling")
            .is_none()
    );
    assert!(
        serve["server"]["declarations"][0]["roles"]["serve"]
            .get("replicas")
            .is_none()
    );
    assert_eq!(
        serve["server"]["declarations"][0]["roles"]["serve"]["settings"]["block_size"],
        16
    );
    assert!(serve["stack"].get("checks").is_none());
    assert_eq!(recipe["measurements"]["gate"], "gsm8k");
    assert_eq!(recipe["measurements"]["evals"][0]["id"], "smoke");
    assert_eq!(recipe["measurements"]["evals"][1]["id"], "gsm8k");
    assert_eq!(
        recipe["measurements"]["evals"][0]["execution"]["kind"],
        "native_openai_smoke"
    );
    // The smoke Eval's declared inputs from the workspace flow into the plan's
    // definition unchanged, so a dropped or mistyped smoke field is caught.
    assert_eq!(
        recipe["measurements"]["evals"][0]["definition"]["prompt"],
        "San Francisco is a city in"
    );
    assert_eq!(
        recipe["measurements"]["evals"][0]["definition"]["max_tokens"],
        16
    );
    assert_eq!(
        recipe["measurements"]["evals"][0]["definition"]["timeout_seconds"],
        60
    );
    assert!(
        recipe["measurements"]["evals"][1]["execution"]["command"]["argv"][0]
            .as_str()
            .is_some_and(|value| value.ends_with("/.pixi/envs/eval/bin/python"))
    );
    assert_eq!(
        recipe["measurements"]["evals"][1]["execution"]["toolchain"]["lm_eval_version"],
        "0.4.12"
    );
    assert_eq!(
        recipe["measurements"]["benches"][0]["execution"]["mode"],
        "matrix"
    );
    assert_eq!(
        recipe["measurements"]["benches"][0]["execution"]["cases"][0]["load_shape"]["kind"],
        "concurrency-limited"
    );
    assert_eq!(
        recipe["measurements"]["benches"][0]["execution"]["cases"][0]["request_count"],
        4
    );
    assert_eq!(
        recipe["measurements"]["benches"][0]["client"]["effective_definition"]["prompt"]["declared"]
            ["kind"],
        "server_chat"
    );
    assert_eq!(
        recipe["measurements"]["benches"][0]["definition"]["warmup_prompts_per_concurrency"],
        0
    );
    assert_eq!(
        recipe["measurements"]["benches"][0]["execution"]["cases"][0]["warmup_request_count"],
        0
    );
    assert_eq!(
        recipe["measurements"]["benches"][0]["execution"]["cases"][1]["request_count"],
        16
    );
    assert_eq!(
        recipe["measurements"]["benches"][0]["execution"]["cases"][2]["load_shape"]["request_rate"],
        1.0
    );
    assert_eq!(
        recipe["measurements"]["benches"][0]["execution"]["cases"][3]["load_shape"]["request_rate"],
        "inf"
    );
    assert_eq!(
        recipe["measurements"]["benches"][1]["execution"]["mode"],
        "adaptive"
    );
    assert_eq!(
        recipe["measurements"]["benches"][1]["execution"]["initial_request_rates"],
        serde_json::json!([1.0, 4.0])
    );
    assert_eq!(
        recipe["measurements"]["benches"][1]["execution"]["policy"],
        "highest-feasible-rate-v1"
    );
    assert_eq!(
        recipe["measurements"]["benches"][1]["execution"]["max_search_steps"],
        3
    );
    assert_eq!(
        recipe["measurements"]["benches"][1]["definition"]["request_slo"]["minimum_good_request_ratio"],
        0.99
    );
    assert!(
        recipe["measurements"]["benches"][0]["client"]["command"]["argv"][1]
            .as_str()
            .is_some_and(|value| {
                value.ends_with("/runner/inferlab_bench_runner/bench_client.py")
            })
    );
    assert_eq!(
        recipe["measurements"]["benches"][0]["client"]["prefix_cache_reset"]["path"],
        "/reset_prefix_cache"
    );
    assert_eq!(serve["workspace"]["dirty"], false);
    assert_eq!(serve["workspace"]["revision_reproducible"], true);
    assert_eq!(
        serve["workspace"]["pixi_manifest_sha256"]
            .as_str()
            .map(str::len),
        Some(64)
    );
    assert_eq!(
        serve["workspace"]["pixi_lock_sha256"]
            .as_str()
            .map(str::len),
        Some(64)
    );
    assert_eq!(serve["stack"]["id"], "vllm");
    assert_eq!(serve["stack"]["pixi_environment"], "vllm");
    assert_eq!(
        serve["server"]["integration"]["adapter_id"],
        "inferlab-vllm"
    );
    assert_eq!(serve["server"]["integration"]["adapter_version"], "0.1.0");
    assert_eq!(
        serve["server"]["integration"]["plan_request_sha256"]
            .as_str()
            .map(str::len),
        Some(64)
    );
    assert_eq!(
        serve["server"]["placement"]["machines"],
        serde_json::json!(["local"])
    );
    let server_rank = resolved_rank(&serve["server"], "server")?;
    let command_prefix: Vec<_> = server_rank
        .command
        .argv
        .iter()
        .take(7)
        .map(String::as_str)
        .collect();
    assert_eq!(
        command_prefix,
        ["pixi", "run", "--as-is", "--executable", "-e", "vllm", "--"]
    );
    assert!(
        server_rank
            .command
            .argv
            .iter()
            .any(|arg| arg == "127.0.0.1")
    );
    assert!(server_rank.command.argv.iter().any(|arg| arg == "8000"));
    assert_eq!(serve["server"]["endpoint"]["host"], "127.0.0.1");
    assert_eq!(serve["server"]["endpoint"]["port"], 8000);
    let ReadinessProjection::Http {
        path,
        timeout_seconds,
    } = &server_rank.readiness
    else {
        return Err("expected HTTP readiness".into());
    };
    assert_eq!(path, "/v1/models");
    assert_eq!(*timeout_seconds, Some(900));
    assert_eq!(server_rank.devices, [0, 1]);
    assert_eq!(server_rank.command.env["CUDA_VISIBLE_DEVICES"], "0,1");
    let cache = &server_rank.runtime_cache;
    let default_cache_root = workspace.root.path().join(".inferlab/cache/runtime");
    assert_eq!(cache.storage_root_source, "workspace-default");
    assert_eq!(cache.storage_root, default_cache_root);
    assert_eq!(
        cache.namespace.workspace_source_digest,
        serve["workspace"]["source_digest"]
            .as_str()
            .ok_or("missing source digest")?
    );
    assert_eq!(cache.namespace.pixi_environment, "vllm");
    assert_eq!(cache.namespace.machine, "local");
    assert_eq!(cache.namespace.process, "server");
    assert!(cache.path.starts_with(&default_cache_root));
    assert!(cache.path.ends_with("local/server"));
    assert_eq!(
        server_rank.command.env["FLASHINFER_WORKSPACE_BASE"],
        cache.path.join("flashinfer").to_string_lossy()
    );
    assert_eq!(
        server_rank.model_locator.as_deref(),
        Some(workspace.private_weight.as_str())
    );
    assert!(serve.to_string().contains(&workspace.private_weight));
    assert!(recipe.to_string().contains(&workspace.private_weight));
    assert!(!workspace.root.path().join(".inferlab/records").exists());
    Ok(())
}

#[test]
fn gateway_single_uses_one_process_only_frontend_without_model_coordinates()
-> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    let manifest_path = workspace.root.path().join(".inferlab/workspace.toml");
    let manifest = fs::read_to_string(&manifest_path)?.replacen(
        "topology = \"single\"\n",
        "topology = \"single\"\nprofiling = true\ngateway_backend = \"fixture-gateway\"\n",
        1,
    );
    fs::write(manifest_path, manifest)?;
    let bindings_path = workspace.root.path().join(".inferlab/local.toml");
    let bindings =
        fs::read_to_string(&bindings_path)?.replacen("ports = [8000]", "ports = [8000, 8001]", 1);
    fs::write(bindings_path, bindings)?;

    let plan = workspace.run_json(&["serve", "start", "dsv4-qualify", "--dry-run"])?;
    let server = &plan["server"];
    let engines = resolved_ranks(server)?;
    let frontend = support::resolved_frontend(server)?;
    let frontend_json = server["frontend"]["processes"][0]
        .as_object()
        .ok_or("routed single did not contain a frontend process")?;

    assert_eq!(engines.len(), 1);
    assert_eq!(engines[0].role_id, "serve");
    assert_eq!(engines[0].rank.endpoint.port, 8000);
    assert_eq!(server["frontend"]["gateway"]["backend"], "fixture-gateway");
    assert_eq!(
        server["frontend"]["gateway"]["targets"][0]["kind"],
        "engine"
    );
    assert_eq!(server["frontend"]["gateway"]["targets"][0]["role"], "serve");
    assert_eq!(server["frontend"]["gateway"]["process_id"], "gateway");
    assert!(server["frontend"]["pd_router"].is_null());
    assert_eq!(server["links"][0]["source"], "gateway");
    assert_eq!(server["links"][0]["targets"], serde_json::json!(["serve"]));

    assert_eq!(frontend.id, "gateway");
    assert_eq!(frontend.process_role, "gateway");
    assert_eq!(frontend.components, ["gateway"]);
    assert_eq!(frontend.dependencies, ["server"]);
    assert!(frontend.devices.is_empty());
    assert_eq!(frontend.endpoint.port, 8001);
    assert_eq!(server["endpoint"]["port"], frontend.endpoint.port);
    assert!(
        frontend
            .command
            .argv
            .iter()
            .any(|arg| arg == "fixture-gateway")
    );
    assert_eq!(
        frontend_json.get("kind"),
        Some(&serde_json::json!("frontend"))
    );
    assert!(!frontend_json.contains_key("model_locator"));
    assert!(!frontend_json.contains_key("replica"));
    assert!(!frontend_json.contains_key("rank"));
    assert!(!frontend_json.contains_key("rank_count"));
    assert_eq!(
        engines[0]
            .rank
            .capture_target
            .as_ref()
            .ok_or("missing capture target")?,
        &support::CaptureTargetProjection {
            window_control_endpoint: "gateway".to_owned(),
            control_process_id: "gateway".to_owned(),
            start: support::HttpActionProjection {
                method: "post".to_owned(),
                path: "/start_profile".to_owned(),
                body: Some(BTreeMap::from([(
                    "activities".to_owned(),
                    serde_json::json!(["CUDA_PROFILER"]),
                )])),
                effective_url: "http://127.0.0.1:8001/start_profile".to_owned(),
            },
            stop: support::HttpActionProjection {
                method: "post".to_owned(),
                path: "/stop_profile".to_owned(),
                body: None,
                effective_url: "http://127.0.0.1:8001/stop_profile".to_owned(),
            },
            escapes: support::NsysEscapesProjection::default(),
        }
    );
    Ok(())
}

#[test]
fn capture_rejects_a_gateway_control_binding_without_a_gateway() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    let manifest_path = workspace.root.path().join(".inferlab/workspace.toml");
    let manifest = fs::read_to_string(&manifest_path)?.replacen(
        "[servers.dsv4-qualify.settings]\n",
        "[servers.dsv4-qualify.settings]\nfixture_capture_gateway = true\n",
        1,
    );
    fs::write(manifest_path, manifest)?;

    let output = workspace.run(&[
        "recipe",
        "run",
        "dsv4-qualify",
        "--capture",
        "c8k1k",
        "--dry-run",
    ])?;
    let stderr = String::from_utf8(output.stderr)?;

    assert!(!output.status.success(), "{stderr}");
    assert!(
        stderr.contains("selected Gateway profiling window control without planning a Gateway"),
        "{stderr}"
    );
    Ok(())
}

#[test]
fn capture_rejects_a_concrete_window_control_url() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    let manifest_path = workspace.root.path().join(".inferlab/workspace.toml");
    let manifest = fs::read_to_string(&manifest_path)?.replacen(
        "[servers.dsv4-qualify.settings]\n",
        "[servers.dsv4-qualify.settings]\nfixture_capture_invalid_path = true\n",
        1,
    );
    fs::write(manifest_path, manifest)?;

    let output = workspace.run(&[
        "recipe",
        "run",
        "dsv4-qualify",
        "--capture",
        "c8k1k",
        "--dry-run",
    ])?;
    let stderr = String::from_utf8(output.stderr)?;

    assert!(!output.status.success(), "{stderr}");
    assert!(
        stderr.contains("capture start path") && stderr.contains("absolute origin path"),
        "{stderr}"
    );
    Ok(())
}

#[test]
fn invocation_cannot_add_a_gateway_absent_from_the_server_base() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    let output = workspace.run(&[
        "serve",
        "start",
        "dsv4-qualify",
        "--set",
        "server.gateway_backend=\"fixture-gateway\"",
        "--dry-run",
    ])?;
    let stderr = String::from_utf8(output.stderr)?;

    assert!(!output.status.success(), "{stderr}");
    assert!(
        stderr.contains(
            "cannot add gateway_backend because server \"dsv4-qualify\" does not declare a Gateway"
        ),
        "{stderr}"
    );
    Ok(())
}

#[test]
fn schema_one_workspace_is_rejected() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    fs::write(
        workspace.root.path().join(".inferlab/workspace.toml"),
        WORKSPACE.replacen("schema_version = 2", "schema_version = 1", 1),
    )?;

    let output = workspace.run(&["serve", "start", "dsv4-qualify", "--dry-run"])?;

    assert!(!output.status.success());
    assert!(output.stdout.is_empty());
    assert!(!workspace.root.path().join(".inferlab/records").exists());
    Ok(())
}

#[test]
fn dry_run_records_launch_files_without_materializing_them() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    let plan = workspace.run_json(&[
        "serve",
        "start",
        "dsv4-qualify",
        "--set",
        "server.settings.fixture_mode=\"launch-file\"",
        "--dry-run",
    ])?;
    let process = resolved_rank(&plan["server"], "server")?;
    let launch_file = &process.launch_files[0];
    let resolved_path = &launch_file.resolved_path;

    assert_eq!(launch_file.text, "fixture: dry-run\nunicode: 雪\n");
    assert!(
        launch_file.relative_path.starts_with("launch-files/")
            && launch_file.relative_path.ends_with("/fixture.yaml")
    );
    assert_eq!(launch_file.sha256.len(), 64);
    assert!(
        process
            .command
            .argv
            .iter()
            .any(|value| Path::new(value) == resolved_path)
    );
    assert!(!resolved_path.exists());
    assert!(!workspace.root.path().join(".inferlab/records").exists());
    Ok(())
}

#[test]
fn recipe_capture_selects_one_workload_and_prepares_the_server() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    let plan = workspace.run_json(&[
        "recipe",
        "run",
        "dsv4-qualify",
        "--capture",
        "c8k1k",
        "--dry-run",
    ])?;

    assert_eq!(plan["measurements"]["evals"][0]["capture"], false);
    assert_eq!(plan["measurements"]["benches"][0]["capture"], true);
    assert_eq!(plan["measurements"]["benches"][1]["capture"], false);
    assert_eq!(plan["server"]["profiling"], true);
    let process = resolved_rank(&plan["server"], "server")?;
    let capture_target = process.capture_target.ok_or("missing capture target")?;
    assert_eq!(capture_target.window_control_endpoint, "replica_entry");
    assert_eq!(capture_target.control_process_id, "server");
    // Capturing this server prepares the adapter-declared profiling control
    // endpoints; pin them so a break in the start/stop wiring is caught.
    assert_eq!(capture_target.start.method, "post");
    assert_eq!(capture_target.start.path, "/start_profile");
    assert_eq!(
        capture_target.start.body,
        Some(BTreeMap::from([(
            "activities".to_owned(),
            serde_json::json!(["CUDA_PROFILER"]),
        )]))
    );
    assert_eq!(
        capture_target.start.effective_url,
        "http://127.0.0.1:8000/start_profile"
    );
    assert_eq!(capture_target.stop.method, "post");
    assert_eq!(capture_target.stop.path, "/stop_profile");
    assert_eq!(
        capture_target.stop.effective_url,
        "http://127.0.0.1:8000/stop_profile"
    );
    Ok(())
}

#[test]
fn ordered_two_node_placement_is_allocated_before_process_rendering() -> Result<(), Box<dyn Error>>
{
    let workspace = TestWorkspace::new()?;
    let node_b_weight = workspace.root.path().join("node-b/dsv4");
    fs::write(
        workspace.root.path().join(".inferlab/local.toml"),
        format!(
            "default_placement = \"pair\"\n\
             \n\
             [model_weights.dsv4]\n\
             locator = {:?}\n\
             \n\
             [model_weights.dsv4.machine_locators]\n\
             node-b = {:?}\n\
             \n\
             [machines.node-a]\n\
             host = \"node-a.example\"\n\
             ports = [8000, 29501]\n\
             devices = [0, 1]\n\
             \n\
             [machines.node-b]\n\
             host = \"node-b.example\"\n\
             ports = [8000]\n\
             devices = [4, 5]\n\
             \n\
             [placements.pair.roles.serve]\n\
             ranks = [\n\
               {{ machine = \"node-a\", devices = [0, 1] }},\n\
               {{ machine = \"node-b\", devices = [4, 5] }},\n\
             ]\n",
            workspace.private_weight,
            node_b_weight.display().to_string(),
        ),
    )?;

    let plan = workspace.run_json(&[
        "serve",
        "start",
        "dsv4-qualify",
        "--case",
        "tp4",
        "--dry-run",
    ])?;

    assert_eq!(
        plan["server"]["placement"]["machines"],
        serde_json::json!(["node-a", "node-b"])
    );
    let first = resolved_rank(&plan["server"], "server-rank-000")?;
    let second = resolved_rank(&plan["server"], "server-rank-001")?;
    assert_eq!(first.id, "server-rank-000");
    assert_eq!(second.id, "server-rank-001");
    assert_eq!(first.devices, [0, 1]);
    assert_eq!(second.devices, [4, 5]);
    assert_eq!(first.ports["master"].port, 29501);
    assert_eq!(second.model_locator.as_deref(), node_b_weight.to_str());
    assert_eq!(plan["server"]["endpoint"]["host"], "node-a.example");
    assert_eq!(plan["server"]["network"]["selected_interface"], "ens-rdma");
    assert_eq!(plan["server"]["network"]["reason"], "common-rdma-interface");
    assert_eq!(
        plan["server"]["network"]["machines"]["node-a"]["default_route_interface"],
        "enx-link-local"
    );
    assert_eq!(first.command.env["NCCL_SOCKET_IFNAME"], "ens-rdma");
    assert_eq!(second.command.env["NCCL_SOCKET_IFNAME"], "ens-rdma");
    let first_cache = &first.runtime_cache;
    let second_cache = &second.runtime_cache;
    assert_ne!(first_cache.path, second_cache.path);
    assert_eq!(first_cache.namespace.machine, "node-a");
    assert_eq!(first_cache.namespace.process, "server-rank-000");
    assert_eq!(second_cache.namespace.machine, "node-b");
    assert_eq!(second_cache.namespace.process, "server-rank-001");
    assert_eq!(
        first.command.env["FLASHINFER_WORKSPACE_BASE"],
        first_cache.path.join("flashinfer").to_string_lossy()
    );
    assert!(second.command.argv.iter().any(|arg| arg == "--headless"));
    Ok(())
}

#[test]
fn device_groups_can_place_multiple_ranks_on_one_machine() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    fs::write(
        workspace.root.path().join(".inferlab/local.toml"),
        format!(
            "default_placement = \"local\"\n\
             \n\
             [model_weights.dsv4]\n\
             locator = {:?}\n\
             \n\
             [machines.local]\n\
             host = \"127.0.0.1\"\n\
             ports = [8000, 8001, 8002]\n\
             devices = [0, 1, 2, 3]\n\
             \n\
             [placements.local.roles.serve]\n\
             ranks = [\n\
               {{ machine = \"local\", devices = [0, 1] }},\n\
               {{ machine = \"local\", devices = [2, 3] }},\n\
             ]\n",
            workspace.private_weight,
        ),
    )?;

    let plan = workspace.run_json(&[
        "serve",
        "start",
        "dsv4-qualify",
        "--case",
        "tp4",
        "--dry-run",
    ])?;
    let processes = resolved_ranks(&plan["server"])?;

    assert_eq!(processes.len(), 2);
    assert_eq!(processes[0].rank.machine, "local");
    assert_eq!(processes[1].rank.machine, "local");
    assert_eq!(processes[0].rank.rank, 0);
    assert_eq!(processes[1].rank.rank, 1);
    assert_eq!(processes[0].rank.devices, [0, 1]);
    assert_eq!(processes[1].rank.devices, [2, 3]);
    assert_eq!(processes[0].rank.ports["master"].port, 8001);
    assert_eq!(processes[1].rank.endpoint.port, 8002);
    assert!(
        processes[0]
            .rank
            .command
            .argv
            .iter()
            .any(|arg| arg == "--nnodes")
    );
    assert!(
        processes[1]
            .rank
            .command
            .argv
            .iter()
            .any(|arg| arg == "--headless")
    );
    Ok(())
}

#[test]
fn static_npmd_on_one_machine_allocates_disjoint_replicas_and_a_public_proxy()
-> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    write_executable(
        &workspace.adapter_bin.join("inferlab-adapter-vllm"),
        PD_ADAPTER,
    )?;
    let config = prefill_decode_workspace("vllm", "mooncake");
    fs::write(
        workspace.root.path().join(".inferlab/workspace.toml"),
        config,
    )?;
    fs::write(
        workspace.root.path().join(".inferlab/local.toml"),
        format!(
            "default_placement = \"local\"\n\
             \n\
             [model_weights.dsv4]\n\
             locator = {:?}\n\
             \n\
             [machines.local]\n\
             host = \"127.0.0.1\"\n\
             ports = [8100, 8101, 8102, 8103, 8200, 8201, 8000]\n\
             devices = [0, 1, 2, 3, 4, 5, 6, 7]\n\
             \n\
             [placements.local]\n\
             machines = [\"local\"]\n",
            workspace.private_weight
        ),
    )?;

    let plan = workspace.run_json(&[
        "recipe",
        "run",
        "dsv4-qualify",
        "--set",
        "server.roles.prefill.replicas=2",
        "--set",
        "server.roles.decode.replicas=2",
        "--dry-run",
    ])?;
    let processes = resolved_ranks(&plan["server"])?;
    let frontend = support::resolved_frontend(&plan["server"])?;

    assert_eq!(plan["server"]["topology"], "prefill_decode");
    assert_eq!(plan["server"]["kv_transfer"], "mooncake");
    assert!(plan["server"].get("routing").is_none());
    assert_eq!(plan["server"]["frontend"]["gateway"]["backend"], "builtin");
    assert_eq!(
        plan["server"]["frontend"]["pd_router"]["backend"],
        "builtin"
    );
    assert_eq!(
        plan["server"]["frontend"]["gateway"]["process_id"],
        "gateway"
    );
    assert_eq!(
        plan["server"]["frontend"]["pd_router"]["process_id"],
        "gateway"
    );
    assert_eq!(
        plan["server"]["frontend"]["processes"]
            .as_array()
            .map(Vec::len),
        Some(1)
    );
    assert_eq!(
        plan["server"]["explicit_overrides"],
        serde_json::json!([
            "server.roles.prefill.replicas=2",
            "server.roles.decode.replicas=2"
        ])
    );
    assert_eq!(
        plan["server"]["frontend"]["pd_router"]["policies"]["prefill"],
        "round_robin"
    );
    assert_eq!(
        plan["server"]["frontend"]["pd_router"]["policies"]["decode"],
        "round_robin"
    );
    assert_eq!(
        plan["server"]["frontend"]["gateway"]["implementation"],
        "vllm_mooncake"
    );
    assert_eq!(
        plan["server"]["frontend"]["pd_router"]["implementation"],
        "vllm_mooncake"
    );
    assert_eq!(processes.len(), 4);
    assert_eq!(
        plan["server"]["declarations"][0]["common"]["kv_transfer"],
        "mooncake"
    );
    assert_eq!(
        plan["server"]["declarations"][2]["source"],
        serde_json::json!({"kind": "invocation", "index": 0})
    );
    assert_eq!(
        plan["server"]["declarations"][2]["roles"]["prefill"]["replicas"],
        2
    );
    assert_eq!(processes[0].replica_id, "prefill-000");
    assert_eq!(processes[0].rank.devices, [0, 1]);
    assert_eq!(processes[0].rank.endpoint.port, 8100);
    assert_eq!(processes[0].rank.ports["bootstrap"].port, 8101);
    assert_eq!(processes[1].replica_id, "prefill-001");
    assert_eq!(processes[1].rank.devices, [2, 3]);
    assert_eq!(processes[1].rank.endpoint.port, 8102);
    assert_eq!(processes[2].replica_id, "decode-000");
    assert_eq!(processes[3].replica_id, "decode-001");
    assert_eq!(
        frontend.dependencies,
        ["prefill-000", "prefill-001", "decode-000", "decode-001"]
    );
    assert_eq!(frontend.id, "gateway");
    assert_eq!(frontend.process_role, "gateway");
    assert_eq!(frontend.components, ["gateway", "pd_router"]);
    assert!(frontend.devices.is_empty());
    assert_eq!(frontend.command.env["CUDA_VISIBLE_DEVICES"], "");
    assert!(
        frontend
            .command
            .explicit_env
            .iter()
            .any(|name| name == "CUDA_VISIBLE_DEVICES")
    );
    assert_eq!(frontend.endpoint.port, 8000);
    assert_eq!(frontend.command.argv[1], "__internal");
    let proxy_argv = &frontend.command.argv;
    assert_eq!(
        proxy_argv.iter().filter(|arg| *arg == "--prefill").count(),
        2
    );
    assert_eq!(
        proxy_argv.iter().filter(|arg| *arg == "--decode").count(),
        2
    );
    assert_eq!(
        plan["server"]["roles"][0]["replicas"]
            .as_array()
            .map(Vec::len),
        Some(2)
    );
    assert_eq!(plan["server"]["roles"][0]["declared_replica_count"], 2);
    assert_eq!(plan["server"]["roles"][0]["effective_replica_count"], 2);
    assert_eq!(
        plan["server"]["roles"][1]["replicas"]
            .as_array()
            .map(Vec::len),
        Some(2)
    );
    assert_eq!(plan["server"]["endpoint"]["port"], 8000);
    assert_eq!(plan["measurements"]["evals"][0]["endpoint"]["port"], 8000);
    assert_eq!(
        plan["server"]["endpoint"]["completions_path"],
        "/v1/completions"
    );
    assert_eq!(
        plan["server"]["endpoint"]["chat_completions_path"],
        "/v1/chat/completions"
    );
    assert_eq!(
        plan["measurements"]["evals"][0]["endpoint"]["completions_path"],
        plan["server"]["endpoint"]["completions_path"]
    );
    assert_eq!(
        plan["measurements"]["evals"][0]["endpoint"]["chat_completions_path"],
        plan["server"]["endpoint"]["chat_completions_path"]
    );
    assert_eq!(plan["server"]["links"][0]["source"], "gateway");
    assert_eq!(plan["server"]["links"][1]["source"], "pd_router");
    assert_eq!(plan["server"]["links"][2]["kind"], "kv_transfer");
    assert!(!workspace.root.path().join(".inferlab/records").exists());
    Ok(())
}

#[test]
fn heterogeneous_pd_parallelism_places_one_prefill_replica_across_nodes()
-> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    write_executable(
        &workspace.adapter_bin.join("inferlab-adapter-vllm"),
        PD_ADAPTER,
    )?;
    fs::write(
        workspace.root.path().join(".inferlab/workspace.toml"),
        prefill_decode_workspace("vllm", "mooncake"),
    )?;
    fs::write(
        workspace.root.path().join(".inferlab/local.toml"),
        format!(
            "default_placement = \"heterogeneous\"\n\
             \n\
             [model_weights.dsv4]\n\
             locator = {:?}\n\
             \n\
             [machines.prefill-a]\n\
             host = \"prefill-a.example\"\n\
             ports = [8100, 8101, 8102]\n\
             devices = [0, 1]\n\
             \n\
             [machines.prefill-b]\n\
             host = \"prefill-b.example\"\n\
             ports = [8110, 8111]\n\
             devices = [2, 3]\n\
             \n\
             [machines.decode]\n\
             host = \"decode.example\"\n\
             ports = [8200, 8201]\n\
             devices = [4, 5]\n\
             \n\
             [machines.gateway]\n\
             host = \"127.0.0.1\"\n\
             ports = [8000]\n\
             devices = []\n\
             \n\
             [placements.heterogeneous.roles.prefill]\n\
             ranks = [\n\
               {{ machine = \"prefill-a\", devices = [0, 1] }},\n\
               {{ machine = \"prefill-b\", devices = [2, 3] }},\n\
             ]\n\
             \n\
             [placements.heterogeneous.roles.decode]\n\
             machine = \"decode\"\n\
             devices = [4, 5]\n\
             \n\
             [placements.heterogeneous.roles.gateway]\n\
             machine = \"gateway\"\n\
             devices = []\n\
             endpoint_port = 8000\n",
            workspace.private_weight,
        ),
    )?;

    let plan = workspace.run_json(&[
        "recipe",
        "run",
        "dsv4-qualify",
        "--set",
        "server.roles.prefill.parallelism.outer.tensor_parallel_size=4",
        "--set",
        "server.roles.decode.parallelism.outer.tensor_parallel_size=2",
        "--dry-run",
    ])?;
    let processes = resolved_ranks(&plan["server"])?;
    let frontend = support::resolved_frontend(&plan["server"])?;
    let prefill = &plan["server"]["roles"][0];
    let decode = &plan["server"]["roles"][1];

    assert_eq!(
        prefill["effective_parallelism"]["outer"]["tensor_parallel_size"],
        4
    );
    assert_eq!(
        prefill["declared_parallelism"]["outer"]["tensor_parallel_size"],
        4
    );
    assert_eq!(prefill["replicas"][0]["device_count"], 4);
    assert_eq!(
        prefill["replicas"][0]["ranks"].as_array().map(Vec::len),
        Some(2)
    );
    assert_eq!(
        decode["effective_parallelism"]["outer"]["tensor_parallel_size"],
        2
    );
    assert_eq!(
        decode["declared_parallelism"]["outer"]["tensor_parallel_size"],
        2
    );
    assert_eq!(decode["replicas"][0]["device_count"], 2);
    assert_eq!(
        decode["replicas"][0]["ranks"].as_array().map(Vec::len),
        Some(1)
    );
    assert_eq!(processes.len(), 3);
    assert_eq!(processes[0].rank.machine, "prefill-a");
    assert_eq!(processes[0].rank.devices, [0, 1]);
    assert_eq!(processes[1].rank.machine, "prefill-b");
    assert_eq!(processes[1].rank.devices, [2, 3]);
    assert_eq!(processes[2].rank.machine, "decode");
    assert_eq!(processes[2].rank.devices, [4, 5]);
    assert_eq!(frontend.machine, "gateway");
    assert!(frontend.devices.is_empty());
    Ok(())
}

#[test]
fn single_replica_list_placement_is_rejected() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    fs::write(
        workspace.root.path().join(".inferlab/local.toml"),
        format!(
            "default_placement = \"local\"\n\
             \n\
             [model_weights.dsv4]\n\
             locator = {:?}\n\
             \n\
             [machines.local]\n\
             host = \"127.0.0.1\"\n\
             ports = [8000]\n\
             devices = [0, 1]\n\
             \n\
             [placements.local.roles.serve]\n\
             replicas = [\n\
               {{ machine = \"local\", devices = [0, 1] }},\n\
             ]\n",
            workspace.private_weight,
        ),
    )?;

    let output = workspace.run(&["serve", "start", "dsv4-qualify", "--dry-run"])?;

    assert!(!output.status.success());
    assert!(output.stdout.is_empty());
    assert!(!workspace.root.path().join(".inferlab/records").exists());
    Ok(())
}

#[test]
fn sglang_builtin_proxy_dry_run_preserves_prefill_bootstrap_triples() -> Result<(), Box<dyn Error>>
{
    let workspace = TestWorkspace::new()?;
    let adapter = PD_ADAPTER
        .replace("framework = \"vllm\"", "framework = \"sglang\"")
        .replace(
            "\"mechanism\": \"mooncake\"",
            "\"mechanism\": input[\"kv_transfer\"]",
        );
    write_executable(
        &workspace.adapter_bin.join("inferlab-adapter-sglang"),
        &adapter,
    )?;
    let manifest_path = workspace.root.path().join("pixi.toml");
    let manifest = fs::read_to_string(&manifest_path)?.replace(
        "inferlab-integration-vllm = \"==0.1.0\"",
        "inferlab-integration-vllm = \"==0.1.0\"\n\
         inferlab-integration-sglang = \"==0.1.0\"",
    );
    fs::write(manifest_path, manifest)?;
    let config = prefill_decode_workspace("sglang", "mooncake");
    fs::write(
        workspace.root.path().join(".inferlab/workspace.toml"),
        config,
    )?;
    fs::write(
        workspace.root.path().join(".inferlab/local.toml"),
        format!(
            "default_placement = \"local\"\n\
             \n\
             [model_weights.dsv4]\n\
             locator = {:?}\n\
             \n\
             [machines.local]\n\
             host = \"127.0.0.1\"\n\
             ports = [8100, 8101, 8102, 8103, 8200, 8201, 8000]\n\
             devices = [0, 1, 2, 3, 4, 5, 6, 7]\n\
             \n\
             [placements.local]\n\
             machines = [\"local\"]\n",
            workspace.private_weight
        ),
    )?;

    for transport in ["mooncake", "nixl"] {
        let transport_override = format!("server.kv_transfer={transport:?}");
        let plan = workspace.run_json(&[
            "recipe",
            "run",
            "dsv4-qualify",
            "--set",
            "server.roles.prefill.replicas=2",
            "--set",
            "server.roles.decode.replicas=2",
            "--set",
            &transport_override,
            "--dry-run",
        ])?;
        let processes = resolved_ranks(&plan["server"])?;
        let frontend = support::resolved_frontend(&plan["server"])?;
        let proxy_argv = &frontend.command.argv;
        assert_eq!(proxy_argv[3], "sglang");

        let actual = proxy_argv
            .windows(4)
            .filter(|window| window[0] == "--prefill")
            .map(|window| window[1..].to_vec())
            .collect::<Vec<_>>();
        let expected = processes
            .iter()
            .filter(|process| process.role_id == "prefill" && process.rank.rank == 0)
            .map(|process| {
                vec![
                    format!(
                        "http://{}:{}",
                        process.rank.endpoint.host, process.rank.endpoint.port
                    ),
                    process.rank.ports["bootstrap"].host.clone(),
                    process.rank.ports["bootstrap"].port.to_string(),
                ]
            })
            .collect::<Vec<_>>();
        assert_eq!(actual, expected, "transport {transport}");
    }
    Ok(())
}

#[test]
fn trtllm_builtin_proxy_dry_run_uses_rank_zero_worker_urls_without_auxiliary_ports()
-> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    let adapter = PD_ADAPTER
        .replace(
            "framework = \"vllm\"",
            "framework = \"tensorrt-llm\"",
        )
        .replace(
            "ports = [\"bootstrap\"] if role[\"kind\"] == \"prefill\" else []",
            "ports = []",
        )
        .replace("\"mechanism\": \"mooncake\"", "\"mechanism\": \"nixl\"")
        .replace(
            "            {\"kind\": \"bootstrap\", \"source\": \"pd_router\", \"target\": \"prefill\", \"port\": \"bootstrap\"},\n",
            "",
        );
    write_executable(
        &workspace.adapter_bin.join("inferlab-adapter-tensorrt-llm"),
        &adapter,
    )?;
    let manifest_path = workspace.root.path().join("pixi.toml");
    let manifest = fs::read_to_string(&manifest_path)?.replace(
        "inferlab-integration-vllm = \"==0.1.0\"",
        "inferlab-integration-vllm = \"==0.1.0\"\n\
         inferlab-integration-tensorrt-llm = \"==0.1.0\"",
    );
    fs::write(manifest_path, manifest)?;
    let config = prefill_decode_workspace("tensorrt-llm", "nixl");
    fs::write(
        workspace.root.path().join(".inferlab/workspace.toml"),
        config,
    )?;
    fs::write(
        workspace.root.path().join(".inferlab/local.toml"),
        format!(
            "default_placement = \"local\"\n\
             \n\
             [model_weights.dsv4]\n\
             locator = {:?}\n\
             \n\
             [machines.local]\n\
             host = \"127.0.0.1\"\n\
             ports = [8100, 8101, 8102, 8103, 8104, 8105, 8106, 8107, 8108, 8109, 8110, 8111, 8000]\n\
             devices = [0, 1, 2, 3, 4, 5, 6, 7]\n\
             \n\
             [[placements.local.roles.prefill.replicas]]\n\
             ranks = [\n\
               {{ machine = \"local\", devices = [0] }},\n\
               {{ machine = \"local\", devices = [1] }},\n\
             ]\n\
             \n\
             [[placements.local.roles.prefill.replicas]]\n\
             ranks = [\n\
               {{ machine = \"local\", devices = [2] }},\n\
               {{ machine = \"local\", devices = [3] }},\n\
             ]\n\
             \n\
             [[placements.local.roles.decode.replicas]]\n\
             ranks = [\n\
               {{ machine = \"local\", devices = [4] }},\n\
               {{ machine = \"local\", devices = [5] }},\n\
             ]\n\
             \n\
             [[placements.local.roles.decode.replicas]]\n\
             ranks = [\n\
               {{ machine = \"local\", devices = [6] }},\n\
               {{ machine = \"local\", devices = [7] }},\n\
             ]\n\
             \n\
             [placements.local.roles.gateway]\n\
             machine = \"local\"\n\
             devices = []\n\
             endpoint_port = 8000\n",
            workspace.private_weight
        ),
    )?;

    let plan = workspace.run_json(&[
        "recipe",
        "run",
        "dsv4-qualify",
        "--set",
        "server.roles.prefill.replicas=2",
        "--set",
        "server.roles.decode.replicas=2",
        "--dry-run",
    ])?;
    let processes = resolved_ranks(&plan["server"])?;
    let frontend = support::resolved_frontend(&plan["server"])?;
    let proxy_argv = &frontend.command.argv;

    assert_eq!(plan["server"]["frontend"]["gateway"]["backend"], "builtin");
    assert_eq!(
        plan["server"]["frontend"]["pd_router"]["backend"],
        "builtin"
    );
    assert_eq!(
        plan["server"]["frontend"]["pd_router"]["policies"]["prefill"],
        "round_robin"
    );
    assert_eq!(
        plan["server"]["frontend"]["gateway"]["implementation"],
        "trtllm"
    );
    assert_eq!(
        plan["server"]["frontend"]["gateway"]["implementation_version"],
        "2"
    );
    assert_eq!(plan["server"]["endpoint"]["port"], 8000);
    assert_eq!(frontend.endpoint.host, plan["server"]["endpoint"]["host"]);
    assert_eq!(frontend.endpoint.port, 8000);
    assert_eq!(proxy_argv[3], "trtllm");

    for role in ["prefill", "decode"] {
        let flag = format!("--{role}");
        let actual = proxy_argv
            .windows(2)
            .filter(|window| window[0] == flag)
            .map(|window| window[1].as_str())
            .collect::<Vec<_>>();
        let expected = processes
            .iter()
            .filter(|process| process.role_id == role && process.rank.rank == 0)
            .map(|process| {
                format!(
                    "http://{}:{}",
                    process.rank.endpoint.host, process.rank.endpoint.port
                )
            })
            .collect::<Vec<_>>();
        assert_eq!(actual.len(), 2);
        assert_eq!(
            actual,
            expected.iter().map(String::as_str).collect::<Vec<_>>()
        );
        assert!(
            processes
                .iter()
                .any(|process| process.role_id == role && process.rank.rank == 1)
        );
    }
    assert_eq!(frontend.components, ["gateway", "pd_router"]);
    assert!(processes.iter().all(|process| {
        !process
            .rank
            .command
            .argv
            .iter()
            .any(|arg| arg == "disaggregated")
    }));
    assert!(processes.iter().all(|process| {
        let ports = &process.rank.ports;
        ports.get("bootstrap").is_none() && ports.get("side_channel").is_none()
    }));
    assert_eq!(
        plan["server"]["links"]
            .as_array()
            .map(|links| links.iter().map(|link| &link["kind"]).collect::<Vec<_>>()),
        Some(vec![
            &serde_json::json!("request_routing"),
            &serde_json::json!("request_routing"),
            &serde_json::json!("kv_transfer")
        ])
    );
    Ok(())
}

#[test]
fn built_in_proxy_prefers_the_local_machine_in_a_remote_first_placement()
-> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    write_executable(
        &workspace.adapter_bin.join("inferlab-adapter-vllm"),
        PD_ADAPTER,
    )?;
    let config = prefill_decode_workspace("vllm", "mooncake");
    fs::write(
        workspace.root.path().join(".inferlab/workspace.toml"),
        config,
    )?;
    fs::write(
        workspace.root.path().join(".inferlab/local.toml"),
        format!(
            "default_placement = \"pair\"\n\
             \n\
             [model_weights.dsv4]\n\
             locator = {:?}\n\
             \n\
             [machines.remote]\n\
             host = \"127.0.0.1\"\n\
             ports = [8100, 8101, 8102]\n\
             devices = [0, 1]\n\
             workspace = {:?}\n\
             launch = {{ kind = \"ssh\", target = \"remote\" }}\n\
             \n\
             [machines.local]\n\
             host = \"127.0.0.1\"\n\
             ports = [8200, 8201]\n\
             devices = [2, 3]\n\
             \n\
             [placements.pair]\n\
             machines = [\"remote\", \"local\"]\n",
            workspace.private_weight,
            workspace.root.path(),
        ),
    )?;

    let plan = workspace.run_json(&["recipe", "run", "dsv4-qualify", "--dry-run"])?;
    let processes = resolved_ranks(&plan["server"])?;
    let frontend = support::resolved_frontend(&plan["server"])?;

    assert_eq!(processes[0].rank.machine, "remote");
    assert_eq!(processes[1].rank.machine, "local");
    assert_eq!(frontend.machine, "local");
    assert_eq!(frontend.launch, LaunchProjection::Local);
    Ok(())
}

#[test]
fn machine_binding_selects_runtime_cache_storage_root() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    let cache_root = workspace.root.path().join("machine-cache");
    fs::write(
        workspace.root.path().join(".inferlab/local.toml"),
        format!(
            "default_placement = \"local\"\n\
             \n\
             [model_weights.dsv4]\n\
             locator = {:?}\n\
             \n\
             [machines.local]\n\
             host = \"127.0.0.1\"\n\
             ports = [8000]\n\
             devices = [0, 1, 2, 3, 4, 5, 6, 7]\n\
             cache_root = {:?}\n\
             \n\
             [placements.local]\n\
             machines = [\"local\"]\n",
            workspace.private_weight, cache_root,
        ),
    )?;

    let plan = workspace.run_json(&["serve", "start", "dsv4-qualify", "--dry-run"])?;
    let process = resolved_rank(&plan["server"], "server")?;
    let cache = &process.runtime_cache;
    assert_eq!(cache.storage_root_source, "machine-binding");
    assert_eq!(cache.storage_root, cache_root);
    assert!(cache.path.starts_with(&cache_root));
    Ok(())
}

#[test]
fn two_node_resolution_rejects_placements_without_a_common_routable_interface()
-> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    fs::write(
        workspace.root.path().join(".inferlab/local.toml"),
        format!(
            "default_placement = \"pair\"\n\
             \n\
             [model_weights.dsv4]\n\
             locator = {:?}\n\
             \n\
             [machines.node-a]\n\
             host = \"node-a.example\"\n\
             ports = [8000, 29501]\n\
             devices = [0, 1]\n\
             \n\
             [machines.node-b]\n\
             host = \"node-b.example\"\n\
             ports = [8000]\n\
             devices = [2, 3]\n\
             \n\
             [placements.pair.roles.serve]\n\
             ranks = [\n\
               {{ machine = \"node-a\", devices = [0, 1] }},\n\
               {{ machine = \"node-b\", devices = [2, 3] }},\n\
             ]\n",
            workspace.private_weight,
        ),
    )?;

    let output = workspace
        .command()
        .env("FAKE_NETWORK_MODE", "link-local-only")
        .args([
            "serve",
            "start",
            "dsv4-qualify",
            "--case",
            "tp4",
            "--dry-run",
        ])
        .output()?;

    assert!(!output.status.success());
    assert!(
        String::from_utf8(output.stderr)?.contains("no common routable communication interface")
    );
    Ok(())
}

#[test]
fn explicit_case_and_server_override_preserve_ordered_declarations() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    let plan = workspace.run_json(&[
        "serve",
        "start",
        "dsv4-qualify",
        "--case",
        "tp4",
        "--set",
        "server.settings.max_model_len=32768",
        "--set",
        "server.parallelism.attention.data_parallel_size=2",
        "--set",
        "server.settings.\"literal.key\"=17",
        "--set",
        "server.roles.serve.settings.block_size=32",
        "--dry-run",
    ])?;

    assert_eq!(plan["server"]["case"]["id"], "tp4");
    assert_eq!(plan["server"]["case"]["selection"], "explicit");
    assert_eq!(
        plan["server"]["roles"][0]["declared_parallelism"]["outer"]["tensor_parallel_size"],
        4
    );
    assert_eq!(
        plan["server"]["roles"][0]["declared_parallelism"]["attention"]["data_parallel_size"],
        2
    );
    assert_eq!(
        plan["server"]["roles"][0]["declared_settings"]["max_model_len"],
        32768
    );
    assert_eq!(
        plan["server"]["roles"][0]["declared_settings"]["literal.key"],
        17
    );
    assert_eq!(
        plan["server"]["explicit_overrides"],
        serde_json::json!([
            "server.settings.max_model_len=32768",
            "server.parallelism.attention.data_parallel_size=2",
            "server.settings.\"literal.key\"=17",
            "server.roles.serve.settings.block_size=32"
        ])
    );
    let declarations = plan["server"]["declarations"]
        .as_array()
        .ok_or("server declarations are not an array")?;
    assert_eq!(declarations.len(), 6);
    assert_eq!(
        declarations[0]["source"],
        serde_json::json!({"kind": "server", "id": "dsv4-qualify"})
    );
    assert_eq!(
        declarations[0]["common"]["parallelism"]["outer"]["pipeline_parallel_size"],
        1
    );
    assert_eq!(
        declarations[0]["roles"]["serve"]["settings"]["block_size"],
        16
    );
    assert_eq!(
        declarations[1]["source"],
        serde_json::json!({"kind": "case", "id": "tp4"})
    );
    assert_eq!(
        declarations[1]["common"]["parallelism"]["outer"]["tensor_parallel_size"],
        4
    );
    assert_eq!(
        declarations[2]["source"],
        serde_json::json!({"kind": "invocation", "index": 0})
    );
    assert_eq!(
        declarations[2]["common"]["settings"]["max_model_len"],
        32768
    );
    assert_eq!(
        declarations[3]["source"],
        serde_json::json!({"kind": "invocation", "index": 1})
    );
    assert_eq!(
        declarations[3]["common"]["parallelism"]["attention"]["data_parallel_size"],
        2
    );
    assert_eq!(
        declarations[5]["source"],
        serde_json::json!({"kind": "invocation", "index": 3})
    );
    assert_eq!(
        declarations[5]["roles"]["serve"]["settings"]["block_size"],
        32
    );
    assert_eq!(
        plan["server"]["roles"][0]["effective_parallelism"]["attention"]["tensor_parallel_size"],
        4
    );
    assert_eq!(plan["server"]["resources"]["device_count"], 8);
    Ok(())
}

#[test]
fn runtime_deadlines_use_the_server_case_and_invocation_patch_precedence()
-> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    let path = workspace.root.path().join(".inferlab/workspace.toml");
    let config = fs::read_to_string(&path)?.replace(
        "[servers.dsv4-qualify.cases.tp4.parallelism.outer]",
        "[servers.dsv4-qualify.cases.tp4]\nreadiness_timeout_seconds = 1200\nreadiness_attempt_timeout_seconds = 45\ncapture_arm_deadline_seconds = 46\ncapture_control_deadline_seconds = 47\ncapture_finalization_deadline_seconds = 48\n\n\
         [servers.dsv4-qualify.cases.tp4.parallelism.outer]",
    );
    fs::write(path, config)?;

    let case_plan = workspace.run_json(&[
        "serve",
        "start",
        "dsv4-qualify",
        "--case",
        "tp4",
        "--dry-run",
    ])?;
    assert_eq!(case_plan["server"]["readiness_timeout_seconds"], 1200);
    assert_eq!(case_plan["server"]["readiness_attempt_timeout_seconds"], 45);
    assert_eq!(case_plan["server"]["capture_arm_deadline_seconds"], 46);
    assert_eq!(case_plan["server"]["capture_control_deadline_seconds"], 47);
    assert_eq!(
        case_plan["server"]["capture_finalization_deadline_seconds"],
        48
    );
    assert_eq!(
        case_plan["server"]["declarations"][1]["source"],
        serde_json::json!({"kind": "case", "id": "tp4"})
    );
    assert_eq!(
        case_plan["server"]["declarations"][1]["common"]["readiness_timeout_seconds"],
        1200
    );
    assert_eq!(
        case_plan["server"]["declarations"][1]["common"]["readiness_attempt_timeout_seconds"],
        45
    );
    assert_eq!(
        case_plan["server"]["declarations"][1]["common"]["capture_arm_deadline_seconds"],
        46
    );
    assert_eq!(
        case_plan["server"]["declarations"][1]["common"]["capture_control_deadline_seconds"],
        47
    );
    assert_eq!(
        case_plan["server"]["declarations"][1]["common"]["capture_finalization_deadline_seconds"],
        48
    );

    let invocation_plan = workspace.run_json(&[
        "serve",
        "start",
        "dsv4-qualify",
        "--case",
        "tp4",
        "--set",
        "server.readiness_timeout_seconds=1800",
        "--set",
        "server.readiness_attempt_timeout_seconds=75",
        "--set",
        "server.capture_arm_deadline_seconds=76",
        "--set",
        "server.capture_control_deadline_seconds=77",
        "--set",
        "server.capture_finalization_deadline_seconds=78",
        "--dry-run",
    ])?;
    assert_eq!(invocation_plan["server"]["readiness_timeout_seconds"], 1800);
    assert_eq!(
        invocation_plan["server"]["readiness_attempt_timeout_seconds"],
        75
    );
    assert_eq!(
        invocation_plan["server"]["capture_arm_deadline_seconds"],
        76
    );
    assert_eq!(
        invocation_plan["server"]["capture_control_deadline_seconds"],
        77
    );
    assert_eq!(
        invocation_plan["server"]["capture_finalization_deadline_seconds"],
        78
    );
    assert_eq!(
        invocation_plan["server"]["declarations"][2]["source"],
        serde_json::json!({"kind": "invocation", "index": 0})
    );
    assert_eq!(
        invocation_plan["server"]["declarations"][3]["source"],
        serde_json::json!({"kind": "invocation", "index": 1})
    );
    assert_eq!(
        invocation_plan["server"]["declarations"][2]["common"]["readiness_timeout_seconds"],
        1800
    );
    assert_eq!(
        invocation_plan["server"]["declarations"][3]["common"]["readiness_attempt_timeout_seconds"],
        75
    );
    assert_eq!(
        invocation_plan["server"]["declarations"][4]["common"]["capture_arm_deadline_seconds"],
        76
    );
    assert_eq!(
        invocation_plan["server"]["declarations"][5]["common"]["capture_control_deadline_seconds"],
        77
    );
    assert_eq!(
        invocation_plan["server"]["declarations"][6]["common"]["capture_finalization_deadline_seconds"],
        78
    );
    Ok(())
}

#[test]
fn readiness_attempt_timeout_must_be_positive() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    let output = workspace.run(&[
        "serve",
        "start",
        "dsv4-qualify",
        "--set",
        "server.readiness_attempt_timeout_seconds=0",
        "--dry-run",
    ])?;

    assert!(!output.status.success());
    assert!(
        String::from_utf8_lossy(&output.stderr)
            .contains("readiness_attempt_timeout_seconds must be nonzero")
    );
    Ok(())
}

#[test]
fn local_adapter_timeout_must_be_positive() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    let local = workspace.root.path().join(".inferlab/local.toml");
    let mut bindings = fs::read_to_string(&local)?;
    bindings.push_str("\n[adapter]\ntimeout_seconds = 0\n");
    fs::write(local, bindings)?;

    let output = workspace.run(&["serve", "start", "dsv4-qualify", "--dry-run"])?;
    assert!(!output.status.success());
    let diagnostics = String::from_utf8_lossy(&output.stderr);
    assert!(
        diagnostics.contains("adapter timeout_seconds must be positive"),
        "{diagnostics}"
    );
    Ok(())
}

#[test]
fn local_adapter_timeout_bounds_the_process_invocation() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    let local = workspace.root.path().join(".inferlab/local.toml");
    let mut bindings = fs::read_to_string(&local)?;
    bindings.push_str("\n[adapter]\ntimeout_seconds = 1\n");
    fs::write(local, bindings)?;

    let started = Instant::now();
    let output = workspace
        .command()
        .env("FIXTURE_ADAPTER_HANG", "1")
        .args(["serve", "start", "dsv4-qualify", "--dry-run"])
        .output()?;
    let elapsed = started.elapsed();

    assert!(!output.status.success());
    let diagnostics = String::from_utf8_lossy(&output.stderr);
    assert!(
        diagnostics.contains("integration \"vllm\" did not finish within 1 seconds"),
        "{diagnostics}"
    );
    assert!(elapsed >= Duration::from_millis(900), "elapsed {elapsed:?}");
    assert!(elapsed < Duration::from_secs(5), "elapsed {elapsed:?}");
    Ok(())
}

#[test]
fn profiler_deadlines_must_be_positive() -> Result<(), Box<dyn Error>> {
    for field in [
        "capture_arm_deadline_seconds",
        "capture_control_deadline_seconds",
        "capture_finalization_deadline_seconds",
    ] {
        let workspace = TestWorkspace::new()?;
        let output = workspace.run(&[
            "serve",
            "start",
            "dsv4-qualify",
            "--set",
            &format!("server.{field}=0"),
            "--dry-run",
        ])?;

        assert!(!output.status.success());
        assert!(
            String::from_utf8_lossy(&output.stderr).contains(&format!("{field} must be nonzero"))
        );
    }
    Ok(())
}
