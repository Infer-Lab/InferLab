//! A Gateway backend that declares a discovery service, registry-membership
//! readiness, and per-target prefix-cache reset resolves into one discovery
//! process beside the Gateway, address-keyed readiness targets, and one reset
//! URL per model-serving replica ([[RFC-0003:C-SERVE-TOPOLOGY]],
//! [[RFC-0006:C-INTEGRATIONS]]).

mod dry_run_support;
mod support;

use std::error::Error;
use std::fs;

use dry_run_support::*;

/// The P/D fixture adapter, turned into an integration-rendered frontend that
/// requires a discovery process and registers workers by request-plane port.
fn discovery_adapter() -> Result<String, Box<dyn Error>> {
    let replacements = [
        (
            r#"ports = ["bootstrap"] if role["kind"] == "prefill" else []"#,
            r#"ports = ["request", "side_channel"]"#,
        ),
        (
            r#"            "render_inputs": [],
        })
        tp = parallelism"#,
            r#"            "render_inputs": [],
            "replica_prefix_cache_reset": {"method": "post", "path": "/engine/flush_cache", "success": {"pointer": "/status", "value": "ok"}},
        })
        tp = parallelism"#,
        ),
        (
            r#"            {"kind": "kv_transfer", "source": "prefill", "target": "decode", "mechanism": "mooncake"},
            {"kind": "bootstrap", "source": "pd_router", "target": "prefill", "port": "bootstrap"},
"#,
            r#"            {"kind": "kv_transfer", "source": "prefill", "target": "decode", "mechanism": "nixl"},
            {"kind": "side_channel", "source": "prefill", "target": "decode", "port": "side_channel"},
"#,
        ),
        (
            r#"            "targets": [{"kind": "pd_router"}],"#,
            r#"            "targets": [{"kind": "pd_router"}],
            "discovery": {"ports": ["peer"], "readiness": {"kind": "http", "path": "/health"}},"#,
        ),
        (
            r#"            "readiness": readiness,
            "handoff": "in_process","#,
            r#"            "readiness": {"kind": "registry_membership", "registry_path": "/health", "target_port": "request", "entries_pointer": "/instances", "entry_filter": {"pointer": "/endpoint", "value": "generate"}, "role_pointer": "/component", "role_values": {"prefill": "prefill", "decode": "backend"}, "address_pointer": "/transport/tcp", "model_list": {"path": "/v1/models", "models_pointer": "/data", "name_pointer": "/id"}},
            "handoff": "in_process","#,
        ),
        (
            r#"            {
                "kind": "model_rank",
                "process": allocation["process"],
                "role": allocation["role"],
                "replica": allocation["replica"],
                "rank": allocation["rank"],
                "rank_count": allocation["rank_count"],
                "launch_files": [],
                "command": {"argv": ["fixture-server", allocation["process"]], "env": {}},
            }
            for allocation in input["allocations"]"#,
            r#"            {
                "kind": "model_rank",
                "process": allocation["process"],
                "role": allocation["role"],
                "replica": allocation["replica"],
                "rank": allocation["rank"],
                "rank_count": allocation["rank_count"],
                "launch_files": [],
                "command": {"argv": ["fixture-server", allocation["process"], "--discovery", allocation.get("discovery") or "none"], "env": {}},
            } if allocation["kind"] == "model_rank" else {
                "kind": allocation["kind"],
                "process": allocation["process"],
                "process_role": allocation["process_role"],
                "components": allocation["components"],
                "launch_files": [],
                "command": {"argv": ["fixture-" + allocation["kind"], allocation["process"]], "env": {}},
            }
            for allocation in input["allocations"]"#,
        ),
    ];
    let mut adapter = PD_ADAPTER.replace(
        r#""render_source": "control_plane","#,
        r#""render_source": "integration","#,
    );
    for (from, to) in replacements {
        if !adapter.contains(from) {
            return Err(format!("fixture adapter no longer contains {from:?}").into());
        }
        adapter = adapter.replacen(from, to, 1);
    }
    Ok(adapter)
}

#[test]
fn a_discovering_gateway_resolves_discovery_registry_targets_and_per_target_resets()
-> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    write_executable(
        &workspace.adapter_bin.join("inferlab-adapter-vllm"),
        &discovery_adapter()?,
    )?;
    fs::write(
        workspace.root.path().join(".inferlab/workspace.toml"),
        prefill_decode_workspace("vllm", "nixl")
            .replace(
                "gateway_backend = \"builtin\"",
                "gateway_backend = \"fixture-discovering\"",
            )
            .replace(
                "pd_router_backend = \"builtin\"",
                "pd_router_backend = \"fixture-discovering\"",
            ),
    )?;
    fs::write(
        workspace.root.path().join(".inferlab/local.toml"),
        format!(
            "default_placement = \"local\"\n\
             \n\
             [model_weights.deepseek-v4-flash]\n\
             locator = {:?}\n\
             \n\
             [machines.local]\n\
             host = \"127.0.0.1\"\n\
             ports = [8100, 8101, 8102, 8103, 8104, 8105, 8106, 8107, 8108, 8109]\n\
             devices = [0, 1, 2, 3, 4, 5, 6, 7]\n\
             \n\
             [placements.local]\n\
             machines = [\"local\"]\n",
            workspace.private_weight
        ),
    )?;

    let plan = workspace.run_json(&["serve", "start", "deepseek-v4-flash-qualify", "--dry-run"])?;
    let server = &plan["server"];

    let discovery = &server["discovery"];
    assert_eq!(discovery["id"], "discovery");
    assert_eq!(discovery["kind"], "discovery");
    assert_eq!(discovery["components"], serde_json::json!(["discovery"]));
    assert_eq!(discovery["devices"], serde_json::json!([]));
    assert_eq!(discovery["readiness"]["kind"], "http");
    assert!(
        discovery["command"]["argv"]
            .as_array()
            .is_some_and(|argv| argv.iter().any(|arg| arg == "fixture-discovery")),
        "the integration renders the discovery process: {}",
        discovery["command"]["argv"]
    );
    assert!(
        discovery["data_directory"]
            .as_str()
            .is_some_and(|path| path.ends_with("/state")),
        "{discovery}"
    );
    let gateway = &server["frontend"]["processes"][0];
    assert_eq!(discovery["machine"], gateway["machine"]);

    let readiness = &gateway["readiness"];
    assert_eq!(readiness["kind"], "registry_membership");
    assert_eq!(readiness["model_list"]["served_model"], "deepseek-v4-flash");
    let roles = server["roles"].as_array().ok_or("roles")?;
    let mut expected = Vec::new();
    let mut resets = Vec::new();
    for role in roles {
        for replica in role["replicas"].as_array().ok_or("replicas")? {
            let entry = &replica["ranks"][0];
            let request = &entry["ports"]["request"];
            expected.push(serde_json::json!({
                "process": entry["id"],
                "role": if role["kind"] == "prefill" { "prefill" } else { "backend" },
                "address": format!("{}:{}", request["host"].as_str().ok_or("host")?, request["port"]),
            }));
            resets.push(format!(
                "http://{}:{}/engine/flush_cache",
                entry["endpoint"]["host"].as_str().ok_or("host")?,
                entry["endpoint"]["port"]
            ));
            assert!(
                entry["command"]["argv"].as_array().is_some_and(|argv| argv
                    .windows(2)
                    .any(|pair| pair[0] == "--discovery" && pair[1] == "discovery")),
                "model ranks receive the discovery link: {}",
                entry["command"]["argv"]
            );
        }
    }
    assert_eq!(
        readiness["expected_targets"],
        serde_json::Value::from(expected)
    );
    let reset_urls = server["endpoint"]["replica_prefix_cache_resets"]
        .as_array()
        .ok_or("per-target resets")?
        .iter()
        .map(|reset| reset["url"].as_str().map(str::to_owned))
        .collect::<Option<Vec<_>>>()
        .ok_or("reset url")?;
    assert_eq!(reset_urls, resets);
    Ok(())
}
