//! Shared fixtures for the independently runnable dry-run integration targets.
//!
//! Cargo compiles each target as its own crate, so each target uses only the
//! subset of this shared fixture API needed by its workflow boundary.
#![allow(dead_code)]

use serde_json::Value;
use std::error::Error;
use std::ffi::OsString;
use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use tempfile::TempDir;

pub(crate) const WORKSPACE: &str = include_str!("../fixtures/dsv4-workspace.toml");

use crate::support::{ResolvedProcessProjection, ResolvedRankProjection};

pub(crate) fn resolved_ranks(
    server: &Value,
) -> Result<Vec<ResolvedProcessProjection>, Box<dyn Error>> {
    crate::support::resolved_processes(server)
}

pub(crate) fn resolved_rank(
    server: &Value,
    id: &str,
) -> Result<ResolvedRankProjection, Box<dyn Error>> {
    crate::support::resolved_process(server, id)
}

pub(crate) struct TestWorkspace {
    pub(crate) root: TempDir,
    pub(crate) adapter_bin: PathBuf,
    pub(crate) data_home: PathBuf,
    pub(crate) private_weight: String,
}

impl TestWorkspace {
    pub(crate) fn new() -> Result<Self, Box<dyn Error>> {
        let root = tempfile::tempdir()?;
        let inferlab_dir = root.path().join(".inferlab");
        let adapter_bin = root.path().join("bin");
        fs::create_dir_all(&inferlab_dir)?;
        fs::create_dir_all(&adapter_bin)?;
        fs::create_dir_all(root.path().join("vendor/vllm"))?;
        fs::create_dir_all(root.path().join("vendor/flashinfer"))?;
        fs::write(inferlab_dir.join("workspace.toml"), WORKSPACE)?;
        fs::write(root.path().join("vendor/vllm/source.txt"), "baseline\n")?;
        fs::write(
            root.path().join("vendor/flashinfer/source.txt"),
            "baseline\n",
        )?;
        fs::write(
            root.path().join("operator-config.yaml"),
            "fixture: dry-run\nunicode: 雪\n",
        )?;
        fs::write(
            root.path().join("pixi.toml"),
            "[workspace]\n\
             channels = [\"conda-forge\"]\n\
             platforms = [\"linux-64\"]\n\
             \n\
             [environments]\n\
             vllm = []\n\
             \n\
             [pypi-dependencies]\n\
             inferlab-integration-vllm = \"==0.1.0\"\n",
        )?;
        fs::write(
            root.path().join("pixi.lock"),
            "version: 6\nenvironments:\n  vllm: {}\n",
        )?;
        // ensure_usable checks this prefix exists on disk before shelling
        // out to pixi at all.
        fs::create_dir_all(root.path().join(".pixi/envs/vllm"))?;
        fs::write(root.path().join(".gitignore"), ".inferlab/local.toml\n")?;

        let private_weight = root
            .path()
            .join("private/weights/deepseek-v4-flash")
            .display()
            .to_string();
        Self::write_local_bindings(&inferlab_dir.join("local.toml"), &private_weight)?;
        Self::write_adapter(&adapter_bin.join("inferlab-adapter-vllm"))?;
        Self::write_pixi(&adapter_bin.join("pixi"))?;
        write_executable(
            &adapter_bin.join("fixture-eval-client"),
            include_str!("../fixtures/bin/recipe-eval-client.py"),
        )?;
        write_executable(
            &adapter_bin.join("fixture-bench-client"),
            include_str!("../fixtures/bin/bench-client.py"),
        )?;
        write_executable(&adapter_bin.join("ip"), NETWORK_IP)?;
        write_executable(&adapter_bin.join("ibdev2netdev"), IBDEV2NETDEV)?;
        write_executable(&adapter_bin.join("ssh"), SSH)?;
        let data_home = root.path().join("data");
        let mut path = OsString::from(&adapter_bin);
        path.push(":");
        path.push(std::env::var_os("PATH").unwrap_or_default());
        let install = Command::new(env!("CARGO_BIN_EXE_inferlab"))
            .current_dir(root.path())
            .env("PATH", path)
            .env("XDG_DATA_HOME", &data_home)
            .args(["toolchain", "install"])
            .output()?;
        if !install.status.success() {
            return Err(format!(
                "toolchain fixture install failed: {}",
                String::from_utf8_lossy(&install.stderr)
            )
            .into());
        }
        Self::git(root.path(), &["init", "-q"])?;
        Self::git(root.path(), &["config", "user.email", "test@example.com"])?;
        Self::git(root.path(), &["config", "user.name", "Inferlab Test"])?;
        Self::git(root.path(), &["add", "."])?;
        Self::git(root.path(), &["commit", "-qm", "fixture"])?;

        Ok(Self {
            root,
            adapter_bin,
            data_home,
            private_weight,
        })
    }

    pub(crate) fn write_local_bindings(
        path: &Path,
        private_weight: &str,
    ) -> Result<(), Box<dyn Error>> {
        fs::write(
            path,
            format!(
                "default_placement = \"local\"\n\
                 \n\
                 [model_weights.deepseek-v4-flash]\n\
                 locator = {private_weight:?}\n\
                 \n\
                 [machines.local]\n\
                 host = \"127.0.0.1\"\n\
                 ports = [8000]\n\
                 devices = [0, 1, 2, 3, 4, 5, 6, 7]\n\
                 \n\
                 [placements.local]\n\
                 machines = [\"local\"]\n"
            ),
        )?;
        Ok(())
    }

    pub(crate) fn write_adapter(path: &Path) -> Result<(), Box<dyn Error>> {
        fs::write(
            path,
            r#"#!/usr/bin/env python3
import hashlib
import json
import os
import sys
import time

if os.environ.get("FIXTURE_ADAPTER_HANG"):
    time.sleep(3600)

request = json.load(sys.stdin)
input = request["input"]
operation = request["operation"]
mechanism = input.get("profiling")
if operation == "plan_serve":
    role = input["roles"][0]
    gateway_backend = input.get("gateway_backend")
    settings = role["settings"]
    declared = role["parallelism"]
    outer = declared.get("outer") or {}
    attention = declared.get("attention") or {}
    experts = declared.get("experts") or {}
    tp = outer.get("tensor_parallel_size") or 1
    pp = outer.get("pipeline_parallel_size") or 1
    dp = attention.get("data_parallel_size") or 1
    ep = experts.get("expert_parallel_size") or 1
    world_size = tp * pp * dp
    effective_settings = dict(settings)
    effective_settings.setdefault("trust_remote_code", False)
    effective_settings["trust_remote_code"] = False
    effective_parallelism = {
        "outer": {"tensor_parallel_size": tp, "pipeline_parallel_size": pp},
        "attention": {
            "tensor_parallel_size": tp,
            "data_parallel_size": dp,
            "context_parallel_size": 1,
        },
        "experts": {
            "tensor_parallel_size": 1 if ep > 1 else tp * dp,
            "data_parallel_size": 1,
            "expert_parallel_size": tp * dp if ep > 1 else 1,
            "dense_tensor_parallel_size": 1,
        },
    }
    output = {
        "integration": {
            "adapter_id": "inferlab-vllm",
            "adapter_version": "0.1.0",
            "framework": "vllm",
            "framework_version": "test",
        },
        "roles": [{
            "id": role["id"],
            "kind": role["kind"],
            "declared_replica_count": role["replica_count"],
            "effective_replica_count": role["replica_count"],
            "effective_settings": effective_settings,
            "effective_parallelism": effective_parallelism,
            **({
                "public_endpoint": {
                    "protocol": "http",
                    "completions_path": "/v1/completions",
                    "chat_completions_path": "/v1/chat/completions",
                    "prefix_cache_reset": {"method": "post", "path": "/reset_prefix_cache"},
                }
            } if not gateway_backend else {}),
            "render_inputs": (
                [{"source_path": "operator-config.yaml"}]
                if settings.get("fixture_mode") == "launch-file"
                else []
            ),
        }],
        "replicas": [{
            "id": "server",
            "role_id": role["id"],
            "replica_index": 0,
            "device_count": world_size,
            "ports": [],
            "primary_ports": ["master"],
            "primary_readiness": {"kind": "http", "path": "/v1/models"},
            "worker_readiness": {"kind": "process_alive"},
            **({
                "capture_target": {
                    "mechanism": mechanism,
                    "window_control": {
                        "endpoint": (
                            "gateway"
                            if gateway_backend or settings.get("fixture_capture_gateway")
                            else "replica_entry"
                        ),
                        "start": {
                            "method": "post",
                            "path": (
                                "http://fixture.invalid/start_profile"
                                if settings.get("fixture_capture_invalid_path")
                                else "/start_profile"
                            ),
                            "body": {
                                "activities": [
                                    "GPU" if mechanism == "engine_trace" else "CUDA_PROFILER"
                                ]
                            },
                        },
                        "stop": {
                            "method": "post",
                            "path": "/stop_profile",
                        },
                    }
                }
            } if mechanism else {}),
        }],
        "links": (
            [{"kind": "request_routing", "source": "gateway", "targets": [role["id"]]}]
            if gateway_backend else []
        ),
        **({
            "gateway": {
                "backend": gateway_backend,
                "implementation": "fixture-gateway",
                "implementation_version": "1",
                "effective_settings": {},
                "endpoint": {
                    "protocol": "http",
                    "completions_path": "/v1/completions",
                    "chat_completions_path": "/v1/chat/completions",
                    "prefix_cache_reset": {"method": "post", "path": "/reset_prefix_cache"},
                },
                "readiness": {"kind": "http", "path": "/healthcheck"},
                "ports": [],
                "targets": [{"kind": "engine", "role": role["id"]}],
                "render_inputs": [],
                "render_source": "integration",
                "co_rendering": {"process_role": "gateway"},
            }
        } if gateway_backend else {}),
    }
    synthetic = input.get("synthetic_acceptance")
    if synthetic:
        # Echo a deterministic valid outcome: the declared length for the
        # explicit form, a fixed resolved pair for the curve form.
        if "explicit" in synthetic:
            output["synthetic_acceptance"] = {
                "acceptance_length": synthetic["explicit"]["acceptance_length"],
            }
        else:
            output["synthetic_acceptance"] = {"acceptance_length": 3.5, "draft_count": 4}
elif operation == "render_serve":
    allocations = input["allocations"]
    master = allocations[0]["ports"].get("master")
    processes = []
    for allocation in allocations:
        if allocation["kind"] == "frontend":
            processes.append({
                "kind": "frontend",
                "process": allocation["process"],
                "process_role": allocation["process_role"],
                "components": allocation["components"],
                "launch_files": [],
                "command": {
                    "argv": [
                        "fixture-gateway",
                        allocation["endpoint"]["host"],
                        str(allocation["endpoint"]["port"]),
                    ],
                    "env": {},
                },
            })
            continue
        parallelism = allocation["effective_parallelism"]
        settings = allocation["effective_settings"]
        tp = parallelism["outer"]["tensor_parallel_size"]
        dp = parallelism["attention"]["data_parallel_size"]
        ep = parallelism["experts"]["expert_parallel_size"]
        cache_root = allocation["cache"]
        argv = [
            "python", "-m", "vllm.entrypoints.cli.main", "serve",
            allocation["model_locator"],
            "--host", allocation["endpoint"]["host"],
            "--port", str(allocation["endpoint"]["port"]),
            "--tensor-parallel-size", str(tp),
        ]
        if dp > 1:
            argv.extend(["--data-parallel-size", str(dp)])
        if ep > 1:
            argv.append("--enable-expert-parallel")
        if allocation["rank_count"] > 1:
            argv.extend([
                "--nnodes", str(allocation["rank_count"]),
                "--node-rank", str(allocation["rank"]),
                "--master-addr", master["host"],
                "--master-port", str(master["port"]),
            ])
            if allocation["rank"]:
                argv.append("--headless")
        launch_files = []
        if settings.get("fixture_mode") == "launch-file":
            text = allocation["render_inputs"][0]["text"]
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
            relative_path = f"launch-files/{digest}/fixture.yaml"
            resolved_path = f"{cache_root}/{relative_path}"
            argv.extend(["--generation-config", resolved_path])
            launch_files.append({
                "relative_path": relative_path,
                "text": text,
                "sha256": digest,
            })
        processes.append({
            "kind": "model_rank",
            "process": allocation["process"],
            "role": allocation["role"],
            "replica": allocation["replica"],
            "rank": allocation["rank"],
            "rank_count": allocation["rank_count"],
            "launch_files": launch_files,
            "command": {
                "argv": argv,
                "env": {
                    "FLASHINFER_WORKSPACE_BASE": f"{cache_root}/flashinfer",
                    "VLLM_CACHE_ROOT": f"{cache_root}/vllm",
                },
            },
        })
    output = {
        "integration": {
            "adapter_id": "inferlab-vllm",
            "adapter_version": "0.1.0",
            "framework": "vllm",
            "framework_version": "test",
        },
        "processes": processes,
    }
else:
    raise ValueError(f"unexpected operation {operation}")
print(json.dumps({
    "status": "ok",
    "protocol_version": "9",
    "result": {
        "operation": operation,
        "output": output,
    },
}))
"#,
        )?;
        let mut permissions = fs::metadata(path)?.permissions();
        permissions.set_mode(0o755);
        fs::set_permissions(path, permissions)?;
        Ok(())
    }

    pub(crate) fn write_pixi(path: &Path) -> Result<(), Box<dyn Error>> {
        fs::write(
            path,
            "#!/bin/sh\n\
             if [ \"$1\" = info ] && [ \"$2\" = --json ]; then\n\
               case \"$(uname -m)\" in\n\
                 x86_64) platform=linux-64 ;;\n\
                 aarch64) platform=linux-aarch64 ;;\n\
                 *) platform=unsupported ;;\n\
               esac\n\
               printf '{\"platform\":\"%s\",\"virtual_packages\":[\"__glibc=2.35=0\"]}\\n' \"$platform\"\n\
               exit 0\n\
             fi\n\
             if [ \"$1\" = install ] && [ \"$2\" = --manifest-path ] && [ \"$4\" = --all ] && [ \"$5\" = --locked ]; then\n\
               prefix=\"$(dirname \"$3\")\"\n\
               mkdir -p \"$prefix/.pixi/envs/eval/bin\" \"$prefix/.pixi/envs/bench/bin\"\n\
               printf '%s\\n' '#!/bin/sh' 'if [ \"$2\" = --handshake ]; then printf '\"'\"'{\"lm_eval_version\":\"0.4.12\"}\\n'\"'\"'; exit 0; fi' 'shift' 'exec fixture-eval-client \"$@\"' > \"$prefix/.pixi/envs/eval/bin/python\"\n\
               printf '%s\\n' '#!/bin/sh' 'if [ \"$2\" = --handshake ]; then printf '\"'\"'{\"aiperf_version\":\"0.12.0\",\"transformers_version\":\"5.12.1\"}\\n'\"'\"'; exit 0; fi' 'if [ \"$1\" = -m ] && [ \"$2\" = inferlab_bench_runner.bench_client ]; then shift 2; else shift; fi' 'exec fixture-bench-client \"$@\"' > \"$prefix/.pixi/envs/bench/bin/python\"\n\
               chmod +x \"$prefix/.pixi/envs/eval/bin/python\" \"$prefix/.pixi/envs/bench/bin/python\"\n\
               exit 0\n\
             fi\n\
             if [ \"$1\" = run ] && [ \"$2\" = --locked ] && [ \"$3\" = --no-install ] && [ \"$4\" = --executable ] && [ \"$5\" = -e ] && [ \"$6\" = vllm ] && [ \"$7\" = -- ]; then\n\
               shift 7\n\
             elif [ \"$1\" = run ] && [ \"$2\" = --as-is ] && [ \"$3\" = --executable ] && [ \"$4\" = -e ] && [ \"$5\" = vllm ] && [ \"$6\" = -- ]; then\n\
               shift 6\n\
             else\n\
               printf 'unexpected pixi fixture arguments\\n' >&2\n\
               exit 2\n\
             fi\n\
             if [ \"${FAKE_PIXI_UNAVAILABLE:-0}\" = 1 ] && [ \"$1\" = true ]; then\n\
               printf 'environment prefix is missing\\n' >&2\n\
               exit 3\n\
             fi\n\
             exec \"$@\"\n",
        )?;
        let mut permissions = fs::metadata(path)?.permissions();
        permissions.set_mode(0o755);
        fs::set_permissions(path, permissions)?;
        Ok(())
    }

    pub(crate) fn git(root: &Path, args: &[&str]) -> Result<(), Box<dyn Error>> {
        let output = Command::new("git").current_dir(root).args(args).output()?;
        if output.status.success() {
            return Ok(());
        }
        Err(format!(
            "git {:?} failed: {}",
            args,
            String::from_utf8_lossy(&output.stderr)
        )
        .into())
    }

    pub(crate) fn command(&self) -> Command {
        let mut command = Command::new(env!("CARGO_BIN_EXE_inferlab"));
        let mut path = OsString::from(&self.adapter_bin);
        path.push(":");
        path.push(std::env::var_os("PATH").unwrap_or_default());
        command
            .current_dir(self.root.path().join("vendor/vllm"))
            .env("PATH", path)
            .env("XDG_DATA_HOME", &self.data_home);
        command
    }

    pub(crate) fn run(&self, args: &[&str]) -> Result<Output, Box<dyn Error>> {
        Ok(self.command().args(args).output()?)
    }

    pub(crate) fn run_json(&self, args: &[&str]) -> Result<Value, Box<dyn Error>> {
        let output = self.run(args)?;
        if !output.status.success() {
            return Err(format!(
                "inferlab {:?} failed: {}",
                args,
                String::from_utf8_lossy(&output.stderr)
            )
            .into());
        }
        Ok(serde_json::from_slice(&output.stdout)?)
    }

    /// Replace the single-file workspace with `root_toml` at
    /// `.inferlab/workspace.toml` and one fragment per `(name, body)` under
    /// `.inferlab/workspace.d/`, then re-commit so the workspace stays clean.
    pub(crate) fn split_workspace(
        &self,
        root_toml: &str,
        fragments: &[(&str, &str)],
    ) -> Result<(), Box<dyn Error>> {
        let inferlab = self.root.path().join(".inferlab");
        fs::write(inferlab.join("workspace.toml"), root_toml)?;
        let fragment_dir = inferlab.join("workspace.d");
        fs::create_dir_all(&fragment_dir)?;
        for (name, body) in fragments {
            fs::write(fragment_dir.join(name), body)?;
        }
        Self::git(self.root.path(), &["add", "-A"])?;
        Self::git(self.root.path(), &["commit", "-qm", "split workspace"])?;
        Ok(())
    }
}

// The single-file fixture partitioned into a root file and two fragments; the
// disjoint union of these three files must reconstruct WORKSPACE exactly. The
// root keeps schema_version and the recipe; one fragment carries the serving
// definitions, the other the measurement definitions.
pub(crate) const SPLIT_ROOT: &str = "\
schema_version = 2

[recipes.dsv4-qualify]
server = \"dsv4-qualify\"
workload_suite = \"qualify\"
";

pub(crate) const SPLIT_SERVING: &str = "\
[models.deepseek-v4-flash]
served_name = \"deepseek-v4-flash\"

[stacks.vllm]
integration = \"vllm\"
pixi_environment = \"vllm\"
source_paths = [\"vendor/vllm\", \"vendor/flashinfer\"]

[servers.dsv4-qualify]
stack = \"vllm\"
model = \"deepseek-v4-flash\"
topology = \"single\"
readiness_timeout_seconds = 900
default_case = \"tp2\"

[servers.dsv4-qualify.parallelism.outer]
pipeline_parallel_size = 1

[servers.dsv4-qualify.settings]
max_model_len = 65536
kv_cache_dtype = \"fp8\"
gpu_memory_utilization = 0.95
trust_remote_code = true
compilation_config = { cudagraph_mode = \"FULL_AND_PIECEWISE\", custom_ops = [\"all\"] }
extra_args = [\"--max-num-seqs\", \"64\", \"--language-model-only\"]

[servers.dsv4-qualify.roles.serve.parallelism.attention]
context_parallel_size = 1

[servers.dsv4-qualify.roles.serve.settings]
block_size = 16

[servers.dsv4-qualify.cases.tp2.parallelism.outer]
tensor_parallel_size = 2

[servers.dsv4-qualify.cases.tp4.settings]
extra_args = [\"--max-num-seqs\", \"128\", \"--enable-prefix-caching\"]

[servers.dsv4-qualify.cases.tp4.parallelism.outer]
tensor_parallel_size = 4
";

pub(crate) const SPLIT_MEASUREMENTS: &str = "\
[evals.smoke]
kind = \"openai-smoke\"
prompt = \"San Francisco is a city in\"
max_tokens = 16
timeout_seconds = 60

[evals.gsm8k]
kind = \"lm-eval\"
task = \"gsm8k\"
prompt = { kind = \"flat\" }
limit = 64
metric = \"exact_match\"
metric_filter = \"strict-match\"
threshold = 0.90
timeout_seconds = 900

[benches.c8k1k]
kind = \"serving\"
request_source = { kind = \"random\", prompt = { kind = \"server_chat\" }, input_tokens = 8192, output_tokens = 1024 }
concurrency = [1, 4]
prompts_per_concurrency = 4
request_rates = [1.0, \"inf\"]
request_count = 32
burstiness = 1.0
cache = { start = \"cold\" }
timeout_seconds = 900

[benches.adaptive-c8k1k]
kind = \"adaptive-serving\"
request_source = { kind = \"random\", prompt = { kind = \"server_chat\" }, input_tokens = 8192, output_tokens = 1024 }
initial_request_rates = [1.0, 4.0]
aggregate_slos = [
    { metric = \"request_throughput\", at_least = 1.0 },
    { metric = \"p99_ttft_ms\", at_most = 1000.0 },
]
request_slo = { ttft_ms = 900.0, minimum_good_request_ratio = 0.99 }
max_search_steps = 3
min_rate_resolution = 0.25
request_count = 32
burstiness = 1.0
cache = { start = \"cold\" }
timeout_seconds = 900

[workload_suites.qualify]
evals = [\"smoke\", \"gsm8k\"]
gate = \"gsm8k\"
benches = [\"c8k1k\", \"adaptive-c8k1k\"]
";

pub(crate) const ORDINARY_DEFAULTED_MEASUREMENTS: &str = "\
[evals.smoke]
kind = \"openai-smoke\"

[benches.fixed-8k1k]
request_source = { kind = \"random\", input_tokens = 8192, output_tokens = 1024 }
concurrency = [1]
prompts_per_concurrency = 1
timeout_seconds = 900

[benches.range-8k1k]
request_source = { kind = \"random\", input_tokens = { min = 6553, max = 8192 }, output_tokens = { min = 819, max = 1024 } }
concurrency = [1]
prompts_per_concurrency = 1
timeout_seconds = 900

[workload_suites.qualify]
evals = [\"smoke\"]
benches = [\"fixed-8k1k\", \"range-8k1k\"]
";

pub(crate) fn write_executable(path: &Path, content: &str) -> Result<(), Box<dyn Error>> {
    fs::write(path, content)?;
    let mut permissions = fs::metadata(path)?.permissions();
    permissions.set_mode(0o755);
    fs::set_permissions(path, permissions)?;
    Ok(())
}

pub(crate) const PD_ADAPTER: &str = r#"#!/usr/bin/env python3
import json
import sys

request = json.load(sys.stdin)
input = request["input"]
operation = request["operation"]
framework = "vllm"

def effective(declared):
    outer = declared.get("outer") or {}
    tp = outer.get("tensor_parallel_size") or 1
    return {
        "outer": {"tensor_parallel_size": tp, "pipeline_parallel_size": 1},
        "attention": {"tensor_parallel_size": tp, "data_parallel_size": 1, "context_parallel_size": 1},
        "experts": {"tensor_parallel_size": tp, "data_parallel_size": 1, "expert_parallel_size": 1, "dense_tensor_parallel_size": 1},
    }

if operation == "plan_serve":
    roles = []
    replicas = []
    for role in input["roles"]:
        parallelism = effective(role["parallelism"])
        settings = dict(role["settings"])
        roles.append({
            "id": role["id"],
            "kind": role["kind"],
            "declared_replica_count": role["replica_count"],
            "effective_replica_count": role["replica_count"],
            "effective_settings": settings,
            "effective_parallelism": parallelism,
            "render_inputs": [],
        })
        tp = parallelism["outer"]["tensor_parallel_size"]
        ports = ["bootstrap"] if role["kind"] == "prefill" else []
        for replica_index in range(role["replica_count"]):
            replica_id = role["id"] if role["replica_count"] == 1 else f'{role["id"]}-{replica_index:03d}'
            replicas.append({
                "id": replica_id,
                "role_id": role["id"],
                "replica_index": replica_index,
                "device_count": tp,
                "ports": ports,
                "primary_ports": ["master"],
                "primary_readiness": {"kind": "http", "path": "/v1/models"},
                "worker_readiness": {"kind": "process_alive"},
            })
    implementation = {
        "vllm": "vllm_mooncake",
        "sglang": "sglang",
        "tensorrt-llm": "trtllm",
    }[framework]
    implementation_version = "2" if framework == "tensorrt-llm" else "1"
    co_rendering = {"process_role": "gateway"}
    readiness = {"kind": "http", "path": "/healthcheck"}
    output = {
        "integration": {
            "adapter_id": "fixture",
            "adapter_version": "1",
            "framework": framework,
            "framework_version": "test",
        },
        "roles": roles,
        "replicas": replicas,
        "links": [
            {"kind": "request_routing", "source": "gateway", "targets": ["pd_router"]},
            {"kind": "request_routing", "source": "pd_router", "targets": ["prefill", "decode"]},
            {"kind": "kv_transfer", "source": "prefill", "target": "decode", "mechanism": "mooncake"},
            {"kind": "bootstrap", "source": "pd_router", "target": "prefill", "port": "bootstrap"},
        ],
        "gateway": {
            "backend": input["gateway_backend"],
            "implementation": implementation,
            "implementation_version": implementation_version,
            "effective_settings": {},
            "endpoint": {
                "protocol": "http",
                "completions_path": "/v1/completions",
                "chat_completions_path": "/v1/chat/completions",
            },
            "readiness": readiness,
            "ports": [],
            "targets": [{"kind": "pd_router"}],
            "render_inputs": [],
            "render_source": "control_plane",
            "co_rendering": co_rendering,
        },
        "pd_router": {
            "backend": input["pd_router_backend"],
            "implementation": implementation,
            "implementation_version": implementation_version,
            "effective_settings": {},
            "policies": {"prefill": "round_robin", "decode": "round_robin"},
            "prefill_role": "prefill",
            "decode_role": "decode",
            "target_scheme": "http",
            "ports": [],
            "readiness": readiness,
            "handoff": "in_process",
            "render_inputs": [],
            "render_source": "control_plane",
            "co_rendering": co_rendering,
        },
    }
elif operation == "render_serve":
    output = {
        "integration": {
            "adapter_id": "fixture",
            "adapter_version": "1",
            "framework": framework,
            "framework_version": "test",
        },
        "processes": [
            {
                "kind": "model_rank",
                "process": allocation["process"],
                "role": allocation["role"],
                "replica": allocation["replica"],
                "rank": allocation["rank"],
                "rank_count": allocation["rank_count"],
                "launch_files": [],
                "command": {"argv": ["fixture-server", allocation["process"]], "env": {}},
            }
            for allocation in input["allocations"]
        ],
    }
else:
    raise ValueError(operation)

print(json.dumps({"status": "ok", "protocol_version": "9", "result": {"operation": operation, "output": output}}))
"#;

pub(crate) fn prefill_decode_workspace(integration: &str, transport: &str) -> String {
    WORKSPACE
        .replacen(
            "integration = \"vllm\"",
            &format!("integration = {integration:?}"),
            1,
        )
        .replacen(
            "topology = \"single\"",
            &format!(
                "topology = \"prefill_decode\"\n\
                 gateway_backend = \"builtin\"\n\
                 pd_router_backend = \"builtin\"\n\
                 kv_transfer = {transport:?}"
            ),
            1,
        )
        .replacen(
            "[servers.dsv4-qualify.roles.serve.parallelism.attention]\n\
             context_parallel_size = 1\n\n\
             [servers.dsv4-qualify.roles.serve.settings]\n\
             block_size = 16",
            "[servers.dsv4-qualify.roles.prefill.parallelism.attention]\n\
             context_parallel_size = 1\n\n\
             [servers.dsv4-qualify.roles.prefill.settings]\n\
             block_size = 16\n\n\
             [servers.dsv4-qualify.roles.decode.parallelism.attention]\n\
             context_parallel_size = 1\n\n\
             [servers.dsv4-qualify.roles.decode.settings]\n\
             block_size = 16",
            1,
        )
        .replace(
            "cache = { start = \"cold\" }",
            "cache = { start = \"uncontrolled\" }",
        )
}

const NETWORK_IP: &str = r#"#!/bin/sh
if [ "$1" = route ] && [ "$2" = get ]; then
  printf '8.8.8.8 dev enx-link-local src 169.254.3.1\n'
  exit 0
fi
if [ "$1" = -o ] && [ "$2" = -4 ] && [ "$3" = addr ]; then
  printf '1: enx-link-local inet 169.254.3.1/24\n'
  if [ "${FAKE_NETWORK_MODE:-default}" != link-local-only ]; then
    printf '2: ens-rdma inet 192.0.2.10/24\n'
  fi
  exit 0
fi
printf 'unexpected ip fixture arguments: %s\n' "$*" >&2
exit 2
"#;

const SSH: &str = r#"#!/bin/sh
while [ "$1" != -- ]; do shift; done
shift
shift
command="$3"
eval "exec bash -c $command"
"#;

const IBDEV2NETDEV: &str = r#"#!/bin/sh
if [ "${FAKE_NETWORK_MODE:-default}" != link-local-only ]; then
  printf 'mlx5_0 port 1 ==> ens-rdma (Up)\n'
fi
"#;
