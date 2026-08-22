#!/usr/bin/env python3
import json
import os
import sys

request = json.load(sys.stdin)
input = request["input"]
operation = request["operation"]
mechanism = input.get("profiling")


def cache_read_capability():
    # The fixture engine reports cache-read usage unless the test suppresses
    # the capability to exercise the planning-time rejection.
    if os.environ.get("FIXTURE_NO_CACHE_READ_REPORTING"):
        return {}
    return {"prompt_cache_read_zero_representation": "explicit"}


if operation == "plan_serve":
    role = input["roles"][0]
    settings = role["settings"]
    declared = role["parallelism"]
    outer = declared.get("outer") or {}
    attention = declared.get("attention") or {}
    tp = outer.get("tensor_parallel_size") or 1
    pp = outer.get("pipeline_parallel_size") or 1
    dp = attention.get("data_parallel_size") or 1
    effective = dict(settings)
    effective.setdefault("trust_remote_code", False)
    effective_parallelism = {
        "outer": {"tensor_parallel_size": tp, "pipeline_parallel_size": pp},
        "attention": {
            "tensor_parallel_size": tp,
            "data_parallel_size": dp,
            "context_parallel_size": 1,
        },
        "experts": {
            "tensor_parallel_size": tp * dp,
            "data_parallel_size": 1,
            "expert_parallel_size": 1,
            "dense_tensor_parallel_size": 1,
        },
    }
    transport = os.environ.get("FIXTURE_PD")
    roles = input["roles"] if transport else [role]
    replicas = []
    for selected_role in roles:
        ports = []
        if transport == "mooncake" and selected_role["kind"] == "prefill":
            ports = ["bootstrap"]
        elif transport == "nixl":
            ports = ["side_channel"]
        for replica_index in range(selected_role["replica_count"]):
            replica_id = (
                "server"
                if not transport
                else selected_role["id"]
                if selected_role["replica_count"] == 1
                else f"{selected_role['id']}-{replica_index:03d}"
            )
            replicas.append(
                {
                    "id": replica_id,
                    "role_id": selected_role["id"],
                    "replica_index": replica_index,
                    "device_count": tp * pp * dp,
                    "ports": ports,
                    "primary_ports": ["master"],
                    "primary_readiness": {"kind": "http", "path": "/v1/models"},
                    "worker_readiness": {"kind": "process_alive"},
                    **(
                        {
                            "capture_target": {
                                "mechanism": mechanism,
                                "window_control": {
                                    "endpoint": "replica_entry",
                                    "start": {
                                        "method": "post",
                                        "path": "/start_profile",
                                        **(
                                            {
                                                "body": {
                                                    "activities": [
                                                        "GPU"
                                                        if mechanism == "engine_trace"
                                                        else "CUDA_PROFILER"
                                                    ]
                                                }
                                            }
                                        ),
                                    },
                                    "stop": {
                                        "method": "post",
                                        "path": "/stop_profile",
                                    },
                                },
                            }
                        }
                        if mechanism
                        else {}
                    ),
                }
            )
    links = (
        []
        if not transport
        else [
            {"kind": "request_routing", "source": "gateway", "targets": ["pd_router"]},
            {"kind": "request_routing", "source": "pd_router", "targets": ["prefill", "decode"]},
            {
                "kind": "kv_transfer",
                "source": "prefill",
                "target": "decode",
                "mechanism": transport,
            },
        ]
    )
    if transport == "mooncake":
        links.append(
            {"kind": "bootstrap", "source": "pd_router", "target": "prefill", "port": "bootstrap"}
        )
    elif transport == "nixl":
        links.append(
            {
                "kind": "side_channel",
                "source": "prefill",
                "target": "decode",
                "port": "side_channel",
            }
        )
    output = {
        "integration": {
            "adapter_id": "fixture",
            "adapter_version": "1",
            "framework": "vllm",
            "framework_version": "test",
        },
        "roles": [
            {
                "id": selected_role["id"],
                "kind": selected_role["kind"],
                "declared_replica_count": selected_role["replica_count"],
                "effective_replica_count": selected_role["replica_count"],
                "effective_settings": effective,
                "effective_parallelism": effective_parallelism,
                **(
                    {
                        "public_endpoint": {
                            "protocol": "http",
                            "completions_path": "/v1/completions",
                            "chat_completions_path": "/v1/chat/completions",
                            "prefix_cache_reset": {"method": "post", "path": "/reset_prefix_cache"},
                            **cache_read_capability(),
                        }
                    }
                    if not transport
                    else {}
                ),
                "render_inputs": [],
            }
            for selected_role in roles
        ],
        "replicas": replicas,
        "links": links,
        **(
            {
                "gateway": {
                    "backend": input["gateway_backend"],
                    "implementation": "vllm_mooncake" if transport == "mooncake" else "vllm_nixl",
                    "implementation_version": "1",
                    "effective_settings": {},
                    "endpoint": {
                        "protocol": "http",
                        "completions_path": "/v1/completions",
                        "chat_completions_path": "/v1/chat/completions",
                        "prefix_cache_reset": {"method": "post", "path": "/reset_prefix_cache"},
                        **cache_read_capability(),
                        **(
                            {}
                            if os.environ.get("FIXTURE_GATEWAY_NO_CONDITIONING")
                            else {
                                "prefix_cache_conditioning": {
                                    "method": "post",
                                    "path": "/prime_prefix_cache",
                                }
                            }
                        ),
                    },
                    "readiness": {"kind": "http", "path": "/healthcheck"},
                    "ports": [],
                    "targets": [{"kind": "pd_router"}],
                    "render_inputs": [],
                    "render_source": "control_plane",
                    "co_rendering": {"process_role": "gateway"},
                },
                "pd_router": {
                    "backend": input["pd_router_backend"],
                    "implementation": "vllm_mooncake" if transport == "mooncake" else "vllm_nixl",
                    "implementation_version": "1",
                    "effective_settings": {},
                    "policies": {"prefill": "round_robin", "decode": "round_robin"},
                    "prefill_role": "prefill",
                    "decode_role": "decode",
                    "target_scheme": "http",
                    "ports": [],
                    "readiness": {"kind": "http", "path": "/healthcheck"},
                    "handoff": "in_process",
                    "render_inputs": [],
                    "render_source": "control_plane",
                    "co_rendering": {"process_role": "gateway"},
                },
            }
            if transport
            else {}
        ),
    }
elif operation == "render_serve":
    server = (
        "fixture-missing-server"
        if os.environ.get("FIXTURE_SERVER_START_FAIL") == "1"
        else "fixture-server"
    )
    allocations = input["allocations"]
    output = {
        "integration": {
            "adapter_id": "fixture",
            "adapter_version": "1",
            "framework": "vllm",
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
                "command": {
                    "argv": [
                        server,
                        allocation["endpoint"]["host"],
                        str(allocation["endpoint"]["port"]),
                        *(
                            [str(allocation["ports"]["bootstrap"]["port"])]
                            if "bootstrap" in allocation["ports"]
                            else []
                        ),
                    ],
                    "env": (
                        {"FIXTURE_CAPTURE_STORAGE": allocation["capture_storage"]}
                        if mechanism == "engine_trace" and allocation.get("capture_storage")
                        else {}
                    ),
                },
            }
            for allocation in allocations
        ],
    }
else:
    raise ValueError(operation)
print(
    json.dumps(
        {
            "status": "ok",
            "protocol_version": "8",
            "result": {"operation": operation, "output": output},
        }
    )
)
