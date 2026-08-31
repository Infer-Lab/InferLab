import importlib.metadata
import json
import sys
from pathlib import Path
from typing import cast

import pytest
from inferlab_adapter_sdk import (
    AdapterOperationError,
    AdapterRequest,
    AdapterRequestPlanServe,
    AdapterRequestRenderServe,
    AdapterResponse,
    PlanServeResult,
    PromptCacheReadZeroRepresentation,
    RenderedServeProcessFrontend,
    SettingValue,
    handle_request,
)
from inferlab_adapter_sdk._generated import AdapterResultPlanServe
from inferlab_integration_vllm import plan_serve, render_serve

ROOT = Path(__file__).parents[3]
FIXTURES = ROOT / "protocol" / "fixtures"


def load_json(path: Path) -> dict[str, object]:
    return cast(dict[str, object], json.loads(path.read_text()))


def load_plan_payload() -> dict[str, object]:
    payload = load_json(FIXTURES / "valid" / "plan-serve-request.json")
    # The shared fixture declares synthetic acceptance; tests that exercise
    # unrelated behavior drop the member instead of spelling an operator
    # speculative configuration into every payload.
    cast(dict[str, object], payload["input"]).pop("synthetic_acceptance")
    return payload


def distribution_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def test_plan_serve_matches_the_shared_vllm_fixture() -> None:
    request = AdapterRequest.model_validate(load_plan_payload())
    expected = AdapterResponse.model_validate(
        load_json(FIXTURES / "valid" / "plan-serve-response.json")
    )

    assert isinstance(request.root, AdapterRequestPlanServe)
    result = plan_serve(request.root.input)

    assert expected.root.status == "ok"
    assert isinstance(expected.root.result.root, AdapterResultPlanServe)
    # The shared fixture pair declares the curve form; this test drops the
    # member (see load_plan_payload), so the expected outcome goes with it.
    expected_output = expected.root.result.root.output.model_copy(
        update={"synthetic_acceptance": None}
    )
    assert result.gateway is not None
    assert result.pd_router is not None
    assert expected_output.gateway is not None
    assert expected_output.pd_router is not None
    normalized = result.model_copy(
        update={
            "integration": expected_output.integration,
            "gateway": result.gateway.model_copy(
                update={"implementation_version": expected_output.gateway.implementation_version}
            ),
            "pd_router": result.pd_router.model_copy(
                update={"implementation_version": expected_output.pd_router.implementation_version}
            ),
        }
    )
    assert normalized == expected_output
    package_version = distribution_version("inferlab-integration-vllm")
    assert result.integration.adapter_version == package_version
    assert result.gateway.implementation_version == package_version
    assert result.pd_router.implementation_version == package_version
    assert result.integration.framework_version == "unavailable"
    assert result.gateway.endpoint.completions_path == "/v1/completions"
    assert result.gateway.endpoint.chat_completions_path == "/v1/chat/completions"
    assert result.gateway.endpoint.server_metrics is None
    assert result.gateway.endpoint.prefix_cache_reset is None
    assert result.gateway.endpoint.prefix_cache_conditioning is None
    assert result.gateway.backend == "vllm-router"
    assert result.pd_router is not None
    assert result.pd_router.backend == "vllm-router"
    assert "vllm" not in sys.modules


def test_unknown_vllm_setting_returns_a_typed_protocol_error() -> None:
    payload = load_plan_payload()
    input_payload = cast(dict[str, object], payload["input"])
    roles = cast(list[dict[str, object]], input_payload["roles"])
    settings = cast(dict[str, object], roles[0]["settings"])
    settings["not_a_vllm_setting"] = True

    response = handle_request(json.dumps(payload), plan_serve)

    assert response.root.status == "error"
    assert response.root.error.code == "invalid_settings"


def test_single_topology_rejects_a_routed_backend() -> None:
    payload = load_plan_payload()
    input_payload = cast(dict[str, object], payload["input"])
    input_payload["topology"] = "single"
    input_payload["gateway_backend"] = "vllm-router"
    input_payload["pd_router_backend"] = None
    input_payload["kv_transfer"] = None
    input_payload["roles"] = [
        {
            "id": "serve",
            "kind": "serve",
            "replica_count": 1,
            "parallelism": {"outer": {"tensor_parallel_size": 2}},
            "settings": {},
        }
    ]

    response = handle_request(json.dumps(payload), plan_serve)

    assert response.root.status == "error"
    assert response.root.error.code == "invalid_settings"


def test_single_topology_declares_its_server_metrics_capability() -> None:
    payload = load_plan_payload()
    input_payload = cast(dict[str, object], payload["input"])
    input_payload["topology"] = "single"
    input_payload["gateway_backend"] = None
    input_payload["pd_router_backend"] = None
    input_payload["kv_transfer"] = None
    input_payload["roles"] = [
        {
            "id": "serve",
            "kind": "serve",
            "replica_count": 1,
            "parallelism": {"outer": {"tensor_parallel_size": 2}},
            "settings": {},
        }
    ]
    request = AdapterRequest.model_validate(payload)

    assert isinstance(request.root, AdapterRequestPlanServe)
    result = plan_serve(request.root.input)
    endpoint = result.roles[0].public_endpoint
    assert endpoint is not None
    assert endpoint.server_metrics is not None
    assert endpoint.server_metrics.path == "/metrics"
    assert endpoint.server_metrics.port is None


def test_single_topology_declares_explicit_zero_cache_usage_when_enabled() -> None:
    payload = load_plan_payload()
    input_payload = cast(dict[str, object], payload["input"])
    input_payload["topology"] = "single"
    input_payload["gateway_backend"] = None
    input_payload["pd_router_backend"] = None
    input_payload["kv_transfer"] = None
    input_payload["roles"] = [
        {
            "id": "serve",
            "kind": "serve",
            "replica_count": 1,
            "parallelism": {"outer": {"tensor_parallel_size": 1}},
            "settings": {"enable_prompt_tokens_details": True},
        }
    ]
    request = AdapterRequest.model_validate(payload)

    assert isinstance(request.root, AdapterRequestPlanServe)
    result = plan_serve(request.root.input)
    endpoint = result.roles[0].public_endpoint
    assert endpoint is not None
    assert (
        endpoint.prompt_cache_read_zero_representation is PromptCacheReadZeroRepresentation.explicit
    )


def test_vllm_rejects_an_expert_size_that_does_not_match_tp_times_dp() -> None:
    payload = load_plan_payload()
    input_payload = cast(dict[str, object], payload["input"])
    roles = cast(list[dict[str, object]], input_payload["roles"])
    prefill_parallelism = cast(dict[str, object], roles[0]["parallelism"])
    prefill_parallelism["experts"] = {"expert_parallel_size": 3}

    response = handle_request(json.dumps(payload), plan_serve)

    assert response.root.status == "error"
    assert response.root.error.code == "invalid_settings"


def test_plan_rejects_inferlab_owned_option_in_extra_args() -> None:
    payload = load_plan_payload()
    input_payload = cast(dict[str, object], payload["input"])
    roles = cast(list[dict[str, object]], input_payload["roles"])
    cast(dict[str, object], roles[0]["settings"])["extra_args"] = ["--block-size", "32"]

    response = handle_request(json.dumps(payload), plan_serve)

    assert response.root.status == "error"
    assert response.root.error.code == "invalid_settings"
    assert "--block-size" in response.root.error.message


def test_render_rejects_inferlab_owned_option_in_extra_args() -> None:
    payload = load_json(FIXTURES / "valid" / "render-serve-request.json")
    input_payload = cast(dict[str, object], payload["input"])
    allocations = cast(list[dict[str, object]], input_payload["allocations"])
    for allocation in allocations[:2]:
        settings = cast(dict[str, object], allocation["effective_settings"])
        settings["extra_args"] = ["--tensor-parallel-size", "99"]

    request = AdapterRequest.model_validate(payload)
    assert isinstance(request.root, AdapterRequestRenderServe)
    with pytest.raises(AdapterOperationError, match="--tensor-parallel-size"):
        render_serve(request.root.input)


def test_render_passes_through_unrecognized_and_passthrough_extra_args() -> None:
    payload = load_json(FIXTURES / "valid" / "render-serve-request.json")
    input_payload = cast(dict[str, object], payload["input"])
    allocations = cast(list[dict[str, object]], input_payload["allocations"])
    for allocation in allocations[:2]:
        settings = cast(dict[str, object], allocation["effective_settings"])
        settings["extra_args"] = [
            "--max-num-seqs",
            "16",
            "--max-num-seqs",
            "32",
            "--",
            "--block-size",
            "32",
        ]

    request = AdapterRequest.model_validate(payload)
    assert isinstance(request.root, AdapterRequestRenderServe)
    result = render_serve(request.root.input)

    rank_zero = result.processes[0].root.command.argv
    assert [
        rank_zero[index + 1] for index, value in enumerate(rank_zero) if value == "--max-num-seqs"
    ] == ["16", "32"]
    assert "--" not in rank_zero, "the composition sentinel never reaches the engine argv"
    assert rank_zero[-2:] == ["--block-size", "32"]


def test_render_enables_prompt_token_details_when_declared() -> None:
    payload = load_json(FIXTURES / "valid" / "render-serve-request.json")
    input_payload = cast(dict[str, object], payload["input"])
    allocations = cast(list[dict[str, object]], input_payload["allocations"])
    settings = cast(dict[str, object], allocations[0]["effective_settings"])
    settings["enable_prompt_tokens_details"] = True

    request = AdapterRequest.model_validate(payload)
    assert isinstance(request.root, AdapterRequestRenderServe)
    result = render_serve(request.root.input)

    assert "--enable-prompt-tokens-details" in result.processes[0].root.command.argv
    assert "--enable-prompt-tokens-details" not in result.processes[1].root.command.argv


def test_render_serve_matches_the_shared_vllm_fixture() -> None:
    request = AdapterRequest.model_validate(
        load_json(FIXTURES / "valid" / "render-serve-request.json")
    )
    assert isinstance(request.root, AdapterRequestRenderServe)
    result = render_serve(request.root.input)

    assert result.integration.framework_version == "unavailable"
    assert len(result.processes) == 3
    assert result.processes[0].root.command.env["VLLM_SERVER_DEV_MODE"] == "1"
    frontend = result.processes[-1].root
    assert isinstance(frontend, RenderedServeProcessFrontend)
    assert frontend.components.model_dump() == (
        "gateway",
        "pd_router",
    )


def test_render_engine_trace_injects_the_assigned_torch_profiler_dir() -> None:
    request = AdapterRequest.model_validate(
        load_json(FIXTURES / "valid" / "render-serve-request.json")
    )
    assert isinstance(request.root, AdapterRequestRenderServe)
    result = render_serve(request.root.input)

    argv = result.processes[0].root.command.argv
    config = argv[argv.index("--profiler-config") + 1]
    assert json.loads(config) == {
        "profiler": "torch",
        "torch_profiler_dir": "/workspace/.inferlab/runtime/engine-trace/serve-fixture/prefill",
    }


def test_render_engine_trace_requires_the_assigned_trace_directory() -> None:
    payload = load_json(FIXTURES / "valid" / "render-serve-request.json")
    input_payload = cast(dict[str, object], payload["input"])
    for allocation in cast(list[dict[str, object]], input_payload["allocations"]):
        allocation.pop("capture_storage", None)
    request = AdapterRequest.model_validate(payload)
    assert isinstance(request.root, AdapterRequestRenderServe)

    with pytest.raises(AdapterOperationError, match="trace directory"):
        render_serve(request.root.input)


def test_render_managed_collection_uses_the_cuda_profiler() -> None:
    payload = load_json(FIXTURES / "valid" / "render-serve-request.json")
    cast(dict[str, object], payload["input"])["profiling"] = "managed_collection"
    request = AdapterRequest.model_validate(payload)
    assert isinstance(request.root, AdapterRequestRenderServe)
    result = render_serve(request.root.input)

    argv = result.processes[0].root.command.argv
    assert argv[argv.index("--profiler-config") + 1] == '{"profiler":"cuda"}'


def test_plan_engine_trace_echoes_the_requested_mechanism() -> None:
    payload = load_plan_payload()
    cast(dict[str, object], payload["input"])["profiling"] = "engine_trace"
    request = AdapterRequest.model_validate(payload)
    assert isinstance(request.root, AdapterRequestPlanServe)
    result = plan_serve(request.root.input)

    target = result.replicas[0].capture_target
    assert target is not None
    assert target.mechanism == "engine_trace"


def test_render_serve_allows_an_explicit_cache_environment_override() -> None:
    payload = load_json(FIXTURES / "valid" / "render-serve-request.json")
    input_payload = cast(dict[str, object], payload["input"])
    allocations = cast(list[dict[str, object]], input_payload["allocations"])
    settings = cast(dict[str, object], allocations[0]["effective_settings"])
    settings["extra_env"] = {"FLASHINFER_WORKSPACE_BASE": "/custom/flashinfer"}

    request = AdapterRequest.model_validate(payload)
    assert isinstance(request.root, AdapterRequestRenderServe)
    result = render_serve(request.root.input)

    process = result.processes[0].root
    assert process.command.env["FLASHINFER_WORKSPACE_BASE"] == "/custom/flashinfer"
    assert process.command.env["TRITON_CACHE_DIR"].endswith("/triton")


def test_render_lowers_published_vllm_settings() -> None:
    payload = load_json(FIXTURES / "valid" / "render-serve-request.json")
    input_payload = cast(dict[str, object], payload["input"])
    allocations = cast(list[dict[str, object]], input_payload["allocations"])
    for allocation in allocations[:2]:
        settings = cast(dict[str, object], allocation["effective_settings"])
        settings.update(
            {
                "tokenizer_mode": "deepseek_v4",
                "tool_call_parser": "deepseek_v4",
                "reasoning_parser": "deepseek_v4",
                "enable_auto_tool_choice": True,
                "reasoning_config": {
                    "reasoning_parser": "deepseek_v4",
                    "reasoning_start_str": "<think>",
                    "reasoning_end_str": "</think>",
                },
                "enable_flashinfer_autotune": False,
            }
        )

    request = AdapterRequest.model_validate(payload)
    assert isinstance(request.root, AdapterRequestRenderServe)
    argv = render_serve(request.root.input).processes[0].root.command.argv

    assert argv[argv.index("--tokenizer-mode") + 1] == "deepseek_v4"
    assert argv[argv.index("--tool-call-parser") + 1] == "deepseek_v4"
    assert argv[argv.index("--reasoning-parser") + 1] == "deepseek_v4"
    assert "--enable-auto-tool-choice" in argv
    assert json.loads(argv[argv.index("--reasoning-config") + 1]) == {
        "reasoning_parser": "deepseek_v4",
        "reasoning_start_str": "<think>",
        "reasoning_end_str": "</think>",
    }
    assert "--no-enable-flashinfer-autotune" in argv


def test_plan_nixl_declares_side_channel_links_and_ports() -> None:
    payload = load_plan_payload()
    input_payload = cast(dict[str, object], payload["input"])
    input_payload["kv_transfer"] = "nixl"

    request = AdapterRequest.model_validate(payload)
    assert isinstance(request.root, AdapterRequestPlanServe)
    result = plan_serve(request.root.input)

    assert [replica.ports for replica in result.replicas] == [
        ["side_channel"],
        ["side_channel"],
    ]
    assert result.links[-1].root.kind == "side_channel"


def test_plan_role_declares_the_whole_replica_accelerator_requirement() -> None:
    payload = load_plan_payload()
    input_payload = cast(dict[str, object], payload["input"])
    roles = cast(list[dict[str, object]], input_payload["roles"])
    parallelism = cast(dict[str, object], roles[0]["parallelism"])
    parallelism["outer"] = {"tensor_parallel_size": 4}

    request = AdapterRequest.model_validate(payload)
    assert isinstance(request.root, AdapterRequestPlanServe)
    result = plan_serve(request.root.input)
    prefill = [replica for replica in result.replicas if replica.role_id == "prefill"]

    assert len(prefill) == 1
    assert prefill[0].device_count == 4
    assert prefill[0].ports == ["bootstrap"]
    assert prefill[0].primary_ports == ["master"]
    assert prefill[0].capture_target is not None
    assert prefill[0].capture_target.model_dump(mode="json") == {
        "mechanism": "managed_collection",
        "window_control": {
            "endpoint": "replica_entry",
            "start": {"method": "post", "path": "/start_profile", "body": None},
            "stop": {"method": "post", "path": "/stop_profile", "body": None},
        },
    }


def test_plan_static_npmd_keeps_replicas_distinct_from_ranks() -> None:
    payload = load_plan_payload()
    input_payload = cast(dict[str, object], payload["input"])
    roles = cast(list[dict[str, object]], input_payload["roles"])
    roles[0]["replica_count"] = 2
    roles[1]["replica_count"] = 3

    request = AdapterRequest.model_validate(payload)
    assert isinstance(request.root, AdapterRequestPlanServe)
    result = plan_serve(request.root.input)

    assert [role.effective_replica_count for role in result.roles] == [2, 3]
    assert [replica.id for replica in result.replicas] == [
        "prefill-000",
        "prefill-001",
        "decode-000",
        "decode-001",
        "decode-002",
    ]
    assert [replica.replica_index for replica in result.replicas] == [0, 1, 0, 1, 2]
    assert all(replica.capture_target is not None for replica in result.replicas)


def test_render_nixl_uses_role_side_channels_and_connector() -> None:
    payload = load_json(FIXTURES / "valid" / "render-serve-request.json")
    input_payload = cast(dict[str, object], payload["input"])
    input_payload["kv_transfer"] = "nixl"
    allocations = cast(list[dict[str, object]], input_payload["allocations"])
    for index, allocation in enumerate(allocations[:2]):
        allocation["ports"] = {"side_channel": {"host": "127.0.0.1", "port": 9000 + index}}

    request = AdapterRequest.model_validate(payload)
    assert isinstance(request.root, AdapterRequestRenderServe)
    result = render_serve(request.root.input)

    for index, wrapped in enumerate(result.processes[:2]):
        process = wrapped.root
        config = process.command.argv[process.command.argv.index("--kv-transfer-config") + 1]
        assert '"kv_connector":"NixlConnector"' in config
        assert process.command.env["VLLM_NIXL_SIDE_CHANNEL_PORT"] == str(9000 + index)


def test_plan_vllm_router_makes_the_external_router_public() -> None:
    payload = load_plan_payload()
    input_payload = cast(dict[str, object], payload["input"])
    input_payload["gateway_backend"] = "vllm-router"
    input_payload["pd_router_backend"] = "vllm-router"

    request = AdapterRequest.model_validate(payload)
    assert isinstance(request.root, AdapterRequestPlanServe)
    result = plan_serve(request.root.input)

    assert all(replica.role_id != "gateway" for replica in result.replicas)
    assert result.gateway is not None
    assert result.gateway.render_source == "integration"
    assert result.pd_router is not None
    assert result.pd_router.render_source == "integration"


def test_builtin_pd_frontends_declare_prefix_cache_reset() -> None:
    payload = load_plan_payload()
    input_payload = cast(dict[str, object], payload["input"])
    input_payload["gateway_backend"] = "builtin"
    input_payload["pd_router_backend"] = "builtin"

    request = AdapterRequest.model_validate(payload)
    assert isinstance(request.root, AdapterRequestPlanServe)
    result = plan_serve(request.root.input)

    assert result.gateway is not None
    assert result.gateway.endpoint.prefix_cache_reset is not None
    assert result.gateway.endpoint.prefix_cache_reset.path == "/reset_prefix_cache"
    assert result.gateway.endpoint.prefix_cache_conditioning is not None
    assert result.gateway.endpoint.prefix_cache_conditioning.path == "/prime_prefix_cache"

    input_payload["gateway_backend"] = "vllm-router"
    input_payload["pd_router_backend"] = "vllm-router"
    request = AdapterRequest.model_validate(payload)
    assert isinstance(request.root, AdapterRequestPlanServe)
    result = plan_serve(request.root.input)

    assert result.gateway is not None
    assert result.gateway.endpoint.prefix_cache_reset is None
    assert result.gateway.endpoint.prefix_cache_conditioning is None


def test_builtin_pd_frontend_declares_cache_read_when_both_roles_report_details() -> None:
    payload = load_plan_payload()
    input_payload = cast(dict[str, object], payload["input"])
    input_payload["gateway_backend"] = "builtin"
    input_payload["pd_router_backend"] = "builtin"
    roles = cast(list[dict[str, object]], input_payload["roles"])
    for role in roles:
        cast(dict[str, object], role["settings"])["enable_prompt_tokens_details"] = True

    request = AdapterRequest.model_validate(payload)
    assert isinstance(request.root, AdapterRequestPlanServe)
    result = plan_serve(request.root.input)

    assert result.gateway is not None
    assert (
        result.gateway.endpoint.prompt_cache_read_zero_representation
        is PromptCacheReadZeroRepresentation.explicit
    )

    # Only one role reporting is not enough: the gateway must not claim a
    # capability a role's responses cannot carry.
    cast(dict[str, object], roles[1]["settings"]).pop("enable_prompt_tokens_details")
    request = AdapterRequest.model_validate(payload)
    assert isinstance(request.root, AdapterRequestPlanServe)
    result = plan_serve(request.root.input)

    assert result.gateway is not None
    assert result.gateway.endpoint.prompt_cache_read_zero_representation is None


def test_vllm_router_targets_replica_entrypoints_and_defers_startup_timeout() -> None:
    payload = load_json(FIXTURES / "valid" / "render-serve-request.json")
    input_payload = cast(dict[str, object], payload["input"])
    input_payload["gateway_backend"] = "vllm-router"
    input_payload["pd_router_backend"] = "vllm-router"
    allocations = cast(list[dict[str, object]], input_payload["allocations"])
    prefill = allocations[0]
    prefill.update(
        {
            "process": "prefill-000-rank-000",
            "replica": 0,
            "rank": 0,
            "rank_count": 2,
            "ports": {
                "bootstrap": {"host": "node-a.example", "port": 29501},
                "master": {"host": "node-a.example", "port": 29502},
            },
        }
    )
    prefill_rank = json.loads(json.dumps(prefill))
    prefill_rank.update(
        {
            "process": "prefill-000-rank-001",
            "machine": "node-b",
            "rank": 1,
            "rank_count": 2,
            "endpoint": {"host": "node-b.example", "port": 8000},
            "ports": {"bootstrap": {"host": "node-b.example", "port": 29501}},
        }
    )
    prefill_replica = json.loads(json.dumps(prefill))
    prefill_replica.update(
        {
            "process": "prefill-001",
            "replica": 1,
            "rank_count": 1,
            "machine": "node-c",
            "cache": "/cache/runtime/node-c/prefill-001",
            "endpoint": {"host": "node-c.example", "port": 8000},
            "ports": {"bootstrap": {"host": "node-c.example", "port": 29501}},
        }
    )
    decode = allocations[1]
    decode.update({"process": "decode-000", "replica": 0})
    decode_replica = json.loads(json.dumps(decode))
    decode_replica.update(
        {
            "process": "decode-001",
            "replica": 1,
            "machine": "node-d",
            "cache": "/cache/runtime/node-d/decode-001",
            "endpoint": {"host": "node-d.example", "port": 8000},
        }
    )
    gateway = allocations[2]
    gateway.update(
        {
            "machine": "local",
            "cache": "/cache/runtime/local/gateway",
            "endpoint": {"host": "127.0.0.1", "port": 8000},
        }
    )
    input_payload["allocations"] = [
        prefill_rank,
        prefill,
        prefill_replica,
        decode,
        decode_replica,
        gateway,
    ]

    request = AdapterRequest.model_validate(payload)
    assert isinstance(request.root, AdapterRequestRenderServe)
    result = render_serve(request.root.input)
    argv = result.processes[-1].root.command.argv
    rank_one = next(
        process.root.command.argv
        for process in result.processes
        if process.root.process == "prefill-000-rank-001"
    )

    assert rank_one[rank_one.index("--master-addr") + 1] == "node-a.example"
    first_prefill = argv.index("http://node-a.example:8000")
    assert argv[first_prefill + 1] == "29501"
    assert [argv[index + 1] for index, arg in enumerate(argv) if arg == "--prefill"] == [
        "http://node-a.example:8000",
        "http://node-c.example:8000",
    ]
    assert [argv[index + 1] for index, arg in enumerate(argv) if arg == "--decode"] == [
        "http://node-b.example:8000",
        "http://node-d.example:8000",
    ]
    assert argv[argv.index("--worker-startup-timeout-secs") + 1] == "2147483647"


def single_plan_payload(parallelism: dict[str, object]) -> dict[str, object]:
    payload = load_plan_payload()
    input_payload = cast(dict[str, object], payload["input"])
    input_payload["topology"] = "single"
    input_payload["gateway_backend"] = None
    input_payload["pd_router_backend"] = None
    input_payload["kv_transfer"] = None
    input_payload["roles"] = [
        {
            "id": "serve",
            "kind": "serve",
            "replica_count": 1,
            "parallelism": parallelism,
            "settings": {},
        }
    ]
    return payload


def test_single_topology_lowers_context_parallelism_to_decode_cp() -> None:
    payload = single_plan_payload(
        {
            "outer": {"tensor_parallel_size": 2},
            "attention": {"context_parallel_size": 2},
        }
    )
    request = AdapterRequest.model_validate(payload)

    assert isinstance(request.root, AdapterRequestPlanServe)
    result = plan_serve(request.root.input)

    attention = result.roles[0].effective_parallelism.attention
    assert attention is not None
    assert attention.tensor_parallel_size == 2
    assert attention.context_parallel_size == 2
    assert result.replicas[0].device_count == 2

    render_payload = load_json(FIXTURES / "valid" / "render-serve-request.json")
    render_input = cast(dict[str, object], render_payload["input"])
    render_input["topology"] = "single"
    render_input["gateway_backend"] = None
    render_input["pd_router_backend"] = None
    render_input["kv_transfer"] = None
    serve_allocation = cast(dict[str, object], cast(list[object], render_input["allocations"])[0])
    serve_allocation.update(
        {"process": "serve", "role": "serve", "role_kind": "serve", "links": []}
    )
    serve_parallelism = cast(dict[str, object], serve_allocation["effective_parallelism"])
    cast(dict[str, object], serve_parallelism["attention"])["context_parallel_size"] = 2
    render_input["allocations"] = [serve_allocation]

    render_request = AdapterRequest.model_validate(render_payload)
    assert isinstance(render_request.root, AdapterRequestRenderServe)
    argv = render_serve(render_request.root.input).processes[0].root.command.argv

    assert argv[argv.index("--decode-context-parallel-size") + 1] == "2"
    assert "--prefill-context-parallel-size" not in argv


def test_decode_context_parallel_size_must_divide_tensor_parallel_size() -> None:
    payload = single_plan_payload(
        {
            "outer": {"tensor_parallel_size": 2},
            "attention": {"context_parallel_size": 3},
        }
    )

    response = handle_request(json.dumps(payload), plan_serve)

    assert response.root.status == "error"
    assert response.root.error.code == "invalid_settings"


def test_prefill_role_lowers_context_parallelism_to_prefill_cp() -> None:
    payload = load_plan_payload()
    input_payload = cast(dict[str, object], payload["input"])
    roles = cast(list[dict[str, object]], input_payload["roles"])
    roles[0]["parallelism"] = {
        "outer": {"tensor_parallel_size": 2},
        "attention": {"context_parallel_size": 2},
    }

    request = AdapterRequest.model_validate(payload)
    assert isinstance(request.root, AdapterRequestPlanServe)
    result = plan_serve(request.root.input)

    attention = result.roles[0].effective_parallelism.attention
    assert attention is not None
    assert attention.tensor_parallel_size == 2
    assert attention.context_parallel_size == 2
    device_counts = {replica.role_id: replica.device_count for replica in result.replicas}
    assert device_counts == {"prefill": 4, "decode": 2}

    render_payload = load_json(FIXTURES / "valid" / "render-serve-request.json")
    render_input = cast(dict[str, object], render_payload["input"])
    allocations = cast(list[dict[str, object]], render_input["allocations"])
    prefill_parallelism = cast(dict[str, object], allocations[0]["effective_parallelism"])
    cast(dict[str, object], prefill_parallelism["attention"])["context_parallel_size"] = 2

    render_request = AdapterRequest.model_validate(render_payload)
    assert isinstance(render_request.root, AdapterRequestRenderServe)
    rendered = render_serve(render_request.root.input)
    prefill_argv = rendered.processes[0].root.command.argv
    decode_argv = rendered.processes[1].root.command.argv

    assert prefill_argv[prefill_argv.index("--prefill-context-parallel-size") + 1] == "2"
    assert "--decode-context-parallel-size" not in prefill_argv
    assert "--prefill-context-parallel-size" not in decode_argv
    assert "--decode-context-parallel-size" not in decode_argv


def test_prefill_context_parallelism_excludes_attention_data_parallelism() -> None:
    payload = load_plan_payload()
    input_payload = cast(dict[str, object], payload["input"])
    roles = cast(list[dict[str, object]], input_payload["roles"])
    roles[0]["parallelism"] = {
        "outer": {"tensor_parallel_size": 2},
        "attention": {"context_parallel_size": 2, "data_parallel_size": 2},
    }

    response = handle_request(json.dumps(payload), plan_serve)

    assert response.root.status == "error"
    assert response.root.error.code == "invalid_settings"
    assert "declared attention.context_parallel_size=2" in response.root.error.message
    assert "attention.data_parallel_size=2" in response.root.error.message


def test_rejects_a_declared_expert_tensor_parallel_size_that_misses_the_derived_value() -> None:
    payload = load_plan_payload()
    input_payload = cast(dict[str, object], payload["input"])
    roles = cast(list[dict[str, object]], input_payload["roles"])
    roles[0]["parallelism"] = {
        "outer": {"tensor_parallel_size": 2},
        "experts": {"tensor_parallel_size": 3},
    }

    response = handle_request(json.dumps(payload), plan_serve)

    assert response.root.status == "error"
    assert response.root.error.code == "invalid_settings"
    assert (
        "outer.tensor_parallel_size * attention.context_parallel_size * "
        "attention.data_parallel_size (2)" in response.root.error.message
    )
    assert "declared 3" in response.root.error.message


def test_prefill_context_parallelism_enters_the_expert_world() -> None:
    payload = load_plan_payload()
    input_payload = cast(dict[str, object], payload["input"])
    roles = cast(list[dict[str, object]], input_payload["roles"])
    roles[0]["parallelism"] = {
        "outer": {"tensor_parallel_size": 2},
        "attention": {"context_parallel_size": 2},
        "experts": {"expert_parallel_size": 4},
    }

    request = AdapterRequest.model_validate(payload)
    assert isinstance(request.root, AdapterRequestPlanServe)
    result = plan_serve(request.root.input)

    experts = result.roles[0].effective_parallelism.experts
    assert experts is not None
    assert experts.expert_parallel_size == 4

    roles[1]["parallelism"] = {
        "outer": {"tensor_parallel_size": 2},
        "attention": {"context_parallel_size": 2},
        "experts": {"expert_parallel_size": 2},
    }
    request = AdapterRequest.model_validate(payload)
    assert isinstance(request.root, AdapterRequestPlanServe)
    result = plan_serve(request.root.input)

    decode_experts = result.roles[1].effective_parallelism.experts
    assert decode_experts is not None
    assert decode_experts.expert_parallel_size == 2


def test_prefill_context_parallelism_rejects_an_expert_size_without_cp() -> None:
    payload = load_plan_payload()
    input_payload = cast(dict[str, object], payload["input"])
    roles = cast(list[dict[str, object]], input_payload["roles"])
    roles[0]["parallelism"] = {
        "outer": {"tensor_parallel_size": 2},
        "attention": {"context_parallel_size": 2},
        "experts": {"expert_parallel_size": 2},
    }

    response = handle_request(json.dumps(payload), plan_serve)

    assert response.root.status == "error"
    assert response.root.error.code == "invalid_settings"


def test_extra_args_rejects_inferlab_owned_context_parallel_options() -> None:
    payload = load_plan_payload()
    input_payload = cast(dict[str, object], payload["input"])
    roles = cast(list[dict[str, object]], input_payload["roles"])
    cast(dict[str, object], roles[0]["settings"])["extra_args"] = [
        "--decode-context-parallel-size",
        "2",
    ]

    response = handle_request(json.dumps(payload), plan_serve)

    assert response.root.status == "error"
    assert response.root.error.code == "invalid_settings"
    assert "--decode-context-parallel-size" in response.root.error.message


def synthetic_plan_payload(
    speculative_config: object,
    synthetic_acceptance: object = None,
) -> dict[str, object]:
    payload = load_json(FIXTURES / "valid" / "plan-serve-request.json")
    input_payload = cast(dict[str, object], payload["input"])
    if synthetic_acceptance is not None:
        input_payload["synthetic_acceptance"] = synthetic_acceptance
    roles = cast(list[dict[str, object]], input_payload["roles"])
    for role in roles:
        settings = cast(dict[str, object], role["settings"])
        if speculative_config is not None:
            settings["extra_args"] = [
                "--speculative-config",
                speculative_config
                if isinstance(speculative_config, str)
                else json.dumps(speculative_config),
            ]
    return payload


def synthetic_error(payload: dict[str, object]) -> str:
    response = handle_request(json.dumps(payload), plan_serve)
    assert response.root.status == "error"
    assert response.root.error.code == "invalid_settings"
    return response.root.error.message


def plan_synthetic(payload: dict[str, object]) -> PlanServeResult:
    request = AdapterRequest.model_validate(payload)
    assert isinstance(request.root, AdapterRequestPlanServe)
    return plan_serve(request.root.input)


def patched_speculative_configs(result: PlanServeResult) -> list[dict[str, object]]:
    configs = []
    for role in result.roles:
        extra_args = role.effective_settings["extra_args"].root
        assert isinstance(extra_args, list)
        index = extra_args.index(SettingValue(root="--speculative-config"))
        token = extra_args[index + 1].root
        assert isinstance(token, str)
        configs.append(json.loads(token))
    return configs


def test_plan_overlays_the_curve_form_onto_the_operator_speculative_config() -> None:
    # The shared fixture declares the curve form: model dsv4, thinking_on,
    # whose text holds draft count 4 -> acceptance length 3.5.
    payload = synthetic_plan_payload({"method": "mtp", "num_speculative_tokens": 4})

    result = plan_synthetic(payload)

    assert (
        patched_speculative_configs(result)
        == [
            {
                "method": "mtp",
                "num_speculative_tokens": 4,
                "rejection_sample_method": "synthetic",
                "synthetic_acceptance_length": 3.5,
            }
        ]
        * 2
    )
    outcome = result.synthetic_acceptance
    assert outcome is not None
    assert outcome.acceptance_length == 3.5
    assert outcome.draft_count == 4


def test_plan_overlays_the_explicit_form_without_a_draft_count() -> None:
    payload = synthetic_plan_payload(
        {"method": "mtp", "num_speculative_tokens": 3},
        synthetic_acceptance={"explicit": {"acceptance_length": 2.25}},
    )

    result = plan_synthetic(payload)

    configs = patched_speculative_configs(result)
    assert configs[0]["synthetic_acceptance_length"] == 2.25
    assert configs[0]["rejection_sample_method"] == "synthetic"
    assert configs[0]["num_speculative_tokens"] == 3
    outcome = result.synthetic_acceptance
    assert outcome is not None
    assert outcome.acceptance_length == 2.25
    assert outcome.draft_count is None


def test_plan_patches_the_equals_spelling_of_speculative_config() -> None:
    payload = synthetic_plan_payload(None)
    input_payload = cast(dict[str, object], payload["input"])
    roles = cast(list[dict[str, object]], input_payload["roles"])
    for role in roles:
        cast(dict[str, object], role["settings"])["extra_args"] = [
            '--speculative-config={"method":"mtp","num_speculative_tokens":4}'
        ]

    result = plan_synthetic(payload)

    for role_result in result.roles:
        extra_args = role_result.effective_settings["extra_args"].root
        assert isinstance(extra_args, list)
        token = extra_args[0].root
        assert isinstance(token, str)
        assert token.startswith("--speculative-config=")
        config = json.loads(token.partition("=")[2])
        assert config["synthetic_acceptance_length"] == 3.5
    outcome = result.synthetic_acceptance
    assert outcome is not None
    assert outcome.draft_count == 4


def test_plan_without_a_speculative_config_cannot_overlay_synthetic_acceptance() -> None:
    payload = synthetic_plan_payload(None)

    message = synthetic_error(payload)

    assert "--speculative-config" in message


def test_plan_rejects_an_unparseable_speculative_config() -> None:
    payload = synthetic_plan_payload("{not json")

    message = synthetic_error(payload)

    assert "--speculative-config" in message


def test_plan_rejects_a_curve_lookup_without_a_determinable_draft_count() -> None:
    payload = synthetic_plan_payload({"method": "mtp"})

    message = synthetic_error(payload)

    assert "num_speculative_tokens" in message


def test_plan_rejects_a_curve_without_the_configured_draft_count_entry() -> None:
    # The shared fixture curve holds only draft count 4.
    payload = synthetic_plan_payload({"method": "mtp", "num_speculative_tokens": 3})

    message = synthetic_error(payload)

    assert "draft count 3" in message


def test_plan_rejects_roles_resolving_different_curve_draft_counts() -> None:
    payload = synthetic_plan_payload(None)
    input_payload = cast(dict[str, object], payload["input"])
    curve = {
        "model_key": "dsv4",
        "thinking_mode": "thinking_on",
        "text": "dsv4:\n  thinking_on:\n    3: 2.6\n    4: 3.5\n",
        "sha256": "f" * 64,
    }
    input_payload["synthetic_acceptance"] = {"curve": curve}
    roles = cast(list[dict[str, object]], input_payload["roles"])
    for role, draft_count in zip(roles, [4, 3], strict=True):
        cast(dict[str, object], role["settings"])["extra_args"] = [
            "--speculative-config",
            json.dumps({"method": "mtp", "num_speculative_tokens": draft_count}),
        ]

    message = synthetic_error(payload)

    assert "different synthetic acceptance outcomes" in message


def test_plan_rejects_an_operator_restated_synthetic_rejection_sampling() -> None:
    payload = synthetic_plan_payload(
        {
            "method": "mtp",
            "num_speculative_tokens": 4,
            "rejection_sample_method": "synthetic",
        }
    )

    message = synthetic_error(payload)

    assert "rejection_sample_method" in message
