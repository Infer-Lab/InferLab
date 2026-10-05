"""vLLM served through Dynamo ([[RFC-0003:C-DYNAMO]])."""

import json
from pathlib import Path
from typing import cast

import pytest
from inferlab_adapter_sdk import (
    AdapterOperationError,
    AdapterRequest,
    AdapterRequestPlanServe,
    AdapterRequestRenderServe,
    PlanServeInput,
    ReadinessProbeHttp,
    ReadinessProbeRegistryMembership,
    RenderServeInput,
)
from inferlab_integration_vllm import plan_serve, render_serve

FIXTURES = Path(__file__).parents[3] / "protocol" / "fixtures"


def plan_input(**overrides: object) -> PlanServeInput:
    payload = json.loads((FIXTURES / "valid" / "plan-serve-request.json").read_text())
    input_payload = cast(dict[str, object], payload["input"])
    input_payload.pop("synthetic_acceptance")
    input_payload.update(
        {"gateway_backend": "dynamo", "pd_router_backend": "dynamo", "kv_transfer": "nixl"}
    )
    input_payload.update(overrides)
    request = AdapterRequest.model_validate(payload)
    assert isinstance(request.root, AdapterRequestPlanServe)
    return request.root.input


def single_plan_input(settings: dict[str, object] | None = None) -> PlanServeInput:
    return plan_input(
        topology="single",
        pd_router_backend=None,
        kv_transfer=None,
        roles=[
            {
                "id": "serve",
                "kind": "serve",
                "replica_count": 2,
                "parallelism": {"outer": {"tensor_parallel_size": 1}},
                "settings": settings or {"reasoning_parser": "qwen3"},
            }
        ],
    )


def render_input(**worker: object) -> RenderServeInput:
    """The shared discovery render fixture with the planned Dynamo P/D
    frontend, request-plane and side-channel ports, and NIXL."""
    plan = plan_serve(plan_input())
    assert plan.gateway is not None and plan.pd_router is not None
    payload = json.loads((FIXTURES / "valid" / "render-serve-request-discovery.json").read_text())
    input_payload = cast(dict[str, object], payload["input"])
    input_payload.update(gateway_backend="dynamo", pd_router_backend="dynamo", kv_transfer="nixl")
    for allocation in cast(list[dict[str, object]], input_payload["allocations"]):
        if allocation["kind"] == "frontend":
            allocation["gateway"] = plan.gateway.model_dump(mode="json")
            allocation["pd_router"] = plan.pd_router.model_dump(mode="json")
            allocation["ports"] = {"response_stream": {"host": "node-a.example", "port": 9001}}
            allocation["discovery"] = "discovery"
        if allocation["kind"] == "model_rank":
            host = "192.0.2.1" if allocation["role"] == "prefill" else "192.0.2.2"
            allocation["endpoint"] = {"host": host, "port": 8000}
            allocation["ports"] = {
                "request": {"host": host, "port": 20001},
                "side_channel": {"host": host, "port": 20002},
            }
            allocation["effective_settings"] = {"reasoning_parser": "qwen3"}
            allocation.update(worker)
    request = AdapterRequest.model_validate(payload)
    assert isinstance(request.root, AdapterRequestRenderServe)
    return request.root.input


def test_dynamo_routed_single_plans_discovery_registry_readiness_and_per_target_reset() -> None:
    result = plan_serve(single_plan_input())

    gateway = result.gateway
    assert gateway is not None and result.pd_router is None
    assert (gateway.backend, gateway.render_source) == ("dynamo", "integration")
    assert gateway.discovery is not None
    assert isinstance(gateway.readiness.root, ReadinessProbeRegistryMembership)
    assert gateway.endpoint.prefix_cache_reset is None
    reset = result.roles[0].replica_prefix_cache_reset
    assert reset is not None and reset.path == "/engine/flush_cache"
    assert reset.success is not None and reset.success.value == "ok"
    for replica in result.replicas:
        assert replica.ports == ["request"]
        assert isinstance(replica.primary_readiness.root, ReadinessProbeHttp)
        assert replica.primary_readiness.root.path == "/health"


def test_dynamo_pd_router_reaches_workers_over_the_tcp_request_plane() -> None:
    pd_router = plan_serve(plan_input()).pd_router

    assert pd_router is not None and pd_router.target_scheme == "tcp"


def test_dynamo_prefill_decode_accepts_only_nixl() -> None:
    with pytest.raises(AdapterOperationError, match="NIXL"):
        plan_serve(plan_input(kv_transfer="mooncake"))


@pytest.mark.parametrize(
    "settings",
    [
        {"tool_call_parser": "hermes"},
        # Past the verbatim sentinel too: the Dynamo worker would abort at launch.
        {"extra_args": ["--", "--tool-call-parser", "hermes"]},
    ],
)
def test_dynamo_rejects_vllm_api_server_options(settings: dict[str, object]) -> None:
    with pytest.raises(AdapterOperationError, match="tool"):
        plan_serve(single_plan_input(settings))


def test_dynamo_prefill_decode_renders_workers_frontend_and_discovery() -> None:
    result = render_serve(render_input())

    by_process = {process.root.process: process.root.command for process in result.processes}
    assert by_process["discovery"].argv[0] == "etcd"
    assert by_process["gateway"].argv[:3] == ["python3", "-m", "dynamo.frontend"]
    for process, mode, kv_role in [
        ("prefill", "prefill", "kv_producer"),
        ("decode", "decode", "kv_consumer"),
    ]:
        command = by_process[process]
        argv = command.argv
        assert argv[:5] == ["python3", "-m", "dynamo.vllm", "--model", "/models/deepseek-v4-flash"]
        assert "--host" not in argv and "--port" not in argv
        assert argv[argv.index("--disaggregation-mode") + 1] == mode
        assert f'"kv_role":"{kv_role}"' in argv[argv.index("--kv-transfer-config") + 1]
        # The Dynamo frontend, not vLLM, post-processes reasoning.
        assert argv[argv.index("--dyn-reasoning-parser") + 1] == "qwen3"
        assert "--reasoning-parser" not in argv
        assert command.env["DYN_SYSTEM_PORT"] == "8000"
        assert command.env["DYN_TCP_RPC_PORT"] == "20001"
        assert command.env["VLLM_NIXL_SIDE_CHANNEL_PORT"] == "20002"
        assert command.env["ETCD_ENDPOINTS"] == "http://127.0.0.1:2379"


def test_dynamo_rejects_a_multi_rank_replica_before_launch() -> None:
    with pytest.raises(AdapterOperationError, match="single-rank"):
        render_serve(render_input(rank_count=2))
