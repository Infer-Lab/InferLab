import json
from pathlib import Path
from typing import cast

import inferlab_gateway_dynamo
import pytest
from inferlab_adapter_sdk import (
    AdapterOperationError,
    AdapterRequest,
    AdapterRequestRenderServe,
    ServeAllocation,
    ServeProcessAllocationDiscovery,
    ServeProcessAllocationFrontend,
    ServeProcessAllocationModelRank,
)

FIXTURES = Path(__file__).parents[3] / "protocol" / "fixtures"


def dynamo_allocations() -> list[ServeAllocation]:
    """The shared discovery render fixture with its fused frontend turned into
    the planned Dynamo Gateway and P/D Router."""
    payload = json.loads((FIXTURES / "valid" / "render-serve-request-discovery.json").read_text())
    allocations = cast(list[dict[str, object]], payload["input"]["allocations"])
    gateway, pd_router = inferlab_gateway_dynamo.prefill_decode_frontends("prefill", "decode")
    for allocation in allocations:
        if allocation["kind"] == "frontend":
            allocation["gateway"] = gateway.model_dump(mode="json")
            allocation["pd_router"] = pd_router.model_dump(mode="json")
            allocation["ports"] = {"response_stream": {"host": "node-a.example", "port": 9001}}
            allocation["discovery"] = "discovery"
        if allocation["kind"] == "model_rank":
            allocation["endpoint"] = {"host": "192.0.2.1", "port": 8000}
            allocation["ports"] = {"request": {"host": "192.0.2.1", "port": 20001}}
    request = AdapterRequest.model_validate(payload)
    assert isinstance(request.root, AdapterRequestRenderServe)
    return [allocation.root for allocation in request.root.input.allocations]


def test_discovery_renders_one_etcd_member_on_its_allocated_listeners() -> None:
    discovery = next(
        item for item in dynamo_allocations() if isinstance(item, ServeProcessAllocationDiscovery)
    )

    rendered = inferlab_gateway_dynamo.render_discovery(discovery).root

    argv = rendered.command.argv
    assert argv[0] == "etcd"
    assert argv[argv.index("--data-dir") + 1] == discovery.data_directory
    assert argv[argv.index("--listen-client-urls") + 1] == "http://127.0.0.1:2379"
    assert argv[argv.index("--initial-cluster") + 1] == "inferlab=http://127.0.0.1:2380"


def test_frontend_receives_discovery_and_only_its_response_stream_port() -> None:
    allocations = dynamo_allocations()
    frontend = next(
        item for item in allocations if isinstance(item, ServeProcessAllocationFrontend)
    )

    command = inferlab_gateway_dynamo.render_frontend(frontend, allocations).root.command

    assert command.argv[:3] == ["python3", "-m", "dynamo.frontend"]
    assert command.argv[command.argv.index("--http-port") + 1] == "9000"
    assert command.argv[command.argv.index("--router-mode") + 1] == "round-robin"
    assert command.env["ETCD_ENDPOINTS"] == "http://127.0.0.1:2379"
    assert command.env["DYN_NAMESPACE"] == inferlab_gateway_dynamo.NAMESPACE
    assert command.env["DYN_TCP_RESPONSE_STREAM_PORT"] == "9001"
    assert "DYN_TCP_RESPONSE_STREAM_HOST" not in command.env


def test_a_control_port_dynamo_cannot_parse_is_rejected_before_launch() -> None:
    allocations = dynamo_allocations()
    worker = next(item for item in allocations if isinstance(item, ServeProcessAllocationModelRank))
    assert worker.endpoint is not None
    worker.endpoint.port = inferlab_gateway_dynamo.SYSTEM_PORT_LIMIT + 1

    with pytest.raises(AdapterOperationError, match="32767"):
        inferlab_gateway_dynamo.worker_env(worker, allocations)


def test_escape_hatches_cannot_rewire_discovery_even_after_the_sentinel() -> None:
    with pytest.raises(AdapterOperationError, match="--namespace"):
        inferlab_gateway_dynamo.validate_escape_hatch(["--", "--namespace", "other"], {})
    # The environment form of an owned option is owned too.
    with pytest.raises(AdapterOperationError, match="DYN_ENDPOINT"):
        inferlab_gateway_dynamo.validate_escape_hatch(
            [], {"DYN_ENDPOINT": "dyn://inferlab.backend.generate"}
        )


def test_a_machine_bound_by_hostname_is_rejected_before_launch() -> None:
    allocations = dynamo_allocations()
    worker = next(item for item in allocations if isinstance(item, ServeProcessAllocationModelRank))
    worker.ports["request"].host = "node-a.example"

    with pytest.raises(AdapterOperationError, match="IP addresses"):
        inferlab_gateway_dynamo.worker_env(worker, allocations)
