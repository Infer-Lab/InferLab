import inferlab_gateway_smg as smg
import pytest
from inferlab_adapter_sdk import (
    AdapterErrorCode,
    AdapterOperationError,
    EndpointAssignment,
    ReadinessProbeHttp,
    ReadinessProbeHttpTargetRegistry,
    TargetEndpointScheme,
)


def endpoint(host: str, port: int) -> EndpointAssignment:
    return EndpointAssignment(host=host, port=port)


GATEWAY = endpoint("10.0.0.1", 30000)
PROMETHEUS = endpoint("10.0.0.1", 30001)


def test_the_backend_is_named_smg_and_the_former_name_is_rejected_with_the_rename() -> None:
    smg.require_backend("smg", component="Gateway")

    with pytest.raises(AdapterOperationError) as former:
        smg.require_backend("tokenspeed-smg", component="P/D Router")
    assert former.value.code == AdapterErrorCode.invalid_settings
    assert 'select "smg"' in former.value.message
    assert "tokenspeed-smg" in former.value.message

    with pytest.raises(AdapterOperationError) as other:
        smg.require_backend("vllm-router", component="Gateway")
    assert other.value.code == AdapterErrorCode.invalid_settings


def test_routed_single_launches_one_grpc_worker_behind_the_gateway() -> None:
    command = smg.routed_single_command(
        endpoint=GATEWAY,
        prometheus=PROMETHEUS,
        model_locator="/models/demo",
        worker=endpoint("10.0.0.2", 40000),
        policy="least_load",
    )
    assert command.argv == [
        "smg",
        "launch",
        "--host",
        "10.0.0.1",
        "--port",
        "30000",
        "--prometheus-port",
        "30001",
        "--worker-startup-timeout-secs",
        "2147483647",
        "--model-path",
        "/models/demo",
        "--tokenizer-path",
        "/models/demo",
        "--worker-urls",
        "grpc://10.0.0.2:40000",
        "--policy",
        "least_load",
        "--disable-retries",
        "--disable-circuit-breaker",
    ]
    assert command.env == {}


def test_prefill_decode_pairs_each_prefill_target_with_its_bootstrap_port() -> None:
    command = smg.prefill_decode_command(
        endpoint=GATEWAY,
        prometheus=PROMETHEUS,
        model_locator="/models/demo",
        prefill=[(endpoint("10.0.0.2", 40000), 41000), (endpoint("10.0.0.3", 40000), 41000)],
        decode=[endpoint("10.0.0.4", 40000)],
        prefill_policy="round_robin",
        decode_policy="round_robin",
    )
    assert command.argv[:2] == ["smg", "launch"]
    tail = command.argv[command.argv.index("--pd-disaggregation") :]
    assert tail == [
        "--pd-disaggregation",
        "--prefill",
        "grpc://10.0.0.2:40000",
        "41000",
        "--prefill",
        "grpc://10.0.0.3:40000",
        "41000",
        "--decode",
        "grpc://10.0.0.4:40000",
        "--policy",
        "round_robin",
        "--prefill-policy",
        "round_robin",
        "--decode-policy",
        "round_robin",
        "--disable-retries",
        "--disable-circuit-breaker",
    ]


def test_readiness_follows_the_gateway_and_the_worker_registry() -> None:
    assert smg.gateway_readiness().root == ReadinessProbeHttp(path="/readiness")
    registry = smg.pd_router_readiness(prefill_bootstrap_port="bootstrap").root
    assert isinstance(registry, ReadinessProbeHttpTargetRegistry)
    assert registry.registry_path == "/workers"
    assert registry.targets_field == "workers"
    assert registry.target_role_field == "worker_type"
    assert registry.target_bootstrap_port_field == "bootstrap_port"
    assert registry.target_scheme == TargetEndpointScheme.grpc
    assert registry.prefill_bootstrap_port == "bootstrap"


def test_the_identity_records_the_installed_gateway_distribution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(smg, "version", lambda distribution: f"{distribution}-9.9")
    assert smg.IMPLEMENTATION == "smg"
    assert smg.installed_version() == "tokenspeed-smg-9.9"

    def missing(distribution: str) -> str:
        raise smg.PackageNotFoundError(distribution)

    monkeypatch.setattr(smg, "version", missing)
    assert smg.installed_version() == "unavailable"
