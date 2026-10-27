import inferlab_kvstore
import pytest
from inferlab_adapter_sdk import (
    AdapterOperationError,
    EndpointAssignment,
    KvStoreInput,
    ReadinessProbeHttp,
    SettingValue,
)
from inferlab_kvstore import mooncake


def store(**settings: object) -> KvStoreInput:
    return KvStoreInput(
        backend="mooncake",
        settings={key: SettingValue.model_validate(value) for key, value in settings.items()},
    )


def test_plan_probes_the_admin_health_path_and_keeps_declared_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mooncake, "installed_version", lambda: "9.9.9")

    plan = mooncake.plan(store(eviction_high_watermark_ratio=0.8))

    assert plan.backend == "mooncake"
    assert plan.implementation_version == "9.9.9"
    assert plan.ports == ["admin"]
    readiness = plan.readiness.root
    assert isinstance(readiness, ReadinessProbeHttp)
    assert (readiness.path, readiness.port) == ("/health", "admin")
    assert plan.effective_settings == {"eviction_high_watermark_ratio": SettingValue(root=0.8)}


def test_master_binds_its_allocated_addresses_without_a_metadata_server(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mooncake, "installed_version", lambda: "9.9.9")
    plan = mooncake.plan(store(default_kv_lease_ttl_ms=5000, extra_args=["-client_ttl=20"]))

    command = mooncake.master_command(
        endpoint=EndpointAssignment(host="store.example", port=50051),
        admin=EndpointAssignment(host="store.example", port=50052),
        store_plan=plan,
    )

    assert command.argv == [
        "mooncake_master",
        "-rpc_address=store.example",
        "-rpc_port=50051",
        "-metrics_host=store.example",
        "-metrics_port=50052",
        "-enable_http_metadata_server=false",
        "-default_kv_lease_ttl=5000",
        "-client_ttl=20",
    ]


def test_extra_args_cannot_restate_an_owned_master_flag() -> None:
    with pytest.raises(AdapterOperationError, match="-rpc_port"):
        mooncake.plan(store(extra_args=["-rpc_port=1"]))


def test_another_backend_is_rejected() -> None:
    with pytest.raises(AdapterOperationError, match="select one of 'mooncake'"):
        inferlab_kvstore.plan(KvStoreInput(backend="redis", settings={}))
