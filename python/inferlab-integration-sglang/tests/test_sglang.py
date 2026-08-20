from typing import cast

import pytest
from inferlab_adapter_sdk import (
    AdapterOperationError,
    KvTransferMechanism,
    Parallelism,
    ParallelismAttention,
    ParallelismExperts,
    ParallelismOuter,
    PlanServeInput,
    PlanServeResult,
    PromptCacheReadZeroRepresentation,
    ReadinessProbeHttp,
    ReadinessProbeHttpTargetRegistry,
    RenderServeInput,
    RenderSource,
    ServeModelInput,
    ServeProcessAllocation,
    ServeRoleInput,
    ServeRoleKind,
    ServeRoleLinkKvTransfer,
    ServeTopology,
    SettingValue,
)
from inferlab_integration_sglang import plan_serve, render_serve


def _plan_input(**overrides: object) -> PlanServeInput:
    parallelism = cast(
        Parallelism,
        overrides.pop(
            "parallelism",
            Parallelism(outer=ParallelismOuter(tensor_parallel_size=2)),
        ),
    )
    settings = cast(
        dict[str, SettingValue],
        overrides.pop("settings", {"trust_remote_code": SettingValue(root=True)}),
    )
    roles = overrides.pop(
        "roles",
        [
            ServeRoleInput(
                id="serve",
                kind=ServeRoleKind.serve,
                replica_count=1,
                parallelism=parallelism,
                settings=settings,
            )
        ],
    )
    base: dict[str, object] = {
        "model": ServeModelInput(id="example", served_name="example"),
        "topology": ServeTopology.single,
        "gateway_backend": None,
        "pd_router_backend": None,
        "kv_transfer": None,
        "roles": roles,
        "profiling": False,
    }
    base.update(overrides)
    return PlanServeInput.model_validate(base)


def test_plan_single_topology() -> None:
    result = plan_serve(_plan_input())

    assert result.integration.framework == "sglang"
    assert [replica.id for replica in result.replicas] == ["server"]
    assert result.replicas[0].device_count == 2
    assert result.replicas[0].capture_target is None
    probe = result.replicas[0].primary_readiness.root
    assert isinstance(probe, ReadinessProbeHttp) and probe.path == "/v1/models"
    endpoint = result.roles[0].public_endpoint
    assert endpoint is not None
    assert endpoint.completions_path == "/v1/completions"
    assert endpoint.chat_completions_path == "/v1/chat/completions"
    assert endpoint.server_metrics is None
    assert endpoint.prefix_cache_reset is not None
    assert endpoint.prefix_cache_reset.path == "/flush_cache"
    assert result.gateway is None
    assert result.pd_router is None
    outer = result.roles[0].effective_parallelism.outer
    assert outer is not None and outer.tensor_parallel_size == 2


def test_plan_single_declares_cuda_profiler_window_control() -> None:
    result = plan_serve(_plan_input(profiling=True))

    assert len(result.replicas) == 1
    target = result.replicas[0].capture_target
    assert target is not None
    assert target.model_dump(mode="json") == {
        "window_control": {
            "endpoint": "replica_entry",
            "start": {
                "method": "post",
                "path": "/start_profile",
                "body": {"activities": ["CUDA_PROFILER"]},
            },
            "stop": {
                "method": "post",
                "path": "/stop_profile",
                "body": None,
            },
        }
    }


def test_plan_rejects_unsupported_shapes() -> None:
    with pytest.raises(AdapterOperationError):
        plan_serve(
            _plan_input(
                roles=[
                    ServeRoleInput(
                        id="serve",
                        kind=ServeRoleKind.serve,
                        replica_count=1,
                        parallelism=Parallelism(
                            outer=ParallelismOuter(tensor_parallel_size=2),
                            attention=ParallelismAttention(data_parallel_size=3),
                        ),
                        settings={},
                    )
                ]
            )
        )
    with pytest.raises(AdapterOperationError):
        plan_serve(_plan_input(settings={"unknown_setting": SettingValue(root=1)}))


def _prefill_decode_roles(
    prefill_replicas: int = 2, decode_replicas: int = 3
) -> list[ServeRoleInput]:
    return [
        ServeRoleInput(
            id="prefill",
            kind=ServeRoleKind.prefill,
            replica_count=prefill_replicas,
            parallelism=Parallelism(outer=ParallelismOuter(tensor_parallel_size=2)),
            settings={},
        ),
        ServeRoleInput(
            id="decode",
            kind=ServeRoleKind.decode,
            replica_count=decode_replicas,
            parallelism=Parallelism(outer=ParallelismOuter(tensor_parallel_size=2)),
            settings={},
        ),
    ]


def _prefill_decode_plan_input(
    *,
    frontend_backend: str = "builtin",
    transport: KvTransferMechanism = KvTransferMechanism.mooncake,
    prefill_replicas: int = 2,
    decode_replicas: int = 3,
    profiling: bool = False,
) -> PlanServeInput:
    return _plan_input(
        topology=ServeTopology.prefill_decode,
        gateway_backend=frontend_backend,
        pd_router_backend=frontend_backend,
        kv_transfer=transport,
        roles=_prefill_decode_roles(prefill_replicas, decode_replicas),
        profiling=profiling,
    )


@pytest.mark.parametrize("transport", [KvTransferMechanism.mooncake, KvTransferMechanism.nixl])
def test_plan_prefill_decode_uses_the_shared_bootstrap_shape(
    transport: KvTransferMechanism,
) -> None:
    result = plan_serve(_prefill_decode_plan_input(transport=transport))

    assert [role.effective_replica_count for role in result.roles] == [2, 3]
    assert [replica.id for replica in result.replicas] == [
        "prefill-000",
        "prefill-001",
        "decode-000",
        "decode-001",
        "decode-002",
    ]
    assert [replica.ports for replica in result.replicas] == [
        ["bootstrap"],
        ["bootstrap"],
        [],
        [],
        [],
    ]
    assert all(replica.capture_target is None for replica in result.replicas)
    assert [link.root.kind for link in result.links] == [
        "request_routing",
        "request_routing",
        "kv_transfer",
        "bootstrap",
    ]
    transfer = result.links[2].root
    assert isinstance(transfer, ServeRoleLinkKvTransfer)
    assert transfer.mechanism == transport
    assert result.gateway is not None
    assert result.gateway.backend == "builtin"
    assert result.gateway.render_source == RenderSource.control_plane
    assert result.gateway.endpoint.completions_path == "/v1/completions"
    assert result.gateway.endpoint.chat_completions_path == "/v1/chat/completions"
    assert result.gateway.endpoint.server_metrics is None
    assert result.gateway.endpoint.prefix_cache_reset is not None
    assert result.gateway.endpoint.prefix_cache_reset.path == "/flush_cache"
    assert result.pd_router is not None
    assert result.pd_router.backend == "builtin"


def test_plan_prefill_decode_declares_every_replica_as_a_capture_target() -> None:
    result = plan_serve(_prefill_decode_plan_input(profiling=True))

    assert len(result.replicas) == 5
    for replica in result.replicas:
        target = replica.capture_target
        assert target is not None
        control = target.window_control
        assert control.endpoint.value == "replica_entry"
        assert control.start.model_dump(mode="json") == {
            "method": "post",
            "path": "/start_profile",
            "body": {"activities": ["CUDA_PROFILER"]},
        }
        assert control.stop.model_dump(mode="json") == {
            "method": "post",
            "path": "/stop_profile",
            "body": None,
        }


def test_plan_sglang_router_declares_worker_aware_readiness() -> None:
    result = plan_serve(
        _prefill_decode_plan_input(
            frontend_backend="sglang-router", prefill_replicas=1, decode_replicas=1
        )
    )

    assert [role.kind for role in result.roles] == [
        ServeRoleKind.prefill,
        ServeRoleKind.decode,
    ]
    assert result.gateway is not None
    assert result.gateway.backend == "sglang-router"
    assert result.pd_router is not None
    readiness = result.pd_router.readiness.root
    assert isinstance(readiness, ReadinessProbeHttpTargetRegistry)
    assert readiness.model_dump() == {
        "kind": "http_target_registry",
        "target_scheme": "http",
        "readiness_path": "/readiness",
        "registry_path": "/workers",
        "targets_field": "workers",
        "target_url_field": "url",
        "target_role_field": "worker_type",
        "target_healthy_field": "is_healthy",
        "target_bootstrap_port_field": "bootstrap_port",
        "prefill_role_value": "prefill",
        "decode_role_value": "decode",
        "prefill_bootstrap_port": "bootstrap",
    }
    assert result.gateway.render_source == RenderSource.integration
    assert result.pd_router.render_source == RenderSource.integration


def test_plan_prefill_decode_rejects_an_unknown_router() -> None:
    with pytest.raises(AdapterOperationError):
        plan_serve(_prefill_decode_plan_input(frontend_backend="unknown"))


def test_plan_expert_parallel_mapping() -> None:
    parallelism = Parallelism(
        outer=ParallelismOuter(tensor_parallel_size=2),
        experts=ParallelismExperts(expert_parallel_size=2),
    )
    result = plan_serve(
        _plan_input(
            parallelism=parallelism,
            roles=[
                ServeRoleInput(
                    id="serve",
                    kind=ServeRoleKind.serve,
                    replica_count=1,
                    parallelism=parallelism,
                    settings={},
                )
            ],
        )
    )
    experts = result.roles[0].effective_parallelism.experts
    assert experts is not None
    assert experts.expert_parallel_size == 2
    assert experts.tensor_parallel_size == 1, "EP divides the TP world"
    assert result.replicas[0].device_count == 2, "the world stays outer TP x PP"


def _plan_with_parallelism(parallelism: Parallelism) -> PlanServeResult:
    return plan_serve(
        _plan_input(
            parallelism=parallelism,
            roles=[
                ServeRoleInput(
                    id="serve",
                    kind=ServeRoleKind.serve,
                    replica_count=1,
                    parallelism=parallelism,
                    settings={},
                )
            ],
        )
    )


def test_plan_rejects_the_moe_dp_combinations_sglang_asserts_on() -> None:
    # The limits SGLang 0.5.14 enforces at server start (server_args.py).
    with pytest.raises(AdapterOperationError):
        _plan_with_parallelism(
            Parallelism(
                outer=ParallelismOuter(tensor_parallel_size=4, pipeline_parallel_size=2),
                attention=ParallelismAttention(context_parallel_size=2),
                experts=ParallelismExperts(data_parallel_size=2),
            )
        )
    with pytest.raises(AdapterOperationError):
        _plan_with_parallelism(
            Parallelism(
                outer=ParallelismOuter(tensor_parallel_size=4),
                experts=ParallelismExperts(data_parallel_size=2),
            )
        )
    with pytest.raises(AdapterOperationError):
        _plan_with_parallelism(
            Parallelism(
                outer=ParallelismOuter(tensor_parallel_size=8),
                attention=ParallelismAttention(context_parallel_size=2),
                experts=ParallelismExperts(expert_parallel_size=2, data_parallel_size=2),
            )
        )


def test_plan_accepts_the_moe_dp_boundary_shapes() -> None:
    # ep * moe-dp == tp with cp == moe-dp: the exact shape 0.5.14 allows.
    exact = _plan_with_parallelism(
        Parallelism(
            outer=ParallelismOuter(tensor_parallel_size=4),
            attention=ParallelismAttention(context_parallel_size=2),
            experts=ParallelismExperts(expert_parallel_size=2, data_parallel_size=2),
        )
    )
    experts = exact.roles[0].effective_parallelism.experts
    assert experts is not None and experts.tensor_parallel_size == 1
    # moe-dp == 1 keeps every previously qualified combination untouched.
    divides = _plan_with_parallelism(
        Parallelism(
            outer=ParallelismOuter(tensor_parallel_size=8),
            experts=ParallelismExperts(expert_parallel_size=2),
        )
    )
    experts = divides.roles[0].effective_parallelism.experts
    assert experts is not None and experts.tensor_parallel_size == 4


def _dp_parallelism() -> Parallelism:
    return Parallelism(
        outer=ParallelismOuter(tensor_parallel_size=4),
        attention=ParallelismAttention(data_parallel_size=2),
        experts=ParallelismExperts(expert_parallel_size=4),
    )


def test_plan_dp_attention_divides_the_world() -> None:
    parallelism = _dp_parallelism()
    result = plan_serve(
        _plan_input(
            parallelism=parallelism,
            roles=[
                ServeRoleInput(
                    id="serve",
                    kind=ServeRoleKind.serve,
                    replica_count=1,
                    parallelism=parallelism,
                    settings={},
                )
            ],
        )
    )
    assert result.replicas[0].device_count == 4, "outer TP is the world size"
    attention = result.roles[0].effective_parallelism.attention
    assert attention is not None
    assert attention.tensor_parallel_size == 2, "attention DP divides the world"
    assert attention.data_parallel_size == 2
    experts = result.roles[0].effective_parallelism.experts
    assert experts is not None
    assert experts.tensor_parallel_size == 1 and experts.expert_parallel_size == 4


def test_render_lowers_dp_attention_and_expert_parallelism() -> None:
    parallelism = _dp_parallelism()
    plan = plan_serve(
        _plan_input(
            parallelism=parallelism,
            roles=[
                ServeRoleInput(
                    id="serve",
                    kind=ServeRoleKind.serve,
                    replica_count=1,
                    parallelism=parallelism,
                    settings={},
                )
            ],
        )
    )
    result = render_serve(
        _render_input(
            parallelism=plan.roles[0].effective_parallelism,
            settings=plan.roles[0].effective_settings,
        )
    )
    argv = result.processes[0].root.command.argv
    assert argv[argv.index("--tensor-parallel-size") + 1] == "4"
    assert argv[argv.index("--data-parallel-size") + 1] == "2"
    assert "--enable-dp-attention" in argv
    assert argv[argv.index("--expert-parallel-size") + 1] == "4"
    assert "--moe-data-parallel-size" not in argv
    assert "--pipeline-parallel-size" not in argv


def test_render_lowers_pipeline_parallelism() -> None:
    parallelism = Parallelism(
        outer=ParallelismOuter(tensor_parallel_size=2, pipeline_parallel_size=2)
    )
    plan = plan_serve(
        _plan_input(
            parallelism=parallelism,
            roles=[
                ServeRoleInput(
                    id="serve",
                    kind=ServeRoleKind.serve,
                    replica_count=1,
                    parallelism=parallelism,
                    settings={},
                )
            ],
        )
    )
    assert plan.replicas[0].device_count == 4
    result = render_serve(
        _render_input(
            parallelism=plan.roles[0].effective_parallelism,
            settings=plan.roles[0].effective_settings,
        )
    )
    argv = result.processes[0].root.command.argv
    assert argv[argv.index("--pipeline-parallel-size") + 1] == "2"
    assert "--enable-dp-attention" not in argv


def _render_input(**overrides: object) -> RenderServeInput:
    profiling = cast(bool, overrides.pop("profiling", False))
    plan = plan_serve(_plan_input(profiling=profiling))
    parallelism = cast(
        Parallelism,
        overrides.pop("parallelism", plan.roles[0].effective_parallelism),
    )
    settings = cast(
        dict[str, SettingValue],
        overrides.pop("settings", plan.roles[0].effective_settings),
    )
    base: dict[str, object] = {
        "model": ServeModelInput(id="example", served_name="example"),
        "topology": ServeTopology.single,
        "gateway_backend": None,
        "pd_router_backend": None,
        "kv_transfer": None,
        "profiling": profiling,
        "allocations": [
            ServeProcessAllocation.model_validate(
                {
                    "kind": "model_rank",
                    "process": "server-rank-000",
                    "role": "serve",
                    "role_kind": "serve",
                    "replica": 0,
                    "rank": 0,
                    "rank_count": 1,
                    "machine": "local",
                    "model_locator": "/models/example",
                    "devices": [0, 1],
                    "endpoint": {"host": "127.0.0.1", "port": 8000},
                    "ports": {},
                    "cache": "/cache/server",
                    "launch": {"kind": "local"},
                    "effective_settings": settings,
                    "effective_parallelism": parallelism,
                    "links": [],
                    "dependencies": [],
                    "render_inputs": [],
                }
            )
        ],
    }
    base.update(overrides)
    return RenderServeInput.model_validate(base)


def _prefill_decode_render_input(
    *,
    frontend_backend: str = "builtin",
    transport: KvTransferMechanism = KvTransferMechanism.mooncake,
    profiling: bool = False,
) -> RenderServeInput:
    plan = plan_serve(
        _prefill_decode_plan_input(
            frontend_backend=frontend_backend,
            transport=transport,
            prefill_replicas=2,
            decode_replicas=2,
            profiling=profiling,
        )
    )
    roles = {role.id: role for role in plan.roles}
    allocations: list[ServeProcessAllocation] = []
    for index, replica in enumerate(plan.replicas):
        role = roles[replica.role_id]
        host = f"node-{index}.example"
        port = 8000
        ports = (
            {"bootstrap": {"host": host, "port": 9000 + index}}
            if "bootstrap" in replica.ports
            else {}
        )
        allocations.append(
            ServeProcessAllocation.model_validate(
                {
                    "kind": "model_rank",
                    "process": replica.id,
                    "role": replica.role_id,
                    "role_kind": role.kind,
                    "replica": replica.replica_index,
                    "rank": 0,
                    "rank_count": 1,
                    "machine": f"machine-{index}",
                    "model_locator": "/models/example",
                    "devices": list(range(replica.device_count)),
                    "endpoint": {"host": host, "port": port},
                    "ports": ports,
                    "cache": f"/cache/{replica.id}",
                    "launch": {"kind": "local"},
                    "effective_settings": role.effective_settings,
                    "effective_parallelism": role.effective_parallelism,
                    "links": plan.links,
                    "dependencies": [],
                    "render_inputs": [],
                }
            )
        )
    if plan.gateway is not None and plan.gateway.render_source == RenderSource.integration:
        assert plan.pd_router is not None
        allocations.append(
            ServeProcessAllocation.model_validate(
                {
                    "kind": "frontend",
                    "process": "gateway",
                    "process_role": "gateway",
                    "components": ["gateway", "pd_router"],
                    "machine": "gateway-machine",
                    "devices": [],
                    "endpoint": {"host": "127.0.0.1", "port": 7000},
                    "ports": {},
                    "cache": "/cache/gateway",
                    "launch": {"kind": "local"},
                    "gateway": plan.gateway,
                    "pd_router": plan.pd_router,
                    "links": plan.links,
                    "dependencies": [allocation.root.process for allocation in allocations],
                    "render_inputs": [],
                }
            )
        )
    return RenderServeInput(
        model=ServeModelInput(id="example", served_name="example"),
        topology=ServeTopology.prefill_decode,
        gateway_backend=frontend_backend,
        pd_router_backend=frontend_backend,
        kv_transfer=transport,
        profiling=profiling,
        allocations=allocations,
    )


def test_render_launches_sglang_server() -> None:
    result = render_serve(_render_input())

    assert len(result.processes) == 1
    process = result.processes[0].root
    argv = process.command.argv
    assert argv[:5] == [
        "python3",
        "-m",
        "sglang.launch_server",
        "--model-path",
        "/models/example",
    ]
    assert argv[argv.index("--tensor-parallel-size") + 1] == "2"
    assert "--enable-dp-attention" not in argv
    assert argv[argv.index("--port") + 1] == "8000"
    assert argv[argv.index("--served-model-name") + 1] == "example"
    assert "--trust-remote-code" in argv
    assert "--data-parallel-size" not in argv
    env = process.command.env
    assert env["TRITON_CACHE_DIR"] == "/cache/server/triton"
    assert env["TORCHINDUCTOR_CACHE_DIR"] == "/cache/server/torchinductor"


def test_metrics_capability_matches_the_effective_sglang_launch() -> None:
    plan = plan_serve(_plan_input(settings={"enable_metrics": SettingValue(root=True)}))
    endpoint = plan.roles[0].public_endpoint
    assert endpoint is not None
    assert endpoint.server_metrics is not None
    assert endpoint.server_metrics.model_dump(mode="json") == {
        "path": "/metrics",
        "port": None,
    }

    result = render_serve(_render_input(settings=plan.roles[0].effective_settings))
    assert "--enable-metrics" in result.processes[0].root.command.argv


def test_render_profiling_keeps_model_server_commands_unchanged() -> None:
    ordinary = render_serve(_render_input())
    profiled = render_serve(_render_input(profiling=True))

    assert profiled.processes == ordinary.processes

    ordinary_pd = render_serve(_prefill_decode_render_input())
    profiled_pd = render_serve(_prefill_decode_render_input(profiling=True))
    assert profiled_pd.processes == ordinary_pd.processes


def test_plan_rejects_inferlab_owned_option_in_extra_args() -> None:
    with pytest.raises(AdapterOperationError, match="--port"):
        plan_serve(
            _plan_input(
                settings={
                    "extra_args": SettingValue.model_validate(["--port", "1"]),
                }
            )
        )


def test_render_passes_through_unrecognized_extra_args() -> None:
    plan = plan_serve(
        _plan_input(
            settings={
                "mem_fraction_static": SettingValue(root=0.8),
                "extra_args": SettingValue.model_validate(["--log-level", "debug"]),
            }
        )
    )
    result = render_serve(_render_input(settings=plan.roles[0].effective_settings))
    argv = result.processes[0].root.command.argv
    assert argv[argv.index("--mem-fraction-static") + 1] == "0.8"
    assert "--log-level" in argv, "unrecognized extra args pass through"


def test_render_lowers_published_sglang_settings() -> None:
    plan = plan_serve(
        _plan_input(
            settings={
                "cuda_graph_max_bs_decode": SettingValue(root=32),
                "moe_runner_backend": SettingValue(root="flashinfer_mxfp4"),
            }
        )
    )
    result = render_serve(_render_input(settings=plan.roles[0].effective_settings))
    argv = result.processes[0].root.command.argv

    assert argv[argv.index("--cuda-graph-max-bs-decode") + 1] == "32"
    assert argv[argv.index("--moe-runner-backend") + 1] == "flashinfer_mxfp4"


def test_render_enables_cache_report_when_declared() -> None:
    plan = plan_serve(_plan_input(settings={"enable_cache_report": SettingValue(root=True)}))

    result = render_serve(_render_input(settings=plan.roles[0].effective_settings))

    assert "--enable-cache-report" in result.processes[0].root.command.argv
    endpoint = plan.roles[0].public_endpoint
    assert endpoint is not None
    assert (
        endpoint.prompt_cache_read_zero_representation is PromptCacheReadZeroRepresentation.omitted
    )


@pytest.mark.parametrize("transport", [KvTransferMechanism.mooncake, KvTransferMechanism.nixl])
def test_render_prefill_decode_lowers_transport_independently_from_routing(
    transport: KvTransferMechanism,
) -> None:
    result = render_serve(_prefill_decode_render_input(transport=transport))

    for wrapped in result.processes[:2]:
        process = wrapped.root
        argv = process.command.argv
        assert argv[argv.index("--disaggregation-mode") + 1] == "prefill"
        assert argv[argv.index("--disaggregation-transfer-backend") + 1] == transport.value
        assert argv[argv.index("--disaggregation-bootstrap-port") + 1] in {"9000", "9001"}
    for wrapped in result.processes[2:]:
        process = wrapped.root
        argv = process.command.argv
        assert argv[argv.index("--disaggregation-mode") + 1] == "decode"
        assert argv[argv.index("--disaggregation-transfer-backend") + 1] == transport.value
        assert "--disaggregation-bootstrap-port" not in argv


def test_render_sglang_router_targets_every_replica_entrypoint() -> None:
    result = render_serve(_prefill_decode_render_input(frontend_backend="sglang-router"))

    argv = result.processes[-1].root.command.argv
    assert argv[:3] == ["python3", "-m", "sglang_router.launch_router"]
    assert "--pd-disaggregation" in argv
    assert "--mini-lb" not in argv
    assert [argv[index + 1] for index, arg in enumerate(argv) if arg == "--prefill"] == [
        "http://node-0.example:8000",
        "http://node-1.example:8000",
    ]
    assert [argv[index + 2] for index, arg in enumerate(argv) if arg == "--prefill"] == [
        "9000",
        "9001",
    ]
    assert [argv[index + 1] for index, arg in enumerate(argv) if arg == "--decode"] == [
        "http://node-2.example:8000",
        "http://node-3.example:8000",
    ]
    assert argv[argv.index("--policy") + 1] == "round_robin"
    assert argv[argv.index("--worker-startup-timeout-secs") + 1] == "2147483647"


def test_render_rejects_multi_node() -> None:
    allocation_payload = _render_input().allocations[0].model_dump()
    allocation_payload.update(
        {
            "process": "server-rank-000",
            "rank_count": 2,
            "machine": "node-a",
            "devices": [0],
        }
    )
    second_payload = {
        **allocation_payload,
        "process": "server-rank-001",
        "rank": 1,
        "machine": "node-b",
    }
    allocation = ServeProcessAllocation.model_validate(allocation_payload)
    second = ServeProcessAllocation.model_validate(second_payload)
    with pytest.raises(AdapterOperationError):
        render_serve(_render_input(allocations=[allocation, second]))
