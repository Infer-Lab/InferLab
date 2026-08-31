import json
from importlib.metadata import PackageNotFoundError
from io import StringIO
from pathlib import Path
from typing import cast

import inferlab_adapter_sdk
import inferlab_adapter_sdk.runtime as adapter_runtime
import pytest
from inferlab_adapter_sdk import (
    AdapterErrorCode,
    AdapterOperationError,
    AdapterRequest,
    AdapterRequestPlanServe,
    AdapterResponse,
    EndpointProtocol,
    EndpointRequirement,
    IntegrationIdentity,
    LaunchFileDeclaration,
    PlanServeInput,
    PlanServeResult,
    ProcessSpec,
    ReadinessProbe,
    ReadinessProbeHttp,
    ReadinessProbeHttpTargetRegistry,
    ReadinessProbeProcessAlive,
    RenderedServeProcessFrontend,
    RenderedServeProcessModelRank,
    RenderInputDeclaration,
    RenderSource,
    ServeProcessAllocationFrontend,
    ServeReplicaRequirement,
    ServeRoleKind,
    ServeRoleResult,
    SettingValue,
    SuppliedRenderInput,
    TargetEndpointScheme,
    effective_settings,
    fused_pd_frontend_plans,
    handle_request,
    integration_identity,
    rendered_frontend,
    rendered_model_rank,
    replica_id,
    require_integration_fused_frontend,
    require_role,
    run_adapter,
    split_serve_allocations,
    validate_settings,
)
from inferlab_adapter_sdk._generated import (
    AdapterRequestRenderServe,
    AdapterResponseError,
    AdapterResponseOk,
    AdapterResultPlanServe,
    AdapterResultRenderServe,
)
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError as JsonSchemaValidationError
from pydantic import BaseModel, ConfigDict
from pydantic import ValidationError as PydanticValidationError

ROOT = Path(__file__).parents[3]
FIXTURES = ROOT / "protocol" / "fixtures"
SCHEMA = ROOT / "protocol" / "schema" / "adapter-protocol-v9.schema.json"


class FixtureSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    port: int


def load_json(path: Path) -> dict[str, object]:
    return cast(dict[str, object], json.loads(path.read_text()))


def test_public_sdk_excludes_measurement_models_and_runtime() -> None:
    assert not hasattr(inferlab_adapter_sdk, "BenchClientRequest")
    assert not hasattr(inferlab_adapter_sdk, "EvalClientRequest")
    assert not hasattr(inferlab_adapter_sdk, "CaseDeadline")


def test_runtime_owns_shared_settings_translation() -> None:
    settings = validate_settings(
        FixtureSettings,
        {"port": SettingValue(root=8000)},
    )

    assert settings.port == 8000
    assert effective_settings(settings) == {"port": SettingValue(root=8000)}
    with pytest.raises(AdapterOperationError) as raised:
        validate_settings(FixtureSettings, {"unknown": SettingValue(root=True)})
    assert raised.value.code == AdapterErrorCode.invalid_settings


def test_runtime_owns_package_identity_and_role_conventions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def distribution_version(distribution: str) -> str:
        if distribution == "fixture-adapter":
            return "1.2.3"
        raise PackageNotFoundError(distribution)

    monkeypatch.setattr(adapter_runtime, "version", distribution_version)
    identity = integration_identity(
        adapter_id="fixture",
        adapter_distribution="fixture-adapter",
        framework="fixture-framework",
        framework_distribution="inferlab-definitely-missing-framework",
    )
    request = AdapterRequest.model_validate(
        load_json(FIXTURES / "valid" / "plan-serve-request.json")
    )
    root = request.root
    assert isinstance(root, AdapterRequestPlanServe)
    plan_input = root.input
    role = require_role(plan_input, ServeRoleKind.prefill)

    assert identity.adapter_version == "1.2.3"
    assert identity.framework_version == "unavailable"
    assert replica_id(role, 0) == "prefill"


def fixture_plan_serve(input: PlanServeInput) -> PlanServeResult:
    return PlanServeResult(
        integration=IntegrationIdentity(
            adapter_id="fixture",
            adapter_version="0.1.0",
            framework="fixture",
            framework_version="test",
        ),
        roles=[
            ServeRoleResult(
                id="serve",
                kind=ServeRoleKind.serve,
                declared_replica_count=1,
                effective_replica_count=1,
                effective_settings=input.roles[0].settings,
                effective_parallelism=input.roles[0].parallelism,
                public_endpoint=EndpointRequirement(
                    protocol=EndpointProtocol(),
                    completions_path="/v1/completions",
                    chat_completions_path="/v1/chat/completions",
                ),
            )
        ],
        replicas=[
            ServeReplicaRequirement(
                id="server",
                role_id="serve",
                replica_index=0,
                device_count=1,
                ports=[],
                primary_ports=[],
                primary_readiness=ReadinessProbe(root=ReadinessProbeHttp(path="/ready")),
                worker_readiness=ReadinessProbe(root=ReadinessProbeProcessAlive()),
            )
        ],
        links=[],
    )


def test_generated_models_accept_shared_valid_fixtures() -> None:
    AdapterRequest.model_validate(load_json(FIXTURES / "valid" / "plan-serve-request.json"))
    plan_response = AdapterResponse.model_validate(
        load_json(FIXTURES / "valid" / "plan-serve-response.json")
    )
    render_request = AdapterRequest.model_validate(
        load_json(FIXTURES / "valid" / "render-serve-request.json")
    )
    AdapterResponse.model_validate(load_json(FIXTURES / "valid" / "render-serve-response.json"))
    AdapterResponse.model_validate(load_json(FIXTURES / "valid" / "error-response.json"))

    assert isinstance(plan_response.root, AdapterResponseOk)
    plan_result = plan_response.root.result.root
    assert isinstance(plan_result, AdapterResultPlanServe)
    assert plan_result.output.gateway is not None
    assert plan_result.output.gateway.backend == "vllm-router"
    assert plan_result.output.pd_router is not None
    assert plan_result.output.pd_router.backend == "vllm-router"
    assert isinstance(render_request.root, AdapterRequestRenderServe)
    frontend = render_request.root.input.allocations[2].root
    assert isinstance(frontend, ServeProcessAllocationFrontend)
    assert frontend.components.model_dump() == ("gateway", "pd_router")
    assert not hasattr(frontend, "model_locator")


def test_sdk_owns_fused_frontend_and_allocation_identity_invariants() -> None:
    request = AdapterRequest.model_validate(
        load_json(FIXTURES / "valid" / "render-serve-request.json")
    )
    assert isinstance(request.root, AdapterRequestRenderServe)
    allocations, model_allocations = split_serve_allocations(request.root.input.allocations)
    assert len(model_allocations) == 2
    frontend = allocations[-1]
    pd_router = require_integration_fused_frontend(
        frontend,
        gateway_backend="vllm-router",
        pd_router_backend="vllm-router",
    )
    assert pd_router.policies.prefill == "round_robin"

    model_process = rendered_model_rank(model_allocations[0], ProcessSpec(argv=["engine"], env={}))
    frontend_process = rendered_frontend(frontend, ProcessSpec(argv=["gateway"], env={}))
    assert isinstance(model_process.root, RenderedServeProcessModelRank)
    assert isinstance(frontend_process.root, RenderedServeProcessFrontend)
    assert frontend_process.root.components.model_dump() == (
        "gateway",
        "pd_router",
    )


def test_sdk_constructs_both_fused_component_plans_from_one_binding() -> None:
    response = AdapterResponse.model_validate(
        load_json(FIXTURES / "valid" / "plan-serve-response.json")
    )
    assert isinstance(response.root, AdapterResponseOk)
    result = response.root.result.root
    assert isinstance(result, AdapterResultPlanServe)
    expected_gateway = result.output.gateway
    expected_pd_router = result.output.pd_router
    assert expected_gateway is not None
    assert expected_pd_router is not None

    gateway, pd_router = fused_pd_frontend_plans(
        gateway_backend="vllm-router",
        pd_router_backend="vllm-router",
        implementation="vllm-router",
        implementation_version=expected_gateway.implementation_version,
        render_source=RenderSource.integration,
        endpoint=expected_gateway.endpoint,
        gateway_readiness=expected_gateway.readiness,
        pd_router_readiness=expected_pd_router.readiness,
        policies=expected_pd_router.policies,
        prefill_role="prefill",
        decode_role="decode",
        target_scheme=TargetEndpointScheme.http,
    )

    assert gateway == expected_gateway
    assert pd_router == expected_pd_router


def test_generated_models_preserve_rendered_launch_files() -> None:
    response = AdapterResponse.model_validate(
        load_json(FIXTURES / "valid" / "render-serve-response-launch-file.json")
    )
    assert isinstance(response.root, AdapterResponseOk)
    result = response.root.result.root
    assert isinstance(result, AdapterResultRenderServe)
    process = result.output.processes[0].root
    assert isinstance(process, RenderedServeProcessModelRank)
    launch_file = process.launch_files[0]

    assert isinstance(launch_file, LaunchFileDeclaration)
    assert launch_file.relative_path.endswith("/generation.yaml")
    assert launch_file.text == "generation_config:\n  temperature: 0.0\n"
    assert launch_file.sha256 == "2bcf56a7e1129e7b0dfbe7ef153a720f020a3dd076700069f9efe53ad9a6d281"


def test_generated_models_preserve_render_inputs() -> None:
    declaration = RenderInputDeclaration.model_validate(
        load_json(FIXTURES / "valid" / "render-input-declaration.json")
    )
    supplied = SuppliedRenderInput.model_validate(
        load_json(FIXTURES / "valid" / "supplied-render-input.json")
    )

    assert declaration.source_path == "configs/operator.yaml"
    assert supplied.source_path == declaration.source_path
    assert supplied.text == "batch_scheduler:\n  enable_chunked_context: true\n"
    assert supplied.sha256 == "898caa1654c13bd4b1f2eba75d17c09b8fc3ea1370e5532a5111be220d50baa3"


def test_generated_models_preserve_http_target_registry_readiness() -> None:
    readiness = ReadinessProbe.model_validate(
        load_json(FIXTURES / "valid" / "http-target-registry-readiness.json")
    ).root

    assert isinstance(readiness, ReadinessProbeHttpTargetRegistry)
    assert readiness.target_scheme == TargetEndpointScheme.http
    assert readiness.readiness_path == "/readiness"
    assert readiness.registry_path == "/workers"
    assert readiness.prefill_bootstrap_port == "bootstrap"


@pytest.mark.parametrize(
    ("model", "fixture"),
    [
        (AdapterRequest, "request-unknown-field.json"),
        (AdapterResponse, "response-wrong-shape.json"),
    ],
)
def test_generated_models_reject_shared_invalid_fixtures(
    model: type[AdapterRequest] | type[AdapterResponse], fixture: str
) -> None:
    with pytest.raises(PydanticValidationError):
        model.model_validate(load_json(FIXTURES / "invalid" / fixture))


def test_generated_schema_classifies_shared_fixtures() -> None:
    request = load_json(FIXTURES / "valid" / "plan-serve-request.json")
    response = load_json(FIXTURES / "valid" / "plan-serve-response.json")
    validator = Draft202012Validator(load_json(SCHEMA))

    validator.validate({"request": request, "response": response})
    with pytest.raises(JsonSchemaValidationError):
        validator.validate(
            {
                "request": load_json(FIXTURES / "invalid" / "request-unknown-field.json"),
                "response": response,
            }
        )


def test_runtime_returns_typed_success_and_error_responses() -> None:
    valid = (FIXTURES / "valid" / "plan-serve-request.json").read_text()

    success = handle_request(valid, fixture_plan_serve)
    failure = handle_request("{}", fixture_plan_serve)

    assert success.root.status == "ok"

    response_error = failure.root
    assert isinstance(response_error, AdapterResponseError)
    assert response_error.status == "error"
    assert response_error.error.code == AdapterErrorCode.invalid_request
    assert response_error.error.message


@pytest.mark.parametrize("complete_body", [True, False])
def test_unsupported_request_protocol_version_is_reported_before_shape(
    complete_body: bool,
) -> None:
    request: dict[str, object] = {"protocol_version": "2"}
    if complete_body:
        valid = load_json(FIXTURES / "valid" / "plan-serve-request.json")
        request = {**valid, "protocol_version": "2"}

    response = handle_request(json.dumps(request), fixture_plan_serve)

    response_error = response.root
    assert isinstance(response_error, AdapterResponseError)
    assert response_error.error.code == AdapterErrorCode.unsupported_protocol_version


def test_protocol_v8_request_is_rejected_instead_of_partially_interpreted() -> None:
    # The fixture is a well-formed protocol-v8 plan request carrying the
    # synthetic acceptance member; protocol v9 MUST reject it outright rather
    # than partially interpret it ([[RFC-0006:C-INTEGRATIONS]]).
    payload = (FIXTURES / "invalid" / "request-protocol-version-8.json").read_text()

    response = handle_request(payload, fixture_plan_serve)

    response_error = response.root
    assert isinstance(response_error, AdapterResponseError)
    assert response_error.error.code == AdapterErrorCode.unsupported_protocol_version
    assert "received protocol version 8" in response_error.error.message
    assert "protocol version 9" in response_error.error.message


def test_malformed_request_json_stays_invalid_request() -> None:
    response = handle_request("{not json", fixture_plan_serve)
    response_error = response.root
    assert isinstance(response_error, AdapterResponseError)
    assert response_error.error.code == AdapterErrorCode.invalid_request


def test_stdio_runner_writes_only_protocol_json() -> None:
    source = StringIO((FIXTURES / "valid" / "plan-serve-request.json").read_text())
    destination = StringIO()
    diagnostics = StringIO()

    assert (
        run_adapter(
            fixture_plan_serve,
            input_stream=source,
            output_stream=destination,
            diagnostics_stream=diagnostics,
        )
        == 0
    )
    AdapterResponse.model_validate_json(destination.getvalue())
    assert diagnostics.getvalue() == ""
