use inferlab_protocol::{
    AdapterRequest, AdapterResponse, AdapterResult, BenchClientRequest, BenchRequestSourceInput,
    EvalClientRequest, EvalClientResult, EvalDefinitionInput, EvalFailureKind,
    EvalMetricComparison, EvalMetricGateConclusion, EvalTaskSourceInput, MEASUREMENT_SCHEMA_ID,
    PROTOCOL_SCHEMA_ID, ProtocolVersion, ReadinessProbe, RenderInputDeclaration, SettingValue,
    SuppliedRenderInput, TargetEndpointScheme, measurement_schema, protocol_schema,
};
use std::error::Error;
use std::path::Path;

const VALID_PLAN_REQUEST: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../protocol/fixtures/valid/plan-serve-request.json"
));
const VALID_PLAN_RESPONSE: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../protocol/fixtures/valid/plan-serve-response.json"
));
const VALID_RENDER_REQUEST: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../protocol/fixtures/valid/render-serve-request.json"
));
const VALID_RENDER_RESPONSE: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../protocol/fixtures/valid/render-serve-response.json"
));
const VALID_LAUNCH_FILE_RESPONSE: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../protocol/fixtures/valid/render-serve-response-launch-file.json"
));
const VALID_ERROR_RESPONSE: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../protocol/fixtures/valid/error-response.json"
));
const INVALID_REQUEST: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../protocol/fixtures/invalid/request-unknown-field.json"
));
const INVALID_RESPONSE: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../protocol/fixtures/invalid/response-wrong-shape.json"
));
const VALID_HTTP_TARGET_REGISTRY_READINESS: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../protocol/fixtures/valid/http-target-registry-readiness.json"
));
const VALID_RENDER_INPUT_DECLARATION: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../protocol/fixtures/valid/render-input-declaration.json"
));
const VALID_SUPPLIED_RENDER_INPUT: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../protocol/fixtures/valid/supplied-render-input.json"
));
const VALID_EVAL_CLIENT_REQUEST_WORKSPACE_YAML: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../protocol/fixtures/valid/eval-client-request-workspace-yaml.json"
));
const VALID_EVAL_CLIENT_REQUEST_BUNDLED: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../protocol/fixtures/valid/eval-client-request-bundled.json"
));
const VALID_EVAL_CLIENT_RESULT_PROBE_FAILURE: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../protocol/fixtures/valid/eval-client-result-probe-failure.json"
));
const VALID_EVAL_CLIENT_RESULT_NORMALIZED_METRIC: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../protocol/fixtures/valid/eval-client-result-normalized-metric.json"
));
const VALID_BENCH_CLIENT_REQUEST_RANDOM_MIXTURE: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../protocol/fixtures/valid/bench-client-request-random-mixture.json"
));
const GENERATED_ADAPTER_SCHEMA: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../protocol/schema/adapter-protocol-v7.schema.json"
));
const GENERATED_MEASUREMENT_SCHEMA: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../protocol/schema/measurement-protocol-v1.schema.json"
));

#[test]
fn protocol_v6_requests_are_rejected_instead_of_partially_interpreted() {
    let request = r#"{
        "operation": "plan_serve",
        "protocol_version": "6",
        "input": {
            "model": {"id": "model", "served_name": "model"},
            "topology": "single",
            "roles": [],
            "profiling": false
        }
    }"#;

    assert!(serde_json::from_str::<AdapterRequest>(request).is_err());
}

#[test]
fn weighted_random_mixture_fixture_round_trips() -> Result<(), Box<dyn Error>> {
    let request: BenchClientRequest =
        serde_json::from_str(VALID_BENCH_CLIENT_REQUEST_RANDOM_MIXTURE)?;
    let BenchRequestSourceInput::RandomMixture {
        shapes,
        total_weight,
        ..
    } = request
        .definition
        .request_source
        .as_ref()
        .ok_or("Bench fixture omitted its request source")?
    else {
        return Err("Bench fixture did not contain a random mixture".into());
    };

    assert_eq!(shapes.len(), 2);
    assert_eq!(*total_weight, 10);
    assert_eq!(
        serde_json::from_str::<BenchClientRequest>(&serde_json::to_string(&request)?)?,
        request
    );
    Ok(())
}

#[test]
fn protocol_v7_rejects_the_pre_binding_capture_control_shape() -> Result<(), Box<dyn Error>> {
    let mut response: serde_json::Value = serde_json::from_str(VALID_PLAN_RESPONSE)?;
    let capture_target = response
        .pointer_mut("/result/output/replicas/0/capture_target")
        .ok_or("plan fixture did not contain a capture target")?;
    *capture_target = serde_json::json!({
        "control": {
            "start_path": "/start_profile",
            "stop_path": "/stop_profile"
        }
    });

    let Err(error) = serde_json::from_value::<AdapterResponse>(response) else {
        return Err("protocol v7 accepted the pre-binding capture-control shape".into());
    };
    assert!(error.to_string().contains("unknown field `control`"));
    Ok(())
}

#[test]
fn protocol_v7_preserves_a_typed_capture_action_body() -> Result<(), Box<dyn Error>> {
    let mut response: serde_json::Value = serde_json::from_str(VALID_PLAN_RESPONSE)?;
    response["result"]["output"]["replicas"][0]["capture_target"]["window_control"]["start"]["body"] =
        serde_json::json!({"activities": ["CUDA_PROFILER"]});
    let response: AdapterResponse = serde_json::from_value(response)?;
    let AdapterResponse::Ok { result, .. } = response else {
        return Err("plan fixture returned an error response".into());
    };
    let AdapterResult::PlanServe { output } = *result else {
        return Err("plan fixture returned a render result".into());
    };
    let body = output.replicas[0]
        .capture_target
        .as_ref()
        .and_then(|target| target.window_control.start.body.as_ref())
        .ok_or("plan fixture did not preserve the start action body")?;

    assert_eq!(
        serde_json::to_value(body)?,
        serde_json::json!({"activities": ["CUDA_PROFILER"]})
    );
    Ok(())
}

#[test]
fn protocol_v7_does_not_attach_capture_bodies_to_prefix_cache_actions() -> Result<(), Box<dyn Error>>
{
    let mut response: serde_json::Value = serde_json::from_str(VALID_PLAN_RESPONSE)?;
    response["result"]["output"]["roles"][0]["public_endpoint"]["prefix_cache_reset"] = serde_json::json!({
        "method": "post",
        "path": "/reset_prefix_cache",
        "body": {"activities": ["CUDA_PROFILER"]}
    });

    let Err(error) = serde_json::from_value::<AdapterResponse>(response) else {
        return Err("protocol v7 accepted a capture body on a prefix-cache action".into());
    };
    assert!(error.to_string().contains("unknown field `body`"));
    Ok(())
}

#[test]
fn frontend_component_schema_uses_stable_binding_names() -> Result<(), Box<dyn Error>> {
    let schema: serde_json::Value = serde_json::from_str(GENERATED_ADAPTER_SCHEMA)?;
    let definitions = schema["$defs"]
        .as_object()
        .ok_or("protocol schema did not contain definitions")?;

    assert!(definitions.contains_key("GatewayFrontendBinding"));
    assert!(definitions.contains_key("GatewayPdRouterFrontendBinding"));
    assert!(!definitions.contains_key("FrontendComponents1"));
    assert!(!definitions.contains_key("FrontendComponents2"));
    Ok(())
}

#[test]
fn valid_fixtures_deserialize_and_round_trip() -> Result<(), Box<dyn Error>> {
    let plan_request: AdapterRequest = serde_json::from_str(VALID_PLAN_REQUEST)?;
    let plan_response: AdapterResponse = serde_json::from_str(VALID_PLAN_RESPONSE)?;
    let render_request: AdapterRequest = serde_json::from_str(VALID_RENDER_REQUEST)?;
    let render_response: AdapterResponse = serde_json::from_str(VALID_RENDER_RESPONSE)?;
    let launch_file_response: AdapterResponse = serde_json::from_str(VALID_LAUNCH_FILE_RESPONSE)?;
    let error_response: AdapterResponse = serde_json::from_str(VALID_ERROR_RESPONSE)?;

    assert_eq!(plan_request.protocol_version(), ProtocolVersion::V7);
    assert_eq!(plan_response.protocol_version(), ProtocolVersion::V7);
    assert_eq!(render_request.protocol_version(), ProtocolVersion::V7);
    assert_eq!(render_response.protocol_version(), ProtocolVersion::V7);
    assert_eq!(error_response.protocol_version(), ProtocolVersion::V7);

    let AdapterResponse::Ok { result, .. } = &plan_response else {
        return Err("plan fixture did not contain a successful response".into());
    };
    let AdapterResult::PlanServe { output } = result.as_ref() else {
        return Err("plan fixture did not contain plan output".into());
    };
    let gateway = output
        .gateway
        .as_ref()
        .ok_or("plan fixture did not contain Gateway")?;
    assert_eq!(gateway.backend, "vllm-router");
    assert_eq!(gateway.endpoint.completions_path, "/v1/completions");
    assert_eq!(
        gateway.endpoint.chat_completions_path,
        "/v1/chat/completions"
    );
    let pd_router = output
        .pd_router
        .as_ref()
        .ok_or("plan fixture did not contain P/D Router")?;
    assert_eq!(pd_router.backend, "vllm-router");
    assert_eq!(pd_router.policies.prefill, "round_robin");

    let AdapterRequest::RenderServe { input, .. } = &render_request else {
        return Err("render fixture did not contain a render request".into());
    };
    let render_json = serde_json::to_value(input)?;
    let allocation = render_json["allocations"][0]
        .as_object()
        .ok_or("render fixture did not contain an allocation object")?;
    assert!(allocation.contains_key("effective_settings"));
    assert!(allocation.contains_key("effective_parallelism"));
    let frontend = render_json["allocations"][2]
        .as_object()
        .ok_or("render fixture did not contain a frontend allocation")?;
    assert_eq!(
        frontend.get("components"),
        Some(&serde_json::json!(["gateway", "pd_router"]))
    );
    assert!(!frontend.contains_key("model_locator"));
    assert!(!frontend.contains_key("replica"));
    assert!(!frontend.contains_key("rank"));

    let AdapterResponse::Ok { result, .. } = &launch_file_response else {
        return Err("launch-file fixture did not contain a successful response".into());
    };
    let AdapterResult::RenderServe { output } = result.as_ref() else {
        return Err("render fixture did not contain render output".into());
    };
    let inferlab_protocol::RenderedServeProcess::ModelRank { launch_files, .. } =
        &output.processes[0]
    else {
        return Err("launch-file fixture did not contain a model-rank process".into());
    };
    let launch_file = launch_files
        .first()
        .ok_or("render fixture did not contain a launch file")?;
    assert_eq!(
        launch_file.relative_path,
        "launch-files/2bcf56a7e1129e7b0dfbe7ef153a720f020a3dd076700069f9efe53ad9a6d281/generation.yaml"
    );
    assert_eq!(
        launch_file.sha256,
        "2bcf56a7e1129e7b0dfbe7ef153a720f020a3dd076700069f9efe53ad9a6d281"
    );
    assert_eq!(launch_file.text, "generation_config:\n  temperature: 0.0\n");
    assert_eq!(
        serde_json::from_str::<AdapterRequest>(&serde_json::to_string(&plan_request)?)?,
        plan_request
    );
    assert_eq!(
        serde_json::from_str::<AdapterRequest>(&serde_json::to_string(&render_request)?)?,
        render_request
    );
    assert_eq!(
        serde_json::from_str::<AdapterResponse>(&serde_json::to_string(&plan_response)?)?,
        plan_response
    );
    assert_eq!(
        serde_json::from_str::<AdapterResponse>(&serde_json::to_string(&render_response)?)?,
        render_response
    );
    assert_eq!(
        serde_json::from_str::<AdapterResponse>(&serde_json::to_string(&launch_file_response)?)?,
        launch_file_response
    );
    assert_eq!(
        serde_json::from_str::<AdapterResponse>(&serde_json::to_string(&error_response)?)?,
        error_response
    );
    Ok(())
}

#[test]
fn http_target_registry_readiness_fixture_preserves_registry_contract() -> Result<(), Box<dyn Error>>
{
    let readiness: ReadinessProbe = serde_json::from_str(VALID_HTTP_TARGET_REGISTRY_READINESS)?;
    let ReadinessProbe::HttpTargetRegistry(registry) = readiness else {
        return Err("fixture did not deserialize as HTTP target-registry readiness".into());
    };
    let inferlab_protocol::HttpTargetRegistryReadiness {
        target_scheme,
        readiness_path,
        registry_path,
        targets_field,
        target_url_field,
        target_role_field,
        target_healthy_field,
        target_bootstrap_port_field,
        prefill_role_value,
        decode_role_value,
        prefill_bootstrap_port,
    } = *registry;

    assert_eq!(target_scheme, TargetEndpointScheme::Http);

    assert_eq!(
        (
            readiness_path.as_str(),
            registry_path.as_str(),
            targets_field.as_str(),
            target_url_field.as_str(),
            target_role_field.as_str(),
            target_healthy_field.as_str(),
            target_bootstrap_port_field.as_str(),
            prefill_role_value.as_str(),
            decode_role_value.as_str(),
            prefill_bootstrap_port.as_str(),
        ),
        (
            "/readiness",
            "/workers",
            "workers",
            "url",
            "worker_type",
            "is_healthy",
            "bootstrap_port",
            "prefill",
            "decode",
            "bootstrap",
        )
    );
    Ok(())
}

#[test]
fn render_input_fixtures_preserve_declared_path_and_supplied_text() -> Result<(), Box<dyn Error>> {
    let declaration: RenderInputDeclaration = serde_json::from_str(VALID_RENDER_INPUT_DECLARATION)?;
    let supplied: SuppliedRenderInput = serde_json::from_str(VALID_SUPPLIED_RENDER_INPUT)?;

    assert_eq!(declaration.source_path, "configs/operator.yaml");
    assert_eq!(supplied.source_path, declaration.source_path);
    assert_eq!(
        supplied.text,
        "batch_scheduler:\n  enable_chunked_context: true\n"
    );
    assert_eq!(
        supplied.sha256,
        "898caa1654c13bd4b1f2eba75d17c09b8fc3ea1370e5532a5111be220d50baa3"
    );
    Ok(())
}

#[test]
fn eval_client_fixture_preserves_workspace_yaml_task_source() -> Result<(), Box<dyn Error>> {
    let request: EvalClientRequest =
        serde_json::from_str(VALID_EVAL_CLIENT_REQUEST_WORKSPACE_YAML)?;
    let EvalDefinitionInput::LmEval {
        task,
        trials,
        metric_filter,
        request_body,
        ..
    } = request.definition
    else {
        return Err("fixture did not contain an lm-eval definition".into());
    };
    let EvalTaskSourceInput::WorkspaceYaml { path } = *task else {
        return Err("fixture did not contain a workspace YAML task source".into());
    };

    assert_eq!(request.protocol_version, ProtocolVersion::V7);
    assert_eq!(request.endpoint.completions_path, "/v1/completions");
    assert_eq!(
        request.endpoint.chat_completions_path,
        "/v1/chat/completions"
    );
    assert_eq!(path, Path::new("/workspace/evals/custom.yaml"));
    assert_eq!(metric_filter.as_deref(), Some("strict-match"));
    assert_eq!(trials, 3);
    assert_eq!(
        request_body.get("reasoning_effort"),
        Some(&SettingValue::String("high".to_owned()))
    );
    assert!(matches!(
        request_body.get("chat_template_kwargs"),
        Some(SettingValue::Object(values))
            if values.get("enable_thinking") == Some(&SettingValue::Bool(true))
    ));
    Ok(())
}

#[test]
fn eval_client_fixture_preserves_bundled_task_identity() -> Result<(), Box<dyn Error>> {
    let request: EvalClientRequest = serde_json::from_str(VALID_EVAL_CLIENT_REQUEST_BUNDLED)?;
    let EvalDefinitionInput::LmEval { task, .. } = request.definition else {
        return Err("fixture did not contain an lm-eval definition".into());
    };
    let EvalTaskSourceInput::Bundled {
        name,
        task_identity,
        task_closure_sha256,
        ..
    } = *task
    else {
        return Err("fixture did not contain a bundled task source".into());
    };

    assert_eq!(name, "estonia");
    assert_eq!(task_identity, "inferlab_estonia");
    assert_eq!(task_closure_sha256.len(), 64);
    Ok(())
}

#[test]
fn eval_client_result_fixture_preserves_typed_probe_failure() -> Result<(), Box<dyn Error>> {
    let result: EvalClientResult = serde_json::from_str(VALID_EVAL_CLIENT_RESULT_PROBE_FAILURE)?;

    assert_eq!(
        result.failure_kind,
        Some(EvalFailureKind::ProbeGeneratedOnlyLogprobs)
    );
    assert_eq!(result.raw_artifacts[0].kind, "prompt-logprob-probe");
    Ok(())
}

#[test]
fn eval_client_result_fixture_preserves_metric_gate_provenance() -> Result<(), Box<dyn Error>> {
    let result: EvalClientResult =
        serde_json::from_str(VALID_EVAL_CLIENT_RESULT_NORMALIZED_METRIC)?;
    let metric = result
        .normalized_metrics
        .get("gsm8k:exact_match,strict-match")
        .ok_or("normalized metric fixture had no metric")?;
    assert_eq!(metric.source_identity, "gsm8k");
    assert_eq!(metric.native_metric_key, "exact_match,strict-match");
    let gate = result.gate.ok_or("normalized metric fixture had no gate")?;
    assert_eq!(gate.comparison, EvalMetricComparison::AtLeast);
    assert_eq!(gate.conclusion, EvalMetricGateConclusion::Passed);
    assert_eq!(result.native_exit_code, Some(0));
    assert!(!result.native_timed_out);
    let summary = result
        .trial_summary
        .ok_or("normalized metric fixture had no trial summary")?;
    assert_eq!(summary.requested_trials, 3);
    assert_eq!(summary.passed_trials, 2);
    Ok(())
}

#[test]
fn invalid_fixtures_are_rejected() -> Result<(), Box<dyn Error>> {
    assert!(serde_json::from_str::<AdapterRequest>(INVALID_REQUEST).is_err());
    assert!(serde_json::from_str::<AdapterResponse>(INVALID_RESPONSE).is_err());
    Ok(())
}

#[test]
fn generated_schemas_are_current_versioned_and_disjoint() -> Result<(), Box<dyn Error>> {
    let mut adapter_rendered = serde_json::to_string_pretty(&protocol_schema())?;
    adapter_rendered.push('\n');
    let mut measurement_rendered = serde_json::to_string_pretty(&measurement_schema())?;
    measurement_rendered.push('\n');

    assert_eq!(adapter_rendered, GENERATED_ADAPTER_SCHEMA);
    assert_eq!(measurement_rendered, GENERATED_MEASUREMENT_SCHEMA);
    let adapter_schema: serde_json::Value = serde_json::from_str(GENERATED_ADAPTER_SCHEMA)?;
    let measurement_schema: serde_json::Value = serde_json::from_str(GENERATED_MEASUREMENT_SCHEMA)?;
    assert_eq!(adapter_schema["$id"], PROTOCOL_SCHEMA_ID);
    assert_eq!(measurement_schema["$id"], MEASUREMENT_SCHEMA_ID);
    assert_eq!(
        adapter_schema["$schema"],
        "https://json-schema.org/draft/2020-12/schema"
    );
    assert_eq!(
        measurement_schema["$schema"],
        "https://json-schema.org/draft/2020-12/schema"
    );
    let adapter_definitions = adapter_schema["$defs"]
        .as_object()
        .ok_or("adapter protocol schema has no definitions")?;
    let measurement_definitions = measurement_schema["$defs"]
        .as_object()
        .ok_or("measurement protocol schema has no definitions")?;
    for adapter_type in [
        "AdapterRequest",
        "AdapterResponse",
        "PlanServeInput",
        "RenderServeInput",
    ] {
        assert!(
            adapter_definitions.contains_key(adapter_type),
            "adapter schema omitted {adapter_type}"
        );
    }
    for measurement_type in [
        "BenchClientRequest",
        "BenchClientResult",
        "EvalClientRequest",
        "EvalClientResult",
    ] {
        assert!(
            measurement_definitions.contains_key(measurement_type),
            "measurement schema omitted {measurement_type}"
        );
    }
    assert!(!adapter_definitions.contains_key("BenchClientRequest"));
    assert!(!adapter_definitions.contains_key("EvalClientRequest"));
    assert!(!measurement_definitions.contains_key("AdapterRequest"));
    assert!(!measurement_definitions.contains_key("AdapterResponse"));

    assert!(!GENERATED_ADAPTER_SCHEMA.contains("lower_bench"));
    assert!(GENERATED_ADAPTER_SCHEMA.contains("prefix_cache_reset"));
    assert!(GENERATED_ADAPTER_SCHEMA.contains("prefill_decode"));
    assert!(!GENERATED_ADAPTER_SCHEMA.contains("inferlab_builtin"));
    assert!(!GENERATED_ADAPTER_SCHEMA.contains("integration_native"));
    assert!(GENERATED_ADAPTER_SCHEMA.contains("gateway_backend"));
    assert!(GENERATED_ADAPTER_SCHEMA.contains("pd_router_backend"));
    assert!(GENERATED_ADAPTER_SCHEMA.contains("render_source"));
    assert!(GENERATED_ADAPTER_SCHEMA.contains("frontend"));
    assert!(GENERATED_ADAPTER_SCHEMA.contains("capture_target"));
    assert!(GENERATED_ADAPTER_SCHEMA.contains("window_control"));
    assert!(GENERATED_ADAPTER_SCHEMA.contains("replica_entry"));
    assert!(GENERATED_ADAPTER_SCHEMA.contains("http_target_registry"));
    assert!(GENERATED_ADAPTER_SCHEMA.contains("launch_files"));
    assert!(GENERATED_ADAPTER_SCHEMA.contains("render_inputs"));
    assert!(GENERATED_MEASUREMENT_SCHEMA.contains("random_mixture"));
    assert!(GENERATED_MEASUREMENT_SCHEMA.contains("prefix_sharing"));
    Ok(())
}
