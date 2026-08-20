//! Workspace aggregate facade. Portable definitions, machine-local bindings,
//! composition, realization checks, and source identity keep separate owners.

mod catalog_validation;
mod composition;
mod definitions;
mod local;
mod realization;
mod source;
mod state;

use crate::InferlabError;

pub(crate) use catalog_validation::{validate_bench, validate_eval, validate_eval_task_source};
pub(crate) use composition::{
    discover_workspace, load_workspace, load_workspace_config, workspace_summary,
};
pub(crate) use composition::{snapshot_workspace, workspace_identity};
#[cfg(test)]
pub(crate) use definitions::BenchRandomShape;
pub(crate) use definitions::{
    AggregateSlo, BenchAgenticSource, BenchArtifactLevel, BenchCacheStart, BenchDefinition,
    BenchRequestSource, BenchSessionSource, BenchTokenSelector, BenchTpotApplicability,
    EvalDefinition, EvalPrompt, EvalTaskSource, JsonValue, ModelDefinition, RecipeDefinition,
    RequestRate, RequestSlo, ServerCaseDefinition, ServerDefinition, StackDefinition,
    WorkloadSuiteDefinition, WorkspaceConfig,
};
pub(crate) use definitions::{
    BenchPrefixSharing, BenchPrompt, BenchPromptSelection, BenchSharedSystemContent,
};
pub(crate) use definitions::{
    DEFAULT_CAPTURE_ARM_DEADLINE_SECONDS, DEFAULT_CAPTURE_CONTROL_DEADLINE_SECONDS,
    DEFAULT_CAPTURE_FINALIZATION_DEADLINE_SECONDS, DEFAULT_READINESS_ATTEMPT_TIMEOUT_SECONDS,
};
#[cfg(test)]
pub(crate) use local::LocalBindings;
pub(crate) use local::MANAGED_CONTAINER_ENV;
pub(crate) use local::{
    AdapterBinding, BuilderKind, LaunchBinding, MachineBinding, ModelWeightBinding,
    PlacementBinding, PlacementRoleBinding,
};
pub(crate) use source::{
    git_status_flags, source_digest_script, source_pathspecs, workspace_mutations,
};
pub(crate) use state::{LoadedWorkspace, WorkspaceSnapshot};

#[cfg(test)]
use catalog_validation::*;
fn invalid<T>(message: String) -> Result<T, InferlabError> {
    Err(InferlabError::InvalidConfig { message })
}

#[cfg(test)]
mod tests {
    use super::definitions::ProfilerEscapes;
    use super::*;
    use inferlab_profiler::plan::NsysEscapes;
    use std::collections::BTreeMap;
    use std::path::PathBuf;

    fn validate_manifest(manifest: &str) -> Result<(), Box<dyn std::error::Error>> {
        let root = tempfile::tempdir()?;
        let config = toml::from_str::<WorkspaceConfig>(manifest)?;
        validate_workspace(root.path(), &config)?;
        Ok(())
    }

    #[test]
    fn prefill_decode_requires_both_frontend_components_on_the_server_base()
    -> Result<(), Box<dyn std::error::Error>> {
        let result = validate_manifest(
            r#"
schema_version = 2

[models.model]
served_name = "model"

[stacks.stack]
integration = "fixture"
pixi_environment = "fixture"

[servers.server]
stack = "stack"
model = "model"
topology = "prefill_decode"
readiness_timeout_seconds = 60

[servers.server.cases.add-frontend]
gateway_backend = "gateway"
pd_router_backend = "router"
"#,
        );
        let Err(error) = result else {
            return Err(std::io::Error::other(
                "a P/D case must not add frontend components absent from the server base",
            )
            .into());
        };

        assert!(
            error
                .to_string()
                .contains("prefill_decode server \"server\" must declare both gateway_backend and pd_router_backend"),
            "{error}"
        );
        Ok(())
    }

    #[test]
    fn single_case_cannot_add_a_gateway_absent_from_the_server_base()
    -> Result<(), Box<dyn std::error::Error>> {
        let result = validate_manifest(
            r#"
schema_version = 2

[models.model]
served_name = "model"

[stacks.stack]
integration = "fixture"
pixi_environment = "fixture"

[servers.server]
stack = "stack"
model = "model"
topology = "single"
readiness_timeout_seconds = 60

[servers.server.cases.add-gateway]
gateway_backend = "gateway"
"#,
        );
        let Err(error) = result else {
            return Err(std::io::Error::other(
                "a single case must not add a Gateway absent from the server base",
            )
            .into());
        };

        assert!(
            error.to_string().contains(
                "cannot add gateway_backend because the server base does not declare a Gateway"
            ),
            "{error}"
        );
        Ok(())
    }

    #[test]
    fn explicitly_declared_aggregate_slos_must_be_nonempty()
    -> Result<(), Box<dyn std::error::Error>> {
        let result = toml::from_str::<WorkspaceConfig>(
            r#"
schema_version = 2
[benches.invalid]
request_source = { kind = "random", prompt = { kind = "server_chat" }, input_tokens = 128, output_tokens = 32 }
aggregate_slos = []
concurrency = [1]
prompts_per_concurrency = 1
timeout_seconds = 60
"#,
        );
        let Err(error) = result else {
            return Err(std::io::Error::other(
                "an explicitly empty aggregate_slos declaration must be rejected",
            )
            .into());
        };

        assert!(error.to_string().contains("non-empty"), "{error}");
        Ok(())
    }

    #[test]
    fn ordinary_serving_authoring_resolves_to_the_explicit_canonical_definition()
    -> Result<(), Box<dyn std::error::Error>> {
        let explicit = toml::from_str::<WorkspaceConfig>(
            r#"
schema_version = 2
[benches.ordinary]
kind = "serving"
request_source = { kind = "random", prompt = { kind = "flat" }, input_tokens = { kind = "inclusive_uniform", min = 64, max = 128 }, output_tokens = 32 }
concurrency = [1, 8]
prompts_per_concurrency = 4
timeout_seconds = 900
"#,
        )?;
        let ordinary = toml::from_str::<WorkspaceConfig>(
            r#"
schema_version = 2
[benches.ordinary]
request_source = { kind = "random", input_tokens = { min = 64, max = 128 }, output_tokens = 32 }
concurrency = [1, 8]
prompts_per_concurrency = 4
timeout_seconds = 900
"#,
        )?;

        assert_eq!(
            serde_json::to_value(&ordinary.benches["ordinary"])?,
            serde_json::to_value(&explicit.benches["ordinary"])?,
            "authoring defaults must disappear into one canonical definition"
        );
        Ok(())
    }

    #[test]
    fn openai_smoke_omission_resolves_to_stable_effective_values()
    -> Result<(), Box<dyn std::error::Error>> {
        let definition = toml::from_str::<EvalDefinition>(r#"kind = "openai-smoke""#)?;
        let value = serde_json::to_value(&definition)?;

        assert_eq!(value["kind"], "openai-smoke");
        assert_eq!(value["prompt"], "Hello");
        assert_eq!(value["max_tokens"], 16);
        assert_eq!(value["timeout_seconds"], 60);
        Ok(())
    }

    #[test]
    fn dataset_request_source_is_one_valid_serving_bench_shape()
    -> Result<(), Box<dyn std::error::Error>> {
        let definition = toml::from_str::<BenchDefinition>(
            r#"
kind = "serving"
request_source = { kind = "dataset", dataset = "sharegpt", max_input_tokens = 8192 }
concurrency = [1]
prompts_per_concurrency = 2
timeout_seconds = 60
"#,
        )?;

        validate_bench("sharegpt", &definition)?;
        let BenchDefinition::Serving { request_source, .. } = definition else {
            return Err(std::io::Error::other("expected a serving Bench").into());
        };
        assert!(matches!(
            &request_source,
            Some(BenchRequestSource::Dataset {
                dataset,
                profile: None,
                max_input_tokens: 8192,
                output_tokens: None,
            }) if dataset == "sharegpt"
        ));
        let Some(request_source) = request_source else {
            return Err(std::io::Error::other("expected a request source").into());
        };
        assert_eq!(
            request_source.tpot_applicability(),
            BenchTpotApplicability::Applicable
        );
        assert_eq!(
            BenchRequestSource::Dataset {
                dataset: "sharegpt".to_owned(),
                profile: None,
                max_input_tokens: 8192,
                output_tokens: Some(1),
            }
            .tpot_applicability(),
            BenchTpotApplicability::Inapplicable
        );
        Ok(())
    }

    #[test]
    fn dataset_profile_is_a_release_catalog_identifier() -> Result<(), Box<dyn std::error::Error>> {
        let definition = toml::from_str::<BenchDefinition>(
            r#"
kind = "serving"
request_source = { kind = "dataset", dataset = "speed_bench", profile = "qualitative_coding", max_input_tokens = 8192, output_tokens = 4096 }
concurrency = [1]
prompts_per_concurrency = 2
timeout_seconds = 60
"#,
        )?;

        validate_bench("speed", &definition)?;
        let BenchDefinition::Serving { request_source, .. } = definition else {
            return Err(std::io::Error::other("expected a serving Bench").into());
        };
        let Some(BenchRequestSource::Dataset {
            dataset, profile, ..
        }) = request_source
        else {
            return Err(std::io::Error::other("expected a dataset request source").into());
        };
        assert_eq!(dataset, "speed_bench");
        assert_eq!(profile.as_deref(), Some("qualitative_coding"));
        Ok(())
    }

    #[test]
    fn dataset_profile_must_resolve_through_the_release_catalog()
    -> Result<(), Box<dyn std::error::Error>> {
        let definition = toml::from_str::<BenchDefinition>(
            r#"
kind = "serving"
request_source = { kind = "dataset", dataset = "speed_bench", profile = "made_up", max_input_tokens = 8192, output_tokens = 4096 }
concurrency = [1]
prompts_per_concurrency = 1
timeout_seconds = 60
"#,
        )?;

        let Err(error) = validate_bench("speed", &definition) else {
            return Err(std::io::Error::other("unknown catalog profile must fail").into());
        };
        assert!(error.to_string().contains("made_up"), "{error}");
        assert!(error.to_string().contains("catalog"), "{error}");
        Ok(())
    }

    #[test]
    fn random_request_source_accepts_bounded_uniform_token_selectors()
    -> Result<(), Box<dyn std::error::Error>> {
        let definition = toml::from_str::<BenchDefinition>(
            r#"
kind = "serving"
request_source = { kind = "random", prompt = { kind = "server_chat" }, input_tokens = { kind = "inclusive_uniform", min = 7000, max = 9000 }, output_tokens = { kind = "inclusive_uniform", min = 900, max = 1100 } }
concurrency = [1]
prompts_per_concurrency = 2
timeout_seconds = 60
"#,
        )?;

        validate_bench("uniform", &definition)?;
        Ok(())
    }

    #[test]
    fn uniform_random_rejects_mixed_tpot_and_accepts_distributed_prefix_input()
    -> Result<(), Box<dyn std::error::Error>> {
        let mixed = toml::from_str::<BenchDefinition>(
            r#"
kind = "serving"
request_source = { kind = "random", prompt = { kind = "server_chat" }, input_tokens = 128, output_tokens = { kind = "inclusive_uniform", min = 1, max = 2 } }
concurrency = [1]
prompts_per_concurrency = 1
timeout_seconds = 60
"#,
        )?;
        let shared = toml::from_str::<BenchDefinition>(
            r#"
kind = "serving"
request_source = { kind = "random", prompt = { kind = "flat" }, input_tokens = { kind = "inclusive_uniform", min = 64, max = 128 }, output_tokens = 32, prefix_sharing = { shared_prefix_ratio = 0.5 } }
concurrency = [1]
prompts_per_concurrency = 1
timeout_seconds = 60
"#,
        )?;

        let Err(mixed_error) = validate_bench("mixed-tpot", &mixed) else {
            return Err(std::io::Error::other("uniform OSL spanning one must fail").into());
        };
        validate_bench("uniform-prefix", &shared)?;
        assert!(mixed_error.to_string().contains("TPOT"), "{mixed_error}");
        Ok(())
    }

    #[test]
    fn serving_bench_rejects_a_public_chat_template_field() -> Result<(), Box<dyn std::error::Error>>
    {
        let Err(error) = toml::from_str::<BenchDefinition>(
            r#"
kind = "serving"
request_source = { kind = "random", prompt = { kind = "server_chat" }, input_tokens = 128, output_tokens = 32 }
chat_template = "templates/qwen.jinja"
concurrency = [1]
prompts_per_concurrency = 1
timeout_seconds = 60
"#,
        ) else {
            return Err(std::io::Error::other("chat_template must not be a Bench field").into());
        };
        assert!(error.to_string().contains("unknown field `chat_template`"));
        Ok(())
    }

    #[test]
    fn serving_bench_preserves_a_server_side_chat_template_request_member()
    -> Result<(), Box<dyn std::error::Error>> {
        let definition = toml::from_str::<BenchDefinition>(
            r#"
kind = "serving"
request_source = { kind = "random", prompt = { kind = "server_chat" }, input_tokens = 128, output_tokens = 32 }
concurrency = [1]
prompts_per_concurrency = 1
timeout_seconds = 60

[request_body]
chat_template = "{% for message in messages %}{{ message.content }}{% endfor %}"
"#,
        )?;

        validate_bench("server-template", &definition)?;
        let BenchDefinition::Serving { request_body, .. } = definition else {
            return Err(std::io::Error::other("fixture should be a serving Bench").into());
        };
        assert!(matches!(
            request_body.get("chat_template"),
            Some(JsonValue::String(value)) if value.contains("message.content")
        ));
        Ok(())
    }

    #[test]
    fn server_metrics_accepts_a_positive_native_warmup() -> Result<(), Box<dyn std::error::Error>> {
        let definition = toml::from_str::<BenchDefinition>(
            r#"
kind = "serving"
request_source = { kind = "random", prompt = { kind = "server_chat" }, input_tokens = 128, output_tokens = 32 }
server_metrics = true
concurrency = [1]
prompts_per_concurrency = 1
warmup_prompts_per_concurrency = 1
timeout_seconds = 60
"#,
        )?;
        validate_bench("metrics-warmup", &definition)?;
        Ok(())
    }

    #[test]
    fn serving_bench_accepts_closed_cache_starts_and_rejects_the_old_boolean()
    -> Result<(), Box<dyn std::error::Error>> {
        for start in ["uncontrolled", "cold", "primed"] {
            let definition = toml::from_str::<BenchDefinition>(&format!(
                r#"
kind = "serving"
request_source = {{ kind = "random", prompt = {{ kind = "flat" }}, input_tokens = 128, output_tokens = 32, prefix_sharing = {{ shared_prefix_tokens = 64 }} }}
concurrency = [1]
prompts_per_concurrency = 1
cache = {{ start = "{start}" }}
timeout_seconds = 60
"#
            ))?;
            validate_bench("cache", &definition)?;
        }

        let error = toml::from_str::<BenchDefinition>(
            r#"
kind = "serving"
request_source = { kind = "random", input_tokens = 128, output_tokens = 32 }
concurrency = [1]
prompts_per_concurrency = 1
reset_prefix_cache = true
timeout_seconds = 60
"#,
        )
        .err()
        .ok_or("the removed reset_prefix_cache field was accepted")?;
        assert!(error.to_string().contains("reset_prefix_cache"), "{error}");
        Ok(())
    }

    #[test]
    fn primed_cache_requires_positive_exact_prefix_geometry()
    -> Result<(), Box<dyn std::error::Error>> {
        let definition = toml::from_str::<BenchDefinition>(
            r#"
kind = "serving"
request_source = { kind = "random", prompt = { kind = "flat" }, input_tokens = 128, output_tokens = 32, prefix_sharing = { shared_prefix_tokens = 0 } }
concurrency = [1]
prompts_per_concurrency = 1
cache = { start = "primed" }
timeout_seconds = 60
"#,
        )?;
        let error = validate_bench("primed", &definition)
            .err()
            .ok_or("primed cache accepted a zero shared prefix")?;
        assert!(
            error.to_string().contains("positive prefix_sharing"),
            "{error}"
        );
        Ok(())
    }

    #[test]
    fn acceptance_slos_belong_only_to_speed_bench_server_metrics()
    -> Result<(), Box<dyn std::error::Error>> {
        let valid = toml::from_str::<BenchDefinition>(
            r#"
kind = "serving"
request_source = { kind = "dataset", dataset = "speed_bench", profile = "qualitative_coding", max_input_tokens = 8192, output_tokens = 128 }
server_metrics = true
aggregate_slos = [{ metric = "acceptance_rate", at_least = 0.5 }]
concurrency = [1]
prompts_per_concurrency = 1
timeout_seconds = 60
"#,
        )?;
        validate_bench("speed", &valid)?;

        let invalid = toml::from_str::<BenchDefinition>(
            r#"
kind = "serving"
request_source = { kind = "random", prompt = { kind = "server_chat" }, input_tokens = 128, output_tokens = 32 }
server_metrics = true
aggregate_slos = [{ metric = "acceptance_rate", at_least = 0.5 }]
concurrency = [1]
prompts_per_concurrency = 1
timeout_seconds = 60
"#,
        )?;
        let error = validate_bench("random", &invalid)
            .err()
            .ok_or("random acceptance-rate SLO was accepted")?;
        assert!(error.to_string().contains("speed_bench"), "{error}");
        Ok(())
    }

    #[test]
    fn random_request_source_accepts_one_shared_prefix_ratio()
    -> Result<(), Box<dyn std::error::Error>> {
        let definition = toml::from_str::<BenchDefinition>(
            r#"
kind = "serving"
request_source = { kind = "random", prompt = { kind = "flat" }, input_tokens = 8000, output_tokens = 1000, prefix_sharing = { shared_prefix_ratio = 0.75 } }
concurrency = [1]
prompts_per_concurrency = 2
timeout_seconds = 60
"#,
        )?;

        validate_bench("shared-prefix", &definition)?;
        let BenchDefinition::Serving { request_source, .. } = definition else {
            return Err(std::io::Error::other("expected a serving Bench").into());
        };
        assert!(matches!(
            request_source,
            Some(BenchRequestSource::Random {
                prompt,
                input_tokens: BenchTokenSelector::Fixed(8000),
                output_tokens: BenchTokenSelector::Fixed(1000),
                prefix_sharing: Some(BenchPrefixSharing::Ratio {
                    shared_prefix_ratio: 0.75,
                }),
                shared_system_content: None,
            }) if prompt.effective() == &BenchPrompt::Flat
        ));
        Ok(())
    }

    #[test]
    fn random_request_source_accepts_a_ratio_that_resolves_to_zero_shared_tokens()
    -> Result<(), Box<dyn std::error::Error>> {
        let definition = toml::from_str::<BenchDefinition>(
            r#"
kind = "serving"
request_source = { kind = "random", prompt = { kind = "flat" }, input_tokens = 1, output_tokens = 1, prefix_sharing = { shared_prefix_ratio = 0.5 } }
concurrency = [1]
prompts_per_concurrency = 1
timeout_seconds = 60
"#,
        )?;
        validate_bench("empty-prefix", &definition)?;
        Ok(())
    }

    #[test]
    fn synthetic_prompt_authority_and_prefix_geometry_validate_as_one_source_boundary()
    -> Result<(), Box<dyn std::error::Error>> {
        let rendered = toml::from_str::<BenchDefinition>(
            r#"
kind = "serving"
request_source = { kind = "random_mixture", prompt = { kind = "rendered_chat", chat_template = "{{ messages }}", chat_template_kwargs = { enable_thinking = false } }, shapes = [
  { input_tokens = 8, output_tokens = 2, weight = 1 },
  { input_tokens = 12, output_tokens = 2, weight = 1 },
], prefix_sharing = { shared_prefix_tokens = 8 } }
concurrency = [1]
prompts_per_concurrency = 1
timeout_seconds = 60
"#,
        )?;
        validate_bench("rendered", &rendered)?;

        let server_chat = toml::from_str::<BenchDefinition>(
            r#"
kind = "serving"
request_source = { kind = "random", prompt = { kind = "server_chat" }, input_tokens = { kind = "inclusive_uniform", min = 8, max = 12 }, output_tokens = 2, shared_system_content = { ratio = 0.5 } }
concurrency = [1]
prompts_per_concurrency = 1
timeout_seconds = 60
"#,
        )?;
        validate_bench("server-chat", &server_chat)?;

        let local_template_conflict = toml::from_str::<BenchDefinition>(
            r#"
kind = "serving"
request_source = { kind = "random", prompt = { kind = "flat" }, input_tokens = 8, output_tokens = 2 }
concurrency = [1]
prompts_per_concurrency = 1
timeout_seconds = 60
[request_body]
chat_template = "{{ messages }}"
"#,
        )?;
        let error = validate_bench("local-conflict", &local_template_conflict)
            .err()
            .ok_or("local prompt accepted a request-body chat template")?;
        assert!(
            error.to_string().contains("local rendering authority"),
            "{error}"
        );

        let default_prompt = toml::from_str::<BenchDefinition>(
            r#"
kind = "serving"
request_source = { kind = "random", input_tokens = 8, output_tokens = 2 }
concurrency = [1]
prompts_per_concurrency = 1
timeout_seconds = 60
"#,
        );
        let default_prompt = default_prompt?;
        let BenchDefinition::Serving { request_source, .. } = default_prompt else {
            return Err(std::io::Error::other("expected a serving Bench").into());
        };
        assert!(matches!(
            request_source,
            Some(BenchRequestSource::Random {
                prompt,
                ..
            }) if prompt.declared().is_none() && prompt.effective() == &BenchPrompt::Flat
        ));
        Ok(())
    }

    #[test]
    fn weighted_random_mixture_owns_exact_shapes_and_one_tpot_class()
    -> Result<(), Box<dyn std::error::Error>> {
        let definition = toml::from_str::<BenchDefinition>(
            r#"
kind = "serving"
request_source = { kind = "random_mixture", prompt = { kind = "server_chat" }, shapes = [
  { input_tokens = 1024, output_tokens = 128, weight = 7 },
  { input_tokens = 8192, output_tokens = 1024, weight = 3 },
] }
concurrency = [1]
prompts_per_concurrency = 2
timeout_seconds = 60
"#,
        )?;

        validate_bench("mixture", &definition)?;
        let BenchDefinition::Serving { request_source, .. } = definition else {
            return Err(std::io::Error::other("expected a serving Bench").into());
        };
        let Some(request_source) = request_source else {
            return Err(std::io::Error::other("expected a request source").into());
        };
        assert_eq!(
            request_source.tpot_applicability(),
            BenchTpotApplicability::Applicable
        );
        assert!(matches!(
            request_source,
            BenchRequestSource::RandomMixture { shapes, .. }
                if shapes
                    == vec![
                        BenchRandomShape {
                            input_tokens: 1024,
                            output_tokens: 128,
                            weight: 7,
                        },
                        BenchRandomShape {
                            input_tokens: 8192,
                            output_tokens: 1024,
                            weight: 3,
                        },
                    ]
        ));
        Ok(())
    }

    #[test]
    fn weighted_random_mixture_prompt_omission_resolves_to_flat()
    -> Result<(), Box<dyn std::error::Error>> {
        let definition = toml::from_str::<BenchDefinition>(
            r#"
kind = "serving"
request_source = { kind = "random_mixture", shapes = [
  { input_tokens = 1024, output_tokens = 128, weight = 7 },
  { input_tokens = 8192, output_tokens = 1024, weight = 3 },
] }
concurrency = [1]
prompts_per_concurrency = 2
timeout_seconds = 60
"#,
        )?;

        let BenchDefinition::Serving { request_source, .. } = definition else {
            return Err(std::io::Error::other("expected a serving Bench").into());
        };
        assert!(matches!(
            request_source,
            Some(BenchRequestSource::RandomMixture {
                prompt,
                ..
            }) if prompt.declared().is_none() && prompt.effective() == &BenchPrompt::Flat
        ));
        Ok(())
    }

    #[test]
    fn agentic_source_is_the_only_required_public_source_input()
    -> Result<(), Box<dyn std::error::Error>> {
        let definition = toml::from_str::<BenchDefinition>(
            r#"
kind = "serving"
agentic_source = { dataset = "semianalysis_agentx_062126_256k", profile = "inferencex" }
concurrency = [2]
timeout_seconds = 3600
"#,
        )?;

        validate_bench("agentx", &definition)?;
        let serialized = toml::to_string(&definition)?;
        assert!(serialized.contains("agentic_source"));
        assert!(serialized.contains("semianalysis_agentx_062126_256k"));
        assert!(serialized.contains("profile = \"inferencex\""));
        Ok(())
    }

    #[test]
    fn agentic_source_rejects_independent_request_controls()
    -> Result<(), Box<dyn std::error::Error>> {
        let definition = toml::from_str::<BenchDefinition>(
            r#"
kind = "serving"
agentic_source = { dataset = "semianalysis_agentx_062126_256k", profile = "inferencex" }
concurrency = [1]
prompts_per_concurrency = 1
timeout_seconds = 3600
"#,
        )?;

        let error = validate_bench("agentx", &definition)
            .err()
            .ok_or("agentic source unexpectedly accepted prompts_per_concurrency")?;
        assert!(
            error.to_string().contains("prompts_per_concurrency"),
            "{error}"
        );
        Ok(())
    }

    #[test]
    fn agentic_source_rejects_duration_below_the_release_profile_minimum()
    -> Result<(), Box<dyn std::error::Error>> {
        let definition = toml::from_str::<BenchDefinition>(
            r#"
kind = "serving"
agentic_source = { dataset = "semianalysis_agentx_062126", profile = "inferencex" }
concurrency = [1]
duration_seconds = 899
timeout_seconds = 3600
"#,
        )?;

        let error = validate_bench("agentx", &definition)
            .err()
            .ok_or("agentic source unexpectedly accepted a short duration")?;
        assert!(error.to_string().contains("at least 900"), "{error}");
        Ok(())
    }

    #[test]
    fn weighted_random_mixture_rejects_duplicate_shapes_and_mixed_tpot_classes()
    -> Result<(), Box<dyn std::error::Error>> {
        let duplicate = toml::from_str::<BenchDefinition>(
            r#"
kind = "serving"
request_source = { kind = "random_mixture", prompt = { kind = "server_chat" }, shapes = [
  { input_tokens = 1024, output_tokens = 128, weight = 7 },
  { input_tokens = 1024, output_tokens = 128, weight = 3 },
] }
concurrency = [1]
prompts_per_concurrency = 2
timeout_seconds = 60
"#,
        )?;
        let mixed_tpot = toml::from_str::<BenchDefinition>(
            r#"
kind = "serving"
request_source = { kind = "random_mixture", prompt = { kind = "server_chat" }, shapes = [
  { input_tokens = 1024, output_tokens = 1, weight = 1 },
  { input_tokens = 8192, output_tokens = 2, weight = 1 },
] }
concurrency = [1]
prompts_per_concurrency = 2
timeout_seconds = 60
"#,
        )?;

        let Err(duplicate_error) = validate_bench("duplicate-mixture", &duplicate) else {
            return Err(std::io::Error::other("duplicate exact shapes must be rejected").into());
        };
        let Err(tpot_error) = validate_bench("mixed-tpot", &mixed_tpot) else {
            return Err(std::io::Error::other(
                "one mixture cannot span TPOT applicability classes",
            )
            .into());
        };

        assert!(
            duplicate_error.to_string().contains("duplicate shape"),
            "unexpected error: {duplicate_error}"
        );
        assert!(
            tpot_error.to_string().contains("TPOT"),
            "unexpected error: {tpot_error}"
        );
        Ok(())
    }

    #[test]
    fn legacy_flat_token_shape_is_not_a_second_bench_authority() {
        let result = toml::from_str::<BenchDefinition>(
            r#"
kind = "serving"
input_tokens = 128
output_tokens = 32
concurrency = [1]
prompts_per_concurrency = 1
timeout_seconds = 60
"#,
        );

        assert!(result.is_err_and(|error| error.to_string().contains("input_tokens")));
    }

    #[test]
    fn aggregate_slo_metric_deserializes_directly_into_the_closed_vocabulary()
    -> Result<(), Box<dyn std::error::Error>> {
        let constraint: AggregateSlo = toml::from_str("metric = \"p95_ttft_ms\"\nat_most = 800.0")?;
        let unknown =
            toml::from_str::<AggregateSlo>("metric = \"aiperf_private_latency\"\nat_most = 800.0");

        assert_eq!(constraint.metric.name(), "p95_ttft_ms");
        assert!(unknown.is_err_and(|error| error.to_string().contains("unknown Bench metric")));
        Ok(())
    }

    #[test]
    fn request_slo_rejects_an_invalid_good_request_ratio() -> Result<(), Box<dyn std::error::Error>>
    {
        let result = validate_bench_slos(
            "latency",
            BenchTpotApplicability::Applicable,
            false,
            false,
            &[],
            &Some(RequestSlo {
                request_latency_ms: None,
                ttft_ms: Some(800.0),
                tpot_ms: None,
                minimum_good_request_ratio: 0.0,
            }),
            false,
        );
        let Err(error) = result else {
            return Err(
                std::io::Error::other("zero cannot be a minimum good-request ratio").into(),
            );
        };
        let error = error.to_string();

        assert!(error.contains("minimum_good_request_ratio"), "{error}");
        Ok(())
    }

    #[test]
    fn bundled_eval_task_uses_the_named_release_catalog_entry()
    -> Result<(), Box<dyn std::error::Error>> {
        let definition: EvalDefinition = toml::from_str(
            r#"
kind = "lm-eval"
task = { bundled = "estonia" }
metric = "estonia_pass"
metric_filter = "strict-terminal-answer"
threshold = 0.5
timeout_seconds = 3600
"#,
        )?;

        let EvalDefinition::LmEval { task, .. } = definition else {
            return Err(std::io::Error::other("fixture should be lm-eval").into());
        };
        assert!(matches!(
            task,
            EvalTaskSource::Bundled { bundled } if bundled == "estonia"
        ));
        Ok(())
    }

    #[test]
    fn an_omitted_eval_prompt_resolves_to_flat_without_claiming_a_declaration()
    -> Result<(), Box<dyn std::error::Error>> {
        let omitted: EvalDefinition = toml::from_str(
            r#"
kind = "lm-eval"
task = "gsm8k"
metric = "exact_match"
threshold = 0.9
timeout_seconds = 300
"#,
        )?;
        let EvalDefinition::LmEval { prompt, .. } = omitted else {
            return Err(std::io::Error::other("fixture should be lm-eval").into());
        };
        assert!(prompt.declared().is_none());
        assert_eq!(prompt.effective(), &EvalPrompt::Flat);

        let declared: EvalDefinition = toml::from_str(
            r#"
kind = "lm-eval"
task = "gsm8k"
prompt = { kind = "server_chat" }
metric = "exact_match"
threshold = 0.9
timeout_seconds = 300
"#,
        )?;
        let EvalDefinition::LmEval { prompt, .. } = declared else {
            return Err(std::io::Error::other("fixture should be lm-eval").into());
        };
        assert_eq!(prompt.declared(), Some(&EvalPrompt::ServerChat));
        assert_eq!(prompt.effective(), &EvalPrompt::ServerChat);
        Ok(())
    }

    #[test]
    fn a_flat_eval_rejects_a_server_owned_chat_template_control()
    -> Result<(), Box<dyn std::error::Error>> {
        let definition: EvalDefinition = toml::from_str(
            r#"
kind = "lm-eval"
task = "gsm8k"
metric = "exact_match"
threshold = 0.9
timeout_seconds = 300

[request_body.chat_template_kwargs]
enable_thinking = true
"#,
        )?;
        let Err(error) = catalog_validation::validate_eval("gsm8k", &definition) else {
            return Err(std::io::Error::other(
                "a flat Eval must reject a server-owned template control",
            )
            .into());
        };
        assert!(
            error.to_string().contains("chat_template_kwargs"),
            "{error}"
        );

        let server_chat: EvalDefinition = toml::from_str(
            r#"
kind = "lm-eval"
task = "gsm8k"
prompt = { kind = "server_chat" }
metric = "exact_match"
threshold = 0.9
timeout_seconds = 300

[request_body.chat_template_kwargs]
enable_thinking = true
"#,
        )?;
        catalog_validation::validate_eval("gsm8k", &server_chat)?;
        Ok(())
    }

    #[test]
    fn inference_request_body_preserves_nested_toml_json_types()
    -> Result<(), Box<dyn std::error::Error>> {
        let definition: EvalDefinition = toml::from_str(
            r#"
kind = "lm-eval"
task = "gsm8k"
metric = "exact_match"
threshold = 0.9
timeout_seconds = 300

[request_body]
temperature = 1.0
logprobs = true
stop_token_ids = [1, 2]

[request_body.chat_template_kwargs]
enable_thinking = false
"#,
        )?;

        let EvalDefinition::LmEval { request_body, .. } = definition else {
            return Err(std::io::Error::other("fixture should be lm-eval").into());
        };
        assert!(matches!(
            request_body.get("temperature"),
            Some(JsonValue::Float(value)) if *value == 1.0
        ));
        assert!(matches!(
            request_body.get("logprobs"),
            Some(JsonValue::Bool(true))
        ));
        assert!(matches!(
            request_body.get("stop_token_ids"),
            Some(JsonValue::Array(values)) if values.len() == 2
        ));
        assert!(matches!(
            request_body.get("chat_template_kwargs"),
            Some(JsonValue::Object(values))
                if values.get("enable_thinking") == Some(&JsonValue::Bool(false))
        ));
        Ok(())
    }

    #[test]
    fn inference_request_body_rejects_owned_members_and_toml_dates()
    -> Result<(), Box<dyn std::error::Error>> {
        let reserved: EvalDefinition = toml::from_str(
            r#"
kind = "lm-eval"
task = "gsm8k"
metric = "exact_match"
threshold = 0.9
timeout_seconds = 300
request_body = { messages = [] }
"#,
        )?;
        let Err(error) = validate_eval("gsm8k", &reserved) else {
            return Err(std::io::Error::other(
                "messages should be owned by the measurement runtime",
            )
            .into());
        };
        let error = error.to_string();
        assert!(error.contains("request_body.messages"), "{error}");

        let Err(date) = toml::from_str::<EvalDefinition>(
            r#"
kind = "lm-eval"
task = "gsm8k"
metric = "exact_match"
threshold = 0.9
timeout_seconds = 300
request_body = { vendor_date = 2026-07-15 }
"#,
        ) else {
            return Err(
                std::io::Error::other("TOML dates should have no exact JSON projection").into(),
            );
        };
        let date = date.to_string();
        assert!(date.contains("JSON-compatible value"), "{date}");
        Ok(())
    }

    // The script text feeds recorded evidence and remote execution; a byte
    // drift here must fail the suite, not surface later as a digest change.
    #[test]
    fn source_digest_script_text_is_pinned() {
        insta::assert_snapshot!(source_digest_script(&[PathBuf::from(".inferlab")]));
    }

    #[test]
    fn role_escapes_merge_into_common_server_escapes() {
        let common = NsysEscapes {
            executable: Some("nsys".to_owned()),
            launch_options: vec!["--cuda-graph-trace=node".to_owned()],
            start_options: vec!["--nic-metrics=true".to_owned()],
            trace: vec!["cuda".to_owned()],
            sampling: Some("cpu".to_owned()),
            context_switch: None,
            env: BTreeMap::from([
                ("NSYS_SHARED".to_owned(), "common".to_owned()),
                ("NSYS_COMMON_ONLY".to_owned(), "1".to_owned()),
            ]),
        };
        let role = NsysEscapes {
            executable: None,
            launch_options: vec!["--nvtx-domain-include=prefill".to_owned()],
            start_options: Vec::new(),
            trace: vec!["cuda".to_owned(), "nvtx".to_owned()],
            sampling: Some("process-tree".to_owned()),
            context_switch: Some("system-wide".to_owned()),
            env: BTreeMap::from([("NSYS_SHARED".to_owned(), "role".to_owned())]),
        };
        let merged = common.merged_with(&role);
        assert_eq!(merged.executable.as_deref(), Some("nsys"));
        assert_eq!(
            merged.launch_options,
            ["--cuda-graph-trace=node", "--nvtx-domain-include=prefill"]
        );
        assert_eq!(merged.start_options, ["--nic-metrics=true"]);
        assert_eq!(merged.trace, ["cuda", "nvtx"]);
        assert_eq!(merged.sampling.as_deref(), Some("process-tree"));
        assert_eq!(merged.context_switch.as_deref(), Some("system-wide"));
        assert_eq!(
            merged.env,
            BTreeMap::from([
                ("NSYS_COMMON_ONLY".to_owned(), "1".to_owned()),
                ("NSYS_SHARED".to_owned(), "role".to_owned()),
            ])
        );
    }

    #[test]
    fn managed_and_dedicated_escape_options_are_rejected_in_both_lists() {
        let rejected = [
            "--session=other",
            "--session-new=other",
            "--output=/tmp/trace",
            "-o=/tmp/trace",
            "--export=sqlite",
            "--force-overwrite=false",
            "-f=false",
            "--capture-range=none",
            "-c=none",
            "--capture-range-end=stop",
            "--wait=none",
            "--trace=cuda",
            "-t=cuda",
            "--sample=cpu",
            "-s=cpu",
            "--cpuctxsw=none",
            "--wait",
            "-tnone",
            "-o/tmp/x",
            "-ftrue",
            "-cnone",
            "-snone",
            "--wai=all",
            "--out=/tmp/x",
            "--force=true",
            "--sess=x",
            "--w",
            "--wai",
        ];
        for option in rejected {
            for field in ["launch_options", "start_options"] {
                let mut escapes = ProfilerEscapes::default();
                let list = if field == "launch_options" {
                    &mut escapes.nsys.launch_options
                } else {
                    &mut escapes.nsys.start_options
                };
                list.push(option.to_owned());
                let error = validate_profiler_escapes("server \"pd\"", &escapes)
                    .err()
                    .map(|error| error.to_string());
                let expected = format!(
                    "server \"pd\" nsys {field} contains managed option {option:?}; \
                     use the dedicated profiler escape field or the inferlab-managed value"
                );
                assert!(
                    error
                        .as_deref()
                        .is_some_and(|error| error.contains(&expected)),
                    "{option} in {field}: {error:?}"
                );
            }
        }
        // Launch's -w is --show-output and -e is --env-var on the qualified
        // nsys; neither names a managed fact, in plain or attached form.
        let permitted = NsysEscapes {
            launch_options: vec![
                "-w=true".to_owned(),
                "-e=NSYS_FIXTURE=1".to_owned(),
                "-eNSYS_ATTACHED=1".to_owned(),
                "--cuda-graph-trace=node".to_owned(),
            ],
            start_options: vec![
                "--nic-metrics=true".to_owned(),
                "--stats=true".to_owned(),
                "-x=true".to_owned(),
                "-xtrue".to_owned(),
            ],
            ..NsysEscapes::default()
        };
        assert!(
            validate_profiler_escapes("server \"pd\"", &ProfilerEscapes { nsys: permitted },)
                .is_ok(),
            "nsys-owned options that name no managed fact pass the load gate"
        );
    }

    // A non-identifier key would be parsed as an option of the environment
    // utility rather than applied as an assignment
    // ([[RFC-0004:C-WORKLOAD-PROFILING]]).
    #[test]
    fn escape_env_keys_must_be_posix_identifiers() {
        for key in ["--unset", "1BAD", "BAD-KEY", "", "BAD KEY"] {
            let mut escapes = ProfilerEscapes::default();
            escapes.nsys.env.insert(key.to_owned(), "value".to_owned());
            let error = validate_profiler_escapes("server \"pd\"", &escapes)
                .err()
                .map(|error| error.to_string());
            let expected = format!(
                "server \"pd\" nsys env contains key {key:?}, which is not a POSIX \
                 identifier; environment entries reach the profiler commands as assignments"
            );
            assert!(
                error
                    .as_deref()
                    .is_some_and(|error| error.contains(&expected)),
                "{key:?}: {error:?}"
            );
        }
        for key in ["_OK", "OK2", "NSYS_FIXTURE"] {
            let mut escapes = ProfilerEscapes::default();
            escapes.nsys.env.insert(key.to_owned(), "value".to_owned());
            assert!(
                validate_profiler_escapes("server \"pd\"", &escapes).is_ok(),
                "{key:?} is a POSIX identifier and passes the load gate"
            );
        }
    }

    // A standalone terminator would splice ahead of the managed tail and
    // demote it to positionals of the wrapped command; on the qualified
    // nsys the start side even swallows it silently
    // ([[RFC-0004:C-WORKLOAD-PROFILING]]).
    #[test]
    fn standalone_terminators_are_rejected_in_both_lists() {
        for option in ["-", "--"] {
            for field in ["launch_options", "start_options"] {
                let mut escapes = ProfilerEscapes::default();
                let list = if field == "launch_options" {
                    &mut escapes.nsys.launch_options
                } else {
                    &mut escapes.nsys.start_options
                };
                list.push(option.to_owned());
                let error = validate_profiler_escapes("server \"pd\"", &escapes)
                    .err()
                    .map(|error| error.to_string());
                let expected = format!(
                    "server \"pd\" nsys {field} contains standalone {option:?}, \
                     which ends option parsing and displaces the inferlab-managed argv tail"
                );
                assert!(
                    error
                        .as_deref()
                        .is_some_and(|error| error.contains(&expected)),
                    "{option} in {field}: {error:?}"
                );
            }
        }
    }
}
