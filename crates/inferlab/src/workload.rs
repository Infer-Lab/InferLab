//! Measurement aggregate facade. Planning freezes intent; runtime modules
//! consume that plan; record projection remains a separate evidence owner.

mod adaptive;
mod domain;
mod plan;
mod planning;
mod record;
mod runtime;
mod wire;

pub(crate) use domain::{
    MeasurementModel, WorkloadEndpoint, WorkloadEndpointProtocol, WorkloadHttpAction,
    WorkloadHttpMethod, WorkloadServerMetricsEndpoint,
};
pub(crate) use record::WorkloadKind;
pub use record::WorkloadStatus;
pub(crate) use runtime::skip;
pub use runtime::{run_bench, run_eval};

pub use plan::{
    BenchCasePlan, BenchExecutionPlan, BenchPlan, ClientCommandPlan, EvalExecutionPlan, EvalPlan,
    LoadShape, MeasurementPlan, MeasurementResolveContext, ResolvedWorkloadPlan,
    WorkloadServerAccess,
};
pub use planning::{resolve_manual_bench, resolve_measurements, resolved_request_count};

#[cfg(test)]
mod tests {
    use crate::workspace::{BenchDefinition, validate_bench};

    #[test]
    fn serving_bench_rejects_ambiguous_or_request_shaped_session_load()
    -> Result<(), Box<dyn std::error::Error>> {
        let both = toml::from_str::<BenchDefinition>(
            r#"
kind = "serving"
request_source = { kind = "random", prompt = { kind = "server_chat" }, input_tokens = 128, output_tokens = 32 }
session_source = { dataset = "sharegpt", max_input_tokens = 8192 }
concurrency = [1]
sessions_per_concurrency = 1
timeout_seconds = 60
"#,
        )?;
        let error = validate_bench("ambiguous", &both)
            .err()
            .ok_or("ambiguous source unexpectedly validated")?;
        assert!(
            error
                .to_string()
                .contains("exactly one of request_source and session_source")
        );

        let request_shaped = toml::from_str::<BenchDefinition>(
            r#"
kind = "serving"
session_source = { dataset = "sharegpt", max_input_tokens = 8192 }
concurrency = [1]
sessions_per_concurrency = 1
prompts_per_concurrency = 1
timeout_seconds = 60
"#,
        )?;
        let error = validate_bench("request-shaped", &request_shaped)
            .err()
            .ok_or("request-shaped session load unexpectedly validated")?;
        assert!(error.to_string().contains("prompts_per_concurrency"));

        let adaptive = toml::from_str::<BenchDefinition>(
            r#"
kind = "adaptive-serving"
request_source = { kind = "random", prompt = { kind = "server_chat" }, input_tokens = 128, output_tokens = 32 }
session_source = { dataset = "sharegpt", max_input_tokens = 8192 }
initial_request_rates = [1.0]
request_count = 1
timeout_seconds = 60
"#,
        );
        assert!(adaptive.is_err());
        Ok(())
    }
}
