//! Domain validation for the portable workspace catalog.

use super::definitions::{
    AggregateSlo, BenchDefinition, BenchRequestSource, BenchSessionSource, BenchTokenSelector,
    BenchTpotApplicability, EvalDefinition, EvalTaskSource, JsonValue, ProfilerEscapes,
    RequestRate, RequestSlo, WorkspaceConfig,
};
use super::invalid;
use super::source::{is_safe_relative, reject_symlink_components};
use crate::InferlabError;
use crate::bench_dataset_catalog;
use inferlab_protocol::{Parallelism, ServeTopology};
use std::collections::{BTreeMap, BTreeSet};
use std::ffi::OsStr;
use std::fs;
use std::path::Path;

pub(super) fn validate_workspace(
    root: &Path,
    config: &WorkspaceConfig,
) -> Result<(), InferlabError> {
    if config.schema_version != 2 {
        return invalid(format!(
            "unsupported workspace schema version {}; expected 2",
            config.schema_version
        ));
    }
    for (id, stack) in &config.stacks {
        require_id("stack", id)?;
        require_nonempty("integration", id, &stack.integration)?;
        require_nonempty("Pixi environment", id, &stack.pixi_environment)?;
        for path in &stack.source_paths {
            if !is_safe_relative(path) {
                return invalid(format!(
                    "stack {id:?} source path {} must be workspace-relative without parent traversal",
                    path.display()
                ));
            }
            reject_symlink_components(root, id, path)?;
            if !root.join(path).exists() {
                return invalid(format!(
                    "stack {id:?} source path {} does not exist",
                    path.display()
                ));
            }
        }
        let mut seen_checks = BTreeSet::new();
        for check in &stack.checks {
            require_id("stack check", &check.id)?;
            if !seen_checks.insert(&check.id) {
                return invalid(format!(
                    "stack {id:?} declares duplicate check id {:?}",
                    check.id
                ));
            }
            validate_environment_script(root, id, "check", &check.id, &check.script)?;
        }
        let mut seen_postprocess = BTreeSet::new();
        for step in &stack.image_postprocess {
            require_id("stack postprocess step", &step.id)?;
            if !seen_postprocess.insert(&step.id) {
                return invalid(format!(
                    "stack {id:?} declares duplicate image postprocess id {:?}",
                    step.id
                ));
            }
            validate_environment_script(
                root,
                id,
                "image postprocess step",
                &step.id,
                &step.script,
            )?;
        }
    }
    for (id, model) in &config.models {
        require_id("model", id)?;
        require_nonempty("served model name", id, &model.served_name)?;
    }
    for (id, server) in &config.servers {
        require_id("server", id)?;
        require_reference("stack", &server.stack, &config.stacks)?;
        require_reference("model", &server.model, &config.models)?;
        if server.readiness_timeout_seconds == 0 {
            return invalid(format!(
                "server {id:?} readiness_timeout_seconds must be nonzero"
            ));
        }
        if server.readiness_attempt_timeout_seconds == Some(0) {
            return invalid(format!(
                "server {id:?} readiness_attempt_timeout_seconds must be nonzero"
            ));
        }
        if server.capture_control_deadline_seconds == Some(0) {
            return invalid(format!(
                "server {id:?} capture_control_deadline_seconds must be nonzero"
            ));
        }
        if server.capture_arm_deadline_seconds == Some(0) {
            return invalid(format!(
                "server {id:?} capture_arm_deadline_seconds must be nonzero"
            ));
        }
        if server.capture_finalization_deadline_seconds == Some(0) {
            return invalid(format!(
                "server {id:?} capture_finalization_deadline_seconds must be nonzero"
            ));
        }
        if server.topology == ServeTopology::Single
            && (server.pd_router_backend.is_some() || server.kv_transfer.is_some())
        {
            return invalid(format!(
                "single-topology server {id:?} must not declare pd_router_backend or kv_transfer"
            ));
        }
        if server.topology == ServeTopology::PrefillDecode
            && (server.gateway_backend.is_none() || server.pd_router_backend.is_none())
        {
            return invalid(format!(
                "prefill_decode server {id:?} must declare both gateway_backend and pd_router_backend"
            ));
        }
        if let Some(backend) = &server.gateway_backend {
            require_nonempty("server Gateway backend", id, backend)?;
        }
        if let Some(backend) = &server.pd_router_backend {
            require_nonempty("server P/D Router backend", id, backend)?;
        }
        validate_parallelism("server", id, &server.parallelism)?;
        validate_profiler_escapes(&format!("server {id:?}"), &server.profiler)?;
        for (role_id, role) in &server.roles {
            require_id("serve role", role_id)?;
            validate_server_role(id, server.topology, role_id)?;
            if role.replicas == Some(0) {
                return invalid(format!(
                    "serve role {role_id:?} replica count must be nonzero"
                ));
            }
            validate_parallelism("serve role", role_id, &role.parallelism)?;
            validate_profiler_escapes(&format!("server {id:?} role {role_id:?}"), &role.profiler)?;
        }
        if let Some(default_case) = &server.default_case
            && !server.cases.contains_key(default_case)
        {
            return invalid(format!(
                "server {id:?} default_case references unknown case {default_case:?}"
            ));
        }
        for (case_id, case) in &server.cases {
            require_id("server case", case_id)?;
            if case.readiness_timeout_seconds == Some(0) {
                return invalid(format!(
                    "server case {case_id:?} readiness_timeout_seconds must be nonzero"
                ));
            }
            if case.readiness_attempt_timeout_seconds == Some(0) {
                return invalid(format!(
                    "server case {case_id:?} readiness_attempt_timeout_seconds must be nonzero"
                ));
            }
            for (name, value) in [
                (
                    "capture_arm_deadline_seconds",
                    case.capture_arm_deadline_seconds,
                ),
                (
                    "capture_control_deadline_seconds",
                    case.capture_control_deadline_seconds,
                ),
                (
                    "capture_finalization_deadline_seconds",
                    case.capture_finalization_deadline_seconds,
                ),
            ] {
                if value == Some(0) {
                    return invalid(format!("server case {case_id:?} {name} must be nonzero"));
                }
            }
            if server.topology == ServeTopology::Single
                && (case.pd_router_backend.is_some() || case.kv_transfer.is_some())
            {
                return invalid(format!(
                    "single-topology server case {case_id:?} must not declare pd_router_backend or kv_transfer"
                ));
            }
            if case.gateway_backend.is_some() && server.gateway_backend.is_none() {
                return invalid(format!(
                    "server case {case_id:?} cannot add gateway_backend because the server base does not declare a Gateway"
                ));
            }
            if case.pd_router_backend.is_some() && server.pd_router_backend.is_none() {
                return invalid(format!(
                    "server case {case_id:?} cannot add pd_router_backend because the server base does not declare a P/D Router"
                ));
            }
            if let Some(backend) = &case.gateway_backend {
                require_nonempty("server case Gateway backend", case_id, backend)?;
            }
            if let Some(backend) = &case.pd_router_backend {
                require_nonempty("server case P/D Router backend", case_id, backend)?;
            }
            validate_parallelism("server case", case_id, &case.parallelism)?;
            for (role_id, role) in &case.roles {
                require_id("server case role", role_id)?;
                validate_server_role(id, server.topology, role_id)?;
                if role.replicas == Some(0) {
                    return invalid(format!(
                        "server case {case_id:?} role {role_id:?} replica count must be nonzero"
                    ));
                }
                validate_parallelism("server case role", role_id, &role.parallelism)?;
            }
        }
    }
    for (id, bench) in &config.benches {
        require_id("bench", id)?;
        validate_bench(id, bench)?;
    }
    for (id, eval) in &config.evals {
        require_id("eval", id)?;
        validate_eval(id, eval)?;
        validate_eval_task_source(root, id, eval)?;
    }

    for (id, suite) in &config.workload_suites {
        require_id("workload suite", id)?;
        if suite.evals.is_empty() && suite.benches.is_empty() {
            return invalid(format!(
                "workload suite {id:?} must select at least one measurement"
            ));
        }
        for eval in &suite.evals {
            require_reference("eval", eval, &config.evals)?;
        }
        for bench in &suite.benches {
            require_reference("bench", bench, &config.benches)?;
        }
        if let Some(gate) = &suite.gate {
            require_reference("eval gate", gate, &config.evals)?;
            if !suite.evals.contains(gate) {
                return invalid(format!(
                    "workload suite {id:?} gate {gate:?} is not in its eval list"
                ));
            }
        }
    }

    for (id, recipe) in &config.recipes {
        require_id("recipe", id)?;
        require_reference("server", &recipe.server, &config.servers)?;
        require_reference(
            "workload suite",
            &recipe.workload_suite,
            &config.workload_suites,
        )?;
    }

    for (id, image) in &config.images {
        require_id("image", id)?;
        require_reference("stack", &image.stack, &config.stacks)?;
        require_nonempty("base image", id, &image.base_image)?;
        if image.base_image.chars().any(char::is_whitespace) {
            return invalid(format!(
                "image {id:?} base image {:?} must not contain whitespace",
                image.base_image
            ));
        }
        if image.platforms.is_empty() {
            return invalid(format!("image {id:?} must declare at least one platform"));
        }
        let mut platforms = BTreeSet::new();
        for platform in &image.platforms {
            let mut parts = platform.split('/');
            let valid = matches!(
                (parts.next(), parts.next(), parts.next()),
                (Some(os), Some(arch), None) if !os.is_empty() && !arch.is_empty()
            );
            if !valid {
                return invalid(format!(
                    "image {id:?} platform {platform:?} must use the os/arch form"
                ));
            }
            if !platforms.insert(platform) {
                return invalid(format!(
                    "image {id:?} declares duplicate platform {platform:?}"
                ));
            }
        }
        if let Some(packages) = &image.packages {
            let stack = &config.stacks[&image.stack];
            for package in packages {
                if !is_safe_relative(package) {
                    return invalid(format!(
                        "image {id:?} package path {} must be workspace-relative without parent \
                         traversal",
                        package.display()
                    ));
                }
                if !stack.source_paths.contains(package) {
                    return invalid(format!(
                        "image {id:?} package path {} is not one of stack {:?}'s source_paths",
                        package.display(),
                        image.stack
                    ));
                }
            }
        }
        for coordinate in &image.validations {
            let Some(recipe) = config.recipes.get(&coordinate.recipe) else {
                return invalid(format!("unknown recipe {:?}", coordinate.recipe));
            };
            let server = &config.servers[&recipe.server];
            if let Some(case) = &coordinate.server_case
                && !server.cases.contains_key(case)
            {
                return invalid(format!(
                    "image {id:?} validation references unknown server case {case:?} of recipe {:?}",
                    coordinate.recipe,
                ));
            }
            if server.stack != image.stack {
                return invalid(format!(
                    "image {id:?} selects stack {:?} but validation recipe {:?} selects server stack {:?}; \
                     a validation recipe must run the serving stack the image contains",
                    image.stack, coordinate.recipe, server.stack
                ));
            }
        }
    }
    for (id, external) in &config.external_images {
        require_id("external image", id)?;
        require_nonempty("external image reference", id, &external.reference)?;
        if external.reference.chars().any(char::is_whitespace) {
            return invalid(format!(
                "external image {id:?} reference {:?} must not contain whitespace",
                external.reference
            ));
        }
        // Digest pinning makes a committed baseline mean one artifact
        // ([[RFC-0003:C-RUNTIME-WORKFLOWS]]).
        let digest_pinned =
            external
                .reference
                .rsplit_once("@sha256:")
                .is_some_and(|(repository, digest)| {
                    !repository.is_empty()
                        && digest.len() == 64
                        && digest.bytes().all(|byte| byte.is_ascii_hexdigit())
                });
        if !digest_pinned {
            return invalid(format!(
                "external image {id:?} reference {:?} must carry its immutable digest \
                 (repository[:tag]@sha256:<64 hex>)",
                external.reference
            ));
        }
        if external.integration.is_empty()
            || !external
                .integration
                .bytes()
                .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-')
        {
            return invalid(format!(
                "external image {id:?} claims invalid integration identifier {:?}",
                external.integration
            ));
        }
        // The integration package's presence in the committed dependency set
        // is verified against the parsed Pixi manifest in `validate_pixi`
        // ([[RFC-0006:C-INTEGRATIONS]]).
    }
    Ok(())
}

pub(super) fn validate_server_role(
    server: &str,
    topology: ServeTopology,
    role: &str,
) -> Result<(), InferlabError> {
    let valid = match topology {
        ServeTopology::Single => role == "serve",
        ServeTopology::PrefillDecode => matches!(role, "prefill" | "decode"),
    };
    if valid {
        Ok(())
    } else {
        invalid(format!(
            "server {server:?} topology {topology:?} does not permit declared role {role:?}; \
             roles are canonical and router is derived"
        ))
    }
}

pub(super) fn validate_parallelism(
    owner: &str,
    id: &str,
    parallelism: &Parallelism,
) -> Result<(), InferlabError> {
    let values = [
        (
            "outer.tensor_parallel_size",
            parallelism
                .outer
                .as_ref()
                .and_then(|outer| outer.tensor_parallel_size),
        ),
        (
            "outer.pipeline_parallel_size",
            parallelism
                .outer
                .as_ref()
                .and_then(|outer| outer.pipeline_parallel_size),
        ),
        (
            "attention.tensor_parallel_size",
            parallelism
                .attention
                .as_ref()
                .and_then(|attention| attention.tensor_parallel_size),
        ),
        (
            "attention.data_parallel_size",
            parallelism
                .attention
                .as_ref()
                .and_then(|attention| attention.data_parallel_size),
        ),
        (
            "attention.context_parallel_size",
            parallelism
                .attention
                .as_ref()
                .and_then(|attention| attention.context_parallel_size),
        ),
        (
            "experts.tensor_parallel_size",
            parallelism
                .experts
                .as_ref()
                .and_then(|experts| experts.tensor_parallel_size),
        ),
        (
            "experts.data_parallel_size",
            parallelism
                .experts
                .as_ref()
                .and_then(|experts| experts.data_parallel_size),
        ),
        (
            "experts.expert_parallel_size",
            parallelism
                .experts
                .as_ref()
                .and_then(|experts| experts.expert_parallel_size),
        ),
        (
            "experts.dense_tensor_parallel_size",
            parallelism
                .experts
                .as_ref()
                .and_then(|experts| experts.dense_tensor_parallel_size),
        ),
    ];
    if let Some((field, _)) = values.into_iter().find(|(_, value)| *value == Some(0)) {
        return invalid(format!(
            "{owner} {id:?} parallelism.{field} must be nonzero"
        ));
    }
    Ok(())
}

pub(crate) fn validate_eval(id: &str, definition: &EvalDefinition) -> Result<(), InferlabError> {
    match definition {
        EvalDefinition::OpenAiSmoke {
            prompt,
            max_tokens,
            timeout_seconds,
        } => {
            require_nonempty("eval prompt", id, prompt)?;
            require_positive("max_tokens", id, u64::from(*max_tokens))?;
            require_positive("timeout_seconds", id, *timeout_seconds)
        }
        EvalDefinition::LmEval {
            task,
            request_body,
            limit,
            seed,
            trials,
            max_tokens,
            concurrency,
            metric,
            metric_filter,
            threshold,
            timeout_seconds,
            ..
        } => {
            match task {
                EvalTaskSource::BuiltIn(task) => require_nonempty("lm-eval task", id, task)?,
                EvalTaskSource::Bundled { bundled } => {
                    require_nonempty("lm-eval bundled task", id, bundled)?
                }
                EvalTaskSource::WorkspaceYaml { .. } => {}
            }
            validate_request_body("eval", id, request_body, &["seed"])?;
            require_nonempty("lm-eval metric", id, metric)?;
            if let Some(metric_filter) = metric_filter {
                require_nonempty("lm-eval metric_filter", id, metric_filter)?;
            }
            require_optional_positive("limit", id, limit.map(u64::from))?;
            require_positive("trials", id, u64::from(*trials))?;
            let base_seed = seed.unwrap_or(1234);
            if base_seed.checked_add(u64::from(*trials - 1)).is_none() {
                return invalid(format!(
                    "eval {id:?} seed schedule exceeds the supported unsigned integer range"
                ));
            }
            require_optional_positive("max_tokens", id, max_tokens.map(u64::from))?;
            require_optional_positive("concurrency", id, concurrency.map(u64::from))?;
            if !threshold.is_finite() {
                return invalid(format!("eval {id:?} threshold must be finite"));
            }
            if *trials > 1 && !(0.0..=1.0).contains(threshold) {
                return invalid(format!(
                    "eval {id:?} threshold must be between zero and one for repeated trials"
                ));
            }
            require_positive("timeout_seconds", id, *timeout_seconds)
        }
    }
}

pub(crate) fn validate_eval_task_source(
    root: &Path,
    id: &str,
    definition: &EvalDefinition,
) -> Result<(), InferlabError> {
    let EvalDefinition::LmEval { task, .. } = definition else {
        return Ok(());
    };
    let EvalTaskSource::WorkspaceYaml { yaml } = task else {
        return Ok(());
    };
    if !is_safe_relative(yaml) {
        return invalid(format!(
            "lm-eval {id:?} task YAML {} must be workspace-relative without parent traversal",
            yaml.display()
        ));
    }
    if !matches!(
        yaml.extension(),
        Some(extension) if extension == OsStr::new("yaml") || extension == OsStr::new("yml")
    ) {
        return invalid(format!(
            "lm-eval {id:?} task YAML {} must use a .yaml or .yml extension supported by the pinned lm-eval runtime",
            yaml.display()
        ));
    }
    reject_symlink_components(root, id, yaml)?;
    let path = root.join(yaml);
    if !path.is_file() {
        return invalid(format!(
            "lm-eval {id:?} task YAML {} is not a regular workspace file",
            yaml.display()
        ));
    }
    Ok(())
}

pub(crate) fn validate_bench(id: &str, definition: &BenchDefinition) -> Result<(), InferlabError> {
    match definition {
        BenchDefinition::Serving {
            request_source,
            session_source,
            server_metrics,
            request_body,
            aggregate_slos,
            request_slo,
            concurrency,
            prompts_per_concurrency,
            warmup_prompts_per_concurrency,
            sessions_per_concurrency,
            warmup_sessions_per_concurrency,
            request_rates,
            request_count,
            duration_seconds,
            burstiness,
            timeout_seconds,
            ..
        } => {
            match (request_source, session_source) {
                (Some(_), Some(_)) | (None, None) => {
                    return invalid(format!(
                        "bench {id:?} requires exactly one of request_source and session_source"
                    ));
                }
                _ => {}
            }
            validate_bench_common(
                id,
                request_source.as_ref(),
                request_body,
                *burstiness,
                *timeout_seconds,
            )?;
            if let Some(session_source) = session_source {
                validate_bench_session_source(id, session_source)?;
            }
            let tpot_applicability = request_source.as_ref().map_or_else(
                || {
                    session_source.as_ref().map_or(
                        BenchTpotApplicability::Inapplicable,
                        BenchSessionSource::tpot_applicability,
                    )
                },
                BenchRequestSource::tpot_applicability,
            );
            validate_bench_slos(
                id,
                tpot_applicability,
                matches!(
                    request_source,
                    Some(BenchRequestSource::Dataset { dataset, .. }) if dataset == "speed_bench"
                ),
                *server_metrics,
                aggregate_slos,
                request_slo,
                false,
            )?;
            if session_source.is_some() {
                if concurrency.is_empty() || concurrency.contains(&0) {
                    return invalid(format!(
                        "bench {id:?} session_source requires non-empty positive concurrency"
                    ));
                }
                if sessions_per_concurrency.is_none_or(|value| value == 0) {
                    return invalid(format!(
                        "bench {id:?} session_source requires positive sessions_per_concurrency"
                    ));
                }
                if prompts_per_concurrency.is_some()
                    || *warmup_prompts_per_concurrency != 0
                    || !request_rates.is_empty()
                    || request_count.is_some()
                    || duration_seconds.is_some()
                    || burstiness.is_some()
                {
                    return invalid(format!(
                        "bench {id:?} session_source rejects prompts_per_concurrency, warmup_prompts_per_concurrency, request_rates, request_count, duration_seconds, and burstiness"
                    ));
                }
                if *server_metrics && *warmup_sessions_per_concurrency != 0 {
                    return invalid(format!(
                        "bench {id:?} server_metrics requires zero warmup_sessions_per_concurrency"
                    ));
                }
                return Ok(());
            }
            if sessions_per_concurrency.is_some() || *warmup_sessions_per_concurrency != 0 {
                return invalid(format!(
                    "bench {id:?} request_source rejects sessions_per_concurrency and warmup_sessions_per_concurrency"
                ));
            }
            if concurrency.is_empty() && request_rates.is_empty() {
                return invalid(format!(
                    "bench {id:?} must define a concurrency or request-rate case"
                ));
            }
            if concurrency.contains(&0) {
                return invalid(format!("bench {id:?} concurrency values must be positive"));
            }
            match (concurrency.is_empty(), prompts_per_concurrency) {
                (false, None) => {
                    return invalid(format!(
                        "bench {id:?} requires prompts_per_concurrency for concurrency cases"
                    ));
                }
                (true, Some(_)) => {
                    return invalid(format!(
                        "bench {id:?} sets prompts_per_concurrency without concurrency cases"
                    ));
                }
                (_, Some(0)) => {
                    return invalid(format!(
                        "bench {id:?} prompts_per_concurrency must be positive"
                    ));
                }
                _ => {}
            }
            if concurrency.is_empty() && *warmup_prompts_per_concurrency != 0 {
                return invalid(format!(
                    "bench {id:?} sets warmup_prompts_per_concurrency without concurrency cases"
                ));
            }
            if *server_metrics && *warmup_prompts_per_concurrency != 0 {
                return invalid(format!(
                    "bench {id:?} server_metrics requires zero warmup_prompts_per_concurrency"
                ));
            }
            validate_request_rates(id, request_rates)?;
            validate_rate_count_policy(
                id,
                !request_rates.is_empty(),
                request_rates.iter().any(|rate| rate.finite().is_none()),
                *request_count,
                *duration_seconds,
            )
        }
        BenchDefinition::AdaptiveServing {
            request_source,
            server_metrics,
            request_body,
            aggregate_slos,
            request_slo,
            initial_request_rates,
            min_rate_resolution,
            request_count,
            duration_seconds,
            burstiness,
            timeout_seconds,
            ..
        } => {
            validate_bench_common(
                id,
                Some(request_source),
                request_body,
                *burstiness,
                *timeout_seconds,
            )?;
            validate_bench_slos(
                id,
                request_source.tpot_applicability(),
                matches!(
                    request_source,
                    BenchRequestSource::Dataset { dataset, .. } if dataset == "speed_bench"
                ),
                *server_metrics,
                aggregate_slos,
                request_slo,
                true,
            )?;
            if initial_request_rates.is_empty()
                || initial_request_rates
                    .iter()
                    .any(|rate| !rate.is_finite() || *rate <= 0.0)
            {
                return invalid(format!(
                    "bench {id:?} initial_request_rates must contain positive finite values"
                ));
            }
            if min_rate_resolution.is_some_and(|value| !value.is_finite() || value <= 0.0) {
                return invalid(format!(
                    "bench {id:?} min_rate_resolution must be positive and finite"
                ));
            }
            validate_rate_count_policy(id, true, false, *request_count, *duration_seconds)
        }
    }
}

pub(super) fn validate_bench_slos(
    id: &str,
    tpot_applicability: BenchTpotApplicability,
    speed_bench_source: bool,
    server_metrics: bool,
    aggregate_slos: &[AggregateSlo],
    request_slo: &Option<RequestSlo>,
    required: bool,
) -> Result<(), InferlabError> {
    if required && aggregate_slos.is_empty() && request_slo.is_none() {
        return invalid(format!(
            "adaptive bench {id:?} requires aggregate_slos, request_slo, or both"
        ));
    }
    for constraint in aggregate_slos {
        let metric = constraint.metric;
        let bound = match (constraint.at_most, constraint.at_least) {
            (Some(value), None) | (None, Some(value)) => value,
            _ => {
                return invalid(format!(
                    "bench {id:?} aggregate_slos metric {:?} requires exactly one of at_most or at_least",
                    metric.name()
                ));
            }
        };
        if !bound.is_finite() {
            return invalid(format!(
                "bench {id:?} aggregate_slos metric {:?} bound must be finite",
                metric.name()
            ));
        }
        if metric.depends_on_tpot() && !tpot_applicability.is_applicable() {
            return invalid(format!(
                "bench {id:?} cannot constrain TPOT when the request source makes TPOT inapplicable"
            ));
        }
        if metric.requires_request_slo() && request_slo.is_none() {
            return invalid(format!(
                "bench {id:?} aggregate metric {:?} requires request_slo",
                metric.name()
            ));
        }
        if metric.requires_speed_bench_server_metrics() && !(server_metrics && speed_bench_source) {
            return invalid(format!(
                "bench {id:?} aggregate metric {:?} requires a speed_bench request source with server_metrics = true",
                metric.name()
            ));
        }
    }
    let Some(request_slo) = request_slo else {
        return Ok(());
    };
    let bounds = [
        ("request_latency_ms", request_slo.request_latency_ms),
        ("ttft_ms", request_slo.ttft_ms),
        ("tpot_ms", request_slo.tpot_ms),
    ];
    if bounds.iter().all(|(_, value)| value.is_none()) {
        return invalid(format!(
            "bench {id:?} request_slo requires at least one request-metric bound"
        ));
    }
    for (name, value) in bounds {
        if value.is_some_and(|value| !value.is_finite() || value < 0.0) {
            return invalid(format!(
                "bench {id:?} request_slo {name} must be finite and non-negative"
            ));
        }
    }
    if request_slo.tpot_ms.is_some() && !tpot_applicability.is_applicable() {
        return invalid(format!(
            "bench {id:?} cannot constrain request TPOT when the request source makes TPOT inapplicable"
        ));
    }
    if !(request_slo.minimum_good_request_ratio.is_finite()
        && request_slo.minimum_good_request_ratio > 0.0
        && request_slo.minimum_good_request_ratio <= 1.0)
    {
        return invalid(format!(
            "bench {id:?} minimum_good_request_ratio must be finite and in (0, 1]"
        ));
    }
    Ok(())
}

pub(super) fn validate_bench_common(
    id: &str,
    request_source: Option<&BenchRequestSource>,
    request_body: &BTreeMap<String, JsonValue>,
    burstiness: Option<f64>,
    timeout_seconds: u64,
) -> Result<(), InferlabError> {
    match request_source {
        None => {}
        Some(request_source) => match request_source {
            BenchRequestSource::Random {
                input_tokens,
                output_tokens,
                prefix_sharing,
            } => {
                validate_bench_token_selector(id, "request_source.input_tokens", input_tokens)?;
                validate_bench_token_selector(id, "request_source.output_tokens", output_tokens)?;
                if matches!(output_tokens, BenchTokenSelector::InclusiveUniform { min: 1, max } if *max >= 2)
                {
                    return invalid(format!(
                        "bench {id:?} request_source.output_tokens must not span TPOT-inapplicable and TPOT-applicable values"
                    ));
                }
                if let Some(prefix_sharing) = prefix_sharing {
                    let Some(input_tokens) = input_tokens.fixed_value() else {
                        return invalid(format!(
                            "bench {id:?} request_source prefix sharing requires fixed input_tokens"
                        ));
                    };
                    let ratio = prefix_sharing.shared_prefix_ratio;
                    if !(ratio.is_finite() && ratio > 0.0 && ratio < 1.0) {
                        return invalid(format!(
                            "bench {id:?} request_source.prefix_sharing.shared_prefix_ratio must be finite and in (0, 1)"
                        ));
                    }
                    let shared_prefix_tokens = (f64::from(input_tokens) * ratio).floor() as u32;
                    if shared_prefix_tokens == 0 {
                        return invalid(format!(
                            "bench {id:?} request_source shared prefix must resolve to at least one token"
                        ));
                    }
                }
            }
            BenchRequestSource::RandomMixture { shapes } => {
                if shapes.len() < 2 {
                    return invalid(format!(
                        "bench {id:?} request_source random_mixture requires at least two shapes"
                    ));
                }
                let mut identities = BTreeSet::new();
                let mut total_weight = 0_u64;
                let first_tpot = BenchTpotApplicability::from_output_tokens(
                    shapes.first().map_or(0, |shape| shape.output_tokens),
                );
                for (index, shape) in shapes.iter().enumerate() {
                    require_positive(
                        &format!("request_source.shapes[{index}].input_tokens"),
                        id,
                        u64::from(shape.input_tokens),
                    )?;
                    require_positive(
                        &format!("request_source.shapes[{index}].output_tokens"),
                        id,
                        u64::from(shape.output_tokens),
                    )?;
                    require_positive(
                        &format!("request_source.shapes[{index}].weight"),
                        id,
                        u64::from(shape.weight),
                    )?;
                    if !identities.insert((shape.input_tokens, shape.output_tokens)) {
                        return invalid(format!(
                            "bench {id:?} request_source random_mixture contains duplicate shape ({}, {})",
                            shape.input_tokens, shape.output_tokens
                        ));
                    }
                    total_weight = total_weight
                    .checked_add(u64::from(shape.weight))
                    .ok_or_else(|| InferlabError::InvalidConfig {
                        message: format!(
                            "bench {id:?} request_source random_mixture total weight exceeds the supported unsigned 64-bit range"
                        ),
                    })?;
                    if BenchTpotApplicability::from_output_tokens(shape.output_tokens) != first_tpot
                    {
                        return invalid(format!(
                            "bench {id:?} request_source random_mixture must not mix TPOT-applicable and TPOT-inapplicable shapes"
                        ));
                    }
                }
            }
            BenchRequestSource::Dataset {
                dataset,
                profile,
                max_input_tokens,
                output_tokens,
            } => {
                let catalog = bench_dataset_catalog::resolve(dataset, profile.as_deref())?;
                require_positive(
                    "request_source.max_input_tokens",
                    id,
                    u64::from(*max_input_tokens),
                )?;
                if let Some(output_tokens) = output_tokens {
                    require_positive(
                        "request_source.output_tokens",
                        id,
                        u64::from(*output_tokens),
                    )?;
                } else if !catalog.provides_output_targets {
                    return invalid(format!(
                        "bench {id:?} dataset {dataset:?} profile {:?} requires fixed output_tokens because its release catalog entry provides no held-out targets",
                        profile.as_deref()
                    ));
                }
            }
        },
    }
    validate_request_body(
        "bench",
        id,
        request_body,
        &["min_tokens", "min_new_tokens", "ignore_eos"],
    )?;
    if burstiness.is_some_and(|value| !value.is_finite() || value <= 0.0) {
        return invalid(format!(
            "bench {id:?} burstiness must be positive and finite"
        ));
    }
    require_positive("timeout_seconds", id, timeout_seconds)
}

fn validate_bench_session_source(
    id: &str,
    source: &BenchSessionSource,
) -> Result<(), InferlabError> {
    let catalog =
        bench_dataset_catalog::resolve_session(&source.dataset, source.profile.as_deref())?;
    require_positive(
        "session_source.max_input_tokens",
        id,
        u64::from(source.max_input_tokens),
    )?;
    if let Some(output_tokens) = source.output_tokens {
        require_positive("session_source.output_tokens", id, u64::from(output_tokens))?;
    } else if !catalog.provides_output_targets {
        return invalid(format!(
            "bench {id:?} session dataset {:?} profile {:?} requires fixed output_tokens because its release catalog entry provides no held-out targets",
            source.dataset,
            source.profile.as_deref()
        ));
    }
    if !source.inter_turn_delay_scale.is_finite() || source.inter_turn_delay_scale < 0.0 {
        return invalid(format!(
            "bench {id:?} session_source.inter_turn_delay_scale must be finite and non-negative"
        ));
    }
    if source
        .max_inter_turn_delay_seconds
        .is_some_and(|value| !value.is_finite() || value < 0.0)
    {
        return invalid(format!(
            "bench {id:?} session_source.max_inter_turn_delay_seconds must be finite and non-negative"
        ));
    }
    Ok(())
}

pub(super) fn validate_bench_token_selector(
    id: &str,
    label: &str,
    selector: &BenchTokenSelector,
) -> Result<(), InferlabError> {
    match selector {
        BenchTokenSelector::Fixed(value) => require_positive(label, id, u64::from(*value)),
        BenchTokenSelector::InclusiveUniform { min, max } => {
            require_positive(&format!("{label}.min"), id, u64::from(*min))?;
            require_positive(&format!("{label}.max"), id, u64::from(*max))?;
            if min >= max {
                return invalid(format!(
                    "bench {id:?} {label} inclusive_uniform requires min less than max"
                ));
            }
            Ok(())
        }
    }
}

pub(super) fn validate_request_body(
    kind: &str,
    id: &str,
    request_body: &BTreeMap<String, JsonValue>,
    additional_reserved: &[&str],
) -> Result<(), InferlabError> {
    const RESERVED: [&str; 8] = [
        "model",
        "prompt",
        "messages",
        "stream",
        "n",
        "max_tokens",
        "max_completion_tokens",
        "stop",
    ];
    if let Some(member) = RESERVED
        .iter()
        .chain(additional_reserved)
        .find(|member| request_body.contains_key(**member))
    {
        return invalid(format!(
            "{kind} {id:?} request_body.{member} conflicts with a measurement-runtime-owned request member"
        ));
    }
    for (member, value) in request_body {
        validate_request_body_value(kind, id, &format!("request_body.{member}"), value)?;
    }
    Ok(())
}

pub(super) fn validate_request_body_value(
    kind: &str,
    id: &str,
    path: &str,
    value: &JsonValue,
) -> Result<(), InferlabError> {
    match value {
        JsonValue::Float(value) if !value.is_finite() => {
            invalid(format!("{kind} {id:?} {path} must be a finite JSON number"))
        }
        JsonValue::Array(values) => {
            for (index, value) in values.iter().enumerate() {
                validate_request_body_value(kind, id, &format!("{path}[{index}]"), value)?;
            }
            Ok(())
        }
        JsonValue::Object(values) => {
            for (member, value) in values {
                validate_request_body_value(kind, id, &format!("{path}.{member}"), value)?;
            }
            Ok(())
        }
        _ => Ok(()),
    }
}

pub(super) fn validate_request_rates(id: &str, rates: &[RequestRate]) -> Result<(), InferlabError> {
    if rates
        .iter()
        .filter_map(RequestRate::finite)
        .any(|rate| !rate.is_finite() || rate <= 0.0)
    {
        return invalid(format!(
            "bench {id:?} request rates must be positive and finite"
        ));
    }
    Ok(())
}

pub(super) fn validate_rate_count_policy(
    id: &str,
    has_rate_cases: bool,
    has_unbounded_rate: bool,
    request_count: Option<u32>,
    duration_seconds: Option<u64>,
) -> Result<(), InferlabError> {
    if !has_rate_cases {
        if request_count.is_some() || duration_seconds.is_some() {
            return invalid(format!(
                "bench {id:?} sets a request-rate count policy without request-rate cases"
            ));
        }
        return Ok(());
    }
    match (request_count, duration_seconds) {
        (Some(0), _) => invalid(format!("bench {id:?} request_count must be positive")),
        (_, Some(0)) => invalid(format!("bench {id:?} duration_seconds must be positive")),
        (Some(_), None) => Ok(()),
        (None, Some(_)) if !has_unbounded_rate => Ok(()),
        (None, Some(_)) => invalid(format!(
            "bench {id:?} cannot combine an unbounded request rate with duration_seconds"
        )),
        _ => invalid(format!(
            "bench {id:?} request-rate cases require exactly one of request_count or duration_seconds"
        )),
    }
}

pub(super) fn require_positive(field: &str, id: &str, value: u64) -> Result<(), InferlabError> {
    if value == 0 {
        invalid(format!("definition {id:?} {field} must be positive"))
    } else {
        Ok(())
    }
}

pub(super) fn require_optional_positive(
    field: &str,
    id: &str,
    value: Option<u64>,
) -> Result<(), InferlabError> {
    value.map_or(Ok(()), |value| require_positive(field, id, value))
}

pub(super) fn require_reference<T>(
    label: &str,
    id: &str,
    definitions: &BTreeMap<String, T>,
) -> Result<(), InferlabError> {
    if definitions.contains_key(id) {
        Ok(())
    } else {
        invalid(format!("unknown {label} {id:?}"))
    }
}

pub(super) fn require_id(label: &str, id: &str) -> Result<(), InferlabError> {
    if !id.is_empty()
        && id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
    {
        Ok(())
    } else {
        invalid(format!("invalid {label} identifier {id:?}"))
    }
}

pub(super) fn require_nonempty(label: &str, id: &str, value: &str) -> Result<(), InferlabError> {
    if value.is_empty() {
        invalid(format!("{label} for {id:?} must not be empty"))
    } else {
        Ok(())
    }
}

/// Escape options that name a managed profiler fact are rejected at load
/// ([[RFC-0004:C-WORKLOAD-PROFILING]]): session identity, report
/// storage/export/overwrite lifecycle, capture-range mechanics, launch
/// wait, and the free-list forms of the dedicated trace, sampling, and
/// context-switch fields — in long, short, and attached short-option-value
/// forms, because nsys 2026.3.1 parses -tnone as --trace=none. Shorthands
/// follow that nsys: launch carries -t for --trace; start carries -o, -f,
/// -c, and -s. Launch's -w is --show-output and -e is --env-var, so neither
/// is rejected. Environment keys must be POSIX identifiers so no key can be
/// parsed as an option of the environment utility.
/// The managed and dedicated-field option names of the profiler escape gate
/// ([[RFC-0004:C-WORKLOAD-PROFILING]]). The strict-prefix abbreviation rule
/// was checked against the qualified nsys 2026.3.1 launch and start option
/// surfaces at qualification (no legitimate option is a strict prefix of a
/// managed name); re-check by hand when the qualified nsys version changes
/// ([[ADR-0006]]).
pub(super) const MANAGED_ESCAPE_OPTIONS: &[&str] = &[
    "--session",
    "--session-new",
    "--output",
    "-o",
    "--export",
    "--force-overwrite",
    "-f",
    "--capture-range",
    "-c",
    "--capture-range-end",
    "--wait",
    "--trace",
    "-t",
    "--sample",
    "-s",
    "--cpuctxsw",
];

pub(super) fn validate_profiler_escapes(
    context: &str,
    escapes: &ProfilerEscapes,
) -> Result<(), InferlabError> {
    const MANAGED: &[&str] = MANAGED_ESCAPE_OPTIONS;
    const MANAGED_SHORT: &[&str] = &["-t", "-o", "-f", "-c", "-s"];
    for (field, options) in [
        ("launch_options", &escapes.nsys.launch_options),
        ("start_options", &escapes.nsys.start_options),
    ] {
        for option in options {
            // A standalone terminator ends option parsing and displaces the
            // managed argv tail into positionals of the wrapped command
            // ([[RFC-0004:C-WORKLOAD-PROFILING]]).
            if option == "-" || option == "--" {
                return invalid(format!(
                    "{context} nsys {field} contains standalone {option:?}, which ends \
                     option parsing and displaces the inferlab-managed argv tail"
                ));
            }
            let name = option.split('=').next().unwrap_or(option.as_str());
            let attached = !name.starts_with("--")
                && MANAGED_SHORT
                    .iter()
                    .any(|short| name.starts_with(short) && name.len() > short.len());
            // The qualified nsys resolves GNU-style abbreviations, so any
            // strict prefix of a managed long name either resolves to the
            // managed option or is an ambiguity
            // ([[RFC-0004:C-WORKLOAD-PROFILING]]).
            let abbreviated = name.starts_with("--")
                && MANAGED
                    .iter()
                    .any(|managed| managed.len() > name.len() && managed.starts_with(name));
            if MANAGED.contains(&name) || attached || abbreviated {
                return invalid(format!(
                    "{context} nsys {field} contains managed option {option:?}; use the \
                     dedicated profiler escape field or the inferlab-managed value"
                ));
            }
        }
    }
    for key in escapes.nsys.env.keys() {
        if !is_posix_identifier(key) {
            return invalid(format!(
                "{context} nsys env contains key {key:?}, which is not a POSIX identifier; \
                 environment entries reach the profiler commands as assignments"
            ));
        }
    }
    Ok(())
}

pub(super) fn is_posix_identifier(name: &str) -> bool {
    let mut characters = name.chars();
    characters
        .next()
        .is_some_and(|first| first.is_ascii_alphabetic() || first == '_')
        && characters.all(|rest| rest.is_ascii_alphanumeric() || rest == '_')
}

pub(super) fn validate_environment_script(
    root: &Path,
    environment: &str,
    label: &str,
    id: &str,
    script: &Path,
) -> Result<(), InferlabError> {
    if !is_safe_relative(script) {
        return invalid(format!(
            "environment {environment:?} {label} {id:?} script {} must be workspace-relative \
             without parent traversal",
            script.display()
        ));
    }
    let target = root.join(script);
    if !target.is_file() {
        return invalid(format!(
            "environment {environment:?} {label} {id:?} script {} does not exist",
            script.display()
        ));
    }
    // A lexically relative path can still resolve outside the workspace
    // through a symlink; scripts are workspace content, so the canonical
    // target must stay inside the (already canonical) root.
    let canonical = fs::canonicalize(&target).map_err(|source| InferlabError::Read {
        path: target,
        source,
    })?;
    if !canonical.starts_with(root) {
        return invalid(format!(
            "environment {environment:?} {label} {id:?} script {} resolves outside the workspace",
            script.display()
        ));
    }
    Ok(())
}
