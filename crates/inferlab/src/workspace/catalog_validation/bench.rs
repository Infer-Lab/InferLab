//! Static and adaptive serving-Bench validation.

use super::{invalid, require_nonempty, require_positive, validate_request_body};
use crate::InferlabError;
use crate::workspace::definitions::{
    AggregateSlo, BenchDefinition, BenchPrefixSharing, BenchPrompt, BenchRequestSource,
    BenchSessionSource, BenchSharedSystemContent, BenchTokenSelector, BenchTpotApplicability,
    JsonValue, RequestRate, RequestSlo,
};
use crate::{bench_agentic_catalog, bench_dataset_catalog};
use std::collections::{BTreeMap, BTreeSet};

pub(crate) fn validate_bench(id: &str, definition: &BenchDefinition) -> Result<(), InferlabError> {
    match definition {
        BenchDefinition::Serving {
            request_source,
            session_source,
            agentic_source,
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
            reset_prefix_cache,
            timeout_seconds,
            ..
        } => {
            if [
                request_source.is_some(),
                session_source.is_some(),
                agentic_source.is_some(),
            ]
            .into_iter()
            .filter(|selected| *selected)
            .count()
                != 1
            {
                return invalid(format!(
                    "bench {id:?} requires exactly one of request_source, session_source, and agentic_source"
                ));
            }
            if let Some(agentic_source) = agentic_source {
                require_nonempty("agentic dataset", id, &agentic_source.dataset)?;
                require_nonempty("agentic profile", id, &agentic_source.profile)?;
                let catalog = bench_agentic_catalog::resolve(
                    &agentic_source.dataset,
                    &agentic_source.profile,
                )?;
                if concurrency.is_empty() || concurrency.contains(&0) {
                    return invalid(format!(
                        "bench {id:?} agentic_source requires non-empty positive concurrency"
                    ));
                }
                if prompts_per_concurrency.is_some()
                    || *warmup_prompts_per_concurrency != 0
                    || sessions_per_concurrency.is_some()
                    || *warmup_sessions_per_concurrency != 0
                    || !request_rates.is_empty()
                    || request_count.is_some()
                    || burstiness.is_some()
                    || !request_body.is_empty()
                    || !aggregate_slos.is_empty()
                    || request_slo.is_some()
                    || *reset_prefix_cache
                {
                    return invalid(format!(
                        "bench {id:?} agentic_source rejects prompts_per_concurrency, warmup_prompts_per_concurrency, sessions_per_concurrency, warmup_sessions_per_concurrency, request_rates, request_count, burstiness, request_body, aggregate_slos, request_slo, and reset_prefix_cache"
                    ));
                }
                if duration_seconds
                    .is_some_and(|duration| duration < catalog.policy.minimum_duration_seconds)
                {
                    return invalid(format!(
                        "bench {id:?} duration_seconds must be at least {} for agentic profile {:?}",
                        catalog.policy.minimum_duration_seconds, agentic_source.profile
                    ));
                }
                return Ok(());
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

fn validate_request_rates(id: &str, rates: &[RequestRate]) -> Result<(), InferlabError> {
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

fn validate_rate_count_policy(
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

pub(in crate::workspace) fn validate_bench_slos(
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

fn validate_bench_common(
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
                prompt,
                input_tokens,
                output_tokens,
                prefix_sharing,
                shared_system_content,
            } => {
                validate_bench_token_selector(id, "request_source.input_tokens", input_tokens)?;
                validate_bench_token_selector(id, "request_source.output_tokens", output_tokens)?;
                if matches!(output_tokens, BenchTokenSelector::InclusiveUniform { min: 1, max } if *max >= 2)
                {
                    return invalid(format!(
                        "bench {id:?} request_source.output_tokens must not span TPOT-inapplicable and TPOT-applicable values"
                    ));
                }
                validate_synthetic_prompt(
                    id,
                    prompt.effective(),
                    prefix_sharing.as_ref(),
                    shared_system_content.as_ref(),
                    input_tokens.minimum(),
                    request_body,
                )?;
            }
            BenchRequestSource::RandomMixture {
                prompt,
                shapes,
                prefix_sharing,
            } => {
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
                let minimum_input = shapes
                    .iter()
                    .map(|shape| shape.input_tokens)
                    .min()
                    .unwrap_or(0);
                validate_synthetic_prompt(
                    id,
                    prompt.effective(),
                    prefix_sharing.as_ref(),
                    None,
                    minimum_input,
                    request_body,
                )?;
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

fn validate_synthetic_prompt(
    id: &str,
    prompt: &BenchPrompt,
    prefix_sharing: Option<&BenchPrefixSharing>,
    shared_system_content: Option<&BenchSharedSystemContent>,
    minimum_input_tokens: u32,
    request_body: &BTreeMap<String, JsonValue>,
) -> Result<(), InferlabError> {
    let locally_rendered = matches!(prompt, BenchPrompt::Flat | BenchPrompt::RenderedChat { .. });
    if locally_rendered {
        for member in ["chat_template", "chat_template_kwargs"] {
            if request_body.contains_key(member) {
                return invalid(format!(
                    "bench {id:?} request_body.{member} conflicts with request_source.prompt local rendering authority"
                ));
            }
        }
        if shared_system_content.is_some() {
            return invalid(format!(
                "bench {id:?} request_source.shared_system_content requires prompt.kind = \"server_chat\""
            ));
        }
    } else if prefix_sharing.is_some() {
        return invalid(format!(
            "bench {id:?} request_source.prefix_sharing requires prompt.kind = \"flat\" or \"rendered_chat\""
        ));
    }

    if let Some(sharing) = prefix_sharing {
        match sharing {
            BenchPrefixSharing::Tokens {
                shared_prefix_tokens,
            } if *shared_prefix_tokens > minimum_input_tokens => {
                return invalid(format!(
                    "bench {id:?} request_source.prefix_sharing.shared_prefix_tokens must not exceed the minimum input-token target {minimum_input_tokens}"
                ));
            }
            BenchPrefixSharing::Ratio {
                shared_prefix_ratio,
            } if !shared_prefix_ratio.is_finite()
                || *shared_prefix_ratio < 0.0
                || *shared_prefix_ratio > 1.0 =>
            {
                return invalid(format!(
                    "bench {id:?} request_source.prefix_sharing.shared_prefix_ratio must be finite and in [0, 1]"
                ));
            }
            _ => {}
        }
    }

    if let Some(sharing) = shared_system_content {
        match sharing {
            BenchSharedSystemContent::Tokens { tokens }
                if *tokens == 0 || *tokens >= minimum_input_tokens =>
            {
                return invalid(format!(
                    "bench {id:?} request_source.shared_system_content.tokens must be positive and less than every input-token target"
                ));
            }
            BenchSharedSystemContent::Ratio { ratio }
                if !ratio.is_finite()
                    || *ratio <= 0.0
                    || *ratio >= 1.0
                    || (f64::from(minimum_input_tokens) * ratio).floor() < 1.0 =>
            {
                return invalid(format!(
                    "bench {id:?} request_source.shared_system_content.ratio must be finite in (0, 1) and resolve to a positive system-content length for every input target"
                ));
            }
            _ => {}
        }
    }
    Ok(())
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

fn validate_bench_token_selector(
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
