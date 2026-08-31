//! Static and adaptive serving-Bench validation.

use super::{
    invalid, require_nonempty, require_positive, validate_expected_digest, validate_request_body,
    validate_workspace_relative_source_path,
};
use crate::InferlabError;
use crate::workspace::definitions::{
    AggregateSlo, BenchCacheStart, BenchDefinition, BenchPrefixSharing, BenchPrompt,
    BenchRequestSource, BenchSessionSource, BenchSharedSystemContent, BenchTokenSelector,
    BenchTpotApplicability, JsonValue, RequestRate, RequestSlo,
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
            cache,
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
                    || cache.is_some()
                {
                    return invalid(format!(
                        "bench {id:?} agentic_source rejects prompts_per_concurrency, warmup_prompts_per_concurrency, sessions_per_concurrency, warmup_sessions_per_concurrency, request_rates, request_count, burstiness, request_body, aggregate_slos, request_slo, and cache"
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
                validate_cache_policy(id, cache.as_ref(), request_source.as_ref())?;
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
            validate_cache_policy(id, cache.as_ref(), request_source.as_ref())?;
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
            cache,
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
            validate_cache_policy(id, cache.as_ref(), Some(request_source))?;
            validate_rate_count_policy(id, true, false, *request_count, *duration_seconds)
        }
    }
}

fn validate_cache_policy(
    id: &str,
    policy: Option<&crate::workspace::definitions::BenchCachePolicy>,
    request_source: Option<&BenchRequestSource>,
) -> Result<(), InferlabError> {
    let Some(policy) = policy else {
        return Ok(());
    };
    if policy.start != BenchCacheStart::Primed {
        return Ok(());
    }
    let has_positive_prefix = match request_source {
        Some(BenchRequestSource::Random {
            prompt,
            input_tokens,
            prefix_sharing: Some(sharing),
            ..
        }) if matches!(
            prompt.effective(),
            BenchPrompt::Flat | BenchPrompt::RenderedChat { .. }
        ) =>
        {
            match sharing {
                BenchPrefixSharing::Tokens {
                    shared_prefix_tokens,
                } => *shared_prefix_tokens > 0,
                BenchPrefixSharing::Ratio {
                    shared_prefix_ratio,
                } => (f64::from(input_tokens.minimum()) * shared_prefix_ratio).floor() >= 1.0,
            }
        }
        Some(BenchRequestSource::RandomMixture {
            prompt,
            shapes,
            prefix_sharing: Some(sharing),
        }) if matches!(
            prompt.effective(),
            BenchPrompt::Flat | BenchPrompt::RenderedChat { .. }
        ) =>
        {
            let minimum_input = shapes
                .iter()
                .map(|shape| shape.input_tokens)
                .min()
                .unwrap_or_default();
            match sharing {
                BenchPrefixSharing::Tokens {
                    shared_prefix_tokens,
                } => *shared_prefix_tokens > 0,
                BenchPrefixSharing::Ratio {
                    shared_prefix_ratio,
                } => (f64::from(minimum_input) * shared_prefix_ratio).floor() >= 1.0,
            }
        }
        // Replay input lengths live in the population file, so a declared
        // positive ratio is accepted here and preparation resolves the
        // geometry from the file entries.
        Some(BenchRequestSource::Replay {
            prompt,
            prefix_sharing: Some(sharing),
            ..
        }) if matches!(
            prompt.effective(),
            BenchPrompt::Flat | BenchPrompt::RenderedChat { .. }
        ) =>
        {
            match sharing {
                BenchPrefixSharing::Tokens {
                    shared_prefix_tokens,
                } => *shared_prefix_tokens > 0,
                BenchPrefixSharing::Ratio {
                    shared_prefix_ratio,
                } => *shared_prefix_ratio > 0.0,
            }
        }
        _ => false,
    };
    if has_positive_prefix {
        Ok(())
    } else {
        invalid(format!(
            "bench {id:?} cache.start = \"primed\" requires flat or rendered_chat prompt authority with positive prefix_sharing"
        ))
    }
}

fn validate_replay_path(id: &str, path: &str) -> Result<(), InferlabError> {
    validate_workspace_relative_source_path(
        &format!("bench {id:?}"),
        "replay request_source.path",
        path,
    )
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
                corpus,
            } => {
                validate_bench_token_selector(id, "request_source.input_tokens", input_tokens)?;
                validate_bench_token_selector(id, "request_source.output_tokens", output_tokens)?;
                if matches!(output_tokens, BenchTokenSelector::InclusiveUniform { min: 1, max } if *max >= 2)
                {
                    return invalid(format!(
                        "bench {id:?} request_source.output_tokens must not span TPOT-inapplicable and TPOT-applicable values"
                    ));
                }
                if let Some(corpus) = corpus {
                    validate_workspace_relative_source_path(
                        &format!("bench {id:?}"),
                        "random request_source.corpus.path",
                        &corpus.path,
                    )?;
                    if let Some(digest) = &corpus.expected_sha256 {
                        validate_expected_digest(
                            &format!("bench {id:?}"),
                            "request_source.corpus.expected_sha256",
                            digest,
                        )?;
                    }
                    // Corpus slices are exact final-prompt token streams; a
                    // chat template would insert tokens between them, so the
                    // corpus supply exists only for flat prompts.
                    if !matches!(prompt.effective(), BenchPrompt::Flat) {
                        return invalid(format!(
                            "bench {id:?} random request_source.corpus requires prompt.kind = \"flat\""
                        ));
                    }
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
            BenchRequestSource::Replay {
                path,
                expected_sha256,
                prompt,
                prefix_sharing,
            } => {
                validate_replay_path(id, path)?;
                if let Some(digest) = expected_sha256 {
                    validate_expected_digest(
                        &format!("bench {id:?}"),
                        "request_source.expected_sha256",
                        digest,
                    )?;
                }
                if prompt.declared().is_none() {
                    return invalid(format!(
                        "bench {id:?} replay request_source requires an explicit prompt kind"
                    ));
                }
                if let Some(sharing) = prefix_sharing {
                    if !matches!(
                        prompt.effective(),
                        BenchPrompt::Flat | BenchPrompt::RenderedChat { .. }
                    ) {
                        return invalid(format!(
                            "bench {id:?} replay request_source.prefix_sharing requires prompt.kind = \"flat\" or \"rendered_chat\""
                        ));
                    }
                    match sharing {
                        BenchPrefixSharing::Tokens {
                            shared_prefix_tokens,
                        } if *shared_prefix_tokens == 0 => {
                            return invalid(format!(
                                "bench {id:?} request_source.prefix_sharing.shared_prefix_tokens must be positive"
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::workspace::definitions::BenchRandomShape;

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
                corpus: None,
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
    fn replay_source_validates_its_declaration_shape() -> Result<(), Box<dyn std::error::Error>> {
        let definition = toml::from_str::<BenchDefinition>(
            r#"
kind = "serving"
request_source = { kind = "replay", path = "populations/x.jsonl", expected_sha256 = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", prompt = { kind = "flat" }, prefix_sharing = { shared_prefix_ratio = 1.0 } }
concurrency = [1]
prompts_per_concurrency = 1
cache = { start = "primed" }
timeout_seconds = 60
"#,
        )?;

        validate_bench("replay", &definition)?;
        let BenchDefinition::Serving { request_source, .. } = definition else {
            return Err(std::io::Error::other("expected a serving Bench").into());
        };
        assert!(matches!(
            request_source,
            Some(BenchRequestSource::Replay {
                ref path,
                prompt,
                prefix_sharing: Some(BenchPrefixSharing::Ratio {
                    shared_prefix_ratio: 1.0,
                }),
                ..
            }) if path == "populations/x.jsonl" && prompt.declared() == Some(&BenchPrompt::Flat)
        ));
        Ok(())
    }

    #[test]
    fn replay_source_requires_an_explicit_prompt_kind() -> Result<(), Box<dyn std::error::Error>> {
        let definition = toml::from_str::<BenchDefinition>(
            r#"
kind = "serving"
request_source = { kind = "replay", path = "populations/x.jsonl" }
concurrency = [1]
prompts_per_concurrency = 1
timeout_seconds = 60
"#,
        )?;

        let error = validate_bench("replay", &definition)
            .err()
            .ok_or("replay source accepted the defaulted prompt kind")?;
        assert!(
            error.to_string().contains("explicit prompt kind"),
            "{error}"
        );
        Ok(())
    }

    #[test]
    fn replay_source_rejects_unsafe_paths_and_digests() -> Result<(), Box<dyn std::error::Error>> {
        for (label, path) in [
            ("absolute", "/tmp/population.jsonl"),
            ("parent-escape", "../outside.jsonl"),
            ("empty", ""),
        ] {
            let definition = toml::from_str::<BenchDefinition>(&format!(
                r#"
kind = "serving"
request_source = {{ kind = "replay", path = {path:?}, prompt = {{ kind = "flat" }} }}
concurrency = [1]
prompts_per_concurrency = 1
timeout_seconds = 60
"#
            ))?;
            let error = validate_bench("replay", &definition)
                .err()
                .ok_or_else(|| format!("replay source accepted a {label} path"))?;
            assert!(
                error.to_string().contains("workspace-relative"),
                "{label}: {error}"
            );
        }
        let bad_digest = toml::from_str::<BenchDefinition>(
            r#"
kind = "serving"
request_source = { kind = "replay", path = "populations/x.jsonl", expected_sha256 = "ABCDEF", prompt = { kind = "flat" } }
concurrency = [1]
prompts_per_concurrency = 1
timeout_seconds = 60
"#,
        )?;
        let error = validate_bench("replay", &bad_digest)
            .err()
            .ok_or("replay source accepted a malformed expected digest")?;
        assert!(
            error.to_string().contains("64 lowercase hexadecimal"),
            "{error}"
        );
        Ok(())
    }

    #[test]
    fn replay_source_gates_prefix_sharing_on_the_prompt_kind()
    -> Result<(), Box<dyn std::error::Error>> {
        let server_chat = toml::from_str::<BenchDefinition>(
            r#"
kind = "serving"
request_source = { kind = "replay", path = "populations/x.jsonl", prompt = { kind = "server_chat" }, prefix_sharing = { shared_prefix_tokens = 8 } }
concurrency = [1]
prompts_per_concurrency = 1
timeout_seconds = 60
"#,
        )?;
        let error = validate_bench("replay", &server_chat)
            .err()
            .ok_or("server-chat replay accepted prefix_sharing")?;
        assert!(
            error.to_string().contains("flat\" or \"rendered_chat"),
            "{error}"
        );

        let zero_tokens = toml::from_str::<BenchDefinition>(
            r#"
kind = "serving"
request_source = { kind = "replay", path = "populations/x.jsonl", prompt = { kind = "flat" }, prefix_sharing = { shared_prefix_tokens = 0 } }
concurrency = [1]
prompts_per_concurrency = 1
timeout_seconds = 60
"#,
        )?;
        let error = validate_bench("replay", &zero_tokens)
            .err()
            .ok_or("replay accepted a zero shared prefix")?;
        assert!(error.to_string().contains("must be positive"), "{error}");
        Ok(())
    }

    #[test]
    fn random_corpus_validates_its_declaration_shape() -> Result<(), Box<dyn std::error::Error>> {
        let definition = toml::from_str::<BenchDefinition>(
            r#"
kind = "serving"
request_source = { kind = "random", input_tokens = 8, output_tokens = 2, corpus = { path = "corpus/shakespeare.txt", expected_sha256 = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa" }, prefix_sharing = { shared_prefix_ratio = 0.8 } }
concurrency = [1]
prompts_per_concurrency = 1
timeout_seconds = 60
"#,
        )?;

        validate_bench("corpus", &definition)?;
        let BenchDefinition::Serving { request_source, .. } = definition else {
            return Err(std::io::Error::other("expected a serving Bench").into());
        };
        assert!(matches!(
            request_source,
            Some(BenchRequestSource::Random {
                corpus: Some(_),
                ..
            })
        ));
        Ok(())
    }

    #[test]
    fn random_corpus_rejects_unsafe_paths_and_digests() -> Result<(), Box<dyn std::error::Error>> {
        for (label, path) in [
            ("empty", ""),
            ("absolute", "/tmp/corpus.txt"),
            ("escaping", "../corpus.txt"),
        ] {
            let definition = toml::from_str::<BenchDefinition>(&format!(
                r#"
kind = "serving"
request_source = {{ kind = "random", input_tokens = 8, output_tokens = 2, corpus = {{ path = {path:?} }} }}
concurrency = [1]
prompts_per_concurrency = 1
timeout_seconds = 60
"#,
            ))?;
            let error = validate_bench("corpus", &definition)
                .err()
                .ok_or_else(|| format!("corpus accepted a {label} path"))?;
            assert!(
                error.to_string().contains("workspace-relative path"),
                "{label}: {error}"
            );
        }
        let bad_digest = toml::from_str::<BenchDefinition>(
            r#"
kind = "serving"
request_source = { kind = "random", input_tokens = 8, output_tokens = 2, corpus = { path = "corpus/x.txt", expected_sha256 = "ABCDEF" } }
concurrency = [1]
prompts_per_concurrency = 1
timeout_seconds = 60
"#,
        )?;
        let error = validate_bench("corpus", &bad_digest)
            .err()
            .ok_or("corpus accepted a malformed expected digest")?;
        assert!(
            error.to_string().contains("64 lowercase hexadecimal"),
            "{error}"
        );
        Ok(())
    }

    #[test]
    fn random_corpus_requires_a_flat_prompt() -> Result<(), Box<dyn std::error::Error>> {
        for (label, prompt) in [
            ("server chat", "{ kind = \"server_chat\" }"),
            ("rendered chat", "{ kind = \"rendered_chat\" }"),
        ] {
            let definition = toml::from_str::<BenchDefinition>(&format!(
                r#"
kind = "serving"
request_source = {{ kind = "random", prompt = {prompt}, input_tokens = 8, output_tokens = 2, corpus = {{ path = "corpus/x.txt" }} }}
concurrency = [1]
prompts_per_concurrency = 1
timeout_seconds = 60
"#,
            ))?;
            let error = validate_bench("corpus", &definition)
                .err()
                .ok_or_else(|| format!("{label} prompt accepted a corpus"))?;
            assert!(
                error.to_string().contains("prompt.kind = \"flat\""),
                "{label}: {error}"
            );
        }
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
}
