//! Static-matrix and adaptive Bench execution-case planning.

use crate::InferlabError;
use crate::bench_agentic_catalog;
use crate::workload::plan::{
    BenchCasePlan, BenchExecutionPlan, LoadShape, session_population_layout,
};
use crate::workspace::{BenchDefinition, RequestRate};

pub(super) fn resolve_bench_execution(
    id: &str,
    definition: &BenchDefinition,
) -> Result<BenchExecutionPlan, InferlabError> {
    match definition {
        BenchDefinition::Serving {
            agentic_source,
            session_source,
            concurrency,
            prompts_per_concurrency,
            warmup_prompts_per_concurrency,
            sessions_per_concurrency,
            warmup_sessions_per_concurrency,
            request_rates,
            request_count,
            duration_seconds,
            burstiness,
            ..
        } => {
            let mut cases = Vec::with_capacity(concurrency.len() + request_rates.len());
            for (index, concurrency) in concurrency.iter().copied().enumerate() {
                if let Some(agentic_source) = agentic_source {
                    let profile = bench_agentic_catalog::resolve(
                        &agentic_source.dataset,
                        &agentic_source.profile,
                    )?;
                    cases.push(BenchCasePlan {
                        id: format!("concurrency-{index:03}"),
                        load_shape: LoadShape::ConcurrencyLimited { concurrency },
                        request_count: 0,
                        warmup_request_count: 0,
                        duration_seconds: Some(
                            duration_seconds.unwrap_or(profile.policy.default_duration_seconds),
                        ),
                        session_count: None,
                        warmup_session_count: None,
                    });
                    continue;
                }
                if session_source.is_some() {
                    let multiplier =
                        sessions_per_concurrency.ok_or_else(|| InferlabError::InvalidConfig {
                            message: format!("bench {id:?} is missing sessions_per_concurrency"),
                        })?;
                    let session_count = concurrency.checked_mul(multiplier).ok_or_else(|| {
                        InferlabError::InvalidConfig {
                            message: format!("bench {id:?} profiling session count exceeds u32"),
                        }
                    })?;
                    let warmup_session_count = concurrency
                        .checked_mul(*warmup_sessions_per_concurrency)
                        .ok_or_else(|| InferlabError::InvalidConfig {
                            message: format!("bench {id:?} warmup session count exceeds u32"),
                        })?;
                    cases.push(BenchCasePlan {
                        id: format!("concurrency-{index:03}"),
                        load_shape: LoadShape::ConcurrencyLimited { concurrency },
                        request_count: 0,
                        warmup_request_count: 0,
                        duration_seconds: None,
                        session_count: Some(session_count),
                        warmup_session_count: Some(warmup_session_count),
                    });
                    continue;
                }
                let multiplier =
                    prompts_per_concurrency.ok_or_else(|| InferlabError::InvalidConfig {
                        message: format!("bench {id:?} is missing prompts_per_concurrency"),
                    })?;
                let request_count = concurrency.checked_mul(multiplier).ok_or_else(|| {
                    InferlabError::InvalidConfig {
                        message: format!("bench {id:?} concurrency request count exceeds u32"),
                    }
                })?;
                let warmup_request_count = concurrency
                    .checked_mul(*warmup_prompts_per_concurrency)
                    .ok_or_else(|| InferlabError::InvalidConfig {
                        message: format!("bench {id:?} warmup request count exceeds u32"),
                    })?;
                cases.push(BenchCasePlan {
                    id: format!("concurrency-{index:03}"),
                    load_shape: LoadShape::ConcurrencyLimited { concurrency },
                    request_count,
                    warmup_request_count,
                    duration_seconds: None,
                    session_count: None,
                    warmup_session_count: None,
                });
            }
            for (index, rate) in request_rates.iter().cloned().enumerate() {
                let count = resolved_request_count(id, &rate, *request_count, *duration_seconds)?;
                cases.push(BenchCasePlan {
                    id: format!("request-rate-{index:03}"),
                    load_shape: LoadShape::RequestRateLimited {
                        request_rate: rate,
                        burstiness: *burstiness,
                    },
                    request_count: count,
                    warmup_request_count: 0,
                    duration_seconds: *duration_seconds,
                    session_count: None,
                    warmup_session_count: None,
                });
            }
            Ok(BenchExecutionPlan::Matrix { cases })
        }
        BenchDefinition::AdaptiveServing {
            initial_request_rates,
            max_search_steps,
            min_rate_resolution,
            request_count,
            duration_seconds,
            ..
        } => {
            let mut initial_request_rates = initial_request_rates.clone();
            initial_request_rates.sort_by(f64::total_cmp);
            initial_request_rates.dedup();
            Ok(BenchExecutionPlan::Adaptive {
                policy: "highest-feasible-rate-v1".to_owned(),
                initial_request_rates,
                max_search_steps: *max_search_steps,
                min_rate_resolution: *min_rate_resolution,
                request_count: *request_count,
                duration_seconds: *duration_seconds,
            })
        }
    }
}

pub(super) fn required_population_count(
    id: &str,
    execution: &BenchExecutionPlan,
) -> Result<u32, InferlabError> {
    match execution {
        BenchExecutionPlan::Matrix { cases } => cases.iter().try_fold(0_u32, |largest, case| {
            let entries = match (case.warmup_session_count, case.session_count) {
                (Some(warmup), Some(profiling)) => session_population_layout(warmup, profiling)
                    .map(|layout| layout.required_entries),
                _ => case.warmup_request_count.checked_add(case.request_count),
            }
            .ok_or_else(|| InferlabError::InvalidConfig {
                message: format!("bench {id:?} population exceeds u32"),
            })?;
            Ok(largest.max(entries))
        }),
        BenchExecutionPlan::Adaptive {
            initial_request_rates,
            max_search_steps,
            request_count,
            duration_seconds,
            ..
        } => {
            if let Some(request_count) = request_count {
                return Ok(*request_count);
            }
            let initial = initial_request_rates
                .iter()
                .copied()
                .max_by(f64::total_cmp)
                .ok_or_else(|| InferlabError::InvalidConfig {
                    message: format!("adaptive bench {id:?} has no initial request rate"),
                })?;
            let factor = 2_f64.powf(f64::from(*max_search_steps));
            let largest_rate = initial * factor;
            if !largest_rate.is_finite() {
                return Err(InferlabError::InvalidConfig {
                    message: format!("adaptive bench {id:?} request population exceeds u32"),
                });
            }
            resolved_request_count(
                id,
                &RequestRate::Finite(largest_rate),
                None,
                *duration_seconds,
            )
        }
    }
}

pub(crate) fn resolved_request_count(
    bench_id: &str,
    rate: &RequestRate,
    request_count: Option<u32>,
    duration_seconds: Option<u64>,
) -> Result<u32, InferlabError> {
    if let Some(request_count) = request_count {
        return Ok(request_count);
    }
    let rate = rate.finite().ok_or_else(|| InferlabError::InvalidConfig {
        message: format!("bench {bench_id:?} cannot derive request count for an unbounded rate"),
    })?;
    let duration = duration_seconds.ok_or_else(|| InferlabError::InvalidConfig {
        message: format!("bench {bench_id:?} has no request count policy"),
    })?;
    let count = (rate * duration as f64).ceil().max(1.0);
    if count > f64::from(u32::MAX) {
        return Err(InferlabError::InvalidConfig {
            message: format!("bench {bench_id:?} request count exceeds u32"),
        });
    }
    Ok(count as u32)
}
