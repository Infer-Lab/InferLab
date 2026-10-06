//! Controlled prefix-cache preparation and its domain evidence.

use crate::workload::BenchPrefixCacheConditioningPlan;
use crate::workload::domain::BenchPopulation;
use crate::workload::domain::{WorkloadEndpoint, WorkloadHttpAction, WorkloadReplicaReset};
use crate::workload::record::{
    BenchCachePreparationEvidence, BenchCachePreparationPhase, BenchCachePreparationTransition,
    PrefixCacheConditioningEvidence, PrefixCacheConditioningRankEvidence, PrefixCacheResetEvidence,
    PrefixCacheResetOutcome,
};
use crate::workspace::BenchCacheStart;
use inferlab_protocol::PromptCacheReadZeroRepresentation;
use inferlab_proxy::core::PrimePrefixCacheResponse;
use inferlab_runtime::operation_bound::{OperationBound, Remaining};
use serde::Deserialize;
use std::collections::BTreeMap;
use std::time::Duration;

/// Delay before the first retry of a declined reset; it doubles per round up
/// to the cap, so a reset that frees within seconds retries promptly and one
/// that waits minutes for a transfer timeout keeps its evidence short.
const DECLINED_RESET_FIRST_DELAY: Duration = Duration::from_millis(500);
const DECLINED_RESET_MAX_DELAY: Duration = Duration::from_secs(15);

struct ConditioningResponse {
    status: u16,
    prompt_tokens: Option<u64>,
    cache_read_tokens: Option<u64>,
}

#[derive(Deserialize)]
struct CompletionResponse {
    usage: Option<CompletionUsage>,
}

#[derive(Deserialize)]
struct CompletionUsage {
    prompt_tokens: Option<u64>,
    prompt_tokens_details: Option<PromptTokenDetails>,
}

#[derive(Deserialize)]
struct PromptTokenDetails {
    cached_tokens: Option<u64>,
}

fn reset_prefix_cache(
    url: String,
    action: &WorkloadHttpAction,
    process: Option<&str>,
    bound: &OperationBound,
) -> PrefixCacheResetEvidence {
    let started_ms = bound.elapsed_ms();
    let result: Result<(u16, Option<Vec<u8>>), CachePreparationError> = (|| {
        let remaining = finite_remaining(bound)?;
        let client = reqwest::blocking::Client::builder()
            .timeout(remaining)
            .connect_timeout(remaining)
            .redirect(reqwest::redirect::Policy::none())
            .no_proxy()
            .build()
            .map_err(|source| CachePreparationError::Request { source })?;
        let mut response = client
            .post(&url)
            .timeout(finite_remaining(bound)?)
            .send()
            .map_err(|source| CachePreparationError::Request { source })?;
        let status = response.status().as_u16();
        // Only a declared success predicate needs the body.
        let body = if action.success.is_some() {
            Some(
                response
                    .bytes()
                    .map_err(|source| CachePreparationError::Request { source })?
                    .to_vec(),
            )
        } else {
            response
                .copy_to(&mut std::io::sink())
                .map_err(|source| CachePreparationError::Request { source })?;
            None
        };
        finite_remaining(bound)?;
        Ok((status, body))
    })();
    let mut evidence = PrefixCacheResetEvidence {
        method: action.method,
        url,
        succeeded: false,
        http_status: None,
        error: None,
        elapsed_ms: 0,
        process: process.map(str::to_owned),
        success: action.success.clone(),
        observed_value: None,
        declined: false,
    };
    match result {
        Ok((status, body)) => {
            evidence.http_status = Some(status);
            if !is_successful_preparation_status(status) {
                evidence.error = Some(format!("prefix-cache reset returned HTTP {status}"));
            } else if let Some(success) = &action.success {
                let observed = body
                    .as_deref()
                    .and_then(|body| serde_json::from_slice::<serde_json::Value>(body).ok())
                    .and_then(|document| document.pointer(&success.pointer).cloned());
                evidence.succeeded = observed
                    .as_ref()
                    .is_some_and(|value| success.value.matches(value));
                if !evidence.succeeded {
                    // Only the opposite boolean is the framework declining
                    // for now, e.g. vLLM while KV blocks are still held; any
                    // other mismatch is a failure
                    // ([[RFC-0004:C-BENCH-CACHE-STATE]]).
                    evidence.declined = matches!(
                        (&success.value, &observed),
                        (
                            inferlab_protocol::JsonScalar::Boolean(expected),
                            Some(serde_json::Value::Bool(reported)),
                        ) if expected != reported
                    );
                    evidence.error = Some(format!(
                        "prefix-cache reset reported {} at {:?}, expected {}",
                        observed
                            .as_ref()
                            .map_or_else(|| "no value".to_owned(), ToString::to_string),
                        success.pointer,
                        success.value
                    ));
                }
                evidence.observed_value = observed;
            } else {
                evidence.succeeded = true;
            }
        }
        Err(error) => evidence.error = Some(error.to_string()),
    }
    evidence.elapsed_ms = bound.elapsed_ms().saturating_sub(started_ms);
    evidence
}

/// The public reset action, or the per-target reset attempted on every
/// model-serving replica even after an earlier target fails. A declined
/// attempt is retried — on its own target only — after a growing delay until
/// it succeeds, fails otherwise, or the case budget cannot hold another
/// delay ([[RFC-0004:C-BENCH-CACHE-STATE]]).
fn reset_outcome(
    input: &CachePreparationInput<'_>,
    bound: &OperationBound,
) -> PrefixCacheResetOutcome {
    let targets = match input.reset {
        ResetCapability::Public(action) => vec![(
            format!(
                "http://{}:{}{}",
                input.endpoint.host, input.endpoint.port, action.path
            ),
            action,
            None,
        )],
        ResetCapability::PerTarget(targets) => targets
            .iter()
            .map(|target| {
                (
                    target.url.clone(),
                    &target.action,
                    Some(target.process.as_str()),
                )
            })
            .collect(),
    };
    let started_ms = bound.elapsed_ms();
    let mut attempts = Vec::new();
    let mut failed = Vec::new();
    let mut pending = (0..targets.len()).collect::<Vec<_>>();
    let mut delay = DECLINED_RESET_FIRST_DELAY;
    let mut exhausted = false;
    while !pending.is_empty() {
        let mut declined = Vec::new();
        for index in pending {
            let (url, action, process) = &targets[index];
            let attempt = reset_prefix_cache(url.clone(), action, *process, bound);
            if attempt.declined {
                declined.push(index);
            } else if !attempt.succeeded {
                failed.push(index);
            }
            attempts.push(attempt);
        }
        // Any failure other than a decline settles the reset as failed, so
        // the declined targets are not retried either.
        if declined.is_empty() || !failed.is_empty() {
            failed.extend(declined);
            break;
        }
        // Retry backoff never outlasts the case budget: the last wait ends
        // with it, and a budget spent while waiting is the declined outcome
        // ([[RFC-0009:C-TIME-CONTROL-OWNERSHIP]]).
        if let Remaining::Finite(remaining) = bound.remaining() {
            std::thread::sleep(delay.min(remaining));
        }
        if !matches!(bound.remaining(), Remaining::Finite(remaining) if !remaining.is_zero()) {
            failed.extend(declined);
            exhausted = true;
            break;
        }
        pending = declined;
        delay = delay.saturating_mul(2).min(DECLINED_RESET_MAX_DELAY);
    }
    failed.sort_unstable();
    let error = (!failed.is_empty()).then(|| {
        let mut message = match input.reset {
            ResetCapability::Public(_) => attempts
                .last()
                .and_then(|attempt| attempt.error.clone())
                .unwrap_or_else(|| "prefix-cache reset failed".to_owned()),
            ResetCapability::PerTarget(_) => format!(
                "prefix-cache reset failed on {}",
                failed
                    .iter()
                    .map(|index| targets[*index].2.unwrap_or(targets[*index].0.as_str()))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        };
        if exhausted {
            message.push_str("; still declined when the case budget expired");
        }
        message
    });
    PrefixCacheResetOutcome {
        succeeded: failed.is_empty(),
        error,
        elapsed_ms: bound.elapsed_ms().saturating_sub(started_ms),
        attempts,
    }
}

/// The server's prefix-cache reset capability: its public action or the
/// per-target reset of every model-serving replica.
#[derive(Clone, Copy)]
pub(super) enum ResetCapability<'a> {
    Public(&'a WorkloadHttpAction),
    PerTarget(&'a [WorkloadReplicaReset]),
}

impl<'a> ResetCapability<'a> {
    /// The capability a Bench plan selected, absent for an uncontrolled start.
    pub(super) fn of(
        public: Option<&'a WorkloadHttpAction>,
        per_target: &'a [WorkloadReplicaReset],
    ) -> Option<Self> {
        if per_target.is_empty() {
            public.map(Self::Public)
        } else {
            Some(Self::PerTarget(per_target))
        }
    }
}

pub(super) struct CachePreparationInput<'a> {
    pub(super) endpoint: &'a WorkloadEndpoint,
    pub(super) reset: ResetCapability<'a>,
    pub(super) start: BenchCacheStart,
    pub(super) conditioning: Option<&'a BenchPrefixCacheConditioningPlan>,
    pub(super) population: Option<&'a BenchPopulation>,
    pub(super) warmup_drained: bool,
}

pub(super) fn prepare_prefix_cache(
    input: CachePreparationInput<'_>,
    bound: &OperationBound,
) -> BenchCachePreparationEvidence {
    let mut transitions = Vec::new();
    if input.warmup_drained {
        transitions.push(BenchCachePreparationTransition {
            phase: BenchCachePreparationPhase::WarmupDrained,
            elapsed_ms: bound.elapsed_ms(),
        });
    }
    let reset = reset_outcome(&input, bound);
    transitions.push(BenchCachePreparationTransition {
        phase: BenchCachePreparationPhase::CacheReset,
        elapsed_ms: bound.elapsed_ms(),
    });
    let conditioning = if reset.succeeded && input.start == BenchCacheStart::Primed {
        input
            .conditioning
            .zip(input.population)
            .map(|(conditioning, population)| {
                let evidence =
                    condition_prefix_cache(input.endpoint, conditioning, population, bound);
                transitions.push(BenchCachePreparationTransition {
                    phase: BenchCachePreparationPhase::CacheConditioned,
                    elapsed_ms: bound.elapsed_ms(),
                });
                evidence
            })
    } else {
        None
    };
    BenchCachePreparationEvidence {
        start: input.start,
        transitions,
        reset,
        conditioning,
    }
}

fn condition_prefix_cache(
    endpoint: &WorkloadEndpoint,
    plan: &BenchPrefixCacheConditioningPlan,
    population: &BenchPopulation,
    bound: &OperationBound,
) -> PrefixCacheConditioningEvidence {
    let started_ms = bound.elapsed_ms();
    let conditioning = population.prefix_conditioning.as_ref();
    let path = conditioning.map_or_else(std::path::PathBuf::new, |item| item.path.clone());
    let sha256 = conditioning.map_or_else(String::new, |item| item.sha256.clone());
    let prompt_tokens = conditioning.map_or(0, |item| item.prompt_tokens);
    let url = format!("http://{}:{}{}", endpoint.host, endpoint.port, plan.route);
    let data_parallel_size = plan.attention_data_parallel_size.max(1);
    let evidence = |ranks: Vec<PrefixCacheConditioningRankEvidence>,
                    succeeded: bool,
                    error: Option<String>,
                    bound: &OperationBound| {
        PrefixCacheConditioningEvidence {
            url: url.clone(),
            model: plan.model.clone(),
            prompt_path: path.clone(),
            prompt_sha256: sha256.clone(),
            prompt_tokens,
            prompt: plan.prompt.clone(),
            request_body: plan.request_body.clone(),
            maximum_shared_prefix_tokens: plan.maximum_shared_prefix_tokens,
            output_tokens: plan.output_tokens,
            consumes_population_entry: plan.consumes_population_entry,
            attention_data_parallel_size: data_parallel_size,
            ranks,
            succeeded,
            elapsed_ms: bound.elapsed_ms().saturating_sub(started_ms),
            error,
        }
    };
    let body = conditioning
        .ok_or_else(|| {
            CachePreparationError::Conditioning(
                "primed cache start has no canonical prefix artifact".to_owned(),
            )
        })
        .and_then(|item| {
            if let Some(maximum) = plan.maximum_shared_prefix_tokens
                && item.prompt_tokens != maximum
            {
                return Err(CachePreparationError::Conditioning(format!(
                    "canonical prefix contains {} tokens, expected {}",
                    item.prompt_tokens, maximum
                )));
            }
            std::fs::read_to_string(&item.path).map_err(|error| {
                CachePreparationError::Conditioning(format!(
                    "failed to read canonical prefix {:?}: {error}",
                    item.path
                ))
            })
        })
        .and_then(|prompt| {
            let mut body = serde_json::to_value(&plan.request_body)
                .map_err(|source| CachePreparationError::Serialization { source })?
                .as_object()
                .cloned()
                .ok_or_else(|| {
                    CachePreparationError::Conditioning(
                        "effective Bench request body did not serialize as an object".to_owned(),
                    )
                })?;
            body.insert(
                "model".to_owned(),
                serde_json::Value::String(plan.model.clone()),
            );
            body.insert("prompt".to_owned(), serde_json::Value::String(prompt));
            body.insert("stream".to_owned(), serde_json::Value::Bool(false));
            body.insert("n".to_owned(), serde_json::Value::from(1));
            body.insert(
                "max_tokens".to_owned(),
                serde_json::Value::from(plan.output_tokens),
            );
            Ok(serde_json::Value::Object(body))
        });
    let body = match body {
        Ok(body) => body,
        Err(error) => return evidence(Vec::new(), false, Some(error.to_string()), bound),
    };
    let client = finite_remaining(bound).and_then(|remaining| {
        reqwest::blocking::Client::builder()
            .timeout(remaining)
            .connect_timeout(remaining)
            .redirect(reqwest::redirect::Policy::none())
            .no_proxy()
            .build()
            .map_err(|source| CachePreparationError::Request { source })
    });
    let client = match client {
        Ok(client) => client,
        Err(error) => return evidence(Vec::new(), false, Some(error.to_string()), bound),
    };
    if plan.frontend_fanout {
        let outcome = finite_remaining(bound).and_then(|remaining| {
            let response = client
                .post(&url)
                .timeout(remaining)
                .json(&body)
                .send()
                .map_err(|source| CachePreparationError::Request { source })?;
            let status = response.status().as_u16();
            let body = response
                .bytes()
                .map_err(|source| CachePreparationError::Request { source })?;
            finite_remaining(bound)?;
            Ok((status, body))
        });
        let (status, body) = match outcome {
            Ok(response) => response,
            Err(error) => return evidence(Vec::new(), false, Some(error.to_string()), bound),
        };
        let fanout = serde_json::from_slice::<PrimePrefixCacheResponse>(&body).map_err(|source| {
            CachePreparationError::Conditioning(format!(
                "frontend conditioning fan-out returned HTTP {status} with an unrecognized response: {source}"
            ))
        });
        let fanout = match fanout {
            Ok(fanout) => fanout,
            Err(error) => return evidence(Vec::new(), false, Some(error.to_string()), bound),
        };
        let mut ranks = Vec::new();
        let mut first_error = None;
        for target in fanout.targets {
            // A target succeeded only when its recorded status is a success
            // status AND the peer reported no error: the cross-process
            // contract does not guarantee that a failing peer fills `error`,
            // so the status is authoritative.
            let failure = match target.http_status {
                Some(status) if is_successful_preparation_status(status) => target.error.clone(),
                Some(status) => Some(
                    target
                        .error
                        .clone()
                        .unwrap_or_else(|| format!("conditioning target returned HTTP {status}")),
                ),
                None => Some(
                    target
                        .error
                        .clone()
                        .unwrap_or_else(|| "conditioning target returned no response".to_owned()),
                ),
            };
            if first_error.is_none() {
                first_error = failure.as_ref().map(|error| {
                    format!(
                        "replica {} data-parallel rank {}: {error}",
                        target.url, target.rank
                    )
                });
            }
            ranks.push(PrefixCacheConditioningRankEvidence {
                target: Some(target.url),
                rank: target.rank,
                http_status: target.http_status,
                backend_prompt_tokens: None,
                backend_cache_read_tokens: None,
                elapsed_ms: target.elapsed_ms,
                error: failure,
            });
        }
        let coverage = reconcile_fanout_coverage(&ranks, data_parallel_size);
        let succeeded = status == 200 && first_error.is_none() && coverage.is_ok();
        let error =
            if succeeded {
                None
            } else {
                Some(first_error.or_else(|| coverage.err()).unwrap_or_else(|| {
                    format!("frontend conditioning fan-out returned HTTP {status}")
                }))
            };
        return evidence(ranks, succeeded, error, bound);
    }
    let mut ranks = Vec::new();
    for rank in 0..data_parallel_size {
        let rank_started_ms = bound.elapsed_ms();
        let outcome = finite_remaining(bound).and_then(|remaining| {
            let mut request = client.post(&url).timeout(remaining).json(&body);
            if data_parallel_size > 1 {
                request = request.header("X-Data-Parallel-Rank", rank.to_string());
            }
            let response = request
                .send()
                .map_err(|source| CachePreparationError::Request { source })?;
            let status = response.status().as_u16();
            let body = response
                .bytes()
                .map_err(|source| CachePreparationError::Request { source })?;
            finite_remaining(bound)?;
            let usage = serde_json::from_slice::<CompletionResponse>(&body)
                .ok()
                .and_then(|response| response.usage);
            let prompt_tokens = usage.as_ref().and_then(|usage| usage.prompt_tokens);
            let cache_read_tokens = usage
                .as_ref()
                .and_then(|usage| usage.prompt_tokens_details.as_ref())
                .and_then(|details| details.cached_tokens)
                .or_else(|| {
                    (prompt_tokens.is_some()
                        && endpoint.prompt_cache_read_zero_representation
                            == Some(PromptCacheReadZeroRepresentation::Omitted))
                    .then_some(0)
                });
            Ok(ConditioningResponse {
                status,
                prompt_tokens,
                cache_read_tokens,
            })
        });
        let rank_elapsed_ms = bound.elapsed_ms().saturating_sub(rank_started_ms);
        let rank_evidence = match outcome {
            Ok(response) if is_successful_preparation_status(response.status) => {
                PrefixCacheConditioningRankEvidence {
                    target: None,
                    rank,
                    http_status: Some(response.status),
                    backend_prompt_tokens: response.prompt_tokens,
                    backend_cache_read_tokens: response.cache_read_tokens,
                    elapsed_ms: rank_elapsed_ms,
                    error: None,
                }
            }
            Ok(response) => PrefixCacheConditioningRankEvidence {
                target: None,
                rank,
                http_status: Some(response.status),
                backend_prompt_tokens: response.prompt_tokens,
                backend_cache_read_tokens: response.cache_read_tokens,
                elapsed_ms: rank_elapsed_ms,
                error: Some(format!(
                    "prefix-cache conditioning returned HTTP {}",
                    response.status
                )),
            },
            Err(error) => PrefixCacheConditioningRankEvidence {
                target: None,
                rank,
                http_status: None,
                backend_prompt_tokens: None,
                backend_cache_read_tokens: None,
                elapsed_ms: rank_elapsed_ms,
                error: Some(error.to_string()),
            },
        };
        let rank_error = rank_evidence.error.clone();
        ranks.push(rank_evidence);
        if let Some(error) = rank_error {
            return evidence(
                ranks,
                false,
                Some(format!("data-parallel rank {rank}: {error}")),
                bound,
            );
        }
    }
    evidence(ranks, true, None, bound)
}

/// The fan-out response must cover every data-parallel rank of every prefill
/// replica it reports, with the replica count derived from the distinct
/// target URLs: `ranks == replicas × attention_data_parallel_size`. An empty
/// or partial set means ranks were never primed, even when every reported
/// target succeeded ([[RFC-0004:C-BENCH-CACHE-STATE]]).
fn reconcile_fanout_coverage(
    ranks: &[PrefixCacheConditioningRankEvidence],
    data_parallel_size: u32,
) -> Result<(), String> {
    if ranks.is_empty() {
        return Err("frontend conditioning fan-out returned no targets".to_owned());
    }
    let expected: Vec<u32> = (0..data_parallel_size).collect();
    let mut by_replica: BTreeMap<&str, Vec<u32>> = BTreeMap::new();
    for rank in ranks {
        let target = rank.target.as_deref().ok_or_else(|| {
            "frontend conditioning fan-out target is missing its replica URL".to_owned()
        })?;
        by_replica.entry(target).or_default().push(rank.rank);
    }
    for (replica, mut covered) in by_replica {
        covered.sort_unstable();
        if covered != expected {
            return Err(format!(
                "frontend conditioning fan-out covered ranks {covered:?} for replica {replica}, expected {expected:?}"
            ));
        }
    }
    Ok(())
}

/// Shared cache-preparation success predicate. 206 Partial Content is never
/// a success here: the built-in proxies use it to report partial fan-out
/// failure, and neither a cache reset nor an engine completions conditioning
/// response can legitimately carry it — a conditioning call that observes
/// 206 is talking to an aggregating frontend whose partial failure must not
/// be recorded as primed.
fn is_successful_preparation_status(status: u16) -> bool {
    (200..300).contains(&status) && status != 206
}

#[derive(Debug, thiserror::Error)]
enum CachePreparationError {
    #[error("measurement-case budget expired")]
    Deadline,
    #[error("prefix-cache reset requires a finite measurement-case budget")]
    UnboundedBudget,
    #[error("prefix-cache HTTP request failed: {source}")]
    Request {
        #[source]
        source: reqwest::Error,
    },
    #[error("failed to serialize prefix-cache conditioning request: {source}")]
    Serialization {
        #[source]
        source: serde_json::Error,
    },
    #[error("{0}")]
    Conditioning(String),
}

fn finite_remaining(bound: &OperationBound) -> Result<std::time::Duration, CachePreparationError> {
    match bound.remaining() {
        Remaining::Finite(duration) => Ok(duration),
        Remaining::Expired => Err(CachePreparationError::Deadline),
        Remaining::Unbounded => Err(CachePreparationError::UnboundedBudget),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resolve::topology::{CHAT_COMPLETIONS_PATH, COMPLETIONS_PATH};
    use crate::workload::domain::{WorkloadEndpointProtocol, WorkloadHttpMethod};
    use std::io::{BufRead, BufReader, Write};
    use std::net::{TcpListener, TcpStream};
    use std::thread;
    use std::time::Duration;

    fn read_request_headers(stream: &mut TcpStream) -> std::io::Result<()> {
        let mut reader = BufReader::new(stream);
        let mut line = String::new();
        loop {
            line.clear();
            match reader.read_line(&mut line)? {
                0 => {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::UnexpectedEof,
                        "request ended before its header terminator",
                    ));
                }
                _ if line == "\r\n" => return Ok(()),
                _ => {}
            }
        }
    }

    fn reset_target(address: std::net::SocketAddr) -> (String, WorkloadHttpAction) {
        (
            format!("http://{address}/reset_prefix_cache"),
            WorkloadHttpAction {
                method: WorkloadHttpMethod::Post,
                path: "/reset_prefix_cache".to_owned(),
                success: None,
            },
        )
    }

    fn endpoint_at(address: std::net::SocketAddr) -> WorkloadEndpoint {
        WorkloadEndpoint {
            protocol: WorkloadEndpointProtocol::Http,
            host: address.ip().to_string(),
            port: address.port(),
            completions_path: COMPLETIONS_PATH.to_owned(),
            chat_completions_path: CHAT_COMPLETIONS_PATH.to_owned(),
            server_metrics: None,
            prompt_cache_read_zero_representation: None,
        }
    }

    /// Serves one canned HTTP response per accepted connection, in order.
    fn serve_responses(
        responses: Vec<&'static str>,
    ) -> std::io::Result<(
        std::net::SocketAddr,
        thread::JoinHandle<std::io::Result<()>>,
    )> {
        let listener = TcpListener::bind("127.0.0.1:0")?;
        let address = listener.local_addr()?;
        let server = thread::spawn(move || -> std::io::Result<()> {
            for response in responses {
                let (mut stream, _) = listener.accept()?;
                read_request_headers(&mut stream)?;
                stream.write_all(response.as_bytes())?;
            }
            Ok(())
        });
        Ok((address, server))
    }

    const OK_BODY: &str = "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 15\r\nConnection: close\r\n\r\n{\"status\":\"ok\"}";
    const ERROR_BODY: &str = "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 18\r\nConnection: close\r\n\r\n{\"status\":\"error\"}";
    const SERVER_ERROR: &str =
        "HTTP/1.1 500 Internal Server Error\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";

    const DECLINED_BODY: &str = "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 17\r\nConnection: close\r\n\r\n{\"success\":false}";
    const ACCEPTED_BODY: &str = "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 16\r\nConnection: close\r\n\r\n{\"success\":true}";

    /// vLLM's reset action: a 200 whose body reports `success` `true`.
    fn vllm_reset() -> WorkloadHttpAction {
        WorkloadHttpAction {
            method: WorkloadHttpMethod::Post,
            path: "/reset_prefix_cache".to_owned(),
            success: Some(crate::workload::domain::WorkloadSuccessMatch {
                pointer: "/success".to_owned(),
                value: inferlab_protocol::JsonScalar::Boolean(true),
            }),
        }
    }

    fn cold<'a>(
        endpoint: &'a WorkloadEndpoint,
        reset: ResetCapability<'a>,
    ) -> CachePreparationInput<'a> {
        CachePreparationInput {
            endpoint,
            reset,
            start: BenchCacheStart::Cold,
            conditioning: None,
            population: None,
            warmup_drained: false,
        }
    }

    #[test]
    fn a_declined_reset_is_retried_only_on_the_declined_target_until_it_succeeds()
    -> Result<(), Box<dyn std::error::Error>> {
        let (declining, declining_server) = serve_responses(vec![DECLINED_BODY, ACCEPTED_BODY])?;
        let (accepting, accepting_server) = serve_responses(vec![ACCEPTED_BODY])?;
        let targets = [
            WorkloadReplicaReset {
                process: "prefill-0-rank-0".to_owned(),
                url: format!("http://{declining}/reset_prefix_cache"),
                action: vllm_reset(),
            },
            WorkloadReplicaReset {
                process: "decode-0-rank-0".to_owned(),
                url: format!("http://{accepting}/reset_prefix_cache"),
                action: vllm_reset(),
            },
        ];
        let endpoint = endpoint_at(declining);
        let outcome = reset_outcome(
            &cold(&endpoint, ResetCapability::PerTarget(&targets)),
            &OperationBound::finite(Duration::from_secs(10)),
        );

        assert!(outcome.succeeded, "{outcome:?}");
        let attempts = outcome
            .attempts
            .iter()
            .map(|attempt| {
                (
                    attempt.process.as_deref(),
                    attempt.succeeded,
                    attempt.declined,
                )
            })
            .collect::<Vec<_>>();
        assert_eq!(
            attempts,
            [
                (Some("prefill-0-rank-0"), false, true),
                (Some("decode-0-rank-0"), true, false),
                (Some("prefill-0-rank-0"), true, false),
            ]
        );
        assert_eq!(
            outcome.attempts[0].observed_value,
            Some(serde_json::Value::Bool(false))
        );
        declining_server
            .join()
            .map_err(|_| "fixture server panicked")??;
        accepting_server
            .join()
            .map_err(|_| "fixture server panicked")??;
        Ok(())
    }

    #[test]
    fn a_failure_other_than_a_decline_stops_retrying_every_target()
    -> Result<(), Box<dyn std::error::Error>> {
        let (failing, failing_server) = serve_responses(vec![SERVER_ERROR])?;
        let (declining, declining_server) = serve_responses(vec![DECLINED_BODY])?;
        let targets = [
            WorkloadReplicaReset {
                process: "prefill-0-rank-0".to_owned(),
                url: format!("http://{failing}/reset_prefix_cache"),
                action: vllm_reset(),
            },
            WorkloadReplicaReset {
                process: "decode-0-rank-0".to_owned(),
                url: format!("http://{declining}/reset_prefix_cache"),
                action: vllm_reset(),
            },
        ];
        let endpoint = endpoint_at(failing);
        let outcome = reset_outcome(
            &cold(&endpoint, ResetCapability::PerTarget(&targets)),
            &OperationBound::finite(Duration::from_secs(10)),
        );

        assert!(!outcome.succeeded, "{outcome:?}");
        assert_eq!(outcome.attempts.len(), 2, "{outcome:?}");
        assert_eq!(
            outcome.error.as_deref(),
            Some("prefix-cache reset failed on prefill-0-rank-0, decode-0-rank-0")
        );
        failing_server
            .join()
            .map_err(|_| "fixture server panicked")??;
        declining_server
            .join()
            .map_err(|_| "fixture server panicked")??;
        Ok(())
    }

    #[test]
    fn only_the_opposite_boolean_is_a_decline() -> Result<(), Box<dyn std::error::Error>> {
        // vLLM before it reported the outcome answered with an empty body;
        // Dynamo answers status error when the reset raised. Neither is the
        // framework declining, so neither may be retried.
        const EMPTY: &str = "HTTP/1.1 200 OK\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
        for (response, action) in [
            (EMPTY, vllm_reset()),
            (ERROR_BODY, predicated("/engine/flush_cache")),
        ] {
            let (address, server) = serve_responses(vec![response])?;
            let endpoint = endpoint_at(address);
            let outcome = reset_outcome(
                &cold(&endpoint, ResetCapability::Public(&action)),
                &OperationBound::finite(Duration::from_secs(10)),
            );
            assert!(!outcome.succeeded, "{outcome:?}");
            assert_eq!(outcome.attempts.len(), 1, "{outcome:?}");
            assert!(!outcome.attempts[0].declined, "{outcome:?}");
            server.join().map_err(|_| "fixture server panicked")??;
        }
        Ok(())
    }

    #[test]
    fn a_reset_still_declined_when_the_case_budget_expires_fails()
    -> Result<(), Box<dyn std::error::Error>> {
        let (address, server) = serve_responses(vec![DECLINED_BODY; 2])?;
        let action = vllm_reset();
        let endpoint = endpoint_at(address);
        let outcome = reset_outcome(
            &cold(&endpoint, ResetCapability::Public(&action)),
            &OperationBound::finite(Duration::from_millis(1_200)),
        );

        assert!(!outcome.succeeded, "{outcome:?}");
        assert_eq!(outcome.attempts.len(), 2, "{outcome:?}");
        assert!(outcome.attempts.iter().all(|attempt| attempt.declined));
        // The whole budget is spent before the reset is called declined.
        assert!(outcome.elapsed_ms >= 1_150, "{outcome:?}");
        assert!(
            outcome
                .error
                .as_deref()
                .is_some_and(|error| error.contains("still declined when the case budget expired")),
            "{outcome:?}"
        );
        server.join().map_err(|_| "fixture server panicked")??;
        Ok(())
    }

    fn predicated(path: &str) -> WorkloadHttpAction {
        WorkloadHttpAction {
            method: WorkloadHttpMethod::Post,
            path: path.to_owned(),
            success: Some(crate::workload::domain::WorkloadSuccessMatch {
                pointer: "/status".to_owned(),
                value: inferlab_protocol::JsonScalar::String("ok".to_owned()),
            }),
        }
    }

    #[test]
    fn a_success_predicate_turns_a_reported_error_inside_a_200_into_a_failed_reset()
    -> Result<(), Box<dyn std::error::Error>> {
        let (address, server) = serve_responses(vec![ERROR_BODY, OK_BODY])?;
        let action = predicated("/engine/flush_cache");
        let bound = OperationBound::finite(Duration::from_secs(5));
        let url = format!("http://{address}/engine/flush_cache");

        let reported_error = reset_prefix_cache(url.clone(), &action, None, &bound);
        let reported_ok = reset_prefix_cache(url, &action, None, &bound);

        assert!(!reported_error.succeeded, "{reported_error:?}");
        assert_eq!(reported_error.http_status, Some(200));
        assert_eq!(
            reported_error.observed_value,
            Some(serde_json::Value::from("error"))
        );
        assert!(reported_ok.succeeded, "{reported_ok:?}");
        server.join().map_err(|_| "fixture server panicked")??;
        Ok(())
    }

    #[test]
    fn a_per_target_reset_attempts_every_replica_after_a_failure_and_names_it()
    -> Result<(), Box<dyn std::error::Error>> {
        let (failing, failing_server) = serve_responses(vec![SERVER_ERROR])?;
        let (healthy, healthy_server) = serve_responses(vec![OK_BODY])?;
        let targets = [
            WorkloadReplicaReset {
                process: "prefill-0-rank-0".to_owned(),
                url: format!("http://{failing}/engine/flush_cache"),
                action: predicated("/engine/flush_cache"),
            },
            WorkloadReplicaReset {
                process: "decode-0-rank-0".to_owned(),
                url: format!("http://{healthy}/engine/flush_cache"),
                action: predicated("/engine/flush_cache"),
            },
        ];
        let (_, public) = reset_target(failing);
        let endpoint = endpoint_at(failing);
        let reset = ResetCapability::of(Some(&public), &targets).ok_or("reset capability")?;
        let outcome = reset_outcome(
            &CachePreparationInput {
                endpoint: &endpoint,
                reset,
                start: BenchCacheStart::Cold,
                conditioning: None,
                population: None,
                warmup_drained: false,
            },
            &OperationBound::finite(Duration::from_secs(5)),
        );

        let evidence = outcome;
        assert!(!evidence.succeeded);
        assert_eq!(evidence.attempts.len(), 2);
        assert!(!evidence.attempts[0].succeeded);
        assert!(evidence.attempts[1].succeeded);
        assert_eq!(
            evidence.error.as_deref(),
            Some("prefix-cache reset failed on prefill-0-rank-0")
        );
        failing_server
            .join()
            .map_err(|_| "fixture server panicked")??;
        healthy_server
            .join()
            .map_err(|_| "fixture server panicked")??;
        Ok(())
    }

    #[test]
    fn reset_can_complete_after_the_former_private_cap() -> Result<(), Box<dyn std::error::Error>> {
        let listener = TcpListener::bind("127.0.0.1:0")?;
        let address = listener.local_addr()?;
        let server = thread::spawn(move || -> std::io::Result<()> {
            let (mut stream, _) = listener.accept()?;
            read_request_headers(&mut stream)?;
            thread::sleep(Duration::from_millis(2_100));
            stream.write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
        });

        let (url, action) = reset_target(address);
        let evidence = reset_prefix_cache(
            url.clone(),
            &action,
            None,
            &OperationBound::finite(Duration::from_secs(3)),
        );

        assert!(evidence.succeeded, "{evidence:?}");
        assert_eq!(evidence.http_status, Some(200));
        server.join().map_err(|_| "fixture server panicked")??;
        Ok(())
    }

    #[test]
    fn reset_deadline_includes_the_complete_response_body() -> Result<(), Box<dyn std::error::Error>>
    {
        let listener = TcpListener::bind("127.0.0.1:0")?;
        let address = listener.local_addr()?;
        let server = thread::spawn(move || -> std::io::Result<()> {
            let (mut stream, _) = listener.accept()?;
            read_request_headers(&mut stream)?;
            stream
                .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 3\r\nConnection: close\r\n\r\nx")?;
            thread::sleep(Duration::from_millis(300));
            stream.write_all(b"y")?;
            thread::sleep(Duration::from_millis(300));
            Ok(())
        });

        let bound = OperationBound::finite(Duration::from_millis(500));
        let (url, action) = reset_target(address);
        let evidence = reset_prefix_cache(url.clone(), &action, None, &bound);
        server.join().map_err(|_| "fixture server panicked")??;

        assert!(
            !evidence.succeeded,
            "evidence={evidence:?}, remaining={:?}, elapsed_ms={}",
            bound.remaining(),
            bound.elapsed_ms()
        );
        Ok(())
    }

    #[test]
    fn reset_rejects_a_complete_response_after_the_owner_deadline()
    -> Result<(), Box<dyn std::error::Error>> {
        let listener = TcpListener::bind("127.0.0.1:0")?;
        let address = listener.local_addr()?;
        let server = thread::spawn(move || -> std::io::Result<()> {
            let (mut stream, _) = listener.accept()?;
            read_request_headers(&mut stream)?;
            stream
                .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 3\r\nConnection: close\r\n\r\nx")?;
            thread::sleep(Duration::from_millis(300));
            stream.write_all(b"y")?;
            thread::sleep(Duration::from_millis(300));
            match stream.write_all(b"z") {
                Ok(()) => Ok(()),
                Err(error)
                    if matches!(
                        error.kind(),
                        std::io::ErrorKind::BrokenPipe | std::io::ErrorKind::ConnectionReset
                    ) =>
                {
                    Ok(())
                }
                Err(error) => Err(error),
            }
        });

        let bound = OperationBound::finite(Duration::from_millis(500));
        let (url, action) = reset_target(address);
        let evidence = reset_prefix_cache(url.clone(), &action, None, &bound);
        server.join().map_err(|_| "fixture server panicked")??;

        assert!(
            !evidence.succeeded,
            "evidence={evidence:?}, remaining={:?}, elapsed_ms={}",
            bound.remaining(),
            bound.elapsed_ms()
        );
        Ok(())
    }

    #[test]
    fn reset_preserves_a_transport_failure_observed_before_the_deadline()
    -> Result<(), Box<dyn std::error::Error>> {
        let listener = TcpListener::bind("127.0.0.1:0")?;
        let address = listener.local_addr()?;
        let server = thread::spawn(move || -> std::io::Result<()> {
            let (mut stream, _) = listener.accept()?;
            read_request_headers(&mut stream)?;
            drop(stream);
            Ok(())
        });

        let (url, action) = reset_target(address);
        let evidence = reset_prefix_cache(
            url.clone(),
            &action,
            None,
            &OperationBound::finite(Duration::from_secs(1)),
        );
        server.join().map_err(|_| "fixture server panicked")??;

        assert!(!evidence.succeeded, "{evidence:?}");
        assert!(evidence.error.is_some());
        Ok(())
    }

    use crate::workload::domain::{BenchPopulation, ResolvedBenchPrompt};
    use crate::workspace::BenchPrompt;
    use inferlab_protocol::BenchPrefixConditioningInput;

    struct FanoutFixture {
        _dir: tempfile::TempDir,
        endpoint: WorkloadEndpoint,
        plan: crate::workload::BenchPrefixCacheConditioningPlan,
        population: BenchPopulation,
    }

    struct FanoutSetup {
        fixture: FanoutFixture,
        server: thread::JoinHandle<std::io::Result<()>>,
    }

    /// A control-plane conditioning fixture: a one-connection mock frontend
    /// answering the fan-out route with a canned status/body, plus the plan
    /// and population `condition_prefix_cache` needs to reach that call.
    fn fanout_fixture(
        data_parallel_size: u32,
        status: u16,
        response_body: &'static str,
    ) -> Result<FanoutSetup, Box<dyn std::error::Error>> {
        let listener = TcpListener::bind("127.0.0.1:0")?;
        let address = listener.local_addr()?;
        let server = thread::spawn(move || -> std::io::Result<()> {
            let (mut stream, _) = listener.accept()?;
            read_request_headers(&mut stream)?;
            let response = format!(
                "HTTP/1.1 {status} Fanout\r\nContent-Length: {}\r\nContent-Type: application/json\r\nConnection: close\r\n\r\n{response_body}",
                response_body.len()
            );
            stream.write_all(response.as_bytes())?;
            Ok(())
        });
        let dir = tempfile::tempdir()?;
        let prompt_path = dir.path().join("prefix.txt");
        std::fs::write(&prompt_path, "canonical prefix")?;
        let endpoint = endpoint_at(address);
        let fixture = FanoutFixture {
            _dir: dir,
            endpoint,
            plan: crate::workload::BenchPrefixCacheConditioningPlan {
                route: "/prime_prefix_cache".to_owned(),
                model: "m".to_owned(),
                prompt: ResolvedBenchPrompt::from_definition(&BenchPrompt::Flat),
                request_body: BTreeMap::new(),
                maximum_shared_prefix_tokens: Some(8),
                output_tokens: 1,
                consumes_population_entry: false,
                attention_data_parallel_size: data_parallel_size,
                frontend_fanout: true,
            },
            population: BenchPopulation {
                path: std::path::PathBuf::from("population.json"),
                evidence_path: std::path::PathBuf::from("population-evidence.json"),
                sha256: "unused".to_owned(),
                entries: 1,
                tpot_applicable: true,
                prefix_conditioning: Some(BenchPrefixConditioningInput {
                    path: prompt_path,
                    sha256: "unused".to_owned(),
                    prompt_tokens: 8,
                }),
                session_templates: Vec::new(),
            },
        };
        Ok(FanoutSetup { fixture, server })
    }

    fn condition_fanout(
        fixture: &FanoutFixture,
    ) -> crate::workload::record::PrefixCacheConditioningEvidence {
        condition_prefix_cache(
            &fixture.endpoint,
            &fixture.plan,
            &fixture.population,
            &OperationBound::finite(Duration::from_secs(5)),
        )
    }

    #[test]
    fn fanout_covers_every_rank_of_every_reported_replica() -> Result<(), Box<dyn std::error::Error>>
    {
        let setup = fanout_fixture(
            2,
            200,
            r#"{"targets": [
                {"url": "http://replica-a", "rank": 0, "http_status": 200, "elapsed_ms": 1, "error": null},
                {"url": "http://replica-a", "rank": 1, "http_status": 200, "elapsed_ms": 1, "error": null},
                {"url": "http://replica-b", "rank": 0, "http_status": 200, "elapsed_ms": 1, "error": null},
                {"url": "http://replica-b", "rank": 1, "http_status": 200, "elapsed_ms": 1, "error": null}
            ]}"#,
        )?;
        let evidence = condition_fanout(&setup.fixture);
        setup
            .server
            .join()
            .map_err(|_| "fixture server panicked")??;

        assert!(evidence.succeeded, "{evidence:?}");
        assert_eq!(evidence.ranks.len(), 4);
        Ok(())
    }

    /// A 200 over an empty target set primes nothing; it must not record a
    /// successful primed start.
    #[test]
    fn fanout_rejects_an_empty_target_set() -> Result<(), Box<dyn std::error::Error>> {
        let setup = fanout_fixture(2, 200, r#"{"targets": []}"#)?;
        let evidence = condition_fanout(&setup.fixture);
        setup
            .server
            .join()
            .map_err(|_| "fixture server panicked")??;

        assert!(!evidence.succeeded, "{evidence:?}");
        let error = evidence.error.ok_or("empty fan-out recorded no error")?;
        assert!(error.contains("no targets"), "{error}");
        Ok(())
    }

    /// Coverage is reconciled against the planned data-parallel size: a
    /// replica missing a rank fails the conditioning even when every
    /// reported target succeeded.
    #[test]
    fn fanout_rejects_partial_rank_coverage() -> Result<(), Box<dyn std::error::Error>> {
        let setup = fanout_fixture(
            2,
            200,
            r#"{"targets": [
                {"url": "http://replica-a", "rank": 0, "http_status": 200, "elapsed_ms": 1, "error": null}
            ]}"#,
        )?;
        let evidence = condition_fanout(&setup.fixture);
        setup
            .server
            .join()
            .map_err(|_| "fixture server panicked")??;

        assert!(!evidence.succeeded, "{evidence:?}");
        let error = evidence.error.ok_or("partial coverage recorded no error")?;
        assert!(error.contains("expected [0, 1]"), "{error}");
        Ok(())
    }

    /// The cross-process contract does not guarantee that a failing peer
    /// fills `error`: a target whose recorded status is not a success status
    /// fails the conditioning even with a null error field.
    #[test]
    fn fanout_target_with_non_success_status_and_no_error_field_fails()
    -> Result<(), Box<dyn std::error::Error>> {
        let setup = fanout_fixture(
            1,
            200,
            r#"{"targets": [
                {"url": "http://replica-a", "rank": 0, "http_status": 500, "elapsed_ms": 1, "error": null}
            ]}"#,
        )?;
        let evidence = condition_fanout(&setup.fixture);
        setup
            .server
            .join()
            .map_err(|_| "fixture server panicked")??;

        assert!(!evidence.succeeded, "{evidence:?}");
        let error = evidence.error.ok_or("failed target recorded no error")?;
        assert!(error.contains("HTTP 500"), "{error}");
        let rank_error = evidence.ranks[0]
            .error
            .as_deref()
            .ok_or("failed rank recorded no error")?;
        assert!(rank_error.contains("HTTP 500"), "{rank_error}");
        Ok(())
    }

    /// The proxy-side empty-target rejection (502 with an error body) is not
    /// a fan-out response: it fails the conditioning with the status named.
    #[test]
    fn fanout_rejection_status_is_not_primed() -> Result<(), Box<dyn std::error::Error>> {
        let setup = fanout_fixture(
            2,
            502,
            r#"{"error": "prefix cache conditioning fan-out has no targets"}"#,
        )?;
        let evidence = condition_fanout(&setup.fixture);
        setup
            .server
            .join()
            .map_err(|_| "fixture server panicked")??;

        assert!(!evidence.succeeded, "{evidence:?}");
        let error = evidence.error.ok_or("rejected fan-out recorded no error")?;
        assert!(error.contains("HTTP 502"), "{error}");
        Ok(())
    }
}
