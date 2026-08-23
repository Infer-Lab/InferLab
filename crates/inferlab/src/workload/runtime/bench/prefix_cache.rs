//! Controlled prefix-cache preparation and its domain evidence.

use crate::workload::BenchPrefixCacheConditioningPlan;
use crate::workload::domain::BenchPopulation;
use crate::workload::domain::{WorkloadEndpoint, WorkloadHttpAction};
use crate::workload::record::{
    BenchCachePreparationEvidence, BenchCachePreparationPhase, BenchCachePreparationTransition,
    PrefixCacheConditioningEvidence, PrefixCacheConditioningRankEvidence, PrefixCacheResetEvidence,
};
use crate::workspace::BenchCacheStart;
use inferlab_protocol::PromptCacheReadZeroRepresentation;
use inferlab_proxy::core::PrimePrefixCacheResponse;
use inferlab_runtime::operation_bound::{OperationBound, Remaining};
use serde::Deserialize;
use std::collections::BTreeMap;

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
    endpoint: &WorkloadEndpoint,
    action: &WorkloadHttpAction,
    bound: &OperationBound,
) -> PrefixCacheResetEvidence {
    let started_ms = bound.elapsed_ms();
    let url = format!("http://{}:{}{}", endpoint.host, endpoint.port, action.path);
    let result: Result<u16, CachePreparationError> = (|| {
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
        response
            .copy_to(&mut std::io::sink())
            .map_err(|source| CachePreparationError::Request { source })?;
        finite_remaining(bound)?;
        Ok(status)
    })();
    match result {
        Ok(status) if is_successful_preparation_status(status) => PrefixCacheResetEvidence {
            method: action.method,
            url,
            succeeded: true,
            http_status: Some(status),
            error: None,
            elapsed_ms: bound.elapsed_ms().saturating_sub(started_ms),
        },
        Ok(status) => PrefixCacheResetEvidence {
            method: action.method,
            url,
            succeeded: false,
            http_status: Some(status),
            error: Some(format!("prefix-cache reset returned HTTP {status}")),
            elapsed_ms: bound.elapsed_ms().saturating_sub(started_ms),
        },
        Err(error) => PrefixCacheResetEvidence {
            method: action.method,
            url,
            succeeded: false,
            http_status: None,
            error: Some(error.to_string()),
            elapsed_ms: bound.elapsed_ms().saturating_sub(started_ms),
        },
    }
}

pub(super) struct CachePreparationInput<'a> {
    pub(super) endpoint: &'a WorkloadEndpoint,
    pub(super) action: &'a WorkloadHttpAction,
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
    let reset = reset_prefix_cache(input.endpoint, input.action, bound);
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

    fn reset_target(address: std::net::SocketAddr) -> (WorkloadEndpoint, WorkloadHttpAction) {
        (
            WorkloadEndpoint {
                protocol: WorkloadEndpointProtocol::Http,
                host: address.ip().to_string(),
                port: address.port(),
                completions_path: "/v1/completions".to_owned(),
                chat_completions_path: "/v1/chat/completions".to_owned(),
                server_metrics: None,
                prompt_cache_read_zero_representation: None,
            },
            WorkloadHttpAction {
                method: WorkloadHttpMethod::Post,
                path: "/reset_prefix_cache".to_owned(),
            },
        )
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

        let (endpoint, action) = reset_target(address);
        let evidence = reset_prefix_cache(
            &endpoint,
            &action,
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
        let (endpoint, action) = reset_target(address);
        let evidence = reset_prefix_cache(&endpoint, &action, &bound);
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
        let (endpoint, action) = reset_target(address);
        let evidence = reset_prefix_cache(&endpoint, &action, &bound);
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

        let (endpoint, action) = reset_target(address);
        let evidence = reset_prefix_cache(
            &endpoint,
            &action,
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
        let (endpoint, _action) = reset_target(address);
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
