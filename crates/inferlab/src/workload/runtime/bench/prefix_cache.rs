//! Controlled prefix-cache preparation and its domain evidence.

use crate::workload::BenchPrefixCacheConditioningPlan;
use crate::workload::domain::BenchPopulation;
use crate::workload::domain::{WorkloadEndpoint, WorkloadHttpAction};
use crate::workload::record::{
    BenchCachePreparationEvidence, BenchCachePreparationPhase, BenchCachePreparationTransition,
    PrefixCacheConditioningEvidence, PrefixCacheResetEvidence,
};
use crate::workspace::BenchCacheStart;
use inferlab_protocol::PromptCacheReadZeroRepresentation;
use inferlab_runtime::operation_bound::{OperationBound, Remaining};
use serde::Deserialize;

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
        Ok(status) if is_successful_cache_reset_status(status) => PrefixCacheResetEvidence {
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
    let outcome = conditioning
        .ok_or_else(|| {
            CachePreparationError::Conditioning(
                "primed cache start has no canonical prefix artifact".to_owned(),
            )
        })
        .and_then(|item| {
            if item.prompt_tokens != plan.maximum_shared_prefix_tokens {
                return Err(CachePreparationError::Conditioning(format!(
                    "canonical prefix contains {} tokens, expected {}",
                    item.prompt_tokens, plan.maximum_shared_prefix_tokens
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
            let remaining = finite_remaining(bound)?;
            let client = reqwest::blocking::Client::builder()
                .timeout(remaining)
                .connect_timeout(remaining)
                .redirect(reqwest::redirect::Policy::none())
                .no_proxy()
                .build()
                .map_err(|source| CachePreparationError::Request { source })?;
            let response = client
                .post(&url)
                .timeout(finite_remaining(bound)?)
                .json(&serde_json::Value::Object(body))
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
    let elapsed_ms = bound.elapsed_ms().saturating_sub(started_ms);
    match outcome {
        Ok(response) if (200..300).contains(&response.status) => PrefixCacheConditioningEvidence {
            url,
            model: plan.model.clone(),
            prompt_path: path,
            prompt_sha256: sha256,
            prompt_tokens,
            prompt: plan.prompt.clone(),
            request_body: plan.request_body.clone(),
            maximum_shared_prefix_tokens: plan.maximum_shared_prefix_tokens,
            output_tokens: plan.output_tokens,
            consumes_population_entry: plan.consumes_population_entry,
            backend_prompt_tokens: response.prompt_tokens,
            backend_cache_read_tokens: response.cache_read_tokens,
            succeeded: true,
            http_status: Some(response.status),
            elapsed_ms,
            error: None,
        },
        Ok(response) => PrefixCacheConditioningEvidence {
            url,
            model: plan.model.clone(),
            prompt_path: path,
            prompt_sha256: sha256,
            prompt_tokens,
            prompt: plan.prompt.clone(),
            request_body: plan.request_body.clone(),
            maximum_shared_prefix_tokens: plan.maximum_shared_prefix_tokens,
            output_tokens: plan.output_tokens,
            consumes_population_entry: plan.consumes_population_entry,
            backend_prompt_tokens: response.prompt_tokens,
            backend_cache_read_tokens: response.cache_read_tokens,
            succeeded: false,
            http_status: Some(response.status),
            elapsed_ms,
            error: Some(format!(
                "prefix-cache conditioning returned HTTP {}",
                response.status
            )),
        },
        Err(error) => PrefixCacheConditioningEvidence {
            url,
            model: plan.model.clone(),
            prompt_path: path,
            prompt_sha256: sha256,
            prompt_tokens,
            prompt: plan.prompt.clone(),
            request_body: plan.request_body.clone(),
            maximum_shared_prefix_tokens: plan.maximum_shared_prefix_tokens,
            output_tokens: plan.output_tokens,
            consumes_population_entry: plan.consumes_population_entry,
            backend_prompt_tokens: None,
            backend_cache_read_tokens: None,
            succeeded: false,
            http_status: None,
            elapsed_ms,
            error: Some(error.to_string()),
        },
    }
}

fn is_successful_cache_reset_status(status: u16) -> bool {
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
}
