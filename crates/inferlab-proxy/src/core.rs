//! Shared HTTP mechanics for the built-in disaggregated-serving proxies.
//!
//! Proxy-specific protocol bodies remain in their owning modules.

use crate::error::ProxyError as ProxyLifecycleError;
use async_stream::try_stream;
use axum::Json;
use axum::body::Body;
use axum::http::{HeaderMap, Response, StatusCode, header};
use axum::response::IntoResponse;
use bytes::Bytes;
use futures_util::{FutureExt, Stream, StreamExt};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::env;
use std::fmt;
use std::future::Future;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;
use tokio::task::JoinHandle;

/// Identity of a built-in proxy. Each proxy module owns its own [`ProxyMeta`]
/// so the proxy crate is the authority for the id/version recorded in
/// `BuiltinProxy` evidence.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProxyMeta {
    pub id: &'static str,
    pub version: u32,
}

/// Build a multi-threaded Tokio runtime and drive `run_async` to completion.
///
/// Built-in proxies share this runtime-builder wrapper; the per-proxy
/// `run` functions call it with their own async entrypoint.
pub fn run<F, Fut>(run_async: F) -> Result<(), ProxyLifecycleError>
where
    F: FnOnce() -> Fut,
    Fut: Future<Output = Result<(), ProxyLifecycleError>>,
{
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map_err(|error| ProxyLifecycleError::Lifecycle {
            message: format!("failed to create proxy tokio runtime: {error}"),
        })?;
    runtime.block_on(run_async())
}

/// Healthcheck response body shared by the proxies.
#[derive(Serialize)]
pub struct ProxyHealthcheckResponse {
    pub ready: bool,
    pub prefill_instances: usize,
    pub decode_instances: usize,
}

/// The shared `/healthcheck` payload: 200 once the proxy reports ready, 503
/// before, with the configured instance counts.
pub(crate) fn healthcheck_response(
    ready: bool,
    prefill_instances: usize,
    decode_instances: usize,
) -> (StatusCode, Json<ProxyHealthcheckResponse>) {
    let status = if ready {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    };
    (
        status,
        Json(ProxyHealthcheckResponse {
            ready,
            prefill_instances,
            decode_instances,
        }),
    )
}

/// Validate that both role endpoint lists are non-empty, naming the proxy in
/// the validation message.
pub(crate) fn require_endpoints(
    proxy_name: &'static str,
    prefill_is_empty: bool,
    decode_is_empty: bool,
) -> Result<(), ProxyLifecycleError> {
    if prefill_is_empty {
        return Err(ProxyLifecycleError::Invalid {
            message: format!("{proxy_name} requires at least one prefill endpoint"),
        });
    }
    if decode_is_empty {
        return Err(ProxyLifecycleError::Invalid {
            message: format!("{proxy_name} requires at least one decode endpoint"),
        });
    }
    Ok(())
}

/// Build the shared pooled client, naming the proxy in the failure message.
pub(crate) fn pooled_client(
    proxy_name: &'static str,
) -> Result<reqwest::Client, ProxyLifecycleError> {
    build_pooled_client().map_err(|error| ProxyLifecycleError::Io {
        message: format!("failed to create {proxy_name} HTTP client: {error}"),
    })
}

/// Bind `host:port` and serve `router` to completion, naming the proxy in
/// bind/serve failure messages.
pub(crate) async fn serve_router(
    proxy_name: &'static str,
    host: &str,
    port: u16,
    router: axum::Router,
) -> Result<(), ProxyLifecycleError> {
    let listener = tokio::net::TcpListener::bind((host, port))
        .await
        .map_err(|error| ProxyLifecycleError::Io {
            message: format!("failed to bind {proxy_name} on {host}:{port}: {error}"),
        })?;
    axum::serve(listener, router)
        .await
        .map_err(|error| ProxyLifecycleError::Io {
            message: format!("{proxy_name} server failed: {error}"),
        })
}

/// Poll every backend `url` at `path` once per second until each answers
/// with a success status. Runs as a background task; the caller marks the
/// proxy ready once it returns.
pub(crate) async fn await_backends(client: reqwest::Client, urls: Vec<String>, path: &'static str) {
    let waits = urls
        .into_iter()
        .map(|url| await_backend(client.clone(), url, path));
    futures_util::future::join_all(waits).await;
}

async fn await_backend(client: reqwest::Client, url: String, path: &'static str) {
    loop {
        if client
            .get(join_path(&url, path))
            .send()
            .await
            .is_ok_and(|response| response.status().is_success())
        {
            return;
        }
        tokio::time::sleep(Duration::from_secs(1)).await;
    }
}

/// Collect the fan-out target URLs shared by the reset/flush sweeps and the
/// readiness wait: every prefill replica URL followed by every decode URL.
pub(crate) fn fanout_target_urls<'a>(
    prefill_urls: impl IntoIterator<Item = &'a str>,
    decode_urls: impl IntoIterator<Item = &'a str>,
) -> Vec<String> {
    prefill_urls
        .into_iter()
        .chain(decode_urls)
        .map(str::to_owned)
        .collect()
}

/// Start a proxy response builder that mirrors the upstream response's
/// status and (when present) content-type.
pub(crate) fn upstream_response_builder(
    response: &reqwest::Response,
) -> Result<axum::http::response::Builder, ProxyHttpError> {
    let mut builder = Response::builder().status(status_code(response.status())?);
    if let Some(content_type) = response
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
    {
        builder = builder.header(header::CONTENT_TYPE, content_type);
    }
    Ok(builder)
}

/// Finish a proxy response builder with `body`.
pub(crate) fn response_body(
    builder: axum::http::response::Builder,
    body: Body,
) -> Result<Response<Body>, ProxyHttpError> {
    builder.body(body).map_err(|error| {
        ProxyHttpError::internal(format!("failed to build proxy response: {error}"))
    })
}

/// Forward an upstream response body verbatim, preserving status and
/// content-type.
pub async fn forward_response(
    response: reqwest::Response,
) -> Result<Response<Body>, ProxyHttpError> {
    let builder = upstream_response_builder(&response)?;
    let bytes = response
        .bytes()
        .await
        .map_err(|error| ProxyHttpError::upstream("upstream response body read failed", error))?;
    response_body(builder, Body::from(bytes))
}

/// Convert an unsuccessful upstream response into a `502 Bad Gateway`
/// [`ProxyHttpError`] that captures the upstream status and body.
pub async fn upstream_status_error(context: &str, response: reqwest::Response) -> ProxyHttpError {
    let status = response.status();
    let body = match response.text().await {
        Ok(text) => text,
        Err(error) => format!("<failed to read upstream error body: {error}>"),
    };
    ProxyHttpError::status(
        StatusCode::BAD_GATEWAY,
        format!("{context} returned HTTP {status}: {body}"),
    )
}

/// Resolve the outbound `Authorization` header from the inbound request or the
/// `OPENAI_API_KEY` environment variable.
pub fn outbound_authorization(headers: &HeaderMap) -> Option<String> {
    headers
        .get(header::AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
        .map(str::to_owned)
        .or_else(|| {
            env::var("OPENAI_API_KEY")
                .ok()
                .map(|key| format!("Bearer {key}"))
        })
}

/// Join a base URL with a path, normalizing a single trailing slash on the
/// base.
pub fn join_path(base: &str, path: &str) -> String {
    format!("{}{}", base.trim_end_matches('/'), path)
}

/// Convert a `reqwest` status code into an `axum`/`http` status code.
pub fn status_code(status: reqwest::StatusCode) -> Result<StatusCode, ProxyHttpError> {
    StatusCode::from_u16(status.as_u16())
        .map_err(|error| ProxyHttpError::internal(format!("invalid upstream status code: {error}")))
}

/// Advance a round-robin cursor and return the selected index into a non-empty
/// target list. Shared by all proxies' prefill/decode selection; each proxy
/// keeps its own cursor and target list (which stay local).
pub(crate) fn round_robin_index(cursor: &AtomicUsize, len: usize) -> usize {
    cursor.fetch_add(1, Ordering::SeqCst) % len
}

/// Build and send a JSON POST to `url` with an optional `X-Request-Id`, any
/// `extra_headers`, and an optional `Authorization`, returning the response or a
/// [`ProxyHttpError`] on transport or non-success status. `context` names the
/// call in error messages (e.g. "decode request"). Owns the transport for every
/// built-in proxy POST. Proxy-specific request ids and headers are optional.
pub(crate) async fn send_json_post(
    client: reqwest::Client,
    url: String,
    body: &Value,
    request_id: Option<&str>,
    authorization: Option<&str>,
    extra_headers: &[(&str, String)],
    context: &'static str,
) -> Result<reqwest::Response, ProxyHttpError> {
    let response = send_json_post_status(
        client,
        url,
        body,
        request_id,
        authorization,
        extra_headers,
        context,
    )
    .await?;
    if !response.status().is_success() {
        return Err(upstream_status_error(context, response).await);
    }
    Ok(response)
}

/// Like [`send_json_post`], but returns the response for any upstream status:
/// fan-out callers record per-target statuses instead of failing fast on the
/// first non-success upstream response.
pub(crate) async fn send_json_post_status(
    client: reqwest::Client,
    url: String,
    body: &Value,
    request_id: Option<&str>,
    authorization: Option<&str>,
    extra_headers: &[(&str, String)],
    context: &'static str,
) -> Result<reqwest::Response, ProxyHttpError> {
    let mut request = client.post(url).json(body);
    if let Some(request_id) = request_id {
        request = request.header("X-Request-Id", request_id);
    }
    // Extra headers precede `Authorization`: header insertion order reaches
    // the wire, and Mooncake's prefill always sent its rank header first.
    for (name, value) in extra_headers {
        request = request.header(*name, value);
    }
    if let Some(authorization) = authorization {
        request = request.header(reqwest::header::AUTHORIZATION, authorization);
    }
    request
        .send()
        .await
        .map_err(|error| ProxyHttpError::upstream(&format!("{context} failed"), error))
}

/// A per-process monotonic request id, `"{pid}-{n}"`, drawn from a proxy-owned
/// counter. Shared by the vLLM proxies so the id scheme has one home.
pub(crate) fn next_request_id(counter: &AtomicUsize) -> String {
    let value = counter.fetch_add(1, Ordering::SeqCst);
    format!("{}-{value}", std::process::id())
}

/// Build the outbound HTTP client shared by the proxies, with the pool tuning
/// (unbounded idle connections per host) both require. Returns the raw
/// `reqwest` error so each proxy keeps its own construction-failure message.
pub(crate) fn build_pooled_client() -> reqwest::Result<reqwest::Client> {
    reqwest::Client::builder()
        .pool_max_idle_per_host(usize::MAX)
        .build()
}

/// Per-target ceiling for cache reset/flush/prime fan-out operations. The
/// pooled client itself carries no timeout because it also serves
/// long-lived streaming decode requests; fan-out targets instead get this
/// per-target bound so one hung engine cannot stall every remaining rank
/// ([[RFC-0004:C-BENCH-CACHE-STATE]]).
pub(crate) const FANOUT_TARGET_TIMEOUT: Duration = Duration::from_secs(60);

/// Failure detail of one reset/flush fan-out target.
#[derive(Debug, Deserialize, Serialize)]
pub struct FanoutFailure {
    pub url: String,
    pub error: String,
}

/// Aggregated response of the cache reset/flush fan-out endpoints. SGLang's
/// `flush_cache` and the vLLM proxies' `reset_prefix_cache` share this wire
/// contract.
#[derive(Debug, Deserialize, Serialize)]
pub struct ResetPrefixCacheResponse {
    pub successful: Vec<String>,
    pub failed: Vec<FanoutFailure>,
}

/// Aggregated response of the prefix-cache conditioning fan-out endpoint.
/// The control plane deserializes this exact shape, so the fields are the
/// cross-process contract.
#[derive(Debug, Deserialize, Serialize)]
pub struct PrimePrefixCacheResponse {
    pub targets: Vec<PrimePrefixCacheTarget>,
}

/// One fanned-out conditioning flow: the prefill replica URL and the pinned
/// data-parallel rank, with the observed status or the failure detail.
#[derive(Debug, Deserialize, Serialize)]
pub struct PrimePrefixCacheTarget {
    pub url: String,
    pub rank: u32,
    pub http_status: Option<u16>,
    pub elapsed_ms: u64,
    pub error: Option<String>,
}

/// Failure of one fanned-out conditioning flow: the upstream status when a
/// response was observed, plus the failure detail.
pub(crate) struct PrimeFlowFailure {
    pub http_status: Option<u16>,
    pub error: String,
}

impl PrimeFlowFailure {
    pub(crate) fn transport(error: ProxyHttpError) -> Self {
        Self {
            http_status: None,
            error: error.to_string(),
        }
    }

    pub(crate) fn status(status: u16, detail: String) -> Self {
        Self {
            http_status: Some(status),
            error: detail,
        }
    }
}

/// Read a fanned-out conditioning response to text, requiring a 2xx status.
/// The body is captured either way so a non-2xx status reports it as the
/// failure detail; a body read failure is a transport failure. Returns the
/// upstream status and body on success.
pub(crate) async fn expect_2xx(
    context: &'static str,
    response: reqwest::Response,
) -> Result<(u16, String), PrimeFlowFailure> {
    let status = response.status().as_u16();
    let text = response.text().await.map_err(|error| {
        PrimeFlowFailure::transport(ProxyHttpError::upstream(
            &format!("{context} response read failed"),
            error,
        ))
    })?;
    if !(200..300).contains(&status) {
        return Err(PrimeFlowFailure::status(
            status,
            format!("{context} returned HTTP {status}: {text}"),
        ));
    }
    Ok((status, text))
}

/// Identity of one prime fan-out target: the prefill replica URL and the
/// data-parallel rank the flow pins.
pub(crate) trait PrimeFanoutTarget {
    fn url(&self) -> &str;
    fn rank(&self) -> u32;
}

/// A prefill replica with a static (config-issued) data-parallel size, as
/// opposed to Mooncake's discovered per-rank engines.
pub(crate) trait PrimeReplica {
    fn url(&self) -> &str;
    fn data_parallel_size(&self) -> u32;
}

/// One prime fan-out target over a static-size replica: the replica and the
/// pinned data-parallel rank.
pub(crate) struct RankedPrimeTarget<R> {
    pub replica: R,
    pub rank: u32,
}

impl<R: PrimeReplica> PrimeFanoutTarget for RankedPrimeTarget<R> {
    fn url(&self) -> &str {
        self.replica.url()
    }

    fn rank(&self) -> u32 {
        self.rank
    }
}

/// Enumerate the prime fan-out targets for static-size replicas, expanding
/// each replica over its data-parallel ranks (at least one).
pub(crate) fn ranked_prime_targets<R: PrimeReplica + Clone>(
    replicas: &[R],
) -> Vec<RankedPrimeTarget<R>> {
    let mut targets = Vec::new();
    for replica in replicas {
        for rank in 0..replica.data_parallel_size().max(1) {
            targets.push(RankedPrimeTarget {
                replica: replica.clone(),
                rank,
            });
        }
    }
    targets
}

/// Run the reset/flush fan-out skeleton: the engine module enumerates the
/// target base URLs and names its endpoint (`path`) and operation; target
/// execution, the per-target timeout, and the response aggregation (200 when
/// every target succeeded, 206 on partial failure) live here. An empty
/// target set is a 502 — "no targets" must not be conflated with success.
pub(crate) async fn run_sweep_fanout(
    client: reqwest::Client,
    operation: &'static str,
    path: &'static str,
    targets: Vec<String>,
    authorization: Option<String>,
) -> Response<Body> {
    if targets.is_empty() {
        return empty_fanout_failure(operation);
    }
    let attempts = targets
        .into_iter()
        .map(|url| sweep_target(client.clone(), operation, path, url, authorization.clone()));
    let mut successful = Vec::new();
    let mut failed = Vec::new();
    for result in futures_util::future::join_all(attempts).await {
        match result {
            Ok(url) => successful.push(url),
            Err(failure) => failed.push(failure),
        }
    }
    let status = if failed.is_empty() {
        StatusCode::OK
    } else {
        StatusCode::PARTIAL_CONTENT
    };
    (
        status,
        Json(ResetPrefixCacheResponse { successful, failed }),
    )
        .into_response()
}

async fn sweep_target(
    client: reqwest::Client,
    operation: &'static str,
    path: &'static str,
    url: String,
    authorization: Option<String>,
) -> Result<String, FanoutFailure> {
    let endpoint = join_path(&url, path);
    let mut request = client.post(endpoint).timeout(FANOUT_TARGET_TIMEOUT);
    if let Some(authorization) = authorization {
        request = request.header(reqwest::header::AUTHORIZATION, authorization);
    }
    let response = request.send().await.map_err(|error| FanoutFailure {
        url: url.clone(),
        error: format!("{operation} request failed: {error}"),
    })?;
    // A 206 from an upstream that is itself an aggregating frontend reports
    // partial failure, not success.
    if response.status().is_success() && response.status() != reqwest::StatusCode::PARTIAL_CONTENT {
        Ok(url)
    } else {
        let status = response.status();
        let detail = response
            .text()
            .await
            .unwrap_or_else(|error| format!("failed to read response body: {error}"));
        Err(FanoutFailure {
            url,
            error: format!("HTTP {status}: {detail}"),
        })
    }
}

/// Run the prefix-cache conditioning fan-out skeleton: the engine module
/// enumerates the (replica, rank) targets and supplies the per-target
/// conditioning flow; sequential target execution with a per-target timeout
/// and the response aggregation (200 when every flow succeeded, 206 on
/// partial failure) live here. An empty target set is a 502 — "no targets"
/// must not be conflated with success.
pub(crate) async fn run_prime_fanout<T, F, Fut>(
    operation: &'static str,
    targets: Vec<T>,
    execute: F,
) -> Response<Body>
where
    T: PrimeFanoutTarget,
    F: FnMut(T) -> Fut,
    Fut: Future<Output = Result<u16, PrimeFlowFailure>>,
{
    run_prime_fanout_with_timeout(operation, targets, execute, FANOUT_TARGET_TIMEOUT).await
}

async fn run_prime_fanout_with_timeout<T, F, Fut>(
    operation: &'static str,
    targets: Vec<T>,
    mut execute: F,
    target_timeout: Duration,
) -> Response<Body>
where
    T: PrimeFanoutTarget,
    F: FnMut(T) -> Fut,
    Fut: Future<Output = Result<u16, PrimeFlowFailure>>,
{
    if targets.is_empty() {
        return empty_fanout_failure(operation);
    }
    let mut results = Vec::new();
    for target in targets {
        let url = target.url().to_owned();
        let rank = target.rank();
        let started = std::time::Instant::now();
        let outcome = tokio::time::timeout(target_timeout, execute(target)).await;
        let elapsed_ms = u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX);
        results.push(match outcome {
            Ok(Ok(status)) => PrimePrefixCacheTarget {
                url,
                rank,
                http_status: Some(status),
                elapsed_ms,
                error: None,
            },
            Ok(Err(failure)) => PrimePrefixCacheTarget {
                url,
                rank,
                http_status: failure.http_status,
                elapsed_ms,
                error: Some(failure.error),
            },
            Err(_elapsed) => PrimePrefixCacheTarget {
                url,
                rank,
                http_status: None,
                elapsed_ms,
                error: Some(format!(
                    "{operation} timed out after {}s",
                    target_timeout.as_secs()
                )),
            },
        });
    }
    let status = if results.iter().all(|target| target.error.is_none()) {
        StatusCode::OK
    } else {
        StatusCode::PARTIAL_CONTENT
    };
    (status, Json(PrimePrefixCacheResponse { targets: results })).into_response()
}

/// "No targets" is a proxy-side failure (502 with an explicit error), never
/// a 200/206 aggregate: an empty fan-out primes or resets nothing.
fn empty_fanout_failure(operation: &str) -> Response<Body> {
    ProxyHttpError::status(
        StatusCode::BAD_GATEWAY,
        format!("{operation} fan-out has no targets: no prefill replica or data-parallel rank is available"),
    )
    .into_response()
}

/// Error type shared by both proxies, carrying an HTTP status and a message.
#[derive(Debug)]
pub struct ProxyHttpError {
    status: StatusCode,
    message: String,
}

impl ProxyHttpError {
    pub fn status(status: StatusCode, message: impl Into<String>) -> Self {
        Self {
            status,
            message: message.into(),
        }
    }

    pub fn upstream(context: &str, error: reqwest::Error) -> Self {
        Self::status(StatusCode::BAD_GATEWAY, format!("{context}: {error}"))
    }

    pub fn internal(message: impl Into<String>) -> Self {
        Self::status(StatusCode::INTERNAL_SERVER_ERROR, message)
    }
}

impl fmt::Display for ProxyHttpError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}", self.message)
    }
}

impl std::error::Error for ProxyHttpError {}

impl IntoResponse for ProxyHttpError {
    fn into_response(self) -> axum::response::Response {
        let body = Json(ProxyErrorResponse {
            error: self.message,
        });
        (self.status, body).into_response()
    }
}

#[derive(Serialize)]
pub struct ProxyErrorResponse {
    pub error: String,
}

/// What happens to the prefill task when the client drops the decode
/// response stream before prefill completes.
#[derive(Clone, Copy, Debug)]
pub(crate) enum OnClientDrop {
    /// Abort the prefill task (via [`AbortOnDrop`]). The default: once the
    /// client is gone, the orphaned prefill request is cancelled.
    Abort,
    /// Leave the prefill task running: it drains to completion in the
    /// background. Required when aborting prefill mid-flight would strand
    /// the paired decode-side engine request (for example the SGLang
    /// prefill/decode bootstrap room, where the decode engine waits for KV
    /// that a cancelled prefill would never deliver).
    Detach,
}

/// Stream a decode response body while a concurrently-running prefill task
/// completes. Used by proxies whose backend protocol starts both roles
/// together; the vLLM NIXL proxy instead forwards them sequentially.
/// `on_client_drop` selects the prefill task's fate when the client drops
/// the response before prefill finishes (see [`OnClientDrop`]); once prefill
/// completes any armed abort is disarmed, and a prefill failure surfaces as
/// a stream error.
pub(crate) fn stream_decode_response(
    response: reqwest::Response,
    prefill_task: JoinHandle<Result<(), ProxyHttpError>>,
    on_client_drop: OnClientDrop,
) -> Result<Response<Body>, ProxyHttpError> {
    let builder = upstream_response_builder(&response)?;
    let stream = decode_response_stream(response.bytes_stream(), prefill_task, on_client_drop);
    response_body(builder, Body::from_stream(stream))
}

/// Stream one successful upstream response without waiting for another role.
pub(crate) fn stream_response(
    response: reqwest::Response,
) -> Result<Response<Body>, ProxyHttpError> {
    let builder = upstream_response_builder(&response)?;
    let stream = response
        .bytes_stream()
        .map(|chunk| chunk.map_err(|error| stream_error(format!("decode stream failed: {error}"))));
    response_body(builder, Body::from_stream(stream))
}

/// The decode byte stream, generic over the decode stream and its error type so
/// it can be exercised without a live `reqwest::Response`. Yields decode bytes
/// in arrival order; concurrently drives `prefill_task` to completion and, per
/// `on_client_drop`, aborts or detaches it if the consumer drops the stream
/// before prefill finishes. On decode EOF before prefill completes, prefill is
/// awaited and its error (if any) surfaces.
pub(crate) fn decode_response_stream<S, E>(
    decode_stream: S,
    prefill_task: JoinHandle<Result<(), ProxyHttpError>>,
    on_client_drop: OnClientDrop,
) -> impl Stream<Item = std::result::Result<Bytes, std::io::Error>>
where
    S: Stream<Item = std::result::Result<Bytes, E>> + Unpin,
    E: fmt::Display,
{
    let prefill_abort = prefill_task.abort_handle();
    try_stream! {
        let mut decode_stream = decode_stream;
        let mut prefill_task = prefill_task;
        // Under `OnClientDrop::Detach` no abort guard is armed at all: dropping
        // the stream (client disconnect) leaves the prefill task draining to
        // completion in the background.
        let mut prefill_abort = match on_client_drop {
            OnClientDrop::Abort => Some(AbortOnDrop::new(prefill_abort)),
            OnClientDrop::Detach => None,
        };
        let mut prefill_done = false;
        loop {
            match next_stream_event(&mut prefill_task, &mut decode_stream, prefill_done).await {
                StreamEvent::Prefill(prefill) => {
                    prefill_done = true;
                    // One-time tie-break: if a decode item was already ready at the
                    // instant prefill completed, handle that single item before
                    // surfacing the prefill outcome. `now_or_never()` polls (and so
                    // consumes) the item, so it must be matched exhaustively: deliver
                    // a ready chunk, and PROPAGATE a ready decode error (an Ok-only
                    // match would drop it and truncate the response into a clean 200).
                    // EOF / not-ready fall through to the
                    // prefill outcome; decode is never indefinitely preferred.
                    match decode_stream.next().now_or_never() {
                        Some(Some(Ok(bytes))) => yield bytes,
                        Some(Some(Err(error))) => {
                            Err(stream_error(format!("decode stream failed: {error}")))?;
                        }
                        Some(None) | None => {}
                    }
                    prefill
                        .map_err(join_error)?
                        .map_err(|error| stream_error(error.to_string()))?;
                    if let Some(abort) = &mut prefill_abort {
                        abort.disarm();
                    }
                }
                StreamEvent::Decode(Some(Ok(bytes))) => yield bytes,
                StreamEvent::Decode(Some(Err(error))) => {
                    Err(stream_error(format!("decode stream failed: {error}")))?;
                }
                StreamEvent::Decode(None) => break,
            }
        }
        if !prefill_done {
            prefill_task
                .await
                .map_err(join_error)?
                .map_err(|error| stream_error(error.to_string()))?;
            if let Some(abort) = &mut prefill_abort {
                abort.disarm();
            }
        }
    }
}

enum StreamEvent<E> {
    Prefill(std::result::Result<Result<(), ProxyHttpError>, tokio::task::JoinError>),
    Decode(Option<std::result::Result<Bytes, E>>),
}

async fn next_stream_event<S, E>(
    prefill_task: &mut JoinHandle<Result<(), ProxyHttpError>>,
    decode_stream: &mut S,
    prefill_done: bool,
) -> StreamEvent<E>
where
    S: Stream<Item = std::result::Result<Bytes, E>> + Unpin,
{
    // Surface the prefill outcome promptly once the prefill task has COMPLETED, so
    // a continuously-ready decode stream cannot defer (and thereby suppress) a
    // prefill failure indefinitely. The caller delivers one already-ready decode
    // chunk before propagating a prefill error (a one-time tie-break), so a chunk
    // that was ready at the instant prefill finished is not dropped — but decode
    // is NOT permanently prioritized.
    if !prefill_done && prefill_task.is_finished() {
        return StreamEvent::Prefill(prefill_task.await);
    }
    // Prefill is still running: deliver decode bytes as they arrive, and otherwise
    // await the prefill task's completion (picked up by the `is_finished` check on
    // the next call). An unbiased race is fine here — there is no completed prefill
    // outcome to drop, and a ready decode chunk taken by its own branch is yielded,
    // not lost.
    tokio::select! {
        prefill = prefill_task, if !prefill_done => StreamEvent::Prefill(prefill),
        chunk = decode_stream.next() => StreamEvent::Decode(chunk),
    }
}

fn join_error(error: tokio::task::JoinError) -> std::io::Error {
    stream_error(format!("prefill task failed: {error}"))
}

fn stream_error(message: String) -> std::io::Error {
    std::io::Error::other(message)
}

/// Aborts the held task when dropped unless [`disarm`](AbortOnDrop::disarm)ed.
struct AbortOnDrop {
    handle: tokio::task::AbortHandle,
    armed: bool,
}

impl AbortOnDrop {
    fn new(handle: tokio::task::AbortHandle) -> Self {
        Self {
            handle,
            armed: true,
        }
    }

    fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for AbortOnDrop {
    fn drop(&mut self) {
        if self.armed {
            self.handle.abort();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::{Context, Result};

    #[test]
    fn join_path_normalizes_single_trailing_slash() {
        assert_eq!(
            join_path("http://h:1/", "/v1/models"),
            "http://h:1/v1/models"
        );
        assert_eq!(
            join_path("http://h:1", "/v1/models"),
            "http://h:1/v1/models"
        );
    }

    #[test]
    fn status_code_maps_reqwest_status() -> Result<()> {
        let mapped = status_code(reqwest::StatusCode::OK)
            .map_err(|error| anyhow::anyhow!(error.to_string()))?;
        assert_eq!(mapped, StatusCode::OK);
        Ok(())
    }

    #[test]
    fn outbound_authorization_prefers_inbound_header() -> Result<()> {
        let mut headers = HeaderMap::new();
        headers.insert(header::AUTHORIZATION, "Bearer inbound".parse()?);
        assert_eq!(
            outbound_authorization(&headers),
            Some("Bearer inbound".to_owned())
        );
        Ok(())
    }

    #[test]
    fn proxy_error_internal_uses_500() {
        let error = ProxyHttpError::internal("boom");
        assert_eq!(error.status, StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(error.to_string(), "boom");
    }

    struct StaticPrimeTarget {
        url: &'static str,
        rank: u32,
    }

    impl PrimeFanoutTarget for StaticPrimeTarget {
        fn url(&self) -> &str {
            self.url
        }

        fn rank(&self) -> u32 {
            self.rank
        }
    }

    /// An empty prime fan-out must be a 502 with an explicit error: a 200
    /// over zero targets would record a primed cache that primed nothing.
    #[test]
    fn prime_fanout_rejects_an_empty_target_set() -> Result<()> {
        let runtime = proxy_test_runtime()?;
        let response = runtime.block_on(run_prime_fanout(
            "prefix cache conditioning",
            Vec::<StaticPrimeTarget>::new(),
            |_target| async { Ok::<u16, PrimeFlowFailure>(200) },
        ));
        assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
        let body = runtime.block_on(axum::body::to_bytes(response.into_body(), usize::MAX))?;
        let value: Value = serde_json::from_slice(&body)?;
        assert!(
            value["error"]
                .as_str()
                .is_some_and(|error| error.contains("no targets")),
            "got {value}"
        );
        Ok(())
    }

    /// Same guard for the reset/flush sweep: zero targets is a proxy
    /// failure, never a clean sweep.
    #[test]
    fn sweep_fanout_rejects_an_empty_target_set() -> Result<()> {
        let runtime = proxy_test_runtime()?;
        let client = build_pooled_client().map_err(|error| anyhow::anyhow!(error.to_string()))?;
        let response = runtime.block_on(run_sweep_fanout(
            client,
            "prefix cache reset",
            "/reset_prefix_cache",
            Vec::new(),
            None,
        ));
        assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
        let body = runtime.block_on(axum::body::to_bytes(response.into_body(), usize::MAX))?;
        let value: Value = serde_json::from_slice(&body)?;
        assert!(
            value["error"]
                .as_str()
                .is_some_and(|error| error.contains("no targets")),
            "got {value}"
        );
        Ok(())
    }

    /// A hung target must not stall the remaining ranks: the flow is bounded
    /// by the per-target timeout and surfaces as a failed target (206).
    #[test]
    fn prime_fanout_times_out_a_hung_target() -> Result<()> {
        let runtime = proxy_test_runtime()?;
        let response = runtime.block_on(run_prime_fanout_with_timeout(
            "prefix cache conditioning",
            vec![StaticPrimeTarget {
                url: "http://127.0.0.1:1",
                rank: 0,
            }],
            |_target| async {
                futures_util::future::pending::<()>().await;
                Ok::<u16, PrimeFlowFailure>(200)
            },
            Duration::from_millis(50),
        ));
        assert_eq!(response.status(), StatusCode::PARTIAL_CONTENT);
        let body = runtime.block_on(axum::body::to_bytes(response.into_body(), usize::MAX))?;
        let value: Value = serde_json::from_slice(&body)?;
        assert_eq!(value["targets"][0]["http_status"], Value::Null);
        assert!(
            value["targets"][0]["error"]
                .as_str()
                .is_some_and(|error| error.contains("timed out")),
            "got {value}"
        );
        Ok(())
    }

    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};

    /// Flips a shared flag when dropped — used to observe that an aborted
    /// prefill task is actually cancelled (its future is dropped).
    struct SetOnDrop(Arc<AtomicBool>);

    impl Drop for SetOnDrop {
        fn drop(&mut self) {
            self.0.store(true, Ordering::SeqCst);
        }
    }

    fn proxy_test_runtime() -> Result<tokio::runtime::Runtime> {
        tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .map_err(|error| anyhow::anyhow!(error.to_string()))
    }

    #[test]
    fn streamed_decode_yields_bytes_in_order_when_prefill_succeeds() -> Result<()> {
        let runtime = proxy_test_runtime()?;
        let bytes = runtime.block_on(async {
            let decode = Box::pin(futures_util::stream::iter(vec![
                std::result::Result::<Bytes, std::io::Error>::Ok(Bytes::from_static(b"hello")),
                Ok(Bytes::from_static(b" world")),
            ]));
            let prefill = tokio::spawn(async { Ok::<(), ProxyHttpError>(()) });
            let mut stream = Box::pin(decode_response_stream(decode, prefill, OnClientDrop::Abort));
            let mut out = Vec::new();
            while let Some(item) = stream.next().await {
                out.push(item.map_err(|error| anyhow::anyhow!(error.to_string()))?);
            }
            anyhow::Ok(out)
        })?;
        let joined: Vec<u8> = bytes.into_iter().flatten().collect();
        assert_eq!(joined, b"hello world");
        Ok(())
    }

    #[test]
    fn streamed_decode_surfaces_prefill_error_after_decode_ends() -> Result<()> {
        let runtime = proxy_test_runtime()?;
        let (bytes, error) = runtime.block_on(async {
            let decode = Box::pin(futures_util::stream::iter(vec![std::result::Result::<
                Bytes,
                std::io::Error,
            >::Ok(
                Bytes::from_static(b"partial"),
            )]));
            let prefill = tokio::spawn(async {
                Err::<(), ProxyHttpError>(ProxyHttpError::internal("prefill boom"))
            });
            let mut stream = Box::pin(decode_response_stream(decode, prefill, OnClientDrop::Abort));
            let mut bytes = Vec::new();
            let mut error = None;
            while let Some(item) = stream.next().await {
                match item {
                    Ok(chunk) => bytes.extend_from_slice(&chunk),
                    Err(stream_error) => {
                        error = Some(stream_error.to_string());
                        break;
                    }
                }
            }
            anyhow::Ok((bytes, error))
        })?;
        assert_eq!(bytes, b"partial");
        let error = error.context("expected a prefill error to surface after decode ended")?;
        assert!(error.contains("prefill boom"), "got {error}");
        Ok(())
    }

    /// A prefill failure must surface even while the decode stream stays
    /// continuously ready. The one-time tie-break delivers an already-ready chunk
    /// but must NOT let an always-ready decode stream defer the prefill error
    /// indefinitely (a permanent decode bias would suppress it).
    #[test]
    fn prefill_error_surfaces_even_while_decode_stays_ready() -> Result<()> {
        let runtime = proxy_test_runtime()?;
        let error = runtime.block_on(async {
            // An unbounded, always-synchronously-ready decode stream.
            let decode = Box::pin(futures_util::stream::repeat_with(|| {
                std::result::Result::<Bytes, std::io::Error>::Ok(Bytes::from_static(b"x"))
            }));
            let prefill = tokio::spawn(async {
                Err::<(), ProxyHttpError>(ProxyHttpError::internal("prefill boom"))
            });
            let mut stream = Box::pin(decode_response_stream(decode, prefill, OnClientDrop::Abort));
            let mut chunks = 0usize;
            let mut error = None;
            while let Some(item) = stream.next().await {
                match item {
                    Ok(_) => {
                        chunks += 1;
                        // Bound: prove the error is not suppressed forever. The fix
                        // surfaces it within a handful of chunks; a regression that
                        // permanently prefers decode would never break out here.
                        assert!(
                            chunks < 100_000,
                            "prefill error was suppressed by a continuously-ready decode stream"
                        );
                    }
                    Err(stream_error) => {
                        error = Some(stream_error.to_string());
                        break;
                    }
                }
            }
            anyhow::Ok(error)
        })?;
        let error = error.context("a prefill error must surface even while decode stays ready")?;
        assert!(error.contains("prefill boom"), "got {error}");
        Ok(())
    }

    /// A decode error that is ALREADY ready at the instant prefill completes must be
    /// propagated by the one-time tie-break, not silently dropped. `now_or_never()`
    /// polls (and thus consumes) that ready item, so an Ok-only match would discard
    /// the error; with a successful prefill the stream would then end cleanly,
    /// turning a decode failure into a truncated 200.
    #[test]
    fn decode_error_ready_at_tiebreak_is_not_swallowed() -> Result<()> {
        let runtime = proxy_test_runtime()?;
        let error = runtime.block_on(async {
            // Force prefill to be FINISHED (Ok) so the first stream event is the
            // prefill outcome and the tie-break is what polls the decode stream.
            let prefill = tokio::spawn(async { Ok::<(), ProxyHttpError>(()) });
            while !prefill.is_finished() {
                tokio::task::yield_now().await;
            }
            // A synchronously-ready decode Err waiting at the tie-break instant.
            let decode = Box::pin(futures_util::stream::iter(vec![std::result::Result::<
                Bytes,
                std::io::Error,
            >::Err(
                std::io::Error::other("decode boom"),
            )]));
            let mut stream = Box::pin(decode_response_stream(decode, prefill, OnClientDrop::Abort));
            let mut error = None;
            while let Some(item) = stream.next().await {
                if let Err(stream_error) = item {
                    error = Some(stream_error.to_string());
                    break;
                }
            }
            anyhow::Ok(error)
        })?;
        let error =
            error.context("a decode error ready at the tie-break must surface, not truncate")?;
        assert!(error.contains("decode boom"), "got {error}");
        Ok(())
    }

    #[test]
    fn dropping_the_stream_before_prefill_finishes_aborts_prefill() -> Result<()> {
        let runtime = proxy_test_runtime()?;
        let aborted = Arc::new(AtomicBool::new(false));
        let flag = aborted.clone();
        let cancelled = runtime.block_on(async move {
            let (started_tx, started_rx) = tokio::sync::oneshot::channel::<()>();
            // Prefill never completes; once aborted, its future is dropped and
            // SetOnDrop flips the flag. It signals `started` only AFTER the guard
            // is constructed, so the drop-abort is observed deterministically (no
            // race on whether the task was polled before the abort fired).
            let prefill = tokio::spawn(async move {
                let _guard = SetOnDrop(flag);
                let _ = started_tx.send(());
                futures_util::future::pending::<()>().await;
                Ok::<(), ProxyHttpError>(())
            });
            let _ = started_rx.await;
            // Decode yields one chunk then stays pending, so the loop neither
            // breaks (decode EOF) nor selects prefill — leaving prefill in flight.
            let decode = Box::pin(
                futures_util::stream::once(async {
                    std::result::Result::<Bytes, std::io::Error>::Ok(Bytes::from_static(b"a"))
                })
                .chain(futures_util::stream::pending::<
                    std::result::Result<Bytes, std::io::Error>,
                >()),
            );
            let mut stream = Box::pin(decode_response_stream(decode, prefill, OnClientDrop::Abort));
            assert!(matches!(stream.next().await, Some(Ok(_))));
            drop(stream);
            for _ in 0..200 {
                if aborted.load(Ordering::SeqCst) {
                    return true;
                }
                tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            }
            false
        });
        assert!(
            cancelled,
            "prefill task was not aborted when the response stream was dropped"
        );
        Ok(())
    }

    /// The counterpart pin for `OnClientDrop::Detach`: dropping the consumer
    /// must NOT abort the prefill task — it keeps running in the background
    /// and completes on its own.
    #[test]
    fn dropping_the_stream_before_prefill_finishes_detaches_prefill() -> Result<()> {
        let runtime = proxy_test_runtime()?;
        let completed = Arc::new(AtomicBool::new(false));
        let flag = completed.clone();
        let finished = runtime.block_on(async move {
            let (started_tx, started_rx) = tokio::sync::oneshot::channel::<()>();
            // Prefill completes after a short delay and flips the flag; if the
            // stream drop aborted it, the flag would never be set.
            let prefill = tokio::spawn(async move {
                let _ = started_tx.send(());
                tokio::time::sleep(std::time::Duration::from_millis(50)).await;
                flag.store(true, Ordering::SeqCst);
                Ok::<(), ProxyHttpError>(())
            });
            let _ = started_rx.await;
            // Decode yields one chunk then stays pending, so prefill is still in
            // flight when the stream is dropped.
            let decode = Box::pin(
                futures_util::stream::once(async {
                    std::result::Result::<Bytes, std::io::Error>::Ok(Bytes::from_static(b"a"))
                })
                .chain(futures_util::stream::pending::<
                    std::result::Result<Bytes, std::io::Error>,
                >()),
            );
            let mut stream = Box::pin(decode_response_stream(
                decode,
                prefill,
                OnClientDrop::Detach,
            ));
            assert!(matches!(stream.next().await, Some(Ok(_))));
            drop(stream);
            for _ in 0..200 {
                if completed.load(Ordering::SeqCst) {
                    return true;
                }
                tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            }
            false
        });
        assert!(
            finished,
            "prefill task was aborted instead of detached when the response stream was dropped"
        );
        Ok(())
    }
}
