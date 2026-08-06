use crate::core::{
    self, ProxyHealthcheckResponse, ProxyHttpError, ProxyMeta, forward_response, join_path,
    outbound_authorization,
};
use crate::error::ProxyError;
use axum::body::Body;
use axum::extract::{Json, State};
use axum::http::{HeaderMap, Response, StatusCode};
use axum::routing::{get, post};
use axum::{Router, serve};
use serde::Serialize;
use serde_json::{Map, Value};
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;
use tokio::net::TcpListener;

/// Identity recorded in `BuiltinProxy` evidence for the NIXL proxy.
pub const ID: &str = "inferlab-vllm-nixl-proxy";
/// Evidence version for the NIXL proxy identity.
pub const VERSION: u32 = 1;

/// Owned identity of the built-in NIXL proxy.
pub fn meta() -> ProxyMeta {
    ProxyMeta {
        id: ID,
        version: VERSION,
    }
}

#[derive(Clone, Debug)]
pub struct Config {
    pub host: String,
    pub port: u16,
    pub prefill: Vec<String>,
    pub decode: Vec<String>,
}

pub fn run(config: Config) -> Result<(), ProxyError> {
    core::run(|| run_async(config))
}

pub async fn run_async(config: Config) -> Result<(), ProxyError> {
    let host = config.host.clone();
    let port = config.port;
    let state = ProxyState::new(config)?;
    let app = router(state);
    let listener = TcpListener::bind((host.as_str(), port))
        .await
        .map_err(|error| ProxyError::Io {
            message: format!("failed to bind vLLM NIXL proxy on {host}:{port}: {error}"),
        })?;
    serve(listener, app).await.map_err(|error| ProxyError::Io {
        message: format!("vLLM NIXL proxy server failed: {error}"),
    })
}

fn router(state: ProxyState) -> Router {
    Router::new()
        .route("/healthcheck", get(healthcheck))
        .route("/v1/models", get(models))
        .route("/v1/completions", post(completions))
        .route("/v1/chat/completions", post(chat_completions))
        .with_state(state)
}

#[derive(Clone)]
struct ProxyState {
    inner: Arc<ProxyStateInner>,
}

struct ProxyStateInner {
    client: reqwest::Client,
    prefill: Vec<String>,
    decode: Vec<String>,
    prefill_cursor: AtomicUsize,
    decode_cursor: AtomicUsize,
    request_counter: AtomicUsize,
}

impl ProxyState {
    fn new(config: Config) -> Result<Self, ProxyError> {
        if config.prefill.is_empty() {
            return Err(ProxyError::Invalid {
                message: "vLLM NIXL proxy requires at least one prefill endpoint".to_owned(),
            });
        }
        if config.decode.is_empty() {
            return Err(ProxyError::Invalid {
                message: "vLLM NIXL proxy requires at least one decode endpoint".to_owned(),
            });
        }
        let client = core::build_pooled_client().map_err(|error| ProxyError::Io {
            message: format!("failed to create vLLM NIXL proxy HTTP client: {error}"),
        })?;
        Ok(Self {
            inner: Arc::new(ProxyStateInner {
                client,
                prefill: config.prefill,
                decode: config.decode,
                prefill_cursor: AtomicUsize::new(0),
                decode_cursor: AtomicUsize::new(0),
                request_counter: AtomicUsize::new(0),
            }),
        })
    }

    fn client(&self) -> reqwest::Client {
        self.inner.client.clone()
    }

    fn next_prefill_url(&self) -> Result<String, ProxyHttpError> {
        let index = core::round_robin_index(&self.inner.prefill_cursor, self.inner.prefill.len());
        Ok(self.inner.prefill[index].clone())
    }

    fn next_decode_url(&self) -> Result<String, ProxyHttpError> {
        let index = core::round_robin_index(&self.inner.decode_cursor, self.inner.decode.len());
        Ok(self.inner.decode[index].clone())
    }

    fn request_id(&self) -> String {
        core::next_request_id(&self.inner.request_counter)
    }
}

async fn healthcheck(State(state): State<ProxyState>) -> Json<ProxyHealthcheckResponse> {
    Json(ProxyHealthcheckResponse {
        ready: true,
        prefill_instances: state.inner.prefill.len(),
        decode_instances: state.inner.decode.len(),
    })
}

async fn models(State(state): State<ProxyState>) -> Result<Response<Body>, ProxyHttpError> {
    let decode_url = state.next_decode_url()?;
    let response = state
        .client()
        .get(join_path(&decode_url, "/v1/models"))
        .send()
        .await
        .map_err(|error| ProxyHttpError::upstream("decode /v1/models request failed", error))?;
    forward_response(response).await
}

async fn completions(
    State(state): State<ProxyState>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> Result<Response<Body>, ProxyHttpError> {
    completion_route(state, headers, body, "/v1/completions").await
}

async fn chat_completions(
    State(state): State<ProxyState>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> Result<Response<Body>, ProxyHttpError> {
    completion_route(state, headers, body, "/v1/chat/completions").await
}

async fn completion_route(
    state: ProxyState,
    headers: HeaderMap,
    body: Value,
    path: &'static str,
) -> Result<Response<Body>, ProxyHttpError> {
    let stream = body.get("stream").and_then(Value::as_bool).unwrap_or(false);
    let prefill_url = state.next_prefill_url()?;
    let decode_url = state.next_decode_url()?;
    let request_id = state.request_id();
    let authorization = outbound_authorization(&headers);
    let client = state.client();
    let prefill_body = prefill_body(&body, &request_id)?;
    let prefill_response = send_prefill_request(
        client.clone(),
        &prefill_url,
        path,
        prefill_body,
        &request_id,
        authorization.as_deref(),
    )
    .await?;
    let decode_body = decode_body(&body, prefill_response.kv_transfer_params)?;
    let decode_response = core::send_json_post(
        client,
        join_path(&decode_url, path),
        &decode_body,
        Some(&request_id),
        authorization.as_deref(),
        &[],
        "decode request",
    )
    .await?;
    if stream {
        core::stream_response(decode_response)
    } else {
        forward_response(decode_response).await
    }
}

#[derive(Debug)]
struct PrefillResponse {
    kv_transfer_params: Value,
}

fn prefill_body(body: &Value, request_id: &str) -> Result<Value, ProxyHttpError> {
    let mut body = body.clone();
    let object = object_mut(&mut body)?;
    object.insert(
        "kv_transfer_params".to_owned(),
        NixlPrefillKvTransferParams::new(request_id).into_protocol_value()?,
    );
    object.insert("stream".to_owned(), Value::Bool(false));
    object.insert("max_tokens".to_owned(), Value::from(1_u8));
    if object.contains_key("max_completion_tokens") {
        object.insert("max_completion_tokens".to_owned(), Value::from(1_u8));
    }
    object.remove("stream_options");
    object.remove("min_tokens");
    object.remove("min_completion_tokens");
    Ok(body)
}

#[derive(Serialize)]
struct NixlPrefillKvTransferParams {
    do_remote_decode: bool,
    do_remote_prefill: bool,
    remote_engine_id: Option<String>,
    remote_block_ids: Option<Vec<u64>>,
    remote_host: Option<String>,
    remote_port: Option<u16>,
    transfer_id: String,
}

impl NixlPrefillKvTransferParams {
    fn new(request_id: &str) -> Self {
        Self {
            do_remote_decode: true,
            do_remote_prefill: false,
            remote_engine_id: None,
            remote_block_ids: None,
            remote_host: None,
            remote_port: None,
            transfer_id: format!("xfer-{request_id}"),
        }
    }

    fn into_protocol_value(self) -> Result<Value, ProxyHttpError> {
        serde_json::to_value(self).map_err(|error| {
            ProxyHttpError::internal(format!(
                "failed to serialize vLLM NIXL prefill transfer params: {error}"
            ))
        })
    }
}

fn decode_body(body: &Value, kv_transfer_params: Value) -> Result<Value, ProxyHttpError> {
    let mut body = body.clone();
    let object = object_mut(&mut body)?;
    object.insert("kv_transfer_params".to_owned(), kv_transfer_params);
    Ok(body)
}

fn object_mut(body: &mut Value) -> Result<&mut Map<String, Value>, ProxyHttpError> {
    body.as_object_mut().ok_or_else(|| {
        ProxyHttpError::status(
            StatusCode::BAD_REQUEST,
            "OpenAI completion request body must be a JSON object",
        )
    })
}

async fn send_prefill_request(
    client: reqwest::Client,
    prefill_url: &str,
    path: &'static str,
    body: Value,
    request_id: &str,
    authorization: Option<&str>,
) -> Result<PrefillResponse, ProxyHttpError> {
    let response = core::send_json_post(
        client,
        join_path(prefill_url, path),
        &body,
        Some(request_id),
        authorization,
        &[],
        "prefill request",
    )
    .await?;
    let body = response
        .json::<Value>()
        .await
        .map_err(|error| ProxyHttpError::upstream("prefill response JSON read failed", error))?;
    let kv_transfer_params = body.get("kv_transfer_params").cloned().ok_or_else(|| {
        ProxyHttpError::status(
            StatusCode::BAD_GATEWAY,
            "prefill response did not include kv_transfer_params",
        )
    })?;
    Ok(PrefillResponse { kv_transfer_params })
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::{Context, Result, bail};
    use async_stream::stream;
    use axum::body::to_bytes;
    use axum::http::{HeaderValue, header};
    use axum::response::IntoResponse;
    use bytes::Bytes;
    use futures_util::StreamExt;
    use serde_json::json;
    use std::time::Duration;
    use tokio::sync::{Mutex, Notify};
    use tokio::task::JoinHandle;

    #[test]
    fn meta_exports_byte_stable_proxy_identity() {
        // AC4: the NIXL proxy owns and exports its own id+version. These exact
        // strings/numbers are persisted in BuiltinProxy evidence, so they must stay
        // byte-stable.
        assert_eq!(ID, "inferlab-vllm-nixl-proxy");
        assert_eq!(VERSION, 1);
        assert_eq!(meta().id, ID);
        assert_eq!(meta().version, VERSION);
    }

    #[tokio::test]
    async fn healthcheck_response_reports_configured_instances() -> Result<()> {
        let state = ProxyState::new(Config {
            host: "127.0.0.1".to_owned(),
            port: 8000,
            prefill: vec![
                "http://127.0.0.1:8010".to_owned(),
                "http://127.0.0.1:8011".to_owned(),
            ],
            decode: vec!["http://127.0.0.1:8020".to_owned()],
        })?;

        let Json(response) = healthcheck(State(state)).await;
        let value = serde_json::to_value(response)?;

        assert_eq!(value.get("ready").and_then(Value::as_bool), Some(true));
        assert_eq!(
            value.get("prefill_instances").and_then(Value::as_u64),
            Some(2)
        );
        assert_eq!(
            value.get("decode_instances").and_then(Value::as_u64),
            Some(1)
        );
        Ok(())
    }

    #[test]
    fn prefill_body_sets_nixl_prefill_transfer_params() -> Result<()> {
        let body = json!({
            "model": "m",
            "prompt": "hello",
            "stream": true,
            "stream_options": {"include_usage": true},
            "max_tokens": 64,
            "max_completion_tokens": 64,
            "min_tokens": 4,
        });

        let lowered =
            prefill_body(&body, "request-1").map_err(|error| anyhow::anyhow!(error.to_string()))?;

        assert_eq!(
            lowered.pointer("/kv_transfer_params/do_remote_decode"),
            Some(&Value::Bool(true))
        );
        assert_eq!(
            lowered.pointer("/kv_transfer_params/do_remote_prefill"),
            Some(&Value::Bool(false))
        );
        assert_eq!(
            lowered
                .pointer("/kv_transfer_params/transfer_id")
                .and_then(Value::as_str),
            Some("xfer-request-1")
        );
        assert_eq!(lowered.get("stream"), Some(&Value::Bool(false)));
        assert_eq!(lowered.get("max_tokens").and_then(Value::as_u64), Some(1));
        assert_eq!(
            lowered.get("max_completion_tokens").and_then(Value::as_u64),
            Some(1)
        );
        assert!(lowered.get("stream_options").is_none());
        assert!(lowered.get("min_tokens").is_none());
        Ok(())
    }

    #[test]
    fn decode_body_forwards_prefill_kv_transfer_params() -> Result<()> {
        let kv_transfer_params = json!({
            "remote_engine_id": "engine-p",
            "remote_host": "10.0.0.1",
            "remote_port": 5600,
        });

        let lowered = decode_body(&json!({"model": "m"}), kv_transfer_params.clone())
            .map_err(|error| anyhow::anyhow!(error.to_string()))?;

        assert_eq!(lowered.get("kv_transfer_params"), Some(&kv_transfer_params));
        Ok(())
    }

    #[tokio::test]
    async fn chat_dispatch_preserves_route_messages_and_unowned_fields() -> Result<()> {
        let prefill_backend = MockBackend::new(true);
        let decode_backend = MockBackend::new(false);
        let (prefill, prefill_server) = spawn_backend(prefill_backend.clone()).await?;
        let (decode, decode_server) = spawn_backend(decode_backend.clone()).await?;
        let state = ProxyState::new(Config {
            host: "127.0.0.1".to_owned(),
            port: 8000,
            prefill: vec![prefill],
            decode: vec![decode],
        })?;
        let request = json!({
            "model": "m",
            "messages": [{"role": "user", "content": "hello"}],
            "temperature": 1.0,
            "reasoning_effort": "high",
            "chat_template_kwargs": {"enable_thinking": true}
        });

        let response = completion_route(
            state,
            HeaderMap::new(),
            request.clone(),
            "/v1/chat/completions",
        )
        .await
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
        assert_eq!(response.status(), StatusCode::OK);
        let _body = to_bytes(response.into_body(), usize::MAX).await?;

        let prefill_requests = prefill_backend.requests.lock().await;
        let decode_requests = decode_backend.requests.lock().await;
        assert_eq!(prefill_requests.len(), 1);
        assert_eq!(decode_requests.len(), 1);
        for key in [
            "messages",
            "temperature",
            "reasoning_effort",
            "chat_template_kwargs",
        ] {
            assert_eq!(prefill_requests[0][key], request[key]);
            assert_eq!(decode_requests[0][key], request[key]);
        }
        prefill_server.abort();
        decode_server.abort();
        Ok(())
    }

    #[tokio::test]
    async fn streaming_decode_reaches_both_public_routes_before_terminal_event() -> Result<()> {
        for path in ["/v1/completions", "/v1/chat/completions"] {
            let terminal_gate = Arc::new(Notify::new());
            let prefill_backend = StreamingBackend::prefill(terminal_gate.clone());
            let decode_backend = StreamingBackend::decode(terminal_gate.clone());
            let (prefill, prefill_server) =
                spawn_streaming_backend(prefill_backend.clone()).await?;
            let (decode, decode_server) = spawn_streaming_backend(decode_backend).await?;
            let state = ProxyState::new(Config {
                host: "127.0.0.1".to_owned(),
                port: 8000,
                prefill: vec![prefill],
                decode: vec![decode],
            })?;
            let request = if path == "/v1/completions" {
                json!({"model": "m", "prompt": "hello", "stream": true})
            } else {
                json!({
                    "model": "m",
                    "messages": [{"role": "user", "content": "hello"}],
                    "stream": true
                })
            };

            let response = tokio::time::timeout(
                Duration::from_secs(1),
                completion_route(state, HeaderMap::new(), request, path),
            )
            .await
            .context("Gateway waited for decode completion before returning response headers")?
            .map_err(|error| anyhow::anyhow!(error.to_string()))?;
            assert_eq!(response.status(), StatusCode::ACCEPTED);
            assert_eq!(
                response.headers().get(header::CONTENT_TYPE),
                Some(&HeaderValue::from_static("text/event-stream"))
            );
            let mut stream = response.into_body().into_data_stream();
            let first = tokio::time::timeout(Duration::from_secs(1), stream.next())
                .await
                .context("Gateway buffered the first SSE event until decode completion")?
                .context("decode stream ended before its first SSE event")??;
            assert_eq!(first, Bytes::from_static(b"data: first\n\n"));

            terminal_gate.notify_one();
            let terminal = stream
                .next()
                .await
                .context("decode stream ended before its terminal SSE event")??;
            assert_eq!(terminal, Bytes::from_static(b"data: [DONE]\n\n"));
            prefill_server.abort();
            decode_server.abort();
        }
        Ok(())
    }

    #[tokio::test]
    async fn streaming_decode_preserves_pre_header_and_post_header_failures() -> Result<()> {
        let terminal_gate = Arc::new(Notify::new());
        let prefill_backend = StreamingBackend::prefill(terminal_gate.clone());
        let decode_backend = StreamingBackend::decode(terminal_gate.clone());
        let (prefill, prefill_server) = spawn_streaming_backend(prefill_backend).await?;
        let (decode, decode_server) = spawn_streaming_backend(decode_backend).await?;
        let state = ProxyState::new(Config {
            host: "127.0.0.1".to_owned(),
            port: 8000,
            prefill: vec![prefill],
            decode: vec![decode],
        })?;

        let error = match completion_route(
            state.clone(),
            HeaderMap::new(),
            json!({
                "model": "m",
                "prompt": "hello",
                "stream": true,
                "mode": "pre-header-error"
            }),
            "/v1/completions",
        )
        .await
        {
            Ok(_) => bail!("decode failure before headers should reject the public request"),
            Err(error) => error,
        };
        assert!(!error.into_response().status().is_success());

        let response = completion_route(
            state,
            HeaderMap::new(),
            json!({
                "model": "m",
                "prompt": "hello",
                "stream": true,
                "mode": "post-header-error"
            }),
            "/v1/completions",
        )
        .await
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
        let mut stream = response.into_body().into_data_stream();
        let first = stream
            .next()
            .await
            .context("decode stream ended before its first SSE event")??;
        assert_eq!(first, Bytes::from_static(b"data: first\n\n"));
        terminal_gate.notify_one();
        let result = stream
            .next()
            .await
            .context("decode stream completed successfully after an upstream body failure")?;
        let error = match result {
            Ok(_) => bail!("upstream body failure should fail the public stream"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("decode stream failed"));
        prefill_server.abort();
        decode_server.abort();
        Ok(())
    }

    #[test]
    fn proxy_state_requires_prefill_and_decode_targets() -> Result<()> {
        let result = ProxyState::new(Config {
            host: "127.0.0.1".to_owned(),
            port: 8000,
            prefill: Vec::new(),
            decode: vec!["http://127.0.0.1:8020".to_owned()],
        });
        let error = match result {
            Ok(_) => bail!("empty prefill targets should fail"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("at least one prefill endpoint"));
        Ok(())
    }

    #[derive(Clone)]
    struct MockBackend {
        requests: Arc<Mutex<Vec<Value>>>,
        prefill: bool,
    }

    impl MockBackend {
        fn new(prefill: bool) -> Self {
            Self {
                requests: Arc::new(Mutex::new(Vec::new())),
                prefill,
            }
        }
    }

    async fn mock_chat(
        State(state): State<MockBackend>,
        Json(body): Json<Value>,
    ) -> Response<Body> {
        state.requests.lock().await.push(body);
        if state.prefill {
            Json(json!({
                "kv_transfer_params": {
                    "remote_engine_id": "prefill-0",
                    "remote_host": "127.0.0.1",
                    "remote_port": 5600
                }
            }))
            .into_response()
        } else {
            Json(json!({"object": "chat.completion", "choices": []})).into_response()
        }
    }

    async fn spawn_backend(state: MockBackend) -> Result<(String, JoinHandle<()>)> {
        let app = Router::new()
            .route("/v1/chat/completions", post(mock_chat))
            .with_state(state);
        let listener = TcpListener::bind("127.0.0.1:0").await?;
        let address = listener.local_addr()?;
        let server = tokio::spawn(async move {
            let _result = serve(listener, app).await;
        });
        Ok((format!("http://{address}"), server))
    }

    #[derive(Clone)]
    struct StreamingBackend {
        prefill: bool,
        terminal_gate: Arc<Notify>,
    }

    impl StreamingBackend {
        fn prefill(terminal_gate: Arc<Notify>) -> Self {
            Self {
                prefill: true,
                terminal_gate,
            }
        }

        fn decode(terminal_gate: Arc<Notify>) -> Self {
            Self {
                prefill: false,
                terminal_gate,
            }
        }
    }

    async fn streaming_response(
        State(state): State<StreamingBackend>,
        Json(body): Json<Value>,
    ) -> Response<Body> {
        if state.prefill {
            return Json(json!({
                "kv_transfer_params": {
                    "remote_engine_id": "prefill-0",
                    "remote_host": "127.0.0.1",
                    "remote_port": 5600
                }
            }))
            .into_response();
        }

        let mode = body.get("mode").and_then(Value::as_str);
        if mode == Some("pre-header-error") {
            return (StatusCode::INTERNAL_SERVER_ERROR, "decode failed").into_response();
        }

        let terminal_gate = state.terminal_gate;
        let fail_after_first_event = mode == Some("post-header-error");
        let body = Body::from_stream(stream! {
            yield Ok::<Bytes, std::io::Error>(Bytes::from_static(b"data: first\n\n"));
            terminal_gate.notified().await;
            if fail_after_first_event {
                yield Err(std::io::Error::other("decode body failed"));
            } else {
                yield Ok(Bytes::from_static(b"data: [DONE]\n\n"));
            }
        });
        (
            StatusCode::ACCEPTED,
            [(header::CONTENT_TYPE, "text/event-stream")],
            body,
        )
            .into_response()
    }

    async fn spawn_streaming_backend(state: StreamingBackend) -> Result<(String, JoinHandle<()>)> {
        let app = Router::new()
            .route("/v1/completions", post(streaming_response))
            .route("/v1/chat/completions", post(streaming_response))
            .with_state(state);
        let listener = TcpListener::bind("127.0.0.1:0").await?;
        let address = listener.local_addr()?;
        let server = tokio::spawn(async move {
            let _result = serve(listener, app).await;
        });
        Ok((format!("http://{address}"), server))
    }
}
