//! SMG gRPC compatibility around the token-only fake Engine.
//!
//! This module is a transport adapter, not part of model execution. It keeps
//! SMG's Gateway responsibilities outside the Engine core while implementing
//! the worker contract used by the routed-single workflow ([[ADR-0022]]).

use std::{future::Future, pin::Pin, sync::Arc};

use futures_util::{Stream, stream};
use smg_grpc_client::{common_proto as common, tokenspeed_proto as wire};
use tokio::net::TcpListener;
use tokio_stream::wrappers::TcpListenerStream;
use tonic::transport::Server;
use tonic::{Request, Response, Status};
use wire::{
    generate_response::Response as GenerateWireResponse,
    token_speed_scheduler_server::{TokenSpeedScheduler, TokenSpeedSchedulerServer},
};

use crate::{EchoEngine, GenerateRequest, TokenEngine};

const MAX_CONTEXT_LENGTH: i32 = 32_768;
const VOCAB_SIZE: i32 = 1_000_000;
const MAX_TOTAL_TOKENS: i32 = 1_000_000;

type GenerateStream =
    Pin<Box<dyn Stream<Item = Result<wire::GenerateResponse, Status>> + Send + 'static>>;
type KvEventStream =
    Pin<Box<dyn Stream<Item = Result<common::KvEventBatch, Status>> + Send + 'static>>;
type TokenizerStream =
    Pin<Box<dyn Stream<Item = Result<common::GetTokenizerChunk, Status>> + Send + 'static>>;

/// Metadata used only by the SMG worker contract and never by token execution.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SmgServiceConfig {
    model_path: String,
    served_model_name: String,
    default_max_output_tokens: u32,
}

impl SmgServiceConfig {
    #[must_use]
    pub fn new(
        model_path: impl Into<String>,
        served_model_name: impl Into<String>,
        default_max_output_tokens: u32,
    ) -> Self {
        Self {
            model_path: model_path.into(),
            served_model_name: served_model_name.into(),
            default_max_output_tokens,
        }
    }
}

/// Serve the token worker protocol and the standard gRPC health service that
/// TokenSpeed SMG uses while registering a worker.
pub async fn serve_smg_worker<F>(
    listener: TcpListener,
    service: SmgService,
    shutdown: F,
) -> Result<(), tonic::transport::Error>
where
    F: Future<Output = ()> + Send + 'static,
{
    let (_health_reporter, health_service) = tonic_health::server::health_reporter();
    Server::builder()
        .add_service(health_service)
        .add_service(TokenSpeedSchedulerServer::new(service))
        .serve_with_incoming_shutdown(TcpListenerStream::new(listener), shutdown)
        .await
}

/// Thin SMG worker service around [`EchoEngine`].
#[derive(Clone, Debug)]
pub struct SmgService {
    config: Arc<SmgServiceConfig>,
    engine: EchoEngine,
}

impl SmgService {
    #[must_use]
    pub fn new(config: SmgServiceConfig) -> Self {
        Self {
            config: Arc::new(config),
            engine: EchoEngine,
        }
    }
}

#[tonic::async_trait]
impl TokenSpeedScheduler for SmgService {
    type GenerateStream = GenerateStream;
    type GetTokenizerStream = TokenizerStream;
    type SubscribeKvEventsStream = KvEventStream;

    async fn generate(
        &self,
        request: Request<wire::GenerateRequest>,
    ) -> Result<Response<Self::GenerateStream>, Status> {
        let request = request.into_inner();
        let tokenized = request
            .tokenized
            .ok_or_else(|| Status::invalid_argument("tokenized input is required"))?;
        let prompt_tokens = u32::try_from(tokenized.input_ids.len())
            .map_err(|_| Status::resource_exhausted("prompt token count exceeds u32"))?;
        let max_output_tokens = request
            .sampling_params
            .and_then(|params| params.max_new_tokens)
            .unwrap_or(self.config.default_max_output_tokens);
        let output = self
            .engine
            .generate(&GenerateRequest {
                prompt_token_ids: tokenized.input_ids,
                max_output_tokens,
            })
            .map_err(|error| Status::invalid_argument(error.to_string()))?;

        let mut responses = Vec::with_capacity(output.token_ids.len().saturating_add(1));
        if request.stream {
            for (index, token_id) in output.token_ids.iter().copied().enumerate() {
                let completion_tokens = u32::try_from(index.saturating_add(1))
                    .map_err(|_| Status::resource_exhausted("output token count exceeds u32"))?;
                responses.push(Ok(wire::GenerateResponse {
                    request_id: request.request_id.clone(),
                    response: Some(GenerateWireResponse::Chunk(wire::GenerateStreamChunk {
                        token_ids: vec![token_id],
                        prompt_tokens,
                        completion_tokens,
                        cached_tokens: 0,
                        output_logprobs: None,
                        index: 0,
                    })),
                }));
            }
        }
        let completion_tokens = u32::try_from(output.token_ids.len())
            .map_err(|_| Status::resource_exhausted("output token count exceeds u32"))?;
        responses.push(Ok(wire::GenerateResponse {
            request_id: request.request_id,
            response: Some(GenerateWireResponse::Complete(wire::GenerateComplete {
                output_ids: output.token_ids,
                finish_reason: output.finish_reason.as_str().to_owned(),
                prompt_tokens,
                completion_tokens,
                cached_tokens: 0,
                output_logprobs: None,
                matched_stop: None,
                index: 0,
            })),
        }));

        Ok(Response::new(Box::pin(stream::iter(responses))))
    }

    async fn health_check(
        &self,
        _request: Request<wire::HealthCheckRequest>,
    ) -> Result<Response<wire::HealthCheckResponse>, Status> {
        Ok(Response::new(wire::HealthCheckResponse {
            healthy: true,
            message: "ready".to_owned(),
        }))
    }

    async fn abort(
        &self,
        _request: Request<wire::AbortRequest>,
    ) -> Result<Response<wire::AbortResponse>, Status> {
        Ok(Response::new(wire::AbortResponse {
            success: true,
            message: "fake generation is synchronous".to_owned(),
        }))
    }

    async fn get_model_info(
        &self,
        _request: Request<wire::GetModelInfoRequest>,
    ) -> Result<Response<wire::GetModelInfoResponse>, Status> {
        Ok(Response::new(wire::GetModelInfoResponse {
            model_path: self.config.model_path.clone(),
            tokenizer_path: self.config.model_path.clone(),
            served_model_name: self.config.served_model_name.clone(),
            model_type: "fake-specialized-engine".to_owned(),
            architectures: vec!["FakeTokenEngine".to_owned()],
            max_context_length: MAX_CONTEXT_LENGTH,
            max_req_input_len: MAX_CONTEXT_LENGTH,
            vocab_size: VOCAB_SIZE,
            eos_token_ids: Vec::new(),
            pad_token_id: 0,
            bos_token_id: 0,
            weight_version: "none".to_owned(),
            default_sampling_params_json: String::new(),
            supports_vision: false,
            supports_multimodal: false,
            supported_modalities: Vec::new(),
            model_dtype: "none".to_owned(),
            multimodal_encoder_dtype: "none".to_owned(),
        }))
    }

    async fn get_server_info(
        &self,
        _request: Request<wire::GetServerInfoRequest>,
    ) -> Result<Response<wire::GetServerInfoResponse>, Status> {
        Ok(Response::new(wire::GetServerInfoResponse {
            server_args: None,
            scheduler_info: None,
            active_requests: 0,
            is_paused: false,
            uptime_seconds: 0.0,
            max_total_num_tokens: MAX_TOTAL_TOKENS,
            tokenspeed_version: format!("inferlab-fake-engine/{}", env!("CARGO_PKG_VERSION")),
            start_time: None,
        }))
    }

    async fn get_loads(
        &self,
        _request: Request<wire::GetLoadsRequest>,
    ) -> Result<Response<wire::GetLoadsResponse>, Status> {
        Ok(Response::new(wire::GetLoadsResponse {
            timestamp: String::new(),
            version: env!("CARGO_PKG_VERSION").to_owned(),
            dp_rank_count: 1,
            loads: vec![wire::SchedulerLoad {
                dp_rank: 0,
                num_running_reqs: 0,
                num_waiting_reqs: 0,
                num_waiting_uncached_tokens: 0,
                num_total_reqs: 0,
                num_used_tokens: 0,
                max_total_num_tokens: MAX_TOTAL_TOKENS,
                max_running_requests: 1,
                token_usage: 0.0,
                gen_throughput: 0.0,
                cache_hit_rate: 0.0,
                utilization: 0.0,
                memory: None,
                queues: None,
            }],
            aggregate: None,
        }))
    }

    async fn flush_cache(
        &self,
        _request: Request<common::FlushCacheRequest>,
    ) -> Result<Response<common::FlushCacheResponse>, Status> {
        Ok(Response::new(common::FlushCacheResponse {
            success: true,
            message: "fake Engine has no KV cache".to_owned(),
        }))
    }

    async fn start_profile(
        &self,
        _request: Request<common::StartProfileRequest>,
    ) -> Result<Response<common::ProfileResponse>, Status> {
        Err(Status::unimplemented("fake Engine profiling"))
    }

    async fn stop_profile(
        &self,
        _request: Request<common::StopProfileRequest>,
    ) -> Result<Response<common::ProfileResponse>, Status> {
        Err(Status::unimplemented("fake Engine profiling"))
    }

    async fn get_tokenizer(
        &self,
        _request: Request<common::GetTokenizerRequest>,
    ) -> Result<Response<Self::GetTokenizerStream>, Status> {
        Err(Status::unimplemented(
            "SMG owns tokenizer loading for the fake Engine workflow",
        ))
    }

    async fn subscribe_kv_events(
        &self,
        _request: Request<common::SubscribeKvEventsRequest>,
    ) -> Result<Response<Self::SubscribeKvEventsStream>, Status> {
        Err(Status::unimplemented("fake Engine has no KV cache"))
    }
}
