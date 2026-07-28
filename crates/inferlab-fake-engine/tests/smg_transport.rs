use std::error::Error;

use futures_util::TryStreamExt;
use inferlab_fake_engine::smg::{SmgService, SmgServiceConfig, serve_smg_worker};
use smg_grpc_client::tokenspeed_proto as wire;
use tonic::transport::Endpoint;
use tonic_health::pb::{
    HealthCheckRequest as StandardHealthCheckRequest, health_check_response::ServingStatus,
    health_client::HealthClient,
};
use wire::token_speed_scheduler_client::TokenSpeedSchedulerClient;
use wire::token_speed_scheduler_server::TokenSpeedScheduler;

#[tokio::test]
async fn smg_generate_discards_original_text_before_entering_the_token_core()
-> Result<(), Box<dyn Error>> {
    let service = SmgService::new(SmgServiceConfig::new("/models/fake-model", "fake-model", 4));
    let response = service
        .generate(tonic::Request::new(wire::GenerateRequest {
            request_id: "request-1".to_owned(),
            tokenized: Some(wire::TokenizedInput {
                input_ids: vec![41, 42],
                original_text: "this field is transport-only".to_owned(),
            }),
            sampling_params: Some(wire::SamplingParams {
                max_new_tokens: Some(3),
                ..Default::default()
            }),
            stream: true,
            ..Default::default()
        }))
        .await?
        .into_inner()
        .try_collect::<Vec<_>>()
        .await?;

    let responses = response
        .iter()
        .filter_map(|item| item.response.as_ref())
        .collect::<Vec<_>>();
    assert_eq!(responses.len(), 4);
    assert!(matches!(
        responses.as_slice(),
        [
            wire::generate_response::Response::Chunk(first),
            wire::generate_response::Response::Chunk(second),
            wire::generate_response::Response::Chunk(third),
            wire::generate_response::Response::Complete(complete),
        ] if first.token_ids == [41]
            && second.token_ids == [42]
            && third.token_ids == [41]
            && complete.output_ids == [41, 42, 41]
            && complete.finish_reason == "length"
    ));
    Ok(())
}

#[tokio::test]
async fn smg_control_surface_reports_the_fixture_identity_and_noop_cache()
-> Result<(), Box<dyn Error>> {
    let service = SmgService::new(SmgServiceConfig::new("/models/fake-model", "fake-model", 4));

    let model = service
        .get_model_info(tonic::Request::new(wire::GetModelInfoRequest {}))
        .await?
        .into_inner();
    assert_eq!(model.model_path, "/models/fake-model");
    assert_eq!(model.served_model_name, "fake-model");
    assert_eq!(model.tokenizer_path, "/models/fake-model");

    let server = service
        .get_server_info(tonic::Request::new(wire::GetServerInfoRequest {}))
        .await?
        .into_inner();
    assert_eq!(server.active_requests, 0);
    assert!(
        server
            .tokenspeed_version
            .starts_with("inferlab-fake-engine/")
    );

    let loads = service
        .get_loads(tonic::Request::new(wire::GetLoadsRequest::default()))
        .await?
        .into_inner();
    assert_eq!(loads.dp_rank_count, 1);
    assert!(matches!(loads.loads.as_slice(), [load] if load.dp_rank == 0));

    let aborted = service
        .abort(tonic::Request::new(wire::AbortRequest::default()))
        .await?
        .into_inner();
    assert!(aborted.success);

    let flushed = service
        .flush_cache(tonic::Request::new(
            smg_grpc_client::common_proto::FlushCacheRequest { timeout_s: 0.0 },
        ))
        .await?
        .into_inner();
    assert!(flushed.success);
    Ok(())
}

#[tokio::test]
async fn published_smg_client_reaches_the_fake_engine_over_grpc() -> Result<(), Box<dyn Error>> {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
    let address = listener.local_addr()?;
    let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel::<()>();
    let service = SmgService::new(SmgServiceConfig::new("/models/fake-model", "fake-model", 2));
    let server = tokio::spawn(serve_smg_worker(listener, service, async {
        let _ = shutdown_rx.await;
    }));

    let endpoint = format!("http://{address}");
    let channel = Endpoint::from_shared(endpoint.clone())?.connect().await?;
    let mut discovery = HealthClient::new(channel);
    let health = discovery
        .check(StandardHealthCheckRequest {
            service: String::new(),
        })
        .await?
        .into_inner();
    assert_eq!(health.status, ServingStatus::Serving as i32);

    let mut client = TokenSpeedSchedulerClient::connect(endpoint).await?;
    let health = client
        .health_check(wire::HealthCheckRequest {})
        .await?
        .into_inner();
    assert!(health.healthy);

    let mut output = client
        .generate(wire::GenerateRequest {
            request_id: "network-request".to_owned(),
            tokenized: Some(wire::TokenizedInput {
                input_ids: vec![7, 8],
                original_text: String::new(),
            }),
            sampling_params: Some(wire::SamplingParams {
                max_new_tokens: Some(2),
                ..Default::default()
            }),
            stream: false,
            ..Default::default()
        })
        .await?
        .into_inner();
    let response = output.message().await?;
    assert!(matches!(
        response.and_then(|item| item.response),
        Some(wire::generate_response::Response::Complete(complete))
            if complete.output_ids == [7, 8]
    ));

    let _ = shutdown_tx.send(());
    server.await??;
    Ok(())
}
