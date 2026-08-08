use std::{
    net::SocketAddr,
    num::{NonZeroU32, NonZeroU64, NonZeroUsize},
    process::ExitCode,
};

use clap::{Args, Parser, Subcommand};
use inferlab_fake_engine::smg::{SmgService, SmgServiceConfig, serve_smg_worker};
use thiserror::Error;

#[derive(Debug, Parser)]
#[command(
    name = "inferlab-token-engine",
    version,
    about = "Token-only Engine fixture compatible with the SMG gRPC Gateway"
)]
struct Cli {
    #[command(subcommand)]
    command: CliCommand,
}

#[derive(Debug, Subcommand)]
enum CliCommand {
    /// Serve the token Engine through SMG's worker protocol.
    SmgWorker(SmgWorkerCli),
}

#[derive(Debug, Args)]
struct SmgWorkerCli {
    /// Socket address allocated to this Engine process.
    #[arg(long)]
    listen: SocketAddr,

    /// Model locator used by the concrete Engine and reported to SMG.
    #[arg(long)]
    model: String,

    /// Public model identity reported to SMG.
    #[arg(long)]
    served_model_name: String,

    /// Number of devices jointly owned by this Engine process for pure TP.
    #[arg(long)]
    tensor_parallel_size: NonZeroU32,

    /// Output-token limit used when a request omits max_new_tokens.
    #[arg(long, default_value_t = 16)]
    default_max_output_tokens: u32,

    /// Maximum aggregate tokens admitted to one scheduled model iteration.
    #[arg(long, default_value = "12288")]
    max_num_batched_tokens: NonZeroUsize,

    /// Share of device memory the Engine may occupy.
    #[arg(long, default_value_t = 100, value_parser = clap::value_parser!(u32).range(1..=100))]
    gpu_memory_utilization_percent: u32,

    /// Device memory withheld from the Engine for execution workspaces.
    #[arg(long, default_value_t = 0)]
    workspace_reserve_mib: u32,

    /// Prefix-cache entries retained on device by each tensor-parallel rank.
    #[arg(long, default_value = "8")]
    prefix_cache_gpu_entries: NonZeroU32,

    /// Share of host memory backing the prefix cache when no explicit per-rank
    /// sizing is supplied.
    #[arg(long, default_value_t = 75, value_parser = clap::value_parser!(u32).range(1..=100))]
    prefix_cache_host_memory_percent: u32,

    /// Host prefix-cache bytes for each rank, paired with the NUMA nodes below
    /// by occurrence order.
    #[arg(long)]
    prefix_cache_cpu_bytes_per_rank: Vec<NonZeroU64>,

    /// NUMA node backing each rank's host prefix cache.
    #[arg(long)]
    prefix_cache_numa_node_per_rank: Vec<u32>,
}

#[derive(Debug, Error)]
enum CliError {
    #[error("fake Engine supports tensor parallel size 1, got {actual}")]
    UnsupportedTensorParallel { actual: u32 },
    #[error("failed to bind the SMG worker at {address}: {source}")]
    Bind {
        address: SocketAddr,
        #[source]
        source: std::io::Error,
    },
    #[error("SMG worker server failed at {address}: {source}")]
    Serve {
        address: SocketAddr,
        #[source]
        source: tonic::transport::Error,
    },
}

#[tokio::main]
async fn main() -> ExitCode {
    match run(Cli::parse()).await {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::FAILURE
        }
    }
}

async fn run(cli: Cli) -> Result<(), CliError> {
    match cli.command {
        CliCommand::SmgWorker(worker) => run_smg_worker(worker).await,
    }
}

async fn run_smg_worker(cli: SmgWorkerCli) -> Result<(), CliError> {
    let tensor_parallel_size = cli.tensor_parallel_size.get();
    if tensor_parallel_size != 1 {
        return Err(CliError::UnsupportedTensorParallel {
            actual: tensor_parallel_size,
        });
    }
    let service = SmgService::new(SmgServiceConfig::new(
        cli.model,
        cli.served_model_name,
        cli.default_max_output_tokens,
    ));
    let listener = tokio::net::TcpListener::bind(cli.listen)
        .await
        .map_err(|source| CliError::Bind {
            address: cli.listen,
            source,
        })?;
    eprintln!(
        "fake Engine listening at {} with max_num_batched_tokens={}",
        cli.listen, cli.max_num_batched_tokens
    );
    serve_smg_worker(listener, service, std::future::pending())
        .await
        .map_err(|source| CliError::Serve {
            address: cli.listen,
            source,
        })
}

#[cfg(test)]
mod tests {
    use clap::Parser;

    use super::{Cli, CliCommand, CliError, run};

    #[test]
    fn parses_the_canonical_token_engine_worker_contract() -> Result<(), clap::Error> {
        let cli = Cli::try_parse_from([
            "inferlab-token-engine",
            "smg-worker",
            "--listen",
            "127.0.0.1:50051",
            "--model",
            "/models/fake-model",
            "--served-model-name",
            "fake-model",
            "--tensor-parallel-size",
            "1",
            "--default-max-output-tokens",
            "3",
            "--max-num-batched-tokens",
            "4096",
        ])?;

        let CliCommand::SmgWorker(worker) = cli.command;
        assert_eq!(worker.listen.to_string(), "127.0.0.1:50051");
        assert_eq!(worker.model, "/models/fake-model");
        assert_eq!(worker.served_model_name, "fake-model");
        assert_eq!(worker.tensor_parallel_size.get(), 1);
        assert_eq!(worker.default_max_output_tokens, 3);
        assert_eq!(worker.max_num_batched_tokens.get(), 4_096);
        Ok(())
    }

    #[test]
    fn parses_the_memory_and_prefix_cache_contract_options() -> Result<(), clap::Error> {
        let cli = Cli::try_parse_from([
            "inferlab-token-engine",
            "smg-worker",
            "--listen",
            "127.0.0.1:50051",
            "--model",
            "/models/fake-model",
            "--served-model-name",
            "fake-model",
            "--tensor-parallel-size",
            "2",
            "--gpu-memory-utilization-percent",
            "90",
            "--workspace-reserve-mib",
            "2048",
            "--prefix-cache-gpu-entries",
            "16",
            "--prefix-cache-cpu-bytes-per-rank",
            "100",
            "--prefix-cache-cpu-bytes-per-rank",
            "200",
            "--prefix-cache-numa-node-per-rank",
            "3",
            "--prefix-cache-numa-node-per-rank",
            "4",
        ])?;

        let CliCommand::SmgWorker(worker) = cli.command;
        assert_eq!(worker.gpu_memory_utilization_percent, 90);
        assert_eq!(worker.workspace_reserve_mib, 2_048);
        assert_eq!(worker.prefix_cache_gpu_entries.get(), 16);
        // The two lists pair by occurrence order, so rank 0 is (100, 3).
        assert_eq!(
            worker
                .prefix_cache_cpu_bytes_per_rank
                .iter()
                .map(|bytes| bytes.get())
                .collect::<Vec<_>>(),
            [100, 200]
        );
        assert_eq!(worker.prefix_cache_numa_node_per_rank, [3, 4]);
        Ok(())
    }

    #[test]
    fn applies_the_worker_defaults_when_the_optional_contract_options_are_absent()
    -> Result<(), clap::Error> {
        let cli = Cli::try_parse_from([
            "inferlab-token-engine",
            "smg-worker",
            "--listen",
            "127.0.0.1:50051",
            "--model",
            "/models/fake-model",
            "--served-model-name",
            "fake-model",
            "--tensor-parallel-size",
            "1",
        ])?;

        let CliCommand::SmgWorker(worker) = cli.command;
        assert_eq!(worker.gpu_memory_utilization_percent, 100);
        assert_eq!(worker.workspace_reserve_mib, 0);
        assert_eq!(worker.prefix_cache_gpu_entries.get(), 8);
        assert_eq!(worker.prefix_cache_host_memory_percent, 75);
        assert!(worker.prefix_cache_cpu_bytes_per_rank.is_empty());
        assert!(worker.prefix_cache_numa_node_per_rank.is_empty());
        Ok(())
    }

    #[tokio::test]
    async fn rejects_a_parallel_width_the_fake_engine_cannot_execute() -> Result<(), clap::Error> {
        let cli = Cli::try_parse_from([
            "inferlab-token-engine",
            "smg-worker",
            "--listen",
            "127.0.0.1:50051",
            "--model",
            "/models/fake-model",
            "--served-model-name",
            "fake-model",
            "--tensor-parallel-size",
            "2",
        ])?;

        assert!(matches!(
            run(cli).await,
            Err(CliError::UnsupportedTensorParallel { actual: 2 })
        ));
        Ok(())
    }
}
