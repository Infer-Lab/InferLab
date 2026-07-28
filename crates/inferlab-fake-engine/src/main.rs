use std::{
    net::SocketAddr,
    num::{NonZeroU32, NonZeroUsize},
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
