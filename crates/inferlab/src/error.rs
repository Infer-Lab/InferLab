use inferlab_protocol::AdapterErrorCode;
use std::path::PathBuf;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum GitError {
    #[error("failed to launch {operation}: {source}")]
    Launch {
        operation: String,
        #[source]
        source: std::io::Error,
    },
    #[error("{operation} stdin was not piped")]
    MissingStdin { operation: String },
    #[error("failed to write {operation} input: {source}")]
    WriteStdin {
        operation: String,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to collect {operation} output: {source}")]
    Wait {
        operation: String,
        #[source]
        source: std::io::Error,
    },
    #[error("{operation} exited with {status}: {diagnostics}")]
    Exit {
        operation: String,
        status: std::process::ExitStatus,
        diagnostics: String,
    },
    #[error("{operation} returned non-UTF-8 output: {source}")]
    Decode {
        operation: String,
        #[source]
        source: std::string::FromUtf8Error,
    },
    #[error("{operation} returned invalid output: {detail}")]
    InvalidOutput { operation: String, detail: String },
}

#[derive(Debug, Error)]
pub enum InferlabError {
    #[error("no `.inferlab/workspace.toml` found from {start} or its parents")]
    WorkspaceNotFound { start: PathBuf },

    #[error("failed to read {path}: {source}")]
    Read {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error(
        "failed to read render input {source_path:?} declared by integration {integration:?} at {path}: {source}"
    )]
    RenderInputRead {
        integration: String,
        source_path: String,
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error(
        "render input {source_path:?} declared by integration {integration:?} at {path} is not UTF-8: {source}"
    )]
    RenderInputUtf8 {
        integration: String,
        source_path: String,
        path: PathBuf,
        #[source]
        source: std::string::FromUtf8Error,
    },

    #[error("failed to parse {path}: {source}")]
    ParseToml {
        path: PathBuf,
        #[source]
        source: toml::de::Error,
    },

    #[error("failed to parse {path}: {source}")]
    ParseYaml {
        path: PathBuf,
        #[source]
        source: yaml_serde::Error,
    },

    #[error("failed to serialize temporary Pixi manifest: {source}")]
    SerializeToml {
        #[source]
        source: toml::ser::Error,
    },

    #[error("invalid configuration: {message}")]
    InvalidConfig { message: String },

    #[error("invalid configuration: network resolution failed: {source}")]
    NetworkResolution {
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    #[error("invalid server override {value:?}: {message}")]
    InvalidOverride { value: String, message: String },

    #[error("git command failed in {root}: {source}")]
    Git {
        root: PathBuf,
        #[source]
        source: GitError,
    },

    #[error("failed to launch pixi {action}: {source}")]
    LaunchPixi {
        action: &'static str,
        #[source]
        source: std::io::Error,
    },

    #[error("pixi {action} exited with {status}; stdout: {stdout}; stderr: {stderr}")]
    PixiExit {
        action: &'static str,
        status: std::process::ExitStatus,
        stdout: String,
        stderr: String,
    },

    #[error("Pixi environment lifecycle failed: {message}")]
    EnvironmentLifecycle { message: String },

    #[error("Pixi environment lifecycle failed: {source}")]
    EnvironmentInterrupt {
        #[source]
        source: inferlab_runtime::interrupt::InterruptInstallError,
    },

    #[error("failed to {operation} at {path}: {source}")]
    EnvironmentIo {
        path: PathBuf,
        operation: &'static str,
        #[source]
        source: std::io::Error,
    },

    #[error(
        "Pixi environment operation failed ({operation}) and workspace restoration also failed ({restoration})"
    )]
    EnvironmentRestore {
        operation: String,
        restoration: String,
    },

    #[error(
        "Pixi environment {environment:?} is not usable without package changes; run `{install_command}` first; diagnostics: {diagnostics}"
    )]
    PixiEnvironmentUnavailable {
        environment: String,
        install_command: String,
        diagnostics: String,
    },

    #[error("one or more stacks are not confirmed usable; see the status report")]
    StackStatusUnconfirmed,

    #[error("the Inferlab measurement toolchain does not support this host platform: {platform}")]
    UnsupportedToolchainPlatform { platform: String },

    #[error("failed to {operation} measurement toolchain path {path}: {source}")]
    ToolchainIo {
        operation: &'static str,
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error(
        "failed to remove held measurement toolchain path {path}: {source}; \
         held by {holders}; terminate the holding processes and rerun \
         `inferlab toolchain install`"
    )]
    ToolchainHeld {
        path: PathBuf,
        holders: String,
        #[source]
        source: std::io::Error,
    },

    #[error("failed to launch {action} for the measurement toolchain: {source}")]
    LaunchToolchain {
        action: &'static str,
        #[source]
        source: std::io::Error,
    },

    #[error(
        "measurement toolchain {action} exited with {status}; stdout: {stdout}; stderr: {stderr}"
    )]
    ToolchainExit {
        action: &'static str,
        status: std::process::ExitStatus,
        stdout: String,
        stderr: String,
    },

    #[error("measurement toolchain verification failed: {message}")]
    ToolchainVerification { message: String },

    #[error(
        "the measurement toolchain for Inferlab {version} on {platform} is not installed or complete; run `inferlab toolchain install` first"
    )]
    ToolchainUnavailable { version: String, platform: String },

    #[error("failed to serialize adapter request: {source}")]
    SerializeAdapterRequest {
        #[source]
        source: serde_json::Error,
    },

    #[error("failed to launch integration {integration:?}: {source}")]
    LaunchAdapter {
        integration: String,
        #[source]
        source: std::io::Error,
    },

    #[error("failed to communicate with integration {integration:?}: {source}")]
    AdapterIo {
        integration: String,
        #[source]
        source: std::io::Error,
    },

    #[error("integration {integration:?} did not finish within {seconds} seconds")]
    AdapterTimeout { integration: String, seconds: u64 },

    #[error("integration {integration:?} exited with {status}: {diagnostics}")]
    AdapterExit {
        integration: String,
        status: std::process::ExitStatus,
        diagnostics: String,
    },

    #[error(
        "integration {integration:?} returned invalid protocol JSON: {source}; diagnostics: {diagnostics}"
    )]
    AdapterProtocol {
        integration: String,
        #[source]
        source: serde_json::Error,
        diagnostics: String,
    },

    #[error("integration {integration:?} rejected the request ({code:?}): {message}")]
    AdapterRejected {
        integration: String,
        code: AdapterErrorCode,
        message: String,
    },

    #[error("{message}")]
    AdapterProtocolVersion { message: String },

    #[error(
        "machine binding {machine:?} provides {available} devices but the server requires {required}"
    )]
    InsufficientDevices {
        machine: String,
        required: u32,
        available: usize,
    },

    #[error("closed-loop recipe failed; record {record_id}")]
    RecipeFailed { record_id: String },

    #[error("Bench execution failed; record {record_id}")]
    BenchFailed { record_id: String },

    #[error("image build failed: {message}")]
    ImageBuild { message: String },

    #[error("image build failed: {source}")]
    ImageInterrupt {
        #[source]
        source: inferlab_runtime::interrupt::InterruptInstallError,
    },

    #[error(
        "failed to stage wheel payload from {source_path} at {staging_path}: {source}; partial staging cleanup failure: {cleanup:?}"
    )]
    WheelCacheStaging {
        source_path: PathBuf,
        staging_path: PathBuf,
        #[source]
        source: std::io::Error,
        cleanup: Option<std::io::Error>,
    },

    #[error(
        "failed to publish staged wheel {staging_path} at {target_path}: {source}; staging cleanup failure: {cleanup:?}"
    )]
    WheelCachePublication {
        staging_path: PathBuf,
        target_path: PathBuf,
        #[source]
        source: std::io::Error,
        cleanup: Option<std::io::Error>,
    },

    #[error("image tool {operation} could not be launched: {source}")]
    ImageToolLaunch {
        operation: String,
        #[source]
        source: std::io::Error,
    },

    #[error("image tool {operation} exited with {status}: {diagnostics}")]
    ImageToolExit {
        operation: String,
        status: std::process::ExitStatus,
        diagnostics: String,
    },

    #[error("image tool {operation} returned invalid JSON: {source}")]
    ImageToolJson {
        operation: String,
        #[source]
        source: serde_json::Error,
    },

    #[error("image selection rejected: {message}")]
    ImageSelection { message: String },

    #[error("image selection rejected: remote container preflight failed: {source}")]
    ImagePreflight {
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    #[error("image build completed with failures; record {record_id}")]
    ImageBuildFailed { record_id: String },

    #[error("server {id:?} already has an active workload or stop operation")]
    ServerBusy { id: String },

    #[error("server lifecycle failed: {message}")]
    ServerLifecycle { message: String },

    #[error("server lifecycle failed: {source}")]
    ServerInterrupt {
        #[source]
        source: inferlab_runtime::interrupt::InterruptInstallError,
    },

    #[error("remote execution preflight failed: {source}")]
    ServerPreflight {
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    #[error("built-in proxy failed: {source}")]
    Proxy {
        #[source]
        source: inferlab_proxy::ProxyError,
    },

    #[error("profiling failed: {source}")]
    Profiling {
        #[source]
        source: inferlab_profiler::error::ProfilerError,
    },

    #[error("profiling failed: {message}")]
    ProfilingEvidence { message: String },

    #[error("profile barrier failed to {operation}: {source}")]
    ProfileBarrierIo {
        operation: &'static str,
        #[source]
        source: std::io::Error,
    },

    #[error("profile barrier protocol failed: {message}")]
    ProfileBarrierProtocol { message: String },

    #[error("scratchpad operation failed: {message}")]
    Scratchpad { message: String },

    #[error("agent plugin operation failed: {message}")]
    Agent { message: String },

    #[error("ad-hoc execution failed: {message}")]
    AdHocRun { message: String },

    #[error("ad-hoc execution failed: {source}")]
    AdHocInterrupt {
        #[source]
        source: inferlab_runtime::interrupt::InterruptInstallError,
    },

    #[error("failed to {operation} dataset path {path}: {source}")]
    DatasetIo {
        operation: &'static str,
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("failed to download dataset from {url}: {source}")]
    DatasetHttp {
        url: String,
        #[source]
        source: reqwest::Error,
    },

    #[error("dataset snapshot at {path} has SHA-256 {observed}, expected {expected}")]
    DatasetDigest {
        path: PathBuf,
        expected: String,
        observed: String,
    },

    #[error("dataset preparation failed: {message}")]
    DatasetPreparation { message: String },

    #[error("failed to access server record {path}: {source}")]
    RecordIo {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("failed to decode server record {path}: {source}")]
    RecordDecode {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },

    #[error("failed to encode server record: {source}")]
    RecordEncode {
        #[source]
        source: serde_json::Error,
    },

    #[error("failed to {operation} operation observation {path}: {source}")]
    OperationObservationIo {
        operation: &'static str,
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("failed to encode operation observation {path}: {source}")]
    OperationObservationEncode {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },

    #[error("failed to write command output: {source}")]
    WriteOutput {
        #[source]
        source: std::io::Error,
    },

    #[error("failed to write command output: failed to install TUI interruption handler: {source}")]
    TuiInterrupt {
        #[source]
        source: inferlab_runtime::interrupt::InterruptInstallError,
    },

    #[error("failed to encode JSON: {source}")]
    EncodeOutput {
        #[source]
        source: serde_json::Error,
    },
}

impl From<inferlab_profiler::error::ProfilerError> for InferlabError {
    fn from(source: inferlab_profiler::error::ProfilerError) -> Self {
        Self::Profiling { source }
    }
}

impl From<inferlab_proxy::ProxyError> for InferlabError {
    fn from(source: inferlab_proxy::ProxyError) -> Self {
        Self::Proxy { source }
    }
}

impl InferlabError {
    /// The stable error code per [[RFC-0001:C-ERROR-CODES]]: the registry is
    /// append-only, so a code never changes meaning and is never reused.
    #[must_use]
    pub const fn code(&self) -> &'static str {
        match self {
            Self::WorkspaceNotFound { .. } => "E1001",
            Self::Read { .. } | Self::RenderInputRead { .. } | Self::RenderInputUtf8 { .. } => {
                "E1002"
            }
            Self::ParseToml { .. } => "E1003",
            Self::ParseYaml { .. } | Self::SerializeToml { .. } => "E1003",
            Self::InvalidConfig { .. } | Self::NetworkResolution { .. } => "E1004",
            Self::InvalidOverride { .. } => "E1005",
            Self::Git { .. } => "E1006",
            Self::LaunchPixi { .. }
            | Self::PixiExit { .. }
            | Self::EnvironmentLifecycle { .. }
            | Self::EnvironmentInterrupt { .. }
            | Self::EnvironmentIo { .. }
            | Self::EnvironmentRestore { .. }
            | Self::PixiEnvironmentUnavailable { .. }
            | Self::StackStatusUnconfirmed => "E1007",
            Self::UnsupportedToolchainPlatform { .. }
            | Self::ToolchainIo { .. }
            | Self::ToolchainHeld { .. }
            | Self::LaunchToolchain { .. }
            | Self::ToolchainExit { .. }
            | Self::ToolchainVerification { .. }
            | Self::ToolchainUnavailable { .. } => "E1008",
            Self::SerializeAdapterRequest { .. }
            | Self::LaunchAdapter { .. }
            | Self::AdapterIo { .. }
            | Self::AdapterTimeout { .. }
            | Self::AdapterExit { .. }
            | Self::AdapterProtocol { .. }
            | Self::AdapterRejected { .. }
            | Self::AdapterProtocolVersion { .. } => "E2001",
            Self::InsufficientDevices { .. } => "E3001",
            Self::RecipeFailed { .. }
            | Self::BenchFailed { .. }
            | Self::ImageBuildFailed { .. } => "E4001",
            Self::ImageBuild { .. }
            | Self::ImageInterrupt { .. }
            | Self::WheelCacheStaging { .. }
            | Self::WheelCachePublication { .. }
            | Self::ImageToolLaunch { .. }
            | Self::ImageToolExit { .. }
            | Self::ImageToolJson { .. }
            | Self::ImageSelection { .. }
            | Self::ImagePreflight { .. } => "E4003",
            Self::ServerLifecycle { .. }
            | Self::ServerInterrupt { .. }
            | Self::ServerPreflight { .. }
            | Self::ServerBusy { .. }
            | Self::Proxy { .. }
            | Self::Profiling { .. }
            | Self::ProfilingEvidence { .. }
            | Self::ProfileBarrierIo { .. }
            | Self::ProfileBarrierProtocol { .. }
            | Self::DatasetIo { .. }
            | Self::DatasetHttp { .. }
            | Self::DatasetDigest { .. }
            | Self::DatasetPreparation { .. } => "E4002",
            Self::RecordIo { .. } | Self::RecordDecode { .. } | Self::RecordEncode { .. } => {
                "E5001"
            }
            Self::OperationObservationIo { .. } | Self::OperationObservationEncode { .. } => {
                "E5002"
            }
            Self::Scratchpad { .. } => "E6001",
            Self::Agent { .. } => "E7001",
            Self::AdHocRun { .. } | Self::AdHocInterrupt { .. } => "E8001",
            Self::WriteOutput { .. } | Self::TuiInterrupt { .. } | Self::EncodeOutput { .. } => {
                "E9001"
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{GitError, InferlabError};
    use std::error::Error;
    use std::path::PathBuf;

    #[test]
    fn profiler_failure_retains_the_profiler_error_as_its_source()
    -> Result<(), Box<dyn std::error::Error>> {
        let error = InferlabError::from(inferlab_profiler::error::ProfilerError::NoTargets);
        let source = error
            .source()
            .ok_or_else(|| std::io::Error::other("profiling error source is missing"))?;
        assert_eq!(
            source.to_string(),
            "managed server has no prepared profiling targets"
        );
        Ok(())
    }

    #[test]
    fn git_failure_retains_root_operation_and_io_source() -> Result<(), Box<dyn std::error::Error>>
    {
        let error = InferlabError::Git {
            root: PathBuf::from("/workspace"),
            source: GitError::Launch {
                operation: "git status".to_owned(),
                source: std::io::Error::other("launch failed"),
            },
        };
        assert!(error.to_string().contains("/workspace"));
        let source = error
            .source()
            .ok_or_else(|| std::io::Error::other("Git boundary source is missing"))?;
        let git = source
            .downcast_ref::<GitError>()
            .ok_or_else(|| std::io::Error::other("Git boundary source has the wrong type"))?;
        assert!(matches!(git, GitError::Launch { operation, .. } if operation == "git status"));
        assert_eq!(
            git.source().map(ToString::to_string).as_deref(),
            Some("launch failed")
        );
        Ok(())
    }
}
