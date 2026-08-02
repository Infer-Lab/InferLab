#[derive(Debug, thiserror::Error)]
pub enum ProfilerError {
    #[error("profiling target {process_id:?} is not a model-rank process")]
    TargetIsNotModelRank { process_id: String },
    #[error(
        "profiling target {process_id:?} references unknown control process {control_process_id:?}"
    )]
    UnknownControlProcess {
        process_id: String,
        control_process_id: String,
    },
    #[error("managed server has no prepared profiling targets")]
    NoTargets,
    #[error("profiling target does not support framework-range control")]
    UnsupportedWindowControl,
    #[error("range-backed profiling requires static workload windows")]
    NoStaticWindows,
    #[error("capture plan contains no window {window_id:?}")]
    UnknownWindow { window_id: String },
    #[error("{message}")]
    WindowStartFailed { window_id: String, message: String },
}
