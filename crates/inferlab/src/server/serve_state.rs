//! The integration's serve-state probe: readiness waits until every
//! model-serving process reports itself engaged, and every attempt is recorded
//! ([[RFC-0003:C-SERVE-KV-STORE]], [[RFC-0006:C-INTEGRATIONS]]).

use crate::InferlabError;
use crate::adapter::ProcessAdapterClient;
use crate::execution::{ResolvedExecution, ServeStateProbePlan, StackPlan};
use inferlab_protocol::{ProbeServeInput, ProbeServeProcessState, ProbeServeResult};
use inferlab_runtime::operation_bound::OperationBound;
use inferlab_runtime::server::SystemProcessRuntime;
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::time::{Duration, Instant};

/// Reads the serve state through the integration; the control plane never
/// speaks the framework's protocol itself.
pub(crate) trait ServeStateProbe {
    fn probe_serve(
        &self,
        root: &Path,
        stack: &StackPlan,
        adapter_timeout: Duration,
        input: ProbeServeInput,
    ) -> Result<ProbeServeResult, InferlabError>;
}

impl ServeStateProbe for SystemProcessRuntime {
    fn probe_serve(
        &self,
        root: &Path,
        stack: &StackPlan,
        adapter_timeout: Duration,
        input: ProbeServeInput,
    ) -> Result<ProbeServeResult, InferlabError> {
        ProcessAdapterClient::new(adapter_timeout).probe_serve(
            root,
            &stack.integration,
            &stack.pixi_environment,
            input,
        )
    }
}

/// One probe attempt: whether every process was engaged, and the
/// integration's per-process answers or why the attempt failed.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ServeStateProbeEvidence {
    pub elapsed_ms: u64,
    pub engaged: bool,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub processes: Vec<ProbeServeProcessState>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

pub(crate) fn probe_input(
    resolved: &ResolvedExecution,
    plan: &ServeStateProbePlan,
) -> ProbeServeInput {
    ProbeServeInput {
        model: inferlab_protocol::ServeModelInput {
            id: resolved.server.model.id.clone(),
            served_name: resolved.server.model.served_name.clone(),
        },
        topology: resolved.server.topology,
        attempt_bound_seconds: plan.attempt_bound_seconds,
        processes: plan.processes.clone(),
        kv_store: plan.kv_store.clone(),
    }
}

/// Probe once and judge the answer: engaged only when the integration read
/// every process and reported each engaged.
pub(crate) fn probe_once<P: ServeStateProbe>(
    root: &Path,
    resolved: &ResolvedExecution,
    plan: &ServeStateProbePlan,
    prober: &P,
    started: Instant,
) -> ServeStateProbeEvidence {
    let outcome = prober.probe_serve(
        root,
        &resolved.stack,
        Duration::from_secs(plan.adapter_timeout_seconds),
        probe_input(resolved, plan),
    );
    judge(outcome, started)
}

/// Probe a bound server from a Bench case at a profiling window boundary
/// ([[RFC-0003:C-SERVE-KV-STORE]]).
pub(crate) fn probe_for_workload(
    root: &Path,
    probe: &crate::workload::WorkloadServeStateProbe,
    started: Instant,
) -> ServeStateProbeEvidence {
    let outcome = ProcessAdapterClient::new(Duration::from_secs(probe.adapter_timeout_seconds))
        .probe_serve(
            root,
            &probe.integration,
            &probe.pixi_environment,
            probe.input.clone(),
        );
    judge(outcome, started)
}

fn judge(
    outcome: Result<ProbeServeResult, InferlabError>,
    started: Instant,
) -> ServeStateProbeEvidence {
    let elapsed_ms = inferlab_runtime::operation_bound::duration_millis(started.elapsed());
    match outcome {
        Ok(result) => ServeStateProbeEvidence {
            elapsed_ms,
            engaged: result
                .processes
                .iter()
                .all(|process| process.engaged && process.error.is_none()),
            processes: result.processes,
            error: None,
        },
        Err(error) => ServeStateProbeEvidence {
            elapsed_ms,
            engaged: false,
            processes: Vec::new(),
            error: Some(error.to_string()),
        },
    }
}

/// Why the server never became engaged: the process that exited, when one
/// did, and the operator-facing description.
pub(super) struct EngagementFailure {
    pub process: Option<String>,
    pub message: String,
}

/// Probe until every process is engaged or the readiness budget ends,
/// recording each attempt. A started process that exits ends the wait at
/// once: `exited` names it ([[RFC-0003:C-RUNTIME-WORKFLOWS]]).
pub(super) fn wait_until_engaged<P: ServeStateProbe>(
    root: &Path,
    resolved: &ResolvedExecution,
    plan: &ServeStateProbePlan,
    prober: &P,
    bound: &OperationBound,
    exited: &dyn Fn() -> Option<String>,
    record: &mut dyn FnMut(ServeStateProbeEvidence) -> Result<(), InferlabError>,
) -> Result<Result<(), EngagementFailure>, InferlabError> {
    use inferlab_runtime::server::readiness_cadence::{
        POLL_INTERVAL, next_probe_interval, sleep_within_readiness,
    };
    let started = Instant::now();
    let mut interval = POLL_INTERVAL;
    loop {
        if let Some(process) = exited() {
            return Ok(Err(EngagementFailure {
                message: format!("process {process:?} exited before the server became ready"),
                process: Some(process),
            }));
        }
        let attempt = probe_once(root, resolved, plan, prober, started);
        let engaged = attempt.engaged;
        let summary = attempt_summary(&attempt);
        record(attempt)?;
        if engaged {
            return Ok(Ok(()));
        }
        if bound.is_expired() {
            return Ok(Err(EngagementFailure {
                process: None,
                message: format!(
                    "the serve-state probe never reported every process engaged within the readiness budget; last attempt: {summary}"
                ),
            }));
        }
        if inferlab_runtime::interrupt::received() {
            return Ok(Err(EngagementFailure {
                process: None,
                message: "server startup was interrupted".to_owned(),
            }));
        }
        sleep_within_readiness(bound, interval);
        interval = next_probe_interval(interval);
    }
}

fn attempt_summary(attempt: &ServeStateProbeEvidence) -> String {
    if let Some(error) = &attempt.error {
        return error.clone();
    }
    attempt
        .processes
        .iter()
        .map(|process| {
            let state = serde_json::to_string(&process.state).unwrap_or_default();
            match &process.error {
                Some(error) => format!("{}: {error}", process.process),
                None => format!("{} engaged={} {state}", process.process, process.engaged),
            }
        })
        .collect::<Vec<_>>()
        .join("; ")
}
