use crate::error::ProfilerError;
use inferlab_protocol::{EndpointAssignment, SettingValue};
use inferlab_runtime::plan::{CommandPlan, LaunchPlan, ProcessEndpointPlan};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

const DEFAULT_TRACE: [&str; 2] = ["cuda", "nvtx"];

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct NsysEscapes {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub executable: Option<String>,
    pub launch_options: Vec<String>,
    pub start_options: Vec<String>,
    pub trace: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sampling: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context_switch: Option<String>,
    pub env: BTreeMap<String, String>,
}

impl NsysEscapes {
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self == &Self::default()
    }

    /// Role escapes merge into common server escapes: scalars replace, option
    /// lists concatenate with the role's after the common values, the trace set
    /// replaces, and environment entries merge with the role value winning
    /// ([[RFC-0004:C-WORKLOAD-PROFILING]]).
    #[must_use]
    pub fn merged_with(&self, role: &Self) -> Self {
        let mut env = self.env.clone();
        env.extend(
            role.env
                .iter()
                .map(|(key, value)| (key.clone(), value.clone())),
        );
        Self {
            executable: role.executable.clone().or_else(|| self.executable.clone()),
            launch_options: [self.launch_options.clone(), role.launch_options.clone()].concat(),
            start_options: [self.start_options.clone(), role.start_options.clone()].concat(),
            trace: if role.trace.is_empty() {
                self.trace.clone()
            } else {
                role.trace.clone()
            },
            sampling: role.sampling.clone().or_else(|| self.sampling.clone()),
            context_switch: role
                .context_switch
                .clone()
                .or_else(|| self.context_switch.clone()),
            env,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ProcessCapturePlan {
    pub window_control_endpoint: CaptureWindowControlEndpointPlan,
    pub control_process_id: String,
    pub start: CaptureWindowActionPlan,
    pub stop: CaptureWindowActionPlan,
    /// The merged escape inputs for this target's role
    /// ([[RFC-0004:C-WORKLOAD-PROFILING]]); the raw common and role
    /// declarations live on the server plan.
    #[serde(default, skip_serializing_if = "NsysEscapes::is_empty")]
    pub escapes: NsysEscapes,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CaptureWindowControlEndpointPlan {
    ReplicaEntry,
    Gateway,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct CaptureWindowActionPlan {
    pub method: CaptureWindowHttpMethodPlan,
    pub path: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub body: Option<BTreeMap<String, SettingValue>>,
    pub effective_url: String,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CaptureWindowHttpMethodPlan {
    Post,
}

pub struct ProcessPreparation<'a> {
    pub record_id: &'a str,
    pub role_id: &'a str,
    pub replica_id: &'a str,
    pub replica_index: u32,
    pub process_id: &'a str,
    pub rank: Option<u32>,
    pub command: &'a CommandPlan,
    pub launch: &'a LaunchPlan,
    pub capture: Option<&'a ProcessCapturePlan>,
    pub control_endpoint: Option<&'a ProcessEndpointPlan>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ProfilerTargetRecord {
    pub process_id: String,
    pub role_id: String,
    pub replica_id: String,
    pub replica_index: u32,
    pub rank: u32,
    pub session: String,
    pub executable: String,
    pub launch: ProfilerLaunch,
    pub finalization: ProfilerFinalization,
    pub control: ProfilerControl,
    pub supported_window_controls: Vec<WindowControlKind>,
    pub command_cwd: PathBuf,
    pub runtime_root: PathBuf,
    pub launch_prefix: Vec<String>,
    #[serde(default, skip_serializing_if = "NsysEscapes::is_empty")]
    pub escapes: NsysEscapes,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum ProfilerFinalization {
    NsysStop,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum WindowControlKind {
    FrameworkRange,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "kebab-case", deny_unknown_fields)]
pub enum ProfilerLaunch {
    Local,
    Ssh { target: String },
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "kebab-case", deny_unknown_fields)]
pub enum ProfilerControl {
    Http {
        window_control_endpoint: CaptureWindowControlEndpointPlan,
        process_id: String,
        endpoint: EndpointAssignment,
        start: CaptureWindowActionPlan,
        stop: CaptureWindowActionPlan,
    },
}

pub struct PreparedProcess {
    pub command: CommandPlan,
    pub target: Option<ProfilerTargetRecord>,
}

pub fn prepare_process(input: ProcessPreparation<'_>) -> Result<PreparedProcess, ProfilerError> {
    let Some(requirement) = input.capture else {
        return Ok(PreparedProcess {
            command: input.command.clone(),
            target: None,
        });
    };
    let rank = input
        .rank
        .ok_or_else(|| ProfilerError::TargetIsNotModelRank {
            process_id: input.process_id.to_owned(),
        })?;
    let session = session_name(input.record_id, input.process_id);
    let escapes = requirement.escapes.clone();
    let executable = escapes
        .executable
        .clone()
        .unwrap_or_else(|| "nsys".to_owned());
    let trace = if escapes.trace.is_empty() {
        DEFAULT_TRACE.join(",")
    } else {
        escapes.trace.join(",")
    };
    let mut launch_prefix = env_prefix(&escapes.env);
    launch_prefix.push(executable.clone());
    launch_prefix.push("launch".to_owned());
    launch_prefix.extend(escapes.launch_options.iter().cloned());
    launch_prefix.extend([
        "--session-new".to_owned(),
        session.clone(),
        format!("--trace={trace}"),
        "--wait=all".to_owned(),
    ]);
    let mut argv = launch_prefix.clone();
    argv.extend(input.command.argv.iter().cloned());
    let control_endpoint =
        input
            .control_endpoint
            .ok_or_else(|| ProfilerError::UnknownControlProcess {
                process_id: input.process_id.to_owned(),
                control_process_id: requirement.control_process_id.clone(),
            })?;
    let control = ProfilerControl::Http {
        window_control_endpoint: requirement.window_control_endpoint,
        process_id: requirement.control_process_id.clone(),
        endpoint: EndpointAssignment {
            host: control_endpoint.host.clone(),
            port: control_endpoint.port,
        },
        start: requirement.start.clone(),
        stop: requirement.stop.clone(),
    };
    Ok(PreparedProcess {
        command: CommandPlan {
            argv,
            env: input.command.env.clone(),
            explicit_env: input.command.explicit_env.clone(),
            pass_env: input.command.pass_env.clone(),
            cwd: input.command.cwd.clone(),
        },
        target: Some(ProfilerTargetRecord {
            process_id: input.process_id.to_owned(),
            role_id: input.role_id.to_owned(),
            replica_id: input.replica_id.to_owned(),
            replica_index: input.replica_index,
            rank,
            session,
            executable,
            launch: match input.launch {
                LaunchPlan::Local => ProfilerLaunch::Local,
                LaunchPlan::Ssh { target } => ProfilerLaunch::Ssh {
                    target: target.clone(),
                },
            },
            finalization: ProfilerFinalization::NsysStop,
            control,
            supported_window_controls: vec![WindowControlKind::FrameworkRange],
            command_cwd: input.command.cwd.clone(),
            runtime_root: input
                .command
                .cwd
                .join("runtime")
                .join(input.record_id)
                .join(input.process_id)
                .join("profiles"),
            launch_prefix,
            escapes,
        }),
    })
}

pub(crate) fn env_prefix(env: &BTreeMap<String, String>) -> Vec<String> {
    if env.is_empty() {
        return Vec::new();
    }
    let mut argv = vec!["env".to_owned(), "--".to_owned()];
    argv.extend(env.iter().map(|(key, value)| format!("{key}={value}")));
    argv
}

fn session_name(record_id: &str, process_id: &str) -> String {
    format!(
        "inferlab-{}-{}",
        sanitize_segment(record_id),
        sanitize_segment(process_id)
    )
}

fn sanitize_segment(value: &str) -> String {
    value
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || matches!(character, '-' | '_') {
                character
            } else {
                '-'
            }
        })
        .collect()
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CapturePlanRecord {
    pub server_record_id: String,
    pub workload_id: String,
    pub deadlines: CaptureDeadlines,
    pub control: WindowControlKind,
    pub windows: Vec<CaptureWindowPlan>,
    pub targets: Vec<CaptureTargetPlan>,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CaptureDeadlines {
    pub capture_arm_deadline_seconds: u64,
    pub capture_control_deadline_seconds: u64,
    pub capture_finalization_deadline_seconds: u64,
}

pub struct CaptureSelection {
    pub targets: Vec<ProfilerTargetRecord>,
    pub deadlines: CaptureDeadlines,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CaptureWindowPlan {
    pub id: String,
    pub range_index: Option<usize>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CaptureTargetPlan {
    pub process_id: String,
    pub role_id: String,
    pub replica_id: String,
    pub replica_index: u32,
    pub rank: u32,
    pub session: String,
    pub expected_range_count: Option<usize>,
    pub output_base: PathBuf,
    pub reports: Vec<PathBuf>,
}

pub(crate) fn compile_plan(
    server_record_id: &str,
    workload_id: &str,
    window_ids: &[String],
    targets: &[ProfilerTargetRecord],
    deadlines: CaptureDeadlines,
) -> Result<CapturePlanRecord, ProfilerError> {
    if targets.is_empty() {
        return Err(ProfilerError::NoTargets);
    }
    if targets.iter().any(|target| {
        !target
            .supported_window_controls
            .contains(&WindowControlKind::FrameworkRange)
    }) {
        return Err(ProfilerError::UnsupportedWindowControl);
    }
    if window_ids.is_empty() {
        return Err(ProfilerError::NoStaticWindows);
    }
    let control = WindowControlKind::FrameworkRange;
    let windows = window_ids
        .iter()
        .enumerate()
        .map(|(index, id)| CaptureWindowPlan {
            id: id.clone(),
            range_index: Some(index + 1),
        })
        .collect::<Vec<_>>();
    let targets = targets
        .iter()
        .map(|target| {
            let output_base = target
                .runtime_root
                .join(sanitize_segment(workload_id))
                .join("trace");
            let reports = windows
                .iter()
                .map(|window| report_path(&output_base, window.range_index))
                .collect();
            CaptureTargetPlan {
                process_id: target.process_id.clone(),
                role_id: target.role_id.clone(),
                replica_id: target.replica_id.clone(),
                replica_index: target.replica_index,
                rank: target.rank,
                session: target.session.clone(),
                expected_range_count: Some(windows.len()),
                output_base,
                reports,
            }
        })
        .collect();
    Ok(CapturePlanRecord {
        server_record_id: server_record_id.to_owned(),
        workload_id: workload_id.to_owned(),
        deadlines,
        control,
        windows,
        targets,
    })
}

fn report_path(output_base: &Path, range_index: Option<usize>) -> PathBuf {
    match range_index {
        Some(index) => PathBuf::from(format!("{}.{index}.nsys-rep", output_base.display())),
        None => output_base.with_extension("nsys-rep"),
    }
}
