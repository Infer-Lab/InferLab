//! Serving topology, case, parallelism, and profiler-escape validation.

use super::{
    invalid, require_id, require_nonempty, require_reference, validate_expected_digest,
    validate_finite_json_value, validate_workspace_relative_source_path,
};
use crate::InferlabError;
use crate::workspace::definitions::{
    JsonValue, ProfilerEscapes, SyntheticAcceptanceDefinition, WorkspaceConfig,
};
use inferlab_protocol::{CaptureMechanism, Parallelism, ServeRoleKind, ServeTopology};
use std::collections::BTreeMap;
use std::path::Path;

pub(super) fn validate(root: &Path, config: &WorkspaceConfig) -> Result<(), InferlabError> {
    for (id, server) in &config.servers {
        require_id("server", id)?;
        require_reference("stack", &server.stack, &config.stacks)?;
        require_reference("model", &server.model, &config.models)?;
        if server.readiness_timeout_seconds == 0 {
            return invalid(format!(
                "server {id:?} readiness_timeout_seconds must be nonzero"
            ));
        }
        if server.readiness_attempt_timeout_seconds == Some(0) {
            return invalid(format!(
                "server {id:?} readiness_attempt_timeout_seconds must be nonzero"
            ));
        }
        if server.capture_control_deadline_seconds == Some(0) {
            return invalid(format!(
                "server {id:?} capture_control_deadline_seconds must be nonzero"
            ));
        }
        if server.capture_arm_deadline_seconds == Some(0) {
            return invalid(format!(
                "server {id:?} capture_arm_deadline_seconds must be nonzero"
            ));
        }
        if server.capture_finalization_deadline_seconds == Some(0) {
            return invalid(format!(
                "server {id:?} capture_finalization_deadline_seconds must be nonzero"
            ));
        }
        if server.topology == ServeTopology::Single
            && (server.pd_router_backend.is_some() || server.kv_transfer.is_some())
        {
            return invalid(format!(
                "single-topology server {id:?} must not declare pd_router_backend or kv_transfer"
            ));
        }
        if server.topology == ServeTopology::PrefillDecode
            && (server.gateway_backend.is_none() || server.pd_router_backend.is_none())
        {
            return invalid(format!(
                "prefill_decode server {id:?} must declare both gateway_backend and pd_router_backend"
            ));
        }
        if let Some(backend) = &server.gateway_backend {
            require_nonempty("server Gateway backend", id, backend)?;
        }
        if let Some(backend) = &server.pd_router_backend {
            require_nonempty("server P/D Router backend", id, backend)?;
        }
        validate_parallelism("server", id, &server.parallelism)?;
        crate::workspace::auxiliary_models::validate_auxiliary_models(
            &format!("server {id:?}"),
            &server.model,
            &server.auxiliary_models,
            config,
        )?;
        validate_synthetic_acceptance(
            root,
            &format!("server {id:?}"),
            &server.synthetic_acceptance,
        )?;
        validate_profiler_escapes(&format!("server {id:?}"), &server.profiler)?;
        validate_settings(&format!("server {id:?}"), &server.settings)?;
        for (role_id, role) in &server.roles {
            require_id("serve role", role_id)?;
            validate_server_role(id, server.topology, role_id)?;
            if role.replicas == Some(0) {
                return invalid(format!(
                    "serve role {role_id:?} replica count must be nonzero"
                ));
            }
            validate_parallelism("serve role", role_id, &role.parallelism)?;
            validate_profiler_escapes(&format!("server {id:?} role {role_id:?}"), &role.profiler)?;
            validate_settings(&format!("server {id:?} role {role_id:?}"), &role.settings)?;
        }
        if let Some(default_case) = &server.default_case
            && !server.cases.contains_key(default_case)
        {
            return invalid(format!(
                "server {id:?} default_case references unknown case {default_case:?}"
            ));
        }
        for (case_id, case) in &server.cases {
            require_id("server case", case_id)?;
            if !case.profiler.nsys.is_empty() {
                return invalid(format!(
                    "server case {case_id:?} declares nsys profiler escapes; escape inputs belong \
                     to the server and its roles, a case may only declare profiler.mechanism"
                ));
            }
            if case.readiness_timeout_seconds == Some(0) {
                return invalid(format!(
                    "server case {case_id:?} readiness_timeout_seconds must be nonzero"
                ));
            }
            if case.readiness_attempt_timeout_seconds == Some(0) {
                return invalid(format!(
                    "server case {case_id:?} readiness_attempt_timeout_seconds must be nonzero"
                ));
            }
            for (name, value) in [
                (
                    "capture_arm_deadline_seconds",
                    case.capture_arm_deadline_seconds,
                ),
                (
                    "capture_control_deadline_seconds",
                    case.capture_control_deadline_seconds,
                ),
                (
                    "capture_finalization_deadline_seconds",
                    case.capture_finalization_deadline_seconds,
                ),
            ] {
                if value == Some(0) {
                    return invalid(format!("server case {case_id:?} {name} must be nonzero"));
                }
            }
            if server.topology == ServeTopology::Single
                && (case.pd_router_backend.is_some() || case.kv_transfer.is_some())
            {
                return invalid(format!(
                    "single-topology server case {case_id:?} must not declare pd_router_backend or kv_transfer"
                ));
            }
            if case.gateway_backend.is_some() && server.gateway_backend.is_none() {
                return invalid(format!(
                    "server case {case_id:?} cannot add gateway_backend because the server base does not declare a Gateway"
                ));
            }
            if case.pd_router_backend.is_some() && server.pd_router_backend.is_none() {
                return invalid(format!(
                    "server case {case_id:?} cannot add pd_router_backend because the server base does not declare a P/D Router"
                ));
            }
            if let Some(backend) = &case.gateway_backend {
                require_nonempty("server case Gateway backend", case_id, backend)?;
            }
            if let Some(backend) = &case.pd_router_backend {
                require_nonempty("server case P/D Router backend", case_id, backend)?;
            }
            validate_parallelism("server case", case_id, &case.parallelism)?;
            validate_synthetic_acceptance(
                root,
                &format!("server case {case_id:?}"),
                &case.synthetic_acceptance,
            )?;
            validate_settings(&format!("server case {case_id:?}"), &case.settings)?;
            for (role_id, role) in &case.roles {
                require_id("server case role", role_id)?;
                validate_server_role(id, server.topology, role_id)?;
                if role.replicas == Some(0) {
                    return invalid(format!(
                        "server case {case_id:?} role {role_id:?} replica count must be nonzero"
                    ));
                }
                validate_parallelism("server case role", role_id, &role.parallelism)?;
                validate_settings(
                    &format!("server case {case_id:?} role {role_id:?}"),
                    &role.settings,
                )?;
            }
        }
    }
    Ok(())
}

/// Settings maps cross the adapter boundary as structured JSON, where a
/// non-finite float would collapse to null; finiteness is checked for every
/// layer before the extra_args segmentation gate.
fn validate_settings(
    context: &str,
    settings: &BTreeMap<String, JsonValue>,
) -> Result<(), InferlabError> {
    for (key, value) in settings {
        validate_finite_json_value(context, &format!("settings.{key}"), value)?;
    }
    validate_extra_args(context, settings)
}

/// [[RFC-0003:C-RESOLUTION]] extra_args segmentation is a workspace-load
/// obligation, not only a composition-time one: a value token preceding any
/// flag and a second bare `--` fail validation even when no overriding layer
/// ever touches the array.
fn validate_extra_args(
    context: &str,
    settings: &BTreeMap<String, JsonValue>,
) -> Result<(), InferlabError> {
    let Some(value) = settings.get("extra_args") else {
        return Ok(());
    };
    let path = format!("{context} settings.extra_args");
    let JsonValue::Array(items) = value else {
        return invalid(format!("extra_args at {path} must be an array"));
    };
    let tokens = items
        .iter()
        .map(|item| match item {
            JsonValue::String(token) => Ok(token.clone()),
            _ => invalid(format!("extra_args entries at {path} must be strings")),
        })
        .collect::<Result<Vec<_>, _>>()?;
    crate::toml_override::validate_extra_args_segmentation(&tokens, &path)
        .map_err(|message| InferlabError::InvalidConfig { message })
}

/// [[RFC-0003:C-SERVE-SYNTHETIC-ACCEPTANCE]] declaration shape: exactly one
/// of the explicit and curve forms, a finite acceptance length of at least
/// one, and well-formed curve coordinates. When the curve file is readable
/// and digest-matched, its two-shape contract, entry values, and the
/// thinking-mode-vs-flat rule also fail here at workspace validation; a
/// missing or unreadable file, a digest mismatch, and the lookup itself
/// belong to resolution.
fn validate_synthetic_acceptance(
    root: &Path,
    context: &str,
    declaration: &Option<SyntheticAcceptanceDefinition>,
) -> Result<(), InferlabError> {
    let Some(declaration) = declaration else {
        return Ok(());
    };
    match (&declaration.acceptance_length, &declaration.curve) {
        (Some(_), Some(_)) | (None, None) => invalid(format!(
            "{context} synthetic_acceptance must declare exactly one of acceptance_length or curve"
        )),
        (Some(length), None) if !length.is_finite() || *length < 1.0 => invalid(format!(
            "{context} synthetic_acceptance.acceptance_length must be a finite number of at least one"
        )),
        (Some(_), None) => Ok(()),
        (None, Some(curve)) => {
            let path = curve
                .path
                .to_str()
                .ok_or_else(|| InferlabError::InvalidConfig {
                    message: format!(
                        "{context} synthetic_acceptance.curve.path must be valid UTF-8"
                    ),
                })?;
            validate_workspace_relative_source_path(
                context,
                "synthetic_acceptance.curve.path",
                path,
            )?;
            validate_expected_digest(
                context,
                "synthetic_acceptance.curve.expected_sha256",
                &curve.expected_sha256,
            )?;
            if curve.model_key.is_empty() {
                return invalid(format!(
                    "{context} synthetic_acceptance.curve.model_key must not be empty"
                ));
            }
            if let Some(mode) = &curve.thinking_mode
                && mode.is_empty()
            {
                return invalid(format!(
                    "{context} synthetic_acceptance.curve.thinking_mode must be non-empty when present"
                ));
            }
            crate::workspace::synthetic_acceptance::validate_curve_shape_at_load(
                root,
                &format!("{context} synthetic_acceptance.curve"),
                curve,
            )
        }
    }
}

fn validate_server_role(
    server: &str,
    topology: ServeTopology,
    role: &str,
) -> Result<(), InferlabError> {
    let valid = match topology {
        ServeTopology::Single => role == ServeRoleKind::Serve.as_str(),
        ServeTopology::PrefillDecode => [
            ServeRoleKind::Prefill.as_str(),
            ServeRoleKind::Decode.as_str(),
        ]
        .contains(&role),
    };
    if valid {
        Ok(())
    } else {
        invalid(format!(
            "server {server:?} topology {topology:?} does not permit declared role {role:?}; \
             roles are canonical and router is derived"
        ))
    }
}

fn validate_parallelism(
    owner: &str,
    id: &str,
    parallelism: &Parallelism,
) -> Result<(), InferlabError> {
    let values = parallelism.field_values();
    if let Some((field, _)) = values.into_iter().find(|(_, value)| *value == Some(0)) {
        return invalid(format!(
            "{owner} {id:?} parallelism.{field} must be nonzero"
        ));
    }
    Ok(())
}

/// The managed and dedicated-field option names of the profiler escape gate
/// ([[RFC-0004:C-WORKLOAD-PROFILING]]). Attached short-option forms are
/// rejected alongside the long names because nsys 2026.3.1 parses them
/// (-tnone is --trace=none), as are strict-prefix abbreviations of a managed
/// name; launch's -w (--show-output) and -e (--env-var) stay allowed. The
/// surface was checked against nsys 2026.3.1 at qualification; re-check by
/// hand when the qualified nsys version changes ([[ADR-0006]]).
const MANAGED_ESCAPE_OPTIONS: &[&str] = &[
    "--session",
    "--session-new",
    "--output",
    "-o",
    "--export",
    "--force-overwrite",
    "-f",
    "--capture-range",
    "-c",
    "--capture-range-end",
    "--wait",
    "--trace",
    "-t",
    "--sample",
    "-s",
    "--cpuctxsw",
];

/// The short-option subset of the managed escape vocabulary: the
/// single-dash entries of [`MANAGED_ESCAPE_OPTIONS`], so adding a short form
/// to the parent list extends the attached-form rejection without a second
/// hand-synced list.
fn managed_short_options() -> impl Iterator<Item = &'static str> {
    MANAGED_ESCAPE_OPTIONS
        .iter()
        .copied()
        .filter(|option| !option.starts_with("--"))
}

pub(in crate::workspace) fn validate_profiler_escapes(
    context: &str,
    escapes: &ProfilerEscapes,
) -> Result<(), InferlabError> {
    // Engine-trace targets declare no profiler escape inputs
    // ([[RFC-0004:C-WORKLOAD-PROFILING]]); the composed view is re-checked at
    // resolution, where case and invocation layers are visible.
    if escapes.mechanism == Some(CaptureMechanism::EngineTrace) && !escapes.nsys.is_empty() {
        return invalid(format!(
            "{context} declares profiler mechanism engine_trace together with nsys escape \
             inputs; engine-trace targets declare no profiler escape inputs"
        ));
    }
    const MANAGED: &[&str] = MANAGED_ESCAPE_OPTIONS;
    for (field, options) in [
        ("launch_options", &escapes.nsys.launch_options),
        ("start_options", &escapes.nsys.start_options),
    ] {
        for option in options {
            // A standalone terminator ends option parsing and displaces the
            // managed argv tail into positionals of the wrapped command
            // ([[RFC-0004:C-WORKLOAD-PROFILING]]).
            if option == "-" || option == "--" {
                return invalid(format!(
                    "{context} nsys {field} contains standalone {option:?}, which ends \
                     option parsing and displaces the inferlab-managed argv tail"
                ));
            }
            let name = option.split('=').next().unwrap_or(option.as_str());
            let attached = !name.starts_with("--")
                && managed_short_options()
                    .any(|short| name.starts_with(short) && name.len() > short.len());
            // The qualified nsys resolves GNU-style abbreviations, so any
            // strict prefix of a managed long name either resolves to the
            // managed option or is an ambiguity
            // ([[RFC-0004:C-WORKLOAD-PROFILING]]).
            let abbreviated = name.starts_with("--")
                && MANAGED
                    .iter()
                    .any(|managed| managed.len() > name.len() && managed.starts_with(name));
            if MANAGED.contains(&name) || attached || abbreviated {
                return invalid(format!(
                    "{context} nsys {field} contains managed option {option:?}; use the \
                     dedicated profiler escape field or the inferlab-managed value"
                ));
            }
        }
    }
    for key in escapes.nsys.env.keys() {
        if !is_posix_identifier(key) {
            return invalid(format!(
                "{context} nsys env contains key {key:?}, which is not a POSIX identifier; \
                 environment entries reach the profiler commands as assignments"
            ));
        }
    }
    Ok(())
}

fn is_posix_identifier(name: &str) -> bool {
    let mut characters = name.chars();
    characters
        .next()
        .is_some_and(|first| first.is_ascii_alphabetic() || first == '_')
        && characters.all(|rest| rest.is_ascii_alphanumeric() || rest == '_')
}

#[cfg(test)]
mod tests {
    use super::*;
    use inferlab_profiler::plan::{
        CaptureDeadlines, CaptureSelection, CaptureWindowActionPlan,
        CaptureWindowControlEndpointPlan, CaptureWindowHttpMethodPlan, NsysEscapes,
        ProcessCapturePlan, ProcessPreparation, prepare_process,
    };
    use inferlab_profiler::record::CaptureActionRecord;
    use inferlab_profiler::session::CaptureSession;
    use inferlab_runtime::plan::{CommandPlan, LaunchPlan, ProcessEndpointPlan};
    use std::error::Error;

    // The short subset is derived from the managed vocabulary, not restated;
    // pin the derivation so a future short form enters the attached-form
    // rejection by extending the parent list alone.
    #[test]
    fn managed_short_options_are_the_short_forms_of_the_managed_vocabulary() {
        let mut derived: Vec<&str> = managed_short_options().collect();
        derived.sort_unstable();
        assert_eq!(derived, ["-c", "-f", "-o", "-s", "-t"]);
    }

    // The managed option vocabulary is spelled twice with no mechanical tie:
    // here as the escape reject list, and inline in inferlab-profiler's
    // launch and start argv rendering ([[RFC-0004:C-WORKLOAD-PROFILING]]).
    // Render one representative managed capture through the profiler's public
    // planning and session APIs and require every emitted long option to be
    // gated; the direction is subset because the gate also lists short forms
    // and configuration-dependent options. A producer-side vocabulary
    // addition fails this pin until the gate learns it.
    #[test]
    fn managed_escape_options_cover_the_emitted_nsys_vocabulary() -> Result<(), Box<dyn Error>> {
        let temp = tempfile::tempdir()?;
        let action = CaptureWindowActionPlan {
            method: CaptureWindowHttpMethodPlan::Post,
            path: "/profile".to_owned(),
            body: None,
            effective_url: "http://127.0.0.1:1/profile".to_owned(),
        };
        let command = CommandPlan {
            argv: vec!["serve-engine".to_owned()],
            env: BTreeMap::new(),
            explicit_env: Vec::new(),
            pass_env: Vec::new(),
            cwd: temp.path().to_path_buf(),
        };
        let capture = ProcessCapturePlan {
            mechanism: CaptureMechanism::ManagedCollection,
            capture_storage: None,
            window_control_endpoint: CaptureWindowControlEndpointPlan::ReplicaEntry,
            control_process_id: "serve-0".to_owned(),
            device_count: 1,
            start: action.clone(),
            stop: action,
            // The no-op executable lets the recorded session arm render the
            // real start argv without an Nsight Systems installation.
            escapes: NsysEscapes {
                executable: Some("true".to_owned()),
                ..NsysEscapes::default()
            },
        };
        let prepared = prepare_process(ProcessPreparation {
            record_id: "serve",
            role_id: "serve",
            replica_id: "serve",
            replica_index: 0,
            process_id: "serve-0",
            rank: Some(0),
            rank_count: Some(1),
            command: &command,
            launch: &LaunchPlan::Local,
            capture: Some(&capture),
            control_endpoint: Some(&ProcessEndpointPlan {
                host: "127.0.0.1".to_owned(),
                port: 1,
            }),
        })?;
        let target = prepared.target.ok_or("missing profiler target")?;
        let session = CaptureSession::open(
            "serve",
            "bench",
            &["w1".to_owned()],
            CaptureSelection {
                targets: vec![target.clone()],
                deadlines: CaptureDeadlines {
                    capture_arm_deadline_seconds: 60,
                    capture_control_deadline_seconds: 60,
                    capture_finalization_deadline_seconds: 60,
                },
            },
        )
        .map_err(|record| format!("capture arming failed: {record:?}"))?;
        let start_argv = session
            .finish()
            .arm
            .into_iter()
            .find_map(|action| match action {
                CaptureActionRecord::Command {
                    operation, argv, ..
                } if operation == "start-range-collection" => Some(argv),
                _ => None,
            })
            .ok_or("capture arm recorded no start-range-collection action")?;

        for (phase, argv) in [("launch", &target.launch_prefix), ("start", &start_argv)] {
            let emitted: Vec<&str> = argv
                .iter()
                .filter(|token| token.starts_with("--"))
                .map(|token| token.split('=').next().unwrap_or(token.as_str()))
                .collect();
            assert!(!emitted.is_empty(), "{phase} argv rendered no long options");
            let ungated: Vec<&str> = emitted
                .into_iter()
                .filter(|name| !MANAGED_ESCAPE_OPTIONS.contains(name))
                .collect();
            assert!(
                ungated.is_empty(),
                "{phase} argv emits managed options missing from MANAGED_ESCAPE_OPTIONS: \
                 {ungated:?}; extend the escape reject list"
            );
        }
        Ok(())
    }

    #[test]
    fn managed_and_dedicated_escape_options_are_rejected_in_both_lists() {
        let rejected = [
            "--session=other",
            "--session-new=other",
            "--output=/tmp/trace",
            "-o=/tmp/trace",
            "--export=sqlite",
            "--force-overwrite=false",
            "-f=false",
            "--capture-range=none",
            "-c=none",
            "--capture-range-end=stop",
            "--wait=none",
            "--trace=cuda",
            "-t=cuda",
            "--sample=cpu",
            "-s=cpu",
            "--cpuctxsw=none",
            "--wait",
            "-tnone",
            "-o/tmp/x",
            "-ftrue",
            "-cnone",
            "-snone",
            "--wai=all",
            "--out=/tmp/x",
            "--force=true",
            "--sess=x",
            "--w",
            "--wai",
        ];
        for option in rejected {
            for field in ["launch_options", "start_options"] {
                let mut escapes = ProfilerEscapes::default();
                let list = if field == "launch_options" {
                    &mut escapes.nsys.launch_options
                } else {
                    &mut escapes.nsys.start_options
                };
                list.push(option.to_owned());
                let error = validate_profiler_escapes("server \"pd\"", &escapes)
                    .err()
                    .map(|error| error.to_string());
                let expected = format!(
                    "server \"pd\" nsys {field} contains managed option {option:?}; \
                     use the dedicated profiler escape field or the inferlab-managed value"
                );
                assert!(
                    error
                        .as_deref()
                        .is_some_and(|error| error.contains(&expected)),
                    "{option} in {field}: {error:?}"
                );
            }
        }
        // Launch's -w is --show-output and -e is --env-var on the qualified
        // nsys; neither names a managed fact, in plain or attached form.
        let permitted = NsysEscapes {
            launch_options: vec![
                "-w=true".to_owned(),
                "-e=NSYS_FIXTURE=1".to_owned(),
                "-eNSYS_ATTACHED=1".to_owned(),
                "--cuda-graph-trace=node".to_owned(),
            ],
            start_options: vec![
                "--nic-metrics=true".to_owned(),
                "--stats=true".to_owned(),
                "-x=true".to_owned(),
                "-xtrue".to_owned(),
            ],
            ..NsysEscapes::default()
        };
        assert!(
            validate_profiler_escapes(
                "server \"pd\"",
                &ProfilerEscapes {
                    mechanism: None,
                    nsys: permitted
                },
            )
            .is_ok(),
            "nsys-owned options that name no managed fact pass the load gate"
        );
    }

    // A non-identifier key would be parsed as an option of the environment
    // utility rather than applied as an assignment
    // ([[RFC-0004:C-WORKLOAD-PROFILING]]).
    #[test]
    fn escape_env_keys_must_be_posix_identifiers() {
        for key in ["--unset", "1BAD", "BAD-KEY", "", "BAD KEY"] {
            let mut escapes = ProfilerEscapes::default();
            escapes.nsys.env.insert(key.to_owned(), "value".to_owned());
            let error = validate_profiler_escapes("server \"pd\"", &escapes)
                .err()
                .map(|error| error.to_string());
            let expected = format!(
                "server \"pd\" nsys env contains key {key:?}, which is not a POSIX \
                 identifier; environment entries reach the profiler commands as assignments"
            );
            assert!(
                error
                    .as_deref()
                    .is_some_and(|error| error.contains(&expected)),
                "{key:?}: {error:?}"
            );
        }
        for key in ["_OK", "OK2", "NSYS_FIXTURE"] {
            let mut escapes = ProfilerEscapes::default();
            escapes.nsys.env.insert(key.to_owned(), "value".to_owned());
            assert!(
                validate_profiler_escapes("server \"pd\"", &escapes).is_ok(),
                "{key:?} is a POSIX identifier and passes the load gate"
            );
        }
    }

    // A standalone terminator would splice ahead of the managed tail and
    // demote it to positionals of the wrapped command; on the qualified
    // nsys the start side even swallows it silently
    // ([[RFC-0004:C-WORKLOAD-PROFILING]]).
    #[test]
    fn standalone_terminators_are_rejected_in_both_lists() {
        for option in ["-", "--"] {
            for field in ["launch_options", "start_options"] {
                let mut escapes = ProfilerEscapes::default();
                let list = if field == "launch_options" {
                    &mut escapes.nsys.launch_options
                } else {
                    &mut escapes.nsys.start_options
                };
                list.push(option.to_owned());
                let error = validate_profiler_escapes("server \"pd\"", &escapes)
                    .err()
                    .map(|error| error.to_string());
                let expected = format!(
                    "server \"pd\" nsys {field} contains standalone {option:?}, \
                     which ends option parsing and displaces the inferlab-managed argv tail"
                );
                assert!(
                    error
                        .as_deref()
                        .is_some_and(|error| error.contains(&expected)),
                    "{option} in {field}: {error:?}"
                );
            }
        }
    }
}
