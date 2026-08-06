//! Serving topology, case, parallelism, and profiler-escape validation.

use super::{invalid, require_id, require_nonempty, require_reference};
use crate::InferlabError;
use crate::workspace::definitions::{ProfilerEscapes, WorkspaceConfig};
use inferlab_protocol::{Parallelism, ServeTopology};

pub(super) fn validate(config: &WorkspaceConfig) -> Result<(), InferlabError> {
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
        validate_profiler_escapes(&format!("server {id:?}"), &server.profiler)?;
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
            for (role_id, role) in &case.roles {
                require_id("server case role", role_id)?;
                validate_server_role(id, server.topology, role_id)?;
                if role.replicas == Some(0) {
                    return invalid(format!(
                        "server case {case_id:?} role {role_id:?} replica count must be nonzero"
                    ));
                }
                validate_parallelism("server case role", role_id, &role.parallelism)?;
            }
        }
    }
    Ok(())
}

fn validate_server_role(
    server: &str,
    topology: ServeTopology,
    role: &str,
) -> Result<(), InferlabError> {
    let valid = match topology {
        ServeTopology::Single => role == "serve",
        ServeTopology::PrefillDecode => matches!(role, "prefill" | "decode"),
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
    let values = [
        (
            "outer.tensor_parallel_size",
            parallelism
                .outer
                .as_ref()
                .and_then(|outer| outer.tensor_parallel_size),
        ),
        (
            "outer.pipeline_parallel_size",
            parallelism
                .outer
                .as_ref()
                .and_then(|outer| outer.pipeline_parallel_size),
        ),
        (
            "attention.tensor_parallel_size",
            parallelism
                .attention
                .as_ref()
                .and_then(|attention| attention.tensor_parallel_size),
        ),
        (
            "attention.data_parallel_size",
            parallelism
                .attention
                .as_ref()
                .and_then(|attention| attention.data_parallel_size),
        ),
        (
            "attention.context_parallel_size",
            parallelism
                .attention
                .as_ref()
                .and_then(|attention| attention.context_parallel_size),
        ),
        (
            "experts.tensor_parallel_size",
            parallelism
                .experts
                .as_ref()
                .and_then(|experts| experts.tensor_parallel_size),
        ),
        (
            "experts.data_parallel_size",
            parallelism
                .experts
                .as_ref()
                .and_then(|experts| experts.data_parallel_size),
        ),
        (
            "experts.expert_parallel_size",
            parallelism
                .experts
                .as_ref()
                .and_then(|experts| experts.expert_parallel_size),
        ),
        (
            "experts.dense_tensor_parallel_size",
            parallelism
                .experts
                .as_ref()
                .and_then(|experts| experts.dense_tensor_parallel_size),
        ),
    ];
    if let Some((field, _)) = values.into_iter().find(|(_, value)| *value == Some(0)) {
        return invalid(format!(
            "{owner} {id:?} parallelism.{field} must be nonzero"
        ));
    }
    Ok(())
}

/// Escape options that name a managed profiler fact are rejected at load
/// ([[RFC-0004:C-WORKLOAD-PROFILING]]): session identity, report
/// storage/export/overwrite lifecycle, capture-range mechanics, launch
/// wait, and the free-list forms of the dedicated trace, sampling, and
/// context-switch fields — in long, short, and attached short-option-value
/// forms, because nsys 2026.3.1 parses -tnone as --trace=none. Shorthands
/// follow that nsys: launch carries -t for --trace; start carries -o, -f,
/// -c, and -s. Launch's -w is --show-output and -e is --env-var, so neither
/// is rejected. Environment keys must be POSIX identifiers so no key can be
/// parsed as an option of the environment utility.
/// The managed and dedicated-field option names of the profiler escape gate
/// ([[RFC-0004:C-WORKLOAD-PROFILING]]). The strict-prefix abbreviation rule
/// was checked against the qualified nsys 2026.3.1 launch and start option
/// surfaces at qualification (no legitimate option is a strict prefix of a
/// managed name); re-check by hand when the qualified nsys version changes
/// ([[ADR-0006]]).
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

pub(in crate::workspace) fn validate_profiler_escapes(
    context: &str,
    escapes: &ProfilerEscapes,
) -> Result<(), InferlabError> {
    const MANAGED: &[&str] = MANAGED_ESCAPE_OPTIONS;
    const MANAGED_SHORT: &[&str] = &["-t", "-o", "-f", "-c", "-s"];
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
                && MANAGED_SHORT
                    .iter()
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
