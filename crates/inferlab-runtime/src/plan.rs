use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::PathBuf;

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ReadinessPlan {
    /// The readiness budget and attempt timeout are server facts resolved
    /// once ([[RFC-0003:C-RESOLUTION]]); a process plan carries only its
    /// probe shape.
    Http {
        path: String,
    },
    HttpTargetRegistry {
        readiness_path: String,
        registry_path: String,
        targets_field: String,
        target_url_field: String,
        target_role_field: String,
        target_healthy_field: String,
        target_bootstrap_port_field: String,
        expected_targets: Vec<TargetRegistryExpectedTarget>,
    },
    /// A framework-neutral registry whose entries must list every expected
    /// target under its role at its allocated `host:port`
    /// ([[RFC-0006:C-INTEGRATIONS]]). Pointers are RFC 6901 JSON Pointers.
    RegistryMembership {
        registry_path: String,
        entries_pointer: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        entry_filter: Option<JsonValueMatchPlan>,
        role_pointer: String,
        address_pointer: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        model_list: Option<ModelListPlan>,
        expected_targets: Vec<RegistryMemberTarget>,
    },
    ProcessAlive,
}

/// A JSON Pointer and the string value expected at it.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct JsonValueMatchPlan {
    pub pointer: String,
    pub value: String,
}

/// A served-model list that must name `served_model`.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct ModelListPlan {
    pub path: String,
    pub models_pointer: String,
    pub name_pointer: String,
    pub served_model: String,
}

/// One rank-zero model-serving process a registry must list.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct RegistryMemberTarget {
    pub process: String,
    pub role: String,
    /// The allocated `host:port` the entry's address member must begin with.
    pub address: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct TargetRegistryExpectedTarget {
    pub url: String,
    pub role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bootstrap_port: Option<u16>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct LaunchFilePlan {
    pub relative_path: String,
    pub resolved_path: PathBuf,
    pub text: String,
    pub sha256: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "kebab-case")]
pub enum LaunchPlan {
    Local,
    Ssh { target: String },
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct CommandPlan {
    pub argv: Vec<String>,
    pub env: BTreeMap<String, String>,
    /// Variables explicitly rendered by resolution or an integration, as
    /// distinct from ambient environment composed for local execution.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub explicit_env: Vec<String>,
    /// Names whose values flow from the launching machine into a
    /// containerized process without projecting remote values into the plan.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub pass_env: Vec<String>,
    pub cwd: PathBuf,
}

impl CommandPlan {
    /// The per-process runtime directory on the process's own machine,
    /// `<cwd>/runtime/<record_id>/<process_id>`: an SSH launch lands its
    /// log and handle files here, and a profiler capture keeps its output
    /// in a `profiles` subdirectory.
    #[must_use]
    pub fn runtime_dir(&self, record_id: &str, process_id: &str) -> PathBuf {
        self.cwd.join("runtime").join(record_id).join(process_id)
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct ProcessEndpointPlan {
    pub host: String,
    pub port: u16,
}
