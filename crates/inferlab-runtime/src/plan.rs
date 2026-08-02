use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::PathBuf;

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ReadinessPlan {
    Http {
        path: String,
        /// `None` when the server is capture-armed: readiness remains
        /// unbounded while process exit and interruption still terminate it.
        timeout_seconds: Option<u64>,
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
        timeout_seconds: Option<u64>,
    },
    ProcessAlive,
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

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct ProcessEndpointPlan {
    pub host: String,
    pub port: u16,
}
