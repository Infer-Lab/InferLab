//! Machine-local binding model and invariants; none of these facts belong to
//! the shareable workspace catalog.

use super::catalog_validation::{require_id, require_nonempty};
use super::invalid;
use crate::InferlabError;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;
use std::time::Duration;

const DEFAULT_ADAPTER_TIMEOUT: Duration = Duration::from_secs(30);
const DEFAULT_IMAGE_ADAPTER_TIMEOUT: Duration = Duration::from_secs(120);

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct LocalBindings {
    #[serde(default)]
    pub default_placement: Option<String>,
    #[serde(default)]
    pub model_weights: BTreeMap<String, ModelWeightBinding>,
    #[serde(default)]
    pub machines: BTreeMap<String, MachineBinding>,
    #[serde(default)]
    pub placements: BTreeMap<String, PlacementBinding>,
    #[serde(default)]
    pub builders: BTreeMap<String, BuilderBinding>,
    #[serde(default)]
    pub adapter: AdapterBinding,
}

/// Machine-private facts for process- and image-backed integration lowering
/// ([[RFC-0003:C-RUNTIME-WORKFLOWS]]), including their independent deadlines
/// and the optional device workaround for container runtimes that reject
/// device-less creation.
#[derive(Clone, Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct AdapterBinding {
    #[serde(default)]
    pub timeout_seconds: Option<u64>,
    #[serde(default)]
    pub image_device: Option<u32>,
    #[serde(default)]
    pub image_timeout_seconds: Option<u64>,
}

impl AdapterBinding {
    pub(crate) fn process_timeout(&self) -> Duration {
        self.timeout_seconds
            .map(Duration::from_secs)
            .unwrap_or(DEFAULT_ADAPTER_TIMEOUT)
    }

    pub(crate) fn image_timeout(&self) -> Duration {
        self.image_timeout_seconds
            .map(Duration::from_secs)
            .unwrap_or(DEFAULT_IMAGE_ADAPTER_TIMEOUT)
    }
}

/// A machine-private image builder declaration. Only a local Docker daemon is
/// supported; the binding shape reserves room for remote builders without
/// changing shareable workspace facts ([[ADR-0005]]).
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct BuilderBinding {
    pub kind: BuilderKind,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub(crate) enum BuilderKind {
    LocalDocker,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ModelWeightBinding {
    #[serde(default)]
    pub locator: Option<String>,
    #[serde(default)]
    pub machine_locators: BTreeMap<String, String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct MachineBinding {
    pub host: String,
    pub devices: Vec<u32>,
    pub ports: Vec<u16>,
    #[serde(default)]
    pub workspace: Option<PathBuf>,
    #[serde(default)]
    pub cache_root: Option<PathBuf>,
    #[serde(default)]
    pub launch: LaunchBinding,
    #[serde(default)]
    pub container: Option<ContainerBinding>,
}

/// Container environment variables Inferlab itself manages: injected at
/// validation launch (HOME, USER, LOGNAME, CUDA_VISIBLE_DEVICES) or owned by
/// the baked image entrypoint (CONDA_PREFIX). One authority for both the
/// pass_env validator and the entrypoint projection, so the two cannot drift
/// ([[RFC-0007:C-IMAGE-BUILD]]).
pub(crate) const MANAGED_CONTAINER_ENV: &[&str] = &[
    "CONDA_PREFIX",
    "CUDA_VISIBLE_DEVICES",
    "HOME",
    "LOGNAME",
    "USER",
];

/// The capabilities the containerized substitution knows how to grant,
/// sized to RDMA-class serving: pinned memory registration (IPC_LOCK),
/// scheduler priorities for communication libraries (SYS_NICE), and
/// cross-process CUDA handle import (SYS_PTRACE). Anything else — and
/// privileged mode categorically — is rejected at load
/// ([[RFC-0007:C-IMAGE-BUILD]]).
pub(crate) const KNOWN_CONTAINER_CAPABILITIES: &[&str] = &["IPC_LOCK", "SYS_NICE", "SYS_PTRACE"];

/// Container launch facts for one machine ([[RFC-0007:C-IMAGE-BUILD]]).
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ContainerBinding {
    /// Environment variable names passed into validation containers by name
    /// reference only (`--env NAME`), so values never enter the launch
    /// command line or the image content. This is the runtime credential
    /// channel. Entries are validated at load: bare names only (no `=`), no
    /// duplicates, and no names Inferlab itself manages in the container.
    #[serde(default)]
    pub pass_env: Vec<String>,
    /// Host device paths granted to every server container on this machine
    /// (`--device`), e.g. `/dev/infiniband` for RDMA KV transfer or
    /// `/dev/gdrdrv` for GPUDirect copies. Operator-declared hardware facts,
    /// never auto-detected; absolute paths only.
    #[serde(default)]
    pub devices: Vec<PathBuf>,
    /// Lift the pinned-memory limit inside server containers
    /// (`--ulimit memlock=-1`); RDMA memory registration needs it.
    #[serde(default)]
    pub memlock_unlimited: bool,
    /// Linux capabilities granted to server containers (`--cap-add`),
    /// validated against [`KNOWN_CONTAINER_CAPABILITIES`]. Privileged mode
    /// is never requested.
    #[serde(default)]
    pub capabilities: Vec<String>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct PlacementBinding {
    #[serde(default)]
    pub machines: Vec<String>,
    #[serde(default)]
    pub roles: BTreeMap<String, PlacementRoleBinding>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(untagged)]
pub(crate) enum PlacementRoleBinding {
    MachinePool(PlacementRoleMachinePoolBinding),
    Direct(RankPlacementBinding),
    MultiRank(MultiRankReplicaPlacementBinding),
    Replicas(PlacementRoleReplicasBinding),
}

impl PlacementRoleBinding {
    pub(crate) const fn uses_machine_pool(&self) -> bool {
        matches!(self, Self::MachinePool(_))
    }

    pub(crate) const fn uses_explicit_replicas(&self) -> bool {
        !self.uses_machine_pool()
    }

    pub(crate) fn machines(&self) -> Option<&[String]> {
        match self {
            Self::MachinePool(binding) => Some(&binding.machines),
            Self::Direct(_) | Self::MultiRank(_) | Self::Replicas(_) => None,
        }
    }

    pub(crate) fn replica_count(&self) -> Option<usize> {
        match self {
            Self::MachinePool(_) => None,
            Self::Direct(_) | Self::MultiRank(_) => Some(1),
            Self::Replicas(binding) => Some(binding.replicas.len()),
        }
    }

    pub(crate) fn ranks_for_replica(
        &self,
        replica_index: usize,
    ) -> Option<&[RankPlacementBinding]> {
        match self {
            Self::MachinePool(_) => None,
            Self::Direct(rank) if replica_index == 0 => Some(std::slice::from_ref(rank)),
            Self::MultiRank(replica) if replica_index == 0 => Some(&replica.ranks),
            Self::Replicas(binding) => binding
                .replicas
                .get(replica_index)
                .map(ReplicaPlacementBinding::ranks),
            Self::Direct(_) | Self::MultiRank(_) => None,
        }
    }

    pub(crate) const fn is_direct_single_replica(&self) -> bool {
        matches!(self, Self::Direct(_))
    }

    pub(crate) fn is_multi_rank_replica(&self, replica_index: usize) -> bool {
        match self {
            Self::MultiRank(_) => replica_index == 0,
            Self::Replicas(binding) => binding
                .replicas
                .get(replica_index)
                .is_some_and(|replica| matches!(replica, ReplicaPlacementBinding::MultiRank(_))),
            Self::MachinePool(_) | Self::Direct(_) => false,
        }
    }
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct PlacementRoleMachinePoolBinding {
    pub machines: Vec<String>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct PlacementRoleReplicasBinding {
    pub replicas: Vec<ReplicaPlacementBinding>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(untagged)]
pub(crate) enum ReplicaPlacementBinding {
    Direct(RankPlacementBinding),
    MultiRank(MultiRankReplicaPlacementBinding),
}

impl ReplicaPlacementBinding {
    pub(crate) fn ranks(&self) -> &[RankPlacementBinding] {
        match self {
            Self::Direct(rank) => std::slice::from_ref(rank),
            Self::MultiRank(replica) => &replica.ranks,
        }
    }
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct MultiRankReplicaPlacementBinding {
    pub ranks: Vec<RankPlacementBinding>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct RankPlacementBinding {
    pub machine: String,
    pub devices: Vec<u32>,
    #[serde(default)]
    pub endpoint_port: Option<u16>,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "kebab-case", deny_unknown_fields)]
pub(crate) enum LaunchBinding {
    #[default]
    Local,
    Ssh {
        target: String,
    },
}

impl LocalBindings {
    /// The device set the default placement assigns to local work, projected
    /// by ad-hoc local execution ([[RFC-0002:C-ADHOC-EXECUTION]]): rank-level
    /// devices plus the full inventories of machines referenced by the
    /// placement or role machine pools. Returns `None` when there is no
    /// default placement, a referenced machine is missing or launches over
    /// SSH, or the resolved set is empty.
    pub(crate) fn default_local_devices(&self) -> Option<Vec<u32>> {
        let placement = self.placements.get(self.default_placement.as_ref()?)?;
        let mut pool_machines: Vec<&str> = placement.machines.iter().map(String::as_str).collect();
        let mut devices: Vec<u32> = Vec::new();
        for role in placement.roles.values() {
            if let Some(pool) = role.machines() {
                pool_machines.extend(pool.iter().map(String::as_str));
            }
            if let Some(replicas) = role.replica_count() {
                for replica in 0..replicas {
                    for rank in role.ranks_for_replica(replica).unwrap_or(&[]) {
                        devices.extend(rank.devices.iter().copied());
                        let machine = self.machines.get(&rank.machine)?;
                        if !matches!(machine.launch, LaunchBinding::Local) {
                            return None;
                        }
                    }
                }
            }
        }
        for id in pool_machines {
            let machine = self.machines.get(id)?;
            if !matches!(machine.launch, LaunchBinding::Local) {
                return None;
            }
            devices.extend(machine.devices.iter().copied());
        }
        devices.sort_unstable();
        devices.dedup();
        (!devices.is_empty()).then_some(devices)
    }
}

pub(super) fn validate_local_bindings(local: &LocalBindings) -> Result<(), InferlabError> {
    if let Some(default_placement) = &local.default_placement {
        require_nonempty("default placement", "local bindings", default_placement)?;
        if !local.placements.contains_key(default_placement) {
            return invalid(format!("unknown default placement {default_placement:?}"));
        }
    }
    if local.adapter.timeout_seconds == Some(0) {
        return invalid(
            "adapter timeout_seconds must be positive; omit it for the default deadline".to_owned(),
        );
    }
    if local.adapter.image_timeout_seconds == Some(0) {
        return invalid(
            "adapter image_timeout_seconds must be positive; omit it for the default deadline"
                .to_owned(),
        );
    }
    for id in local.builders.keys() {
        require_id("builder binding", id)?;
    }
    for (id, weight) in &local.model_weights {
        require_id("model weight binding", id)?;
        if let Some(locator) = &weight.locator {
            require_nonempty("model weight locator", id, locator)?;
        }
        if weight.locator.is_none() && weight.machine_locators.is_empty() {
            return invalid(format!(
                "model weight binding {id:?} must declare locator, machine_locators, or both"
            ));
        }
        for (machine, locator) in &weight.machine_locators {
            if !local.machines.contains_key(machine) {
                return invalid(format!(
                    "model weight binding {id:?} references unknown machine {machine:?}"
                ));
            }
            require_nonempty("machine model weight locator", machine, locator)?;
        }
    }
    for (id, machine) in &local.machines {
        require_id("machine binding", id)?;
        require_nonempty("machine host", id, &machine.host)?;
        let unique: BTreeSet<_> = machine.devices.iter().collect();
        if unique.len() != machine.devices.len() {
            return invalid(format!("machine binding {id:?} contains duplicate devices"));
        }
        let mut ports = BTreeSet::new();
        for port in &machine.ports {
            if *port == 0 {
                return invalid(format!("machine binding {id:?} port must be nonzero"));
            }
            if !ports.insert(*port) {
                return invalid(format!("machine binding {id:?} contains duplicate ports"));
            }
        }
        if let Some(container) = &machine.container {
            let mut seen = BTreeSet::new();
            for name in &container.pass_env {
                // A POSIX shell identifier, exactly: the launch scripts
                // splice these names into shell parameter references, where
                // anything richer (a bash array subscript, for one) carries
                // expansion side effects — and a non-identifier name could
                // not be referenced by ${NAME} at all
                // ([[RFC-0007:C-IMAGE-BUILD]]).
                let identifier = !name.is_empty()
                    && !name.as_bytes()[0].is_ascii_digit()
                    && name
                        .bytes()
                        .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_');
                if !identifier {
                    return invalid(format!(
                        "machine binding {id:?} pass_env entry {name:?} must be a POSIX \
                         shell identifier; values are never declared here \
                         (name-reference-only pass-through)"
                    ));
                }
                if MANAGED_CONTAINER_ENV.contains(&name.as_str()) {
                    return invalid(format!(
                        "machine binding {id:?} pass_env entry {name:?} collides with a \
                         container variable Inferlab manages"
                    ));
                }
                if !seen.insert(name) {
                    return invalid(format!(
                        "machine binding {id:?} pass_env contains duplicate entry {name:?}"
                    ));
                }
            }
            let mut devices = BTreeSet::new();
            for device in &container.devices {
                if !device.is_absolute() {
                    return invalid(format!(
                        "machine binding {id:?} container device {} must be an absolute \
                         host path",
                        device.display()
                    ));
                }
                if !devices.insert(device) {
                    return invalid(format!(
                        "machine binding {id:?} contains duplicate container device {}",
                        device.display()
                    ));
                }
            }
            let mut capabilities = BTreeSet::new();
            for capability in &container.capabilities {
                if !KNOWN_CONTAINER_CAPABILITIES.contains(&capability.as_str()) {
                    return invalid(format!(
                        "machine binding {id:?} container capability {capability:?} is not \
                         a capability Inferlab grants (known: {})",
                        KNOWN_CONTAINER_CAPABILITIES.join(", ")
                    ));
                }
                if !capabilities.insert(capability) {
                    return invalid(format!(
                        "machine binding {id:?} contains duplicate container capability \
                         {capability:?}"
                    ));
                }
            }
        }
        if machine
            .cache_root
            .as_ref()
            .is_some_and(|path| !path.is_absolute())
        {
            return invalid(format!(
                "machine binding {id:?} cache_root must be an absolute path"
            ));
        }
        match &machine.launch {
            LaunchBinding::Local if machine.workspace.is_some() => {
                return invalid(format!(
                    "local machine binding {id:?} uses the controller workspace and must not set workspace"
                ));
            }
            LaunchBinding::Local => {}
            LaunchBinding::Ssh { target } => {
                require_nonempty("SSH target", id, target)?;
                if machine.workspace.is_none() {
                    return invalid(format!(
                        "SSH machine binding {id:?} requires an execution-visible workspace"
                    ));
                }
            }
        }
    }
    for (id, placement) in &local.placements {
        require_id("placement binding", id)?;
        let uses_role_pools = placement
            .roles
            .values()
            .any(PlacementRoleBinding::uses_machine_pool);
        let uses_explicit_replicas = placement
            .roles
            .values()
            .any(PlacementRoleBinding::uses_explicit_replicas);
        let uses_pools = !placement.machines.is_empty() || uses_role_pools;
        if uses_pools == uses_explicit_replicas {
            return invalid(format!(
                "placement binding {id:?} must use exactly one of machine pools or explicit replicas"
            ));
        }
        let mut machines = BTreeSet::new();
        for machine in &placement.machines {
            if !machines.insert(machine) {
                return invalid(format!(
                    "placement binding {id:?} contains duplicate machine {machine:?}"
                ));
            }
            if !local.machines.contains_key(machine) {
                return invalid(format!(
                    "placement binding {id:?} references unknown machine {machine:?}"
                ));
            }
        }
        let mut explicit_devices = BTreeSet::new();
        let mut explicit_ports = BTreeSet::new();
        for (role, role_placement) in &placement.roles {
            require_id("placement role", role)?;
            if let Some(role_machines) = role_placement.machines() {
                if role_machines.is_empty() {
                    return invalid(format!(
                        "placement binding {id:?} role {role:?} machine pool must not be empty"
                    ));
                }
                let mut role_seen = BTreeSet::new();
                for machine in role_machines {
                    if !role_seen.insert(machine) {
                        return invalid(format!(
                            "placement binding {id:?} role {role:?} contains duplicate machine {machine:?}"
                        ));
                    }
                    if !local.machines.contains_key(machine) {
                        return invalid(format!(
                            "placement binding {id:?} role {role:?} references unknown machine {machine:?}"
                        ));
                    }
                }
                continue;
            }
            if !matches!(role.as_str(), "serve" | "prefill" | "decode" | "gateway") {
                return invalid(format!(
                    "placement binding {id:?} contains non-canonical role {role:?}"
                ));
            }
            if role == "gateway" && !role_placement.is_direct_single_replica() {
                return invalid(format!(
                    "placement binding {id:?} Gateway must contain exactly one direct replica"
                ));
            }
            let replica_count =
                role_placement
                    .replica_count()
                    .ok_or_else(|| InferlabError::InvalidConfig {
                        message: format!(
                            "placement binding {id:?} role {role:?} does not define replicas"
                        ),
                    })?;
            if matches!(role_placement, PlacementRoleBinding::Replicas(_)) && replica_count < 2 {
                return invalid(format!(
                    "placement binding {id:?} role {role:?} replicas form requires at least two replicas"
                ));
            }

            for replica_index in 0..replica_count {
                let replica_index =
                    u32::try_from(replica_index).map_err(|_| InferlabError::InvalidConfig {
                        message: format!(
                            "placement binding {id:?} role {role:?} has too many replicas"
                        ),
                    })?;
                let ranks = role_placement
                    .ranks_for_replica(replica_index as usize)
                    .ok_or_else(|| InferlabError::InvalidConfig {
                        message: format!(
                            "placement binding {id:?} role {role:?} replica {replica_index} is missing"
                        ),
                    })?;
                if role_placement.is_multi_rank_replica(replica_index as usize) && ranks.len() < 2 {
                    return invalid(format!(
                        "placement binding {id:?} role {role:?} replica {replica_index} multi-rank form requires at least two ranks"
                    ));
                }
                for (rank_index, rank) in ranks.iter().enumerate() {
                    let rank_index =
                        u32::try_from(rank_index).map_err(|_| InferlabError::InvalidConfig {
                            message: format!(
                                "placement binding {id:?} role {role:?} replica {replica_index} has too many ranks"
                            ),
                        })?;
                    let machine = local.machines.get(&rank.machine).ok_or_else(|| {
                        InferlabError::InvalidConfig {
                            message: format!(
                                "placement binding {id:?} rank ({role:?}, {replica_index}, {rank_index}) references unknown machine {:?}",
                                rank.machine
                            ),
                        }
                    })?;
                    if role == "gateway" && !rank.devices.is_empty() {
                        return invalid(format!(
                            "placement binding {id:?} Gateway must use no devices"
                        ));
                    }
                    if role != "gateway" && rank.devices.is_empty() {
                        return invalid(format!(
                            "placement binding {id:?} rank ({role:?}, {replica_index}, {rank_index}) must bind at least one device"
                        ));
                    }
                    let mut rank_devices = BTreeSet::new();
                    for device in &rank.devices {
                        if !rank_devices.insert(device) {
                            return invalid(format!(
                                "placement binding {id:?} rank ({role:?}, {replica_index}, {rank_index}) contains duplicate device {device}"
                            ));
                        }
                        if !machine.devices.contains(device) {
                            return invalid(format!(
                                "placement binding {id:?} references unavailable device {}:{device}",
                                rank.machine
                            ));
                        }
                        if !explicit_devices.insert((&rank.machine, *device)) {
                            return invalid(format!(
                                "placement binding {id:?} assigns device {}:{device} more than once",
                                rank.machine
                            ));
                        }
                    }
                    if let Some(port) = rank.endpoint_port {
                        if !machine.ports.contains(&port) {
                            return invalid(format!(
                                "placement binding {id:?} rank ({role:?}, {replica_index}, {rank_index}) endpoint_port {port} is not in machine {:?}'s port pool",
                                rank.machine
                            ));
                        }
                        if !explicit_ports.insert((&rank.machine, port)) {
                            return invalid(format!(
                                "placement binding {id:?} assigns endpoint port {}:{port} more than once",
                                rank.machine
                            ));
                        }
                    }
                }
            }
        }
    }
    Ok(())
}
