//! Machine-pool occupancy observed before allocation
//! ([[RFC-0002:C-LOCAL-PLACEMENT]], [[ADR-0065]]): a pool port is unavailable
//! when a TCP socket already listens on it on the machine's launch path, a
//! pool device when it reports a compute application, and either when a
//! starting or running record of the workspace allocated it on that launch
//! path. Each launch path is observed once per resolution.

use crate::error::InferlabError;
use crate::workspace::MachineBinding;
use inferlab_runtime::operation_bound::OperationBound;
use inferlab_runtime::plan::LaunchPlan;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;
use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::process::Command;

const LISTEN_MARKER: &str = "INFERLAB_LISTEN\t";
const ALLOCATION_LOCK_FILE: &str = "allocation.lock";
const ALLOCATION_HOLDER_FILE: &str = "allocation.json";

/// What made one pool port or device unavailable when resolution observed it.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub(crate) enum Holder {
    ListeningSocket {
        /// The owning process, when the launch path shows it to the
        /// launching user.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        process: Option<String>,
    },
    ComputeApplication {
        occupant: String,
    },
    Record {
        record_id: String,
        record_kind: RecordKind,
        status: String,
    },
}

/// The kind of a holding record, which decides the command that stops it.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum RecordKind {
    Server,
    Recipe,
}

impl std::fmt::Display for Holder {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ListeningSocket { process: None } => formatter.write_str("a listening socket"),
            Self::ListeningSocket {
                process: Some(process),
            } => write!(formatter, "a socket {process} listens on"),
            Self::ComputeApplication { occupant } => formatter.write_str(occupant),
            Self::Record {
                record_id,
                record_kind,
                status,
            } => {
                let (kind, command) = match record_kind {
                    RecordKind::Server => ("server", "serve stop"),
                    RecordKind::Recipe => ("recipe", "recipe stop"),
                };
                write!(
                    formatter,
                    "{kind} record {record_id} ({status}); release it with `inferlab {command} {record_id}`"
                )
            }
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct UnavailablePort {
    pub port: u16,
    pub holder: Holder,
}

/// Each used machine's unavailable pool ports, keyed by machine identifier.
pub(crate) type UnavailablePorts = BTreeMap<String, Vec<UnavailablePort>>;

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct UnavailableDevice {
    pub device: u32,
    pub holder: Holder,
}

/// Each used machine's unavailable pool devices, keyed by machine identifier.
pub(crate) type UnavailableDevices = BTreeMap<String, Vec<UnavailableDevice>>;

/// What resolution observed held on the machines that received processes.
#[derive(Debug, Default)]
pub(crate) struct Unavailable {
    pub ports: UnavailablePorts,
    pub devices: UnavailableDevices,
}

/// A port or device a record allocated on one launch path.
#[derive(Clone, Copy, Eq, PartialEq)]
enum Resource {
    Port(u16),
    Device(u32),
}

/// The workspace allocation lock a launching workflow holds from port
/// observation until its server record exists, so two launches in one
/// workspace cannot allocate the same port; dropping it releases the lock
/// once a record carrying the allocation exists.
#[derive(Debug)]
pub(crate) struct AllocationLock {
    _file: File,
}

pub(crate) fn acquire_allocation_lock(
    root: &Path,
    progress: &crate::progress::Progress,
) -> Result<AllocationLock, InferlabError> {
    let runtime = root.join(crate::record::RUNTIME_DIR);
    let file = crate::controller::lock_waiting(
        &runtime.join(ALLOCATION_LOCK_FILE),
        &runtime.join(ALLOCATION_HOLDER_FILE),
        "the workspace allocation lock",
        progress,
    )?;
    Ok(AllocationLock { _file: file })
}

/// Observed occupancy, cached per launch path for one resolution.
pub(super) struct Occupancy<'a> {
    root: &'a Path,
    listening: BTreeMap<String, BTreeMap<u16, Option<String>>>,
    occupied: BTreeMap<String, BTreeMap<u32, String>>,
    reserved: Option<Vec<(Holder, LaunchPlan, Resource)>>,
}

impl<'a> Occupancy<'a> {
    pub(super) const fn new(root: &'a Path) -> Self {
        Self {
            root,
            listening: BTreeMap::new(),
            occupied: BTreeMap::new(),
            reserved: None,
        }
    }

    /// The machine's pool ports that are unavailable on its launch path.
    pub(super) fn unavailable(
        &mut self,
        machine_id: &str,
        machine: &MachineBinding,
    ) -> Result<BTreeMap<u16, Holder>, InferlabError> {
        let launch = super::allocation::launch_plan(&machine.launch);
        let key = launch_key(&launch);
        if !self.listening.contains_key(&key) {
            let ports = listening_ports(machine_id, &launch)?;
            self.listening.insert(key.clone(), ports);
        }
        if self.reserved.is_none() {
            self.reserved = Some(reserved_resources(self.root)?);
        }
        let listening = self.listening.get(&key);
        let mut unavailable = BTreeMap::new();
        for &port in &machine.ports {
            if let Some(process) = listening.and_then(|ports| ports.get(&port)) {
                unavailable.insert(
                    port,
                    Holder::ListeningSocket {
                        process: process.clone(),
                    },
                );
            } else if let Some(holder) = self.reserved_by(&key, Resource::Port(port)) {
                unavailable.insert(port, holder);
            }
        }
        Ok(unavailable)
    }

    /// The machine's pool devices that are unavailable on its launch path:
    /// those reporting a compute application to the same probe the
    /// launch-time device check runs, and those a live record allocated.
    pub(super) fn unavailable_devices(
        &mut self,
        machine_id: &str,
        machine: &MachineBinding,
    ) -> Result<BTreeMap<u32, Holder>, InferlabError> {
        if machine.devices.is_empty() {
            return Ok(BTreeMap::new());
        }
        let launch = super::allocation::launch_plan(&machine.launch);
        let key = launch_key(&launch);
        if !self.occupied.contains_key(&key) {
            let occupied = crate::server::occupied_devices(&launch, machine_id, &machine.devices)
                .map_err(|message| InferlabError::PortObservation {
                machine: machine_id.to_owned(),
                message,
            })?;
            self.occupied.insert(key.clone(), occupied);
        }
        if self.reserved.is_none() {
            self.reserved = Some(reserved_resources(self.root)?);
        }
        let mut unavailable = BTreeMap::new();
        for &device in &machine.devices {
            if let Some(occupant) = self.occupied.get(&key).and_then(|held| held.get(&device)) {
                unavailable.insert(
                    device,
                    Holder::ComputeApplication {
                        occupant: occupant.clone(),
                    },
                );
            } else if let Some(holder) = self.reserved_by(&key, Resource::Device(device)) {
                unavailable.insert(device, holder);
            }
        }
        Ok(unavailable)
    }

    fn reserved_by(&self, key: &str, resource: Resource) -> Option<Holder> {
        self.reserved
            .iter()
            .flatten()
            .find(|(_, launch, reserved)| *reserved == resource && launch_key(launch) == key)
            .map(|(holder, _, _)| holder.clone())
    }
}

fn launch_key(launch: &LaunchPlan) -> String {
    match launch {
        LaunchPlan::Local => "local".to_owned(),
        LaunchPlan::Ssh { target } => format!("ssh:{target}"),
    }
}

/// The launch path's listening TCP ports with their owning process when the
/// launching user may see it, read with `ss -Hltnp` through the same launch
/// path serving processes use. The marker prefix keeps SSH login banners out
/// of the parsed rows, and the command substitution keeps the tool's exit
/// status authoritative.
fn listening_ports(
    machine_id: &str,
    launch: &LaunchPlan,
) -> Result<BTreeMap<u16, Option<String>>, InferlabError> {
    let script = format!(
        "set -eu; printf '\\n'; out=$(ss -Hltnp); printf '%s\\n' \"$out\" | sed 's/^/{LISTEN_MARKER}/'"
    );
    let failure = |message: String| InferlabError::PortObservation {
        machine: machine_id.to_owned(),
        message,
    };
    let output = match launch {
        LaunchPlan::Local => Command::new("sh")
            .args(["-c", &script])
            .output()
            .map_err(|error| failure(format!("failed to run ss: {error}")))?,
        LaunchPlan::Ssh { target } => {
            inferlab_runtime::ssh::ssh_output(target, &script, &OperationBound::unbounded())
                .map_err(|error| failure(error.to_string()))?
        }
    };
    if !output.status.success() {
        return Err(failure(format!(
            "ss -Hltnp exited with {}: {}",
            output.status,
            String::from_utf8_lossy(&output.stderr).trim()
        )));
    }
    parse_listening_ports(&String::from_utf8_lossy(&output.stdout)).map_err(failure)
}

/// Each `ss -Hltnp` row's fourth column is the local address, whose port
/// follows the last colon (`0.0.0.0:22`, `[::]:22`, `*:22`); a trailing
/// `users:(("name",pid=N,fd=M))` names the owning process when visible.
fn parse_listening_ports(stdout: &str) -> Result<BTreeMap<u16, Option<String>>, String> {
    let mut ports = BTreeMap::new();
    for line in stdout.lines() {
        let Some(row) = line.strip_prefix(LISTEN_MARKER) else {
            continue;
        };
        if row.trim().is_empty() {
            continue;
        }
        let port = row
            .split_whitespace()
            .nth(3)
            .and_then(|address| address.rsplit_once(':'))
            .and_then(|(_, port)| port.parse::<u16>().ok())
            .ok_or_else(|| format!("ss -Hltnp returned an unparsable row {row:?}"))?;
        let process = row.split_once("users:((\"").and_then(|(_, users)| {
            let (name, rest) = users.split_once('"')?;
            let pid = rest
                .split_once("pid=")?
                .1
                .split(|c: char| !c.is_ascii_digit())
                .next()?;
            Some(format!("pid {pid} ({name})"))
        });
        let entry = ports.entry(port).or_insert(None);
        if entry.is_none() {
            *entry = process;
        }
    }
    Ok(ports)
}

/// Resources the workspace's records hold for the window observation cannot
/// see, with the launch path each was allocated on: a current-schema server
/// record holds its ports while starting or running and its devices while
/// starting; a current-schema recipe record holds both while running before
/// its server record exists. Earlier-schema records hold nothing. Records are
/// read as plain JSON; one that cannot be read holds nothing.
fn reserved_resources(root: &Path) -> Result<Vec<(Holder, LaunchPlan, Resource)>, InferlabError> {
    let records = root.join(crate::record::RECORDS_DIR);
    let entries = match fs::read_dir(&records) {
        Ok(entries) => entries,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(source) => {
            return Err(InferlabError::RecordIo {
                path: records,
                source,
            });
        }
    };
    let mut entries = entries.flatten().collect::<Vec<_>>();
    entries.sort_by_key(std::fs::DirEntry::file_name);
    let mut reserved = Vec::new();
    for entry in entries {
        let path: PathBuf = entry.path().join("record.json");
        let Ok(text) = fs::read_to_string(&path) else {
            continue;
        };
        let Ok(record) = serde_json::from_str::<Value>(&text) else {
            continue;
        };
        let (Some(record_id), Some(status), Some(schema)) = (
            record["id"].as_str(),
            record["status"].as_str(),
            record["schema_version"].as_u64(),
        ) else {
            continue;
        };
        let (kind, holds_ports, holds_devices) = if record.get("process_evidence").is_some() {
            if schema != u64::from(crate::server::ServerRecord::SCHEMA_VERSION) {
                continue;
            }
            match status {
                "starting" => (RecordKind::Server, true, true),
                "running" => (RecordKind::Server, true, false),
                _ => continue,
            }
        } else if let Some(server_id) = record["server"]["id"].as_str() {
            let server_started = records.join(server_id).join("record.json").exists();
            if schema != u64::from(crate::recipe::RecipeRecord::SCHEMA_VERSION)
                || status != "running"
                || server_started
            {
                continue;
            }
            (RecordKind::Recipe, true, true)
        } else {
            continue;
        };
        let holder = Holder::Record {
            record_id: record_id.to_owned(),
            record_kind: kind,
            status: status.to_owned(),
        };
        collect_process_resources(&record["resolved"]["server"], &mut |launch, resource| {
            let held = match resource {
                Resource::Port(_) => holds_ports,
                Resource::Device(_) => holds_devices,
            };
            if held {
                reserved.push((holder.clone(), launch, resource));
            }
        });
    }
    Ok(reserved)
}

/// Every process plan carries its `launch` path beside its `devices`,
/// `endpoint`, and named `ports`; the walk finds them wherever the topology
/// nests them.
fn collect_process_resources(value: &Value, found: &mut impl FnMut(LaunchPlan, Resource)) {
    match value {
        Value::Object(object) => {
            if let (Some(launch), Some(ports)) = (object.get("launch"), object.get("ports"))
                && let Ok(launch) = serde_json::from_value::<LaunchPlan>(launch.clone())
            {
                let port_of = |value: &Value| {
                    value["port"]
                        .as_u64()
                        .and_then(|port| u16::try_from(port).ok())
                };
                if let Some(port) = object.get("endpoint").and_then(port_of) {
                    found(launch.clone(), Resource::Port(port));
                }
                if let Some(ports) = ports.as_object() {
                    for port in ports.values().filter_map(port_of) {
                        found(launch.clone(), Resource::Port(port));
                    }
                }
                for device in object
                    .get("devices")
                    .and_then(Value::as_array)
                    .into_iter()
                    .flatten()
                    .filter_map(Value::as_u64)
                    .filter_map(|device| u32::try_from(device).ok())
                {
                    found(launch.clone(), Resource::Device(device));
                }
                return;
            }
            for child in object.values() {
                collect_process_resources(child, found);
            }
        }
        Value::Array(items) => {
            for item in items {
                collect_process_resources(item, found);
            }
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::{LISTEN_MARKER, parse_listening_ports};

    #[test]
    fn listening_rows_parse_ipv4_ipv6_and_wildcard_addresses() -> Result<(), String> {
        let stdout = format!(
            "login banner\n\
             {LISTEN_MARKER}LISTEN 0      4096       0.0.0.0:8000 0.0.0.0:*\n\
             {LISTEN_MARKER}LISTEN 0      10         [fe80::1]%ens5:53 [::]:*\n\
             {LISTEN_MARKER}LISTEN 0      128        *:29501 *:* users:((\"fixture-engine\",pid=4242,fd=7))\n\
             {LISTEN_MARKER}\n"
        );
        let ports = parse_listening_ports(&stdout)?;
        assert_eq!(ports.keys().copied().collect::<Vec<_>>(), [53, 8000, 29501]);
        assert_eq!(ports[&29501], Some("pid 4242 (fixture-engine)".to_owned()));
        assert_eq!(ports[&8000], None);
        Ok(())
    }
}
