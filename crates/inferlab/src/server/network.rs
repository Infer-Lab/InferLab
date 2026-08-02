use crate::execution::{
    ActiveRdmaInterfacePlan, NetworkMachinePlan, NetworkPlan, NetworkSelectionReason, ProcessPlan,
};
use inferlab_runtime::plan::LaunchPlan;
use std::collections::{BTreeMap, BTreeSet};
use std::net::Ipv4Addr;
use std::process::{Command, Output};
use thiserror::Error;

const NETWORK_MARKER: &str = "INFERLAB_NETWORK\t";
const EXCLUDED_INTERFACE_PREFIXES: [&str; 4] = ["br-", "docker", "veth", "virbr"];
const PROBE_SCRIPT: &str = r#"set -eu
route_iface=$(ip route get 8.8.8.8 2>/dev/null | sed -n 's/.* dev \([^ ]*\).*/\1/p' | head -n1 || true)
printf 'INFERLAB_NETWORK\tROUTE\t%s\n' "$route_iface"
ip -o -4 addr show scope global up | awk '{printf "INFERLAB_NETWORK\tADDR\t%s\t%s\n", $2, $4}'
if command -v ibdev2netdev >/dev/null 2>&1; then
    ibdev2netdev | awk '/\(Up\)/ {printf "INFERLAB_NETWORK\tRDMA\t%s\t%s\n", $5, $1}'
fi"#;

#[derive(Debug, Error)]
pub(super) enum NetworkResolutionError {
    #[error("failed to launch network probe for local machine {machine:?}: {source}")]
    LocalLaunch {
        machine: String,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to launch network probe for machine {machine:?}: {source}")]
    Ssh {
        machine: String,
        #[source]
        source: inferlab_runtime::ssh::SshError,
    },
    #[error("network probe for machine {machine:?} exited with {status}: {stderr}")]
    Exit {
        machine: String,
        status: std::process::ExitStatus,
        stderr: String,
    },
    #[error("network probe for machine {machine:?} returned non-UTF-8 output: {source}")]
    NonUtf8 {
        machine: String,
        #[source]
        source: std::string::FromUtf8Error,
    },
    #[error("invalid network probe for {machine:?}: {source}")]
    InvalidEvidence {
        machine: String,
        #[source]
        source: NetworkEvidenceError,
    },
    #[error(
        "no common routable communication interface across machines [{machines}]; verify that every machine has a common global IPv4 interface and that RDMA links are up"
    )]
    NoCommonInterface { machines: String },
}

#[derive(Debug, Error)]
pub(super) enum NetworkEvidenceError {
    #[error("{kind} evidence has no {field}")]
    MissingField {
        kind: &'static str,
        field: &'static str,
    },
    #[error("unknown evidence kind {kind:?}")]
    UnknownKind { kind: String },
    #[error("empty evidence line")]
    EmptyLine,
    #[error("probe returned no route evidence")]
    MissingRoute,
}

pub(super) fn resolve(
    processes: &[ProcessPlan],
) -> Result<Option<NetworkPlan>, NetworkResolutionError> {
    let mut seen = BTreeSet::new();
    let machines = processes
        .iter()
        .filter(|process| seen.insert(process.machine.clone()))
        .collect::<Vec<_>>();
    if machines.len() <= 1 {
        return Ok(None);
    }

    let probes = machines
        .into_iter()
        .map(|process| probe_machine(process).map(|probe| (process.machine.clone(), probe)))
        .collect::<Result<Vec<_>, _>>()?;
    let (selected_interface, reason) = select_interface(&probes).ok_or_else(|| {
        let machine_ids = probes
            .iter()
            .map(|(machine, _)| machine.as_str())
            .collect::<Vec<_>>()
            .join(", ");
        NetworkResolutionError::NoCommonInterface {
            machines: machine_ids,
        }
    })?;

    Ok(Some(NetworkPlan {
        selected_interface,
        reason,
        machines: probes.into_iter().collect(),
    }))
}

fn probe_machine(process: &ProcessPlan) -> Result<NetworkMachinePlan, NetworkResolutionError> {
    let output = match &process.launch {
        LaunchPlan::Local => Command::new("bash")
            .args(["-c", PROBE_SCRIPT])
            .output()
            .map_err(|source| NetworkResolutionError::LocalLaunch {
                machine: process.machine.clone(),
                source,
            })?,
        LaunchPlan::Ssh { target } => inferlab_runtime::ssh::ssh_output(target, PROBE_SCRIPT)
            .map_err(|source| NetworkResolutionError::Ssh {
                machine: process.machine.clone(),
                source,
            })?,
    };
    parse_output(&process.machine, output)
}

fn parse_output(
    machine: &str,
    output: Output,
) -> Result<NetworkMachinePlan, NetworkResolutionError> {
    if !output.status.success() {
        return Err(NetworkResolutionError::Exit {
            machine: machine.to_owned(),
            status: output.status,
            stderr: String::from_utf8_lossy(&output.stderr).trim().to_owned(),
        });
    }
    let stdout =
        String::from_utf8(output.stdout).map_err(|source| NetworkResolutionError::NonUtf8 {
            machine: machine.to_owned(),
            source,
        })?;
    parse_probe(&stdout).map_err(|source| NetworkResolutionError::InvalidEvidence {
        machine: machine.to_owned(),
        source,
    })
}

fn parse_probe(output: &str) -> Result<NetworkMachinePlan, NetworkEvidenceError> {
    let mut saw_route = false;
    let mut default_route_interface = None;
    let mut addresses = BTreeMap::<String, Vec<String>>::new();
    let mut address_order = Vec::new();
    let mut active_rdma_interfaces = Vec::new();

    for payload in output
        .lines()
        .filter_map(|line| line.strip_prefix(NETWORK_MARKER))
    {
        let mut fields = payload.split('\t');
        match fields.next() {
            Some("ROUTE") => {
                saw_route = true;
                default_route_interface = fields
                    .next()
                    .filter(|interface| !interface.is_empty())
                    .map(str::to_owned);
            }
            Some("ADDR") => {
                let interface = fields
                    .next()
                    .filter(|interface| !interface.is_empty())
                    .ok_or(NetworkEvidenceError::MissingField {
                        kind: "address",
                        field: "interface",
                    })?;
                let address = fields.next().filter(|address| !address.is_empty()).ok_or(
                    NetworkEvidenceError::MissingField {
                        kind: "address",
                        field: "address",
                    },
                )?;
                if !addresses.contains_key(interface) {
                    address_order.push(interface.to_owned());
                }
                addresses
                    .entry(interface.to_owned())
                    .or_default()
                    .push(address.to_owned());
            }
            Some("RDMA") => {
                let interface = fields
                    .next()
                    .filter(|interface| !interface.is_empty())
                    .ok_or(NetworkEvidenceError::MissingField {
                        kind: "RDMA",
                        field: "interface",
                    })?;
                let device = fields.next().filter(|device| !device.is_empty()).ok_or(
                    NetworkEvidenceError::MissingField {
                        kind: "RDMA",
                        field: "device",
                    },
                )?;
                active_rdma_interfaces.push(ActiveRdmaInterfacePlan {
                    interface: interface.to_owned(),
                    device: device.to_owned(),
                });
            }
            Some(kind) => {
                return Err(NetworkEvidenceError::UnknownKind {
                    kind: kind.to_owned(),
                });
            }
            None => return Err(NetworkEvidenceError::EmptyLine),
        }
    }
    if !saw_route {
        return Err(NetworkEvidenceError::MissingRoute);
    }

    let candidates = address_order
        .into_iter()
        .filter(|interface| {
            is_routable_interface(
                interface,
                addresses.get(interface).map_or(&[], Vec::as_slice),
            )
        })
        .collect();
    Ok(NetworkMachinePlan {
        default_route_interface,
        addresses,
        active_rdma_interfaces,
        candidates,
    })
}

fn is_routable_interface(interface: &str, addresses: &[String]) -> bool {
    if interface == "lo"
        || EXCLUDED_INTERFACE_PREFIXES
            .iter()
            .any(|prefix| interface.starts_with(prefix))
    {
        return false;
    }
    let parsed = addresses
        .iter()
        .filter_map(|address| address.split('/').next()?.parse::<Ipv4Addr>().ok())
        .collect::<Vec<_>>();
    !parsed.is_empty()
        && parsed.iter().all(|address| {
            !address.is_loopback()
                && !address.is_link_local()
                && !address.is_unspecified()
                && !address.is_multicast()
                && *address != Ipv4Addr::BROADCAST
        })
}

fn select_interface(
    probes: &[(String, NetworkMachinePlan)],
) -> Option<(String, NetworkSelectionReason)> {
    let route_rdma = probes
        .iter()
        .map(|(_, probe)| {
            probe
                .default_route_interface
                .iter()
                .filter(|interface| probe.candidates.contains(interface))
                .filter(|interface| {
                    probe
                        .active_rdma_interfaces
                        .iter()
                        .any(|rdma| &rdma.interface == *interface)
                })
                .cloned()
                .collect()
        })
        .collect::<Vec<Vec<String>>>();
    if let Some(interface) = first_common_in_order(&route_rdma) {
        return Some((interface, NetworkSelectionReason::RdmaDefaultRoute));
    }

    let rdma = probes
        .iter()
        .map(|(_, probe)| {
            let mut seen = BTreeSet::new();
            probe
                .active_rdma_interfaces
                .iter()
                .map(|rdma| rdma.interface.clone())
                .filter(|interface| probe.candidates.contains(interface))
                .filter(|interface| seen.insert(interface.clone()))
                .collect()
        })
        .collect::<Vec<Vec<String>>>();
    if let Some(interface) = first_common_in_order(&rdma) {
        return Some((interface, NetworkSelectionReason::Rdma));
    }

    let routes = probes
        .iter()
        .map(|(_, probe)| {
            probe
                .default_route_interface
                .iter()
                .filter(|interface| probe.candidates.contains(interface))
                .cloned()
                .collect()
        })
        .collect::<Vec<Vec<String>>>();
    if let Some(interface) = first_common_in_order(&routes) {
        return Some((interface, NetworkSelectionReason::DefaultRoute));
    }

    let candidates = probes
        .iter()
        .map(|(_, probe)| probe.candidates.clone())
        .collect::<Vec<_>>();
    first_common_in_order(&candidates)
        .map(|interface| (interface, NetworkSelectionReason::Routable))
}

fn first_common_in_order(values_by_machine: &[Vec<String>]) -> Option<String> {
    let (first, rest) = values_by_machine.split_first()?;
    first
        .iter()
        .find(|candidate| rest.iter().all(|values| values.contains(candidate)))
        .cloned()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn probe(
        route: &str,
        addresses: &[(&str, &str)],
        rdma: &[(&str, &str)],
    ) -> Result<NetworkMachinePlan, String> {
        let mut text = format!("{NETWORK_MARKER}ROUTE\t{route}\n");
        for (interface, address) in addresses {
            text.push_str(&format!("{NETWORK_MARKER}ADDR\t{interface}\t{address}\n"));
        }
        for (interface, device) in rdma {
            text.push_str(&format!("{NETWORK_MARKER}RDMA\t{interface}\t{device}\n"));
        }
        parse_probe(&text).map_err(|error| error.to_string())
    }

    #[test]
    fn common_rdma_interface_wins_over_a_link_local_default_route() -> Result<(), String> {
        let probes = vec![
            (
                "node-a".to_owned(),
                probe(
                    "enx-link-local",
                    &[
                        ("enx-link-local", "169.254.3.1/24"),
                        ("ens-rdma", "192.0.2.10/24"),
                    ],
                    &[("ens-rdma", "mlx5_0")],
                )?,
            ),
            (
                "node-b".to_owned(),
                probe(
                    "enx-link-local",
                    &[
                        ("enx-link-local", "169.254.3.1/24"),
                        ("ens-rdma", "192.0.2.11/24"),
                    ],
                    &[("ens-rdma", "mlx5_0")],
                )?,
            ),
        ];

        let Some((interface, reason)) = select_interface(&probes) else {
            return Err("expected a common interface".to_owned());
        };
        assert_eq!(interface, "ens-rdma");
        assert!(matches!(reason, NetworkSelectionReason::Rdma));
        Ok(())
    }

    #[test]
    fn selection_falls_back_to_a_common_routable_interface() -> Result<(), String> {
        let probes = vec![
            (
                "node-a".to_owned(),
                probe(
                    "eth-a",
                    &[("eth-a", "192.0.2.10/24"), ("fabric", "198.51.100.10/24")],
                    &[],
                )?,
            ),
            (
                "node-b".to_owned(),
                probe(
                    "eth-b",
                    &[("eth-b", "192.0.2.11/24"), ("fabric", "198.51.100.11/24")],
                    &[],
                )?,
            ),
        ];

        let Some((interface, reason)) = select_interface(&probes) else {
            return Err("expected a common interface".to_owned());
        };
        assert_eq!(interface, "fabric");
        assert!(matches!(reason, NetworkSelectionReason::Routable));
        Ok(())
    }

    #[test]
    fn link_local_only_probes_have_no_candidate() -> Result<(), String> {
        let probes = vec![
            (
                "node-a".to_owned(),
                probe(
                    "enx-link-local",
                    &[("enx-link-local", "169.254.3.1/24")],
                    &[],
                )?,
            ),
            (
                "node-b".to_owned(),
                probe(
                    "enx-link-local",
                    &[("enx-link-local", "169.254.3.1/24")],
                    &[],
                )?,
            ),
        ];

        assert!(select_interface(&probes).is_none());
        Ok(())
    }
}
