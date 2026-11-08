//! One lookup of each used machine's `host`, whose address every rendered
//! listen address, endpoint, and client target then carries
//! ([[RFC-0002:C-LOCAL-PLACEMENT]], [[ADR-0074]]).

use crate::InferlabError;
use crate::execution::MachineAddressPlan;
use crate::workspace::MachineBinding;
use std::collections::BTreeMap;
use std::net::IpAddr;
use std::process::Command;

#[derive(Default)]
pub(super) struct MachineAddresses {
    resolved: BTreeMap<String, MachineAddressPlan>,
}

impl MachineAddresses {
    pub(super) fn address(
        &mut self,
        machine_id: &str,
        machine: &MachineBinding,
    ) -> Result<String, InferlabError> {
        if let Some(resolved) = self.resolved.get(machine_id) {
            return Ok(resolved.address.clone());
        }
        let address = lookup(machine_id, &machine.host)?.to_string();
        self.resolved.insert(
            machine_id.to_owned(),
            MachineAddressPlan {
                host: machine.host.clone(),
                address: address.clone(),
            },
        );
        Ok(address)
    }

    pub(super) fn finish(self) -> crate::execution::MachineAddresses {
        self.resolved
    }
}

/// The warning for a server that uses a machine launched over SSH while a
/// used machine's address is loopback: only processes and clients on this
/// machine reach a loopback address, which holds for an SSH launch only when
/// its target is this machine, a fact InferLab cannot observe
/// ([[RFC-0002:C-LOCAL-PLACEMENT]]).
pub(crate) fn loopback_warning(server: &crate::execution::ServerPlan) -> Option<String> {
    let remote = server
        .processes()
        .find_map(|process| match &process.launch {
            inferlab_runtime::plan::LaunchPlan::Ssh { .. } => Some(process.machine.as_str()),
            inferlab_runtime::plan::LaunchPlan::Local => None,
        })?;
    let (machine_id, loopback) = server.placement.addresses.iter().find(|(_, resolved)| {
        resolved
            .address
            .parse::<IpAddr>()
            .is_ok_and(|address| address.is_loopback())
    })?;
    Some(format!(
        "machine {machine_id:?} host {:?} resolves to loopback address {} on this machine, while machine {remote:?} launches over SSH; unless that SSH target is this machine, set host to an address every machine of the server reaches",
        loopback.host, loopback.address
    ))
}

/// The address of `host` through the system resolver (`getent ahosts`): the
/// first IPv4 address it returns, otherwise its first IPv6 address. Pinning
/// one address removes the fallback a client gets from trying each result,
/// and external components are not all known to accept IPv6, so IPv4 wins
/// on a dual-stack name. An IP address is used as written.
fn lookup(machine_id: &str, host: &str) -> Result<IpAddr, InferlabError> {
    if let Ok(address) = host.parse::<IpAddr>() {
        return Ok(address);
    }
    let output = Command::new("getent")
        .args(["ahosts", host])
        .output()
        .map_err(|source| InferlabError::InvalidConfig {
            message: format!(
                "failed to run getent to look up host {host:?} of machine {machine_id:?}: {source}"
            ),
        })?;
    // getent exits 2 when the name is not found.
    if output.status.code() == Some(2) {
        return Err(InferlabError::InvalidConfig {
            message: format!(
                "host {host:?} of machine {machine_id:?} does not resolve on this machine"
            ),
        });
    }
    if !output.status.success() {
        return Err(InferlabError::InvalidConfig {
            message: format!(
                "looking up host {host:?} of machine {machine_id:?} failed ({}): {}",
                output.status,
                String::from_utf8_lossy(&output.stderr).trim()
            ),
        });
    }
    preferred_address(&String::from_utf8_lossy(&output.stdout)).ok_or_else(|| {
        InferlabError::InvalidConfig {
            message: format!(
                "looking up host {host:?} of machine {machine_id:?} returned no address"
            ),
        }
    })
}

/// Each `getent ahosts` row starts with an address, in resolver order.
fn preferred_address(rows: &str) -> Option<IpAddr> {
    let addresses = rows
        .lines()
        .filter_map(|row| row.split_whitespace().next()?.parse::<IpAddr>().ok())
        .collect::<Vec<_>>();
    addresses
        .iter()
        .find(|address| address.is_ipv4())
        .or_else(|| addresses.first())
        .copied()
}

#[cfg(test)]
mod tests {
    use super::preferred_address;

    #[test]
    fn a_dual_stack_name_resolves_to_its_first_ipv4_address() {
        let rows = "2001:db8::1     STREAM node-a.example\n2001:db8::1     DGRAM  \n192.0.2.11      STREAM \n192.0.2.12      STREAM \n";
        assert_eq!(
            preferred_address(rows).map(|address| address.to_string()),
            Some("192.0.2.11".to_owned())
        );
        let rows = "2001:db8::1     STREAM node-a.example\n2001:db8::1     DGRAM  \n";
        assert_eq!(
            preferred_address(rows).map(|address| address.to_string()),
            Some("2001:db8::1".to_owned())
        );
    }
}
