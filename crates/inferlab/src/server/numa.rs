//! NUMA pinning for server processes ([[RFC-0002:C-LOCAL-PLACEMENT]],
//! [[ADR-0053]]): a machine binding's declared `numa_nodes` bind every server
//! process launched on it. Each pinned machine is probed once on its launch
//! path for the nodes' CPU lists and, for host launches, `numactl`; the
//! binding then enters the resolved commands, where dry-run and records
//! already show it.

use crate::execution::ProcessPlan;
use crate::image::launch::CONTAINER_RUN;
use crate::workspace::MachineBinding;
use inferlab_runtime::operation_bound::OperationBound;
use inferlab_runtime::plan::LaunchPlan;
use std::collections::BTreeMap;
use std::process::{Command, Output};
use thiserror::Error;

const NUMA_MARKER: &str = "INFERLAB_NUMA\t";

#[derive(Debug, Error)]
pub enum NumaPinningError {
    #[error("failed to launch the NUMA probe for machine {machine:?}: {source}")]
    LocalLaunch {
        machine: String,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to launch the NUMA probe for machine {machine:?}: {source}")]
    Ssh {
        machine: String,
        #[source]
        source: inferlab_runtime::ssh::SshError,
    },
    #[error("NUMA probe for machine {machine:?} exited with {status}: {stderr}")]
    Exit {
        machine: String,
        status: std::process::ExitStatus,
        stderr: String,
    },
    #[error("machine {machine:?} declares NUMA node {node}, which it does not have")]
    AbsentNode { machine: String, node: String },
    #[error("NUMA probe for machine {machine:?} did not report declared node {node}")]
    Unreported { machine: String, node: u32 },
    #[error("machine {machine:?} declares NUMA node {node}, which has no CPUs to bind")]
    NodeWithoutCpus { machine: String, node: String },
    #[error(
        "machine {machine:?} pins host processes to NUMA nodes but provides no numactl; install it there or remove numa_nodes"
    )]
    MissingNumactl { machine: String },
}

/// What one pinned machine reported.
struct MachineNuma {
    cpus: Vec<String>,
    numactl: bool,
}

/// Bind every server process on a machine that declares `numa_nodes`: a host
/// process runs under `numactl`, a containerized one receives the container
/// runtime's cpuset. Runs after containerization, so a process's command is
/// already the one that will launch.
pub(crate) fn apply<'a>(
    processes: impl IntoIterator<Item = &'a mut ProcessPlan>,
    machines: &BTreeMap<String, MachineBinding>,
) -> Result<(), NumaPinningError> {
    apply_with(processes, machines, probe)
}

fn apply_with<'a>(
    processes: impl IntoIterator<Item = &'a mut ProcessPlan>,
    machines: &BTreeMap<String, MachineBinding>,
    mut probe: impl FnMut(&str, &LaunchPlan, &[u32]) -> Result<MachineNuma, NumaPinningError>,
) -> Result<(), NumaPinningError> {
    let mut probed = BTreeMap::<String, MachineNuma>::new();
    for process in processes {
        let Some(nodes) = machines
            .get(&process.machine)
            .and_then(|machine| machine.numa_nodes.as_deref())
        else {
            continue;
        };
        if !probed.contains_key(&process.machine) {
            let observed = probe(&process.machine, &process.launch, nodes)?;
            probed.insert(process.machine.clone(), observed);
        }
        let Some(observed) = probed.get(&process.machine) else {
            continue;
        };
        let node_list = nodes
            .iter()
            .map(u32::to_string)
            .collect::<Vec<_>>()
            .join(",");
        if process.container.is_some() {
            let cpuset = [
                "--cpuset-cpus".to_owned(),
                observed.cpus.join(","),
                "--cpuset-mems".to_owned(),
                node_list,
            ];
            let options = CONTAINER_RUN.len();
            process.command.argv.splice(options..options, cpuset);
        } else {
            if !observed.numactl {
                return Err(NumaPinningError::MissingNumactl {
                    machine: process.machine.clone(),
                });
            }
            let prefix = [
                "numactl".to_owned(),
                format!("--cpunodebind={node_list}"),
                format!("--membind={node_list}"),
                "--".to_owned(),
            ];
            process.command.argv.splice(0..0, prefix);
        }
    }
    Ok(())
}

/// The probe script spells its marker through the parser's own constant, so
/// the emitted rows and the parsed prefix cannot drift apart. Its leading
/// newline keeps a login banner printed without one off the first row.
fn probe_script(nodes: &[u32]) -> String {
    let nodes = nodes
        .iter()
        .map(u32::to_string)
        .collect::<Vec<_>>()
        .join(" ");
    format!(
        "set -eu; printf '\\n'; for node in {nodes}; do file=/sys/devices/system/node/node$node/cpulist; \
         if [ -r \"$file\" ]; then printf '{NUMA_MARKER}CPUS\\t%s\\t%s\\n' \"$node\" \"$(cat \"$file\")\"; \
         else printf '{NUMA_MARKER}ABSENT\\t%s\\n' \"$node\"; fi; done; \
         if command -v numactl >/dev/null 2>&1; then printf '{NUMA_MARKER}NUMACTL\\n'; fi"
    )
}

fn probe(
    machine: &str,
    launch: &LaunchPlan,
    nodes: &[u32],
) -> Result<MachineNuma, NumaPinningError> {
    let script = probe_script(nodes);
    let output = match launch {
        LaunchPlan::Local => {
            Command::new("sh")
                .args(["-c", &script])
                .output()
                .map_err(|source| NumaPinningError::LocalLaunch {
                    machine: machine.to_owned(),
                    source,
                })?
        }
        LaunchPlan::Ssh { target } => {
            inferlab_runtime::ssh::ssh_output(target, &script, &OperationBound::unbounded())
                .map_err(|source| NumaPinningError::Ssh {
                    machine: machine.to_owned(),
                    source,
                })?
        }
    };
    parse(machine, nodes, &output)
}

/// Every declared node must be reported: a row lost to banner noise would
/// otherwise leave a container's cpuset empty, which binds no CPUs at all.
fn parse(machine: &str, nodes: &[u32], output: &Output) -> Result<MachineNuma, NumaPinningError> {
    if !output.status.success() {
        return Err(NumaPinningError::Exit {
            machine: machine.to_owned(),
            status: output.status,
            stderr: String::from_utf8_lossy(&output.stderr).trim().to_owned(),
        });
    }
    let mut cpus = BTreeMap::<u32, String>::new();
    let mut numactl = false;
    for line in String::from_utf8_lossy(&output.stdout).lines() {
        let Some(row) = line.strip_prefix(NUMA_MARKER) else {
            continue;
        };
        let fields = row.split('\t').collect::<Vec<_>>();
        match fields.as_slice() {
            ["CPUS", node, ""] => {
                return Err(NumaPinningError::NodeWithoutCpus {
                    machine: machine.to_owned(),
                    node: (*node).to_owned(),
                });
            }
            ["CPUS", node, list] => {
                if let Ok(node) = node.parse::<u32>() {
                    cpus.insert(node, (*list).to_owned());
                }
            }
            ["ABSENT", node] => {
                return Err(NumaPinningError::AbsentNode {
                    machine: machine.to_owned(),
                    node: (*node).to_owned(),
                });
            }
            ["NUMACTL"] => numactl = true,
            _ => {}
        }
    }
    let cpus = nodes
        .iter()
        .map(|node| {
            cpus.remove(node)
                .ok_or_else(|| NumaPinningError::Unreported {
                    machine: machine.to_owned(),
                    node: *node,
                })
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(MachineNuma { cpus, numactl })
}

#[cfg(test)]
mod tests {
    use super::{MachineNuma, NUMA_MARKER, NumaPinningError, apply_with, parse, probe_script};
    use crate::execution::ProcessPlan;
    use crate::image::launch::CONTAINER_RUN;
    use crate::workspace::MachineBinding;

    fn fixture(argv: &[&str]) -> ProcessPlan {
        let mut process = crate::server::tests::process(0);
        process.machine = "local".to_owned();
        process.command.argv = argv.iter().map(|arg| (*arg).to_owned()).collect();
        process
    }
    use std::collections::BTreeMap;
    use std::os::unix::process::ExitStatusExt;
    use std::process::{Command, ExitStatus, Output};

    fn machines(
        numa_nodes: Option<Vec<u32>>,
    ) -> Result<BTreeMap<String, MachineBinding>, toml::de::Error> {
        let mut binding: MachineBinding =
            toml::from_str("host = \"127.0.0.1\"\ndevices = [0]\nports = [8000]\n")?;
        binding.numa_nodes = numa_nodes;
        Ok(BTreeMap::from([("local".to_owned(), binding)]))
    }

    fn observed(numactl: bool) -> MachineNuma {
        MachineNuma {
            cpus: vec!["0-3".to_owned()],
            numactl,
        }
    }

    fn succeeded(stdout: String) -> Output {
        Output {
            status: ExitStatus::from_raw(0),
            stdout: stdout.into_bytes(),
            stderr: Vec::new(),
        }
    }

    #[test]
    fn host_processes_run_under_numactl_and_containers_get_a_cpuset()
    -> Result<(), Box<dyn std::error::Error>> {
        let mut host = fixture(&["pixi", "run"]);
        let mut container = fixture(&[CONTAINER_RUN.as_slice(), &["--rm"]].concat());
        container.container = Some(crate::execution::ContainerPlan {
            name: "inferlab-fixture".to_owned(),
            image: "fixture".to_owned(),
        });
        let mut probes = 0;
        apply_with(
            [&mut host, &mut container],
            &machines(Some(vec![0]))?,
            |_, _, _| {
                probes += 1;
                Ok(observed(true))
            },
        )?;
        assert_eq!(probes, 1, "one probe per pinned machine");
        assert_eq!(
            host.command.argv,
            [
                "numactl",
                "--cpunodebind=0",
                "--membind=0",
                "--",
                "pixi",
                "run"
            ]
        );
        assert_eq!(
            container.command.argv,
            [
                "docker",
                "run",
                "--cpuset-cpus",
                "0-3",
                "--cpuset-mems",
                "0",
                "--rm"
            ]
        );
        Ok(())
    }

    #[test]
    fn an_unpinned_machine_is_never_probed() -> Result<(), Box<dyn std::error::Error>> {
        let mut host = fixture(&["pixi"]);
        apply_with([&mut host], &machines(None)?, |_, _, _| {
            Err(NumaPinningError::MissingNumactl {
                machine: "local".to_owned(),
            })
        })?;
        assert_eq!(host.command.argv, ["pixi"]);
        Ok(())
    }

    #[test]
    fn a_host_launch_without_numactl_fails_naming_the_machine()
    -> Result<(), Box<dyn std::error::Error>> {
        let mut host = fixture(&["pixi"]);
        let missing = apply_with([&mut host], &machines(Some(vec![0]))?, |_, _, _| {
            Ok(observed(false))
        });
        assert!(
            matches!(&missing, Err(NumaPinningError::MissingNumactl { machine }) if machine == "local"),
            "{missing:?}"
        );
        Ok(())
    }

    #[test]
    fn the_probe_script_reports_an_absent_node_behind_a_login_banner()
    -> Result<(), Box<dyn std::error::Error>> {
        // A login profile may print without a trailing newline; the probe
        // rows must still start on their own lines. No machine has node 4096.
        let output = Command::new("sh")
            .args([
                "-c",
                &format!("printf 'login banner'; {}", probe_script(&[4096])),
            ])
            .output()?;
        let absent = parse("local", &[4096], &output);
        assert!(
            matches!(&absent, Err(NumaPinningError::AbsentNode { machine, node }) if machine == "local" && node == "4096"),
            "{:?}",
            absent.err()
        );
        Ok(())
    }

    #[test]
    fn a_probe_that_omits_a_declared_node_fails_naming_it() {
        let unreported = parse(
            "local",
            &[0, 2],
            &succeeded(format!("{NUMA_MARKER}CPUS\t0\t0-3\n{NUMA_MARKER}NUMACTL\n")),
        );
        assert!(
            matches!(&unreported, Err(NumaPinningError::Unreported { machine, node }) if machine == "local" && *node == 2),
            "{:?}",
            unreported.err()
        );
    }

    #[test]
    fn a_declared_node_without_cpus_fails_naming_the_machine_and_node() {
        let cpuless = parse(
            "local",
            &[0, 2],
            &succeeded(format!(
                "{NUMA_MARKER}CPUS\t0\t0-3\n{NUMA_MARKER}CPUS\t2\t\n"
            )),
        );
        assert!(
            matches!(&cpuless, Err(NumaPinningError::NodeWithoutCpus { machine, node }) if machine == "local" && node == "2"),
            "{:?}",
            cpuless.err()
        );
    }
}
