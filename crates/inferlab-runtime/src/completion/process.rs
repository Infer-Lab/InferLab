//! Process exit as a completion wait ([[RFC-0009:C-COMPLETION-WAITS]]):
//! after SIGKILL a process tears down in its remaining threads — the leader
//! can already read as a zombie while another thread spends many seconds
//! releasing pinned memory — so a process has exited only when none of its
//! threads lives. Progress is the live-thread count falling or the threads'
//! CPU time rising. One shell script observes both launch paths, locally or
//! over SSH.

use super::{CompletionTarget, Observation};
use crate::container::{BoundedError, BoundedWait, run_cleanup_with_bound};
use crate::operation_bound::OperationBound;
use crate::ssh::{SSH_ENV_REMOVE, ssh_argv};

const PROGRESS_MARKER: &str = "INFERLAB_EXIT_PROGRESS\t";

/// Which processes a wait follows.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Processes {
    /// Every process in the group.
    Group(u32),
    /// These exact processes, each named by pid and start time so a reused
    /// pid never stands in for one that exited.
    Exact(Vec<(u32, u64)>),
    /// Processes the caller listed moments ago without their start times.
    Listed(Vec<u32>),
}

/// Wait until every thread of the selected processes has exited, on the
/// local machine or on the SSH `target`.
#[derive(Clone, Debug)]
pub struct ProcessExit {
    pub target: Option<String>,
    pub processes: Processes,
}

impl ProcessExit {
    /// The observation as a POSIX shell script, for a launch machine reached
    /// through a transport: it prints one result line that
    /// [`parse_exit_observation`] reads.
    pub fn script(&self) -> String {
        let select = match &self.processes {
            Processes::Group(group) => format!(
                "for s in /proc/[0-9]*/stat; do read -r line < \"$s\" 2>/dev/null || continue; \
                 rest=${{line##*) }}; set -- $rest; [ \"$3\" = {group} ] || continue; \
                 p=${{s#/proc/}}; pids=\"$pids ${{p%/stat}}\"; done"
            ),
            Processes::Exact(processes) => {
                let pairs = processes
                    .iter()
                    .map(|(pid, start)| format!("{pid}:{start}"))
                    .collect::<Vec<_>>()
                    .join(" ");
                format!(
                    "for pair in {pairs}; do p=${{pair%:*}}; want=${{pair#*:}}; \
                     read -r line < /proc/$p/stat 2>/dev/null || continue; \
                     rest=${{line##*) }}; set -- $rest; [ \"${{20}}\" = \"$want\" ] || continue; \
                     pids=\"$pids $p\"; done"
                )
            }
            Processes::Listed(pids) => format!(
                "pids='{}'",
                pids.iter()
                    .map(u32::to_string)
                    .collect::<Vec<_>>()
                    .join(" ")
            ),
        };
        format!(
            "set +e; printf '\\n'; pids=; {select}; live=0; ticks=0; left=; \
             for p in $pids; do n=0; for t in /proc/$p/task/*/stat; do \
             read -r line < \"$t\" 2>/dev/null || continue; rest=${{line##*) }}; set -- $rest; \
             ticks=$((ticks + ${{12}} + ${{13}})); case $1 in Z|X) ;; *) n=$((n + 1)) ;; esac; \
             done; if [ $n -gt 0 ]; then live=$((live + n)); left=\"$left $p\"; fi; done; \
             printf '{PROGRESS_MARKER}%s\\t%s\\t%s\\n' \"$live\" \"$ticks\" \"$left\""
        )
    }
}

impl CompletionTarget for ProcessExit {
    fn awaited(&self) -> String {
        let what = match &self.processes {
            Processes::Group(group) => format!("process group {group}"),
            Processes::Exact(processes) => format!(
                "processes {}",
                processes
                    .iter()
                    .map(|(pid, _)| pid.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            Processes::Listed(pids) => format!(
                "processes {}",
                pids.iter()
                    .map(u32::to_string)
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        };
        match &self.target {
            Some(target) => format!("every thread of {what} on {target} has exited"),
            None => format!("every thread of {what} has exited"),
        }
    }

    fn observe(&mut self, attempt: &OperationBound) -> Result<Observation, String> {
        // Locally the observation reads /proc in-process: it spawns nothing,
        // so terminating an expired command — which waits through this same
        // observation — cannot recurse into another bounded command.
        let (live, ticks, left) = match self.target.as_deref() {
            None => observe_local(&self.processes),
            Some(target) => {
                parse_exit_observation(&run_script(Some(target), &self.script(), attempt)?)?
            }
        };
        Ok(Observation {
            reached: live == 0,
            signals: vec![("live_threads", live), ("cpu_ticks", ticks)],
            unavailable: Vec::new(),
            remaining: format!("{live} live threads in processes {left}"),
        })
    }
}

/// Parse the exit observation script's result line.
pub fn parse_exit_observation(stdout: &str) -> Result<(u64, u64, String), String> {
    let line = stdout
        .lines()
        .rev()
        .find_map(|line| line.strip_prefix(PROGRESS_MARKER))
        .ok_or_else(|| "the exit observation returned no result".to_owned())?;
    let mut fields = line.split('\t');
    let number = |field: Option<&str>| {
        field
            .and_then(|value| value.trim().parse::<u64>().ok())
            .ok_or_else(|| format!("malformed exit observation {line:?}"))
    };
    let live = number(fields.next())?;
    let ticks = number(fields.next())?;
    Ok((live, ticks, fields.next().unwrap_or("").trim().to_owned()))
}

/// The fields of a `/proc` stat line after the command name: state first.
fn stat_fields(path: &str) -> Option<Vec<String>> {
    let stat = std::fs::read_to_string(path).ok()?;
    let rest = &stat[stat.rfind(") ")? + 2..];
    Some(rest.split_whitespace().map(str::to_owned).collect())
}

/// The same observation as [`ProcessExit::script`], read in-process.
fn observe_local(processes: &Processes) -> (u64, u64, String) {
    let pids: Vec<u32> = match processes {
        Processes::Group(group) => std::fs::read_dir("/proc")
            .into_iter()
            .flatten()
            .flatten()
            .filter_map(|entry| entry.file_name().to_str()?.parse::<u32>().ok())
            .filter(|pid| {
                stat_fields(&format!("/proc/{pid}/stat"))
                    .is_some_and(|fields| fields.get(2).is_some_and(|g| *g == group.to_string()))
            })
            .collect(),
        Processes::Exact(processes) => processes
            .iter()
            .filter(|(pid, start)| {
                stat_fields(&format!("/proc/{pid}/stat"))
                    .is_some_and(|fields| fields.get(19).is_some_and(|s| *s == start.to_string()))
            })
            .map(|(pid, _)| *pid)
            .collect(),
        Processes::Listed(pids) => pids.clone(),
    };
    let (mut live, mut ticks, mut left) = (0_u64, 0_u64, Vec::new());
    for pid in pids {
        let mut threads = 0_u64;
        for task in std::fs::read_dir(format!("/proc/{pid}/task"))
            .into_iter()
            .flatten()
            .flatten()
        {
            let Some(fields) = stat_fields(&format!("{}/stat", task.path().display())) else {
                continue;
            };
            let field = |index: usize| fields.get(index).and_then(|v| v.parse::<u64>().ok());
            ticks += field(11).unwrap_or(0) + field(12).unwrap_or(0);
            if !matches!(fields.first().map(String::as_str), Some("Z" | "X")) {
                threads += 1;
            }
        }
        if threads > 0 {
            live += threads;
            left.push(pid.to_string());
        }
    }
    (live, ticks, left.join(" "))
}

/// Run one observation or action script on the launch machine under its
/// attempt bound, returning stdout of a successful run.
pub(crate) fn run_script(
    target: Option<&str>,
    script: &str,
    attempt: &OperationBound,
) -> Result<String, String> {
    let (argv, env_remove): (Vec<String>, &[&str]) = match target {
        Some(target) => (ssh_argv(target, script), SSH_ENV_REMOVE),
        None => (
            vec!["sh".to_owned(), "-c".to_owned(), script.to_owned()],
            &[],
        ),
    };
    match run_cleanup_with_bound(&argv, env_remove, None, None, attempt, None) {
        Ok(BoundedWait::Exited {
            status,
            stdout,
            stderr,
        }) => {
            if status.success() {
                Ok(String::from_utf8_lossy(&stdout).into_owned())
            } else {
                Err(format!(
                    "observation exited with {status}: {}",
                    String::from_utf8_lossy(&stderr).trim()
                ))
            }
        }
        Ok(BoundedWait::Expired { .. }) => Err("observation attempt expired".to_owned()),
        Ok(BoundedWait::Interrupted { .. }) => Err("observation attempt interrupted".to_owned()),
        Err(BoundedError::Launch(error)) => Err(format!("observation failed to launch: {error}")),
        Err(
            BoundedError::Stdin(error)
            | BoundedError::Wait(error)
            | BoundedError::WaitCleanup { source: error, .. },
        ) => Err(format!("observation failed: {error}")),
    }
}

#[cfg(test)]
mod tests {
    use super::{ProcessExit, Processes};
    use crate::completion::CompletionTarget;
    use crate::operation_bound::OperationBound;
    use std::os::unix::process::CommandExt;
    use std::process::Command;
    use std::time::Duration;

    fn observe(exit: &mut ProcessExit) -> Result<crate::completion::Observation, String> {
        exit.observe(&OperationBound::finite(Duration::from_secs(10)))
    }

    #[test]
    fn a_live_group_reports_its_threads_and_an_exited_one_is_reached()
    -> Result<(), Box<dyn std::error::Error>> {
        let mut child = Command::new("sleep").arg("30").process_group(0).spawn()?;
        let mut exit = ProcessExit {
            target: None,
            processes: Processes::Group(child.id()),
        };
        let live = observe(&mut exit)?;
        assert!(!live.reached, "{live:?}");
        assert_eq!(live.signals[0], ("live_threads", 1));
        child.kill()?;
        child.wait()?;
        assert!(observe(&mut exit)?.reached);
        Ok(())
    }

    #[test]
    fn an_exact_process_with_another_start_time_is_not_followed()
    -> Result<(), Box<dyn std::error::Error>> {
        let mut child = Command::new("sleep").arg("30").spawn()?;
        let start = crate::process_group::process_start_time(child.id())?.ok_or("no start")?;
        let mut same = ProcessExit {
            target: None,
            processes: Processes::Exact(vec![(child.id(), start)]),
        };
        let mut reused = ProcessExit {
            target: None,
            processes: Processes::Exact(vec![(child.id(), start + 1)]),
        };
        let followed = observe(&mut same)?;
        let other = observe(&mut reused)?;
        child.kill()?;
        child.wait()?;
        assert!(!followed.reached);
        assert!(other.reached, "a pid with another start time is not ours");
        Ok(())
    }

    /// The shell script a remote launch machine runs counts the same live
    /// threads as the in-process local observation.
    #[test]
    fn the_remote_script_agrees_with_the_local_observation()
    -> Result<(), Box<dyn std::error::Error>> {
        let mut child = Command::new("python3")
            .args(["-c", "import threading,time\nthreading.Thread(target=lambda: time.sleep(30)).start()\ntime.sleep(30)\n"])
            .process_group(0)
            .spawn()?;
        std::thread::sleep(Duration::from_millis(300));
        let exit = ProcessExit {
            target: None,
            processes: Processes::Group(child.id()),
        };
        let output = Command::new("sh").args(["-c", &exit.script()]).output()?;
        let (script_live, _, script_left) =
            super::parse_exit_observation(&String::from_utf8_lossy(&output.stdout))?;
        let (local_live, _, local_left) = super::observe_local(&exit.processes);
        child.kill()?;
        child.wait()?;
        assert_eq!(script_live, 2);
        assert_eq!((script_live, script_left), (local_live, local_left));
        Ok(())
    }

    /// A leader that already reads as a zombie while another of its threads
    /// still runs has not exited: that thread is where teardown happens.
    #[test]
    fn a_zombie_leader_with_a_live_thread_has_not_exited() -> Result<(), Box<dyn std::error::Error>>
    {
        // The main thread ends with pthread_exit while a worker thread
        // sleeps, so the leader reads as a zombie with one live thread.
        let mut child = Command::new("python3")
            .args([
                "-c",
                "import threading,time,ctypes\n\
                 threading.Thread(target=lambda: time.sleep(30)).start()\n\
                 ctypes.CDLL(None).pthread_exit(None)\n",
            ])
            .process_group(0)
            .spawn()?;
        let mut exit = ProcessExit {
            target: None,
            processes: Processes::Group(child.id()),
        };
        let deadline = std::time::Instant::now() + Duration::from_secs(10);
        let state = loop {
            let stat = std::fs::read_to_string(format!("/proc/{}/stat", child.id()))?;
            let state = stat
                .rsplit(") ")
                .next()
                .and_then(|rest| rest.chars().next());
            if state == Some('Z') || std::time::Instant::now() > deadline {
                break state;
            }
            std::thread::sleep(Duration::from_millis(20));
        };
        let observed = observe(&mut exit)?;
        child.kill()?;
        child.wait()?;
        assert_eq!(state, Some('Z'), "the leader thread did not exit first");
        assert!(!observed.reached, "{observed:?}");
        assert_eq!(observed.signals[0], ("live_threads", 1));
        Ok(())
    }
}
