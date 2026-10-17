//! Container disappearance as a completion wait
//! ([[RFC-0009:C-COMPLETION-WAITS]]): every container InferLab starts runs
//! with `--rm`, so after a removal request the daemon removes it once its
//! processes have exited. That can outlast the daemon's own kill wait while
//! the container releases pinned memory or writes back dirty pages, so the
//! wait follows the container's cgroup — CPU time, charged memory, dirty and
//! writeback bytes — and its status. A container that has exited but is
//! still present is asked to be removed again.

use super::process::run_script;
use super::{CompletionTarget, Observation};
use crate::operation_bound::OperationBound;
use crate::shell::shell_quote;

const CONTAINER_MARKER: &str = "INFERLAB_CONTAINER_PROGRESS\t";

#[derive(Clone, Debug)]
pub struct ContainerGone {
    pub target: Option<String>,
    pub container: String,
    /// The container's cgroup directory, learned on the first observation
    /// that sees the container's init process.
    cgroup: String,
}

impl ContainerGone {
    pub fn new(target: Option<&str>, container: &str) -> Self {
        Self {
            target: target.map(str::to_owned),
            container: container.to_owned(),
            cgroup: String::new(),
        }
    }

    fn script(&self) -> String {
        format!(
            "set +e; printf '\\n'; c={container}; cg={cgroup}; \
             out=$(docker container inspect --format '{{{{.State.Status}}}} {{{{.State.Pid}}}}' \"$c\" 2>&1); \
             if [ $? -ne 0 ]; then case $out in *'No such container'*) printf '{marker}absent\\n'; exit 0 ;; esac; \
             printf '%s\\n' \"$out\" >&2; exit 1; fi; set -- $out; status=$1; pid=${{2:-0}}; \
             case $status in exited|dead|created) docker rm -f \"$c\" >/dev/null 2>&1 ;; esac; \
             if [ -z \"$cg\" ] && [ \"$pid\" != 0 ]; then \
             rel=$(sed -n 's/^0:://p' /proc/$pid/cgroup 2>/dev/null); [ -n \"$rel\" ] && cg=/sys/fs/cgroup$rel; fi; \
             cpu=-; mem=-; wb=-; dirty=-; \
             if [ -n \"$cg\" ] && [ -r \"$cg/cpu.stat\" ]; then \
             while read -r k v; do [ \"$k\" = usage_usec ] && cpu=$v; done < \"$cg/cpu.stat\"; \
             read -r mem < \"$cg/memory.current\"; \
             while read -r k v; do case $k in file_writeback) wb=$v ;; file_dirty) dirty=$v ;; esac; done < \"$cg/memory.stat\"; fi; \
             printf '{marker}present\\t%s\\t%s\\t%s\\t%s\\t%s\\t%s\\n' \"$status\" \"$cg\" \"$cpu\" \"$mem\" \"$wb\" \"$dirty\"",
            container = shell_quote(&self.container),
            cgroup = shell_quote(&self.cgroup),
            marker = CONTAINER_MARKER,
        )
    }
}

fn status_rank(status: &str) -> u64 {
    match status {
        "created" => 1,
        "running" => 2,
        "paused" => 3,
        "restarting" => 4,
        "removing" => 5,
        "exited" => 6,
        "dead" => 7,
        _ => 0,
    }
}

impl CompletionTarget for ContainerGone {
    fn awaited(&self) -> String {
        match &self.target {
            Some(target) => format!("container {} on {target} is gone", self.container),
            None => format!("container {} is gone", self.container),
        }
    }

    fn observe(&mut self, attempt: &OperationBound) -> Result<Observation, String> {
        let stdout = run_script(self.target.as_deref(), &self.script(), attempt)?;
        let line = stdout
            .lines()
            .rev()
            .find_map(|line| line.strip_prefix(CONTAINER_MARKER))
            .ok_or_else(|| "the container observation returned no result".to_owned())?;
        let mut fields = line.split('\t');
        match fields.next() {
            Some("absent") => Ok(Observation {
                reached: true,
                ..Observation::default()
            }),
            Some("present") => {
                let status = fields.next().unwrap_or("").to_owned();
                if let Some(cgroup) = fields.next().filter(|cgroup| !cgroup.is_empty()) {
                    cgroup.clone_into(&mut self.cgroup);
                }
                let mut observation = Observation {
                    signals: vec![("container_status", status_rank(&status))],
                    remaining: format!("container is {status}"),
                    ..Observation::default()
                };
                for name in [
                    "cgroup_cpu_usec",
                    "cgroup_memory_bytes",
                    "cgroup_writeback_bytes",
                    "cgroup_dirty_bytes",
                ] {
                    match fields.next().and_then(|value| value.parse::<u64>().ok()) {
                        Some(value) => observation.signals.push((name, value)),
                        None => observation.unavailable.push(name),
                    }
                }
                Ok(observation)
            }
            _ => Err(format!("malformed container observation {line:?}")),
        }
    }
}
