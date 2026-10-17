//! Environment completion waits ([[RFC-0009:C-COMPLETION-WAITS]],
//! [[ADR-0067]]): waiting for the environment to finish an action InferLab
//! already requested — a process exiting after SIGKILL, a container
//! disappearing after its removal, device memory being released. How long
//! that takes grows with what the target holds, so no deadline predicts it.
//! A wait ends when the awaited state is observed, when a new operator
//! interrupt arrives, when no attributable progress has been observed for the
//! stall window, or when the attention ceiling elapses; each observation is
//! one attempt with its own bound below the stall window.

pub mod container;
pub mod process;

use crate::operation_bound::{OperationBound, duration_millis};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::time::{Duration, Instant};

/// How long a wait goes without any attributable progress before it is
/// judged stalled: it measures the absence of progress, not the work, so it
/// does not scale with the target.
pub const STALL_WINDOW: Duration = Duration::from_secs(30);
/// How long a workflow withholds control before recording a cleanup as
/// unverified even while its target keeps working; expiry leaves the cleanup
/// retryable.
pub const ATTENTION_CEILING: Duration = Duration::from_secs(30 * 60);
/// The bound of each observation or action attempt inside a wait, kept below
/// the stall window so one slow attempt is not mistaken for a stall. It
/// leaves room for a container runtime's own 12-second kill wait.
pub const ATTEMPT_BOUND: Duration = Duration::from_secs(20);
/// The cadence between observations: polling, not a bound.
pub const OBSERVATION_INTERVAL: Duration = Duration::from_millis(500);
/// When a wait still running first tells the operator what it awaits.
pub const NOTICE_DELAY: Duration = Duration::from_secs(10);

/// The fixed policy one wait runs under; tests shrink it.
#[derive(Clone, Copy, Debug)]
pub struct Policy {
    pub stall_window: Duration,
    pub attention_ceiling: Duration,
    pub attempt_bound: Duration,
    pub interval: Duration,
    pub notice_delay: Duration,
}

impl Policy {
    pub const PRODUCT: Self = Self {
        stall_window: STALL_WINDOW,
        attention_ceiling: ATTENTION_CEILING,
        attempt_bound: ATTEMPT_BOUND,
        interval: OBSERVATION_INTERVAL,
        notice_delay: NOTICE_DELAY,
    };
}

/// One successful observation of a wait's targets.
#[derive(Clone, Debug, Default)]
pub struct Observation {
    /// The awaited state holds.
    pub reached: bool,
    /// Attributable progress signals and their current values; a changed
    /// value between two observations is progress.
    pub signals: Vec<(&'static str, u64)>,
    /// Signals the target offers but could not read this time.
    pub unavailable: Vec<&'static str>,
    /// What still remains, in the operator's terms.
    pub remaining: String,
}

/// What a completion wait waits for.
pub trait CompletionTarget {
    /// The awaited state, for diagnostics and evidence.
    fn awaited(&self) -> String;
    /// Observe the targets once within `attempt`; an error is one failed
    /// observation, never by itself the end of the wait.
    fn observe(&mut self, attempt: &OperationBound) -> Result<Observation, String>;
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CompletionEnd {
    Reached,
    Stalled,
    AttentionCeiling,
    Interrupted,
}

/// How one completion wait went ([[RFC-0009:C-CLEANUP-GRACE]]).
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CompletionWaitEvidence {
    pub awaited: String,
    pub ended: CompletionEnd,
    pub elapsed_ms: u64,
    pub stall_window_ms: u64,
    pub attention_ceiling_ms: u64,
    pub attempt_bound_ms: u64,
    pub observations: u32,
    pub failed_observations: u32,
    /// Signals read at least once.
    pub signals: Vec<String>,
    /// Signals the targets offer that were never read.
    pub unavailable_signals: Vec<String>,
    /// Elapsed time at which progress was last observed; 0 when none was.
    pub last_progress_ms: u64,
    /// What remained when the wait ended without reaching its state.
    pub remaining: Option<String>,
    pub last_error: Option<String>,
}

impl CompletionWaitEvidence {
    #[must_use]
    pub fn reached(&self) -> bool {
        self.ended == CompletionEnd::Reached
    }

    /// One sentence for an unverified cleanup's error.
    #[must_use]
    pub fn failure(&self) -> Option<String> {
        let remaining = self
            .remaining
            .as_deref()
            .map_or(String::new(), |remaining| {
                format!("; remaining: {remaining}")
            });
        let error = self.last_error.as_deref().map_or(String::new(), |error| {
            format!("; last observation error: {error}")
        });
        match self.ended {
            CompletionEnd::Reached => None,
            CompletionEnd::Stalled => Some(format!(
                "{} did not happen: no progress for {} s{remaining}{error}",
                self.awaited,
                self.stall_window_ms / 1000
            )),
            CompletionEnd::AttentionCeiling => Some(format!(
                "{} did not happen within the {} s attention ceiling{remaining}{error}",
                self.awaited,
                self.attention_ceiling_ms / 1000
            )),
            CompletionEnd::Interrupted => Some(format!(
                "waiting until {} was interrupted{remaining}",
                self.awaited
            )),
        }
    }
}

/// Wait for `target` under the product policy; a new operator interrupt —
/// one arriving after the wait starts — ends it.
pub fn wait(
    target: &mut dyn CompletionTarget,
    notice: &mut dyn FnMut(&str),
) -> CompletionWaitEvidence {
    let start = crate::interrupt::signals();
    wait_with(target, Policy::PRODUCT, notice, &mut || {
        crate::interrupt::signals() != start
    })
}

pub fn wait_with(
    target: &mut dyn CompletionTarget,
    policy: Policy,
    notice: &mut dyn FnMut(&str),
    interrupted: &mut dyn FnMut() -> bool,
) -> CompletionWaitEvidence {
    let started = Instant::now();
    let awaited = target.awaited();
    let mut last_progress = started;
    let mut progressed = false;
    let mut previous = BTreeMap::<&'static str, u64>::new();
    let mut read = BTreeSet::<&'static str>::new();
    let mut offered = BTreeSet::<&'static str>::new();
    let mut observations = 0_u32;
    let mut failed = 0_u32;
    let mut remaining = None;
    let mut last_error = None;
    let mut noticed = false;
    let end = loop {
        let attempt = OperationBound::finite(policy.attempt_bound);
        match target.observe(&attempt) {
            Ok(observation) => {
                observations += 1;
                offered.extend(observation.unavailable.iter().copied());
                if observation.reached {
                    remaining = None;
                    break CompletionEnd::Reached;
                }
                let mut changed = false;
                for (name, value) in observation.signals {
                    offered.insert(name);
                    read.insert(name);
                    if previous
                        .insert(name, value)
                        .is_some_and(|before| before != value)
                    {
                        changed = true;
                    }
                }
                if changed {
                    last_progress = Instant::now();
                    progressed = true;
                }
                remaining = Some(observation.remaining);
            }
            Err(error) => {
                failed += 1;
                last_error = Some(error);
            }
        }
        if interrupted() {
            break CompletionEnd::Interrupted;
        }
        if last_progress.elapsed() >= policy.stall_window {
            break CompletionEnd::Stalled;
        }
        if started.elapsed() >= policy.attention_ceiling {
            break CompletionEnd::AttentionCeiling;
        }
        if !noticed && started.elapsed() >= policy.notice_delay {
            noticed = true;
            notice(&format!(
                "waiting until {awaited}; {}",
                remaining.as_deref().unwrap_or("not yet observed")
            ));
        }
        std::thread::sleep(policy.interval);
    };
    CompletionWaitEvidence {
        awaited,
        ended: end,
        elapsed_ms: duration_millis(started.elapsed()),
        stall_window_ms: duration_millis(policy.stall_window),
        attention_ceiling_ms: duration_millis(policy.attention_ceiling),
        attempt_bound_ms: duration_millis(policy.attempt_bound),
        observations,
        failed_observations: failed,
        signals: read.iter().map(|name| (*name).to_owned()).collect(),
        unavailable_signals: offered
            .difference(&read)
            .map(|name| (*name).to_owned())
            .collect(),
        last_progress_ms: if progressed {
            duration_millis(last_progress.duration_since(started))
        } else {
            0
        },
        remaining,
        last_error,
    }
}

#[cfg(test)]
mod tests {
    use super::{CompletionEnd, CompletionTarget, Observation, Policy, wait_with};
    use crate::operation_bound::OperationBound;
    use std::time::Duration;

    const FAST: Policy = Policy {
        stall_window: Duration::from_millis(120),
        attention_ceiling: Duration::from_millis(600),
        attempt_bound: Duration::from_millis(50),
        interval: Duration::from_millis(10),
        notice_delay: Duration::from_millis(30),
    };

    /// A target that reports a scripted sequence of observations, repeating
    /// the last one.
    struct Scripted(Vec<Result<Observation, String>>, usize);

    impl CompletionTarget for Scripted {
        fn awaited(&self) -> String {
            "the fixture target exits".to_owned()
        }
        fn observe(&mut self, _: &OperationBound) -> Result<Observation, String> {
            let index = self.1.min(self.0.len() - 1);
            self.1 += 1;
            self.0[index].clone()
        }
    }

    fn working(cpu: u64) -> Result<Observation, String> {
        Ok(Observation {
            signals: vec![("cpu_ticks", cpu), ("tasks", 1)],
            unavailable: vec!["memory"],
            remaining: "1 task".to_owned(),
            ..Observation::default()
        })
    }

    fn run(target: &mut Scripted) -> super::CompletionWaitEvidence {
        wait_with(target, FAST, &mut |_| {}, &mut || false)
    }

    #[test]
    fn a_target_that_keeps_progressing_outlasts_the_stall_window() {
        // CPU advances on every observation for longer than the stall window.
        let mut script = (0..40).map(working).collect::<Vec<_>>();
        script.push(Ok(Observation {
            reached: true,
            ..Observation::default()
        }));
        let evidence = run(&mut Scripted(script, 0));
        assert_eq!(evidence.ended, CompletionEnd::Reached);
        assert!(evidence.elapsed_ms > 120, "{evidence:?}");
        assert_eq!(evidence.remaining, None);
        assert_eq!(evidence.signals, ["cpu_ticks", "tasks"]);
        assert_eq!(evidence.unavailable_signals, ["memory"]);
    }

    #[test]
    fn a_target_without_progress_stalls_after_the_window() {
        let evidence = run(&mut Scripted(vec![working(7)], 0));
        assert_eq!(evidence.ended, CompletionEnd::Stalled);
        assert_eq!(evidence.last_progress_ms, 0);
        assert_eq!(evidence.remaining.as_deref(), Some("1 task"));
        assert!(
            evidence
                .failure()
                .is_some_and(|f| f.contains("no progress"))
        );
    }

    #[test]
    fn a_target_that_works_without_finishing_ends_at_the_ceiling() {
        let script = (0..1000).map(working).collect::<Vec<_>>();
        let evidence = run(&mut Scripted(script, 0));
        assert_eq!(evidence.ended, CompletionEnd::AttentionCeiling);
        assert!(evidence.last_progress_ms > 0);
    }

    #[test]
    fn failed_observations_do_not_end_the_wait_and_are_counted() {
        let mut script = vec![Err("transport closed".to_owned()); 3];
        script.push(Ok(Observation {
            reached: true,
            ..Observation::default()
        }));
        let evidence = run(&mut Scripted(script, 0));
        assert_eq!(evidence.ended, CompletionEnd::Reached);
        assert_eq!(evidence.failed_observations, 3);
        assert_eq!(evidence.last_error.as_deref(), Some("transport closed"));
    }

    #[test]
    fn only_failed_observations_stall() {
        let evidence = run(&mut Scripted(vec![Err("unreachable".to_owned())], 0));
        assert_eq!(evidence.ended, CompletionEnd::Stalled);
        assert_eq!(evidence.observations, 0);
    }

    #[test]
    fn a_new_interrupt_ends_the_wait_and_a_long_wait_says_what_it_awaits() {
        let mut calls = 0;
        let mut notices = Vec::new();
        let script = (0..1000).map(working).collect::<Vec<_>>();
        let evidence = wait_with(
            &mut Scripted(script, 0),
            FAST,
            &mut |message| notices.push(message.to_owned()),
            &mut || {
                calls += 1;
                calls > 8
            },
        );
        assert_eq!(evidence.ended, CompletionEnd::Interrupted);
        assert_eq!(notices.len(), 1);
        assert!(
            notices[0].contains("the fixture target exits"),
            "{notices:?}"
        );
    }
}
