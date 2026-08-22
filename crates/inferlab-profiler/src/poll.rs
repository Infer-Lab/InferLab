use inferlab_runtime::operation_bound::{OperationBound, Remaining};
use std::time::Duration;

/// One probe outcome inside a budget-aware polling loop.
pub(crate) enum Poll<T> {
    /// The polled condition is decided; polling stops with this outcome.
    Done(T),
    /// The condition does not hold yet; carries the outcome to record if the
    /// owning budget expires before the condition holds.
    Pending(T),
}

/// Probe until the condition is decided or the owning operation's budget
/// expires, doubling the wait between probes from `initial_interval` up to
/// `max_interval` (equal bounds give a fixed interval) and never sleeping
/// past the remaining budget.
pub(crate) fn poll_until<T>(
    bound: &OperationBound,
    initial_interval: Duration,
    max_interval: Duration,
    mut probe: impl FnMut() -> Poll<T>,
) -> T {
    let mut interval = initial_interval;
    loop {
        let pending = match probe() {
            Poll::Done(outcome) => return outcome,
            Poll::Pending(outcome) => outcome,
        };
        if bound.is_expired() {
            return pending;
        }
        match bound.remaining() {
            Remaining::Finite(remaining) => std::thread::sleep(remaining.min(interval)),
            Remaining::Expired => return pending,
            Remaining::Unbounded => std::thread::sleep(interval),
        }
        interval = interval.saturating_mul(2).min(max_interval);
    }
}
