//! One background refresh worker per registered workspace ([[ADR-0056]]):
//! each collects with the TUI's collector on the display cadence and
//! publishes complete generations, so a slow or failing workspace never
//! holds up a request or another workspace ([[RFC-0012:C-VIEWS]]).

use crate::console::Snapshot;
use crate::console::collector::Collector;
use crate::console::presentation::Presentation;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, RecvTimeoutError};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::sync::watch;

/// One complete display generation of a workspace.
pub(super) struct Generation {
    pub(super) number: u64,
    pub(super) completed: Instant,
    pub(super) snapshot: Snapshot,
    pub(super) presentation: Presentation,
}

pub(super) struct Feed {
    pub(super) id: String,
    pub(super) root: PathBuf,
    pub(super) latest: watch::Receiver<Option<Arc<Generation>>>,
    refresh: mpsc::Sender<()>,
}

impl Feed {
    /// The workspace's directory name, as pages title it.
    pub(super) fn name(&self) -> String {
        self.root.file_name().map_or_else(
            || self.root.display().to_string(),
            |name| name.to_string_lossy().into_owned(),
        )
    }

    /// Ask the worker for an immediate refresh that re-reads every source.
    pub(super) fn refresh_now(&self) {
        let _ = self.refresh.send(());
    }
}

pub(super) struct Hub {
    interval: Duration,
    feeds: Mutex<BTreeMap<PathBuf, Arc<Feed>>>,
}

impl Hub {
    pub(super) fn new(interval: Duration) -> Self {
        Self {
            interval,
            feeds: Mutex::new(BTreeMap::new()),
        }
    }

    pub(super) fn interval(&self) -> Duration {
        self.interval
    }

    /// Keep exactly one worker per registered root: start the missing and
    /// stop those whose root is no longer registered.
    pub(super) fn track(&self, roots: &[PathBuf]) {
        let Ok(mut feeds) = self.feeds.lock() else {
            return;
        };
        feeds.retain(|root, _| roots.contains(root));
        for root in roots {
            feeds
                .entry(root.clone())
                .or_insert_with(|| Arc::new(spawn(root, self.interval)));
        }
    }

    /// Every tracked workspace, in root order.
    pub(super) fn feeds(&self) -> Vec<Arc<Feed>> {
        self.feeds
            .lock()
            .map(|feeds| feeds.values().cloned().collect())
            .unwrap_or_default()
    }

    pub(super) fn feed(&self, id: &str) -> Option<Arc<Feed>> {
        self.feeds
            .lock()
            .ok()?
            .values()
            .find(|feed| feed.id == id)
            .cloned()
    }
}

/// How surfaces that combine workspaces name each one
/// ([[RFC-0012:C-WORKSPACES]]): the root's directory name, extended by as
/// many parent directories as it takes to tell same-named roots apart.
pub(super) fn labels(roots: &[&Path]) -> Vec<String> {
    let suffix = |root: &Path, depth: usize| {
        let components = root
            .components()
            .filter_map(|component| match component {
                std::path::Component::Normal(name) => Some(name.to_string_lossy()),
                _ => None,
            })
            .collect::<Vec<_>>();
        let start = components.len().saturating_sub(depth);
        let label = components[start..].join("/");
        if start == 0 {
            root.display().to_string()
        } else {
            label
        }
    };
    let mut depths = vec![1_usize; roots.len()];
    loop {
        let labels = roots
            .iter()
            .zip(&depths)
            .map(|(root, depth)| suffix(root, *depth))
            .collect::<Vec<_>>();
        let mut extended = false;
        for (index, label) in labels.iter().enumerate() {
            let shared = labels.iter().filter(|other| *other == label).count() > 1;
            let deeper = roots[index].components().count() > depths[index];
            if shared && deeper {
                depths[index] += 1;
                extended = true;
            }
        }
        if !extended {
            return labels;
        }
    }
}

/// A short, stable path segment for a workspace root.
pub(super) fn workspace_id(root: &Path) -> String {
    let digest = Sha256::digest(root.as_os_str().as_encoded_bytes());
    digest
        .iter()
        .take(6)
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn spawn(root: &Path, interval: Duration) -> Feed {
    let (publish, latest) = watch::channel(None);
    let (refresh, requests) = mpsc::channel();
    let worker_root = root.to_path_buf();
    // The worker ends when its feed is dropped: the refresh channel closes.
    let spawned = std::thread::Builder::new()
        .name("inferlab-web-refresh".to_owned())
        .spawn(move || {
            let mut collector = Collector::new(interval);
            let mut number = 0_u64;
            let mut force = true;
            loop {
                let snapshot = collector.collect(&worker_root, None, force);
                let presentation = Presentation::from_snapshot(&snapshot);
                number += 1;
                publish.send_replace(Some(Arc::new(Generation {
                    number,
                    completed: Instant::now(),
                    snapshot,
                    presentation,
                })));
                force = match requests.recv_timeout(interval) {
                    Ok(()) => true,
                    Err(RecvTimeoutError::Timeout) => false,
                    Err(RecvTimeoutError::Disconnected) => return,
                };
            }
        });
    if let Err(error) = spawned {
        eprintln!(
            "warning: could not start the refresh worker for {}: {error}",
            root.display()
        );
    }
    Feed {
        id: workspace_id(root),
        root: root.to_path_buf(),
        latest,
        refresh,
    }
}

#[cfg(test)]
mod tests {
    use super::labels;
    use std::path::Path;

    #[test]
    fn same_named_roots_gain_parents_until_distinct() {
        assert_eq!(
            labels(&[
                Path::new("/srv/a/lab"),
                Path::new("/srv/b/lab"),
                Path::new("/srv/other")
            ]),
            ["a/lab", "b/lab", "other"]
        );
    }
}
