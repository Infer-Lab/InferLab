//! The persisted workspace registry ([[RFC-0012:C-WORKSPACES]]): canonical
//! workspace roots in one user-level state file, replaced atomically.

use crate::InferlabError;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct RegistryFile {
    workspaces: Vec<PathBuf>,
}

pub(super) struct Registry {
    path: PathBuf,
    roots: Vec<PathBuf>,
}

/// Whether a directory is a workspace root: it holds the workspace manifest,
/// whether or not its definitions load.
pub(super) fn is_workspace_root(directory: &Path) -> bool {
    directory.join(crate::workspace::WORKSPACE_FILE).is_file()
}

impl Registry {
    /// The registry under `XDG_STATE_HOME`, or `~/.local/state` when unset. A
    /// missing file is an empty registry; a malformed one fails the start and
    /// is left as it is.
    pub(super) fn load() -> Result<Self, InferlabError> {
        let path = location()?;
        let roots = read(&path)?;
        Ok(Self { path, roots })
    }

    /// Re-read the file so a change starts from what another console may
    /// have written since.
    fn reload(&mut self) -> Result<(), InferlabError> {
        self.roots = read(&self.path)?;
        Ok(())
    }

    pub(super) fn roots(&self) -> &[PathBuf] {
        &self.roots
    }

    /// Register a workspace root by its canonical form; a root already
    /// registered is left as it is.
    pub(super) fn register(&mut self, directory: &Path) -> Result<(), RegisterError> {
        let root = directory.canonicalize().map_err(|error| {
            RegisterError::Rejected(format!("{}: {error}", directory.display()))
        })?;
        if !is_workspace_root(&root) {
            return Err(RegisterError::Rejected(format!(
                "{} is not a workspace root: it has no {}",
                root.display(),
                crate::workspace::WORKSPACE_FILE
            )));
        }
        self.reload().map_err(RegisterError::Persist)?;
        if self.roots.contains(&root) {
            return Ok(());
        }
        self.roots.push(root);
        self.persist().map_err(RegisterError::Persist)
    }

    /// Unregister a root; its files are untouched.
    pub(super) fn unregister(&mut self, root: &Path) -> Result<(), InferlabError> {
        self.reload()?;
        let before = self.roots.len();
        self.roots.retain(|registered| registered != root);
        if self.roots.len() == before {
            return Ok(());
        }
        self.persist()
    }

    fn persist(&self) -> Result<(), InferlabError> {
        crate::atomic_json::write(
            &self.path,
            &RegistryFile {
                workspaces: self.roots.clone(),
            },
        )
        .map_err(|error| InferlabError::WebRegistry {
            path: self.path.clone(),
            message: format!("could not be written: {error:?}"),
        })
    }
}

pub(super) enum RegisterError {
    /// The operator chose a directory that is not a workspace root.
    Rejected(String),
    Persist(InferlabError),
}

/// The registered roots in the file; a missing file registers none.
fn read(path: &Path) -> Result<Vec<PathBuf>, InferlabError> {
    Ok(match std::fs::read(path) {
        Ok(bytes) => {
            serde_json::from_slice::<RegistryFile>(&bytes)
                .map_err(|error| InferlabError::WebRegistry {
                    path: path.to_path_buf(),
                    message: format!("is malformed: {error}"),
                })?
                .workspaces
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Vec::new(),
        Err(error) => {
            return Err(InferlabError::WebRegistry {
                path: path.to_path_buf(),
                message: format!("could not be read: {error}"),
            });
        }
    })
}

fn location() -> Result<PathBuf, InferlabError> {
    let state = match std::env::var_os("XDG_STATE_HOME").filter(|value| !value.is_empty()) {
        Some(state) => PathBuf::from(state),
        None => std::env::var_os("HOME")
            .map(|home| PathBuf::from(home).join(".local/state"))
            .ok_or_else(|| InferlabError::WebRegistry {
                path: PathBuf::from("inferlab/web/workspaces.json"),
                message: "has no location: neither XDG_STATE_HOME nor HOME is set".to_owned(),
            })?,
    };
    Ok(state.join("inferlab/web/workspaces.json"))
}
