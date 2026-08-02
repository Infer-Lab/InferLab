//! Workspace discovery and root-plus-fragment composition. This is the only
//! loader that constructs and validates a `WorkspaceConfig` aggregate.

use super::catalog_validation::validate_workspace;
use super::definitions::{
    BenchDefinition, DEFAULT_LOCAL_FILE, EvalDefinition, ExternalImageDefinition, ImageDefinition,
    ModelDefinition, RecipeDefinition, ServerDefinition, StackDefinition, WORKSPACE_FILE,
    WORKSPACE_FRAGMENT_DIR, WorkloadSuiteDefinition, WorkspaceConfig,
};
use super::invalid;
use super::local::{LocalBindings, validate_local_bindings};
use super::realization::validate_pixi;
use super::source::{git_text, inspect_workspace, symlink_guard, workspace_mutations};
use super::state::{LoadedWorkspace, WorkspaceIdentity, WorkspaceSnapshot};
use crate::InferlabError;
use inferlab_protocol::ServeTopology;
use serde::Deserialize;
use std::collections::BTreeMap;
use std::ffi::OsStr;
use std::fs;
use std::path::{Path, PathBuf};

/// A workspace fragment under `.inferlab/workspace.d/*.toml`
/// ([[RFC-0002:C-WORKSPACE-AUTHORITY]]): the identifier-keyed sections of
/// [`WorkspaceConfig`] and nothing else. It reuses the very same section
/// definition types as the root, so the section shapes have one authority;
/// this struct only re-lists which sections a fragment may carry. It omits
/// `schema_version` (and any future workspace-global scalar) deliberately —
/// those live only in the root file, and a fragment declaring one is rejected
/// before deserialization here so the operator gets a message naming the
/// fragment rather than a bare serde error.
#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct WorkspaceFragment {
    #[serde(default)]
    models: BTreeMap<String, ModelDefinition>,
    #[serde(default)]
    stacks: BTreeMap<String, StackDefinition>,
    #[serde(default)]
    servers: BTreeMap<String, ServerDefinition>,
    #[serde(default)]
    evals: BTreeMap<String, EvalDefinition>,
    #[serde(default)]
    benches: BTreeMap<String, BenchDefinition>,
    #[serde(default)]
    workload_suites: BTreeMap<String, WorkloadSuiteDefinition>,
    #[serde(default)]
    recipes: BTreeMap<String, RecipeDefinition>,
    #[serde(default)]
    images: BTreeMap<String, ImageDefinition>,
    #[serde(default)]
    external_images: BTreeMap<String, ExternalImageDefinition>,
}
/// Lightweight projection of the same Git revision and dirty authority used
/// by resolved execution snapshots. Runtime records, observations, the local
/// binding file, caches, and scratchpads remain outside source identity.
pub(crate) fn workspace_identity(root: &Path) -> Result<WorkspaceIdentity, InferlabError> {
    let exclusions = [
        PathBuf::from(DEFAULT_LOCAL_FILE),
        PathBuf::from(".inferlab/cache"),
        PathBuf::from(".inferlab/records"),
        PathBuf::from(".inferlab/runtime"),
        PathBuf::from(".inferlab/scratchpads"),
    ];
    Ok(WorkspaceIdentity {
        revision: git_text(root, &["rev-parse", "HEAD"])?,
        dirty: !workspace_mutations(root, &exclusions)?.is_empty(),
    })
}

pub fn discover_workspace(explicit: Option<&Path>) -> Result<PathBuf, InferlabError> {
    if let Some(path) = explicit {
        let root = if path.ends_with(WORKSPACE_FILE) {
            path.parent()
                .and_then(Path::parent)
                .map(Path::to_path_buf)
                .ok_or_else(|| InferlabError::InvalidConfig {
                    message: format!("invalid workspace file path {}", path.display()),
                })?
        } else {
            path.to_path_buf()
        };
        return canonicalize_root(root);
    }

    let start = std::env::current_dir().map_err(|source| InferlabError::Read {
        path: PathBuf::from("."),
        source,
    })?;
    for candidate in start.ancestors() {
        if candidate.join(WORKSPACE_FILE).is_file() {
            return canonicalize_root(candidate.to_path_buf());
        }
    }
    Err(InferlabError::WorkspaceNotFound { start })
}

pub fn load_workspace(
    root: PathBuf,
    local: Option<&Path>,
) -> Result<LoadedWorkspace, InferlabError> {
    // Resolved before the committed configuration loads: a fresh checkout
    // missing this git-ignored file deserves that guidance as the first
    // error a new operator sees, ahead of any workspace-config problem.
    let local_path = local
        .map(Path::to_path_buf)
        .unwrap_or_else(|| root.join(DEFAULT_LOCAL_FILE));
    let local_path = match fs::canonicalize(&local_path) {
        Ok(path) => path,
        // The first file a new operator is missing deserves guidance, not a
        // bare OS error: name what the file is for and the alternative.
        Err(source) if source.kind() == std::io::ErrorKind::NotFound => {
            return Err(InferlabError::InvalidConfig {
                message: format!(
                    "local bindings not found at {}: this git-ignored file supplies the \
                     machine-private facts recipes resolve against (machines, devices, \
                     model locators, launch access); create it, or select another file \
                     with --local <FILE>",
                    local_path.display()
                ),
            });
        }
        Err(source) => {
            return Err(InferlabError::Read {
                path: local_path,
                source,
            });
        }
    };
    let config = load_workspace_config(&root)?;
    let bindings: LocalBindings = load_toml(&local_path)?;
    validate_local_bindings(&bindings)?;
    let snapshot = inspect_workspace(&root, &local_path, &config)?;
    Ok(LoadedWorkspace {
        root,
        config,
        local: bindings,
        snapshot,
    })
}

/// Load and validate the committed workspace configuration alone, without
/// the machine-private local bindings `load_workspace` also requires. Serves
/// callers that only need declared facts — environment identifiers, for
/// instance — before an operator has bound this machine's local facts
/// ([[RFC-0002:C-PIXI-ENVIRONMENT-LIFECYCLE]]).
pub fn load_workspace_config(root: &Path) -> Result<WorkspaceConfig, InferlabError> {
    // The shared parent of WORKSPACE_FILE and WORKSPACE_FRAGMENT_DIR: a
    // symlinked `.inferlab` would route every final-node guard below through
    // the link, so the intermediate component is guarded first.
    symlink_guard(&root.join(".inferlab"), ".inferlab")?;
    let workspace_path = root.join(WORKSPACE_FILE);
    symlink_guard(&workspace_path, WORKSPACE_FILE)?;
    let mut config: WorkspaceConfig = load_toml(&workspace_path)?;
    compose_workspace_fragments(root, &mut config)?;
    validate_workspace(root, &config)?;
    validate_pixi(root, &config)?;
    Ok(config)
}

pub fn workspace_summary(config: &WorkspaceConfig) -> String {
    let mut output = format!("workspace schema {}\n", config.schema_version);
    push_catalog_section(
        &mut output,
        "stacks",
        config.stacks.iter().map(|(id, stack)| {
            format!(
                "{id} (integration: {}, pixi: {})",
                stack.integration, stack.pixi_environment
            )
        }),
    );
    push_catalog_section(
        &mut output,
        "models",
        config
            .models
            .iter()
            .map(|(id, model)| format!("{id} (served name: {})", model.served_name)),
    );
    push_catalog_section(
        &mut output,
        "servers",
        config.servers.iter().map(|(id, server)| {
            let cases = if server.cases.is_empty() {
                "none".to_owned()
            } else {
                server.cases.keys().cloned().collect::<Vec<_>>().join(", ")
            };
            let selection = case_selection_label(server);
            format!(
                "{id} (stack: {}, model: {}, topology: {}, cases: {cases}, selection: {selection})",
                server.stack,
                server.model,
                topology_label(server.topology)
            )
        }),
    );
    push_catalog_section(&mut output, "evals", config.evals.keys().cloned());
    push_catalog_section(&mut output, "benches", config.benches.keys().cloned());
    push_catalog_section(
        &mut output,
        "workload suites",
        config.workload_suites.iter().map(|(id, suite)| {
            let gate = suite.gate.as_deref().unwrap_or("none");
            format!(
                "{id} (evals: [{}], benches: [{}], gate: {gate})",
                suite.evals.join(", "),
                suite.benches.join(", ")
            )
        }),
    );
    push_catalog_section(
        &mut output,
        "recipes",
        config.recipes.iter().map(|(id, recipe)| {
            format!(
                "{id} (server: {}, workload suite: {})",
                recipe.server, recipe.workload_suite
            )
        }),
    );
    output
}

fn push_catalog_section(
    output: &mut String,
    label: &str,
    values: impl IntoIterator<Item = String>,
) {
    output.push('\n');
    output.push_str(label);
    output.push_str(":\n");
    let mut empty = true;
    for value in values {
        empty = false;
        output.push_str("  ");
        output.push_str(&value);
        output.push('\n');
    }
    if empty {
        output.push_str("  (none)\n");
    }
}

fn case_selection_label(server: &ServerDefinition) -> String {
    if let Some(default) = &server.default_case {
        format!("default {default}")
    } else if server.cases.len() == 1 {
        server.cases.keys().next().map_or_else(
            || "base server".to_owned(),
            |case| format!("sole case {case}"),
        )
    } else {
        "base server".to_owned()
    }
}

const fn topology_label(topology: ServeTopology) -> &'static str {
    match topology {
        ServeTopology::Single => "single",
        ServeTopology::PrefillDecode => "prefill-decode",
    }
}

pub(crate) fn snapshot_workspace(
    root: &Path,
    config: &WorkspaceConfig,
) -> Result<WorkspaceSnapshot, InferlabError> {
    inspect_workspace(root, &root.join(DEFAULT_LOCAL_FILE), config)
}

fn canonicalize_root(root: PathBuf) -> Result<PathBuf, InferlabError> {
    if !root.join(WORKSPACE_FILE).is_file() {
        return Err(InferlabError::WorkspaceNotFound { start: root });
    }
    fs::canonicalize(&root).map_err(|source| InferlabError::Read { path: root, source })
}

fn load_toml<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T, InferlabError> {
    let content = fs::read_to_string(path).map_err(|source| InferlabError::Read {
        path: path.to_path_buf(),
        source,
    })?;
    toml::from_str(&content).map_err(|source| InferlabError::ParseToml {
        path: path.to_path_buf(),
        source,
    })
}

/// Compose fragments under `.inferlab/workspace.d/*.toml` into the root
/// configuration as a disjoint union of identifier-keyed definitions
/// ([[RFC-0002:C-WORKSPACE-AUTHORITY]]). File organization creates no implicit
/// precedence: the union is disjoint by construction, and an identifier
/// declared by two files is a load error naming both. Fragments are visited in
/// sorted filename order so a collision reports the same pair of files however
/// the filesystem enumerates the directory. A workspace with no
/// `workspace.d/` directory (or an empty one) composes to the root config
/// unchanged.
fn compose_workspace_fragments(
    root: &Path,
    config: &mut WorkspaceConfig,
) -> Result<(), InferlabError> {
    let fragment_dir = root.join(WORKSPACE_FRAGMENT_DIR);
    symlink_guard(&fragment_dir, WORKSPACE_FRAGMENT_DIR)?;
    let entries = match fs::read_dir(&fragment_dir) {
        Ok(entries) => entries,
        Err(source) if source.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(source) => {
            return Err(InferlabError::Read {
                path: PathBuf::from(WORKSPACE_FRAGMENT_DIR),
                source,
            });
        }
    };

    // Only regular `*.toml` files are fragments; a subdirectory or any other
    // extension under workspace.d is ignored, while a symlinked `*.toml` is
    // rejected rather than followed or dropped
    // ([[RFC-0002:C-WORKSPACE-AUTHORITY]]). Sorting by file name makes the
    // merge — and thus every collision error — order-independent.
    let mut fragments: Vec<PathBuf> = Vec::new();
    for entry in entries {
        let entry = entry.map_err(|source| InferlabError::Read {
            path: PathBuf::from(WORKSPACE_FRAGMENT_DIR),
            source,
        })?;
        let path = entry.path();
        if path.extension() != Some(OsStr::new("toml")) {
            continue;
        }
        let file_type = entry.file_type().map_err(|source| InferlabError::Read {
            path: path.clone(),
            source,
        })?;
        if file_type.is_symlink() {
            return invalid(format!(
                "workspace fragment {WORKSPACE_FRAGMENT_DIR}/{} must be a regular \
                 filesystem entry, not a symbolic link; the workspace source digest \
                 records link text rather than target content",
                path.file_name().unwrap_or_default().to_string_lossy()
            ));
        }
        if file_type.is_file() {
            fragments.push(path);
        }
    }
    fragments.sort();

    // (section, identifier) -> the workspace-relative path of the FRAGMENT
    // that declared it; root declarations need no entry because the collision
    // check consults the composed map and attributes unknown declarers to the
    // root file. Load-local only; it never reaches the workspace struct or
    // any record.
    let mut provenance: BTreeMap<(&'static str, String), String> = BTreeMap::new();

    for path in fragments {
        let relative = format!(
            "{WORKSPACE_FRAGMENT_DIR}/{}",
            path.file_name().unwrap_or_default().to_string_lossy()
        );
        let content = fs::read_to_string(&path).map_err(|source| InferlabError::Read {
            path: PathBuf::from(&relative),
            source,
        })?;
        // A fragment may not carry `schema_version` or any workspace-global
        // scalar; those live only in the root file. Detect it on the parsed
        // table so the operator sees the fragment named, not a serde error
        // about an unknown field.
        let table: toml::Table =
            toml::from_str(&content).map_err(|source| InferlabError::ParseToml {
                path: PathBuf::from(&relative),
                source,
            })?;
        if table.contains_key("schema_version") {
            return invalid(format!(
                "workspace fragment {relative} declares schema_version, which lives only in the \
                 root workspace file {WORKSPACE_FILE}"
            ));
        }
        // Typed parsing re-reads the source text rather than converting the
        // already-parsed table: `toml::from_str` keeps line/column spans, so a
        // type error or unknown field names its position like the root file.
        let fragment: WorkspaceFragment =
            toml::from_str(&content).map_err(|source| InferlabError::ParseToml {
                path: PathBuf::from(&relative),
                source,
            })?;
        merge_fragment(config, &mut provenance, fragment, &relative)?;
    }
    Ok(())
}

/// Fold one parsed fragment into the composed config, rejecting any identifier
/// already declared by an earlier-visited file (the root or a lower-sorted
/// fragment) with an error naming both files, the section, and the identifier.
fn merge_fragment(
    config: &mut WorkspaceConfig,
    provenance: &mut BTreeMap<(&'static str, String), String>,
    fragment: WorkspaceFragment,
    file: &str,
) -> Result<(), InferlabError> {
    merge_section(
        &mut config.models,
        fragment.models,
        "model",
        file,
        provenance,
    )?;
    merge_section(
        &mut config.stacks,
        fragment.stacks,
        "stack",
        file,
        provenance,
    )?;
    merge_section(
        &mut config.servers,
        fragment.servers,
        "server",
        file,
        provenance,
    )?;
    merge_section(&mut config.evals, fragment.evals, "eval", file, provenance)?;
    merge_section(
        &mut config.benches,
        fragment.benches,
        "bench",
        file,
        provenance,
    )?;
    merge_section(
        &mut config.workload_suites,
        fragment.workload_suites,
        "workload suite",
        file,
        provenance,
    )?;
    merge_section(
        &mut config.recipes,
        fragment.recipes,
        "recipe",
        file,
        provenance,
    )?;
    merge_section(
        &mut config.images,
        fragment.images,
        "image",
        file,
        provenance,
    )?;
    merge_section(
        &mut config.external_images,
        fragment.external_images,
        "external image",
        file,
        provenance,
    )
}

/// Insert one section's definitions into the composed map, rejecting a
/// collision against whichever file already declared the identifier. The
/// check consults the composed map itself, so a root-declared identifier
/// collides without any seeding step: an identifier present in the map but
/// absent from `provenance` was necessarily declared by the root file.
fn merge_section<T>(
    target: &mut BTreeMap<String, T>,
    incoming: BTreeMap<String, T>,
    label: &'static str,
    file: &str,
    provenance: &mut BTreeMap<(&'static str, String), String>,
) -> Result<(), InferlabError> {
    for (id, definition) in incoming {
        if target.contains_key(&id) {
            let existing = provenance
                .get(&(label, id.clone()))
                .map(String::as_str)
                .unwrap_or(WORKSPACE_FILE);
            return invalid(format!(
                "{label} {id:?} is declared by both {existing} and {file}"
            ));
        }
        provenance.insert((label, id.clone()), file.to_owned());
        target.insert(id, definition);
    }
    Ok(())
}
