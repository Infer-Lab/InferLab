//! Workspace discovery and root-plus-fragment composition. This is the only
//! loader that constructs and validates a `WorkspaceConfig` aggregate.

use super::catalog_validation::validate_workspace;
use super::definitions::{
    BenchDefinition, DEFAULT_LOCAL_FILE, EvalDefinition, ExternalImageDefinition, ImageDefinition,
    ImplicitCaseSelection, ModelDefinition, RecipeDefinition, ServerDefinition, StackDefinition,
    WORKSPACE_FILE, WORKSPACE_FRAGMENT_DIR, WORKSPACE_LOCAL_FRAGMENT_DIR, WorkloadSuiteDefinition,
    WorkspaceConfig, deserialize_defaulted_bench_definitions,
};
use super::invalid;
use super::local::{LocalBindings, validate_local_bindings, validate_recipe_placements};
use super::realization::validate_pixi;
use super::source::{git_text, inspect_workspace, symlink_guard, workspace_mutations};
use super::state::{LoadedWorkspace, LocalDefinitionFile, WorkspaceIdentity, WorkspaceSnapshot};
use crate::InferlabError;
use crate::record::STATE_DIR;
use inferlab_protocol::ServeTopology;
use serde::Deserialize;
use sha2::Digest;
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
    #[serde(default, deserialize_with = "deserialize_defaulted_bench_definitions")]
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
/// binding file, caches, and scratchpads remain outside source identity. The
/// exclusion set comes from the shared constructor in `source` so this
/// verdict can never diverge from the snapshot's.
pub(crate) fn workspace_identity(
    root: &Path,
    local: Option<&Path>,
) -> Result<WorkspaceIdentity, InferlabError> {
    // The effective local bindings file, resolved as a run would resolve it:
    // an explicit selection wins, otherwise the default in-root file. A
    // missing file still yields its declared path for exclusion purposes.
    let local_path = local
        .map(Path::to_path_buf)
        .unwrap_or_else(|| root.join(DEFAULT_LOCAL_FILE));
    let local_path = fs::canonicalize(&local_path).unwrap_or(local_path);
    let exclusions = super::source::source_exclusions(root, &local_path);
    Ok(WorkspaceIdentity {
        revision: git_text(root, &["rev-parse", "HEAD"])?,
        dirty: !workspace_mutations(root, &exclusions)?.is_empty(),
    })
}

pub(crate) fn discover_workspace(explicit: Option<&Path>) -> Result<PathBuf, InferlabError> {
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

pub(crate) fn load_workspace(
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
    validate_recipe_placements(&bindings, &config)?;
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
pub(crate) fn load_workspace_config(root: &Path) -> Result<WorkspaceConfig, InferlabError> {
    // The shared parent of WORKSPACE_FILE and WORKSPACE_FRAGMENT_DIR: a
    // symlinked `.inferlab` would route every final-node guard below through
    // the link, so the intermediate component is guarded first.
    symlink_guard(&root.join(STATE_DIR), STATE_DIR)?;
    let workspace_path = root.join(WORKSPACE_FILE);
    symlink_guard(&workspace_path, WORKSPACE_FILE)?;
    let mut config: WorkspaceConfig = load_toml(&workspace_path)?;
    compose_workspace_fragments(root, &mut config)?;
    validate_workspace(root, &config)?;
    validate_pixi(root, &config)?;
    Ok(config)
}

pub(crate) fn workspace_summary(config: &WorkspaceConfig) -> String {
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
    match server.implicit_case_selection() {
        ImplicitCaseSelection::Default(case) => format!("default {case}"),
        ImplicitCaseSelection::Sole(case) => format!("sole case {case}"),
        ImplicitCaseSelection::Base => "base server".to_owned(),
        // Resolution rejects this selection; name it instead of claiming a
        // base server that would fail to resolve.
        ImplicitCaseSelection::Ambiguous => "ambiguous (no default case)".to_owned(),
    }
}

const fn topology_label(topology: ServeTopology) -> &'static str {
    match topology {
        ServeTopology::Single => "single",
        ServeTopology::PrefillDecode => "prefill-decode",
    }
}

/// Load the workspace for a workflow that composes local definitions —
/// `serve start` and `recipe run` — after the committed workspace stands
/// validated on its own, so no committed definition can depend on one
/// ([[RFC-0002:C-WORKSPACE-AUTHORITY]]).
pub(crate) fn load_workspace_with_local_definitions(
    root: PathBuf,
    local: Option<&Path>,
) -> Result<LoadedWorkspace, InferlabError> {
    let mut workspace = load_workspace(root, local)?;
    let local_definitions =
        compose_validated_local_definitions(&workspace.root, &mut workspace.config)?;
    record_local_definitions(&mut workspace.snapshot, local_definitions);
    Ok(workspace)
}

/// The committed configuration plus local definitions, for a manual Bench,
/// which resolves without local bindings.
pub(crate) fn load_workspace_config_with_local_definitions(
    root: &Path,
) -> Result<(WorkspaceConfig, WorkspaceSnapshot), InferlabError> {
    let mut config = load_workspace_config(root)?;
    let local_definitions = compose_validated_local_definitions(root, &mut config)?;
    let mut snapshot = snapshot_workspace(root, &config)?;
    record_local_definitions(&mut snapshot, local_definitions);
    Ok((config, snapshot))
}

fn compose_validated_local_definitions(
    root: &Path,
    config: &mut WorkspaceConfig,
) -> Result<Vec<LocalDefinitionFile>, InferlabError> {
    let local_definitions = compose_local_definitions(root, config)?;
    if !local_definitions.is_empty() {
        validate_workspace(root, config)?;
    }
    Ok(local_definitions)
}

/// A run that composed local definitions names them, and its workspace
/// revision alone can no longer reproduce it ([[RFC-0005:C-EVIDENCE]]).
fn record_local_definitions(snapshot: &mut WorkspaceSnapshot, files: Vec<LocalDefinitionFile>) {
    if !files.is_empty() {
        snapshot.revision_reproducible = false;
    }
    snapshot.local_definitions = files;
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

/// Best-effort read of the machine-private local bindings for callers whose
/// operation must not depend on them: absent, unreadable, or invalid bindings
/// yield `None` instead of an error ([[RFC-0002:C-ADHOC-EXECUTION]]). A file
/// that exists but fails to load or validate warns on stderr, so a broken
/// binding never silently disables the ad-hoc device projection.
pub(crate) fn soft_local_bindings(root: &Path) -> Option<LocalBindings> {
    let path = root.join(DEFAULT_LOCAL_FILE);
    if !path.exists() {
        return None;
    }
    let loaded = load_toml(&path).and_then(|bindings: LocalBindings| {
        validate_local_bindings(&bindings)?;
        Ok(bindings)
    });
    match loaded {
        Ok(bindings) => Some(bindings),
        Err(error) => {
            eprintln!(
                "warning: ignoring local bindings at {}: {error}",
                path.display()
            );
            None
        }
    }
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
    compose_fragment_dir(root, config, WORKSPACE_FRAGMENT_DIR, false).map(|_| ())
}

/// Compose the git-ignored local definitions under
/// `.inferlab/workspace.local.d/*.toml` into an already composed and
/// validated committed workspace: the same disjoint union, restricted to
/// measurement and serving definitions, outside source identity, and named
/// with its digests for the run's records ([[RFC-0002:C-WORKSPACE-AUTHORITY]]).
fn compose_local_definitions(
    root: &Path,
    config: &mut WorkspaceConfig,
) -> Result<Vec<LocalDefinitionFile>, InferlabError> {
    let directory = root.join(WORKSPACE_LOCAL_FRAGMENT_DIR);
    if !directory.exists() {
        return Ok(Vec::new());
    }
    // Inside source identity the directory would dirty the workspace and
    // change its digest, so it must be ignored the way local.toml is.
    let ignored = std::process::Command::new("git")
        .current_dir(root)
        .args(["check-ignore", "-q", "--", WORKSPACE_LOCAL_FRAGMENT_DIR])
        .status()
        .map_err(|source| InferlabError::Read {
            path: PathBuf::from(WORKSPACE_LOCAL_FRAGMENT_DIR),
            source,
        })?;
    if !ignored.success() {
        return invalid(format!(
            "local workspace definitions under {WORKSPACE_LOCAL_FRAGMENT_DIR}/ must be \
             git-ignored; add `{WORKSPACE_LOCAL_FRAGMENT_DIR}/` to .gitignore"
        ));
    }
    // A file force-added despite the ignore rule is inside source identity.
    let tracked = git_text(root, &["ls-files", "--", WORKSPACE_LOCAL_FRAGMENT_DIR])?;
    if !tracked.trim().is_empty() {
        return invalid(format!(
            "local workspace definitions must be untracked, but git tracks {}; remove them \
             from the index with `git rm --cached`",
            tracked.trim().lines().collect::<Vec<_>>().join(", ")
        ));
    }
    compose_fragment_dir(root, config, WORKSPACE_LOCAL_FRAGMENT_DIR, true)
}

/// Compose every regular `*.toml` fragment of one directory, in sorted name
/// order. Local fragments may carry only models, servers, Evals, Benches,
/// workload suites, and recipes, and each is returned with its digest.
fn compose_fragment_dir(
    root: &Path,
    config: &mut WorkspaceConfig,
    fragment_dir_name: &str,
    local: bool,
) -> Result<Vec<LocalDefinitionFile>, InferlabError> {
    let mut composed = Vec::new();
    let fragment_dir = root.join(fragment_dir_name);
    symlink_guard(&fragment_dir, fragment_dir_name)?;
    let entries = match fs::read_dir(&fragment_dir) {
        Ok(entries) => entries,
        Err(source) if source.kind() == std::io::ErrorKind::NotFound => return Ok(composed),
        Err(source) => {
            return Err(InferlabError::Read {
                path: PathBuf::from(fragment_dir_name),
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
            path: PathBuf::from(fragment_dir_name),
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
                "workspace fragment {fragment_dir_name}/{} must be a regular \
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
            "{fragment_dir_name}/{}",
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
        if local {
            for (section, declared) in [
                ("stacks", !fragment.stacks.is_empty()),
                ("images", !fragment.images.is_empty()),
                ("external_images", !fragment.external_images.is_empty()),
            ] {
                if declared {
                    return invalid(format!(
                        "local workspace definition {relative} declares {section}; local \
                         definitions may declare only models, servers, evals, benches, \
                         workload_suites, and recipes"
                    ));
                }
            }
            composed.push(LocalDefinitionFile {
                path: relative.clone(),
                sha256: base16ct::lower::encode_string(&sha2::Sha256::digest(content.as_bytes())),
            });
        }
        merge_fragment(config, &mut provenance, fragment, &relative)?;
    }
    Ok(composed)
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
