//! Source identity, Git evidence, and symlink containment for workspace
//! snapshots.

use super::definitions::WorkspaceConfig;
use super::invalid;
use super::state::WorkspaceSnapshot;
use crate::InferlabError;
use std::collections::{BTreeMap, BTreeSet};
use std::ffi::{OsStr, OsString};
use std::fs;
use std::path::{Component, Path, PathBuf};
use std::process::Command;

/// [[RFC-0002:C-WORKSPACE-AUTHORITY]]: every symbolic link effectively
/// present in the digested worktree must carry a target that resolves to
/// identity-covered workspace content. The walk covers the whole digested
/// worktree rather than the declared stack source subtrees because the digest
/// pathspec covers the root: a link outside every stack source still enters
/// identity as link text, so every intermediate link is enumerated and
/// judged on its own by construction. The walk reads the filesystem rather
/// than the git index because untracked and ignored links — and links
/// replacing tracked entries — carry the same digest blindness as tracked
/// ones; tracking state affects dirtiness, not containment. Resolution
/// stays lexical because physical resolution would depend on machine state;
/// a target resolving onto or through an enumerated link is judged against
/// its link-resolved destination because git refuses pathspecs beyond a
/// symbolic link.
pub(super) fn reject_uncovered_worktree_links(
    root: &Path,
    config: &WorkspaceConfig,
    exclusions: &[PathBuf],
) -> Result<(), InferlabError> {
    let links = collect_digested_worktree_symlinks(root, exclusions)?;
    // Phase one judges every link's own containment, so an escaping
    // intermediate is named as the root cause before any link resolving
    // through it is judged. The map carries each link's visibility because
    // substitution is defined only through digest-visible links
    // ([[RFC-0002:C-WORKSPACE-AUTHORITY]]).
    let mut link_map: BTreeMap<PathBuf, (PathBuf, bool)> = BTreeMap::new();
    let mut direct = Vec::new();
    for (link, target) in &links {
        let scope = link_scope(config, link);
        // A git-ignored link is machine-local state no identity claim
        // covers (editable installs plant absolute links to in-root
        // content), so containment alone binds it; a digest-visible
        // link must also have an identity-covered target.
        let machine_local = link_is_git_ignored(root, link)?;
        link_map.insert(link.clone(), (target.clone(), machine_local));
        let resolved = if target.is_absolute() {
            if !machine_local {
                return invalid(format!(
                    "{scope} targets absolute path {}; the workspace source digest records \
                     link text rather than target content",
                    target.display(),
                ));
            }
            target
                .strip_prefix(root)
                .ok()
                .and_then(|in_root| lexical_resolution(Path::new(""), in_root))
        } else {
            lexical_resolution(link.parent().unwrap_or(Path::new("")), target)
        };
        let Some(resolved) = resolved else {
            let judgement = if target.is_absolute() {
                "resolves"
            } else {
                "lexically resolves"
            };
            return invalid(format!(
                "{scope} targets {}, which {judgement} outside the workspace root; the \
                 workspace source digest records link text rather than target content",
                target.display(),
            ));
        };
        if contains_git_component(&resolved) {
            return invalid(format!(
                "{scope} targets {}, which resolves into git metadata at {}; the workspace \
                 source digest records link text rather than target content",
                target.display(),
                resolved.display(),
            ));
        }
        if !machine_local {
            direct.push((scope, link, target, resolved));
        }
    }
    // Phase two judges the link-resolved destination: substitution through
    // the enumerated links keeps a benign in-root chain judgeable and stops
    // a covered-looking path from riding another link's target
    // ([[RFC-0002:C-WORKSPACE-AUTHORITY]]).
    let mut ignore_candidates = Vec::new();
    for (scope, link, target, resolved) in direct {
        let resolved = resolve_through_links(root, &link_map, resolved, &scope, target)?;
        if contains_git_component(&resolved) {
            return invalid(format!(
                "{scope} targets {}, which resolves into git metadata at {}; the workspace \
                 source digest records link text rather than target content",
                target.display(),
                resolved.display(),
            ));
        }
        if let Some(exclusion) = exclusions
            .iter()
            .find(|exclusion| resolved.starts_with(exclusion))
        {
            return invalid(format!(
                "{scope} targets {}, which resolves into the workspace source exclusion {}; \
                 the workspace source digest records link text rather than target content",
                target.display(),
                exclusion.display(),
            ));
        }
        ignore_candidates.push((scope, link.clone(), target.clone(), resolved));
    }
    reject_ignored_targets(root, ignore_candidates)
}

/// Rejection evidence names the declaring stack when one covers the
/// link ([[RFC-0002:C-WORKSPACE-AUTHORITY]]).
pub(super) fn link_scope(config: &WorkspaceConfig, link: &Path) -> String {
    let stack = config.stacks.iter().find_map(|(name, stack)| {
        stack
            .source_paths
            .iter()
            .any(|path| link.starts_with(path))
            .then_some(name)
    });
    match stack {
        Some(name) => format!("stack {name:?} source symlink {}", link.display()),
        None => format!("workspace symlink {}", link.display()),
    }
}

pub(super) fn contains_git_component(path: &Path) -> bool {
    path.components()
        .any(|component| component.as_os_str() == ".git")
}

/// Substitute enumerated link text into `resolved` until no enumerated link
/// component remains, rejecting substitution chains that revisit a link
/// (a cycle) or step outside the root ([[RFC-0002:C-WORKSPACE-AUTHORITY]]).
pub(super) fn resolve_through_links(
    root: &Path,
    link_map: &BTreeMap<PathBuf, (PathBuf, bool)>,
    mut resolved: PathBuf,
    scope: &str,
    target: &Path,
) -> Result<PathBuf, InferlabError> {
    let mut visited = BTreeSet::new();
    loop {
        // The shortest link prefix substitutes first, mirroring component-
        // by-component path resolution.
        let mut prefix = PathBuf::new();
        let link_prefix = resolved.components().find_map(|component| {
            prefix.push(component);
            link_map.contains_key(&prefix).then(|| prefix.clone())
        });
        let Some(link_prefix) = link_prefix else {
            return Ok(resolved);
        };
        if !visited.insert(link_prefix.clone()) {
            return invalid(format!(
                "{scope} targets {}, which resolves through a symbolic-link cycle at {}; \
                 the workspace source digest records link text rather than target content",
                target.display(),
                link_prefix.display(),
            ));
        }
        let rest = resolved
            .strip_prefix(&link_prefix)
            .unwrap_or(Path::new(""))
            .to_path_buf();
        let (link_target, machine_local) = &link_map[&link_prefix];
        // Substitution is defined only through digest-visible links: a
        // machine-local link's text is outside the recorded identity, so a
        // digest-visible resolution riding it could change effective content
        // under an unchanged digest ([[RFC-0002:C-WORKSPACE-AUTHORITY]]).
        if *machine_local {
            return invalid(format!(
                "{scope} targets {}, which resolves through the git-ignored link {}; the \
                 machine-local link text is outside the workspace source digest",
                target.display(),
                link_prefix.display(),
            ));
        }
        let base = if link_target.is_absolute() {
            link_target
                .strip_prefix(root)
                .ok()
                .and_then(|in_root| lexical_resolution(Path::new(""), in_root))
        } else {
            lexical_resolution(link_prefix.parent().unwrap_or(Path::new("")), link_target)
        };
        let Some(base) = base else {
            return invalid(format!(
                "{scope} targets {}, which resolves outside the workspace root through {}; \
                 the workspace source digest records link text rather than target content",
                target.display(),
                link_prefix.display(),
            ));
        };
        resolved = base.join(rest);
    }
}

/// Whether the link itself is git-ignored in its owning repository — with
/// the same tracked-overrides-pattern correction as the target verdict,
/// because a tracked link matching an ignore pattern is still digest-visible
/// and must keep the full coverage requirement.
pub(super) fn link_is_git_ignored(root: &Path, link: &Path) -> Result<bool, InferlabError> {
    let repo = owning_repo(root, link);
    let repo_dir = root.join(&repo);
    let relative = link.strip_prefix(&repo).unwrap_or(link);
    if !git_in(
        &repo_dir,
        ["check-ignore", "-q", "--", &path_text(relative)],
    )? {
        return Ok(false);
    }
    let tracked = git_in(
        &repo_dir,
        ["ls-files", "--error-unmatch", "--", &path_text(relative)],
    )?;
    Ok(!tracked)
}

/// Every symlink effectively present in the digested worktree, collected by
/// `lstat` without following links, skipping `.git` entries, the workspace
/// source exclusions, and git-ignored directories — machine-local trees the
/// digest cannot see and digest-visible links cannot target
/// ([[RFC-0002:C-WORKSPACE-AUTHORITY]]). The walk proceeds level by level so
/// ignored directories are pruned in one batched judgment per owning repo
/// before their (possibly enormous) contents are read; entries are sorted
/// per directory so rejection order is stable; unreadable directories are
/// the shape checks' problem, not this walk's.
pub(super) fn collect_digested_worktree_symlinks(
    root: &Path,
    exclusions: &[PathBuf],
) -> Result<Vec<(PathBuf, PathBuf)>, InferlabError> {
    let mut links = Vec::new();
    let mut frontier = vec![PathBuf::new()];
    while !frontier.is_empty() {
        let mut directories = Vec::new();
        for dir in frontier.drain(..) {
            let Ok(entries) = fs::read_dir(root.join(&dir)) else {
                continue;
            };
            let mut children: Vec<_> = entries.flatten().collect();
            children.sort_by_key(std::fs::DirEntry::file_name);
            for child in children {
                if child.file_name() == ".git" {
                    continue;
                }
                let relative = dir.join(child.file_name());
                if exclusions
                    .iter()
                    .any(|exclusion| relative.starts_with(exclusion))
                {
                    continue;
                }
                let Ok(file_type) = child.file_type() else {
                    continue;
                };
                if file_type.is_symlink() {
                    if let Ok(target) = fs::read_link(child.path()) {
                        links.push((relative, target));
                    }
                } else if file_type.is_dir() {
                    directories.push(relative);
                }
            }
        }
        frontier = retain_walked_directories(root, directories)?;
    }
    Ok(links)
}

/// Directories the walk descends into: everything not git-ignored, judged
/// in one `check-ignore --stdin` batch per owning repo. A flagged directory
/// still holding tracked content is kept — `check-ignore` matches patterns
/// without consulting the index, and tracked content stays digest-visible.
pub(super) fn retain_walked_directories(
    root: &Path,
    directories: Vec<PathBuf>,
) -> Result<Vec<PathBuf>, InferlabError> {
    if directories.is_empty() {
        return Ok(directories);
    }
    let mut groups: BTreeMap<PathBuf, Vec<usize>> = BTreeMap::new();
    for (index, directory) in directories.iter().enumerate() {
        groups
            .entry(owning_repo(root, directory))
            .or_default()
            .push(index);
    }
    let mut pruned = vec![false; directories.len()];
    for (repo, indexes) in groups {
        let repo_dir = root.join(&repo);
        let paths = indexes
            .iter()
            .map(|index| {
                directories[*index]
                    .strip_prefix(&repo)
                    .unwrap_or(&directories[*index])
                    .to_path_buf()
            })
            .collect::<Vec<_>>();
        let flagged = git_check_ignore_batch(&repo_dir, &paths)?;
        for (index, path) in indexes.iter().zip(&paths) {
            if !flagged.contains(path) {
                continue;
            }
            let tracked = git_in(
                &repo_dir,
                ["ls-files", "--error-unmatch", "--", &path_text(path)],
            )?;
            if !tracked {
                pruned[*index] = true;
            }
        }
    }
    Ok(directories
        .into_iter()
        .enumerate()
        .filter(|(index, _)| !pruned[*index])
        .map(|(_, directory)| directory)
        .collect())
}

/// The subset of `paths` git-ignore patterns flag, in one batched
/// `check-ignore --stdin -z` call. Exit 0 means some matched, exit 1 means
/// none did; anything else is a git failure.
pub(super) fn git_check_ignore_batch(
    repo_dir: &Path,
    paths: &[PathBuf],
) -> Result<BTreeSet<PathBuf>, InferlabError> {
    use std::io::Write as _;
    let mut child = Command::new("git")
        .current_dir(repo_dir)
        .args(["check-ignore", "--stdin", "-z"])
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .map_err(|source| InferlabError::Git {
            root: repo_dir.to_path_buf(),
            source: crate::error::GitError::Launch {
                operation: "git check-ignore --stdin".to_owned(),
                source,
            },
        })?;
    let mut input = Vec::new();
    for path in paths {
        input.extend_from_slice(path_text(path).as_bytes());
        input.push(0);
    }
    let mut stdin = child.stdin.take().ok_or_else(|| InferlabError::Git {
        root: repo_dir.to_path_buf(),
        source: crate::error::GitError::MissingStdin {
            operation: "git check-ignore".to_owned(),
        },
    })?;
    stdin
        .write_all(&input)
        .map_err(|source| InferlabError::Git {
            root: repo_dir.to_path_buf(),
            source: crate::error::GitError::WriteStdin {
                operation: "git check-ignore".to_owned(),
                source,
            },
        })?;
    drop(stdin);
    let output = child
        .wait_with_output()
        .map_err(|source| InferlabError::Git {
            root: repo_dir.to_path_buf(),
            source: crate::error::GitError::Wait {
                operation: "git check-ignore".to_owned(),
                source,
            },
        })?;
    match output.status.code() {
        Some(0 | 1) => Ok(output
            .stdout
            .split(|byte| *byte == 0)
            .filter(|entry| !entry.is_empty())
            .map(|entry| PathBuf::from(String::from_utf8_lossy(entry).into_owned()))
            .collect()),
        _ => Err(InferlabError::Git {
            root: repo_dir.to_path_buf(),
            source: crate::error::GitError::Exit {
                operation: "git check-ignore --stdin".to_owned(),
                status: output.status,
                diagnostics: String::from_utf8_lossy(&output.stderr).trim().to_owned(),
            },
        }),
    }
}

/// `target` resolved lexically against the root-relative `base` directory,
/// or `None` when any step climbs above the workspace root.
pub(super) fn lexical_resolution(base: &Path, target: &Path) -> Option<PathBuf> {
    let mut resolved = base.components().collect::<Vec<_>>();
    for component in target.components() {
        match component {
            Component::ParentDir => {
                resolved.pop()?;
            }
            Component::Normal(_) => resolved.push(component),
            Component::CurDir => {}
            Component::RootDir | Component::Prefix(_) => return None,
        }
    }
    Some(resolved.iter().collect())
}

/// Reject candidates whose resolved target is git-ignored, judged by the
/// target's owning repository (the nearest ancestor with a `.git` entry) so
/// submodule ignore rules govern submodule content. `git check-ignore`
/// matches patterns without consulting the index, so a flagged target is
/// re-checked for trackedness — a tracked file matching an ignore pattern is
/// still identity-covered. Dangling targets are judged too: an ignored
/// namespace fills with uncovered bytes later without another snapshot.
pub(super) fn reject_ignored_targets(
    root: &Path,
    mut candidates: Vec<(String, PathBuf, PathBuf, PathBuf)>,
) -> Result<(), InferlabError> {
    if candidates.is_empty() {
        return Ok(());
    }
    candidates.sort_by(|left, right| left.1.cmp(&right.1));
    let mut groups: BTreeMap<PathBuf, Vec<usize>> = BTreeMap::new();
    for (index, (_, _, _, resolved)) in candidates.iter().enumerate() {
        groups
            .entry(owning_repo(root, resolved))
            .or_default()
            .push(index);
    }
    for (repo, indexes) in groups {
        let repo_dir = root.join(&repo);
        let paths = indexes
            .iter()
            .map(|index| {
                candidates[*index]
                    .3
                    .strip_prefix(&repo)
                    .unwrap_or(&candidates[*index].3)
                    .to_path_buf()
            })
            .collect::<Vec<_>>();
        for (index, path) in indexes.iter().zip(&paths) {
            let flagged = git_in(&repo_dir, ["check-ignore", "-q", "--", &path_text(path)])?;
            if !flagged {
                continue;
            }
            let tracked = git_in(
                &repo_dir,
                ["ls-files", "--error-unmatch", "--", &path_text(path)],
            )?;
            if tracked {
                continue;
            }
            let (scope, _, target, resolved) = &candidates[*index];
            return invalid(format!(
                "{scope} targets {}, which resolves to git-ignored content at {}; the \
                 workspace source digest records link text rather than target content",
                target.display(),
                resolved.display(),
            ));
        }
    }
    Ok(())
}

pub(super) fn path_text(path: &Path) -> String {
    path.display().to_string()
}

/// Run a git query returning whether it affirmed (exit 0) or denied (exit 1);
/// any other exit is a git failure.
pub(super) fn git_in<const N: usize>(dir: &Path, args: [&str; N]) -> Result<bool, InferlabError> {
    let output = Command::new("git")
        .current_dir(dir)
        .args(args)
        .output()
        .map_err(|source| InferlabError::Git {
            root: dir.to_path_buf(),
            source: crate::error::GitError::Launch {
                operation: format!("git {args:?}"),
                source,
            },
        })?;
    match output.status.code() {
        Some(0) => Ok(true),
        Some(1) => Ok(false),
        _ => Err(InferlabError::Git {
            root: dir.to_path_buf(),
            source: crate::error::GitError::Exit {
                operation: format!("git {args:?}"),
                status: output.status,
                diagnostics: String::from_utf8_lossy(&output.stderr).trim().to_owned(),
            },
        }),
    }
}

/// The nearest ancestor of `resolved` (relative to the workspace root, which
/// is itself the outermost owner) containing a `.git` entry.
pub(super) fn owning_repo(root: &Path, resolved: &Path) -> PathBuf {
    let mut dir = resolved.parent().unwrap_or(Path::new(""));
    loop {
        if dir.as_os_str().is_empty() {
            return PathBuf::new();
        }
        if root.join(dir).join(".git").exists() {
            return dir.to_path_buf();
        }
        dir = dir.parent().unwrap_or(Path::new(""));
    }
}

pub(super) fn inspect_workspace(
    root: &Path,
    local_path: &Path,
    config: &WorkspaceConfig,
) -> Result<WorkspaceSnapshot, InferlabError> {
    let revision = git_text(root, &["rev-parse", "HEAD"])?;
    let mut source_exclusions = local_path
        .strip_prefix(root)
        .ok()
        .filter(|relative| is_safe_relative(relative))
        .map(Path::to_path_buf)
        .into_iter()
        .collect::<Vec<_>>();
    source_exclusions.extend([
        PathBuf::from(".inferlab/cache"),
        PathBuf::from(".inferlab/records"),
        PathBuf::from(".inferlab/runtime"),
        // Operator journal state: narrative, never a source fact
        // ([[RFC-0005:C-SCRATCHPAD-JOURNAL]]).
        PathBuf::from(".inferlab/scratchpads"),
    ]);
    // The containment guard precedes the identity reads: a snapshot must not
    // be claimed over a tree whose effective bytes live outside it.
    reject_uncovered_worktree_links(root, config, &source_exclusions)?;
    let dirty = !workspace_mutations(root, &source_exclusions)?.is_empty();
    let source_digest = workspace_source_digest(root, &source_exclusions)?;
    Ok(WorkspaceSnapshot {
        revision,
        dirty,
        source_digest,
        source_exclusions,
        revision_reproducible: !dirty,
        pixi_manifest_sha256: crate::digest::hash_file(&root.join("pixi.toml"))?,
        pixi_lock_sha256: crate::digest::hash_file(&root.join("pixi.lock"))?,
    })
}

/// The `git status` flags that define workspace dirtiness: the porcelain
/// format the mutation scan parses, plus the two flags that widen the scan to
/// untracked files and submodule state. The remote preflight's dirty check
/// and the source-digest scripts derive their script text from the same set
/// so every scan of the effective source state agrees byte-for-byte.
pub(super) const GIT_STATUS_FLAGS: [&str; 3] = [
    "--porcelain=v1",
    "--untracked-files=all",
    "--ignore-submodules=none",
];

/// The dirty-check `git status` flags joined for embedding in a shell script.
pub(crate) fn git_status_flags() -> String {
    GIT_STATUS_FLAGS.join(" ")
}

/// The dirty-check flags with git's NUL output selector interspersed, as the
/// source-digest scripts embed them.
pub(super) fn git_status_flags_z() -> String {
    format!(
        "{} -z {} {}",
        GIT_STATUS_FLAGS[0], GIT_STATUS_FLAGS[1], GIT_STATUS_FLAGS[2]
    )
}

/// Workspace paths that differ from the committed source state, under the
/// same exclusions the snapshot uses. The dirty gate consumes this at
/// resolution; package builds consume it afterwards to detect mutation by
/// external build tooling ([[RFC-0007:C-IMAGE-BUILD]]).
pub(crate) fn workspace_mutations(
    root: &Path,
    exclusions: &[PathBuf],
) -> Result<Vec<String>, InferlabError> {
    // `-z` NUL-separates the machine-readable scan the parser below consumes;
    // it follows the porcelain flag and precedes the scan-widening flags.
    let mut status_args = vec![
        "status".to_owned(),
        GIT_STATUS_FLAGS[0].to_owned(),
        "-z".to_owned(),
        GIT_STATUS_FLAGS[1].to_owned(),
        GIT_STATUS_FLAGS[2].to_owned(),
        "--".to_owned(),
        ".".to_owned(),
    ];
    status_args.extend(
        exclusions
            .iter()
            .map(|path| source_exclusion_pathspec(path)),
    );
    let status = git_bytes(root, status_args)?;
    Ok(status
        .split(|byte| *byte == 0)
        .filter(|entry| !entry.is_empty())
        .map(|entry| String::from_utf8_lossy(entry).into_owned())
        .collect())
}

pub(crate) fn source_digest_script(exclusions: &[PathBuf]) -> String {
    let pathspecs = source_pathspecs(exclusions);
    let status_flags_z = git_status_flags_z();
    format!(
        r#"set -euo pipefail
untracked=$(mktemp)
trap 'rm -f "$untracked"' EXIT
{{
printf 'revision\0'; git rev-parse HEAD
printf 'submodules\0'; git submodule status --recursive
printf 'status\0'; git status {status_flags_z} -- {pathspecs}
printf 'diff\0'; git diff --binary --submodule=diff HEAD -- {pathspecs}
printf 'untracked\0'
git ls-files --others --exclude-standard -z -- {pathspecs} > "$untracked"
while IFS= read -r -d '' path; do
  printf '%s\0' "$path"
  if [ -L "$path" ]; then
    printf 'link\0'; readlink -- "$path"
  elif [ -f "$path" ]; then
    printf 'file\0'; sha256sum < "$path"
  fi
done < "$untracked"
git submodule foreach --quiet --recursive 'set -eu; printf "submodule-worktree\0%s\0" "$displaypath"; git status {status_flags_z}; git diff --binary HEAD; untracked=$(mktemp); trap "rm -f \"$untracked\"" EXIT; git ls-files --others --exclude-standard -z > "$untracked"; xargs -0 -r sh -c '\''set -eu; for path in "$@"; do printf "%s\0" "$path"; if [ -L "$path" ]; then printf "link\0"; readlink -- "$path"; elif [ -f "$path" ]; then printf "file\0"; sha256sum < "$path"; fi; done'\'' classify < "$untracked"'
}} | sha256sum | awk '{{print $1}}'"#
    )
}

pub(crate) fn source_pathspecs(exclusions: &[PathBuf]) -> String {
    std::iter::once("'.'".to_owned())
        .chain(
            exclusions
                .iter()
                .map(|path| source_exclusion_pathspec(path))
                .map(|pathspec| inferlab_runtime::shell::shell_quote(&pathspec)),
        )
        .collect::<Vec<_>>()
        .join(" ")
}

pub(super) fn workspace_source_digest(
    root: &Path,
    exclusions: &[PathBuf],
) -> Result<String, InferlabError> {
    let script = source_digest_script(exclusions);
    let output = Command::new("bash")
        .current_dir(root)
        .args(["-c", &script])
        .output()
        .map_err(|source| InferlabError::Git {
            root: root.to_path_buf(),
            source: crate::error::GitError::Launch {
                operation: "workspace source digest".to_owned(),
                source,
            },
        })?;
    if !output.status.success() {
        return Err(InferlabError::Git {
            root: root.to_path_buf(),
            source: crate::error::GitError::Exit {
                operation: "workspace source digest".to_owned(),
                status: output.status,
                diagnostics: String::from_utf8_lossy(&output.stderr).trim().to_owned(),
            },
        });
    }
    let digest = String::from_utf8(output.stdout)
        .map(|digest| digest.trim().to_owned())
        .map_err(|error| InferlabError::Git {
            root: root.to_path_buf(),
            source: crate::error::GitError::Decode {
                operation: "workspace source digest".to_owned(),
                source: error,
            },
        })?;
    if digest.len() != 64
        || !digest
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(InferlabError::Git {
            root: root.to_path_buf(),
            source: crate::error::GitError::InvalidOutput {
                operation: "workspace source digest".to_owned(),
                detail: format!("invalid SHA-256 {digest:?}"),
            },
        });
    }
    Ok(digest)
}

pub(super) fn source_exclusion_pathspec(path: &Path) -> String {
    format!(":(top,literal,exclude){}", path.to_string_lossy())
}

pub(super) fn git_text(root: &Path, args: &[&str]) -> Result<String, InferlabError> {
    let bytes = git_bytes(root, args.iter().copied())?;
    let text = String::from_utf8(bytes).map_err(|error| InferlabError::Git {
        root: root.to_path_buf(),
        source: crate::error::GitError::Decode {
            operation: format!("git {args:?}"),
            source: error,
        },
    })?;
    Ok(text.trim().to_owned())
}

pub(super) fn git_bytes<I, S>(root: &Path, args: I) -> Result<Vec<u8>, InferlabError>
where
    I: IntoIterator<Item = S>,
    S: AsRef<OsStr>,
{
    let args: Vec<OsString> = args
        .into_iter()
        .map(|value| value.as_ref().to_os_string())
        .collect();
    let rendered_args = args
        .iter()
        .map(|value| value.to_string_lossy())
        .collect::<Vec<_>>()
        .join(" ");
    let output = Command::new("git")
        .current_dir(root)
        .args(&args)
        .output()
        .map_err(|source| InferlabError::Git {
            root: root.to_path_buf(),
            source: crate::error::GitError::Launch {
                operation: format!("git {rendered_args}"),
                source,
            },
        })?;
    if output.status.success() {
        return Ok(output.stdout);
    }
    Err(InferlabError::Git {
        root: root.to_path_buf(),
        source: crate::error::GitError::Exit {
            operation: format!("git {rendered_args}"),
            status: output.status,
            diagnostics: String::from_utf8_lossy(&output.stderr).trim().to_owned(),
        },
    })
}

pub(super) fn is_safe_relative(path: &Path) -> bool {
    !path.as_os_str().is_empty()
        && !path.is_absolute()
        && path
            .components()
            .all(|component| !matches!(component, Component::ParentDir | Component::RootDir))
}

/// Reject a symbolic link anywhere along a declared stack source path
/// ([[RFC-0002:C-WORKSPACE-AUTHORITY]]): the source digest walks git's view
/// of the tree, which records link text rather than target content, so a
/// linked component would let the served source drift under an unchanged
/// digest. Symlinks buried deeper inside a source tree share git's own
/// link-text semantics and stay out of scope here.
pub(super) fn reject_symlink_components(
    root: &Path,
    stack: &str,
    path: &Path,
) -> Result<(), InferlabError> {
    let mut absolute = root.to_path_buf();
    let mut relative = PathBuf::new();
    for component in path.components() {
        absolute.push(component);
        relative.push(component);
        symlink_guard(
            &absolute,
            &format!(
                "stack {stack:?} source path component {}",
                relative.display()
            ),
        )?;
    }
    Ok(())
}

/// Reject a symbolic link where shareable workspace content must live
/// ([[RFC-0002:C-WORKSPACE-AUTHORITY]]): the source digest records link text
/// rather than target content, so a followed link would let the loaded
/// configuration drift under an unchanged digest. Absence passes — the
/// callers own their missing-file handling.
pub(super) fn symlink_guard(absolute: &Path, described: &str) -> Result<(), InferlabError> {
    match fs::symlink_metadata(absolute) {
        Ok(metadata) if metadata.file_type().is_symlink() => invalid(format!(
            "{described} must be a regular filesystem entry, not a symbolic link; \
             the workspace source digest records link text rather than target content"
        )),
        _ => Ok(()),
    }
}
