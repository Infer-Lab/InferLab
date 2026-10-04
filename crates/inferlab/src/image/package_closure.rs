//! Pixi-owned locked package closure and editable source identities.

use crate::InferlabError;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Package build-procedure identity ([[RFC-0007:C-IMAGE-BUILD]]): bump when
/// the wheel build procedure changes behavior (copy sanitization, build
/// flags, cache layout or key derivation, publication protocol) without a
/// crate version change. Enters both the wheel cache key and the image
/// content closure, so a procedure change invalidates cached wheels and
/// changes the closure digest together.
/// Epoch 6: compiler path maps reach every build, and built packages are
/// made path-neutral and verified before caching ([[ADR-0057]]).
pub(super) const WHEEL_BUILD_EPOCH: u32 = 6;

/// Map an OCI platform (`linux/amd64`) to the Pixi platform (`linux-64`).
pub(super) fn pixi_platform(oci_platform: &str) -> Result<&'static str, InferlabError> {
    match oci_platform {
        "linux/amd64" => Ok("linux-64"),
        "linux/arm64" => Ok("linux-aarch64"),
        other => Err(InferlabError::ImageBuild {
            message: format!("unsupported target platform {other:?}"),
        }),
    }
}

#[derive(Debug, Deserialize)]
struct PixiListEntry {
    name: String,
    kind: String,
    #[serde(default)]
    url: Option<String>,
    #[serde(default)]
    sha256: Option<String>,
}

/// List the locked packages of one environment via `pixi list --json`; Pixi
/// remains the only package authority. `--locked --no-install` keeps the
/// listing read-only: a stale lock fails instead of being re-solved, so image
/// resolution never rewrites the committed lock it derives facts from. The listing covers the host platform,
/// which is the only platform the local builder assembles. Entries without a
/// registry hash are source-backed (editable path dependencies) and are
/// either replaced by locally built wheels or deliberately excluded.
pub(super) fn locked_packages(
    root: &Path,
    environment: &str,
) -> Result<Vec<PackageSpec>, InferlabError> {
    let output = Command::new("pixi")
        .current_dir(root)
        .args([
            "list",
            "--json",
            "--environment",
            environment,
            "--locked",
            "--no-install",
        ])
        .output()
        .map_err(|source| InferlabError::LaunchPixi {
            action: "list",
            source,
        })?;
    if !output.status.success() {
        return Err(InferlabError::ImageToolExit {
            operation: format!("pixi list for environment {environment:?}"),
            status: output.status,
            diagnostics: String::from_utf8_lossy(&output.stderr).trim().to_owned(),
        });
    }
    let entries: Vec<PixiListEntry> =
        serde_json::from_slice(&output.stdout).map_err(|source| InferlabError::ImageToolJson {
            operation: "pixi list".to_owned(),
            source,
        })?;
    if entries.is_empty() {
        return Err(InferlabError::ImageBuild {
            message: format!("pixi list reported no packages for environment {environment:?}"),
        });
    }
    Ok(entries
        .into_iter()
        .map(|entry| {
            let editable = entry.kind != "conda"
                && entry.sha256.is_none()
                && entry.url.as_deref().and_then(local_project_path).is_some();
            PackageSpec {
                name: entry.name,
                kind: match entry.kind.as_str() {
                    "conda" => PackageKind::Conda,
                    _ => PackageKind::Pypi,
                },
                url: entry.url,
                sha256: entry.sha256,
                editable,
            }
        })
        .collect())
}

/// Workspace-relative project directory of a locked location, or `None` for
/// a registry location. Pixi reports source-backed projects as relative paths
/// (`./sglang/python`); any location carrying a URL scheme is an artifact.
fn local_project_path(url: &str) -> Option<PathBuf> {
    if url.contains("://") {
        return None;
    }
    Some(without_current_dir(Path::new(url)))
}

/// A workspace-relative path without `.` components, so `./sglang` and
/// `sglang` compare as the same path.
pub(super) fn without_current_dir(path: &Path) -> PathBuf {
    path.components()
        .filter(|component| !matches!(component, std::path::Component::CurDir))
        .collect()
}

/// Directories to build for the selected image package paths
/// ([[RFC-0007:C-IMAGE-BUILD]]): every source-backed project the environment
/// installs from inside a selected path, at the directory Pixi reports. A
/// selected path with no such project is an error rather than a root build,
/// because that wheel would have no counterpart in the realized environment.
pub(super) fn package_project_directories(
    packages: &[PackageSpec],
    selected: &[PathBuf],
    environment: &str,
) -> Result<Vec<PathBuf>, InferlabError> {
    let projects: Vec<PathBuf> = packages
        .iter()
        .filter(|package| package.editable)
        .filter_map(|package| package.url.as_deref().and_then(local_project_path))
        .collect();
    let mut directories: Vec<PathBuf> = Vec::new();
    for path in selected {
        let normalized = without_current_dir(path);
        let mut inside: Vec<&PathBuf> = projects
            .iter()
            .filter(|project| project.starts_with(&normalized))
            .collect();
        if inside.is_empty() {
            return Err(InferlabError::ImageBuild {
                message: format!(
                    "image package path {} contains no source-backed project of Pixi \
                     environment {environment:?}; install it from that path or remove it \
                     from the image packages",
                    path.display()
                ),
            });
        }
        inside.sort();
        for project in inside {
            if !directories.contains(project) {
                directories.push(project.clone());
            }
        }
    }
    Ok(directories)
}

/// Locked registry packages that the generated image context cannot pin by
/// hash, so the committed lock cannot reproduce them ([[RFC-0007:C-IMAGE-BUILD]]).
/// Image resolution rejects them before any package build; context
/// generation keeps the same rule as its final guard.
pub(super) fn unpinned_registry_packages(packages: &[PackageSpec]) -> Vec<&str> {
    packages
        .iter()
        .filter(|package| !package.editable && (package.url.is_none() || package.sha256.is_none()))
        .map(|package| package.name.as_str())
        .collect()
}

/// Content identity of the selected environment's locked package closure
/// ([[RFC-0007:C-IMAGE-BUILD]]): only pinned upstream packages move this
/// digest — editable entries are excluded because their identity is the
/// committed source state, which the cache key already carries exactly.
/// Unlike whole-file manifest digests, unrelated environments, platforms,
/// tasks, and format churn leave this stable.
pub(super) fn locked_closure_digest(packages: &[PackageSpec]) -> String {
    let canonical: Vec<String> = packages
        .iter()
        .filter(|package| !package.editable)
        .map(|package| {
            format!(
                "{}\u{1f}{:?}\u{1f}{}\u{1f}{}",
                package.name,
                package.kind,
                package.url.as_deref().unwrap_or(""),
                package.sha256.as_deref().unwrap_or("")
            )
        })
        .collect();
    format!("{:x}", Sha256::digest(canonical.join("\u{1e}").as_bytes()))
}

/// Content identities for editable packages installed outside the stack sources
/// ([[RFC-0007:C-IMAGE-BUILD]]): stack-source editables are covered exactly by
/// committed git identities, but external path dependencies (for example
/// Inferlab's own adapter and integration packages from a sibling repository)
/// sit in the build environment with no other identity, so their tree content
/// enters the cache key. Paths come from `pixi list` itself — the package
/// authority — not from a second manifest parse.
pub(super) fn editable_identities(
    root: &Path,
    packages: &[PackageSpec],
    source_paths: &[PathBuf],
) -> Result<Vec<String>, InferlabError> {
    let mut identities = Vec::new();
    for package in packages {
        if !package.editable {
            continue;
        }
        let Some(url) = &package.url else {
            return Err(InferlabError::ImageBuild {
                message: format!(
                    "editable package {:?} reports no source path; its build-input identity \
                     cannot be derived",
                    package.name
                ),
            });
        };
        let relative = Path::new(url.strip_prefix("./").unwrap_or(url));
        if source_paths.iter().any(|path| relative.starts_with(path)) {
            continue;
        }
        let digest = tree_digest(&root.join(relative))?;
        identities.push(format!("{}\u{1f}{digest}", relative.display()));
    }
    identities.sort();
    Ok(identities)
}

/// Deterministic content digest of one source tree: files sorted by relative
/// path, hashing path and bytes. Derived artifacts that churn without a
/// source change (`.git`, `__pycache__`, `*.egg-info`) are excluded; symlinks
/// contribute their target string.
fn tree_digest(path: &Path) -> Result<String, InferlabError> {
    fn walk(dir: &Path, files: &mut Vec<PathBuf>) -> Result<(), InferlabError> {
        let entries = fs::read_dir(dir).map_err(|source| InferlabError::Read {
            path: dir.to_path_buf(),
            source,
        })?;
        for entry in entries {
            let entry = entry.map_err(|source| InferlabError::Read {
                path: dir.to_path_buf(),
                source,
            })?;
            let name = entry.file_name().to_string_lossy().into_owned();
            if name == ".git" || name == "__pycache__" || name.ends_with(".egg-info") {
                continue;
            }
            let file_type = entry.file_type().map_err(|source| InferlabError::Read {
                path: entry.path(),
                source,
            })?;
            if file_type.is_dir() {
                walk(&entry.path(), files)?;
            } else {
                files.push(entry.path());
            }
        }
        Ok(())
    }
    let mut files = Vec::new();
    walk(path, &mut files)?;
    files.sort();
    let mut hasher = Sha256::new();
    for file in &files {
        let relative = file.strip_prefix(path).unwrap_or(file);
        hasher.update(relative.display().to_string().as_bytes());
        hasher.update([0]);
        if file.is_symlink() {
            let target = fs::read_link(file).map_err(|source| InferlabError::Read {
                path: file.clone(),
                source,
            })?;
            hasher.update(target.display().to_string().as_bytes());
        } else {
            let bytes = fs::read(file).map_err(|source| InferlabError::Read {
                path: file.clone(),
                source,
            })?;
            hasher.update(&bytes);
        }
        hasher.update([0]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

/// PEP 503 package-name normalization for wheel/lock comparisons.
pub(super) fn normalize_package_name(name: &str) -> String {
    let mut normalized = String::with_capacity(name.len());
    let mut previous_separator = false;
    for character in name.chars() {
        if matches!(character, '-' | '_' | '.') {
            if !previous_separator {
                normalized.push('-');
            }
            previous_separator = true;
        } else {
            normalized.push(character.to_ascii_lowercase());
            previous_separator = false;
        }
    }
    normalized
}

#[derive(Clone, Debug)]
pub(super) struct PackageSpec {
    pub name: String,
    pub kind: PackageKind,
    pub url: Option<String>,
    pub sha256: Option<String>,
    pub editable: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum PackageKind {
    Conda,
    Pypi,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn project_directories_cover_every_project_under_a_selected_path() -> Result<(), InferlabError>
    {
        let source_backed = |name: &str, url: &str| PackageSpec {
            name: name.to_owned(),
            kind: PackageKind::Pypi,
            url: Some(url.to_owned()),
            sha256: None,
            editable: true,
        };
        let packages = vec![
            source_backed("flashinfer-python", "./flashinfer"),
            source_backed("flashinfer-cubin", "./flashinfer/flashinfer-cubin"),
            source_backed("sglang", "./sglang/python"),
        ];
        // Nested selected paths share a project; it is built once.
        let directories = package_project_directories(
            &packages,
            &[
                PathBuf::from("sglang"),
                PathBuf::from("flashinfer"),
                PathBuf::from("flashinfer/flashinfer-cubin"),
            ],
            "sglang",
        )?;
        assert_eq!(
            directories,
            vec![
                PathBuf::from("sglang/python"),
                PathBuf::from("flashinfer"),
                PathBuf::from("flashinfer/flashinfer-cubin"),
            ]
        );
        // The workspace accepts `./`-spelled source paths; they select the
        // same projects.
        assert_eq!(
            package_project_directories(&packages, &[PathBuf::from("./sglang")], "sglang")?,
            vec![PathBuf::from("sglang/python")]
        );
        Ok(())
    }

    #[test]
    fn editable_identities_cover_external_paths_only() -> Result<(), InferlabError> {
        let scratch = tempfile::tempdir().map_err(|source| InferlabError::EnvironmentIo {
            path: PathBuf::from("tempdir"),
            operation: "create test workspace",
            source,
        })?;
        let external = scratch.path().join("sibling/pkg");
        fs::create_dir_all(external.join("src")).map_err(|source| {
            InferlabError::EnvironmentIo {
                path: external.clone(),
                operation: "create external package",
                source,
            }
        })?;
        fs::create_dir_all(external.join("__pycache__")).map_err(|source| {
            InferlabError::EnvironmentIo {
                path: external.clone(),
                operation: "create pycache",
                source,
            }
        })?;
        let write = |relative: &str, content: &str| -> Result<(), InferlabError> {
            fs::write(external.join(relative), content).map_err(|source| {
                InferlabError::EnvironmentIo {
                    path: external.join(relative),
                    operation: "write external file",
                    source,
                }
            })
        };
        write("src/module.py", "VALUE = 1\n")?;
        write("__pycache__/module.cpython-312.pyc", "derived")?;
        let root = scratch.path().join("workspace");
        fs::create_dir_all(&root).map_err(|source| InferlabError::EnvironmentIo {
            path: root.clone(),
            operation: "create workspace root",
            source,
        })?;
        let editable = |name: &str, url: &str| PackageSpec {
            name: name.to_owned(),
            kind: PackageKind::Pypi,
            url: Some(url.to_owned()),
            sha256: None,
            editable: true,
        };
        let packages = vec![
            editable("vllm", "./vllm"),
            editable("inferlab-adapter-sdk", "../sibling/pkg"),
            PackageSpec {
                name: "torch".to_owned(),
                kind: PackageKind::Pypi,
                url: Some("https://example/torch".to_owned()),
                sha256: Some("aa".to_owned()),
                editable: false,
            },
        ];
        let source_paths = [PathBuf::from("vllm")];
        let baseline = editable_identities(&root, &packages, &source_paths)?;
        assert_eq!(baseline.len(), 1, "only the external editable is keyed");
        assert!(baseline[0].starts_with("../sibling/pkg\u{1f}"));
        write("__pycache__/module.cpython-312.pyc", "different derived")?;
        assert_eq!(
            editable_identities(&root, &packages, &source_paths)?,
            baseline
        );
        write("src/module.py", "VALUE = 2\n")?;
        assert_ne!(
            editable_identities(&root, &packages, &source_paths)?,
            baseline
        );
        let no_url = vec![PackageSpec {
            name: "mystery".to_owned(),
            kind: PackageKind::Pypi,
            url: None,
            sha256: None,
            editable: true,
        }];
        assert!(
            editable_identities(&root, &no_url, &source_paths).is_err(),
            "an editable without a source path cannot be silently skipped"
        );
        Ok(())
    }

    #[test]
    fn locked_closure_digest_tracks_pinned_packages_only() {
        let pinned = |name: &str, sha: &str| PackageSpec {
            name: name.to_owned(),
            kind: PackageKind::Pypi,
            url: Some(format!("https://example/{name}")),
            sha256: Some(sha.to_owned()),
            editable: false,
        };
        let editable = PackageSpec {
            name: "vllm".to_owned(),
            kind: PackageKind::Pypi,
            url: None,
            sha256: None,
            editable: true,
        };
        let with_editable = vec![pinned("torch", "1"), editable.clone()];
        let without_editable = vec![pinned("torch", "1")];
        assert_eq!(
            locked_closure_digest(&with_editable),
            locked_closure_digest(&without_editable),
            "editable entries never move the closure digest"
        );
        let pin_changed = vec![pinned("torch", "2"), editable];
        assert_ne!(
            locked_closure_digest(&with_editable),
            locked_closure_digest(&pin_changed),
            "a pinned package change moves the closure digest"
        );
    }

    #[test]
    fn pixi_platform_maps_supported_targets() -> Result<(), InferlabError> {
        assert_eq!(pixi_platform("linux/amd64")?, "linux-64");
        assert_eq!(pixi_platform("linux/arm64")?, "linux-aarch64");
        assert!(pixi_platform("windows/amd64").is_err());
        Ok(())
    }
}
