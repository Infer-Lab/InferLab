//! Embeds the product-owned Python toolchain and agent plugin from their one
//! source authority. Repository builds read the canonical monorepo trees;
//! `scripts/package-inferlab-crate.sh` projects those same trees into the
//! crate-local `resources/` paths before `cargo package`, so the published
//! crate stays self-contained without a committed editable mirror.

use flate2::{Compression, GzBuilder};
use std::env;
use std::error::Error;
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};

/// 2026-01-01T00:00:00Z, matching `scripts/pack-plugin.sh`'s `--mtime`.
const FIXED_MTIME: u64 = 1_767_225_600;

fn main() -> Result<(), Box<dyn Error>> {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR")?);
    let out_dir = PathBuf::from(env::var("OUT_DIR")?);
    generate_toolchain_python_manifest(&manifest_dir, &out_dir)?;

    let members = plugin_sources(&manifest_dir)?;

    let out_path = out_dir.join("inferlab-plugin.tar.gz");
    let file = File::create(&out_path)?;
    // `GzBuilder::mtime(0)` and no filename/comment: the gzip stream itself
    // carries no host- or time-dependent bytes, matching `gzip -n`.
    let encoder = GzBuilder::new()
        .mtime(0)
        .write(file, Compression::default());
    let mut builder = tar::Builder::new(encoder);

    for (member, source) in &members {
        let contents = fs::read(source)?;
        let mut header = tar::Header::new_gnu();
        header.set_size(contents.len() as u64);
        header.set_mode(0o644);
        header.set_mtime(FIXED_MTIME);
        header.set_uid(0);
        header.set_gid(0);
        builder.append_data(&mut header, member, contents.as_slice())?;
    }

    let encoder = builder.into_inner()?;
    encoder.finish()?;
    Ok(())
}

/// Generates the one compile-time inventory for the measurement runtime
/// payload. Runtime materialization and content identity consume this same
/// sorted set, so adding a Python module cannot require a second manual list.
fn generate_toolchain_python_manifest(
    manifest_dir: &Path,
    out_dir: &Path,
) -> Result<(), Box<dyn Error>> {
    let members = toolchain_python_sources(manifest_dir)?;

    let mut output = File::create(out_dir.join("toolchain_python_files.rs"))?;
    writeln!(output, "const TOOLCHAIN_PYTHON_FILES: &[(&str, &str)] = &[")?;
    for (member, source) in &members {
        let relative = member
            .to_str()
            .ok_or("toolchain Python path is not UTF-8")?;
        let source = source
            .to_str()
            .ok_or("toolchain Python source path is not UTF-8")?;
        writeln!(output, "    ({relative:?}, include_str!({source:?})),")?;
    }
    writeln!(output, "];")?;
    for (name, relative) in [
        (
            "ESTONIA_TASK",
            "inferlab_eval_runner/bundled_tasks/estonia/estonia.yaml",
        ),
        (
            "ESTONIA_PROMPT",
            "inferlab_eval_runner/bundled_tasks/estonia/prompt.txt",
        ),
        (
            "ESTONIA_DATASET",
            "inferlab_eval_runner/bundled_tasks/estonia/dataset.json",
        ),
        (
            "ESTONIA_SCORER",
            "inferlab_eval_runner/bundled_tasks/estonia/estonia.py",
        ),
    ] {
        let source = members
            .iter()
            .find_map(|(member, source)| (member == Path::new(relative)).then_some(source))
            .ok_or_else(|| format!("toolchain payload is missing {relative}"))?;
        let source = source
            .to_str()
            .ok_or("bundled Eval task source path is not UTF-8")?;
        writeln!(output, "const {name}: &str = include_str!({source:?});")?;
    }
    Ok(())
}

fn toolchain_python_sources(
    manifest_dir: &Path,
) -> Result<Vec<(PathBuf, PathBuf)>, Box<dyn Error>> {
    let staged = manifest_dir.join("resources/toolchain-python");
    if staged.is_dir() {
        println!("cargo:rerun-if-changed={}", staged.display());
        return collect_source_tree(&staged, &staged, None);
    }

    let repository = manifest_dir.join("../..");
    // The toolchain Python payload membership has one manifest, shared with
    // the crate staging script and the packaging test
    // (scripts/toolchain-python-members.txt).
    let manifest = repository.join("scripts/toolchain-python-members.txt");
    println!("cargo:rerun-if-changed={}", manifest.display());
    let members = std::fs::read_to_string(&manifest)?;
    let mut sources = Vec::new();
    for member in members.lines().filter(|line| !line.is_empty()) {
        let (relative, package) = member.split_once(' ').ok_or_else(|| {
            format!("toolchain Python manifest member has no package name: {member}")
        })?;
        let root = repository.join(relative);
        println!("cargo:rerun-if-changed={}", root.display());
        sources.extend(collect_source_tree(&root, &root, Some(Path::new(package)))?);
    }
    sources.sort_by(|left, right| left.0.cmp(&right.0));
    Ok(sources)
}

fn plugin_sources(manifest_dir: &Path) -> Result<Vec<(PathBuf, PathBuf)>, Box<dyn Error>> {
    let staged = manifest_dir.join("resources/plugin");
    if staged.is_dir() {
        println!("cargo:rerun-if-changed={}", staged.display());
        return collect_source_tree(&staged, &staged, None);
    }

    let repository = manifest_dir.join("../..");
    // The plugin payload membership has one manifest, shared with the release
    // tarball and the crate staging scripts (scripts/plugin-package-members.txt).
    let manifest = repository.join("scripts/plugin-package-members.txt");
    println!("cargo:rerun-if-changed={}", manifest.display());
    let members = std::fs::read_to_string(&manifest)?;
    let mut sources = Vec::new();
    for member in members.lines().filter(|line| !line.is_empty()) {
        let source = repository.join(member);
        println!("cargo:rerun-if-changed={}", source.display());
        if source.is_dir() {
            sources.extend(collect_source_tree(&repository, &source, None)?);
        } else {
            sources.push((PathBuf::from(member), source));
        }
    }
    sources.sort_by(|left, right| left.0.cmp(&right.0));
    Ok(sources)
}

fn collect_source_tree(
    root: &Path,
    dir: &Path,
    prefix: Option<&Path>,
) -> Result<Vec<(PathBuf, PathBuf)>, Box<dyn Error>> {
    let mut members = Vec::new();
    collect_files(root, dir, &mut members)?;
    let mut sources = members
        .into_iter()
        .map(|member| {
            let relative = prefix.map_or_else(|| member.clone(), |prefix| prefix.join(&member));
            (relative, root.join(member))
        })
        .collect::<Vec<_>>();
    sources.sort_by(|left, right| left.0.cmp(&right.0));
    Ok(sources)
}

/// Recursively collects every file under `dir`, as paths relative to
/// `root`. The caller sorts the result; traversal order here does not
/// matter for the final archive's determinism.
fn collect_files(root: &Path, dir: &Path, out: &mut Vec<PathBuf>) -> Result<(), Box<dyn Error>> {
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if entry.file_type()?.is_dir() {
            if entry.file_name() == "__pycache__" {
                continue;
            }
            collect_files(root, &path, out)?;
        } else if path.extension().is_none_or(|extension| extension != "pyc") {
            out.push(path.strip_prefix(root)?.to_path_buf());
        }
    }
    Ok(())
}
