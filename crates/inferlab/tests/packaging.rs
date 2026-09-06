use flate2::read::GzDecoder;
use std::collections::BTreeMap;
use std::error::Error;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::Command;

#[test]
fn packaged_licenses_match_the_repository_notice() -> Result<(), Box<dyn Error>> {
    let crate_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let root = crate_dir.join("../..");

    let repository_license = fs::read(root.join("LICENSE"))?;
    // The covered set derives from the workspace layout, so a new crate or
    // Python package cannot escape the comparison by missing a manual list;
    // absence fails loudly too, naming the member.
    for group in ["crates", "python"] {
        let mut members = fs::read_dir(root.join(group))?.collect::<Result<Vec<_>, _>>()?;
        members.sort_by_key(|entry| entry.file_name());
        for member in members {
            if !member.file_type()?.is_dir() {
                continue;
            }
            let copy = member.path().join("LICENSE");
            let bytes = fs::read(&copy).map_err(|source| {
                format!(
                    "{}: every {group}/ member must carry the repository LICENSE: {source}",
                    member.path().display()
                )
            })?;
            assert_eq!(
                bytes,
                repository_license,
                "{} drifted from the repository LICENSE",
                copy.strip_prefix(&root)?.display()
            );
        }
    }
    let embedded = Command::new(env!("CARGO_BIN_EXE_inferlab"))
        .args(["license"])
        .output()?;
    assert_eq!(
        embedded.stdout, repository_license,
        "the embedded notice drifted from the repository LICENSE"
    );

    Ok(())
}

#[test]
fn staged_crate_contains_the_canonical_product_payload() -> Result<(), Box<dyn Error>> {
    let crate_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let root = crate_dir.join("../..");
    assert!(!crate_dir.join("resources/toolchain-python").exists());
    assert!(!crate_dir.join("resources/plugin").exists());

    let output_dir = tempfile::tempdir()?;
    let package = Command::new(root.join("scripts/package-inferlab-crate.sh"))
        .arg(output_dir.path())
        .output()?;
    assert!(
        package.status.success(),
        "crate staging failed: {}",
        String::from_utf8_lossy(&package.stderr)
    );
    let artifact = PathBuf::from(String::from_utf8(package.stdout)?.trim());
    let files = crate_archive_files(&artifact)?;
    assert_eq!(
        files.get(Path::new("resources/bench-agentic-sources.toml")),
        Some(&fs::read(
            crate_dir.join("resources/bench-agentic-sources.toml")
        )?),
        "the staged crate omitted or changed the AgentX source catalog"
    );
    assert!(
        !files.contains_key(Path::new("resources/plugin/docs/workspace-authoring.md")),
        "the staged plugin must not carry an aggregate workspace-authoring copy"
    );

    // The member set has one manifest, shared with the crate build script and
    // the crate staging script (scripts/toolchain-python-members.txt).
    let manifest = fs::read_to_string(root.join("scripts/toolchain-python-members.txt"))?;
    for member in manifest.lines().filter(|line| !line.is_empty()) {
        let (source, package) = member.split_once(' ').ok_or_else(|| {
            format!("toolchain Python manifest member has no package name: {member}")
        })?;
        assert_tree_in_archive(
            &root.join(source),
            &Path::new("resources/toolchain-python").join(package),
            &files,
        )?;
    }

    // The member set has one manifest, shared with the crate build script and
    // the release scripts (scripts/plugin-package-members.txt).
    let manifest = fs::read_to_string(root.join("scripts/plugin-package-members.txt"))?;
    let mut plugin_sources = Vec::new();
    for member in manifest.lines().filter(|line| !line.is_empty()) {
        let source = root.join(member);
        if source.is_dir() {
            collect_source_files(&root, &source, &mut plugin_sources)?;
        } else {
            plugin_sources.push((PathBuf::from(member), source));
        }
    }
    for (relative, source) in plugin_sources {
        let packaged = Path::new("resources/plugin").join(&relative);
        assert_eq!(
            files.get(&packaged),
            Some(&fs::read(source)?),
            "staged plugin file {} differs from its canonical source",
            relative.display()
        );
    }
    Ok(())
}

/// The agent-install payload (build.rs embeds it) and the release asset
/// (scripts/pack-plugin.sh produces it) are two producers of one artifact;
/// the shared fixed mtime exists so they can agree. Compare the entry
/// surface — names, modes, contents, and mtimes — so a one-sided edit to
/// either producer fails here instead of forking the distributables.
#[test]
fn embedded_and_release_plugin_tarballs_carry_identical_entries() -> Result<(), Box<dyn Error>> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let embedded = Path::new(env!("OUT_DIR")).join("inferlab-plugin.tar.gz");

    let output_dir = tempfile::tempdir()?;
    let release = output_dir.path().join("plugin.tar.gz");
    let status = Command::new(root.join("scripts/pack-plugin.sh"))
        .arg(&release)
        .current_dir(&root)
        .status()?;
    assert!(status.success(), "pack-plugin.sh failed");

    let embedded_entries = plugin_tarball_entries(&embedded)?;
    let release_entries = plugin_tarball_entries(&release)?;
    assert_eq!(
        embedded_entries, release_entries,
        "the embedded plugin tarball and the release tarball diverge in member set, modes, contents, or mtimes"
    );
    Ok(())
}

/// File entries of a plugin tarball as path → (mtime, mode, contents) — the
/// surface on which the two plugin artifact producers must agree.
type PluginEntries = BTreeMap<PathBuf, (u64, u32, Vec<u8>)>;

fn plugin_tarball_entries(path: &Path) -> Result<PluginEntries, Box<dyn Error>> {
    let decoder = GzDecoder::new(fs::File::open(path)?);
    let mut archive = tar::Archive::new(decoder);
    let mut entries = BTreeMap::new();
    for entry in archive.entries()? {
        let mut entry = entry?;
        if !entry.header().entry_type().is_file() {
            continue;
        }
        let mtime = entry.header().mtime()?;
        let mode = entry.header().mode()?;
        let mut bytes = Vec::new();
        entry.read_to_end(&mut bytes)?;
        entries.insert(entry.path()?.into_owned(), (mtime, mode, bytes));
    }
    Ok(entries)
}

fn crate_archive_files(path: &Path) -> Result<BTreeMap<PathBuf, Vec<u8>>, Box<dyn Error>> {
    let decoder = GzDecoder::new(fs::File::open(path)?);
    let mut archive = tar::Archive::new(decoder);
    let mut files = BTreeMap::new();
    for entry in archive.entries()? {
        let mut entry = entry?;
        if !entry.header().entry_type().is_file() {
            continue;
        }
        let path = entry.path()?.into_owned();
        let relative = path.components().skip(1).collect::<PathBuf>();
        let mut bytes = Vec::new();
        entry.read_to_end(&mut bytes)?;
        files.insert(relative, bytes);
    }
    Ok(files)
}

fn assert_tree_in_archive(
    source_root: &Path,
    packaged_root: &Path,
    files: &BTreeMap<PathBuf, Vec<u8>>,
) -> Result<(), Box<dyn Error>> {
    let mut sources = Vec::new();
    collect_source_files(source_root, source_root, &mut sources)?;
    for (relative, source) in sources {
        let packaged = packaged_root.join(&relative);
        assert_eq!(
            files.get(&packaged),
            Some(&fs::read(source)?),
            "staged payload file {} differs from its canonical source",
            relative.display()
        );
    }
    Ok(())
}

#[test]
fn plugin_manifests_match_the_crate_version() -> Result<(), Box<dyn Error>> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let crate_version = env!("CARGO_PKG_VERSION");
    for (manifest, pointer) in [
        ("plugins/inferlab/.claude-plugin/plugin.json", "/version"),
        ("plugins/inferlab/.codex-plugin/plugin.json", "/version"),
        (".claude-plugin/marketplace.json", "/plugins/0/version"),
    ] {
        let bytes = fs::read(root.join(manifest))?;
        let value: serde_json::Value = serde_json::from_slice(&bytes)?;
        let version = value
            .pointer(pointer)
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| format!("{manifest} has no string at {pointer}"))?;
        assert_eq!(
            version, crate_version,
            "{manifest} must match the crate version ([[RFC-0008:C-AGENT-PLUGIN]])"
        );
    }
    Ok(())
}

fn collect_source_files(
    root: &Path,
    dir: &Path,
    out: &mut Vec<(PathBuf, PathBuf)>,
) -> Result<(), Box<dyn Error>> {
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if entry.file_type()?.is_dir() {
            if entry.file_name() == "__pycache__" {
                continue;
            }
            collect_source_files(root, &path, out)?;
        } else if path.extension().is_none_or(|extension| extension != "pyc") {
            out.push((path.strip_prefix(root)?.to_path_buf(), path));
        }
    }
    Ok(())
}
