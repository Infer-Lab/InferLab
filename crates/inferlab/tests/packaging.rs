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
    for crate_name in [
        "inferlab",
        "inferlab-runtime",
        "inferlab-profiler",
        "inferlab-protocol",
        "inferlab-proxy",
    ] {
        let copy = fs::read(root.join("crates").join(crate_name).join("LICENSE"))?;
        assert_eq!(
            copy, repository_license,
            "crates/{crate_name}/LICENSE drifted from the repository LICENSE"
        );
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

    assert_tree_in_archive(
        &root.join("python/inferlab-eval-runner/src/inferlab_eval_runner"),
        Path::new("resources/toolchain-python/inferlab_eval_runner"),
        &files,
    )?;
    assert_tree_in_archive(
        &root.join("python/inferlab-bench-runner/src/inferlab_bench_runner"),
        Path::new("resources/toolchain-python/inferlab_bench_runner"),
        &files,
    )?;
    assert_tree_in_archive(
        &root.join("python/inferlab-measurement-sdk/src/inferlab_measurement_sdk"),
        Path::new("resources/toolchain-python/inferlab_measurement_sdk"),
        &files,
    )?;

    let mut plugin_sources = vec![
        (PathBuf::from("LICENSE"), root.join("LICENSE")),
        (
            PathBuf::from("docs/workspace-authoring.md"),
            root.join("docs/workspace-authoring.md"),
        ),
        (
            PathBuf::from("docs/backend-support.md"),
            root.join("docs/backend-support.md"),
        ),
    ];
    for top in [".claude-plugin", ".agents", "plugins"] {
        collect_source_files(&root, &root.join(top), &mut plugin_sources)?;
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
