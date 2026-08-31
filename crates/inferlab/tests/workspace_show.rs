use serde_json::Value;
use std::error::Error;
use std::fs;
use std::process::Command;

const WORKSPACE: &str = include_str!("fixtures/dsv4-workspace.toml");

fn workspace_without_local_bindings() -> Result<tempfile::TempDir, Box<dyn Error>> {
    let root = tempfile::tempdir()?;
    fs::create_dir_all(root.path().join(".inferlab"))?;
    fs::create_dir_all(root.path().join("vendor/vllm"))?;
    fs::create_dir_all(root.path().join("vendor/flashinfer"))?;
    fs::write(root.path().join(".inferlab/workspace.toml"), WORKSPACE)?;
    fs::write(root.path().join("operator-config.yaml"), "fixture: show\n")?;
    fs::write(
        root.path().join("pixi.toml"),
        "[workspace]\nchannels = [\"conda-forge\"]\nplatforms = [\"linux-64\"]\n\n\
         [environments]\nvllm = []\n\n\
         [pypi-dependencies]\ninferlab-integration-vllm = \"==0.1.0\"\n",
    )?;
    fs::write(
        root.path().join("pixi.lock"),
        "version: 6\nenvironments:\n  vllm: {}\n",
    )?;
    Ok(root)
}

#[test]
fn workspace_show_json_returns_the_merged_public_definition_without_local_bindings()
-> Result<(), Box<dyn Error>> {
    let root = workspace_without_local_bindings()?;
    let output = Command::new(env!("CARGO_BIN_EXE_inferlab"))
        .current_dir(root.path())
        .args(["workspace", "show", "--json"])
        .output()?;

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let value: Value = serde_json::from_slice(&output.stdout)?;
    assert_eq!(value["schema_version"], 2);
    assert_eq!(value["stacks"]["vllm"]["integration"], "vllm");
    assert_eq!(
        value["servers"]["dsv4-qualify"]["model"],
        "deepseek-v4-flash"
    );
    assert_eq!(value["recipes"]["dsv4-qualify"]["server"], "dsv4-qualify");
    assert!(!root.path().join(".inferlab/local.toml").exists());
    Ok(())
}

#[test]
fn workspace_show_human_view_does_not_require_local_bindings() -> Result<(), Box<dyn Error>> {
    let root = workspace_without_local_bindings()?;
    let output = Command::new(env!("CARGO_BIN_EXE_inferlab"))
        .current_dir(root.path())
        .args(["workspace", "show"])
        .output()?;

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(!output.stdout.is_empty());
    Ok(())
}

/// The canonical merged output preserves the declared synthetic-acceptance
/// configuration ([[RFC-0003:C-SERVE-SYNTHETIC-ACCEPTANCE]]): resolution
/// depends on the selected case, so only the declaration appears here — both
/// forms, at server and case level.
#[test]
fn workspace_show_json_preserves_the_synthetic_acceptance_declaration() -> Result<(), Box<dyn Error>>
{
    let root = workspace_without_local_bindings()?;
    let manifest = root.path().join(".inferlab/workspace.toml");
    fs::write(
        &manifest,
        format!(
            "{}\n\
             [servers.spec-explicit]\n\
             stack = \"vllm\"\n\
             model = \"deepseek-v4-flash\"\n\
             topology = \"single\"\n\
             readiness_timeout_seconds = 900\n\
             synthetic_acceptance = {{ acceptance_length = 2.5 }}\n\
             \n\
             [servers.spec-curve]\n\
             stack = \"vllm\"\n\
             model = \"deepseek-v4-flash\"\n\
             topology = \"single\"\n\
             readiness_timeout_seconds = 900\n\
             default_case = \"short\"\n\
             synthetic_acceptance = {{ curve = {{ path = \"curves/golden.yaml\", expected_sha256 = \"{}\", model_key = \"deepseek-v4-flash\", thinking_mode = \"thinking_off\" }} }}\n\
             \n\
             [servers.spec-curve.cases.short]\n\
             synthetic_acceptance = {{ acceptance_length = 1.5 }}\n",
            fs::read_to_string(&manifest)?,
            "a".repeat(64),
        ),
    )?;
    let output = Command::new(env!("CARGO_BIN_EXE_inferlab"))
        .current_dir(root.path())
        .args(["workspace", "show", "--json"])
        .output()?;

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let value: Value = serde_json::from_slice(&output.stdout)?;
    let explicit = &value["servers"]["spec-explicit"]["synthetic_acceptance"];
    assert_eq!(explicit["acceptance_length"], 2.5);
    assert!(
        explicit.get("curve").is_none(),
        "the explicit form omits the curve member: {explicit}"
    );
    let curve = &value["servers"]["spec-curve"]["synthetic_acceptance"]["curve"];
    assert_eq!(curve["path"], "curves/golden.yaml");
    assert_eq!(curve["expected_sha256"], "a".repeat(64));
    assert_eq!(curve["model_key"], "deepseek-v4-flash");
    assert_eq!(curve["thinking_mode"], "thinking_off");
    assert!(
        curve.get("num_speculative_tokens").is_none(),
        "the declaration carries no draft count: {curve}"
    );
    let case = &value["servers"]["spec-curve"]["cases"]["short"]["synthetic_acceptance"];
    assert_eq!(case["acceptance_length"], 1.5);
    Ok(())
}
