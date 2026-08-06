mod dry_run_support;
mod support;

use serde_json::Value;
use std::error::Error;
use std::fs;
use std::process::Command;

use dry_run_support::*;

#[test]
fn unresolved_typed_reference_is_rejected() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    let path = workspace.root.path().join(".inferlab/workspace.toml");
    fs::write(
        &path,
        WORKSPACE.replace("model = \"dsv4\"", "model = \"missing\""),
    )?;
    let output = workspace.run(&["recipe", "run", "dsv4-qualify", "--dry-run"])?;

    assert!(!output.status.success());
    assert!(String::from_utf8(output.stderr)?.contains("unknown model"));
    Ok(())
}

#[test]
fn dirty_workspace_reports_a_digest_and_effective_values() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    fs::write(
        workspace.root.path().join("vendor/vllm/source.txt"),
        "local edit\n",
    )?;
    let plan = workspace.run_json(&["recipe", "run", "dsv4-qualify", "--dry-run"])?;

    assert_eq!(plan["workspace"]["dirty"], true);
    assert_eq!(plan["workspace"]["revision_reproducible"], false);
    assert_eq!(
        plan["workspace"]["source_digest"].as_str().map(str::len),
        Some(64)
    );
    assert!(plan.to_string().contains(&workspace.private_weight));
    Ok(())
}

#[test]
fn scratchpad_state_stays_outside_source_identity() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    let baseline = workspace.run_json(&["serve", "start", "dsv4-qualify", "--dry-run"])?;
    assert_eq!(baseline["workspace"]["dirty"], false);

    let note = workspace.run(&[
        "scratchpad",
        "note",
        "journal text is not a source fact",
        "--topic",
        "pd-debug",
    ])?;
    assert!(
        note.status.success(),
        "{}",
        String::from_utf8_lossy(&note.stderr)
    );

    let after = workspace.run_json(&["serve", "start", "dsv4-qualify", "--dry-run"])?;
    assert_eq!(after["workspace"]["dirty"], false);
    assert_eq!(
        after["workspace"]["source_digest"],
        baseline["workspace"]["source_digest"]
    );
    Ok(())
}

#[test]
fn explicit_local_bindings_file_replaces_the_default() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    let alternate = workspace.root.path().join("alternate-local.toml");
    TestWorkspace::write_local_bindings(&alternate, &workspace.private_weight)?;
    fs::remove_file(workspace.root.path().join(".inferlab/local.toml"))?;

    let plan = workspace.run_json(&[
        "serve",
        "start",
        "dsv4-qualify",
        "--local",
        alternate.to_str().ok_or("non-UTF-8 test path")?,
        "--dry-run",
    ])?;
    assert_eq!(plan["server"]["case"]["id"], "tp2");
    assert_eq!(plan["workspace"]["dirty"], false);
    Ok(())
}

#[test]
fn missing_weight_binding_is_reported_before_lowering() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    fs::write(
        workspace.root.path().join(".inferlab/local.toml"),
        "default_placement = \"local\"\n\
         \n\
         [model_weights]\n\
         \n\
         [machines.local]\n\
         host = \"127.0.0.1\"\n\
         ports = [8000]\n\
         devices = [0, 1]\n\
         \n\
         [placements.local]\n\
         machines = [\"local\"]\n",
    )?;

    let output = workspace.run(&["serve", "start", "dsv4-qualify", "--dry-run"])?;

    assert!(!output.status.success());
    assert!(String::from_utf8(output.stderr)?.contains("missing model weight binding"));
    Ok(())
}

#[test]
fn placement_role_must_belong_to_the_resolved_topology() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    let path = workspace.root.path().join(".inferlab/local.toml");
    let mut local = fs::read_to_string(&path)?;
    local.push_str("\n[placements.local.roles.typo]\nmachines = [\"local\"]\n");
    fs::write(path, local)?;

    let output = workspace
        .command()
        .args(["serve", "start", "dsv4-qualify", "--dry-run"])
        .output()?;

    assert!(!output.status.success());
    assert!(String::from_utf8(output.stderr)?.contains(
        "placement references role \"typo\", which is not part of the resolved topology"
    ));
    Ok(())
}

#[test]
fn case_and_invocation_roles_must_belong_to_the_selected_topology() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    let invocation = workspace.run(&[
        "serve",
        "start",
        "dsv4-qualify",
        "--set",
        "server.roles.typo.replicas=2",
        "--dry-run",
    ])?;
    assert!(!invocation.status.success());

    let path = workspace.root.path().join(".inferlab/workspace.toml");
    let mut config = fs::read_to_string(&path)?;
    config.push_str(
        "\n[servers.dsv4-qualify.cases.tp4.roles.typo]\n\
         replicas = 2\n",
    );
    fs::write(path, config)?;
    let case = workspace.run(&[
        "serve",
        "start",
        "dsv4-qualify",
        "--case",
        "tp4",
        "--dry-run",
    ])?;
    assert!(!case.status.success());
    Ok(())
}

#[test]
fn insufficient_devices_are_reported_after_lowering() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    fs::write(
        workspace.root.path().join(".inferlab/local.toml"),
        format!(
            "default_placement = \"local\"\n\
             \n\
             [model_weights.dsv4]\n\
             locator = {:?}\n\
             \n\
             [machines.local]\n\
             host = \"127.0.0.1\"\n\
             ports = [8000]\n\
             devices = [0]\n\
             \n\
             [placements.local]\n\
             machines = [\"local\"]\n",
            workspace.private_weight
        ),
    )?;

    let output = workspace.run(&["serve", "start", "dsv4-qualify", "--dry-run"])?;

    assert!(!output.status.success());
    assert!(String::from_utf8(output.stderr)?.contains("provides 1 devices"));
    Ok(())
}

#[test]
fn unknown_pixi_environment_is_rejected_before_lowering() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    let path = workspace.root.path().join(".inferlab/workspace.toml");
    fs::write(
        &path,
        WORKSPACE.replace(
            "pixi_environment = \"vllm\"",
            "pixi_environment = \"missing\"",
        ),
    )?;

    let output = workspace.run(&["serve", "start", "dsv4-qualify", "--dry-run"])?;

    assert!(!output.status.success());
    assert!(String::from_utf8(output.stderr)?.contains("unknown Pixi environment"));
    Ok(())
}

#[test]
fn integration_must_be_selected_by_the_pixi_manifest() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    fs::write(
        workspace.root.path().join("pixi.toml"),
        "[workspace]\n\
         channels = [\"conda-forge\"]\n\
         platforms = [\"linux-64\"]\n\
         \n\
         [environments]\n\
         vllm = []\n",
    )?;

    let output = workspace.run(&["serve", "start", "dsv4-qualify", "--dry-run"])?;

    assert!(!output.status.success());
    assert!(String::from_utf8(output.stderr)?.contains("is not selected by Pixi environment"));
    Ok(())
}

#[test]
fn dirty_submodule_state_changes_workspace_evidence() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    let origin = tempfile::tempdir()?;
    fs::write(origin.path().join("source.txt"), "submodule baseline\n")?;
    TestWorkspace::git(origin.path(), &["init", "-q"])?;
    TestWorkspace::git(origin.path(), &["config", "user.email", "test@example.com"])?;
    TestWorkspace::git(origin.path(), &["config", "user.name", "Inferlab Test"])?;
    TestWorkspace::git(origin.path(), &["add", "."])?;
    TestWorkspace::git(origin.path(), &["commit", "-qm", "submodule fixture"])?;

    TestWorkspace::git(workspace.root.path(), &["rm", "-qr", "vendor/flashinfer"])?;
    TestWorkspace::git(
        workspace.root.path(),
        &[
            "-c",
            "protocol.file.allow=always",
            "submodule",
            "add",
            "-q",
            origin.path().to_str().ok_or("non-UTF-8 test path")?,
            "vendor/flashinfer",
        ],
    )?;
    TestWorkspace::git(workspace.root.path(), &["commit", "-qam", "use submodule"])?;
    let clean = workspace.run_json(&["serve", "start", "dsv4-qualify", "--dry-run"])?;

    fs::write(
        workspace.root.path().join("vendor/flashinfer/source.txt"),
        "submodule local edit\n",
    )?;
    let dirty = workspace.run_json(&["serve", "start", "dsv4-qualify", "--dry-run"])?;

    assert_eq!(clean["workspace"]["dirty"], false);
    assert_eq!(dirty["workspace"]["dirty"], true);
    assert_ne!(
        clean["workspace"]["source_digest"],
        dirty["workspace"]["source_digest"]
    );
    Ok(())
}

/// A workspace whose vendor/flashinfer is a real file-protocol submodule,
/// for digest tests over submodule worktree state. The origin tempdir must
/// outlive the workspace.
fn workspace_with_file_submodule() -> Result<(TestWorkspace, tempfile::TempDir), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    let origin = tempfile::tempdir()?;
    fs::write(origin.path().join("real"), "hello\n")?;
    TestWorkspace::git(origin.path(), &["init", "-q"])?;
    TestWorkspace::git(origin.path(), &["config", "user.email", "test@example.com"])?;
    TestWorkspace::git(origin.path(), &["config", "user.name", "Inferlab Test"])?;
    TestWorkspace::git(origin.path(), &["add", "."])?;
    TestWorkspace::git(origin.path(), &["commit", "-qm", "submodule fixture"])?;
    TestWorkspace::git(workspace.root.path(), &["rm", "-qr", "vendor/flashinfer"])?;
    TestWorkspace::git(
        workspace.root.path(),
        &[
            "-c",
            "protocol.file.allow=always",
            "submodule",
            "add",
            "-q",
            origin.path().to_str().ok_or("non-UTF-8 test path")?,
            "vendor/flashinfer",
        ],
    )?;
    TestWorkspace::git(workspace.root.path(), &["commit", "-qam", "use submodule"])?;
    Ok((workspace, origin))
}

#[test]
fn submodule_local_refs_do_not_change_source_digest() -> Result<(), Box<dyn Error>> {
    let (workspace, _origin) = workspace_with_file_submodule()?;
    let submodule = workspace.root.path().join("vendor/flashinfer");
    let baseline = workspace.run_json(&["serve", "start", "dsv4-qualify", "--dry-run"])?;

    TestWorkspace::git(&submodule, &["tag", "local-presentation-only"])?;
    let tagged = workspace.run_json(&["serve", "start", "dsv4-qualify", "--dry-run"])?;

    assert_eq!(baseline["workspace"]["dirty"], false);
    assert_eq!(tagged["workspace"]["dirty"], false);
    assert_eq!(
        baseline["workspace"]["source_digest"], tagged["workspace"]["source_digest"],
        "a local ref must not enter workspace source identity"
    );

    TestWorkspace::git(
        &submodule,
        &[
            "-c",
            "user.email=test@example.com",
            "-c",
            "user.name=Inferlab Test",
            "commit",
            "-qm",
            "different effective submodule head",
            "--allow-empty",
        ],
    )?;
    let moved = workspace.run_json(&["serve", "start", "dsv4-qualify", "--dry-run"])?;
    assert_ne!(
        tagged["workspace"]["source_digest"], moved["workspace"]["source_digest"],
        "the effective submodule HEAD must remain part of workspace source identity"
    );

    TestWorkspace::git(&submodule, &["tag", "moved-presentation-only"])?;
    let moved_tagged = workspace.run_json(&["serve", "start", "dsv4-qualify", "--dry-run"])?;
    assert_eq!(
        moved["workspace"]["source_digest"], moved_tagged["workspace"]["source_digest"],
        "local refs must stay outside identity when the submodule HEAD differs from its gitlink"
    );
    Ok(())
}

/// A submodule untracked entry enters the source digest classified as the
/// top level classifies it ([[RFC-0002:C-WORKSPACE-AUTHORITY]]): a regular
/// file and a same-content symlink at the same path digest differently, and
/// the link's target text alone changes the digest.
#[test]
fn submodule_untracked_links_enter_the_source_digest() -> Result<(), Box<dyn Error>> {
    let (workspace, _origin) = workspace_with_file_submodule()?;
    let probe = workspace.root.path().join("vendor/flashinfer/probe");

    fs::write(&probe, "hello\n")?;
    let as_file = workspace.run_json(&["serve", "start", "dsv4-qualify", "--dry-run"])?;

    fs::remove_file(&probe)?;
    std::os::unix::fs::symlink("real", &probe)?;
    let as_link = workspace.run_json(&["serve", "start", "dsv4-qualify", "--dry-run"])?;
    assert_ne!(
        as_file["workspace"]["source_digest"], as_link["workspace"]["source_digest"],
        "a same-content link must not digest like the regular file it replaced"
    );

    fs::remove_file(&probe)?;
    std::os::unix::fs::symlink("./real", &probe)?;
    let retargeted = workspace.run_json(&["serve", "start", "dsv4-qualify", "--dry-run"])?;
    assert_ne!(
        as_link["workspace"]["source_digest"], retargeted["workspace"]["source_digest"],
        "the link target text alone must change the digest"
    );
    Ok(())
}

/// A dangling untracked link inside a submodule is a permitted shape
/// ([[RFC-0002:C-WORKSPACE-AUTHORITY]]) and no longer kills digest
/// computation; its text alone identifies it.
#[test]
fn dangling_submodule_links_do_not_kill_the_digest() -> Result<(), Box<dyn Error>> {
    let (workspace, _origin) = workspace_with_file_submodule()?;
    let probe = workspace.root.path().join("vendor/flashinfer/probe");

    std::os::unix::fs::symlink("missing", &probe)?;
    let dangling = workspace.run_json(&["serve", "start", "dsv4-qualify", "--dry-run"])?;
    let first = dangling["workspace"]["source_digest"]
        .as_str()
        .ok_or("dry run carries no source digest")?
        .to_owned();

    fs::remove_file(&probe)?;
    std::os::unix::fs::symlink("missing-elsewhere", &probe)?;
    let retargeted = workspace.run_json(&["serve", "start", "dsv4-qualify", "--dry-run"])?;
    assert_ne!(
        retargeted["workspace"]["source_digest"].as_str(),
        Some(first.as_str()),
        "the dangling link text alone must change the digest"
    );
    Ok(())
}

// (a) A workspace spread across the root file and two workspace.d fragments
// composes to the same definitions as the equivalent single file: the resolved
// server and measurement plan is identical. Only the file layout differs, so
// the workspace snapshot (digest, revision) is not compared.
#[test]
fn definitions_split_across_fragments_resolve_identically() -> Result<(), Box<dyn Error>> {
    // Capture the single-file baseline, then reorganize the same workspace onto
    // the fragment layout. Reusing one workspace keeps the model-weight locator
    // path identical, so the resolved server plan (including the adapter
    // request digest, which the locator flows into) can be compared exactly.
    let workspace = TestWorkspace::new()?;
    let single_plan = workspace.run_json(&["recipe", "run", "dsv4-qualify", "--dry-run"])?;

    workspace.split_workspace(
        SPLIT_ROOT,
        &[
            ("serving.toml", SPLIT_SERVING),
            ("measurements.toml", SPLIT_MEASUREMENTS),
        ],
    )?;
    let split_plan = workspace.run_json(&["recipe", "run", "dsv4-qualify", "--dry-run"])?;

    assert_eq!(split_plan["workspace"]["dirty"], false);
    // The composed workspace resolves the same server topology, settings,
    // parallelism, and every measurement definition as the single-file source.
    // The render-phase adapter digests and the runtime cache path derive from
    // the workspace source digest, which legitimately changes when the file
    // layout changes; strip those before comparing so the assertion pins the
    // resolved definitions rather than the on-disk file identity.
    assert_eq!(
        strip_source_derived(single_plan["server"].clone()),
        strip_source_derived(split_plan["server"].clone()),
    );
    assert_eq!(split_plan["measurements"], single_plan["measurements"]);
    assert_eq!(split_plan["recipe"], single_plan["recipe"]);
    // The full serving definition the adapter plans against (its request and
    // response digest) is identical, so the composed definitions match byte for
    // byte through the resolution the source digest does not touch.
    assert_eq!(
        split_plan["server"]["integration"]["plan_request_sha256"],
        single_plan["server"]["integration"]["plan_request_sha256"],
    );
    assert_eq!(
        split_plan["server"]["integration"]["plan_response_sha256"],
        single_plan["server"]["integration"]["plan_response_sha256"],
    );
    Ok(())
}

/// Drop the resolved-server fields that derive from the workspace source
/// digest (the render-phase adapter digests and every runtime cache subtree),
/// so two workspaces with identical definitions but different file layouts
/// compare equal on the definition-derived content.
fn strip_source_derived(mut server: Value) -> Value {
    if let Some(integration) = server.get_mut("integration").and_then(Value::as_object_mut) {
        integration.remove("render_request_sha256");
        integration.remove("render_response_sha256");
    }
    if let Some(roles) = server.get_mut("roles").and_then(Value::as_array_mut) {
        for role in roles {
            if let Some(replicas) = role.get_mut("replicas").and_then(Value::as_array_mut) {
                for replica in replicas {
                    if let Some(ranks) = replica.get_mut("ranks").and_then(Value::as_array_mut) {
                        for rank in ranks {
                            // The rendered command embeds cache-root paths in its env; the
                            // plan-phase resolution above already pins the definitions.
                            if let Some(rank) = rank.as_object_mut() {
                                rank.remove("runtime_cache");
                                rank.remove("command");
                            }
                        }
                    }
                }
            }
        }
    }
    server
}

// (b) An identifier declared in two workspace files is rejected at load. The
// collision is detected both across the root and a fragment, and across two
// fragments, and the message names the section, the identifier, and both
// files.
#[test]
fn identifier_declared_by_two_files_is_rejected_naming_both() -> Result<(), Box<dyn Error>> {
    // Root + fragment collision: the root file declares model "dsv4" (it lives
    // in SPLIT_SERVING, so a root variant that inlines the models section
    // collides against a fragment still supplying it). The root file is always
    // named first: its declarations occupy the composed map before any
    // fragment is visited, and an occupant without fragment provenance is
    // attributed to the root file.
    let root_with_model = format!("{SPLIT_ROOT}\n[models.dsv4]\nserved_name = \"dsv4\"\n");
    let root_fragment = TestWorkspace::new()?;
    root_fragment.split_workspace(
        &root_with_model,
        &[
            ("serving.toml", SPLIT_SERVING),
            ("measurements.toml", SPLIT_MEASUREMENTS),
        ],
    )?;
    let output = root_fragment.run(&["recipe", "run", "dsv4-qualify", "--dry-run"])?;
    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr)?;
    assert!(
        stderr.contains(
            "model \"dsv4\" is declared by both .inferlab/workspace.toml \
             and .inferlab/workspace.d/serving.toml"
        ),
        "root+fragment collision message was: {stderr}"
    );

    // Fragment + fragment collision: two fragments both declare eval "smoke".
    // Sorted filename order fixes which file is named first: a-dup.toml sorts
    // before measurements.toml, so measurements.toml is the second declarer.
    let two_fragments = TestWorkspace::new()?;
    two_fragments.split_workspace(
        SPLIT_ROOT,
        &[
            ("serving.toml", SPLIT_SERVING),
            ("measurements.toml", SPLIT_MEASUREMENTS),
            (
                "a-dup.toml",
                "[evals.smoke]\n\
                 kind = \"openai-smoke\"\n\
                 prompt = \"duplicate\"\n\
                 max_tokens = 8\n\
                 timeout_seconds = 30\n",
            ),
        ],
    )?;
    let output = two_fragments.run(&["recipe", "run", "dsv4-qualify", "--dry-run"])?;
    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr)?;
    assert!(
        stderr.contains(
            "eval \"smoke\" is declared by both .inferlab/workspace.d/a-dup.toml \
             and .inferlab/workspace.d/measurements.toml"
        ),
        "fragment+fragment collision message was: {stderr}"
    );
    Ok(())
}

// (c) A fragment that declares schema_version is rejected at load with a
// message naming the fragment; the scalar lives only in the root file.
#[test]
fn schema_version_in_a_fragment_is_rejected() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    let serving_with_scalar = format!("schema_version = 1\n\n{SPLIT_SERVING}");
    workspace.split_workspace(
        SPLIT_ROOT,
        &[
            ("serving.toml", &serving_with_scalar),
            ("measurements.toml", SPLIT_MEASUREMENTS),
        ],
    )?;
    let output = workspace.run(&["recipe", "run", "dsv4-qualify", "--dry-run"])?;
    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr)?;
    assert!(
        stderr.contains(
            "workspace fragment .inferlab/workspace.d/serving.toml declares schema_version, \
             which lives only in the root workspace file .inferlab/workspace.toml"
        ),
        "schema_version rejection message was: {stderr}"
    );
    Ok(())
}

// (d) A workspace.d directory with no fragments composes to exactly the
// single-file result; the existing single-file loader path (exercised by
// `serve_and_recipe_dry_run_share_the_default_case`, which builds the fixture
// with no workspace.d directory at all) is unchanged by construction.
#[test]
fn empty_fragment_directory_leaves_the_single_file_workspace_unchanged()
-> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    fs::create_dir_all(workspace.root.path().join(".inferlab/workspace.d"))?;
    // A non-toml file and a subdirectory under workspace.d are ignored.
    fs::write(
        workspace
            .root
            .path()
            .join(".inferlab/workspace.d/README.md"),
        "notes\n",
    )?;
    fs::create_dir_all(workspace.root.path().join(".inferlab/workspace.d/nested"))?;
    TestWorkspace::git(workspace.root.path(), &["add", "-A"])?;
    TestWorkspace::git(
        workspace.root.path(),
        &["commit", "-qm", "empty fragment dir"],
    )?;

    let plan = workspace.run_json(&["recipe", "run", "dsv4-qualify", "--dry-run"])?;
    assert_eq!(plan["workspace"]["dirty"], false);
    assert_eq!(plan["server"]["case"]["id"], "tp2");
    assert_eq!(plan["measurements"]["gate"], "gsm8k");
    Ok(())
}

// A symbolic link anywhere shareable workspace content lives escapes the
// source digest — the digest records link text, not target content — so the
// loader rejects all three shapes: a linked fragment, a linked workspace.d
// directory, and a linked root workspace file.
#[test]
fn symlinked_workspace_files_are_rejected() -> Result<(), Box<dyn Error>> {
    // Linked fragment: a *.toml symlink under workspace.d is an error, not a
    // followed file and not a silently ignored one.
    let fragment_link = TestWorkspace::new()?;
    fragment_link.split_workspace(
        SPLIT_ROOT,
        &[
            ("serving.toml", SPLIT_SERVING),
            ("measurements.toml", SPLIT_MEASUREMENTS),
        ],
    )?;
    let outside = fragment_link.root.path().join("outside.toml");
    fs::write(
        &outside,
        "[models.outside]\nweight = \"x\"\nserved_name = \"x\"\n",
    )?;
    std::os::unix::fs::symlink(
        &outside,
        fragment_link
            .root
            .path()
            .join(".inferlab/workspace.d/extra.toml"),
    )?;
    let output = fragment_link.run(&["recipe", "run", "dsv4-qualify", "--dry-run"])?;
    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr)?;
    assert!(
        stderr.contains(
            "workspace fragment .inferlab/workspace.d/extra.toml must be a regular \
             filesystem entry, not a symbolic link; the workspace source digest \
             records link text rather than target content"
        ),
        "fragment symlink rejection message was: {stderr}"
    );

    // Linked workspace.d directory.
    let dir_link = TestWorkspace::new()?;
    let real_dir = dir_link.root.path().join("fragments-elsewhere");
    fs::create_dir_all(&real_dir)?;
    std::os::unix::fs::symlink(
        &real_dir,
        dir_link.root.path().join(".inferlab/workspace.d"),
    )?;
    let output = dir_link.run(&["recipe", "run", "dsv4-qualify", "--dry-run"])?;
    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr)?;
    assert!(
        stderr.contains(
            ".inferlab/workspace.d must be a regular filesystem entry, not a symbolic link"
        ),
        "workspace.d symlink rejection message was: {stderr}"
    );

    // Linked root workspace file.
    let root_link = TestWorkspace::new()?;
    let inferlab = root_link.root.path().join(".inferlab");
    fs::rename(
        inferlab.join("workspace.toml"),
        inferlab.join("workspace-real.toml"),
    )?;
    std::os::unix::fs::symlink(
        inferlab.join("workspace-real.toml"),
        inferlab.join("workspace.toml"),
    )?;
    let output = root_link.run(&["recipe", "run", "dsv4-qualify", "--dry-run"])?;
    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr)?;
    assert!(
        stderr.contains(
            ".inferlab/workspace.toml must be a regular filesystem entry, not a symbolic link"
        ),
        "root symlink rejection message was: {stderr}"
    );
    Ok(())
}

// Fragment type errors carry TOML line/column like the root file: the typed
// parse re-reads the source text instead of converting the span-less table.
#[test]
fn fragment_type_errors_name_their_position() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    workspace.split_workspace(
        SPLIT_ROOT,
        &[
            ("serving.toml", SPLIT_SERVING),
            ("measurements.toml", SPLIT_MEASUREMENTS),
            (
                "broken.toml",
                "[models.broken]\nweight = 5\nserved_name = \"x\"\n",
            ),
        ],
    )?;
    let output = workspace.run(&["recipe", "run", "dsv4-qualify", "--dry-run"])?;
    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr)?;
    assert!(
        stderr.contains("broken.toml") && stderr.contains("line 2"),
        "fragment type error lost its position: {stderr}"
    );
    Ok(())
}

// A symlinked `.inferlab` directory routes every final-node guard through the
// link (symlink_metadata follows intermediate components), so the shared
// parent is guarded first.
#[test]
fn symlinked_inferlab_directory_is_rejected() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    let root = workspace.root.path();
    fs::rename(root.join(".inferlab"), root.join(".inferlab-real"))?;
    std::os::unix::fs::symlink(root.join(".inferlab-real"), root.join(".inferlab"))?;
    let output = workspace.run(&["recipe", "run", "dsv4-qualify", "--dry-run"])?;
    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr)?;
    assert!(
        stderr.contains(".inferlab must be a regular filesystem entry, not a symbolic link"),
        ".inferlab symlink rejection message was: {stderr}"
    );
    Ok(())
}

// A declared stack source path must be symlink-free along every component: a
// linked declared root and a linked intermediate directory both escape the
// source digest identically (git records link text, not target content).
#[test]
fn symlinked_stack_source_components_are_rejected() -> Result<(), Box<dyn Error>> {
    // Declared root: vendor/flashinfer becomes a link to a real directory.
    let linked_root = TestWorkspace::new()?;
    let root = linked_root.root.path();
    fs::remove_dir_all(root.join("vendor/flashinfer"))?;
    fs::create_dir_all(root.join("flashinfer-elsewhere"))?;
    std::os::unix::fs::symlink(
        root.join("flashinfer-elsewhere"),
        root.join("vendor/flashinfer"),
    )?;
    let output = linked_root.run(&["recipe", "run", "dsv4-qualify", "--dry-run"])?;
    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr)?;
    assert!(
        stderr.contains(
            "stack \"vllm\" source path component vendor/flashinfer must be a regular \
             filesystem entry, not a symbolic link"
        ),
        "linked stack-source root rejection message was: {stderr}"
    );

    // Intermediate component: vendor itself becomes a link.
    let linked_parent = TestWorkspace::new()?;
    let root = linked_parent.root.path();
    fs::rename(root.join("vendor"), root.join("vendor-elsewhere"))?;
    std::os::unix::fs::symlink(root.join("vendor-elsewhere"), root.join("vendor"))?;
    let output = linked_parent.run(&["recipe", "run", "dsv4-qualify", "--dry-run"])?;
    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr)?;
    assert!(
        stderr.contains(
            "stack \"vllm\" source path component vendor must be a regular \
             filesystem entry, not a symbolic link"
        ),
        "linked intermediate component rejection message was: {stderr}"
    );
    Ok(())
}

/// Symlinks whose targets leave the workspace root are rejected when the
/// snapshot claims source identity ([[RFC-0002:C-WORKSPACE-AUTHORITY]]): the
/// digest records only link text, so out-of-root bytes could drift without
/// changing the recorded identity.
#[test]
fn escaping_source_links_are_rejected() -> Result<(), Box<dyn Error>> {
    // An absolute target, even a dangling one, is machine-specific link text.
    let absolute = TestWorkspace::new()?;
    let root = absolute.root.path();
    std::os::unix::fs::symlink(
        "/outside-nowhere/module.py",
        root.join("vendor/vllm/absolute-link"),
    )?;
    TestWorkspace::git(root, &["add", "."])?;
    TestWorkspace::git(root, &["commit", "-qm", "absolute link"])?;
    let output = absolute.run(&["recipe", "run", "dsv4-qualify", "--dry-run"])?;
    assert!(!output.status.success());

    // A relative target that lexically steps above the workspace root.
    let escaping = TestWorkspace::new()?;
    let root = escaping.root.path();
    std::os::unix::fs::symlink(
        "../../../outside/module.py",
        root.join("vendor/vllm/escape-link"),
    )?;
    TestWorkspace::git(root, &["add", "."])?;
    TestWorkspace::git(root, &["commit", "-qm", "escaping link"])?;
    let output = escaping.run(&["recipe", "run", "dsv4-qualify", "--dry-run"])?;
    assert!(!output.status.success());

    // An internal-looking link routing through an escaping intermediate is
    // caught through the intermediate's own rejection: resolution stays
    // lexical because every link is enumerated on its own.
    let chained = TestWorkspace::new()?;
    let root = chained.root.path();
    std::os::unix::fs::symlink("../../../outside-dir", root.join("vendor/vllm/mid"))?;
    std::os::unix::fs::symlink("mid/module.py", root.join("vendor/vllm/deep"))?;
    TestWorkspace::git(root, &["add", "."])?;
    TestWorkspace::git(root, &["commit", "-qm", "chained links"])?;
    let output = chained.run(&["recipe", "run", "dsv4-qualify", "--dry-run"])?;
    assert!(!output.status.success());
    Ok(())
}

/// Containment covers the digested worktree, not only stack-source subtrees
/// ([[RFC-0002:C-WORKSPACE-AUTHORITY]]): a root-level bridge link outside
/// every stack source is digested as link text, so a stack-source link resolving
/// onto it was a two-hop escape until the walk enumerated the bridge itself.
#[test]
fn out_of_stack_source_bridge_links_are_contained() -> Result<(), Box<dyn Error>> {
    // Resolving ONTO the bridge: the bridge's own verdict names the escape.
    let onto = TestWorkspace::new()?;
    let root = onto.root.path();
    std::os::unix::fs::symlink("/outside-nowhere", root.join("bridge"))?;
    std::os::unix::fs::symlink("../../bridge", root.join("vendor/vllm/deep"))?;
    TestWorkspace::git(root, &["add", "."])?;
    TestWorkspace::git(root, &["commit", "-qm", "bridge"])?;
    let output = onto.run(&["recipe", "run", "dsv4-qualify", "--dry-run"])?;
    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr)?;
    assert!(
        stderr.contains(
            "workspace symlink bridge targets absolute path /outside-nowhere; the \
             workspace source digest records link text rather than target content"
        ),
        "the bridge outside every stack source is rejected on its own: {stderr}"
    );

    // Resolving THROUGH the bridge: still a containment verdict, not a git
    // hard error about pathspecs beyond a symbolic link.
    let through = TestWorkspace::new()?;
    let root = through.root.path();
    std::os::unix::fs::symlink("/outside-nowhere", root.join("bridge"))?;
    std::os::unix::fs::symlink("../../bridge/module.py", root.join("vendor/vllm/deep"))?;
    TestWorkspace::git(root, &["add", "."])?;
    TestWorkspace::git(root, &["commit", "-qm", "bridge"])?;
    let output = through.run(&["recipe", "run", "dsv4-qualify", "--dry-run"])?;
    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr)?;
    assert!(
        stderr.contains("workspace symlink bridge targets absolute path /outside-nowhere"),
        "the through-link shape gets a containment verdict: {stderr}"
    );
    assert!(
        !stderr.contains("beyond a symbolic link") && !stderr.contains("git command failed"),
        "no git pathspec hard error may stand in for containment: {stderr}"
    );
    Ok(())
}

/// A benign in-root chain through a covered link directory is accepted: the
/// ignore judgment runs on the link-resolved destination, because git
/// refuses pathspecs beyond a symbolic link
/// ([[RFC-0002:C-WORKSPACE-AUTHORITY]]).
#[test]
fn chains_through_covered_link_directories_are_accepted() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    let root = workspace.root.path();
    fs::create_dir(root.join("vendor/vllm/real-dir"))?;
    fs::write(root.join("vendor/vllm/real-dir/module.py"), "content\n")?;
    std::os::unix::fs::symlink("real-dir", root.join("vendor/vllm/dir-link"))?;
    std::os::unix::fs::symlink("dir-link/module.py", root.join("vendor/vllm/through"))?;
    TestWorkspace::git(root, &["add", "."])?;
    TestWorkspace::git(root, &["commit", "-qm", "benign chain"])?;
    let output = workspace.run(&["recipe", "run", "dsv4-qualify", "--dry-run"])?;
    assert!(
        output.status.success(),
        "a covered chain must pass containment: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    Ok(())
}

/// A digest-visible link resolving onto or through a machine-local link is
/// rejected: the machine-local link's text is outside the recorded
/// identity, so retargeting it would change effective content under an
/// unchanged digest ([[RFC-0002:C-WORKSPACE-AUTHORITY]]).
#[test]
fn digest_visible_links_may_not_ride_machine_local_links() -> Result<(), Box<dyn Error>> {
    // ONTO: a tracked link pointing at a git-ignored link.
    let onto = TestWorkspace::new()?;
    let root = onto.root.path();
    fs::write(
        root.join(".gitignore"),
        ".inferlab/local.toml\nvendor/vllm/bridge-ig\n",
    )?;
    fs::write(root.join("vendor/vllm/a.py"), "content a\n")?;
    std::os::unix::fs::symlink("bridge-ig", root.join("vendor/vllm/deep"))?;
    TestWorkspace::git(root, &["add", "."])?;
    TestWorkspace::git(root, &["commit", "-qm", "onto shape"])?;
    std::os::unix::fs::symlink("a.py", root.join("vendor/vllm/bridge-ig"))?;
    let output = onto.run(&["recipe", "run", "dsv4-qualify", "--dry-run"])?;
    assert!(!output.status.success());

    // THROUGH: a tracked link routing through a git-ignored link directory.
    let through = TestWorkspace::new()?;
    let root = through.root.path();
    fs::write(
        root.join(".gitignore"),
        ".inferlab/local.toml\nvendor/vllm/ig-dir\n",
    )?;
    fs::create_dir(root.join("vendor/vllm/real-dir"))?;
    fs::write(root.join("vendor/vllm/real-dir/module.py"), "content\n")?;
    std::os::unix::fs::symlink("ig-dir/module.py", root.join("vendor/vllm/deep"))?;
    TestWorkspace::git(root, &["add", "."])?;
    TestWorkspace::git(root, &["commit", "-qm", "through shape"])?;
    std::os::unix::fs::symlink("real-dir", root.join("vendor/vllm/ig-dir"))?;
    let output = through.run(&["recipe", "run", "dsv4-qualify", "--dry-run"])?;
    assert!(!output.status.success());
    Ok(())
}

/// A substitution chain that revisits a link is a cycle and is rejected
/// naming the starting link and its target
/// ([[RFC-0002:C-WORKSPACE-AUTHORITY]]).
#[test]
fn symlink_cycles_are_rejected() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    let root = workspace.root.path();
    std::os::unix::fs::symlink("cycle-b", root.join("vendor/vllm/cycle-a"))?;
    std::os::unix::fs::symlink("cycle-a", root.join("vendor/vllm/cycle-b"))?;
    TestWorkspace::git(root, &["add", "."])?;
    TestWorkspace::git(root, &["commit", "-qm", "cycle"])?;
    let output = workspace.run(&["recipe", "run", "dsv4-qualify", "--dry-run"])?;
    assert!(!output.status.success());
    Ok(())
}

/// Containment covers every symlink effectively present in the worktree:
/// untracked, git-ignored, and index-type-replaced escaping links carry the
/// same digest blindness as tracked ones, and the ignored shape is invisible
/// to the dirty gate entirely.
#[test]
fn uncovered_links_are_rejected_regardless_of_tracking_state() -> Result<(), Box<dyn Error>> {
    // Untracked escaping link: dirty, but dirtiness does not exempt it.
    let untracked = TestWorkspace::new()?;
    let root = untracked.root.path();
    std::os::unix::fs::symlink(
        "/outside-nowhere/module.py",
        root.join("vendor/vllm/untracked-escape"),
    )?;
    let output = untracked.run(&["recipe", "run", "dsv4-qualify", "--dry-run"])?;
    assert!(!output.status.success());

    // Ignored escaping link: git status and the digest see nothing at all,
    // which is exactly why the walk must.
    let ignored = TestWorkspace::new()?;
    let root = ignored.root.path();
    fs::write(
        root.join(".gitignore"),
        ".inferlab/local.toml\nvendor/vllm/ignored-escape\n",
    )?;
    TestWorkspace::git(root, &["add", "."])?;
    TestWorkspace::git(root, &["commit", "-qm", "ignore the link"])?;
    std::os::unix::fs::symlink(
        "/outside-nowhere/module.py",
        root.join("vendor/vllm/ignored-escape"),
    )?;
    let output = ignored.run(&["recipe", "run", "dsv4-qualify", "--dry-run"])?;
    assert!(!output.status.success());

    // A tracked regular file replaced in the worktree by an escaping link.
    let replaced = TestWorkspace::new()?;
    let root = replaced.root.path();
    fs::write(root.join("vendor/vllm/swapped.py"), "original\n")?;
    TestWorkspace::git(root, &["add", "."])?;
    TestWorkspace::git(root, &["commit", "-qm", "regular file"])?;
    fs::remove_file(root.join("vendor/vllm/swapped.py"))?;
    std::os::unix::fs::symlink(
        "/outside-nowhere/swapped.py",
        root.join("vendor/vllm/swapped.py"),
    )?;
    let output = replaced.run(&["recipe", "run", "dsv4-qualify", "--dry-run"])?;
    assert!(!output.status.success());
    Ok(())
}

/// A lexically internal target is not enough: it must be identity-covered.
/// Source exclusions, git metadata, and git-ignored content never enter the
/// digest, so links into them let uncovered bytes wear a covered identity.
#[test]
fn identity_uncovered_targets_are_rejected() -> Result<(), Box<dyn Error>> {
    // A target inside a workspace source exclusion; rejected even though the
    // path is dangling, because the excluded namespace fills at runtime.
    let excluded = TestWorkspace::new()?;
    let root = excluded.root.path();
    std::os::unix::fs::symlink(
        "../../.inferlab/cache/generated.py",
        root.join("vendor/vllm/cache-link"),
    )?;
    TestWorkspace::git(root, &["add", "."])?;
    TestWorkspace::git(root, &["commit", "-qm", "cache link"])?;
    let output = excluded.run(&["recipe", "run", "dsv4-qualify", "--dry-run"])?;
    assert!(!output.status.success());

    // A tracked link to a git-ignored target: the link is committed and the
    // tree is clean, yet the target's bytes are outside the digest.
    let ignored_target = TestWorkspace::new()?;
    let root = ignored_target.root.path();
    fs::write(
        root.join(".gitignore"),
        ".inferlab/local.toml\nvendor/vllm/generated.py\n",
    )?;
    std::os::unix::fs::symlink("generated.py", root.join("vendor/vllm/gen-link"))?;
    TestWorkspace::git(root, &["add", "."])?;
    TestWorkspace::git(root, &["commit", "-qm", "link to ignored"])?;
    fs::write(root.join("vendor/vllm/generated.py"), "uncovered\n")?;
    let output = ignored_target.run(&["recipe", "run", "dsv4-qualify", "--dry-run"])?;
    assert!(!output.status.success());

    // A target inside git metadata.
    let git_target = TestWorkspace::new()?;
    let root = git_target.root.path();
    std::os::unix::fs::symlink("../../.git/config", root.join("vendor/vllm/git-link"))?;
    let output = git_target.run(&["recipe", "run", "dsv4-qualify", "--dry-run"])?;
    assert!(!output.status.success());
    Ok(())
}

/// A submodule's own ignore rules govern targets inside it, and the walk
/// enumerates links across the submodule boundary as plain directories.
#[test]
fn submodule_ignore_rules_govern_submodule_targets() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    let root = workspace.root.path();
    let sub_src = root.join("sub-src");
    fs::create_dir_all(&sub_src)?;
    fs::write(sub_src.join(".gitignore"), "generated.py\n")?;
    fs::write(sub_src.join("module.py"), "covered\n")?;
    TestWorkspace::git(&sub_src, &["init", "-q"])?;
    TestWorkspace::git(&sub_src, &["config", "user.email", "test@example.com"])?;
    TestWorkspace::git(&sub_src, &["config", "user.name", "Inferlab Test"])?;
    TestWorkspace::git(&sub_src, &["add", "."])?;
    TestWorkspace::git(&sub_src, &["commit", "-qm", "sub"])?;
    TestWorkspace::git(
        root,
        &[
            "-c",
            "protocol.file.allow=always",
            "submodule",
            "add",
            "-q",
            "./sub-src",
            "vendor/vllm/subrepo",
        ],
    )?;
    TestWorkspace::git(root, &["commit", "-qm", "add submodule"])?;
    // Ignored by the submodule's rules, invisible to the parent's.
    fs::write(root.join("vendor/vllm/subrepo/generated.py"), "uncovered\n")?;
    std::os::unix::fs::symlink("generated.py", root.join("vendor/vllm/subrepo/inner-link"))?;
    let output = workspace.run(&["recipe", "run", "dsv4-qualify", "--dry-run"])?;
    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr)?;
    assert!(
        stderr.contains(
            "symlink vendor/vllm/subrepo/inner-link targets generated.py, which \
             resolves to git-ignored content at vendor/vllm/subrepo/generated.py"
        ),
        "submodule-ignored-target rejection message was: {stderr}"
    );
    Ok(())
}

/// Identity-covered internal targets stay permitted regardless of the link's
/// tracking state, and a dangling internal target is identified by its link
/// text alone.
#[test]
fn internal_source_links_are_permitted() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    let root = workspace.root.path();
    // Contains `..` but lexically stays inside the root.
    std::os::unix::fs::symlink(
        "../flashinfer/source.txt",
        root.join("vendor/vllm/sibling-link"),
    )?;
    std::os::unix::fs::symlink("source.txt", root.join("vendor/vllm/local-link"))?;
    std::os::unix::fs::symlink("missing-file.py", root.join("vendor/vllm/dangling-link"))?;
    TestWorkspace::git(root, &["add", "."])?;
    TestWorkspace::git(root, &["commit", "-qm", "internal links"])?;
    // An untracked internal link to covered content: ordinary dirty state.
    std::os::unix::fs::symlink("source.txt", root.join("vendor/vllm/untracked-internal"))?;
    let output = workspace.run(&["recipe", "run", "dsv4-qualify", "--dry-run"])?;
    assert!(
        output.status.success(),
        "identity-covered internal links must be permitted: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    Ok(())
}

/// Git-ignored links are machine-local state bound by containment alone —
/// the two shapes real trees plant: an editable install's absolute link to
/// in-root content, and a build checkout's ignored-to-ignored internal link.
#[test]
fn ignored_links_to_in_root_content_are_machine_local() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    let root = workspace.root.path();
    fs::write(
        root.join(".gitignore"),
        ".inferlab/local.toml\nvendor/vllm/data-link\nvendor/vllm/.deps/\n",
    )?;
    TestWorkspace::git(root, &["add", "."])?;
    TestWorkspace::git(root, &["commit", "-qm", "ignore machine-local links"])?;
    // The flashinfer editable-install shape: ignored link, absolute target
    // resolving under this workspace root.
    std::os::unix::fs::symlink(
        root.canonicalize()?.join("vendor/flashinfer/source.txt"),
        root.join("vendor/vllm/data-link"),
    )?;
    // The vllm .deps shape: ignored link to an ignored internal target.
    fs::create_dir_all(root.join("vendor/vllm/.deps"))?;
    fs::write(root.join("vendor/vllm/.deps/notes.md"), "machine local\n")?;
    std::os::unix::fs::symlink("notes.md", root.join("vendor/vllm/.deps/notes-link"))?;
    let output = workspace.run(&["recipe", "run", "dsv4-qualify", "--dry-run"])?;
    assert!(
        output.status.success(),
        "ignored links to in-root content must be permitted: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    Ok(())
}

#[test]
fn missing_local_bindings_error_guides_the_operator() -> Result<(), Box<dyn Error>> {
    // A fresh workspace before any bindings exist: the first error a new
    // operator sees names what the file is for, not a bare OS error.
    let root = tempfile::tempdir()?;
    fs::create_dir_all(root.path().join(".inferlab"))?;
    fs::write(
        root.path().join(".inferlab/workspace.toml"),
        "schema_version = 1\n",
    )?;
    let output = Command::new(env!("CARGO_BIN_EXE_inferlab"))
        .current_dir(root.path())
        .args(["recipe", "run", "any", "--dry-run"])
        .output()?;
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("machine-private facts"), "{stderr}");
    assert!(stderr.contains("--local <FILE>"), "{stderr}");
    Ok(())
}

/// An adapter that answers a well-formed success response but stamps it with
/// protocol version 2: the cross-version combination the wheel-distribution
/// switch makes constructible ([[RFC-0006:C-INTEGRATIONS]]).
const WRONG_VERSION_ADAPTER: &str = r#"#!/usr/bin/env python3
import json
import sys

json.load(sys.stdin)
print(json.dumps({
    "status": "ok",
    "protocol_version": "2",
    "result": {"operation": "plan_serve", "output": {}},
}))
"#;

/// An adapter that recognizes the mismatch itself and answers a structured
/// unsupported-protocol-version rejection naming both versions.
const UNSUPPORTED_VERSION_ADAPTER: &str = r#"#!/usr/bin/env python3
import json
import sys

json.load(sys.stdin)
print(json.dumps({
    "status": "error",
    "protocol_version": "7",
    "error": {
        "code": "unsupported_protocol_version",
        "message": "received protocol version 7; this integration supports protocol version 6",
    },
}))
"#;

#[test]
fn protocol_version_mismatch_names_both_versions_and_the_remedy() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;

    // A raw-stamped foreign version is caught before the response even
    // deserializes; the failure names both versions and the remedy.
    write_executable(
        &workspace.adapter_bin.join("inferlab-adapter-vllm"),
        WRONG_VERSION_ADAPTER,
    )?;
    let output = workspace.run(&["recipe", "run", "dsv4-qualify", "--dry-run"])?;
    assert!(
        !output.status.success(),
        "a protocol version 2 answer must fail the command"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("protocol version 2") && stderr.contains("protocol version 7"),
        "the mismatch names both versions: {stderr}"
    );
    assert!(
        stderr.contains("bump the workspace adapter pins and relock")
            && stderr.contains("run a release whose binary speaks"),
        "the mismatch names the remedy: {stderr}"
    );

    // A structured unsupported-protocol-version rejection surfaces the same
    // both-versions-plus-remedy shape.
    write_executable(
        &workspace.adapter_bin.join("inferlab-adapter-vllm"),
        UNSUPPORTED_VERSION_ADAPTER,
    )?;
    let output = workspace.run(&["recipe", "run", "dsv4-qualify", "--dry-run"])?;
    assert!(
        !output.status.success(),
        "a structured unsupported-protocol-version rejection must fail the command"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("protocol version 7") && stderr.contains("protocol version 6"),
        "the structured rejection names both versions: {stderr}"
    );
    assert!(
        stderr.contains("bump the workspace adapter pins and relock"),
        "the structured rejection names the remedy: {stderr}"
    );
    Ok(())
}
