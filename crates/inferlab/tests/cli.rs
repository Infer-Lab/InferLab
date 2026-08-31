use std::error::Error;
use std::process::Command;

#[test]
fn help_is_a_runnable_minimal_surface() -> Result<(), Box<dyn Error>> {
    let output = Command::new(env!("CARGO_BIN_EXE_inferlab"))
        .arg("--help")
        .output()?;

    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout)?;
    assert!(stdout.contains("Usage: inferlab"));
    assert!(stdout.contains("Run reproducible LLM inference experiments"));
    for subcommand in [
        "tui",
        "workspace",
        "stack",
        "toolchain",
        "serve",
        "recipe",
        "bench",
        "run",
        "image",
        "scratchpad",
        "agent",
        "license",
    ] {
        assert!(
            stdout
                .lines()
                .any(|line| line.split_whitespace().next() == Some(subcommand)),
            "help must advertise the {subcommand:?} subcommand: {stdout}"
        );
    }
    assert!(
        !stdout.contains("__internal"),
        "help must not advertise the hidden __internal command: {stdout}"
    );
    Ok(())
}

#[test]
fn detailed_help_states_the_operator_boundaries_it_owns() -> Result<(), Box<dyn Error>> {
    const CASES: &[(&[&str], &[&str])] = &[
        (&[], &["file-first evidence", "--dry-run"]),
        (&["tui"], &["view-only", "does not launch workflows"]),
        (
            &["stack", "status"],
            &["does not require machine-local bindings", "repair hint"],
        ),
        (
            &["toolchain", "install"],
            &["lm-eval and AIPerf", "serving stack's Pixi environment"],
        ),
        (
            &["serve", "start"],
            &[
                "creates a server record before launch",
                "integration planning still runs",
            ],
        ),
        (
            &["recipe", "run"],
            &[
                "recorded closed loop",
                "Failure still finalizes",
                "kind = \"openai-smoke\"",
            ],
        ),
        (
            &["bench"],
            &[
                "explicit running managed-server record",
                "sends no measurement traffic",
                "input_tokens = 8192",
                "input_tokens = { min = 6553, max = 8192 }",
            ],
        ),
        (
            &["workspace", "show"],
            &[
                "canonical default-expanded",
                "inferlab workspace show --json",
            ],
        ),
        (
            &["run"],
            &[
                "writes no execution record",
                "no host mount or device implicitly",
            ],
        ),
        (
            &["image", "build"],
            &["requires a clean workspace", "never pushes to a registry"],
        ),
        (
            &["scratchpad", "note"],
            &["Entries may link existing records", "newest local record"],
        ),
        (
            &["scratchpad", "show"],
            &["recent tail", "never alters workspace resolution"],
        ),
        (
            &["agent", "install"],
            &["embedded in this binary", "one JSON report"],
        ),
        (
            &["agent", "doctor"],
            &["Read-only diagnosis", "does not install, update, or remove"],
        ),
    ];

    for (args, expected) in CASES {
        let output = Command::new(env!("CARGO_BIN_EXE_inferlab"))
            .args(*args)
            .arg("--help")
            .output()?;
        assert!(
            output.status.success(),
            "help failed for {args:?}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let stdout = String::from_utf8(output.stdout)?;
        for text in *expected {
            assert!(
                stdout.contains(text),
                "help for {args:?} omitted {text:?}: {stdout}"
            );
        }
    }
    Ok(())
}

#[test]
fn recipe_help_advertises_only_valid_measurement_override_paths() -> Result<(), Box<dyn Error>> {
    let output = Command::new(env!("CARGO_BIN_EXE_inferlab"))
        .args(["recipe", "run", "--help"])
        .output()?;

    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout)?;
    assert!(stdout.contains("evals.gsm8k.limit=10"), "{stdout}");
    assert!(!stdout.contains("evals.smoke.limit"), "{stdout}");
    Ok(())
}

#[test]
fn version_output_states_the_adapter_protocol_version() -> Result<(), Box<dyn Error>> {
    let output = Command::new(env!("CARGO_BIN_EXE_inferlab"))
        .arg("--version")
        .output()?;

    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout)?;
    let protocol = inferlab_protocol::ProtocolVersion::CURRENT;
    assert_eq!(
        stdout.trim(),
        format!(
            "inferlab {} (adapter protocol v{})",
            env!("CARGO_PKG_VERSION"),
            protocol.as_str()
        ),
        "version output must state the wire protocol version: {stdout}"
    );
    Ok(())
}
