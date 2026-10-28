//! Pixi activation projection and the generated runtime entrypoint contract.

use crate::InferlabError;
use crate::environment::PIXI_MANIFEST;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

pub(crate) const ENTRYPOINT_PATH: &str = "/usr/local/bin/inferlab-entrypoint";
pub(crate) const ENV_PREFIX: &str = "/opt/inferlab-env";

/// The entrypoint/command contract contribution to the content closure. The
/// contract is hashed over the rendered script so activation projections are
/// behavior-affecting closure inputs, while the closure map itself carries no
/// in-image program paths.
pub(super) fn entrypoint_contract_digest(rendered_entrypoint: &str) -> String {
    base16ct::lower::encode_string(&Sha256::digest(
        format!("{ENTRYPOINT_PATH}\u{1e}{rendered_entrypoint}").as_bytes(),
    ))
}

/// Activation the selected environment composes on one platform: the merged
/// activation variables and the activation scripts in execution order, as
/// declared (workspace-relative, unvalidated).
#[derive(Debug, Default, PartialEq, Eq)]
pub(super) struct ComposedActivation {
    pub env: BTreeMap<String, String>,
    pub scripts: Vec<String>,
}

/// Directory, inside the image, holding the projected activation scripts.
pub(super) const ACTIVATION_SCRIPT_DIR: &str = "/opt/inferlab-activation";

/// File name of the `index`-th projected activation script, shared by the
/// build context (`activation-scripts/<name>`) and the image
/// (`ACTIVATION_SCRIPT_DIR/<name>`). Names carry no workspace text, so the
/// Dockerfile and entrypoint never quote workspace-authored paths.
pub(super) fn activation_script_name(index: usize) -> String {
    format!("{:02}.sh", index + 1)
}

/// The composed sources for an environment, in Pixi's precedence order:
/// activation declared on the environment itself, then its features in
/// listed order, then the default feature (the workspace-level tables)
/// unless the environment sets `no-default-feature`. An environment absent
/// from `[environments]` (for example the implicit `default`) composes only
/// the default feature.
fn composed_sources<'a>(data: &'a toml::Value, environment: &str) -> Vec<&'a toml::Value> {
    let declared = data
        .get("environments")
        .and_then(|environments| environments.get(environment));
    let (inline, features, include_default) = match declared {
        Some(toml::Value::Array(features)) => (None, Some(features), true),
        Some(table @ toml::Value::Table(entries)) => (
            Some(table),
            entries.get("features").and_then(toml::Value::as_array),
            !entries
                .get("no-default-feature")
                .and_then(toml::Value::as_bool)
                .unwrap_or(false),
        ),
        _ => (None, None, true),
    };
    let mut sources: Vec<&toml::Value> = inline.into_iter().collect();
    for name in features
        .into_iter()
        .flatten()
        .filter_map(toml::Value::as_str)
    {
        if let Some(feature) = data.get("feature").and_then(|table| table.get(name)) {
            sources.push(feature);
        }
    }
    if include_default {
        sources.push(data);
    }
    sources
}

/// The activation tables of one source that apply on `pixi_platform`, most
/// specific first: the platform name, its operating-system family, `unix`,
/// then the untargeted table.
fn applicable_tables<'a>(source: &'a toml::Value, pixi_platform: &str) -> Vec<&'a toml::Value> {
    let family = pixi_platform.split('-').next().unwrap_or(pixi_platform);
    let mut selectors = vec![pixi_platform];
    if family != pixi_platform && matches!(family, "linux" | "osx" | "win") {
        selectors.push(family);
    }
    if matches!(family, "linux" | "osx") {
        selectors.push("unix");
    }
    let target = source.get("target");
    let mut tables: Vec<&toml::Value> = selectors
        .into_iter()
        .filter_map(|selector| {
            target
                .and_then(|target| target.get(selector))
                .and_then(|table| table.get("activation"))
        })
        .collect();
    tables.extend(source.get("activation"));
    tables
}

/// Compose the committed Pixi manifest's activation for one environment and
/// platform — the workspace's single authority for serving activation facts,
/// projected into the image instead of a framework-specific template.
///
/// Composition follows observed Pixi precedence ([[RFC-0007:C-IMAGE-BUILD]],
/// pinned by the manual differential test): within a source the script list
/// comes from the most specific applicable table declaring `scripts`, and
/// variables merge with more specific tables overriding; across sources
/// scripts concatenate in source order and the earliest source supplies a
/// variable's value.
pub(super) fn compose_activation(
    root: &Path,
    pixi_platform: &str,
    environment: &str,
) -> Result<ComposedActivation, InferlabError> {
    let manifest = root.join(PIXI_MANIFEST);
    let text = fs::read_to_string(&manifest).map_err(|source| InferlabError::Read {
        path: manifest.clone(),
        source,
    })?;
    let data: toml::Value = toml::from_str(&text).map_err(|source| InferlabError::ParseToml {
        path: manifest,
        source,
    })?;
    let mut composed = ComposedActivation::default();
    for source in composed_sources(&data, environment) {
        let tables = applicable_tables(source, pixi_platform);
        if let Some(scripts) = tables
            .iter()
            .find_map(|table| table.get("scripts").and_then(toml::Value::as_array))
        {
            composed.scripts.extend(
                scripts
                    .iter()
                    .filter_map(toml::Value::as_str)
                    .map(str::to_owned),
            );
        }
        let mut source_env = BTreeMap::new();
        for table in tables.iter().rev() {
            if let Some(entries) = table.get("env").and_then(toml::Value::as_table) {
                for (name, value) in entries {
                    if let Some(value) = value.as_str() {
                        source_env.insert(name.clone(), value.to_owned());
                    }
                }
            }
        }
        for (name, value) in source_env {
            composed.env.entry(name).or_insert(value);
        }
    }
    Ok(composed)
}

/// The rendered activation entrypoint. Projection rules, combining the v0 and
/// v1 precedents ([[ADR-0005]]) with projected activation scripts
/// ([[ADR-0070]]):
/// - `CONDA_PREFIX` is the baked prefix and `PATH` starts with its `bin`
///   before anything else runs; `$CONDA_PREFIX` references resolve to it.
/// - Projected activation scripts are sourced by bash in composition order
///   with `PIXI_PROJECT_ROOT` unset and projected variables not yet visible;
///   a script ending the shell with a non-zero status fails the entrypoint
///   and is named on stderr.
/// - Values referencing `$PIXI_PROJECT_ROOT` are workspace-tree facts with no
///   in-image equivalent; they are skipped and reported.
/// - Credential-named variables are skipped and reported; credentials never
///   enter portable artifacts.
/// - Every projected value expands against the environment the scripts
///   left. Self-referencing values (`CPATH=...:$CPATH`) compose onto it; all
///   others apply unless the variable was non-empty when the entrypoint
///   started, so `docker run --env` and image configuration win.
/// - Inferlab-managed container variables hold their starting values after
///   activation scripts run.
pub(super) struct RenderedEntrypoint {
    pub text: String,
    pub skipped: Vec<String>,
}

pub(super) fn render_entrypoint(
    activation: &BTreeMap<String, String>,
    scripts: &[String],
) -> Result<RenderedEntrypoint, InferlabError> {
    let shell = if scripts.is_empty() {
        "#!/bin/sh"
    } else {
        "#!/bin/bash"
    };
    let mut projected = Vec::new();
    let mut skipped = Vec::new();
    for (name, value) in activation {
        if crate::workspace::MANAGED_CONTAINER_ENV.contains(&name.as_str()) {
            // Inferlab-managed container variables never project from
            // activation: CONDA_PREFIX is owned by the baked prefix,
            // CUDA_VISIBLE_DEVICES is a runtime placement fact, and
            // HOME/USER/LOGNAME are injected at validation launch.
            skipped.push(name.clone());
            continue;
        }
        if credential_name(name) {
            // Credential material must never enter portable artifacts;
            // runtime credentials reach validation containers through the
            // per-machine `container.pass_env` binding (`--env NAME`), which
            // a non-empty starting value lets win.
            skipped.push(name.clone());
            continue;
        }
        if value.contains("$PIXI_PROJECT_ROOT") || value.contains("${PIXI_PROJECT_ROOT") {
            skipped.push(name.clone());
            continue;
        }
        if !shell_identifier(name) {
            return Err(InferlabError::ImageBuild {
                message: format!(
                    "activation variable {name:?} is not a shell identifier and cannot be \
                     projected into the image entrypoint"
                ),
            });
        }
        if value.contains('"')
            || value.contains('`')
            || value.contains('\\')
            || value.contains("$(")
        {
            return Err(InferlabError::ImageBuild {
                message: format!(
                    "activation value {name:?} contains shell-active characters and cannot be \
                     projected into the image entrypoint"
                ),
            });
        }
        let self_reference =
            value.contains(&format!("${name}")) || value.contains(&format!("${{{name}}}"));
        // Self-references use default-empty form so they compose under
        // `set -u` in a fresh container.
        projected.push((name, guard_self_reference(value, name), self_reference));
    }
    let mut lines = vec![shell.to_owned(), "set -eu".to_owned()];
    for (name, _, self_reference) in &projected {
        if !self_reference {
            lines.push(format!("__inferlab_s_{name}=\"${{{name}:+1}}\""));
        }
    }
    let restored: Vec<&str> = crate::workspace::MANAGED_CONTAINER_ENV
        .iter()
        .copied()
        .filter(|name| *name != "CONDA_PREFIX")
        .collect();
    if !scripts.is_empty() {
        for name in &restored {
            lines.push(format!("__inferlab_m_{name}_set=\"${{{name}+1}}\""));
            lines.push(format!("__inferlab_m_{name}=\"${{{name}-}}\""));
        }
    }
    lines.push(format!("export CONDA_PREFIX=\"{ENV_PREFIX}\""));
    lines.push(format!("export PATH=\"{ENV_PREFIX}/bin:${{PATH:-}}\""));
    if !scripts.is_empty() {
        lines.push("unset PIXI_PROJECT_ROOT".to_owned());
        lines.push("__inferlab_script=".to_owned());
        lines.push(
            "trap '__inferlab_status=$?; if [ -n \"$__inferlab_script\" ] && \
             [ \"$__inferlab_status\" -ne 0 ]; then printf \"inferlab-entrypoint: activation \
             script %s exited with status %s\\n\" \"$__inferlab_script\" \"$__inferlab_status\" \
             >&2; fi' EXIT"
                .to_owned(),
        );
        lines.push("set +eu".to_owned());
        for (index, script) in scripts.iter().enumerate() {
            lines.push(format!("__inferlab_script={}", single_quoted(script)));
            lines.push(format!(
                ". '{ACTIVATION_SCRIPT_DIR}/{}'",
                activation_script_name(index)
            ));
        }
        lines.push("__inferlab_script=".to_owned());
        lines.push("trap - EXIT".to_owned());
        lines.push("set -eu".to_owned());
        // Managed variables are restored before projected values expand, so
        // `$CONDA_PREFIX` references always resolve to the baked prefix.
        lines.push(format!("export CONDA_PREFIX=\"{ENV_PREFIX}\""));
        for name in &restored {
            lines.push(format!(
                "if [ -n \"$__inferlab_m_{name}_set\" ]; then export {name}=\"$__inferlab_m_{name}\"; fi"
            ));
        }
    }
    for (name, value, _) in &projected {
        lines.push(format!("__inferlab_p_{name}=\"{value}\""));
    }
    for (name, _, self_reference) in &projected {
        if *self_reference {
            lines.push(format!("export {name}=\"$__inferlab_p_{name}\""));
        } else {
            lines.push(format!(
                "if [ -z \"$__inferlab_s_{name}\" ]; then export {name}=\"$__inferlab_p_{name}\"; fi"
            ));
        }
    }
    lines.push("exec \"$@\"".to_owned());
    lines.push(String::new());
    Ok(RenderedEntrypoint {
        text: lines.join("\n"),
        skipped,
    })
}

fn shell_identifier(name: &str) -> bool {
    let mut chars = name.chars();
    chars
        .next()
        .is_some_and(|first| first.is_ascii_alphabetic() || first == '_')
        && chars.all(|next| next.is_ascii_alphanumeric() || next == '_')
}

fn single_quoted(text: &str) -> String {
    format!("'{}'", text.replace('\'', "'\\''"))
}

/// Variable names that identify credential material
/// ([[RFC-0007:C-IMAGE-BUILD]]). Matched on `_`-separated name segments so
/// `HF_TOKEN` is excluded while `TOKENIZERS_PARALLELISM` projects. This is
/// best-effort protection for committed workspace content, not a secret
/// scanner; runtime credentials reach validation containers through the
/// per-machine `container.pass_env` binding instead.
fn credential_name(name: &str) -> bool {
    name.split('_').any(|segment| {
        matches!(
            segment.to_ascii_uppercase().as_str(),
            "TOKEN"
                | "TOKENS"
                | "SECRET"
                | "SECRETS"
                | "PASSWORD"
                | "PASSWD"
                | "CREDENTIAL"
                | "CREDENTIALS"
                | "APIKEY"
                | "KEY"
                | "PAT"
                | "AUTH"
                | "NETRC"
        )
    })
}

fn guard_self_reference(value: &str, name: &str) -> String {
    let braced = format!("${{{name}}}");
    let guarded = format!("${{{name}:-}}");
    let mut result = value.replace(&braced, &guarded);
    let bare = format!("${name}");
    let mut output = String::with_capacity(result.len());
    let mut rest = result.as_str();
    while let Some(position) = rest.find(&bare) {
        let after = &rest[position + bare.len()..];
        let boundary = after
            .chars()
            .next()
            .is_none_or(|next| !next.is_ascii_alphanumeric() && next != '_');
        output.push_str(&rest[..position]);
        if boundary {
            output.push_str(&guarded);
        } else {
            output.push_str(&bare);
        }
        rest = after;
    }
    output.push_str(rest);
    result = output;
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::process::Command;

    #[test]
    fn entrypoint_contract_digest_tracks_rendered_text() -> Result<(), InferlabError> {
        let empty = render_entrypoint(&BTreeMap::new(), &[])?;
        let mut activation = BTreeMap::new();
        activation.insert("TORCH_CUDA_ARCH_LIST".to_owned(), "12.0".to_owned());
        let projected = render_entrypoint(&activation, &[])?;
        assert_eq!(
            entrypoint_contract_digest(&empty.text),
            entrypoint_contract_digest(&empty.text)
        );
        assert_ne!(
            entrypoint_contract_digest(&empty.text),
            entrypoint_contract_digest(&projected.text)
        );
        Ok(())
    }

    #[test]
    fn entrypoint_rejects_shell_active_values() {
        for value in ["$(id)", "`id`", "broken\"quote", "back\\slash"] {
            let mut activation = BTreeMap::new();
            activation.insert("PROBE".to_owned(), (*value).to_owned());
            assert!(render_entrypoint(&activation, &[]).is_err());
        }
        let mut activation = BTreeMap::new();
        activation.insert("NOT-AN-IDENTIFIER".to_owned(), "1".to_owned());
        assert!(render_entrypoint(&activation, &[]).is_err());
    }

    #[test]
    fn entrypoint_projection_applies_the_v0_v1_rules() -> Result<(), InferlabError> {
        let mut activation = BTreeMap::new();
        activation.insert(
            "DG_JIT_NVCC_COMPILER".to_owned(),
            "$CONDA_PREFIX/bin/nvcc".to_owned(),
        );
        activation.insert(
            "DEEPGEMM_SRC_DIR".to_owned(),
            "$PIXI_PROJECT_ROOT/DeepGEMM".to_owned(),
        );
        activation.insert(
            "CPATH".to_owned(),
            "$CONDA_PREFIX/include:$CPATH".to_owned(),
        );
        activation.insert("PATH".to_owned(), "$CONDA_PREFIX/nvvm/bin:$PATH".to_owned());
        activation.insert("CONDA_PREFIX".to_owned(), "/somewhere/else".to_owned());
        let rendered = render_entrypoint(&activation, &[])?;
        assert!(rendered.text.starts_with("#!/bin/sh"));
        assert!(
            rendered
                .text
                .contains("export CONDA_PREFIX=\"/opt/inferlab-env\"")
        );
        assert!(
            rendered
                .text
                .contains("__inferlab_p_DG_JIT_NVCC_COMPILER=\"$CONDA_PREFIX/bin/nvcc\"")
        );
        assert!(
            rendered
                .text
                .contains("__inferlab_p_CPATH=\"$CONDA_PREFIX/include:${CPATH:-}\"")
        );
        assert!(
            rendered
                .text
                .contains("__inferlab_p_PATH=\"$CONDA_PREFIX/nvvm/bin:${PATH:-}\"")
        );
        let observed = run_entrypoint(
            &rendered.text,
            &[
                ("DG_JIT_NVCC_COMPILER", "/injected/nvcc"),
                ("CPATH", "/base"),
            ],
            &["DG_JIT_NVCC_COMPILER", "CPATH"],
        )?;
        assert_eq!(observed.status, Some(0));
        assert_eq!(
            observed.values,
            ["/injected/nvcc", "/opt/inferlab-env/include:/base"]
        );
        assert!(!rendered.text.contains("DEEPGEMM_SRC_DIR"));
        assert_eq!(rendered.skipped, ["CONDA_PREFIX", "DEEPGEMM_SRC_DIR"]);
        Ok(())
    }

    #[test]
    fn entrypoint_excludes_credential_named_activation() -> Result<(), InferlabError> {
        let mut activation = BTreeMap::new();
        activation.insert("HF_TOKEN".to_owned(), "hf-credential-value".to_owned());
        activation.insert("AWS_SECRET_ACCESS_KEY".to_owned(), "aws-value".to_owned());
        activation.insert("DB_PASSWORD".to_owned(), "db-value".to_owned());
        activation.insert("TOKENIZERS_PARALLELISM".to_owned(), "false".to_owned());
        let rendered = render_entrypoint(&activation, &[])?;
        for leak in [
            "HF_TOKEN",
            "hf-credential-value",
            "AWS_SECRET_ACCESS_KEY",
            "aws-value",
            "DB_PASSWORD",
            "db-value",
        ] {
            assert!(!rendered.text.contains(leak));
        }
        assert!(
            rendered
                .text
                .contains("__inferlab_p_TOKENIZERS_PARALLELISM=\"false\"")
        );
        assert_eq!(
            rendered.skipped,
            ["AWS_SECRET_ACCESS_KEY", "DB_PASSWORD", "HF_TOKEN"]
        );
        Ok(())
    }

    fn temp_dir() -> Result<tempfile::TempDir, InferlabError> {
        tempfile::tempdir().map_err(|source| InferlabError::EnvironmentIo {
            path: PathBuf::from("tempdir"),
            operation: "create test directory",
            source,
        })
    }

    fn write(path: &Path, content: &str) -> Result<(), InferlabError> {
        fs::write(path, content).map_err(|source| InferlabError::EnvironmentIo {
            path: path.to_path_buf(),
            operation: "write test file",
            source,
        })
    }

    struct Observed {
        status: Option<i32>,
        values: Vec<String>,
        stdout: String,
        stderr: String,
    }

    /// Run a rendered entrypoint under its own interpreter with a minimal
    /// starting environment, printing the named variables (`<unset>` when
    /// absent) from the command it execs.
    fn run_entrypoint(
        text: &str,
        start: &[(&str, &str)],
        print: &[&str],
    ) -> Result<Observed, InferlabError> {
        run_entrypoint_with_scripts(text, &[], start, print)
    }

    fn run_entrypoint_with_scripts(
        text: &str,
        scripts: &[&str],
        start: &[(&str, &str)],
        print: &[&str],
    ) -> Result<Observed, InferlabError> {
        let dir = temp_dir()?;
        for (index, script) in scripts.iter().enumerate() {
            write(&dir.path().join(activation_script_name(index)), script)?;
        }
        let text = text.replace(ACTIVATION_SCRIPT_DIR, &dir.path().display().to_string());
        let entrypoint = dir.path().join("entrypoint");
        write(&entrypoint, &text)?;
        let interpreter = text
            .lines()
            .next()
            .and_then(|line| line.strip_prefix("#!"))
            .unwrap_or("/bin/sh")
            .to_owned();
        let probe = print
            .iter()
            .map(|name| format!("printf '%s\\n' \"${{{name}-<unset>}}\""))
            .collect::<Vec<_>>()
            .join("; ");
        let output = Command::new(interpreter)
            .arg(&entrypoint)
            .args(["/bin/sh", "-c", &probe])
            .env_clear()
            .env("PATH", "/usr/bin:/bin")
            .envs(start.iter().copied())
            .output()
            .map_err(|source| InferlabError::EnvironmentIo {
                path: entrypoint.clone(),
                operation: "run test entrypoint",
                source,
            })?;
        let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
        Ok(Observed {
            status: output.status.code(),
            values: stdout.lines().map(str::to_owned).collect(),
            stdout,
            stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
        })
    }

    fn compose(manifest: &str, environment: &str) -> Result<ComposedActivation, InferlabError> {
        let dir = temp_dir()?;
        write(&dir.path().join("pixi.toml"), manifest)?;
        compose_activation(dir.path(), "linux-64", environment)
    }

    const SCRIPT_PROBE_MANIFEST: &str = "\
[activation]\nscripts = [\"root.sh\"]\n\
[target.unix.activation]\nscripts = [\"root-unix.sh\"]\n\
[feature.a.activation]\nscripts = [\"a.sh\"]\nenv = { X = \"a\", U = \"a\" }\n\
[feature.a.target.linux.activation]\nscripts = [\"a-linux.sh\"]\n\
[feature.a.target.unix.activation.env]\nU = \"a-unix\"\n\
[feature.a.target.linux-aarch64.activation]\nscripts = [\"a-arm.sh\"]\n\
[feature.b.activation]\nscripts = [\"b.sh\"]\nenv = { X = \"b\", B = \"b\" }\n\
[environments]\n\
ab = [\"a\", \"b\"]\n\
inline = { features = [\"b\"], activation = { scripts = [\"env.sh\"], env = { X = \"inline\" } } }\n\
nodef = { features = [\"b\"], no-default-feature = true }\n";

    #[test]
    fn composition_orders_scripts_and_applies_target_selectors() -> Result<(), InferlabError> {
        let manifest = SCRIPT_PROBE_MANIFEST;
        let ab = compose(manifest, "ab")?;
        assert_eq!(ab.scripts, ["a-linux.sh", "b.sh", "root-unix.sh"]);
        assert_eq!(ab.env.get("X").map(String::as_str), Some("a"));
        assert_eq!(ab.env.get("U").map(String::as_str), Some("a-unix"));
        assert_eq!(ab.env.get("B").map(String::as_str), Some("b"));
        let inline = compose(manifest, "inline")?;
        assert_eq!(inline.scripts, ["env.sh", "b.sh", "root-unix.sh"]);
        assert_eq!(inline.env.get("X").map(String::as_str), Some("inline"));
        assert_eq!(compose(manifest, "nodef")?.scripts, ["b.sh"]);
        assert_eq!(compose(manifest, "default")?.scripts, ["root-unix.sh"]);
        Ok(())
    }

    #[test]
    fn entrypoint_sources_scripts_before_projected_values() -> Result<(), InferlabError> {
        let mut activation = BTreeMap::new();
        activation.insert("SEEN".to_owned(), "projected".to_owned());
        activation.insert("FROM_SCRIPT".to_owned(), "projected".to_owned());
        activation.insert("KEPT".to_owned(), "projected".to_owned());
        activation.insert("REF".to_owned(), "$CHAIN".to_owned());
        activation.insert("CHAIN".to_owned(), "projected".to_owned());
        activation.insert(
            "LD_LIBRARY_PATH".to_owned(),
            "$CONDA_PREFIX/lib:$LD_LIBRARY_PATH".to_owned(),
        );
        let scripts = ["tools/first.sh".to_owned(), "tools/second.sh".to_owned()];
        let rendered = render_entrypoint(&activation, &scripts)?;
        assert!(rendered.text.starts_with("#!/bin/bash\n"));
        let first = "\
echo \"SEEN=${SEEN-<unset>} ROOT=${PIXI_PROJECT_ROOT-<unset>} PREFIX=$CONDA_PREFIX\" >&2\n\
case \"$PATH\" in /opt/inferlab-env/bin:*) echo path-ok >&2;; esac\n\
export FROM_SCRIPT=script KEPT=script CHAIN=script\n\
export LD_LIBRARY_PATH=\"$CONDA_PREFIX/cuda-compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}\"\n\
export HOME=/script-home CONDA_PREFIX=/elsewhere USER=script-user\n\
false\n";
        let second = "echo \"second sees FROM_SCRIPT=$FROM_SCRIPT\" >&2\nreturn 3\n";
        let observed = run_entrypoint_with_scripts(
            &rendered.text,
            &[first, second],
            &[
                ("KEPT", "start"),
                ("HOME", "/start-home"),
                ("PIXI_PROJECT_ROOT", "/workspace"),
            ],
            &[
                "FROM_SCRIPT",
                "KEPT",
                "REF",
                "CHAIN",
                "LD_LIBRARY_PATH",
                "HOME",
                "CONDA_PREFIX",
                "USER",
                "PIXI_PROJECT_ROOT",
            ],
        )?;
        assert_eq!(observed.status, Some(0), "{}", observed.stderr);
        assert!(
            observed
                .stderr
                .contains("SEEN=<unset> ROOT=<unset> PREFIX=/opt/inferlab-env")
        );
        assert!(observed.stderr.contains("path-ok"));
        assert!(observed.stderr.contains("second sees FROM_SCRIPT=script"));
        assert_eq!(
            observed.values,
            [
                "projected",
                "script",
                "script",
                "projected",
                "/opt/inferlab-env/lib:/opt/inferlab-env/cuda-compat",
                "/start-home",
                "/opt/inferlab-env",
                "script-user",
                "<unset>",
            ],
            "{}",
            observed.stdout
        );
        Ok(())
    }

    #[test]
    fn entrypoint_fails_when_an_activation_script_exits_non_zero() -> Result<(), InferlabError> {
        let scripts = ["tools/ok.sh".to_owned(), "tools/it's-broken.sh".to_owned()];
        let rendered = render_entrypoint(&BTreeMap::new(), &scripts)?;
        for failing in ["exit 7\n", "set -e\nfalse\necho unreachable\n"] {
            let observed =
                run_entrypoint_with_scripts(&rendered.text, &[":\n", failing], &[], &["PATH"])?;
            assert_ne!(observed.status, Some(0));
            assert!(observed.stdout.is_empty(), "{}", observed.stdout);
            assert!(
                observed
                    .stderr
                    .contains("activation script tools/it's-broken.sh exited with status"),
                "{}",
                observed.stderr
            );
        }
        Ok(())
    }

    const MERGE_PROBE_MANIFEST: &str = "\
[activation.env]\nX = \"default\"\nD = \"default-only\"\n\
[feature.a.activation.env]\nX = \"a\"\nA = \"a-only\"\n\
[feature.a.target.linux-64.activation.env]\nX = \"a-target\"\nT = \"a-target-only\"\n\
[feature.b.activation.env]\nX = \"b\"\nB = \"b-only\"\n\
[environments]\nab = [\"a\", \"b\"]\nba = [\"b\", \"a\"]\n\
nodef = { features = [\"a\"], no-default-feature = true }\n";

    #[test]
    fn activation_env_composes_features_per_pixi_precedence() -> Result<(), InferlabError> {
        let probe = |environment: &str| {
            compose(MERGE_PROBE_MANIFEST, environment).map(|composed| composed.env)
        };
        let ab = probe("ab")?;
        assert_eq!(ab.get("X").map(String::as_str), Some("a-target"));
        assert_eq!(ab.get("A").map(String::as_str), Some("a-only"));
        assert_eq!(ab.get("B").map(String::as_str), Some("b-only"));
        assert_eq!(ab.get("D").map(String::as_str), Some("default-only"));
        assert_eq!(ab.get("T").map(String::as_str), Some("a-target-only"));
        assert_eq!(probe("ba")?.get("X").map(String::as_str), Some("b"));
        let nodef = probe("nodef")?;
        assert_eq!(nodef.get("X").map(String::as_str), Some("a-target"));
        assert_eq!(nodef.get("D"), None);
        assert_eq!(
            probe("default")?.get("X").map(String::as_str),
            Some("default")
        );
        Ok(())
    }

    /// Differential check of [[RFC-0007:C-IMAGE-BUILD]] composition against
    /// the installed Pixi: every activation variable the probe manifests
    /// declare, and the order in which their scripts run, must match what
    /// `pixi run` applies. Requires `pixi` on PATH and network access to
    /// conda-forge; run with `cargo test -p inferlab manual_activation -- --ignored`.
    #[test]
    #[ignore]
    fn manual_activation_merge_matches_real_pixi() -> Result<(), InferlabError> {
        for (probe, environments, variables) in [
            (
                MERGE_PROBE_MANIFEST,
                &["ab", "ba", "nodef", "default"][..],
                &["X", "A", "B", "D", "T"][..],
            ),
            (
                SCRIPT_PROBE_MANIFEST,
                &["ab", "inline", "nodef", "default"][..],
                &["X", "U", "B"][..],
            ),
        ] {
            let dir = temp_dir()?;
            let manifest = format!(
                "[workspace]\nname = \"probe\"\nchannels = [\"conda-forge\"]\n\
                 platforms = [\"linux-64\"]\n\n{probe}"
            );
            write(&dir.path().join("pixi.toml"), &manifest)?;
            for script in [
                "root.sh",
                "root-unix.sh",
                "a.sh",
                "a-linux.sh",
                "a-arm.sh",
                "b.sh",
                "env.sh",
            ] {
                write(
                    &dir.path().join(script),
                    &format!("export TRACE=\"${{TRACE:+$TRACE,}}{script}\"\n"),
                )?;
            }
            let run = |args: &[&str]| -> Result<String, InferlabError> {
                let output = Command::new("pixi")
                    .current_dir(dir.path())
                    .args(args)
                    .output()
                    .map_err(|source| InferlabError::LaunchPixi {
                        action: "probe",
                        source,
                    })?;
                if !output.status.success() {
                    return Err(InferlabError::ImageBuild {
                        message: format!(
                            "pixi {args:?} failed: {}",
                            String::from_utf8_lossy(&output.stderr).trim()
                        ),
                    });
                }
                Ok(String::from_utf8_lossy(&output.stdout).into_owned())
            };
            run(&["lock"])?;
            for environment in environments {
                let composed = compose_activation(dir.path(), "linux-64", environment)?;
                let observed_value = |variable: &str| -> Result<Option<String>, InferlabError> {
                    let observed = run(&[
                        "run",
                        "--no-progress",
                        "-e",
                        environment,
                        "--",
                        "sh",
                        "-c",
                        &format!("printenv {variable} || true"),
                    ])?;
                    let observed = observed.trim();
                    Ok((!observed.is_empty()).then(|| observed.to_owned()))
                };
                for variable in variables {
                    assert_eq!(
                        composed.env.get(*variable).cloned(),
                        observed_value(variable)?,
                        "{environment}: {variable}"
                    );
                }
                assert_eq!(
                    (!composed.scripts.is_empty()).then(|| composed.scripts.join(",")),
                    observed_value("TRACE")?,
                    "{environment}: script order"
                );
            }
        }
        Ok(())
    }
}
