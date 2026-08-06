//! Pixi activation projection and the generated runtime entrypoint contract.

use crate::InferlabError;
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
    format!(
        "{:x}",
        Sha256::digest(format!("{ENTRYPOINT_PATH}\u{1e}{rendered_entrypoint}").as_bytes())
    )
}

/// The selected environment's feature composition: the named features in
/// declaration order, and whether the default feature (the workspace-level
/// tables) applies. An environment absent from `[environments]` (for example
/// the implicit `default`) composes no named features.
fn composed_features(data: &toml::Value, environment: &str) -> (Vec<String>, bool) {
    match data
        .get("environments")
        .and_then(|environments| environments.get(environment))
    {
        Some(toml::Value::Array(features)) => (
            features
                .iter()
                .filter_map(toml::Value::as_str)
                .map(str::to_owned)
                .collect(),
            true,
        ),
        Some(toml::Value::Table(table)) => (
            table
                .get("features")
                .and_then(toml::Value::as_array)
                .map(|features| {
                    features
                        .iter()
                        .filter_map(toml::Value::as_str)
                        .map(str::to_owned)
                        .collect()
                })
                .unwrap_or_default(),
            !table
                .get("no-default-feature")
                .and_then(toml::Value::as_bool)
                .unwrap_or(false),
        ),
        _ => (Vec::new(), true),
    }
}

/// Whether a feature-shaped table (the workspace root or one `[feature.*]`
/// entry) declares activation scripts, directly or under a target.
fn declares_activation_scripts(node: &toml::Value, pixi_platform: &str) -> bool {
    let direct = node
        .get("activation")
        .and_then(|activation| activation.get("scripts"))
        .is_some();
    let targeted = node
        .get("target")
        .and_then(|target| target.get(pixi_platform))
        .and_then(|target| target.get("activation"))
        .and_then(|activation| activation.get("scripts"))
        .is_some();
    direct || targeted
}

/// Reject activation scripts that bind to the selected environment
/// ([[RFC-0007:C-IMAGE-BUILD]]): a script's exports are not statically
/// knowable, so the cache key, sanitized-view redirects, and entrypoint
/// projection would all silently miss them. Feature-scoped `activation.env`
/// is composed by the projection and is legal; scripts under features the
/// environment does not compose are inert and stay legal too.
pub(super) fn guard_unmodeled_activation(
    root: &Path,
    pixi_platform: &str,
    environment: &str,
) -> Result<(), InferlabError> {
    let manifest = root.join("pixi.toml");
    let text = fs::read_to_string(&manifest).map_err(|source| InferlabError::Read {
        path: manifest.clone(),
        source,
    })?;
    let data: toml::Value = toml::from_str(&text).map_err(|source| InferlabError::ParseToml {
        path: manifest,
        source,
    })?;
    let (features, include_default) = composed_features(&data, environment);
    if include_default && declares_activation_scripts(&data, pixi_platform) {
        return Err(InferlabError::ImageBuild {
            message: "pixi.toml declares [activation] scripts, which image production cannot \
                      project; move the exports into [activation.env]"
                .to_owned(),
        });
    }
    for name in &features {
        let declares = data
            .get("feature")
            .and_then(|table| table.get(name))
            .is_some_and(|feature| declares_activation_scripts(feature, pixi_platform));
        if declares {
            return Err(InferlabError::ImageBuild {
                message: format!(
                    "environment {environment:?} composes feature {name:?}, which declares \
                     activation scripts that image production cannot project; move the \
                     exports into the feature's [activation.env]"
                ),
            });
        }
    }
    Ok(())
}

/// Read the committed Pixi manifest's activation env for one environment and
/// platform — the workspace's single authority for serving activation facts,
/// projected into the image instead of a framework-specific template.
///
/// Feature composition follows Pixi's observed precedence (verified against
/// pixi 0.71.2 and pinned by the manual differential test): earlier-listed
/// features win over later ones, a feature's target table overrides its
/// untargeted table, and the default feature (the workspace-level tables)
/// applies last unless the environment sets `no-default-feature`. The merge
/// inserts in reverse precedence so later inserts overwrite.
pub(super) fn activation_env(
    root: &Path,
    pixi_platform: &str,
    environment: &str,
) -> Result<BTreeMap<String, String>, InferlabError> {
    let manifest = root.join("pixi.toml");
    let text = fs::read_to_string(&manifest).map_err(|source| InferlabError::Read {
        path: manifest.clone(),
        source,
    })?;
    let data: toml::Value = toml::from_str(&text).map_err(|source| InferlabError::ParseToml {
        path: manifest,
        source,
    })?;
    let (features, include_default) = composed_features(&data, environment);
    let mut env = BTreeMap::new();
    let mut merge = |node: &toml::Value| {
        let tables = [
            node.get("activation"),
            node.get("target")
                .and_then(|target| target.get(pixi_platform))
                .and_then(|target| target.get("activation")),
        ];
        for activation in tables.into_iter().flatten() {
            if let Some(entries) = activation.get("env").and_then(toml::Value::as_table) {
                for (name, value) in entries {
                    if let Some(value) = value.as_str() {
                        env.insert(name.clone(), value.to_owned());
                    }
                }
            }
        }
    };
    if include_default {
        merge(&data);
    }
    for name in features.iter().rev() {
        if let Some(feature) = data.get("feature").and_then(|table| table.get(name)) {
            merge(feature);
        }
    }
    Ok(env)
}

/// The rendered activation entrypoint. Projection rules, combining the v0 and
/// v1 precedents ([[ADR-0005]]):
/// - `$CONDA_PREFIX` references stay unexpanded and resolve to the baked
///   prefix exported first.
/// - Values referencing `$PIXI_PROJECT_ROOT` are workspace-tree facts with no
///   in-image equivalent; they are skipped and reported (DeepGEMM sources are
///   compiled into the vLLM wheel at build time, not consumed at runtime).
/// - Credential-named variables are skipped and reported; credentials never
///   enter portable artifacts.
/// - Self-referencing values (`CPATH=...:$CPATH`) export plainly so injected
///   values compose; all others use `${VAR:-...}` so `docker run --env` wins.
pub(super) struct RenderedEntrypoint {
    pub text: String,
    pub skipped: Vec<String>,
}

pub(super) fn render_entrypoint(
    activation: &BTreeMap<String, String>,
) -> Result<RenderedEntrypoint, InferlabError> {
    let mut lines = vec![
        "#!/bin/sh".to_owned(),
        "set -eu".to_owned(),
        format!("export CONDA_PREFIX=\"{ENV_PREFIX}\""),
        format!("export PATH=\"{ENV_PREFIX}/bin:${{PATH:-}}\""),
    ];
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
            // the `${VAR:-...}` export form honors.
            skipped.push(name.clone());
            continue;
        }
        if value.contains("$PIXI_PROJECT_ROOT") || value.contains("${PIXI_PROJECT_ROOT") {
            skipped.push(name.clone());
            continue;
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
        if self_reference {
            // Rewrite the self-reference to default-empty form so the export
            // composes under `set -u` in a fresh container.
            let guarded = guard_self_reference(value, name);
            lines.push(format!("export {name}=\"{guarded}\""));
        } else {
            lines.push(format!("export {name}=\"${{{name}:-{value}}}\""));
        }
    }
    lines.push("exec \"$@\"".to_owned());
    lines.push(String::new());
    Ok(RenderedEntrypoint {
        text: lines.join("\n"),
        skipped,
    })
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
        let empty = render_entrypoint(&BTreeMap::new())?;
        let mut activation = BTreeMap::new();
        activation.insert("TORCH_CUDA_ARCH_LIST".to_owned(), "12.0".to_owned());
        let projected = render_entrypoint(&activation)?;
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
            assert!(render_entrypoint(&activation).is_err());
        }
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
        let rendered = render_entrypoint(&activation)?;
        assert!(rendered.text.starts_with("#!/bin/sh"));
        assert!(
            rendered
                .text
                .contains("export CONDA_PREFIX=\"/opt/inferlab-env\"")
        );
        assert!(rendered.text.contains(
            "export DG_JIT_NVCC_COMPILER=\"${DG_JIT_NVCC_COMPILER:-$CONDA_PREFIX/bin/nvcc}\""
        ));
        assert!(
            rendered
                .text
                .contains("export CPATH=\"$CONDA_PREFIX/include:${CPATH:-}\"")
        );
        assert!(
            rendered
                .text
                .contains("export PATH=\"$CONDA_PREFIX/nvvm/bin:${PATH:-}\"")
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
        let rendered = render_entrypoint(&activation)?;
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
                .contains("export TOKENIZERS_PARALLELISM=\"${TOKENIZERS_PARALLELISM:-false}\"")
        );
        assert_eq!(
            rendered.skipped,
            ["AWS_SECRET_ACCESS_KEY", "DB_PASSWORD", "HF_TOKEN"]
        );
        Ok(())
    }

    #[test]
    fn unmodeled_activation_branches_are_rejected() -> Result<(), InferlabError> {
        let write_manifest = |content: &str| -> Result<tempfile::TempDir, InferlabError> {
            let dir = tempfile::tempdir().map_err(|source| InferlabError::EnvironmentIo {
                path: PathBuf::from("tempdir"),
                operation: "create test workspace",
                source,
            })?;
            fs::write(dir.path().join("pixi.toml"), content).map_err(|source| {
                InferlabError::EnvironmentIo {
                    path: dir.path().join("pixi.toml"),
                    operation: "write test manifest",
                    source,
                }
            })?;
            Ok(dir)
        };
        for (manifest, environment) in [
            (
                "[activation.env]\nX = \"1\"\n[target.linux-aarch64.activation.env]\nX = \"2\"\n",
                "default",
            ),
            (
                "[feature.cuda.activation.env]\nX = \"1\"\n[environments]\nvllm = [\"cuda\"]\n",
                "vllm",
            ),
            (
                "[feature.dev.activation]\nscripts = [\"env.sh\"]\n[environments]\nvllm = [\"cuda\"]\n",
                "vllm",
            ),
            (
                "[activation]\nscripts = [\"env.sh\"]\n[environments]\nvllm = { features = [\"cuda\"], no-default-feature = true }\n",
                "vllm",
            ),
        ] {
            let dir = write_manifest(manifest)?;
            assert!(guard_unmodeled_activation(dir.path(), "linux-64", environment).is_ok());
        }
        for (manifest, environment) in [
            ("[activation]\nscripts = [\"env.sh\"]\n", "default"),
            (
                "[feature.cuda.activation]\nscripts = [\"env.sh\"]\n[environments]\nvllm = { features = [\"cuda\"] }\n",
                "vllm",
            ),
            (
                "[feature.cuda.target.linux-64.activation]\nscripts = [\"env.sh\"]\n[environments]\nvllm = [\"cuda\"]\n",
                "vllm",
            ),
        ] {
            let dir = write_manifest(manifest)?;
            assert!(guard_unmodeled_activation(dir.path(), "linux-64", environment).is_err());
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
        let dir = tempfile::tempdir().map_err(|source| InferlabError::EnvironmentIo {
            path: PathBuf::from("tempdir"),
            operation: "create test workspace",
            source,
        })?;
        fs::write(dir.path().join("pixi.toml"), MERGE_PROBE_MANIFEST).map_err(|source| {
            InferlabError::EnvironmentIo {
                path: dir.path().join("pixi.toml"),
                operation: "write test manifest",
                source,
            }
        })?;
        let probe = |environment: &str| activation_env(dir.path(), "linux-64", environment);
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

    #[test]
    #[ignore]
    fn manual_activation_merge_matches_real_pixi() -> Result<(), InferlabError> {
        let dir = tempfile::tempdir().map_err(|source| InferlabError::EnvironmentIo {
            path: PathBuf::from("tempdir"),
            operation: "create probe workspace",
            source,
        })?;
        let manifest = format!(
            "[workspace]\nname = \"probe\"\nchannels = [\"conda-forge\"]\n\
             platforms = [\"linux-64\"]\n\n{MERGE_PROBE_MANIFEST}"
        );
        fs::write(dir.path().join("pixi.toml"), manifest).map_err(|source| {
            InferlabError::EnvironmentIo {
                path: dir.path().join("pixi.toml"),
                operation: "write probe manifest",
                source,
            }
        })?;
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
        for environment in ["ab", "ba", "nodef"] {
            let projected = activation_env(dir.path(), "linux-64", environment)?;
            for variable in ["X", "A", "B", "D", "T"] {
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
                assert_eq!(
                    projected.get(variable).map(String::as_str),
                    (!observed.is_empty()).then_some(observed)
                );
            }
        }
        Ok(())
    }
}
