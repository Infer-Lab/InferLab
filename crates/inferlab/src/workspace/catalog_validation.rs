//! Domain validation for the portable workspace catalog.

mod bench;
mod core;
mod eval;
mod image;
mod serve;

use super::definitions::{JsonValue, WorkspaceConfig};
use super::invalid;
use crate::InferlabError;
use std::collections::BTreeMap;
use std::path::Path;

pub(crate) use bench::validate_bench;
pub(crate) use eval::{validate_eval, validate_eval_task_source};

pub(super) fn validate_workspace(
    root: &Path,
    config: &WorkspaceConfig,
) -> Result<(), InferlabError> {
    if config.schema_version != 2 {
        return invalid(format!(
            "unsupported workspace schema version {}; expected 2",
            config.schema_version
        ));
    }
    core::validate(root, config)?;
    serve::validate(root, config)?;
    for (id, bench) in &config.benches {
        require_id("bench", id)?;
        validate_bench(id, bench)?;
    }
    for (id, eval) in &config.evals {
        require_id("eval", id)?;
        validate_eval(id, eval)?;
        validate_eval_task_source(root, id, eval)?;
    }

    for (id, suite) in &config.workload_suites {
        require_id("workload suite", id)?;
        if suite.evals.is_empty() && suite.benches.is_empty() {
            return invalid(format!(
                "workload suite {id:?} must select at least one measurement"
            ));
        }
        for eval in &suite.evals {
            require_reference("eval", eval, &config.evals)?;
        }
        for bench in &suite.benches {
            require_reference("bench", bench, &config.benches)?;
        }
        if let Some(gate) = &suite.gate {
            require_reference("eval gate", gate, &config.evals)?;
            if !suite.evals.contains(gate) {
                return invalid(format!(
                    "workload suite {id:?} gate {gate:?} is not in its eval list"
                ));
            }
        }
    }

    for (id, recipe) in &config.recipes {
        require_id("recipe", id)?;
        require_reference("server", &recipe.server, &config.servers)?;
        require_reference(
            "workload suite",
            &recipe.workload_suite,
            &config.workload_suites,
        )?;
    }

    image::validate(config)?;
    Ok(())
}

fn validate_request_body(
    kind: &str,
    id: &str,
    request_body: &BTreeMap<String, JsonValue>,
    additional_reserved: &[&str],
) -> Result<(), InferlabError> {
    const RESERVED: [&str; 8] = [
        "model",
        "prompt",
        "messages",
        "stream",
        "n",
        "max_tokens",
        "max_completion_tokens",
        "stop",
    ];
    if let Some(member) = RESERVED
        .iter()
        .chain(additional_reserved)
        .find(|member| request_body.contains_key(**member))
    {
        return invalid(format!(
            "{kind} {id:?} request_body.{member} conflicts with a measurement-runtime-owned request member"
        ));
    }
    for (member, value) in request_body {
        validate_request_body_value(kind, id, &format!("request_body.{member}"), value)?;
    }
    Ok(())
}

fn validate_request_body_value(
    kind: &str,
    id: &str,
    path: &str,
    value: &JsonValue,
) -> Result<(), InferlabError> {
    match value {
        JsonValue::Float(value) if !value.is_finite() => {
            invalid(format!("{kind} {id:?} {path} must be a finite JSON number"))
        }
        JsonValue::Array(values) => {
            for (index, value) in values.iter().enumerate() {
                validate_request_body_value(kind, id, &format!("{path}[{index}]"), value)?;
            }
            Ok(())
        }
        JsonValue::Object(values) => {
            for (member, value) in values {
                validate_request_body_value(kind, id, &format!("{path}.{member}"), value)?;
            }
            Ok(())
        }
        _ => Ok(()),
    }
}

fn require_positive(field: &str, id: &str, value: u64) -> Result<(), InferlabError> {
    if value == 0 {
        invalid(format!("definition {id:?} {field} must be positive"))
    } else {
        Ok(())
    }
}

fn require_optional_positive(
    field: &str,
    id: &str,
    value: Option<u64>,
) -> Result<(), InferlabError> {
    value.map_or(Ok(()), |value| require_positive(field, id, value))
}

fn require_reference<T>(
    label: &str,
    id: &str,
    definitions: &BTreeMap<String, T>,
) -> Result<(), InferlabError> {
    if definitions.contains_key(id) {
        Ok(())
    } else {
        invalid(format!("unknown {label} {id:?}"))
    }
}

pub(super) fn require_id(label: &str, id: &str) -> Result<(), InferlabError> {
    if !id.is_empty()
        && id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
    {
        Ok(())
    } else {
        invalid(format!("invalid {label} identifier {id:?}"))
    }
}

pub(super) fn require_nonempty(label: &str, id: &str, value: &str) -> Result<(), InferlabError> {
    if value.is_empty() {
        invalid(format!("{label} for {id:?} must not be empty"))
    } else {
        Ok(())
    }
}

// One shared locator rule for operator-supplied workspace files: the bench
// replay population file, the random source's corpus
// ([[RFC-0004:C-BENCH-REQUEST-SOURCES]]), and the synthetic-acceptance golden
// curve ([[RFC-0003:C-SERVE-SYNTHETIC-ACCEPTANCE]]).
pub(super) fn validate_workspace_relative_source_path(
    owner: &str,
    field: &str,
    path: &str,
) -> Result<(), InferlabError> {
    let candidate = Path::new(path);
    if path.is_empty()
        || candidate.is_absolute()
        || candidate.components().any(|component| {
            matches!(
                component,
                std::path::Component::Prefix(_) | std::path::Component::ParentDir
            )
        })
    {
        return invalid(format!(
            "{owner} {field} must be a workspace-relative path without parent traversal"
        ));
    }
    Ok(())
}

pub(super) fn validate_expected_digest(
    owner: &str,
    field: &str,
    digest: &str,
) -> Result<(), InferlabError> {
    if digest.len() != 64
        || !digest
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    {
        return invalid(format!(
            "{owner} {field} must be 64 lowercase hexadecimal characters"
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn validate_manifest(manifest: &str) -> Result<(), Box<dyn std::error::Error>> {
        let root = tempfile::tempdir()?;
        let config = toml::from_str::<WorkspaceConfig>(manifest)?;
        validate_workspace(root.path(), &config)?;
        Ok(())
    }

    #[test]
    fn prefill_decode_requires_both_frontend_components_on_the_server_base()
    -> Result<(), Box<dyn std::error::Error>> {
        let result = validate_manifest(
            r#"
schema_version = 2

[models.model]
served_name = "model"

[stacks.stack]
integration = "fixture"
pixi_environment = "fixture"

[servers.server]
stack = "stack"
model = "model"
topology = "prefill_decode"
readiness_timeout_seconds = 60

[servers.server.cases.add-frontend]
gateway_backend = "gateway"
pd_router_backend = "router"
"#,
        );
        let Err(error) = result else {
            return Err(std::io::Error::other(
                "a P/D case must not add frontend components absent from the server base",
            )
            .into());
        };

        assert!(
            error
                .to_string()
                .contains("prefill_decode server \"server\" must declare both gateway_backend and pd_router_backend"),
            "{error}"
        );
        Ok(())
    }

    #[test]
    fn single_case_cannot_add_a_gateway_absent_from_the_server_base()
    -> Result<(), Box<dyn std::error::Error>> {
        let result = validate_manifest(
            r#"
schema_version = 2

[models.model]
served_name = "model"

[stacks.stack]
integration = "fixture"
pixi_environment = "fixture"

[servers.server]
stack = "stack"
model = "model"
topology = "single"
readiness_timeout_seconds = 60

[servers.server.cases.add-gateway]
gateway_backend = "gateway"
"#,
        );
        let Err(error) = result else {
            return Err(std::io::Error::other(
                "a single case must not add a Gateway absent from the server base",
            )
            .into());
        };

        assert!(
            error.to_string().contains(
                "cannot add gateway_backend because the server base does not declare a Gateway"
            ),
            "{error}"
        );
        Ok(())
    }

    #[test]
    fn extra_args_segmentation_validates_at_workspace_load_across_layers()
    -> Result<(), Box<dyn std::error::Error>> {
        const HEADER: &str = r#"
schema_version = 2

[models.model]
served_name = "model"

[stacks.stack]
integration = "fixture"
pixi_environment = "fixture"

[servers.server]
stack = "stack"
model = "model"
topology = "single"
readiness_timeout_seconds = 60
"#;
        // A value token preceding any flag fails load validation at every
        // settings layer, even when no overriding layer touches the array
        // ([[RFC-0003:C-RESOLUTION]]).
        for (layer, path) in [
            (
                "[servers.server.settings]\nextra_args = [\"stray\", \"--a\"]",
                "server \"server\" settings.extra_args",
            ),
            (
                "[servers.server.roles.serve.settings]\nextra_args = [\"stray\"]",
                "server \"server\" role \"serve\" settings.extra_args",
            ),
            (
                "[servers.server.cases.c.settings]\nextra_args = [\"stray\"]",
                "server case \"c\" settings.extra_args",
            ),
            (
                "[servers.server.cases.c.roles.serve.settings]\nextra_args = [\"stray\"]",
                "server case \"c\" role \"serve\" settings.extra_args",
            ),
        ] {
            let Err(error) = validate_manifest(&format!("{HEADER}\n{layer}\n")) else {
                return Err(
                    std::io::Error::other(format!("{layer} must fail load validation")).into(),
                );
            };
            assert!(error.to_string().contains("precedes any flag"), "{error}");
            assert!(error.to_string().contains(path), "{error}");
        }

        // A second bare `--` is malformed even without any patch layer.
        let result = validate_manifest(&format!(
            "{HEADER}\n[servers.server.settings]\nextra_args = [\"--\", \"--a\", \"--\", \"--b\"]\n"
        ));
        let Err(error) = result else {
            return Err(std::io::Error::other(
                "a second verbatim sentinel must fail load validation",
            )
            .into());
        };
        assert!(error.to_string().contains("second bare `--`"), "{error}");

        // A well-segmented declaration, including one verbatim block, loads.
        validate_manifest(&format!(
            "{HEADER}\n[servers.server.settings]\nextra_args = [\"--max-num-seqs\", \"256\", \"--\", \"--block-size\", \"32\"]\n"
        ))?;
        Ok(())
    }

    // [[RFC-0003:C-SERVE-SYNTHETIC-ACCEPTANCE]] declaration validation at
    // workspace load: exactly one form, a finite acceptance length of at
    // least one, and well-formed curve coordinates.
    #[test]
    fn synthetic_acceptance_declaration_validation() -> Result<(), Box<dyn std::error::Error>> {
        const HEADER: &str = r#"
schema_version = 2

[models.model]
served_name = "model"

[stacks.stack]
integration = "fixture"
pixi_environment = "fixture"

[servers.server]
stack = "stack"
model = "model"
topology = "single"
readiness_timeout_seconds = 60
"#;
        const CURVE: &str = "curve = { path = \"curves/golden.yaml\", expected_sha256 = \"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\", model_key = \"model\" }";

        // Both forms load at either layer.
        validate_manifest(&format!(
            "{HEADER}\nsynthetic_acceptance = {{ acceptance_length = 3.5 }}\n"
        ))?;
        validate_manifest(&format!("{HEADER}\nsynthetic_acceptance = {{ {CURVE} }}\n"))?;
        validate_manifest(&format!(
            "{HEADER}\n[servers.server.cases.c]\nsynthetic_acceptance = {{ acceptance_length = 1.0 }}\n"
        ))?;
        validate_manifest(&format!(
            "{HEADER}\n[servers.server.cases.c]\nsynthetic_acceptance = {{ {CURVE} }}\n"
        ))?;

        for (declaration, expected) in [
            (
                "synthetic_acceptance = { acceptance_length = 3.5, curve = { path = \"curves/golden.yaml\", expected_sha256 = \"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\", model_key = \"model\" } }",
                "exactly one of acceptance_length or curve",
            ),
            (
                "synthetic_acceptance = { }",
                "exactly one of acceptance_length or curve",
            ),
            // [[RFC-0003:C-SERVE-SYNTHETIC-ACCEPTANCE]] InferLab does not
            // model the speculative method or draft model; both forms reject
            // unknown members at parse.
            (
                "synthetic_acceptance = { acceptance_length = 3.5, method = \"mtp\" }",
                "unknown field `method`",
            ),
            (
                "synthetic_acceptance = { curve = { path = \"curves/golden.yaml\", expected_sha256 = \"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\", model_key = \"model\", draft_model = \"draft\" } }",
                "unknown field `draft_model`",
            ),
            (
                "synthetic_acceptance = { acceptance_length = 0.5 }",
                "finite number of at least one",
            ),
            (
                "synthetic_acceptance = { acceptance_length = 0.0 }",
                "finite number of at least one",
            ),
            (
                "synthetic_acceptance = { acceptance_length = inf }",
                "finite number of at least one",
            ),
            (
                "synthetic_acceptance = { acceptance_length = nan }",
                "finite number of at least one",
            ),
            (
                "synthetic_acceptance = { curve = { path = \"/abs/golden.yaml\", expected_sha256 = \"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\", model_key = \"model\" } }",
                "workspace-relative path without parent traversal",
            ),
            (
                "synthetic_acceptance = { curve = { path = \"../golden.yaml\", expected_sha256 = \"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\", model_key = \"model\" } }",
                "workspace-relative path without parent traversal",
            ),
            (
                "synthetic_acceptance = { curve = { path = \"curves/golden.yaml\", expected_sha256 = \"AAAA\", model_key = \"model\" } }",
                "64 lowercase hexadecimal characters",
            ),
            (
                "synthetic_acceptance = { curve = { path = \"curves/golden.yaml\", expected_sha256 = \"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\", model_key = \"\" } }",
                "model_key must not be empty",
            ),
            (
                "synthetic_acceptance = { curve = { path = \"curves/golden.yaml\", expected_sha256 = \"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\", model_key = \"model\", thinking_mode = \"\" } }",
                "thinking_mode must be non-empty when present",
            ),
        ] {
            let result = validate_manifest(&format!("{HEADER}\n{declaration}\n"));
            let Err(error) = result else {
                return Err(std::io::Error::other(format!(
                    "{declaration} must fail load validation"
                ))
                .into());
            };
            assert!(
                error.to_string().contains(expected),
                "{declaration}: {error}"
            );
        }
        Ok(())
    }

    // [[RFC-0003:C-SERVE-SYNTHETIC-ACCEPTANCE]]: when the curve file is
    // readable and digest-matched, the two-shape contract, entry values, and
    // the thinking-mode-vs-flat rule fail workspace validation at either
    // declaration layer. Missing files, digest mismatches, and unknown model
    // keys defer to resolution.
    #[test]
    fn synthetic_acceptance_curve_shape_validates_at_load() -> Result<(), Box<dyn std::error::Error>>
    {
        use sha2::{Digest, Sha256};

        const HEADER: &str = r#"
schema_version = 2

[models.model]
served_name = "model"

[stacks.stack]
integration = "fixture"
pixi_environment = "fixture"

[servers.server]
stack = "stack"
model = "model"
topology = "single"
readiness_timeout_seconds = 60
"#;
        const SERVER_LAYER: &str = "{declaration}";
        const CASE_LAYER: &str = "[servers.server.cases.c]\n{declaration}";

        #[allow(clippy::too_many_arguments)]
        fn load_manifest(
            root: &std::path::Path,
            layer: &str,
            curve_text: Option<&str>,
            digest_override: Option<&str>,
            model_key: &str,
            thinking_mode: Option<&str>,
        ) -> Result<(), String> {
            let digest = match digest_override {
                Some(digest) => digest.to_owned(),
                None => match curve_text {
                    Some(text) => format!("{:x}", Sha256::digest(text.as_bytes())),
                    None => "a".repeat(64),
                },
            };
            if let Some(text) = curve_text {
                let curves = root.join("curves");
                std::fs::create_dir_all(&curves).map_err(|error| error.to_string())?;
                std::fs::write(curves.join("golden.yaml"), text)
                    .map_err(|error| error.to_string())?;
            }
            let mode = thinking_mode
                .map(|mode| format!(", thinking_mode = \"{mode}\""))
                .unwrap_or_default();
            let declaration = format!(
                "synthetic_acceptance = {{ curve = {{ path = \"curves/golden.yaml\", expected_sha256 = \"{digest}\", model_key = \"{model_key}\"{mode} }} }}"
            );
            let manifest = format!(
                "{HEADER}\n{}\n",
                layer.replace("{declaration}", &declaration)
            );
            let config =
                toml::from_str::<WorkspaceConfig>(&manifest).map_err(|error| error.to_string())?;
            validate_workspace(root, &config).map_err(|error| error.to_string())
        }

        // Malformed shapes and non-finite values fail at both declaration
        // layers, including a case that is never selected.
        for (layer, context) in [
            (SERVER_LAYER, "server \"server\" synthetic_acceptance.curve"),
            (CASE_LAYER, "server case \"c\" synthetic_acceptance.curve"),
        ] {
            for (text, expected) in [
                ("- 1\n- 2\n", "must map model keys"),
                (
                    "model: 3.5\n",
                    "flat list of draft-length entries or a thinking-mode mapping",
                ),
                ("model:\n  - 4\n", "single-entry mapping"),
                ("model:\n  - 0: 3.5\n", "must be a positive integer"),
                ("model:\n  - 4: later\n", "must be a finite number"),
                ("model:\n  - 4: .inf\n", "must be a finite number"),
                ("model:\n  thinking_on: 3.5\n", "must map draft lengths"),
                ("model:\n  1:\n    4: 3.5\n", "must be a string"),
                // The whole document is shape-validated, not only the
                // declared model key's entry.
                (
                    "other: 3.5\nmodel:\n  - 4: 3.5\n",
                    "flat list of draft-length entries or a thinking-mode mapping",
                ),
            ] {
                let root = tempfile::tempdir()?;
                let error = load_manifest(root.path(), layer, Some(text), None, "model", None)
                    .err()
                    .ok_or(format!("{text:?} must fail load validation"))?;
                assert!(error.contains(expected), "{layer} {text:?}: {error}");
                assert!(error.contains(context), "{layer} {text:?}: {error}");
            }
        }

        // A declared thinking mode against a flat entry fails at both layers.
        for layer in [SERVER_LAYER, CASE_LAYER] {
            let root = tempfile::tempdir()?;
            let error = load_manifest(
                root.path(),
                layer,
                Some("model:\n  - 4: 3.5\n"),
                None,
                "model",
                Some("thinking_on"),
            )
            .err()
            .ok_or("thinking_mode against a flat entry must fail load validation")?;
            assert!(error.contains("no mode applies"), "{layer}: {error}");
        }

        // A valid flat curve and a valid matrix curve load cleanly.
        for (text, mode) in [
            ("model:\n  - 4: 3.5\n", None),
            ("model:\n  thinking_on:\n    4: 3.5\n", None),
            ("model:\n  thinking_on:\n    4: 3.5\n", Some("thinking_on")),
        ] {
            let root = tempfile::tempdir()?;
            load_manifest(root.path(), SERVER_LAYER, Some(text), None, "model", mode)
                .map_err(|error| format!("{text:?} must load: {error}"))?;
        }

        // A missing file, a digest mismatch, and an unknown model key defer
        // to resolution: load validation passes.
        let root = tempfile::tempdir()?;
        load_manifest(root.path(), SERVER_LAYER, None, None, "model", None)
            .map_err(|error| format!("a missing curve file defers to resolution: {error}"))?;
        load_manifest(
            root.path(),
            SERVER_LAYER,
            Some("model:\n  - 4: 3.5\n"),
            Some(&"b".repeat(64)),
            "model",
            None,
        )
        .map_err(|error| format!("a digest mismatch defers to resolution: {error}"))?;
        load_manifest(
            root.path(),
            SERVER_LAYER,
            Some("model:\n  - 4: 3.5\n"),
            None,
            "other",
            None,
        )
        .map_err(|error| format!("an unknown model key defers to resolution: {error}"))?;
        Ok(())
    }
}
