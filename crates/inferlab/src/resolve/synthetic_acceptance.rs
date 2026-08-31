//! Digest-verified golden-curve resolution for the synthetic acceptance
//! overlay ([[RFC-0003:C-SERVE-SYNTHETIC-ACCEPTANCE]]). The control plane owns
//! declaration composition, curve integrity (digest verification and shape
//! validation, shared with workspace-load validation through
//! [`crate::workspace::synthetic_acceptance`]), and the effective thinking
//! mode; the keyed acceptance-length lookup belongs to the selected
//! integration, where the operator's speculative configuration supplies the
//! draft count ([[ADR-0043]]).

use crate::InferlabError;
use crate::workspace::synthetic_acceptance::{
    CurveModelEntry, invalid_curve, validate_curve_document,
};
use crate::workspace::{SyntheticAcceptanceCurveDefinition, SyntheticAcceptanceDefinition};
use inferlab_protocol::{SyntheticAcceptanceCurveInput, SyntheticAcceptanceInput};
use sha2::{Digest, Sha256};
use std::path::Path;

/// The thinking mode a matrix-shape curve entry falls back to when the
/// declaration omits `thinking_mode`
/// ([[RFC-0003:C-SERVE-SYNTHETIC-ACCEPTANCE]]).
const DEFAULT_THINKING_MODE: &str = "thinking_on";

/// The resolved synthetic acceptance request material: the effective
/// declaration after case composition plus the wire input carrying, for the
/// curve form, the digest-verified curve text and the effective thinking
/// mode. The acceptance length itself is resolved and returned by the
/// integration ([[ADR-0043]]).
pub(super) struct ResolvedSyntheticAcceptance {
    pub declared: SyntheticAcceptanceDefinition,
    pub input: SyntheticAcceptanceInput,
}

impl ResolvedSyntheticAcceptance {
    /// The effective thinking mode of a matrix-shape curve entry; `None` for
    /// the explicit form and for flat entries, where no mode applies.
    pub(super) fn effective_thinking_mode(&self) -> Option<&str> {
        match &self.input {
            SyntheticAcceptanceInput::Explicit { .. } => None,
            SyntheticAcceptanceInput::Curve(curve) => curve.thinking_mode.as_deref(),
        }
    }
}

/// Resolve the synthetic acceptance request material for the composed server
/// definition. Declaration shape was validated at workspace load; resolution
/// digest-verifies the curve file before it is used and shape-validates it
/// again (resolution must not depend on load-time findings), failing with a
/// typed error that names the missing or mismatched element. The keyed curve
/// lookup is the integration's planning-stage obligation, so an unknown
/// model key or a missing thinking-mode entry does not fail here.
pub(super) fn resolve_synthetic_acceptance(
    workspace_root: &Path,
    server_id: &str,
    declared: &SyntheticAcceptanceDefinition,
) -> Result<ResolvedSyntheticAcceptance, InferlabError> {
    match (&declared.acceptance_length, &declared.curve) {
        (Some(length), None) => {
            if !length.is_finite() || *length < 1.0 {
                return Err(InferlabError::InvalidConfig {
                    message: format!(
                        "server {server_id:?} synthetic_acceptance.acceptance_length must be a finite number of at least one, resolved {length}"
                    ),
                });
            }
            Ok(ResolvedSyntheticAcceptance {
                declared: declared.clone(),
                input: SyntheticAcceptanceInput::Explicit {
                    acceptance_length: *length,
                },
            })
        }
        (None, Some(curve)) => resolve_curve(workspace_root, server_id, declared, curve),
        // Workspace load validation owns the exactly-one-form rule; reaching
        // this arm means a layer composition defect, not operator input.
        (Some(_), Some(_)) | (None, None) => Err(InferlabError::InvalidConfig {
            message: format!(
                "server {server_id:?} synthetic_acceptance must declare exactly one of acceptance_length or curve"
            ),
        }),
    }
}

fn resolve_curve(
    workspace_root: &Path,
    server_id: &str,
    declared: &SyntheticAcceptanceDefinition,
    curve: &SyntheticAcceptanceCurveDefinition,
) -> Result<ResolvedSyntheticAcceptance, InferlabError> {
    let context = format!("server {server_id:?} synthetic_acceptance.curve");
    let resolved_path = workspace_root.join(&curve.path);
    let bytes = std::fs::read(&resolved_path).map_err(|source| {
        invalid_curve(
            &context,
            curve,
            format!("the curve file is missing or unreadable: {source}"),
        )
    })?;
    let observed = format!("{:x}", Sha256::digest(&bytes));
    if observed != curve.expected_sha256 {
        return Err(InferlabError::InvalidConfig {
            message: format!(
                "{context} digest mismatch for {:?}: expected {}, observed {observed}",
                curve.path, curve.expected_sha256
            ),
        });
    }
    let entries = validate_curve_document(&resolved_path, &bytes, &context, curve)?;
    let text = String::from_utf8(bytes).map_err(|_| {
        invalid_curve(
            &context,
            curve,
            "the curve file is not valid UTF-8".to_owned(),
        )
    })?;
    let thinking_mode = match entries.get(&curve.model_key) {
        Some(CurveModelEntry::Flat) => None,
        Some(CurveModelEntry::Matrix) => Some(
            curve
                .thinking_mode
                .clone()
                .unwrap_or_else(|| DEFAULT_THINKING_MODE.to_owned()),
        ),
        // The keyed lookup belongs to the integration: an unknown model key
        // fails its planning, so the declared mode crosses as-is.
        None => curve.thinking_mode.clone(),
    };
    Ok(ResolvedSyntheticAcceptance {
        declared: declared.clone(),
        input: SyntheticAcceptanceInput::Curve(SyntheticAcceptanceCurveInput {
            model_key: curve.model_key.clone(),
            thinking_mode,
            text,
            sha256: curve.expected_sha256.clone(),
        }),
    })
}

#[cfg(test)]
mod tests {
    use super::{DEFAULT_THINKING_MODE, resolve_synthetic_acceptance};
    use crate::workspace::{SyntheticAcceptanceCurveDefinition, SyntheticAcceptanceDefinition};
    use inferlab_protocol::SyntheticAcceptanceInput;
    use sha2::{Digest, Sha256};
    use std::path::{Path, PathBuf};

    fn curve_declaration(path: &str, bytes: &[u8]) -> SyntheticAcceptanceDefinition {
        SyntheticAcceptanceDefinition {
            acceptance_length: None,
            curve: Some(SyntheticAcceptanceCurveDefinition {
                path: PathBuf::from(path),
                expected_sha256: format!("{:x}", Sha256::digest(bytes)),
                model_key: "model".to_owned(),
                thinking_mode: None,
            }),
        }
    }

    fn write_curve(
        root: &Path,
        path: &str,
        text: &str,
    ) -> Result<SyntheticAcceptanceDefinition, Box<dyn std::error::Error>> {
        std::fs::create_dir_all(root.join("curves"))?;
        std::fs::write(root.join(path), text)?;
        Ok(curve_declaration(path, text.as_bytes()))
    }

    fn resolve_input(
        root: &Path,
        declared: &SyntheticAcceptanceDefinition,
    ) -> Result<SyntheticAcceptanceInput, String> {
        resolve_synthetic_acceptance(root, "server", declared)
            .map(|resolved| resolved.input)
            .map_err(|error| error.to_string())
    }

    fn curve_payload(
        input: &SyntheticAcceptanceInput,
    ) -> Result<(String, Option<String>, String, String), String> {
        let SyntheticAcceptanceInput::Curve(curve) = input else {
            return Err(format!("expected the curve form, got {input:?}"));
        };
        Ok((
            curve.model_key.clone(),
            curve.thinking_mode.clone(),
            curve.text.clone(),
            curve.sha256.clone(),
        ))
    }

    #[test]
    fn flat_curve_ships_the_digest_verified_text_without_a_mode()
    -> Result<(), Box<dyn std::error::Error>> {
        let text = "model:\n  - 1: 2.1\n  - 2: 2.8\n  - 4: 3.5\n";
        let root = tempfile::tempdir()?;
        let declared = write_curve(root.path(), "curves/golden.yaml", text)?;

        let (model_key, thinking_mode, shipped_text, sha256) =
            curve_payload(&resolve_input(root.path(), &declared)?)?;
        assert_eq!(model_key, "model");
        assert_eq!(thinking_mode, None);
        assert_eq!(shipped_text, text);
        assert_eq!(
            sha256,
            format!("{:x}", Sha256::digest(text.as_bytes())),
            "the shipped digest is the digest-verified declaration pin"
        );
        Ok(())
    }

    #[test]
    fn matrix_curve_omission_computes_the_thinking_on_default()
    -> Result<(), Box<dyn std::error::Error>> {
        let root = tempfile::tempdir()?;
        let mut declared = write_curve(
            root.path(),
            "curves/golden.yaml",
            "model:\n  thinking_on:\n    4: 3.5\n  thinking_off:\n    4: 1.5\n",
        )?;

        let (_, thinking_mode, _, _) = curve_payload(&resolve_input(root.path(), &declared)?)?;
        assert_eq!(thinking_mode.as_deref(), Some(DEFAULT_THINKING_MODE));

        declared.curve.as_mut().ok_or("curve form")?.thinking_mode =
            Some("thinking_off".to_owned());
        let (_, thinking_mode, _, _) = curve_payload(&resolve_input(root.path(), &declared)?)?;
        assert_eq!(thinking_mode.as_deref(), Some("thinking_off"));
        Ok(())
    }

    #[test]
    fn declared_thinking_mode_against_a_flat_entry_fails() -> Result<(), Box<dyn std::error::Error>>
    {
        let root = tempfile::tempdir()?;
        let mut declared = write_curve(root.path(), "curves/golden.yaml", "model:\n  - 4: 3.5\n")?;
        declared.curve.as_mut().ok_or("curve form")?.thinking_mode = Some("thinking_on".to_owned());

        let error = resolve_input(root.path(), &declared)
            .err()
            .ok_or("must fail")?;
        assert!(error.contains("thinking_mode"), "{error}");
        assert!(error.contains("flat list"), "{error}");
        assert!(error.contains("model"), "{error}");
        Ok(())
    }

    #[test]
    fn digest_mismatch_names_both_digests() -> Result<(), Box<dyn std::error::Error>> {
        let root = tempfile::tempdir()?;
        let mut declared = write_curve(root.path(), "curves/golden.yaml", "model:\n  - 4: 3.5\n")?;
        declared.curve.as_mut().ok_or("curve form")?.expected_sha256 = "a".repeat(64);

        let error = resolve_input(root.path(), &declared)
            .err()
            .ok_or("must fail")?;
        assert!(error.contains("digest mismatch"), "{error}");
        assert!(error.contains(&"a".repeat(64)), "{error}");
        Ok(())
    }

    #[test]
    fn missing_curve_file_names_the_context_and_path() -> Result<(), Box<dyn std::error::Error>> {
        let root = tempfile::tempdir()?;
        let declared = curve_declaration("curves/absent.yaml", b"model:\n  - 4: 3.5\n");

        let error = resolve_input(root.path(), &declared)
            .err()
            .ok_or("must fail")?;
        assert!(
            error.contains("server \"server\" synthetic_acceptance.curve"),
            "{error}"
        );
        assert!(error.contains("curves/absent.yaml"), "{error}");
        assert!(error.contains("missing or unreadable"), "{error}");
        Ok(())
    }

    // The keyed lookup belongs to the integration ([[ADR-0043]]): an unknown
    // model key ships the text with the declared mode so the integration can
    // fail planning naming the missing element.
    #[test]
    fn unknown_model_key_defers_to_the_integration() -> Result<(), Box<dyn std::error::Error>> {
        let root = tempfile::tempdir()?;
        let mut declared = write_curve(
            root.path(),
            "curves/golden.yaml",
            "model:\n  thinking_on:\n    4: 3.5\n",
        )?;
        let curve = declared.curve.as_mut().ok_or("curve form")?;
        curve.model_key = "other".to_owned();
        curve.thinking_mode = Some("thinking_off".to_owned());

        let (model_key, thinking_mode, _, _) =
            curve_payload(&resolve_input(root.path(), &declared)?)?;
        assert_eq!(model_key, "other");
        assert_eq!(thinking_mode.as_deref(), Some("thinking_off"));
        Ok(())
    }

    #[test]
    fn invalid_curve_shapes_and_values_fail_resolution() -> Result<(), Box<dyn std::error::Error>> {
        let root = tempfile::tempdir()?;
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
            // Every model entry is shape-validated, not only the declared key.
            (
                "other: 3.5\nmodel:\n  - 4: 3.5\n",
                "flat list of draft-length entries or a thinking-mode mapping",
            ),
        ] {
            let declared = write_curve(root.path(), "curves/golden.yaml", text)?;
            let error = resolve_input(root.path(), &declared)
                .err()
                .ok_or("must fail")?;
            assert!(error.contains(expected), "{text:?}: {error}");
        }
        Ok(())
    }

    #[test]
    fn explicit_form_ships_the_declared_length() -> Result<(), Box<dyn std::error::Error>> {
        let root = tempfile::tempdir()?;
        let declared = SyntheticAcceptanceDefinition {
            acceptance_length: Some(2.25),
            curve: None,
        };

        let input = resolve_input(root.path(), &declared)?;
        let SyntheticAcceptanceInput::Explicit { acceptance_length } = input else {
            return Err(format!("expected the explicit form, got {input:?}").into());
        };
        assert_eq!(acceptance_length, 2.25);
        Ok(())
    }
}
