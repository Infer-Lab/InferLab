//! Golden acceptance-length curve parsing and shape validation
//! ([[RFC-0003:C-SERVE-SYNTHETIC-ACCEPTANCE]]). The control plane owns curve
//! integrity — digest verification and the two-shape contract — while the
//! keyed acceptance-length lookup belongs to the selected integration, where
//! the operator's speculative configuration supplies the draft count
//! ([[ADR-0043]]). Workspace-load validation and serve resolution share this
//! one parser so both stages reject the same document shapes with the same
//! diagnostics.

use crate::InferlabError;
use crate::workspace::definitions::SyntheticAcceptanceCurveDefinition;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::path::Path;

/// One model's shape-validated curve entry kind: the flat list shape or the
/// thinking-mode matrix shape. The values are validated but not consumed by
/// the control plane — the keyed lookup belongs to the integration
/// ([[ADR-0043]]).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum CurveModelEntry {
    Flat,
    Matrix,
}

/// Load-time curve shape validation ([[RFC-0003:C-SERVE-SYNTHETIC-ACCEPTANCE]]).
/// When the curve file is readable and its digest matches the declaration,
/// every model entry must satisfy the two-shape contract with finite values,
/// and a declared thinking mode against the declared model key's flat entry
/// fails because no mode applies. A missing or unreadable file and a digest
/// mismatch are assigned to workspace resolution and defer; an unknown model
/// key belongs to the integration's planning-stage lookup and also defers.
pub(crate) fn validate_curve_shape_at_load(
    workspace_root: &Path,
    context: &str,
    curve: &SyntheticAcceptanceCurveDefinition,
) -> Result<(), InferlabError> {
    let resolved_path = workspace_root.join(&curve.path);
    let Ok(bytes) = std::fs::read(&resolved_path) else {
        return Ok(());
    };
    if format!("{:x}", Sha256::digest(&bytes)) != curve.expected_sha256 {
        return Ok(());
    }
    validate_curve_document(&resolved_path, &bytes, context, curve)?;
    Ok(())
}

/// Parse and shape-validate the complete curve document: every model key is
/// a string, and every model entry is exactly one of the two shapes with
/// positive integer draft lengths and finite acceptance-length values
/// throughout. A declared thinking mode against the declared model key's flat
/// entry fails at both stages; an unknown model key does not fail here — the
/// integration owns the keyed lookup.
pub(crate) fn validate_curve_document(
    resolved_path: &Path,
    bytes: &[u8],
    context: &str,
    curve: &SyntheticAcceptanceCurveDefinition,
) -> Result<BTreeMap<String, CurveModelEntry>, InferlabError> {
    let document: yaml_serde::Value =
        yaml_serde::from_slice(bytes).map_err(|source| InferlabError::ParseYaml {
            path: resolved_path.to_path_buf(),
            source,
        })?;
    let yaml_serde::Value::Mapping(models) = document else {
        return Err(invalid_curve(
            context,
            curve,
            "the curve document must map model keys to acceptance-length entries".to_owned(),
        ));
    };
    let mut entries = BTreeMap::new();
    for (key, entry) in &models {
        let Some(model) = key.as_str() else {
            return Err(invalid_curve(
                context,
                curve,
                format!("curve model key {key:?} must be a string"),
            ));
        };
        entries.insert(
            model.to_owned(),
            shape_validate_entry(context, curve, model, entry)?,
        );
    }
    if let Some(CurveModelEntry::Flat) = entries.get(&curve.model_key)
        && let Some(mode) = &curve.thinking_mode
    {
        return Err(invalid_curve(
            context,
            curve,
            format!(
                "thinking_mode {mode:?} was declared but the curve entry for model key {:?} is a flat list; no mode applies to that entry",
                curve.model_key
            ),
        ));
    }
    Ok(entries)
}

/// Validate the complete shape of one model entry: exactly one of the flat
/// list and thinking-mode matrix shapes, positive integer draft lengths, and
/// finite acceptance-length values throughout — not only at the lookup
/// coordinates ([[RFC-0003:C-SERVE-SYNTHETIC-ACCEPTANCE]]).
fn shape_validate_entry(
    context: &str,
    curve: &SyntheticAcceptanceCurveDefinition,
    model: &str,
    entry: &yaml_serde::Value,
) -> Result<CurveModelEntry, InferlabError> {
    match entry {
        yaml_serde::Value::Sequence(entries) => {
            validate_flat_entries(context, curve, entries)?;
            Ok(CurveModelEntry::Flat)
        }
        yaml_serde::Value::Mapping(modes) => {
            for (mode, drafts) in modes {
                let Some(mode) = mode.as_str() else {
                    return Err(invalid_curve(
                        context,
                        curve,
                        format!(
                            "curve thinking-mode key {mode:?} for model key {model:?} must be a string"
                        ),
                    ));
                };
                let yaml_serde::Value::Mapping(drafts) = drafts else {
                    return Err(invalid_curve(
                        context,
                        curve,
                        format!(
                            "the curve thinking mode {mode:?} for model key {model:?} must map draft lengths to acceptance lengths"
                        ),
                    ));
                };
                validate_matrix_entries(context, curve, drafts)?;
            }
            Ok(CurveModelEntry::Matrix)
        }
        _ => Err(invalid_curve(
            context,
            curve,
            format!(
                "the curve entry for model key {model:?} must be a flat list of draft-length entries or a thinking-mode mapping"
            ),
        )),
    }
}

/// The flat shape: a list of single-entry mappings from a positive integer
/// draft length to a mean acceptance length
/// ([[RFC-0003:C-SERVE-SYNTHETIC-ACCEPTANCE]]).
fn validate_flat_entries(
    context: &str,
    curve: &SyntheticAcceptanceCurveDefinition,
    entries: &[yaml_serde::Value],
) -> Result<(), InferlabError> {
    for entry in entries {
        let yaml_serde::Value::Mapping(mapping) = entry else {
            return Err(invalid_curve(
                context,
                curve,
                "every flat curve entry must be a single-entry mapping from a draft length to a mean acceptance length"
                    .to_owned(),
            ));
        };
        if mapping.len() != 1 {
            return Err(invalid_curve(
                context,
                curve,
                "every flat curve entry must carry exactly one draft length".to_owned(),
            ));
        }
        let (draft, length) = mapping
            .iter()
            .next()
            .ok_or_else(|| invalid_curve(context, curve, "flat curve entry is empty".to_owned()))?;
        draft_length(context, curve, draft)?;
        acceptance(context, curve, length)?;
    }
    Ok(())
}

/// The matrix shape's per-mode mapping from a positive integer draft length
/// to a mean acceptance length ([[RFC-0003:C-SERVE-SYNTHETIC-ACCEPTANCE]]).
fn validate_matrix_entries(
    context: &str,
    curve: &SyntheticAcceptanceCurveDefinition,
    drafts: &yaml_serde::Mapping,
) -> Result<(), InferlabError> {
    for (draft, length) in drafts {
        draft_length(context, curve, draft)?;
        acceptance(context, curve, length)?;
    }
    Ok(())
}

fn draft_length(
    context: &str,
    curve: &SyntheticAcceptanceCurveDefinition,
    key: &yaml_serde::Value,
) -> Result<u64, InferlabError> {
    match key.as_u64() {
        Some(draft) if draft > 0 => Ok(draft),
        _ => Err(invalid_curve(
            context,
            curve,
            format!("curve draft-length key {key:?} must be a positive integer"),
        )),
    }
}

fn acceptance(
    context: &str,
    curve: &SyntheticAcceptanceCurveDefinition,
    value: &yaml_serde::Value,
) -> Result<f64, InferlabError> {
    match value.as_f64() {
        Some(length) if length.is_finite() => Ok(length),
        _ => Err(invalid_curve(
            context,
            curve,
            format!("curve acceptance-length value {value:?} must be a finite number"),
        )),
    }
}

/// The shared curve diagnostic: the synthetic-acceptance context and the
/// declared workspace-relative path frame every shape and integrity failure.
pub(crate) fn invalid_curve(
    context: &str,
    curve: &SyntheticAcceptanceCurveDefinition,
    detail: String,
) -> InferlabError {
    InferlabError::InvalidConfig {
        message: format!("{context} at {:?}: {detail}", curve.path),
    }
}
