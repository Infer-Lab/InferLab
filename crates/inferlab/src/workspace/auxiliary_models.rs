//! The governed auxiliary weight kind vocabulary and server-declaration
//! validation ([[RFC-0003:C-SERVE-AUXILIARY-MODELS]]). The kind names the
//! role of a weight artifact, not a speculative method; the vocabulary grows
//! only with a demonstrated workflow. Resolution reuses the standing
//! machine-local model weight bindings ([[RFC-0002:C-LOCAL-PLACEMENT]]).

use crate::InferlabError;
use crate::workspace::definitions::WorkspaceConfig;
use crate::workspace::invalid;
use inferlab_protocol::AuxiliaryModelKind as WireKind;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum AuxiliaryModelKind {
    /// Separate weights the operator's framework speculative decoding
    /// configuration consumes as its draft.
    DraftModel,
}

impl AuxiliaryModelKind {
    pub(crate) const DRAFT_MODEL: &'static str = "draft-model";

    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::DraftModel => Self::DRAFT_MODEL,
        }
    }

    pub(crate) fn parse(context: &str, declared: &str) -> Result<Self, InferlabError> {
        match declared {
            Self::DRAFT_MODEL => Ok(Self::DraftModel),
            _ => invalid(format!(
                "{context} auxiliary_models declares unknown kind {declared:?}; the governed vocabulary is {:?}",
                [Self::DRAFT_MODEL]
            )),
        }
    }

    pub(crate) const fn to_wire(self) -> WireKind {
        match self {
            Self::DraftModel => WireKind::DraftModel,
        }
    }
}

pub(crate) fn validate_auxiliary_models(
    context: &str,
    server_model: &str,
    auxiliary_models: &std::collections::BTreeMap<String, String>,
    config: &WorkspaceConfig,
) -> Result<(), InferlabError> {
    for (kind, model) in auxiliary_models {
        AuxiliaryModelKind::parse(context, kind)?;
        if model == server_model {
            return invalid(format!(
                "{context} auxiliary_models kind {kind:?} references the server's own model {model:?}; auxiliary weights must be independent of the server's model"
            ));
        }
        if !config.models.contains_key(model) {
            return invalid(format!(
                "{context} auxiliary_models kind {kind:?} references unknown model {model:?}"
            ));
        }
    }
    Ok(())
}
