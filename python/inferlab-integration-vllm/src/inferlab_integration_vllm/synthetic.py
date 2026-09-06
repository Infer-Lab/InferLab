"""Synthetic acceptance overlay onto the operator's speculative configuration.

[[RFC-0003:C-SERVE-SYNTHETIC-ACCEPTANCE]]: the integration overlays the
effective acceptance length onto the operator-declared `--speculative-config`
JSON in the role's `extra_args`; InferLab never models the speculative method
or draft model. For the curve form the integration determines the draft count
from that JSON's `num_speculative_tokens` and resolves the effective length
from the digest-verified curve text ([[ADR-0043]]). The patch lands at plan
time so the plan response's effective settings carry the final injected JSON
as evidence, and rendering consumes those effective settings unchanged.
"""

from inferlab_adapter_sdk import (
    AdapterErrorCode,
    AdapterOperationError,
    SyntheticAcceptanceInput,
    SyntheticAcceptanceInput2,
    SyntheticAcceptanceOutcome,
    resolve_golden_acceptance_length,
)

from .settings import VllmServeSettings
from .speculative import _SPECULATIVE_CONFIG_OPTION, _speculative_config_target

# The rejection-sampling keys the overlay owns; an operator restating them
# would create a second authority for the effective acceptance length.
_OVERLAY_KEYS = ("rejection_sample_method", "synthetic_acceptance_length")
_DRAFT_COUNT_KEY = "num_speculative_tokens"
_PURPOSE = "the synthetic acceptance overlay"


def _draft_count(config: dict[str, object], role_id: str) -> int:
    """The operator's draft count: the sole authority for the curve lookup."""
    declared = config.get(_DRAFT_COUNT_KEY)
    if isinstance(declared, bool):
        declared = None
    if isinstance(declared, float) and declared.is_integer():
        declared = int(declared)
    if not isinstance(declared, int):
        raise AdapterOperationError(
            AdapterErrorCode.invalid_settings,
            f"the {_SPECULATIVE_CONFIG_OPTION} JSON in role {role_id!r} extra_args does not "
            f"determine an integer {_DRAFT_COUNT_KEY}; the curve form of the "
            "synthetic acceptance declaration needs it as the curve lookup "
            "coordinate (use the explicit form otherwise)",
        )
    return declared


def apply_synthetic_acceptance(
    settings: VllmServeSettings,
    synthetic: SyntheticAcceptanceInput,
    role_id: str,
) -> SyntheticAcceptanceOutcome:
    """Patch the role's speculative-config JSON and return the outcome."""
    context = f"role {role_id!r}"
    config = _speculative_config_target(settings, context, purpose=_PURPOSE)
    carried = [key for key in _OVERLAY_KEYS if key in config]
    if carried:
        raise AdapterOperationError(
            AdapterErrorCode.invalid_settings,
            f"the {_SPECULATIVE_CONFIG_OPTION} JSON in role {role_id!r} extra_args already sets "
            f"{', '.join(carried)}; the synthetic acceptance declaration is the "
            "single authority for those keys",
        )
    form = synthetic.root
    draft_count: int | None = None
    if isinstance(form, SyntheticAcceptanceInput2):
        draft_count = _draft_count(config, role_id)
        acceptance_length = resolve_golden_acceptance_length(
            curve_text=form.curve.text,
            model_key=form.curve.model_key,
            thinking_mode=form.curve.thinking_mode,
            draft_count=draft_count,
        )
    else:
        acceptance_length = form.explicit.acceptance_length
    _speculative_config_target(
        settings,
        context,
        purpose=_PURPOSE,
        patch={
            "rejection_sample_method": "synthetic",
            "synthetic_acceptance_length": acceptance_length,
        },
    )
    return SyntheticAcceptanceOutcome(
        acceptance_length=acceptance_length,
        draft_count=draft_count,
    )
