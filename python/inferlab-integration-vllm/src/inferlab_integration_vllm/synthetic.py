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

import json

from inferlab_adapter_sdk import (
    AdapterErrorCode,
    AdapterOperationError,
    SyntheticAcceptanceInput,
    SyntheticAcceptanceInput2,
    SyntheticAcceptanceOutcome,
    resolve_golden_acceptance_length,
)

from .settings import VllmServeSettings

_OVERLAY_OPTION = "--speculative-config"
# The rejection-sampling keys the overlay owns; an operator restating them
# would create a second authority for the effective acceptance length.
_OVERLAY_KEYS = ("rejection_sample_method", "synthetic_acceptance_length")
_DRAFT_COUNT_KEY = "num_speculative_tokens"


def _overlay_target(extra_args: list[str], role_id: str) -> tuple[int, bool]:
    """Locate the effective speculative-config value token.

    Returns ``(index, inline)`` where ``inline`` selects the
    ``--speculative-config=<json>`` spelling over the separate-token spelling.
    Engine last-wins parsing makes the last occurrence the effective one.
    """
    target: tuple[int, bool] | None = None
    index = 0
    while index < len(extra_args):
        argument = extra_args[index]
        if argument == _OVERLAY_OPTION:
            if index + 1 >= len(extra_args):
                raise AdapterOperationError(
                    AdapterErrorCode.invalid_settings,
                    f"role {role_id!r} extra_args entry {_OVERLAY_OPTION!r} is missing its "
                    "JSON value; the synthetic acceptance overlay needs that operator "
                    "speculative configuration as its target",
                )
            target = (index + 1, False)
            index += 2
            continue
        if argument.startswith(f"{_OVERLAY_OPTION}="):
            target = (index, True)
        index += 1
    if target is None:
        raise AdapterOperationError(
            AdapterErrorCode.invalid_settings,
            f"role {role_id!r} declares no {_OVERLAY_OPTION} JSON in extra_args; the "
            "synthetic acceptance overlay requires the operator's speculative "
            "configuration as its target",
        )
    return target


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
            f"the {_OVERLAY_OPTION} JSON in role {role_id!r} extra_args does not "
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
    extra_args = list(settings.extra_args or [])
    index, inline = _overlay_target(extra_args, role_id)
    token = extra_args[index]
    text = token.partition("=")[2] if inline else token
    try:
        config: object = json.loads(text)
    except json.JSONDecodeError as error:
        raise AdapterOperationError(
            AdapterErrorCode.invalid_settings,
            f"cannot parse the {_OVERLAY_OPTION} JSON in role {role_id!r} extra_args: "
            f"{error}; the synthetic acceptance overlay needs that operator "
            "speculative configuration as its target",
        ) from error
    if not isinstance(config, dict):
        raise AdapterOperationError(
            AdapterErrorCode.invalid_settings,
            f"the {_OVERLAY_OPTION} JSON in role {role_id!r} extra_args must be an "
            "object for the synthetic acceptance overlay to patch",
        )
    carried = [key for key in _OVERLAY_KEYS if key in config]
    if carried:
        raise AdapterOperationError(
            AdapterErrorCode.invalid_settings,
            f"the {_OVERLAY_OPTION} JSON in role {role_id!r} extra_args already sets "
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
    config["rejection_sample_method"] = "synthetic"
    config["synthetic_acceptance_length"] = acceptance_length
    patched = json.dumps(config, sort_keys=True, separators=(",", ":"))
    extra_args[index] = f"{_OVERLAY_OPTION}={patched}" if inline else patched
    settings.extra_args = extra_args
    return SyntheticAcceptanceOutcome(
        acceptance_length=acceptance_length,
        draft_count=draft_count,
    )
