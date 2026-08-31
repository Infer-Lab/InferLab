"""Synthetic acceptance overlay onto the operator's speculative configuration.

[[RFC-0003:C-SERVE-SYNTHETIC-ACCEPTANCE]]: the integration overlays the
effective acceptance length onto the operator-declared speculative
configuration; InferLab never models the speculative method or draft model.
SGLang takes the overlay as `SGLANG_SIMULATE_ACC_*` environment variables,
rendered per engine process. For the curve form the integration determines
the draft count from the operator's `--speculative-num-steps` and resolves
the effective length from the digest-verified curve text ([[ADR-0043]]); the
same resolution runs at plan and at render, so both see one effective value.
"""

from inferlab_adapter_sdk import (
    AdapterErrorCode,
    AdapterOperationError,
    SyntheticAcceptanceInput,
    SyntheticAcceptanceInput2,
    SyntheticAcceptanceOutcome,
    resolve_golden_acceptance_length,
)

from .settings import SglangServeSettings

# The environment keys the overlay owns; an operator restating them in
# extra_env would create a second authority for the effective acceptance
# length. Spellings match the upstream InferenceX injector.
_OVERLAY_ENV_PREFIX = "SGLANG_SIMULATE_ACC_"
_SPECULATIVE_FLAG_PREFIX = "--speculative-"
_NUM_STEPS_FLAG = "--speculative-num-steps"


def _parse_num_steps(value: str, role_id: str) -> int:
    try:
        return int(value)
    except ValueError as error:
        raise AdapterOperationError(
            AdapterErrorCode.invalid_settings,
            f"role {role_id!r} declares {_NUM_STEPS_FLAG}={value!r}, which is not "
            "an integer; the curve form of the synthetic acceptance declaration "
            "needs it as the curve lookup coordinate (use the explicit form "
            "otherwise)",
        ) from error


def _draft_count(extra_args: list[str], role_id: str) -> int:
    """The operator's speculative step count: the curve lookup coordinate.

    Engine last-wins parsing makes the last occurrence the effective one. A
    present flag with a malformed value is determinable intent, not an absent
    signal: fail rather than silently skip the determination.
    """
    num_steps: int | None = None
    index = 0
    while index < len(extra_args):
        argument = extra_args[index]
        if argument == _NUM_STEPS_FLAG and index + 1 < len(extra_args):
            num_steps = _parse_num_steps(extra_args[index + 1], role_id)
            index += 2
            continue
        if argument.startswith(f"{_NUM_STEPS_FLAG}="):
            num_steps = _parse_num_steps(argument.partition("=")[2], role_id)
        index += 1
    if num_steps is None:
        raise AdapterOperationError(
            AdapterErrorCode.invalid_settings,
            f"role {role_id!r} extra_args do not determine {_NUM_STEPS_FLAG}; the "
            "curve form of the synthetic acceptance declaration needs it as the "
            "curve lookup coordinate (use the explicit form otherwise)",
        )
    return num_steps


def resolve_synthetic_acceptance(
    settings: SglangServeSettings,
    synthetic: SyntheticAcceptanceInput,
    role_id: str,
) -> SyntheticAcceptanceOutcome:
    """Validate the overlay target and resolve the effective acceptance length."""
    restated = [key for key in (settings.extra_env or {}) if key.startswith(_OVERLAY_ENV_PREFIX)]
    if restated:
        raise AdapterOperationError(
            AdapterErrorCode.invalid_settings,
            f"role {role_id!r} extra_env restates {', '.join(sorted(restated))}; the "
            "synthetic acceptance declaration is the single authority for those keys",
        )
    extra_args = settings.extra_args or []
    if not any(argument.startswith(_SPECULATIVE_FLAG_PREFIX) for argument in extra_args):
        raise AdapterOperationError(
            AdapterErrorCode.invalid_settings,
            f"role {role_id!r} declares no --speculative-* flag in extra_args; the "
            "synthetic acceptance overlay requires the operator's speculative "
            "configuration as its target",
        )
    form = synthetic.root
    if isinstance(form, SyntheticAcceptanceInput2):
        # The golden curve's lookup key is the speculative step count
        # (num-steps semantics) from the operator's flags.
        draft_count = _draft_count(extra_args, role_id)
        acceptance_length = resolve_golden_acceptance_length(
            curve_text=form.curve.text,
            model_key=form.curve.model_key,
            thinking_mode=form.curve.thinking_mode,
            draft_count=draft_count,
        )
        return SyntheticAcceptanceOutcome(
            acceptance_length=acceptance_length,
            draft_count=draft_count,
        )
    return SyntheticAcceptanceOutcome(acceptance_length=form.explicit.acceptance_length)


def synthetic_acceptance_env(acceptance_length: float) -> dict[str, str]:
    """The per-process environment carrying the overlay into the engine."""
    return {
        "SGLANG_SIMULATE_ACC_LEN": f"{acceptance_length:g}",
        "SGLANG_SIMULATE_ACC_METHOD": "match-expected",
        "SGLANG_SIMULATE_ACC_TOKEN_MODE": "real-draft-token",
    }
