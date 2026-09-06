"""Auxiliary weight artifact consumption ([[RFC-0003:C-SERVE-AUXILIARY-MODELS]]).

A declared ``draft-model`` auxiliary is the separate draft weights the
operator's SGLang speculative decoding configuration drafts with. Planning
validates that every model-serving role declares speculative decoding
(``--speculative-algorithm``) as the splice target and that its extra_args do
not already spell ``--speculative-draft-model-path`` — the declaration is the
single authority for the artifact. The machine-resolved locator arrives on
each model-rank allocation, and rendering splices it into the role's
extra_args so the rendered command carries the final value as evidence.
"""

from inferlab_adapter_sdk import (
    AdapterErrorCode,
    AdapterOperationError,
    AuxiliaryModelInput,
    PlanServeInput,
    ServeProcessAllocationModelRank,
    resolve_draft_model_locator,
)

from .settings import SglangServeSettings, _settings

# The operator's speculative decoding declaration: the splice target the
# draft-model auxiliary rides on. Without it the engine never consumes a
# draft model path.
_SPLICE_TARGET = "--speculative-algorithm"
# The framework spelling of the draft weights artifact; an operator restating
# it would create a second authority for the artifact.
_DRAFT_MODEL_FLAG = "--speculative-draft-model-path"


def _mentions(extra_args: list[str], flag: str) -> bool:
    """Whether the flag appears in either the separate-token or the
    ``--flag=value`` spelling. Engine last-wins parsing makes any mention
    effective, so the scan does not stop at the composition sentinel."""
    return any(argument == flag or argument.startswith(f"{flag}=") for argument in extra_args)


def _check_splice_target(extra_args: list[str], context: str) -> None:
    if not _mentions(extra_args, _SPLICE_TARGET):
        raise AdapterOperationError(
            AdapterErrorCode.invalid_settings,
            f"{context} declares no {_SPLICE_TARGET} flag in extra_args; the "
            "draft-model auxiliary model requires the operator's speculative "
            "decoding configuration as its splice target",
        )


def _check_no_collision(extra_args: list[str], context: str) -> None:
    if _mentions(extra_args, _DRAFT_MODEL_FLAG):
        raise AdapterOperationError(
            AdapterErrorCode.invalid_settings,
            f"{context} extra_args already spell {_DRAFT_MODEL_FLAG!r}; the "
            "auxiliary_models declaration is the single authority for the "
            "draft-model artifact",
        )


def validate_auxiliary_models(input: PlanServeInput) -> None:
    """Validate at plan time that every requested role can consume each
    declared auxiliary kind; unconsumable declarations fail before allocation.
    A kind outside the wire vocabulary never reaches here — request
    deserialization rejects it ([[RFC-0006:C-INTEGRATIONS]])."""
    if not input.auxiliary_models:
        return
    for role in input.roles:
        settings = _settings(role.settings)
        extra_args = list(settings.extra_args or [])
        _check_splice_target(extra_args, f"role {role.id!r}")
        _check_no_collision(extra_args, f"role {role.id!r}")


def splice_draft_model(
    settings: SglangServeSettings,
    allocation: ServeProcessAllocationModelRank,
    declared: list[AuxiliaryModelInput] | None,
) -> None:
    """Splice the allocation's resolved draft-model locator into the role's
    extra_args as ``--speculative-draft-model-path``."""
    locator = resolve_draft_model_locator(allocation, declared)
    if locator is None:
        return
    extra_args = list(settings.extra_args or [])
    context = f"allocation {allocation.process!r}"
    # Plan-time validation already ran on these settings; revalidating here
    # keeps a hand-built render request from bypassing it.
    _check_splice_target(extra_args, context)
    _check_no_collision(extra_args, context)
    pair = [_DRAFT_MODEL_FLAG, locator]
    if "--" in extra_args:
        # The spliced pair belongs with the operator args; the post-sentinel
        # block stays the operator's deliberate verbatim override tail.
        index = extra_args.index("--")
        extra_args[index:index] = pair
    else:
        extra_args.extend(pair)
    settings.extra_args = extra_args
