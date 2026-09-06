"""Auxiliary weight artifact consumption ([[RFC-0003:C-SERVE-AUXILIARY-MODELS]]).

A declared ``draft-model`` auxiliary is the separate weights the operator's
``--speculative-config`` JSON drafts with. Planning validates that every
model-serving role carries that splice target and that the JSON does not
already set its ``model`` key — the declaration is the single authority for
the artifact. The machine-resolved locator arrives on each model-rank
allocation, and rendering splices it into the effective speculative-config
JSON so the rendered command carries the final value as evidence.
"""

from inferlab_adapter_sdk import (
    AdapterErrorCode,
    AdapterOperationError,
    AuxiliaryModelInput,
    PlanServeInput,
    ServeProcessAllocationModelRank,
    resolve_draft_model_locator,
)

from .settings import VllmServeSettings, _settings
from .speculative import _SPECULATIVE_CONFIG_OPTION, _speculative_config_target

# The draft weights' key inside the speculative-config JSON; an operator
# restating it would create a second authority for the artifact.
_DRAFT_MODEL_KEY = "model"
_PURPOSE = "the draft-model auxiliary splice"


def _target_config(settings: VllmServeSettings, context: str) -> dict[str, object]:
    """Locate and parse the splice-target speculative-config JSON object."""
    return _speculative_config_target(settings, context, purpose=_PURPOSE)


def _check_no_collision(config: dict[str, object], context: str) -> None:
    if _DRAFT_MODEL_KEY in config:
        raise AdapterOperationError(
            AdapterErrorCode.invalid_settings,
            f"the {_SPECULATIVE_CONFIG_OPTION} JSON in {context} extra_args already sets "
            f"{_DRAFT_MODEL_KEY!r}; the auxiliary_models declaration is the single "
            "authority for the draft-model artifact",
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
        config = _target_config(settings, f"role {role.id!r}")
        _check_no_collision(config, f"role {role.id!r}")


def splice_draft_model(
    settings: VllmServeSettings,
    allocation: ServeProcessAllocationModelRank,
    declared: list[AuxiliaryModelInput] | None,
) -> None:
    """Splice the allocation's resolved draft-model locator into the effective
    speculative-config JSON carried by the role's extra_args."""
    locator = resolve_draft_model_locator(allocation, declared)
    if locator is None:
        return
    context = f"allocation {allocation.process!r}"
    config = _target_config(settings, context)
    _check_no_collision(config, context)
    _speculative_config_target(
        settings, context, purpose=_PURPOSE, patch={_DRAFT_MODEL_KEY: locator}
    )
