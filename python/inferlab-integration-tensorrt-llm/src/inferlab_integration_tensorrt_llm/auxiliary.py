"""Auxiliary weight artifact consumption ([[RFC-0003:C-SERVE-AUXILIARY-MODELS]]).

A declared ``draft-model`` auxiliary is the separate weights the operator's
``speculative_config`` drafts with. Planning validates that every
model-serving role's effective ``extra_llm_api_options`` (source YAML plus
patch) carries that mapping as the splice target and does not already set its
``speculative_model`` key — the declaration is the single authority for the
artifact. The machine-resolved locator arrives on each model-rank allocation,
and rendering splices it into the merged launch-file YAML so the rendered
configuration carries the final value as evidence.
"""

from inferlab_adapter_sdk import (
    AdapterErrorCode,
    AdapterOperationError,
    AuxiliaryModelInput,
    PlanServeInput,
    ServeProcessAllocationModelRank,
    resolve_draft_model_locator,
)

from .settings import _operator_config, _settings, _yaml_mapping

_SPLICE_TARGET = "speculative_config"
# The draft weights' key inside the speculative_config mapping; an operator
# restating it would create a second authority for the artifact.
_DRAFT_MODEL_KEY = "speculative_model"


def _splice_target(config: dict[str, object], context: str) -> dict[str, object]:
    """The speculative_config mapping the draft-model locator splices into."""
    value = config.get(_SPLICE_TARGET)
    if value is None:
        raise AdapterOperationError(
            AdapterErrorCode.invalid_settings,
            f"{context} declares no {_SPLICE_TARGET} in its extra_llm_api_options "
            "source YAML or patch; the draft-model auxiliary model requires the "
            "operator's speculative configuration as its splice target",
        )
    return _yaml_mapping(value, _SPLICE_TARGET)


def _check_no_collision(speculative: dict[str, object], context: str) -> None:
    if _DRAFT_MODEL_KEY in speculative:
        raise AdapterOperationError(
            AdapterErrorCode.invalid_settings,
            f"the {_SPLICE_TARGET} in {context} extra_llm_api_options already sets "
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
        config = _operator_config(settings, input.state_dir)
        _check_no_collision(_splice_target(config, f"role {role.id!r}"), f"role {role.id!r}")


def splice_draft_model(
    config: dict[str, object],
    allocation: ServeProcessAllocationModelRank,
    auxiliary_models: list[AuxiliaryModelInput] | None,
) -> None:
    """Splice the allocation's resolved draft-model locator into the merged
    extra_llm_api_options mapping that rendering turns into the launch file."""
    locator = resolve_draft_model_locator(allocation, auxiliary_models)
    if locator is None:
        return
    context = f"allocation {allocation.process!r}"
    speculative = _splice_target(config, context)
    _check_no_collision(speculative, context)
    speculative[_DRAFT_MODEL_KEY] = locator
