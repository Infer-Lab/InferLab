"""Auxiliary weight artifact rejection ([[RFC-0003:C-SERVE-AUXILIARY-MODELS]]).

The Specialized Engine contract renders a fixed token-worker command with no
speculative-decoding configuration, so this integration consumes no auxiliary
model kind. Planning rejects any declaration with a typed error naming the
kind, and rendering rejects an allocation that still carries resolved
locators rather than silently dropping a declared artifact from the record.
"""

from inferlab_adapter_sdk import (
    AdapterErrorCode,
    AdapterOperationError,
    PlanServeInput,
    ServeProcessAllocationModelRank,
)


def validate_auxiliary_models(input: PlanServeInput) -> None:
    """Reject every declared auxiliary kind before allocation; the Specialized
    Engine contract has no splice target for auxiliary weights."""
    for auxiliary in input.auxiliary_models or []:
        raise AdapterOperationError(
            AdapterErrorCode.invalid_settings,
            f"the Specialized Engine integration does not consume the "
            f"{auxiliary.kind.root!r} auxiliary model kind; remove the "
            "auxiliary_models declaration",
        )


def reject_auxiliary_locators(allocation: ServeProcessAllocationModelRank) -> None:
    """Reject a model-rank allocation carrying resolved auxiliary locators;
    planning already refused the declaration, so rendering must not drop the
    resolved artifact from the record silently."""
    for entry in allocation.auxiliary_model_locators or []:
        raise AdapterOperationError(
            AdapterErrorCode.invalid_request,
            f"allocation {allocation.process!r} carries a resolved "
            f"{entry.kind.root!r} auxiliary model locator, but the Specialized "
            "Engine integration consumes no auxiliary model kind",
        )
