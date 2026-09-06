"""Auxiliary weight artifact rejection ([[RFC-0003:C-SERVE-AUXILIARY-MODELS]]).

The TokenSpeed render contract carries no speculative-decoding draft weights,
so the integration consumes no declared auxiliary kind. Planning rejects any
declaration with a typed error naming the kind instead of silently dropping
the artifact the operator declared; the declaration is the single authority
for the artifact, and an integration that cannot splice it must say so before
allocation.
"""

from inferlab_adapter_sdk import (
    AdapterErrorCode,
    AdapterOperationError,
    PlanServeInput,
)


def validate_auxiliary_models(input: PlanServeInput) -> None:
    """Reject every declared auxiliary kind; a kind outside the wire
    vocabulary never reaches here — request deserialization rejects it
    ([[RFC-0006:C-INTEGRATIONS]])."""
    for auxiliary in input.auxiliary_models or []:
        raise AdapterOperationError(
            AdapterErrorCode.invalid_settings,
            f"the TokenSpeed integration does not consume the "
            f"{auxiliary.kind.root!r} auxiliary model kind; remove it from the "
            "server's auxiliary_models declaration",
        )
