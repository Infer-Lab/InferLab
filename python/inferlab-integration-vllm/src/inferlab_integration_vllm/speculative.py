"""Locating and patching the operator's speculative-config JSON in settings.

The draft-model auxiliary splice ([[RFC-0003:C-SERVE-AUXILIARY-MODELS]]) and
the synthetic acceptance overlay ([[RFC-0003:C-SERVE-SYNTHETIC-ACCEPTANCE]])
patch the same operator-declared ``--speculative-config`` value. This module
owns the shared mechanics — the SDK's engine last-wins scan of the two
option spellings, the JSON object shape, and the serialize-and-writeback of
a consumer's patch — so each consumer applies only its own keys. The error
messages are one canonical set parameterized by the consumer's ``purpose``
noun-phrase; the context (role or allocation id) and the typed error code
carry the actionable identity.
"""

import json
from collections.abc import Mapping

from inferlab_adapter_sdk import AdapterErrorCode, AdapterOperationError, last_option_value

from .settings import VllmServeSettings

_SPECULATIVE_CONFIG_OPTION = "--speculative-config"


def _speculative_config_target(
    settings: VllmServeSettings,
    context: str,
    *,
    purpose: str,
    patch: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Locate and parse the effective speculative-config JSON object.

    With ``patch``, merge its keys into the located object and write the
    patched JSON back over the effective option token, preserving the
    operator's spelling; the mutated settings carry the final value as
    evidence. Without ``patch`` the settings stay untouched, which is the
    plan-time validation form.
    """
    extra_args = list(settings.extra_args or [])
    target = last_option_value(
        extra_args, _SPECULATIVE_CONFIG_OPTION, context=context, purpose=purpose
    )
    if target is None:
        raise AdapterOperationError(
            AdapterErrorCode.invalid_settings,
            f"{context} declares no {_SPECULATIVE_CONFIG_OPTION} JSON in extra_args; "
            f"{purpose} requires it as its target",
        )
    index, inline, text = target
    try:
        config: object = json.loads(text)
    except json.JSONDecodeError as error:
        raise AdapterOperationError(
            AdapterErrorCode.invalid_settings,
            f"cannot parse the {_SPECULATIVE_CONFIG_OPTION} JSON in {context} extra_args: "
            f"{error}; {purpose} needs it",
        ) from error
    if not isinstance(config, dict):
        raise AdapterOperationError(
            AdapterErrorCode.invalid_settings,
            f"the {_SPECULATIVE_CONFIG_OPTION} JSON in {context} extra_args must be an "
            f"object for {purpose}",
        )
    if patch is not None:
        config.update(patch)
        patched = json.dumps(config, sort_keys=True, separators=(",", ":"))
        extra_args[index] = f"{_SPECULATIVE_CONFIG_OPTION}={patched}" if inline else patched
        settings.extra_args = extra_args
    return config
