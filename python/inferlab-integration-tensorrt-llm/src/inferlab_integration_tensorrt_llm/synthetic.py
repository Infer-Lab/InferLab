"""Synthetic acceptance overlay spelling for TensorRT-LLM.

[[RFC-0003:C-SERVE-SYNTHETIC-ACCEPTANCE]]: the integration overlays the
resolved acceptance length onto the operator-declared speculative
configuration; InferLab never models the speculative method or draft model.
TensorRT-LLM takes the overlay as a forced accepted-token count in the
process environment; planning validates that the operator's
`extra_llm_api_options` actually declare a `speculative_config` to overlay.
"""

# The environment variable the overlay owns; an operator restating it in
# extra_env would create a second authority for the effective acceptance
# length. Spelling matches the upstream InferenceX recipe.
FORCE_ACCEPTED_TOKENS_ENV = "TLLM_SPEC_DECODE_FORCE_NUM_ACCEPTED_TOKENS"


def synthetic_acceptance_env(acceptance_length: float) -> dict[str, str]:
    """The per-process environment carrying the overlay into the engine.

    The variable counts accepted draft tokens and excludes the bonus
    verification token, so it carries one less than the golden mean
    acceptance length (the upstream off-by-one); fractional values are
    supported and rendered without trailing zeros.
    """
    return {FORCE_ACCEPTED_TOKENS_ENV: f"{acceptance_length - 1:g}"}
