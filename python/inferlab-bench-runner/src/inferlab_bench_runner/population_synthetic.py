"""Materialize deterministic synthetic request populations."""

import hashlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from inferlab_measurement_sdk import (
    BenchInclusiveUniformInput,
    BenchPopulationInput,
    BenchPopulationPreparationRequest,
    BenchPopulationPreparationResult,
    BenchPromptTemplateProjection,
    BenchPromptTemplateSource,
    BenchPromptTokenTargetingSummary,
    BenchRequestSourceInputRandom,
    BenchRequestSourceInputRandomMixture,
    BenchTokenSelectorInput,
    BenchTokenSelectorInput1,
    ClientStatus,
    JsonObject,
    plain_setting,
)

from inferlab_bench_runner.chat_tokens import required_messages_content_tokens
from inferlab_bench_runner.population_types import (
    ChatTokenizer,
    count_summary,
    json_line,
    unbiased_index,
)

SYNTHETIC_MATERIALIZATION_IDENTITY = "inferlab-synthetic-prompt-target-v3"
MAX_EXACT_TARGETING_ATTEMPTS = 32
MAX_EXACT_CONTENT_VARIANTS = 16


class SyntheticTextFactory:
    def __init__(self, tokenizer: ChatTokenizer) -> None:
        self.tokenizer = tokenizer

    @staticmethod
    def _corpus_word(
        seed: int,
        population_index: int,
        label: str,
        word_index: int,
    ) -> str:
        identity = f"{seed}\0{population_index}\0{label}\0{word_index}".encode()
        return f"inferlab_{hashlib.sha256(identity).hexdigest()}"

    def exact_text(
        self,
        target_tokens: int,
        seed: int,
        population_index: int,
        label: str,
    ) -> str:
        words: list[str] = []
        token_ids: list[int] = []
        next_word_count = 128
        while len(token_ids) < target_tokens + 128:
            words.extend(
                self._corpus_word(seed, population_index, label, word_index)
                for word_index in range(len(words), next_word_count)
            )
            token_ids = self.tokenizer.encode(" ".join(words), add_special_tokens=False)
            next_word_count *= 2
        starts = len(token_ids) - target_tokens + 1
        first = unbiased_index(seed, population_index, f"{label}-offset", starts)
        for attempt in range(min(starts, 128)):
            start = (first + attempt) % starts
            text = self.tokenizer.decode(
                token_ids[start : start + target_tokens],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            if len(self.tokenizer.encode(text, add_special_tokens=False)) == target_tokens:
                return text
        raise ValueError(
            f"tokenizer could not round-trip a synthetic {label} of {target_tokens} tokens"
        )


@dataclass(frozen=True)
class SyntheticPromptTargeting:
    messages: list[dict[str, str]]
    pre_template_content_tokens: int
    locally_predicted_prompt_tokens: int | None
    exact: bool
    fallback_reason: str | None
    fallback_detail: str | None


@dataclass(frozen=True)
class PromptProjection:
    template: BenchPromptTemplateProjection
    chat_template_kwargs: dict[str, object]


def token_count(value: object) -> int:
    if isinstance(value, Mapping) and "input_ids" in value:
        return token_count(value["input_ids"])
    if isinstance(value, list):
        return len(value)
    if hasattr(value, "shape"):
        shape = value.shape
        if isinstance(shape, tuple) and shape:
            return int(shape[-1])
    raise TypeError("tokenizer returned an unsupported token container")


def selected_tokens(
    selector: BenchTokenSelectorInput,
    seed: int,
    population_index: int,
    label: str,
) -> int:
    value = selector.root
    if isinstance(value, BenchTokenSelectorInput1):
        return value.root
    if isinstance(value, BenchInclusiveUniformInput):
        return value.min + unbiased_index(seed, population_index, label, value.max - value.min + 1)
    raise TypeError(f"unsupported token selector {type(value).__name__}")


def _resolve_prompt_projection(
    request: BenchPopulationPreparationRequest,
    tokenizer: ChatTokenizer,
) -> tuple[PromptProjection | None, str | None, str | None]:
    request_body = {key: plain_setting(value) for key, value in request.request_body.items()}
    requested_template: str | None = None
    template_source = BenchPromptTemplateSource.tokenizer_default
    if "chat_template" in request_body:
        raw_template = request_body["chat_template"]
        if not isinstance(raw_template, str):
            return (
                None,
                "chat_template_control_not_projectable",
                "request_body.chat_template is not a string",
            )
        requested_template = raw_template
        template_source = BenchPromptTemplateSource.request_body
    raw_kwargs = request_body.get("chat_template_kwargs", {})
    if not isinstance(raw_kwargs, dict):
        return (
            None,
            "chat_template_control_not_projectable",
            "request_body.chat_template_kwargs is not an object",
        )
    template_kwargs = cast(dict[str, object], raw_kwargs)
    candidate = getattr(tokenizer, "get_chat_template", None)
    if not callable(candidate):
        return (
            None,
            "chat_template_resolution_unavailable",
            "resolved tokenizer does not expose get_chat_template",
        )
    get_chat_template = cast(Callable[..., object], candidate)
    try:
        resolved_template = get_chat_template(
            chat_template=requested_template,
            tools=template_kwargs.get("tools"),
        )
    except Exception as error:
        return (
            None,
            "chat_template_resolution_failed",
            f"{type(error).__name__}: {error}",
        )
    if not isinstance(resolved_template, str):
        return (
            None,
            "chat_template_resolution_failed",
            "resolved tokenizer returned a non-string chat template",
        )
    return (
        PromptProjection(
            template=BenchPromptTemplateProjection(
                source=template_source,
                content=resolved_template,
                sha256=hashlib.sha256(resolved_template.encode()).hexdigest(),
            ),
            chat_template_kwargs=template_kwargs,
        ),
        None,
        None,
    )


def _project_prompt_tokens(
    tokenizer: ChatTokenizer,
    messages: list[dict[str, str]],
    projection: PromptProjection,
) -> tuple[int | None, str | None, str | None]:
    candidate = getattr(tokenizer, "apply_chat_template", None)
    if not callable(candidate):
        return (
            None,
            "chat_template_projection_unavailable",
            "resolved tokenizer does not expose apply_chat_template",
        )
    apply_chat_template = cast(Callable[..., object], candidate)
    kwargs = dict(projection.chat_template_kwargs)
    kwargs["chat_template"] = projection.template.content
    try:
        projected = apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            **kwargs,
        )
        return token_count(projected), None, None
    except Exception as error:
        return (
            None,
            "chat_template_projection_failed",
            f"{type(error).__name__}: {error}",
        )


def _target_synthetic_prompt(
    tokenizer: ChatTokenizer,
    text_factory: SyntheticTextFactory,
    projection: PromptProjection | None,
    projection_fallback_reason: str | None,
    projection_fallback_detail: str | None,
    fixed_messages: list[dict[str, str]],
    selected_prompt_tokens: int,
    unadjusted_content_tokens: int,
    seed: int,
    population_index: int,
    label: str,
) -> SyntheticPromptTargeting:
    def candidate_messages(content_tokens: int, variant: int | None = None) -> list[dict[str, str]]:
        candidate_label = label if variant is None else f"{label}-exact-{variant}"
        return [
            *fixed_messages,
            {
                "role": "user",
                "content": text_factory.exact_text(
                    content_tokens,
                    seed,
                    population_index,
                    candidate_label,
                ),
            },
        ]

    def projected_candidate(
        content_tokens: int,
    ) -> tuple[SyntheticPromptTargeting | None, int | None, str | None, str | None]:
        if projection is None:
            raise ValueError("prompt projection is unavailable")
        first_prediction: int | None = None
        for variant in range(MAX_EXACT_CONTENT_VARIANTS):
            try:
                messages = candidate_messages(content_tokens, variant)
            except ValueError:
                continue
            prediction, reason, detail = _project_prompt_tokens(
                tokenizer,
                messages,
                projection,
            )
            if prediction is None:
                return None, None, reason, detail
            if first_prediction is None:
                first_prediction = prediction
            if prediction == selected_prompt_tokens:
                return (
                    SyntheticPromptTargeting(
                        messages=messages,
                        pre_template_content_tokens=required_messages_content_tokens(
                            messages, tokenizer
                        ),
                        locally_predicted_prompt_tokens=prediction,
                        exact=True,
                        fallback_reason=None,
                        fallback_detail=None,
                    ),
                    prediction,
                    None,
                    None,
                )
        return None, first_prediction, None, None

    unadjusted_messages = candidate_messages(unadjusted_content_tokens)
    if projection is None:
        if projection_fallback_reason is None:
            raise ValueError("unavailable prompt projection omitted its fallback reason")
        return SyntheticPromptTargeting(
            messages=unadjusted_messages,
            pre_template_content_tokens=required_messages_content_tokens(
                unadjusted_messages, tokenizer
            ),
            locally_predicted_prompt_tokens=None,
            exact=False,
            fallback_reason=projection_fallback_reason,
            fallback_detail=projection_fallback_detail,
        )

    unadjusted_prediction, projection_reason, projection_detail = _project_prompt_tokens(
        tokenizer,
        unadjusted_messages,
        projection,
    )
    if unadjusted_prediction is None:
        return SyntheticPromptTargeting(
            messages=unadjusted_messages,
            pre_template_content_tokens=required_messages_content_tokens(
                unadjusted_messages, tokenizer
            ),
            locally_predicted_prompt_tokens=None,
            exact=False,
            fallback_reason=projection_reason,
            fallback_detail=projection_detail,
        )

    if unadjusted_prediction == selected_prompt_tokens:
        return SyntheticPromptTargeting(
            messages=unadjusted_messages,
            pre_template_content_tokens=required_messages_content_tokens(
                unadjusted_messages, tokenizer
            ),
            locally_predicted_prompt_tokens=unadjusted_prediction,
            exact=True,
            fallback_reason=None,
            fallback_detail=None,
        )

    tried = {unadjusted_content_tokens}
    next_content_tokens = unadjusted_content_tokens + selected_prompt_tokens - unadjusted_prediction
    last_content_tokens = next_content_tokens
    for _ in range(MAX_EXACT_TARGETING_ATTEMPTS):
        if next_content_tokens <= 0 or next_content_tokens in tried:
            break
        tried.add(next_content_tokens)
        last_content_tokens = next_content_tokens
        targeting, prediction, reason, detail = projected_candidate(next_content_tokens)
        if targeting is not None:
            return targeting
        if prediction is None:
            if reason is None:
                break
            return SyntheticPromptTargeting(
                messages=unadjusted_messages,
                pre_template_content_tokens=required_messages_content_tokens(
                    unadjusted_messages, tokenizer
                ),
                locally_predicted_prompt_tokens=unadjusted_prediction,
                exact=False,
                fallback_reason=reason,
                fallback_detail=detail,
            )
        next_content_tokens += selected_prompt_tokens - prediction

    neighborhood = (
        value
        for distance in range(1, MAX_EXACT_TARGETING_ATTEMPTS + 1)
        for value in (last_content_tokens - distance, last_content_tokens + distance)
    )
    for content_tokens in neighborhood:
        if content_tokens <= 0 or content_tokens in tried:
            continue
        tried.add(content_tokens)
        targeting, _, _, _ = projected_candidate(content_tokens)
        if targeting is not None:
            return targeting

    return SyntheticPromptTargeting(
        messages=unadjusted_messages,
        pre_template_content_tokens=required_messages_content_tokens(
            unadjusted_messages, tokenizer
        ),
        locally_predicted_prompt_tokens=unadjusted_prediction,
        exact=False,
        fallback_reason="exact_prompt_length_unreachable",
        fallback_detail=(
            f"selected prompt target {selected_prompt_tokens} could not be reached from "
            f"unadjusted content length {unadjusted_content_tokens}"
        ),
    )


def _selected_mixture_shape(
    source: BenchRequestSourceInputRandomMixture,
    seed: int,
    population_index: int,
) -> tuple[int, int]:
    selected = unbiased_index(seed, population_index, "shape", source.total_weight)
    cumulative = 0
    for shape in source.shapes:
        cumulative += shape.weight
        if selected < cumulative:
            return shape.input_tokens, shape.output_tokens
    raise ValueError("random_mixture weights do not cover their resolved total")


def write_synthetic_population(
    request: BenchPopulationPreparationRequest,
    tokenizer: ChatTokenizer,
    source: BenchRequestSourceInputRandom | BenchRequestSourceInputRandomMixture,
) -> BenchPopulationPreparationResult:
    required = request.required_entries
    if required <= 0:
        raise ValueError("population preparation requires at least one entry")
    artifact_dir = Path(request.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    population_path = artifact_dir / "population.jsonl"
    evidence_path = artifact_dir / "population-evidence.jsonl"
    population_digest = hashlib.sha256()
    selected_prompt_counts: list[int] = []
    pre_template_content_counts: list[int] = []
    output_counts: list[int] = []
    exact_entries = 0
    fallback_entries = 0
    fallback_reasons: dict[str, int] = {}
    projection, projection_reason, projection_detail = _resolve_prompt_projection(
        request, tokenizer
    )
    text_factory = SyntheticTextFactory(tokenizer)
    shared_prefix: str | None = None
    if isinstance(source, BenchRequestSourceInputRandom) and source.prefix_sharing is not None:
        shared_prefix = text_factory.exact_text(
            source.prefix_sharing.shared_prefix_tokens,
            request.seed,
            0,
            "shared-prefix",
        )
    with population_path.open("wb") as population_file, evidence_path.open("wb") as evidence_file:
        for index in range(required):
            if isinstance(source, BenchRequestSourceInputRandom):
                input_tokens = selected_tokens(
                    source.input_tokens, request.seed, index, "input_tokens"
                )
                output_tokens = selected_tokens(
                    source.output_tokens, request.seed, index, "output_tokens"
                )
            else:
                input_tokens, output_tokens = _selected_mixture_shape(source, request.seed, index)
            if shared_prefix is None:
                fixed_messages: list[dict[str, str]] = []
                unadjusted_content_tokens = input_tokens
                content_label = "user-content"
            else:
                if not isinstance(source, BenchRequestSourceInputRandom):
                    raise TypeError("only random sources may declare prefix sharing")
                sharing = source.prefix_sharing
                if sharing is None:
                    raise ValueError("resolved prefix-sharing source omitted its split")
                fixed_messages = [{"role": "system", "content": shared_prefix}]
                unadjusted_content_tokens = sharing.unique_suffix_tokens
                content_label = "user-suffix"
            targeting = _target_synthetic_prompt(
                tokenizer,
                text_factory,
                projection,
                projection_reason,
                projection_detail,
                fixed_messages,
                input_tokens,
                unadjusted_content_tokens,
                request.seed,
                index,
                content_label,
            )
            if targeting.exact:
                exact_entries += 1
            else:
                reason = targeting.fallback_reason
                if reason is None:
                    raise ValueError("synthetic prompt targeting fallback omitted its reason")
                fallback_entries += 1
                fallback_reasons[reason] = fallback_reasons.get(reason, 0) + 1
            population_value: JsonObject = {
                "session_id": f"inferlab-{index:08}",
                "messages": targeting.messages,
                "output_length": output_tokens,
                "extra": {"ignore_eos": True, "min_tokens": output_tokens},
            }
            population_line = json_line(population_value)
            population_file.write(population_line)
            population_digest.update(population_line)
            evidence_file.write(
                json_line(
                    {
                        "population_index": index,
                        "source_sample_id": f"synthetic-{index:08}",
                        "messages": targeting.messages,
                        "selected_prompt_tokens": input_tokens,
                        "selected_output_tokens": output_tokens,
                        "pre_template_content_tokens": (targeting.pre_template_content_tokens),
                        "locally_predicted_prompt_tokens": (
                            targeting.locally_predicted_prompt_tokens
                        ),
                        "prompt_token_targeting": ("exact" if targeting.exact else "fallback"),
                        "prompt_token_fallback_reason": targeting.fallback_reason,
                        "prompt_token_fallback_detail": targeting.fallback_detail,
                    }
                )
            )
            selected_prompt_counts.append(input_tokens)
            pre_template_content_counts.append(targeting.pre_template_content_tokens)
            output_counts.append(output_tokens)
    return BenchPopulationPreparationResult(
        schema_version=1,
        status=ClientStatus.succeeded,
        materialization_identity=SYNTHETIC_MATERIALIZATION_IDENTITY,
        requested_entries=required,
        candidate_entries=required,
        admitted_entries=required,
        ineligible_entries=0,
        ineligible_reasons={},
        population=BenchPopulationInput(
            path=str(population_path),
            sha256=population_digest.hexdigest(),
            entries=required,
            tpot_applicable=all(value >= 2 for value in output_counts),
        ),
        input_tokens=count_summary(selected_prompt_counts),
        output_tokens=count_summary(output_counts),
        prompt_token_targeting=BenchPromptTokenTargetingSummary(
            selected_prompt_tokens=count_summary(selected_prompt_counts),
            pre_template_content_tokens=count_summary(pre_template_content_counts),
            projection_template=(projection.template if projection is not None else None),
            exact_entries=exact_entries,
            fallback_entries=fallback_entries,
            fallback_reasons=fallback_reasons,
        ),
        evidence_path=str(evidence_path),
        error=None,
    )
