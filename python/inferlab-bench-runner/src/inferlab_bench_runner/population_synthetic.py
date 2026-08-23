"""Materialize deterministic synthetic request populations."""

import hashlib
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from inferlab_measurement_sdk import (
    BenchCacheStartInput,
    BenchInclusiveUniformInput,
    BenchPopulationInput,
    BenchPopulationPreparationRequest,
    BenchPopulationPreparationResult,
    BenchPrefixConditioningInput,
    BenchPrefixGeometrySummary,
    BenchPrefixSharingInput1,
    BenchPrefixSharingInput2,
    BenchPromptInputFlat,
    BenchPromptInputRenderedChat,
    BenchPromptInputServerChat,
    BenchPromptTemplateProjection,
    BenchPromptTemplateSource,
    BenchPromptTokenTargetingSummary,
    BenchRequestSourceInputRandom,
    BenchRequestSourceInputRandomMixture,
    BenchSharedSystemContentInput1,
    BenchSharedSystemContentInput2,
    BenchSharedSystemContentSummary,
    BenchTokenSelectorInput,
    BenchTokenSelectorInput1,
    ClientStatus,
    JsonObject,
    plain_setting,
)

from inferlab_bench_runner.chat_tokens import required_messages_content_tokens
from inferlab_bench_runner.population_types import (
    ChatTokenizer,
    common_prefix_length,
    count_summary,
    decode_exact,
    json_line,
    token_stream_digest,
    unbiased_index,
)

SYNTHETIC_MATERIALIZATION_IDENTITY = "inferlab-synthetic-prompt-authority-v4"
CORPUS_MATERIALIZATION_IDENTITY = "inferlab-corpus-slice-v1"
MAX_EXACT_TARGETING_ATTEMPTS = 32
MAX_EXACT_CONTENT_VARIANTS = 16


@dataclass(frozen=True)
class CorpusSlice:
    text: str
    offset: int
    length: int


class CorpusTextFactory:
    """Draw exact-length slices from one operator-supplied corpus token stream.

    The corpus replaces only the hash-word supply of the random source: every
    entry takes exactly its selected input-token target as one slice whose
    offset is determined by the Bench seed and the entry's population index
    alone, under the same round-trip verification as synthetic prompts
    ([[RFC-0004:C-BENCH-REQUEST-SOURCES]]). Slices are drawn independently and
    may overlap; that incidental sharing is natural reuse, never measured.
    """

    def __init__(self, tokenizer: ChatTokenizer, corpus_ids: list[int]) -> None:
        self.tokenizer = tokenizer
        self.corpus_ids = corpus_ids

    def exact_slice(
        self,
        target_tokens: int,
        seed: int,
        population_index: int,
        label: str,
    ) -> CorpusSlice:
        starts = len(self.corpus_ids) - target_tokens + 1
        if starts < 1:
            raise ValueError(
                f"corpus token stream has {len(self.corpus_ids)} tokens, shorter than "
                f"the requested {target_tokens}-token slice"
            )
        first = unbiased_index(seed, population_index, f"{label}-offset", starts)
        for attempt in range(min(starts, 128)):
            start = (first + attempt) % starts
            text = self.tokenizer.decode(
                self.corpus_ids[start : start + target_tokens],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            if len(self.tokenizer.encode(text, add_special_tokens=False)) == target_tokens:
                return CorpusSlice(text=text, offset=start, length=target_tokens)
        raise ValueError(
            f"tokenizer could not round-trip a corpus {label} of {target_tokens} tokens"
        )


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
    transport_prompt: str | None
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
    prompt = request.prompt.root
    if isinstance(prompt, BenchPromptInputFlat):
        return None, None, None
    request_body = {key: plain_setting(value) for key, value in request.request_body.items()}
    requested_template: str | None = None
    template_source = BenchPromptTemplateSource.tokenizer_default
    template_kwargs: dict[str, object]
    if isinstance(prompt, BenchPromptInputRenderedChat):
        requested_template = prompt.chat_template
        template_kwargs = {
            key: plain_setting(value) for key, value in prompt.chat_template_kwargs.items()
        }
        if requested_template is not None:
            template_source = BenchPromptTemplateSource.prompt_table
    elif isinstance(prompt, BenchPromptInputServerChat) and "chat_template" in request_body:
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
    else:
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


def _render_prompt(
    tokenizer: ChatTokenizer,
    messages: list[dict[str, str]],
    projection: PromptProjection,
) -> str:
    candidate = getattr(tokenizer, "apply_chat_template", None)
    if not callable(candidate):
        raise ValueError("resolved tokenizer does not expose apply_chat_template")
    apply_chat_template = cast(Callable[..., object], candidate)
    kwargs = dict(projection.chat_template_kwargs)
    kwargs["chat_template"] = projection.template.content
    rendered = apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        **kwargs,
    )
    if not isinstance(rendered, str):
        raise ValueError("resolved tokenizer returned a non-string rendered prompt")
    return rendered


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
                        transport_prompt=None,
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
            transport_prompt=None,
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
            transport_prompt=None,
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
            transport_prompt=None,
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
                transport_prompt=None,
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
        transport_prompt=None,
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


def _maximum_selected_input(
    source: BenchRequestSourceInputRandom | BenchRequestSourceInputRandomMixture,
) -> int:
    if isinstance(source, BenchRequestSourceInputRandom):
        selector = source.input_tokens.root
        if isinstance(selector, BenchTokenSelectorInput1):
            return selector.root
        if isinstance(selector, BenchInclusiveUniformInput):
            return selector.max
        raise TypeError(f"unsupported token selector {type(selector).__name__}")
    return max(shape.input_tokens for shape in source.shapes)


def _resolved_prefix_tokens(
    source: BenchRequestSourceInputRandom | BenchRequestSourceInputRandomMixture,
    input_tokens: int,
) -> int | None:
    sharing = source.prefix_sharing
    if sharing is None:
        return None
    value = sharing.root
    if isinstance(value, BenchPrefixSharingInput1):
        return value.shared_prefix_tokens
    if isinstance(value, BenchPrefixSharingInput2):
        return math.floor(input_tokens * value.shared_prefix_ratio)
    raise TypeError(f"unsupported prefix sharing {type(value).__name__}")


def _resolved_system_tokens(source: BenchRequestSourceInputRandom, input_tokens: int) -> int | None:
    sharing = source.shared_system_content
    if sharing is None:
        return None
    value = sharing.root
    if isinstance(value, BenchSharedSystemContentInput1):
        return value.tokens
    if isinstance(value, BenchSharedSystemContentInput2):
        return math.floor(input_tokens * value.ratio)
    raise TypeError(f"unsupported shared system content {type(value).__name__}")


def _flat_prompt(
    tokenizer: ChatTokenizer,
    text_factory: SyntheticTextFactory,
    canonical_ids: list[int] | None,
    input_tokens: int,
    shared_prefix_tokens: int | None,
    seed: int,
    population_index: int,
) -> str:
    if shared_prefix_tokens is None or shared_prefix_tokens == 0:
        return text_factory.exact_text(input_tokens, seed, population_index, "flat-unique-prompt")
    if canonical_ids is None:
        raise ValueError("flat prefix geometry omitted its canonical token stream")
    if shared_prefix_tokens == input_tokens:
        return decode_exact(
            tokenizer,
            canonical_ids[:input_tokens],
            "full shared prompt",
        )
    for variant in range(MAX_EXACT_CONTENT_VARIANTS):
        suffix = text_factory.exact_text(
            input_tokens - shared_prefix_tokens,
            seed,
            population_index,
            f"flat-unique-suffix-{variant}",
        )
        suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
        candidate_ids = canonical_ids[:shared_prefix_tokens] + suffix_ids
        try:
            return decode_exact(tokenizer, candidate_ids, "flat prompt")
        except ValueError:
            continue
    raise ValueError(
        f"tokenizer could not construct an exact flat prompt with {shared_prefix_tokens} "
        f"shared tokens and {input_tokens - shared_prefix_tokens} unique tokens"
    )


def _corpus_flat_prompt(
    tokenizer: ChatTokenizer,
    corpus_factory: CorpusTextFactory,
    canonical_ids: list[int] | None,
    canonical_offset: int | None,
    input_tokens: int,
    shared_prefix_tokens: int | None,
    seed: int,
    population_index: int,
) -> tuple[str, CorpusSlice]:
    """One corpus-backed flat prompt and the slice evidence for its entry.

    Without declared prefix geometry the whole entry is one independent
    slice. With it, the shared prefix is one fixed corpus slice and each
    unique suffix is drawn independently.
    """
    if shared_prefix_tokens is None or shared_prefix_tokens == 0:
        slice_ = corpus_factory.exact_slice(
            input_tokens, seed, population_index, "corpus-unique-prompt"
        )
        return slice_.text, slice_
    if canonical_ids is None or canonical_offset is None:
        raise ValueError("corpus prefix geometry omitted its canonical slice")
    if shared_prefix_tokens == input_tokens:
        text = decode_exact(
            tokenizer,
            canonical_ids[:input_tokens],
            "full shared prompt",
        )
        return text, CorpusSlice(text=text, offset=canonical_offset, length=input_tokens)
    for variant in range(MAX_EXACT_CONTENT_VARIANTS):
        suffix = corpus_factory.exact_slice(
            input_tokens - shared_prefix_tokens,
            seed,
            population_index,
            f"corpus-unique-suffix-{variant}",
        )
        suffix_ids = tokenizer.encode(suffix.text, add_special_tokens=False)
        candidate_ids = canonical_ids[:shared_prefix_tokens] + suffix_ids
        try:
            return decode_exact(tokenizer, candidate_ids, "flat prompt"), suffix
        except ValueError:
            continue
    raise ValueError(
        f"tokenizer could not construct an exact corpus flat prompt with "
        f"{shared_prefix_tokens} shared tokens and "
        f"{input_tokens - shared_prefix_tokens} unique tokens"
    )


def _rendered_prompt_without_geometry(
    tokenizer: ChatTokenizer,
    text_factory: SyntheticTextFactory,
    projection: PromptProjection,
    input_tokens: int,
    seed: int,
    population_index: int,
    label: str,
) -> SyntheticPromptTargeting:
    targeting = _target_synthetic_prompt(
        tokenizer,
        text_factory,
        projection,
        None,
        None,
        [],
        input_tokens,
        input_tokens,
        seed,
        population_index,
        label,
    )
    if not targeting.exact:
        raise ValueError(
            targeting.fallback_detail
            or targeting.fallback_reason
            or "exact rendered-chat prompt construction failed"
        )
    rendered = _render_prompt(tokenizer, targeting.messages, projection)
    if len(tokenizer.encode(rendered, add_special_tokens=False)) != input_tokens:
        raise ValueError("rendered-chat prompt did not preserve its exact selected token target")
    return SyntheticPromptTargeting(
        messages=targeting.messages,
        transport_prompt=rendered,
        pre_template_content_tokens=targeting.pre_template_content_tokens,
        locally_predicted_prompt_tokens=input_tokens,
        exact=True,
        fallback_reason=None,
        fallback_detail=None,
    )


def _rendered_prompt_with_geometry(
    tokenizer: ChatTokenizer,
    text_factory: SyntheticTextFactory,
    projection: PromptProjection,
    canonical_targeting: SyntheticPromptTargeting,
    canonical_ids: list[int],
    input_tokens: int,
    shared_prefix_tokens: int,
    seed: int,
    population_index: int,
) -> SyntheticPromptTargeting:
    if shared_prefix_tokens == input_tokens:
        if input_tokens != len(canonical_ids):
            raise ValueError(
                "rendered-chat full-prefix geometry is incompatible with variable final "
                "prompt lengths"
            )
        return canonical_targeting

    independent = _rendered_prompt_without_geometry(
        tokenizer,
        text_factory,
        projection,
        input_tokens,
        seed,
        population_index,
        "rendered-independent-probe",
    )
    independent_prompt = independent.transport_prompt
    canonical_prompt = canonical_targeting.transport_prompt
    if independent_prompt is None or canonical_prompt is None:
        raise ValueError("rendered-chat geometry omitted a final transport prompt")
    independent_ids = tokenizer.encode(independent_prompt, add_special_tokens=False)
    invariant_prefix = common_prefix_length(canonical_ids, independent_ids)
    if shared_prefix_tokens < invariant_prefix:
        raise ValueError(
            f"rendered-chat template contributes {invariant_prefix} invariant prefix tokens, "
            f"which is incompatible with requested prefix length {shared_prefix_tokens}"
        )
    if shared_prefix_tokens == 0:
        return independent

    canonical_content = canonical_targeting.messages[-1]["content"]
    canonical_content_ids = tokenizer.encode(canonical_content, add_special_tokens=False)
    empty_rendered = _render_prompt(
        tokenizer,
        [{"role": "user", "content": ""}],
        projection,
    )
    estimated_content_tokens = max(
        0,
        input_tokens - len(tokenizer.encode(empty_rendered, add_special_tokens=False)),
    )
    estimated_shared_content = max(0, shared_prefix_tokens - invariant_prefix)
    shared_candidates = sorted(
        {
            value
            for delta in range(-8, 9)
            if 0 <= (value := estimated_shared_content + delta) <= len(canonical_content_ids)
        }
    )
    content_candidates = sorted(
        {value for delta in range(-16, 17) if (value := estimated_content_tokens + delta) >= 0}
    )
    for shared_content_tokens in shared_candidates:
        try:
            shared_text = decode_exact(
                tokenizer,
                canonical_content_ids[:shared_content_tokens],
                "rendered shared content",
            )
        except ValueError:
            continue
        for content_tokens in content_candidates:
            unique_tokens = content_tokens - shared_content_tokens
            if unique_tokens < 0:
                continue
            for variant in range(4):
                unique_text = text_factory.exact_text(
                    unique_tokens,
                    seed,
                    population_index,
                    f"rendered-unique-suffix-{variant}",
                )
                separator = " " if shared_text and unique_text else ""
                messages = [{"role": "user", "content": shared_text + separator + unique_text}]
                rendered = _render_prompt(tokenizer, messages, projection)
                rendered_ids = tokenizer.encode(rendered, add_special_tokens=False)
                if len(rendered_ids) != input_tokens:
                    continue
                if rendered_ids[:shared_prefix_tokens] != canonical_ids[:shared_prefix_tokens]:
                    continue
                return SyntheticPromptTargeting(
                    messages=messages,
                    transport_prompt=rendered,
                    pre_template_content_tokens=required_messages_content_tokens(
                        messages, tokenizer
                    ),
                    locally_predicted_prompt_tokens=input_tokens,
                    exact=True,
                    fallback_reason=None,
                    fallback_detail=None,
                )
    raise ValueError(
        f"rendered-chat prompt cannot satisfy exact target {input_tokens} and shared prefix "
        f"length {shared_prefix_tokens} under the frozen template"
    )


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
    selected_shapes: list[tuple[int, int]] = []
    for index in range(required):
        if isinstance(source, BenchRequestSourceInputRandom):
            selected_shapes.append(
                (
                    selected_tokens(source.input_tokens, request.seed, index, "input_tokens"),
                    selected_tokens(source.output_tokens, request.seed, index, "output_tokens"),
                )
            )
        else:
            selected_shapes.append(_selected_mixture_shape(source, request.seed, index))
    selected_prompt_counts = [shape[0] for shape in selected_shapes]
    output_counts = [shape[1] for shape in selected_shapes]
    resolved_prefix_counts = [
        _resolved_prefix_tokens(source, input_tokens) for input_tokens in selected_prompt_counts
    ]
    resolved_system_counts = [
        _resolved_system_tokens(source, input_tokens)
        if isinstance(source, BenchRequestSourceInputRandom)
        else None
        for input_tokens in selected_prompt_counts
    ]
    corpus_factory: CorpusTextFactory | None = None
    if isinstance(source, BenchRequestSourceInputRandom) and source.corpus is not None:
        # The corpus replaces only the content supply: tokenize it once with
        # the resolved model tokenizer, then serve exact-length slices
        # ([[RFC-0004:C-BENCH-REQUEST-SOURCES]]).
        corpus = source.corpus
        if not isinstance(request.prompt.root, BenchPromptInputFlat):
            raise ValueError("a corpus request source requires prompt kind flat")
        if request.source_path is None:
            raise ValueError("corpus preparation requires a source path")
        corpus_bytes = Path(request.source_path).read_bytes()
        if corpus.expected_sha256 is not None:
            observed_sha256 = hashlib.sha256(corpus_bytes).hexdigest()
            if observed_sha256 != corpus.expected_sha256:
                raise ValueError(
                    "corpus SHA-256 does not match the declared expected digest: "
                    f"expected {corpus.expected_sha256}, observed {observed_sha256}"
                )
        corpus_ids = tokenizer.encode(corpus_bytes.decode("utf-8"), add_special_tokens=False)
        largest_selected = max(selected_prompt_counts)
        if len(corpus_ids) < largest_selected:
            raise ValueError(
                f"corpus token stream has {len(corpus_ids)} tokens, shorter than the "
                f"largest selected input-token target {largest_selected}"
            )
        corpus_factory = CorpusTextFactory(tokenizer, corpus_ids)

    prompt = request.prompt.root
    projection, projection_reason, projection_detail = _resolve_prompt_projection(
        request, tokenizer
    )
    exact_entries = 0
    fallback_entries = 0
    fallback_reasons: dict[str, int] = {}
    text_factory = SyntheticTextFactory(tokenizer)
    canonical_ids: list[int] | None = None
    canonical_offset: int | None = None
    canonical_targeting: SyntheticPromptTargeting | None = None
    maximum_possible_input = _maximum_selected_input(source)
    maximum_prefix_tokens = _resolved_prefix_tokens(source, maximum_possible_input)
    if maximum_prefix_tokens is not None:
        if isinstance(prompt, BenchPromptInputFlat):
            if corpus_factory is not None:
                # The shared prefix is one fixed corpus slice at a
                # seed-determined offset, independent of the population size.
                if len(corpus_factory.corpus_ids) < maximum_prefix_tokens:
                    raise ValueError(
                        f"corpus token stream has {len(corpus_factory.corpus_ids)} tokens, "
                        f"shorter than the resolved maximum shared prefix "
                        f"{maximum_prefix_tokens}"
                    )
                canonical_offset = unbiased_index(
                    request.seed,
                    0,
                    "corpus-canonical-slice",
                    len(corpus_factory.corpus_ids) - maximum_prefix_tokens + 1,
                )
                canonical_ids = corpus_factory.corpus_ids[
                    canonical_offset : canonical_offset + maximum_prefix_tokens
                ]
            else:
                canonical_text = text_factory.exact_text(
                    maximum_possible_input,
                    request.seed,
                    0,
                    "canonical-shared-stream",
                )
                canonical_ids = tokenizer.encode(canonical_text, add_special_tokens=False)
        elif isinstance(prompt, BenchPromptInputRenderedChat):
            if projection is None:
                raise ValueError(
                    projection_detail
                    or projection_reason
                    or "rendered-chat prompt has no usable local template"
                )
            canonical_targeting = _rendered_prompt_without_geometry(
                tokenizer,
                text_factory,
                projection,
                maximum_possible_input,
                request.seed,
                0,
                "canonical-rendered-prompt",
            )
            if canonical_targeting.transport_prompt is None:
                raise ValueError("rendered-chat canonical prompt was not frozen")
            canonical_ids = tokenizer.encode(
                canonical_targeting.transport_prompt, add_special_tokens=False
            )
        else:
            raise ValueError("server-chat prompt cannot declare exact prefix geometry")
    canonical_system_text: str | None = None
    maximum_system_tokens: int | None = None
    if (
        isinstance(source, BenchRequestSourceInputRandom)
        and source.shared_system_content is not None
    ):
        maximum_system_tokens = _resolved_system_tokens(source, maximum_possible_input)
        if maximum_system_tokens is None:
            raise ValueError("shared system content omitted its resolved maximum length")
        canonical_system_text = text_factory.exact_text(
            maximum_system_tokens,
            request.seed,
            0,
            "canonical-system-content",
        )

    prefix_conditioning: BenchPrefixConditioningInput | None = None
    if (
        request.cache_start is BenchCacheStartInput.primed
        and maximum_prefix_tokens is not None
        and maximum_prefix_tokens > 0
    ):
        if canonical_ids is None:
            raise ValueError("canonical prefix token stream was not frozen")
        canonical_prefix = decode_exact(
            tokenizer,
            canonical_ids[:maximum_prefix_tokens],
            "canonical prefix conditioning prompt",
        )
        conditioning_path = artifact_dir / "canonical-prefix.txt"
        conditioning_path.write_text(canonical_prefix, encoding="utf-8")
        prefix_conditioning = BenchPrefixConditioningInput(
            path=str(conditioning_path),
            sha256=hashlib.sha256(canonical_prefix.encode()).hexdigest(),
            prompt_tokens=maximum_prefix_tokens,
        )

    pre_template_content_counts: list[int] = []
    with population_path.open("wb") as population_file, evidence_path.open("wb") as evidence_file:
        for index, (input_tokens, output_tokens) in enumerate(selected_shapes):
            shared_prefix_tokens = resolved_prefix_counts[index]
            system_content_tokens = resolved_system_counts[index]
            corpus_slice: CorpusSlice | None = None
            if isinstance(prompt, BenchPromptInputFlat):
                if corpus_factory is not None:
                    transport_prompt, corpus_slice = _corpus_flat_prompt(
                        tokenizer,
                        corpus_factory,
                        canonical_ids,
                        canonical_offset,
                        input_tokens,
                        shared_prefix_tokens,
                        request.seed,
                        index,
                    )
                else:
                    transport_prompt = _flat_prompt(
                        tokenizer,
                        text_factory,
                        canonical_ids,
                        input_tokens,
                        shared_prefix_tokens,
                        request.seed,
                        index,
                    )
                targeting = SyntheticPromptTargeting(
                    messages=[],
                    transport_prompt=transport_prompt,
                    pre_template_content_tokens=input_tokens,
                    locally_predicted_prompt_tokens=input_tokens,
                    exact=True,
                    fallback_reason=None,
                    fallback_detail=None,
                )
            elif isinstance(prompt, BenchPromptInputRenderedChat):
                if projection is None:
                    raise ValueError(
                        projection_detail
                        or projection_reason
                        or "rendered-chat prompt has no usable local template"
                    )
                if shared_prefix_tokens is None:
                    targeting = _rendered_prompt_without_geometry(
                        tokenizer,
                        text_factory,
                        projection,
                        input_tokens,
                        request.seed,
                        index,
                        "rendered-user-content",
                    )
                else:
                    if canonical_targeting is None or canonical_ids is None:
                        raise ValueError(
                            "rendered-chat prefix geometry omitted its canonical prompt"
                        )
                    targeting = _rendered_prompt_with_geometry(
                        tokenizer,
                        text_factory,
                        projection,
                        canonical_targeting,
                        canonical_ids,
                        input_tokens,
                        shared_prefix_tokens,
                        request.seed,
                        index,
                    )
            elif system_content_tokens is None:
                fixed_messages: list[dict[str, str]] = []
                unadjusted_content_tokens = input_tokens
                content_label = "user-content"
            else:
                if canonical_system_text is None or maximum_system_tokens is None:
                    raise ValueError("shared system content omitted its canonical stream")
                system_ids = tokenizer.encode(canonical_system_text, add_special_tokens=False)
                system_text = decode_exact(
                    tokenizer,
                    system_ids[:system_content_tokens],
                    "shared system content",
                )
                fixed_messages = [{"role": "system", "content": system_text}]
                unadjusted_content_tokens = input_tokens - system_content_tokens
                content_label = "user-suffix"
            if isinstance(prompt, BenchPromptInputServerChat):
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
                "output_length": output_tokens,
                "extra": {"ignore_eos": True, "min_tokens": output_tokens},
            }
            if targeting.transport_prompt is None:
                population_value["messages"] = targeting.messages
            else:
                population_value["text_input"] = targeting.transport_prompt
            population_line = json_line(population_value)
            population_file.write(population_line)
            population_digest.update(population_line)
            evidence_file.write(
                json_line(
                    {
                        "population_index": index,
                        "source_sample_id": f"synthetic-{index:08}",
                        "messages": targeting.messages,
                        "transport_prompt_sha256": (
                            hashlib.sha256(targeting.transport_prompt.encode()).hexdigest()
                            if targeting.transport_prompt is not None
                            else None
                        ),
                        "request_representation": prompt.request_representation.value,
                        "route": prompt.route.value,
                        "rendering_authority": prompt.rendering_authority.value,
                        "prompt_kind": prompt.kind,
                        "selected_prompt_tokens": input_tokens,
                        "selected_output_tokens": output_tokens,
                        "pre_template_content_tokens": (targeting.pre_template_content_tokens),
                        "locally_predicted_prompt_tokens": (
                            targeting.locally_predicted_prompt_tokens
                        ),
                        "prompt_token_targeting": ("exact" if targeting.exact else "fallback"),
                        "prompt_token_fallback_reason": targeting.fallback_reason,
                        "prompt_token_fallback_detail": targeting.fallback_detail,
                        "resolved_shared_prefix_tokens": shared_prefix_tokens,
                        "resolved_unique_suffix_tokens": (
                            input_tokens - shared_prefix_tokens
                            if shared_prefix_tokens is not None
                            else None
                        ),
                        "resolved_system_content_tokens": system_content_tokens,
                        "resolved_user_content_tokens": (
                            input_tokens - system_content_tokens
                            if system_content_tokens is not None
                            else None
                        ),
                        "corpus_slice_offset": (
                            corpus_slice.offset if corpus_slice is not None else None
                        ),
                        "corpus_slice_length": (
                            corpus_slice.length if corpus_slice is not None else None
                        ),
                        "corpus_shared_slice_offset": (
                            canonical_offset if shared_prefix_tokens else None
                        ),
                    }
                )
            )
            pre_template_content_counts.append(targeting.pre_template_content_tokens)
    prefix_counts = [value for value in resolved_prefix_counts if value is not None]
    system_counts = [value for value in resolved_system_counts if value is not None]
    return BenchPopulationPreparationResult(
        schema_version=1,
        status=ClientStatus.succeeded,
        materialization_identity=(
            CORPUS_MATERIALIZATION_IDENTITY
            if corpus_factory is not None
            else SYNTHETIC_MATERIALIZATION_IDENTITY
        ),
        requested_entries=required,
        candidate_entries=required,
        admitted_entries=required,
        ineligible_entries=0,
        ineligible_reasons={},
        population=BenchPopulationInput(
            path=str(population_path),
            evidence_path=str(evidence_path),
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
        prefix_geometry=(
            BenchPrefixGeometrySummary(
                shared_prefix_tokens=count_summary(prefix_counts),
                unique_suffix_tokens=count_summary(
                    [
                        input_tokens - prefix_tokens
                        for input_tokens, prefix_tokens in zip(
                            selected_prompt_counts, prefix_counts, strict=True
                        )
                    ]
                ),
                maximum_shared_prefix_tokens=max(prefix_counts),
                canonical_prefix_sha256=token_stream_digest(
                    (canonical_ids or [])[: max(prefix_counts)]
                ),
                full_prompt_entries=sum(
                    input_tokens == prefix_tokens
                    for input_tokens, prefix_tokens in zip(
                        selected_prompt_counts, prefix_counts, strict=True
                    )
                ),
            )
            if prefix_counts
            else None
        ),
        prefix_conditioning=prefix_conditioning,
        shared_system_content=(
            BenchSharedSystemContentSummary(
                system_content_tokens=count_summary(system_counts),
                user_content_tokens=count_summary(
                    [
                        input_tokens - system_tokens
                        for input_tokens, system_tokens in zip(
                            selected_prompt_counts, system_counts, strict=True
                        )
                    ]
                ),
                canonical_system_content_sha256=hashlib.sha256(
                    (canonical_system_text or "").encode()
                ).hexdigest(),
            )
            if system_counts
            else None
        ),
        evidence_path=str(evidence_path),
        error=None,
    )
