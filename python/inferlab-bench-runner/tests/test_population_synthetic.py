import hashlib
import json
import math
from pathlib import Path
from typing import cast

import pytest
from inferlab_bench_runner.aiperf import (
    aiperf_config,
    inference_request_config,
)
from inferlab_bench_runner.population import prepare_population
from inferlab_bench_runner.population_synthetic import token_count
from inferlab_bench_runner.result_population import (
    prompt_token_reconciliation,
)
from inferlab_measurement_sdk import (
    BenchPopulationPreparationRequest,
    ClientStatus,
)

from .support import (
    FakeTokenizer,
    request,
    resolved_prompt_input,
)


class ExactChatTokenizer(FakeTokenizer):
    def get_chat_template(self, chat_template: str | None = None, tools: object = None) -> str:
        assert chat_template == "{{ messages }}"
        assert tools is None
        return chat_template

    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        chat_template: str | None = None,
        **kwargs: object,
    ) -> list[int] | str:
        assert add_generation_prompt is True
        assert chat_template == "{{ messages }}"
        assert kwargs == {"enable_thinking": True}
        contents = " ".join(message["content"] for message in conversation)
        if not tokenize:
            return f"frame0 frame1 {contents} generation"
        content_tokens = sum(len(message["content"].split()) for message in conversation)
        return list(range(content_tokens + 3))


class DefaultChatTokenizer(FakeTokenizer):
    def get_chat_template(self, chat_template: str | None = None, tools: object = None) -> str:
        assert chat_template is None
        assert tools is None
        return "{{ default_messages }}"

    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        chat_template: str | None = None,
        **kwargs: object,
    ) -> list[int] | str:
        assert add_generation_prompt is True
        assert chat_template == "{{ default_messages }}"
        assert kwargs == {}
        contents = " ".join(message["content"] for message in conversation)
        if not tokenize:
            return f"frame {contents} generation"
        content_tokens = sum(len(message["content"].split()) for message in conversation)
        return list(range(content_tokens + 2))


class UnreachableChatTokenizer(FakeTokenizer):
    def get_chat_template(self, chat_template: str | None = None, tools: object = None) -> str:
        assert chat_template is None
        assert tools is None
        return "{{ unreachable }}"

    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        chat_template: str | None = None,
        **kwargs: object,
    ) -> list[int] | str:
        assert add_generation_prompt is True
        assert chat_template == "{{ unreachable }}"
        assert kwargs == {}
        contents = " ".join(message["content"] for message in conversation)
        if not tokenize:
            doubled = " ".join(word for word in contents.split() for _ in range(2))
            return f"frame {doubled} generation"
        content_tokens = sum(len(message["content"].split()) for message in conversation)
        return list(range(content_tokens * 2 + 2))


class NonRoundTripTokenizer(FakeTokenizer):
    def decode(self, token_ids: list[int], **kwargs: object) -> str:
        assert token_ids
        assert kwargs == {
            "skip_special_tokens": True,
            "clean_up_tokenization_spaces": False,
        }
        return ""


class PeriodicCorpusTokenizer:
    """Model the short token period that exposed the 0.8.0 fallback regression."""

    _synthetic_corpus = (
        "Reproducible inference measurements need stable prompts, explicit evidence, "
        "and independently selected request shapes. "
    )
    _decoded_prefix = "synthetic_token_"

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert not add_special_tokens
        words = text.split()
        if words and all(word.startswith(self._decoded_prefix) for word in words):
            return [int(word.removeprefix(self._decoded_prefix)) for word in words]
        if text and len(text) % len(self._synthetic_corpus) == 0:
            repetitions = len(text) // len(self._synthetic_corpus)
            if text == self._synthetic_corpus * repetitions:
                return list(range(20)) * repetitions
        return [int.from_bytes(hashlib.sha256(word.encode()).digest()[:8], "big") for word in words]

    def decode(self, token_ids: list[int], **kwargs: object) -> str:
        assert kwargs == {"skip_special_tokens": True, "clean_up_tokenization_spaces": False}
        return " ".join(f"{self._decoded_prefix}{token_id}" for token_id in token_ids)


class ExactIdentityChatTokenizer(PeriodicCorpusTokenizer):
    def get_chat_template(self, chat_template: str | None = None, tools: object = None) -> str:
        assert chat_template == "{{ messages }}"
        assert tools is None
        return chat_template

    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        chat_template: str | None = None,
        **kwargs: object,
    ) -> list[int] | str:
        assert add_generation_prompt is True
        assert chat_template == "{{ messages }}"
        assert kwargs == {"enable_thinking": True}
        contents = " ".join(message["content"] for message in conversation)
        rendered = f"frame0 frame1 {contents} generation"
        return self.encode(rendered, add_special_tokens=False) if tokenize else rendered


class NonRoundTripTemplatePrefixTokenizer(ExactIdentityChatTokenizer):
    """A complete rendered prompt is exact even when its frame prefix is not."""

    def get_chat_template(self, chat_template: str | None = None, tools: object = None) -> str:
        assert chat_template is None
        assert tools is None
        return "{{ default_messages }}"

    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        chat_template: str | None = None,
        **kwargs: object,
    ) -> list[int] | str:
        assert add_generation_prompt is True
        assert chat_template == "{{ default_messages }}"
        assert kwargs == {}
        contents = " ".join(message["content"] for message in conversation)
        rendered = f"frame0 frame1 {contents} generation"
        return self.encode(rendered, add_special_tokens=False) if tokenize else rendered

    def decode(self, token_ids: list[int], **kwargs: object) -> str:
        frame_token = self.encode("frame0", add_special_tokens=False)[0]
        if token_ids and token_ids[0] == frame_token:
            return ""
        return super().decode(token_ids, **kwargs)


def test_token_count_accepts_transformers_batch_encoding_shape() -> None:
    assert token_count({"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}) == 3


def random_preparation_request(
    tmp_path: Path,
    required_entries: int,
    *,
    request_source: dict[str, object],
    artifact_name: str = "population",
    request_body: dict[str, object] | None = None,
    seed: int = 7,
    cache_start: str = "uncontrolled",
) -> BenchPopulationPreparationRequest:
    effective_source = dict(request_source)
    prompt = effective_source.pop("prompt", {"kind": "server_chat"})
    if not isinstance(prompt, dict):
        raise ValueError("test prompt must be an object")
    effective_prompt = resolved_prompt_input(prompt)
    return BenchPopulationPreparationRequest.model_validate(
        {
            "protocol_version": "9",
            "model": {"locator": "/models/dsv4", "served_name": "dsv4"},
            "tokenizer_backend": "huggingface",
            "transformers_version": "5.12.1",
            "request_source": effective_source,
            "prompt": effective_prompt,
            "cache_start": cache_start,
            "source_path": None,
            "required_entries": required_entries,
            "seed": seed,
            "request_body": request_body or {},
            "artifact_dir": str(tmp_path / artifact_name),
        }
    )


def test_uniform_selectors_freeze_a_prefix_stable_population(tmp_path: Path) -> None:
    source: dict[str, object] = {
        "kind": "random",
        "input_tokens": {"kind": "inclusive_uniform", "min": 7, "max": 11},
        "output_tokens": {"kind": "inclusive_uniform", "min": 3, "max": 5},
        "prefix_sharing": None,
    }
    first = prepare_population(
        random_preparation_request(tmp_path, 4, request_source=source, artifact_name="first"),
        FakeTokenizer(),
    )
    larger = prepare_population(
        random_preparation_request(tmp_path, 8, request_source=source, artifact_name="larger"),
        FakeTokenizer(),
    )

    assert first.status == ClientStatus.succeeded
    assert larger.status == ClientStatus.succeeded
    assert first.evidence_path is not None
    assert larger.evidence_path is not None
    first_rows = [json.loads(line) for line in Path(first.evidence_path).read_text().splitlines()]
    larger_rows = [json.loads(line) for line in Path(larger.evidence_path).read_text().splitlines()]
    assert larger_rows[:4] == first_rows
    assert all(7 <= row["selected_prompt_tokens"] <= 11 for row in larger_rows)
    assert all(3 <= row["selected_output_tokens"] <= 5 for row in larger_rows)
    assert all(
        row["pre_template_content_tokens"] == row["selected_prompt_tokens"] for row in larger_rows
    )
    assert all(row["prompt_token_targeting"] == "fallback" for row in larger_rows)


def test_synthetic_population_preserves_structured_messages_and_configured_isl(
    tmp_path: Path,
) -> None:
    result = prepare_population(
        random_preparation_request(
            tmp_path,
            2,
            request_source={
                "kind": "random",
                "input_tokens": 8,
                "output_tokens": 4,
                "prefix_sharing": None,
            },
        ),
        FakeTokenizer(),
    )

    assert result.status == ClientStatus.succeeded
    assert result.population is not None
    assert result.evidence_path is not None
    population = [
        json.loads(line) for line in Path(result.population.path).read_text().splitlines()
    ]
    evidence = [json.loads(line) for line in Path(result.evidence_path).read_text().splitlines()]
    assert all("messages" in row and "text_input" not in row for row in population)
    assert all(row["selected_prompt_tokens"] == 8 for row in evidence)
    assert all(
        row["messages"] == population[index]["messages"] for index, row in enumerate(evidence)
    )
    assert all("rendered_prompt" not in row for row in evidence)


def test_synthetic_population_targets_the_complete_local_chat_projection(
    tmp_path: Path,
) -> None:
    result = prepare_population(
        random_preparation_request(
            tmp_path,
            2,
            request_source={
                "kind": "random",
                "input_tokens": 8,
                "output_tokens": 4,
                "prefix_sharing": None,
            },
            request_body={
                "chat_template": "{{ messages }}",
                "chat_template_kwargs": {"enable_thinking": True},
            },
        ),
        ExactChatTokenizer(),
    )

    assert result.status == ClientStatus.succeeded
    assert result.prompt_token_targeting is not None
    assert result.prompt_token_targeting.exact_entries == 2
    assert result.prompt_token_targeting.fallback_entries == 0
    assert result.prompt_token_targeting.fallback_reasons == {}
    assert result.prompt_token_targeting.selected_prompt_tokens.minimum == 8
    assert result.prompt_token_targeting.pre_template_content_tokens.minimum == 5
    assert result.prompt_token_targeting.projection_template is not None
    assert result.prompt_token_targeting.projection_template.source == "request_body"
    assert result.prompt_token_targeting.projection_template.content == "{{ messages }}"
    assert (
        result.prompt_token_targeting.projection_template.sha256
        == hashlib.sha256(b"{{ messages }}").hexdigest()
    )
    assert result.evidence_path is not None
    evidence = [json.loads(line) for line in Path(result.evidence_path).read_text().splitlines()]
    assert all(row["selected_prompt_tokens"] == 8 for row in evidence)
    assert all(row["pre_template_content_tokens"] == 5 for row in evidence)
    assert all(row["locally_predicted_prompt_tokens"] == 8 for row in evidence)
    assert all(row["prompt_token_targeting"] == "exact" for row in evidence)
    assert all(row["prompt_token_fallback_reason"] is None for row in evidence)


def test_synthetic_population_uses_the_tokenizer_default_chat_template(
    tmp_path: Path,
) -> None:
    result = prepare_population(
        random_preparation_request(
            tmp_path,
            1,
            request_source={
                "kind": "random",
                "input_tokens": 8,
                "output_tokens": 4,
                "prefix_sharing": None,
            },
        ),
        DefaultChatTokenizer(),
    )

    assert result.prompt_token_targeting is not None
    assert result.prompt_token_targeting.exact_entries == 1
    assert result.prompt_token_targeting.pre_template_content_tokens.minimum == 6
    assert result.prompt_token_targeting.projection_template is not None
    assert result.prompt_token_targeting.projection_template.source == "tokenizer_default"
    assert result.prompt_token_targeting.projection_template.content == "{{ default_messages }}"


def test_synthetic_population_records_unmodified_fallback_without_a_template_projection(
    tmp_path: Path,
) -> None:
    result = prepare_population(
        random_preparation_request(
            tmp_path,
            1,
            request_source={
                "kind": "random",
                "input_tokens": 8,
                "output_tokens": 4,
                "prefix_sharing": None,
            },
        ),
        FakeTokenizer(),
    )

    assert result.status == ClientStatus.succeeded
    assert result.prompt_token_targeting is not None
    assert result.prompt_token_targeting.exact_entries == 0
    assert result.prompt_token_targeting.fallback_entries == 1
    assert result.prompt_token_targeting.fallback_reasons == {
        "chat_template_resolution_unavailable": 1
    }
    assert result.prompt_token_targeting.projection_template is None
    assert result.evidence_path is not None
    evidence = json.loads(Path(result.evidence_path).read_text())
    assert evidence["selected_prompt_tokens"] == 8
    assert evidence["pre_template_content_tokens"] == 8
    assert evidence["locally_predicted_prompt_tokens"] is None
    assert evidence["prompt_token_targeting"] == "fallback"
    assert evidence["prompt_token_fallback_reason"] == "chat_template_resolution_unavailable"


@pytest.mark.parametrize(
    "request_source",
    [
        {
            "kind": "random",
            "input_tokens": 8192,
            "output_tokens": 1,
            "prefix_sharing": None,
        },
        {
            "kind": "random_mixture",
            "shapes": [
                {"input_tokens": 8192, "output_tokens": 1, "weight": 1},
            ],
            "total_weight": 1,
        },
    ],
    ids=["random", "random-mixture"],
)
def test_fallback_population_keeps_long_independent_prompts_distinct(
    tmp_path: Path,
    request_source: dict[str, object],
) -> None:
    result = prepare_population(
        random_preparation_request(
            tmp_path,
            4,
            request_source=request_source,
            seed=0,
        ),
        PeriodicCorpusTokenizer(),
    )

    assert result.population is not None
    population = [
        json.loads(line) for line in Path(result.population.path).read_text().splitlines()
    ]
    contents = [row["messages"][0]["content"] for row in population]
    assert len(set(contents)) == 4
    assert all(len(content.split()) == 8192 for content in contents)
    assert result.prompt_token_targeting is not None
    assert result.prompt_token_targeting.exact_entries == 0
    assert result.prompt_token_targeting.fallback_entries == 4
    assert result.prompt_token_targeting.fallback_reasons == {
        "chat_template_resolution_unavailable": 4
    }


def test_synthetic_population_keeps_unadjusted_content_when_exact_target_is_unreachable(
    tmp_path: Path,
) -> None:
    result = prepare_population(
        random_preparation_request(
            tmp_path,
            1,
            request_source={
                "kind": "random",
                "input_tokens": 9,
                "output_tokens": 4,
                "prefix_sharing": None,
            },
        ),
        UnreachableChatTokenizer(),
    )

    assert result.prompt_token_targeting is not None
    assert result.prompt_token_targeting.exact_entries == 0
    assert result.prompt_token_targeting.fallback_entries == 1
    assert result.prompt_token_targeting.fallback_reasons == {"exact_prompt_length_unreachable": 1}
    assert result.prompt_token_targeting.projection_template is not None
    assert result.evidence_path is not None
    evidence = json.loads(Path(result.evidence_path).read_text())
    assert evidence["pre_template_content_tokens"] == 9
    assert evidence["locally_predicted_prompt_tokens"] == 20
    assert evidence["prompt_token_targeting"] == "fallback"


def test_synthetic_population_fails_when_the_unadjusted_content_cannot_be_constructed(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        ValueError,
        match="could not round-trip a synthetic user-content of 9 tokens",
    ):
        prepare_population(
            random_preparation_request(
                tmp_path,
                1,
                request_source={
                    "kind": "random",
                    "input_tokens": 9,
                    "output_tokens": 4,
                    "prefix_sharing": None,
                },
            ),
            NonRoundTripTokenizer(),
        )


def test_server_chat_targeting_keeps_shared_system_content_and_adjusts_only_the_user_suffix(
    tmp_path: Path,
) -> None:
    result = prepare_population(
        random_preparation_request(
            tmp_path,
            2,
            request_source={
                "kind": "random",
                "input_tokens": 12,
                "output_tokens": 4,
                "prefix_sharing": None,
                "shared_system_content": {"ratio": 0.5},
            },
            request_body={
                "chat_template": "{{ messages }}",
                "chat_template_kwargs": {"enable_thinking": True},
            },
        ),
        ExactChatTokenizer(),
    )

    assert result.status == ClientStatus.succeeded
    assert result.population is not None
    population = [
        json.loads(line) for line in Path(result.population.path).read_text().splitlines()
    ]
    assert len({row["messages"][0]["content"] for row in population}) == 1
    assert all(len(row["messages"][0]["content"].split()) == 6 for row in population)
    assert all(len(row["messages"][1]["content"].split()) == 3 for row in population)
    assert len({row["messages"][1]["content"] for row in population}) == 2
    assert result.prompt_token_targeting is not None
    assert result.prompt_token_targeting.pre_template_content_tokens.minimum == 9
    assert result.prompt_token_targeting.exact_entries == 2


def test_flat_population_is_exact_and_uses_the_completions_route(tmp_path: Path) -> None:
    source: dict[str, object] = {
        "kind": "random",
        "prompt": {"kind": "flat"},
        "input_tokens": 8,
        "output_tokens": 4,
        "prefix_sharing": None,
    }
    result = prepare_population(
        random_preparation_request(tmp_path, 4, request_source=source),
        PeriodicCorpusTokenizer(),
    )

    assert result.population is not None
    assert result.prompt_token_targeting is not None
    assert result.prompt_token_targeting.exact_entries == 4
    assert result.prompt_token_targeting.fallback_entries == 0
    population = [
        json.loads(line) for line in Path(result.population.path).read_text().splitlines()
    ]
    assert all("text_input" in row and "messages" not in row for row in population)
    assert all(
        len(PeriodicCorpusTokenizer().encode(row["text_input"], add_special_tokens=False)) == 8
        for row in population
    )

    value = request(
        tmp_path / "case",
        {"kind": "concurrency_limited", "concurrency": 1},
        request_body={},
        request_source=source,
    ).model_copy(update={"population": result.population})
    benchmark = cast(dict[str, object], aiperf_config(value)["benchmark"])
    endpoint = cast(dict[str, object], benchmark["endpoint"])
    assert endpoint["path"] == "/v1/completions"
    assert endpoint["type"] == "completions"
    request_config = inference_request_config(value)
    assert request_config["selected_named_route"] == "completions_path"
    assert request_config["prompt_authority"] == {
        "kind": "flat",
        "request_representation": "flat_prompt",
        "route": "completions",
        "rendering_authority": "local_flat",
    }


@pytest.mark.parametrize(
    ("declaration", "expected_shared"),
    [
        ({"shared_prefix_tokens": 0}, 0),
        ({"shared_prefix_ratio": 1.0}, 8),
    ],
)
def test_flat_prefix_geometry_supports_zero_and_full_sharing(
    tmp_path: Path,
    declaration: dict[str, object],
    expected_shared: int,
) -> None:
    result = prepare_population(
        random_preparation_request(
            tmp_path,
            3,
            request_source={
                "kind": "random",
                "prompt": {"kind": "flat"},
                "input_tokens": 8,
                "output_tokens": 2,
                "prefix_sharing": declaration,
            },
        ),
        PeriodicCorpusTokenizer(),
    )

    assert result.population is not None
    assert result.prefix_geometry is not None
    assert result.prefix_geometry.shared_prefix_tokens.minimum == expected_shared
    rows = [json.loads(line) for line in Path(result.population.path).read_text().splitlines()]
    prompts = [row["text_input"] for row in rows]
    assert (len(set(prompts)) == 1) is (expected_shared == 8)


def test_primed_flat_population_freezes_the_exact_maximum_prefix_artifact(
    tmp_path: Path,
) -> None:
    tokenizer = PeriodicCorpusTokenizer()
    result = prepare_population(
        random_preparation_request(
            tmp_path,
            3,
            request_source={
                "kind": "random",
                "prompt": {"kind": "flat"},
                "input_tokens": {"kind": "inclusive_uniform", "min": 8, "max": 12},
                "output_tokens": 2,
                "prefix_sharing": {"shared_prefix_ratio": 0.5},
            },
            cache_start="primed",
        ),
        tokenizer,
    )

    assert result.prefix_conditioning is not None
    conditioning = result.prefix_conditioning
    content = Path(conditioning.path).read_text(encoding="utf-8")
    assert conditioning.prompt_tokens == 6
    assert len(tokenizer.encode(content, add_special_tokens=False)) == 6
    assert hashlib.sha256(content.encode()).hexdigest() == conditioning.sha256


def test_distributed_flat_ratio_uses_nested_prefixes_and_equivalent_fixed_geometry(
    tmp_path: Path,
) -> None:
    distributed = prepare_population(
        random_preparation_request(
            tmp_path,
            8,
            request_source={
                "kind": "random",
                "prompt": {"kind": "flat"},
                "input_tokens": {"kind": "inclusive_uniform", "min": 7, "max": 11},
                "output_tokens": 2,
                "prefix_sharing": {"shared_prefix_ratio": 0.5},
            },
            artifact_name="distributed",
        ),
        PeriodicCorpusTokenizer(),
    )
    assert distributed.population is not None
    assert distributed.evidence_path is not None
    prompts = [
        json.loads(line)["text_input"]
        for line in Path(distributed.population.path).read_text().splitlines()
    ]
    evidence = [
        json.loads(line) for line in Path(distributed.evidence_path).read_text().splitlines()
    ]
    tokenizer = PeriodicCorpusTokenizer()
    longest = max(
        range(len(evidence)), key=lambda index: evidence[index]["resolved_shared_prefix_tokens"]
    )
    canonical = tokenizer.encode(prompts[longest], add_special_tokens=False)
    for prompt, entry in zip(prompts, evidence, strict=True):
        token_ids = tokenizer.encode(prompt, add_special_tokens=False)
        shared = entry["resolved_shared_prefix_tokens"]
        assert shared == math.floor(entry["selected_prompt_tokens"] * 0.5)
        assert token_ids[:shared] == canonical[:shared]

    common = {
        "kind": "random",
        "prompt": {"kind": "flat"},
        "input_tokens": 8,
        "output_tokens": 2,
    }
    by_tokens = prepare_population(
        random_preparation_request(
            tmp_path,
            3,
            request_source={**common, "prefix_sharing": {"shared_prefix_tokens": 4}},
            artifact_name="tokens",
        ),
        PeriodicCorpusTokenizer(),
    )
    by_ratio = prepare_population(
        random_preparation_request(
            tmp_path,
            3,
            request_source={**common, "prefix_sharing": {"shared_prefix_ratio": 0.5}},
            artifact_name="ratio",
        ),
        PeriodicCorpusTokenizer(),
    )
    assert by_tokens.population is not None and by_ratio.population is not None
    assert (
        Path(by_tokens.population.path).read_bytes() == Path(by_ratio.population.path).read_bytes()
    )


def test_rendered_chat_freezes_custom_template_into_an_exact_flat_population(
    tmp_path: Path,
) -> None:
    source: dict[str, object] = {
        "kind": "random",
        "prompt": {
            "kind": "rendered_chat",
            "chat_template": "{{ messages }}",
            "chat_template_kwargs": {"enable_thinking": True},
        },
        "input_tokens": 8,
        "output_tokens": 4,
        "prefix_sharing": {"shared_prefix_tokens": 4},
    }
    result = prepare_population(
        random_preparation_request(tmp_path, 4, request_source=source),
        ExactIdentityChatTokenizer(),
    )

    assert result.population is not None
    assert result.prompt_token_targeting is not None
    assert result.prompt_token_targeting.projection_template is not None
    assert result.prompt_token_targeting.projection_template.source == "prompt_table"
    rows = [json.loads(line) for line in Path(result.population.path).read_text().splitlines()]
    assert all("text_input" in row and "messages" not in row for row in rows)
    assert all(len(row["text_input"].split()) == 8 for row in rows)

    value = request(
        tmp_path / "case",
        {"kind": "concurrency_limited", "concurrency": 1},
        request_body={},
        request_source=source,
    ).model_copy(update={"population": result.population})
    benchmark = cast(dict[str, object], aiperf_config(value)["benchmark"])
    endpoint = cast(dict[str, object], benchmark["endpoint"])
    assert endpoint["path"] == "/v1/completions"
    assert endpoint["type"] == "completions"


def test_rendered_chat_freezes_the_tokenizer_default_into_an_exact_flat_population(
    tmp_path: Path,
) -> None:
    result = prepare_population(
        random_preparation_request(
            tmp_path,
            2,
            request_source={
                "kind": "random",
                "prompt": {"kind": "rendered_chat"},
                "input_tokens": 8,
                "output_tokens": 4,
                "prefix_sharing": None,
            },
            request_body={},
        ),
        DefaultChatTokenizer(),
    )

    assert result.status == ClientStatus.succeeded
    assert result.population is not None
    assert result.prompt_token_targeting is not None
    assert result.prompt_token_targeting.exact_entries == 2
    assert result.prompt_token_targeting.fallback_entries == 0
    assert result.prompt_token_targeting.projection_template is not None
    assert result.prompt_token_targeting.projection_template.source == "tokenizer_default"
    rows = [json.loads(line) for line in Path(result.population.path).read_text().splitlines()]
    assert all("text_input" in row and "messages" not in row for row in rows)
    assert all(len(row["text_input"].split()) == 8 for row in rows)


def test_rendered_prefix_does_not_require_the_template_frame_to_round_trip_alone(
    tmp_path: Path,
) -> None:
    result = prepare_population(
        random_preparation_request(
            tmp_path,
            2,
            request_source={
                "kind": "random",
                "prompt": {"kind": "rendered_chat"},
                "input_tokens": 8,
                "output_tokens": 2,
                "prefix_sharing": {"shared_prefix_tokens": 4},
            },
            request_body={},
        ),
        NonRoundTripTemplatePrefixTokenizer(),
    )

    assert result.status == ClientStatus.succeeded
    assert result.population is not None
    assert result.prefix_geometry is not None
    assert result.prefix_geometry.shared_prefix_tokens.minimum == 4
    rows = [json.loads(line) for line in Path(result.population.path).read_text().splitlines()]
    tokenizer = NonRoundTripTemplatePrefixTokenizer()
    token_rows = [tokenizer.encode(row["text_input"], add_special_tokens=False) for row in rows]
    assert all(len(tokens) == 8 for tokens in token_rows)
    assert len({tuple(tokens[:4]) for tokens in token_rows}) == 1


def test_primed_rendered_prefix_requires_an_independently_exact_conditioning_prompt(
    tmp_path: Path,
) -> None:
    request = random_preparation_request(
        tmp_path,
        2,
        request_source={
            "kind": "random",
            "prompt": {"kind": "rendered_chat"},
            "input_tokens": 8,
            "output_tokens": 2,
            "prefix_sharing": {"shared_prefix_tokens": 4},
        },
        request_body={},
        cache_start="primed",
    )

    with pytest.raises(ValueError, match="conditioning prompt token stream"):
        prepare_population(request, NonRoundTripTemplatePrefixTokenizer())


def test_weighted_mixture_uses_one_canonical_prefix_stream_across_shapes(
    tmp_path: Path,
) -> None:
    result = prepare_population(
        random_preparation_request(
            tmp_path,
            16,
            request_source={
                "kind": "random_mixture",
                "prompt": {"kind": "flat"},
                "shapes": [
                    {"input_tokens": 8, "output_tokens": 2, "weight": 1},
                    {"input_tokens": 12, "output_tokens": 3, "weight": 1},
                ],
                "total_weight": 2,
                "prefix_sharing": {"shared_prefix_tokens": 4},
            },
        ),
        PeriodicCorpusTokenizer(),
    )

    assert result.population is not None
    assert result.evidence_path is not None
    rows = [json.loads(line) for line in Path(result.population.path).read_text().splitlines()]
    evidence = [json.loads(line) for line in Path(result.evidence_path).read_text().splitlines()]
    tokenizer = PeriodicCorpusTokenizer()
    token_rows = [tokenizer.encode(row["text_input"], add_special_tokens=False) for row in rows]
    assert {entry["selected_prompt_tokens"] for entry in evidence} == {8, 12}
    assert len({tuple(tokens[:4]) for tokens in token_rows}) == 1
    assert all(
        len(tokens) == entry["selected_prompt_tokens"]
        for tokens, entry in zip(token_rows, evidence, strict=True)
    )
    assert all(entry["resolved_shared_prefix_tokens"] == 4 for entry in evidence)


def test_local_prompt_targets_reconcile_with_backend_prompt_tokens(tmp_path: Path) -> None:
    source: dict[str, object] = {
        "kind": "random",
        "prompt": {"kind": "flat"},
        "input_tokens": 8,
        "output_tokens": 2,
        "prefix_sharing": None,
    }
    prepared = prepare_population(
        random_preparation_request(tmp_path, 4, request_source=source),
        PeriodicCorpusTokenizer(),
    )
    assert prepared.population is not None
    bench_request = request(
        tmp_path / "case",
        {"kind": "concurrency_limited", "concurrency": 1},
        request_body={},
        request_source=source,
    ).model_copy(update={"population": prepared.population})
    profiling_path = tmp_path / "profiling.jsonl"
    records = [
        {
            "metadata": {
                "benchmark_phase": "profiling",
                "session_num": index,
                "was_cancelled": False,
            },
            "metrics": {"input_sequence_length": {"value": 8, "unit": "tokens"}},
            "error": None,
        }
        for index in range(4)
    ]
    profiling_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )

    evidence, error = prompt_token_reconciliation(bench_request, profiling_path)
    assert error is None
    assert len(evidence) == 4
    assert all(item.reconciled for item in evidence)

    records[2]["metrics"] = {"input_sequence_length": {"value": 9, "unit": "tokens"}}
    profiling_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    evidence, error = prompt_token_reconciliation(bench_request, profiling_path)
    assert error == "profiling population entry 2 planned 8 prompt tokens, backend reported 9"
    assert evidence[2].reconciled is False
