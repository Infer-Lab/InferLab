import json
from pathlib import Path
from typing import cast

import pytest
from inferlab_bench_runner.aiperf import (
    aiperf_config,
    inference_request_config,
    prepare_aiperf_execution,
)
from inferlab_bench_runner.aiperf_images import (
    IMAGE_DECORATION_ENV,
    MMAP_CACHE_DISABLE_ENV,
    ImageDecorationSpec,
    ImageSourceSpec,
    _decorate_turn,
    _parse_spec,
    decoration_environment,
    image_decoration,
    native_image_options,
)
from inferlab_measurement_sdk import (
    BenchClientRequest,
    CaseDeadline,
)

from .support import (
    dataset_request,
    request,
)


def images_source(
    source: dict[str, object] | None = None,
) -> dict[str, object]:
    images: dict[str, object] = {"width": 512, "height": 384, "count": 2}
    if source is not None:
        images["source"] = source
    return {
        "kind": "random",
        "input_tokens": 8000,
        "output_tokens": 1000,
        "prefix_sharing": None,
        "images": images,
    }


def directory_source() -> dict[str, object]:
    return {
        "path": "images/pool",
        "resolved_path": "/workspace/images/pool",
        "expected_sha256": "c" * 64,
        "sampling": "shuffle-cycle",
    }


def decoration(request_value: BenchClientRequest) -> ImageDecorationSpec:
    value = image_decoration(request_value)
    assert value is not None
    return value


def test_image_decoration_extracts_the_noise_supply(tmp_path: Path) -> None:
    value = request(
        tmp_path, {"kind": "concurrency_limited", "concurrency": 1}, request_source=images_source()
    )

    assert decoration(value) == ImageDecorationSpec(count=2, width=512, height=384, source=None)


def test_image_decoration_extracts_the_directory_source(tmp_path: Path) -> None:
    value = request(
        tmp_path,
        {"kind": "concurrency_limited", "concurrency": 1},
        request_source=images_source(directory_source()),
    )

    assert decoration(value) == ImageDecorationSpec(
        count=2,
        width=512,
        height=384,
        source=ImageSourceSpec(path="/workspace/images/pool", sampling="shuffle-cycle"),
    )


def test_image_decoration_is_absent_without_images(tmp_path: Path) -> None:
    value = request(tmp_path, {"kind": "concurrency_limited", "concurrency": 1})
    assert image_decoration(value) is None
    dataset = dataset_request(tmp_path / "dataset")
    assert image_decoration(dataset) is None


def test_image_decoration_rejects_a_non_chat_route(tmp_path: Path) -> None:
    value = request(
        tmp_path,
        {"kind": "concurrency_limited", "concurrency": 1},
        request_source=images_source(),
    )
    raw = value.model_dump(mode="json")
    raw["definition"]["prompt"] = {
        "kind": "flat",
        "request_representation": "flat_prompt",
        "route": "completions",
        "rendering_authority": "local_flat",
    }
    with pytest.raises(ValueError, match="chat-completions route"):
        image_decoration(BenchClientRequest.model_validate(raw))


def test_image_decoration_rejects_non_positive_dimensions(tmp_path: Path) -> None:
    value = request(
        tmp_path,
        {"kind": "concurrency_limited", "concurrency": 1},
        request_source=images_source(),
    )
    raw = value.model_dump(mode="json")
    cast(dict[str, object], raw["definition"]["request_source"])["images"] = {
        "width": 0,
        "height": 384,
        "count": 2,
    }
    with pytest.raises(ValueError, match="positive count, width, and height"):
        image_decoration(BenchClientRequest.model_validate(raw))


def test_native_image_options_render_the_fixed_dimensions() -> None:
    assert native_image_options(
        ImageDecorationSpec(count=2, width=512, height=384, source=None)
    ) == {
        "width": {"mean": 512, "stddev": 0},
        "height": {"mean": 384, "stddev": 0},
        "batch_size": 2,
    }
    assert native_image_options(
        ImageDecorationSpec(
            count=1,
            width=64,
            height=64,
            source=ImageSourceSpec(path="/workspace/images/pool", sampling="sequential-cycle"),
        )
    ) == {
        "width": {"mean": 64, "stddev": 0},
        "height": {"mean": 64, "stddev": 0},
        "batch_size": 1,
        "source": "/workspace/images/pool",
        "source_sampling": "sequential-cycle",
    }


def test_decoration_environment_round_trips_the_spec() -> None:
    spec = ImageDecorationSpec(
        count=2,
        width=512,
        height=384,
        source=ImageSourceSpec(path="/workspace/images/pool", sampling="shuffle-cycle"),
    )
    environment = decoration_environment(spec)

    assert environment[MMAP_CACHE_DISABLE_ENV] == "false"
    assert _parse_spec(environment[IMAGE_DECORATION_ENV]) == spec


def test_parse_spec_rejects_malformed_payloads() -> None:
    with pytest.raises(ValueError, match="JSON object"):
        _parse_spec(json.dumps([1, 2]))
    with pytest.raises(ValueError, match="positive integer"):
        _parse_spec(json.dumps({"count": 0, "width": 64, "height": 64, "source": None}))
    with pytest.raises(ValueError, match="non-empty path"):
        _parse_spec(
            json.dumps(
                {
                    "count": 1,
                    "width": 64,
                    "height": 64,
                    "source": {"path": "", "sampling": "shuffle-cycle"},
                }
            )
        )
    with pytest.raises(ValueError, match="not supported"):
        _parse_spec(
            json.dumps(
                {
                    "count": 1,
                    "width": 64,
                    "height": 64,
                    "source": {"path": "/pool", "sampling": "round-robin"},
                }
            )
        )


def test_synthetic_dataset_takes_the_native_image_options(tmp_path: Path) -> None:
    value = request(
        tmp_path,
        {"kind": "concurrency_limited", "concurrency": 1},
        request_source=images_source(directory_source()),
    )

    benchmark = cast(dict[str, object], aiperf_config(value)["benchmark"])
    dataset = cast(dict[str, object], benchmark["dataset"])
    assert dataset["type"] == "synthetic"
    assert dataset["images"] == {
        "width": {"mean": 512, "stddev": 0},
        "height": {"mean": 384, "stddev": 0},
        "batch_size": 2,
        "source": "/workspace/images/pool",
        "source_sampling": "shuffle-cycle",
    }


def test_frozen_population_arms_the_decorating_composer_environment(tmp_path: Path) -> None:
    value = request(
        tmp_path,
        {"kind": "concurrency_limited", "concurrency": 1},
        request_source=images_source(directory_source()),
    )
    raw = value.model_dump(mode="json")
    population_path = tmp_path / "population.jsonl"
    population_path.write_text(
        "".join(
            json.dumps(
                {
                    "session_id": f"inferlab-{index:08}",
                    "messages": [{"role": "user", "content": f"fixture prompt {index}"}],
                }
            )
            + "\n"
            for index in range(4)
        ),
        encoding="utf-8",
    )
    raw["population"] = {
        "path": str(population_path),
        "evidence_path": str(population_path),
        "sha256": "1" * 64,
        "entries": 4,
        "tpot_applicable": True,
    }
    populated = BenchClientRequest.model_validate(raw)

    prepared = prepare_aiperf_execution(populated, CaseDeadline(3600))

    assert prepared.environment[MMAP_CACHE_DISABLE_ENV] == "false"
    spec = _parse_spec(prepared.environment[IMAGE_DECORATION_ENV])
    assert spec == ImageDecorationSpec(
        count=2,
        width=512,
        height=384,
        source=ImageSourceSpec(path="/workspace/images/pool", sampling="shuffle-cycle"),
    )
    # The file dataset itself never carries the synthetic-only image options.
    config = json.loads(prepared.config_path.read_text(encoding="utf-8"))
    dataset = cast(dict[str, object], config["benchmark"]["dataset"])
    assert dataset["type"] == "file"
    assert "images" not in dataset
    # The inference-request evidence reports the decoration as the native
    # option rendering.
    request_config = json.loads(prepared.request_config_path.read_text(encoding="utf-8"))
    assert request_config["image_decoration"] == native_image_options(spec)


class FakeGenerator:
    def __init__(self) -> None:
        self.generated = 0

    def generate(self) -> str:
        self.generated += 1
        return f"data:image/png;base64,fake-{self.generated}"


class FakeTurn:
    def __init__(self, raw_messages: object) -> None:
        self.raw_messages: list[dict[str, object]] = cast(list[dict[str, object]], raw_messages)


def test_decorate_turn_appends_image_parts_after_the_text() -> None:
    turn = FakeTurn([{"role": "user", "content": "describe the image"}])
    generator = FakeGenerator()

    _decorate_turn(turn, generator, 2)

    assert turn.raw_messages[0]["content"] == [
        {"type": "text", "text": "describe the image"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,fake-1"}},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,fake-2"}},
    ]


def test_decorate_turn_preserves_existing_parts() -> None:
    existing = [
        {"type": "text", "text": "caption"},
        {"type": "image_url", "image_url": {"url": "u"}},
    ]
    turn = FakeTurn([{"role": "user", "content": existing}])

    _decorate_turn(turn, FakeGenerator(), 1)

    content = cast(list[dict[str, object]], turn.raw_messages[0]["content"])
    assert content[:2] == existing
    assert content[2]["type"] == "image_url"


def test_decorate_turn_requires_exactly_one_user_message() -> None:
    with pytest.raises(ValueError, match="exactly one user message"):
        _decorate_turn(FakeTurn([{"role": "system", "content": "s"}]), FakeGenerator(), 1)
    with pytest.raises(ValueError, match="exactly one user message"):
        _decorate_turn(
            FakeTurn(
                [
                    {"role": "user", "content": "a"},
                    {"role": "user", "content": "b"},
                ]
            ),
            FakeGenerator(),
            1,
        )
    with pytest.raises(ValueError, match="structured-message entries"):
        _decorate_turn(FakeTurn("not-a-list"), FakeGenerator(), 1)
    with pytest.raises(ValueError, match="string or parts"):
        _decorate_turn(FakeTurn([{"role": "user", "content": 7}]), FakeGenerator(), 1)


def test_inference_request_config_reports_no_decoration_without_images(
    tmp_path: Path,
) -> None:
    value = request(tmp_path, {"kind": "concurrency_limited", "concurrency": 1})
    assert inference_request_config(value)["image_decoration"] is None
