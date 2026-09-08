"""Request-time image decoration for image-declared random Bench sources.

A random request source's `images` decoration
([[RFC-0004:C-BENCH-REQUEST-SOURCES]]) attaches `image_url` content parts to
each request on the chat-completions route at measurement-run time; the
frozen population carries text only. The pinned AIPerf owns the image
semantics — its native `dataset.images` options exist only on the synthetic
dataset, while InferLab's frozen population always arrives as a file dataset —
so the runner arms this release-owned composer plugin through AIPerf's plugin
registry. The plugin reuses AIPerf's native `ImageGenerator`, the component
behind `--image-width-mean`/`--image-height-mean`/`--image-batch-size` and
`--image-source`/`--image-source-sampling`, so image generation, source
indexing, and the sampling policies stay AIPerf-owned and seed-determined.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from importlib import import_module
from importlib.metadata import version
from typing import Any, Protocol, cast

from inferlab_measurement_sdk import (
    BenchClientRequest,
    BenchPromptRouteInput,
    BenchRequestSourceInputRandom,
)

IMAGE_DECORATION_ENV = "INFERLAB_AIPERF_IMAGE_DECORATION"

# The pinned AIPerf dataset mmap cache keys off the input file and settings,
# not this decoration; a hit would replay an undecorated (or differently
# decorated) composition. Decoration runs always re-compose.
MMAP_CACHE_DISABLE_ENV = "AIPERF_DATASET_MMAP_CACHE_ENABLED"


@dataclass(frozen=True)
class ImageSourceSpec:
    path: str
    sampling: str


@dataclass(frozen=True)
class ImageDecorationSpec:
    count: int
    width: int
    height: int
    source: ImageSourceSpec | None

    def as_dict(self) -> dict[str, object]:
        return {
            "count": self.count,
            "width": self.width,
            "height": self.height,
            "source": (
                None
                if self.source is None
                else {"path": self.source.path, "sampling": self.source.sampling}
            ),
        }


def image_decoration(request: BenchClientRequest) -> ImageDecorationSpec | None:
    """The effective image decoration of a random source, when declared.

    Images attach only on the chat-completions route
    ([[RFC-0004:C-BENCH-REQUEST-SOURCES]]); any other route carrying the
    decoration is a contract violation the runner refuses."""
    source_input = request.definition.request_source
    if source_input is None:
        return None
    source = source_input.root
    if not isinstance(source, BenchRequestSourceInputRandom):
        return None
    images = source.images
    if images is None:
        return None
    if request.definition.prompt.root.route is not BenchPromptRouteInput.chat_completions:
        raise ValueError("Bench image decoration requires the chat-completions route")
    if images.count < 1 or images.width < 1 or images.height < 1:
        raise ValueError("Bench image decoration requires positive count, width, and height")
    source_spec = None
    if images.source is not None:
        source_spec = ImageSourceSpec(
            path=images.source.resolved_path,
            sampling=str(images.source.sampling),
        )
    return ImageDecorationSpec(
        count=images.count,
        width=images.width,
        height=images.height,
        source=source_spec,
    )


def native_image_options(decoration: ImageDecorationSpec) -> dict[str, object]:
    """The decoration as AIPerf's native image options (its `dataset.images`
    subtable): fixed dimensions as zero-variance means, the per-request batch
    size, and the directory source with its sampling policy. The noise supply
    is AIPerf's default source and is not spelled out."""
    options: dict[str, object] = {
        "width": {"mean": decoration.width, "stddev": 0},
        "height": {"mean": decoration.height, "stddev": 0},
        "batch_size": decoration.count,
    }
    if decoration.source is not None:
        options["source"] = decoration.source.path
        options["source_sampling"] = decoration.source.sampling
    return options


def decoration_environment(decoration: ImageDecorationSpec) -> dict[str, str]:
    return {
        IMAGE_DECORATION_ENV: json.dumps(decoration.as_dict()),
        MMAP_CACHE_DISABLE_ENV: "false",
    }


class _PluginRegistry(Protocol):
    def register(
        self, category: object, name: str, cls: type[object], *, priority: int = 0
    ) -> None: ...

    def get_class(self, category: object, name: str) -> type[object]: ...


def register_image_decoration(supported_aiperf_version: str) -> None:
    """Register the decorating dataset composer over AIPerf's file-dataset
    composer. Called inside the AIPerf process only when the decoration
    environment is set; the pinned-version guard fails closed because the
    composer boundary is version-sensitive."""
    observed_version = version("aiperf")
    if observed_version != supported_aiperf_version:
        raise RuntimeError(
            "InferLab's image decoration requires "
            f"AIPerf {supported_aiperf_version}, found {observed_version}"
        )
    raw = os.environ.get(IMAGE_DECORATION_ENV)
    if raw is None:
        raise RuntimeError(f"{IMAGE_DECORATION_ENV} is not set")
    spec = _parse_spec(raw)
    composer_class = _decorating_composer_class(spec)
    plugin_module = import_module("aiperf.plugin")
    enum_module = import_module("aiperf.plugin.enums")
    registry = cast(_PluginRegistry, plugin_module.plugins)
    category = enum_module.PluginType.DATASET_COMPOSER
    name = str(enum_module.ComposerType.CUSTOM)
    registry.register(category, name, composer_class, priority=1)
    if registry.get_class(category, name) is not composer_class:
        raise RuntimeError("AIPerf did not activate InferLab's image-decorating composer")


def _parse_spec(raw: str) -> ImageDecorationSpec:
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("image decoration must be a JSON object")
    unknown = sorted(set(data) - {"count", "width", "height", "source"})
    if unknown:
        raise ValueError(f"image decoration carries unknown members: {unknown}")
    missing = [key for key in ("count", "width", "height") if key not in data]
    if missing:
        raise ValueError(f"image decoration is missing required members: {missing}")
    count, width, height = data["count"], data["width"], data["height"]
    # JSON booleans are ints in Python; True is not a valid count or dimension.
    if not (
        isinstance(count, int)
        and not isinstance(count, bool)
        and isinstance(width, int)
        and not isinstance(width, bool)
        and isinstance(height, int)
        and not isinstance(height, bool)
        and count >= 1
        and width >= 1
        and height >= 1
    ):
        raise ValueError("image decoration requires positive integer count, width, and height")
    source_data = data.get("source")
    source = None
    if source_data is not None:
        if not isinstance(source_data, dict):
            raise ValueError("image decoration source must be a JSON object")
        unknown_source = sorted(set(source_data) - {"path", "sampling"})
        if unknown_source:
            raise ValueError(f"image decoration source carries unknown members: {unknown_source}")
        path, sampling = source_data.get("path"), source_data.get("sampling")
        if not isinstance(path, str) or not path:
            raise ValueError("image decoration source requires a non-empty path")
        if sampling not in ("random-with-replacement", "shuffle-cycle", "sequential-cycle"):
            raise ValueError(f"image decoration source sampling {sampling!r} is not supported")
        source = ImageSourceSpec(path=path, sampling=cast(str, sampling))
    return ImageDecorationSpec(count=count, width=width, height=height, source=source)


def _decorating_composer_class(spec: ImageDecorationSpec) -> type[object]:
    """Build the composer subclass with the pinned AIPerf's classes; framework
    imports stay local to this registration path."""
    custom_module = import_module("aiperf.dataset.composer.custom")
    content_module = import_module("aiperf.config.dataset.content")
    image_module = import_module("aiperf.dataset.generator.image")
    base_composer = custom_module.CustomDatasetComposer

    class InferlabImageDecoratingComposer(base_composer):  # type: ignore[misc, valid-type]
        """Attach the declared images to every replayed turn after the frozen
        population is loaded; the population file itself carries text only."""

        def create_dataset(self) -> Any:
            conversations = super().create_dataset()
            image_config = content_module.ImageConfig(**native_image_options(spec))
            generator = image_module.ImageGenerator(image_config)
            for conversation in conversations:
                for turn in conversation.turns:
                    _decorate_turn(turn, generator, spec.count)
            return conversations

    return InferlabImageDecoratingComposer


def _decorate_turn(turn: Any, generator: Any, count: int) -> None:
    """Rewrite the turn's user message so its content carries the text part
    followed by the request's image_url parts. The frozen population's
    structured messages replay verbatim downstream, so the decoration writes
    the parts into that raw message."""
    raw_messages = getattr(turn, "raw_messages", None)
    if not isinstance(raw_messages, list):
        raise ValueError(
            "image decoration requires the frozen population's structured-message entries"
        )
    user_messages = [
        message
        for message in raw_messages
        if isinstance(message, dict) and message.get("role") == "user"
    ]
    if len(user_messages) != 1:
        raise ValueError("image decoration requires exactly one user message per population entry")
    message = user_messages[0]
    content = message.get("content")
    parts: list[dict[str, object]]
    if isinstance(content, str):
        parts = [{"type": "text", "text": content}]
    elif isinstance(content, list):
        parts = list(content)
    else:
        raise ValueError("image decoration requires a string or parts user content")
    for _ in range(count):
        parts.append({"type": "image_url", "image_url": {"url": generator.generate()}})
    message["content"] = parts
