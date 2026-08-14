"""Block-layout conversion helpers for Anima LoRAs.

Anima-2.9B expands the original 28 transformer blocks to 40 blocks.  The
inserted blocks were initialized from specific original blocks, so a LoRA can
be expanded without copying tensor storage by exposing the same tensor under
each corresponding 40-block key.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import TypeVar


ANIMA_BASE_BLOCKS = 28
ANIMA_29B_BLOCKS = 40

# Positions inserted by the published Anima-2.9B expansion manifest.
ANIMA_29B_INSERTED_BLOCKS = (2, 5, 8, 11, 14, 17, 21, 24, 27, 30, 33, 36)

# Each inserted block was initialized from this block in the 28-block model.
_ANIMA_29B_INSERTED_TO_BASE = {
    2: 1,
    5: 3,
    8: 5,
    11: 7,
    14: 9,
    17: 11,
    21: 14,
    24: 16,
    27: 18,
    30: 20,
    33: 22,
    36: 24,
}

ANIMA_BASE_TO_29B = tuple(block for block in range(ANIMA_29B_BLOCKS) if block not in ANIMA_29B_INSERTED_BLOCKS)
_ANIMA_29B_ORIGINAL_TO_BASE = {expanded: base for base, expanded in enumerate(ANIMA_BASE_TO_29B)}
ANIMA_29B_TO_BASE = tuple(
    _ANIMA_29B_INSERTED_TO_BASE.get(expanded, _ANIMA_29B_ORIGINAL_TO_BASE.get(expanded))
    for expanded in range(ANIMA_29B_BLOCKS)
)
_ANIMA_BASE_TO_29B_TARGETS = tuple(
    tuple(expanded for expanded, base in enumerate(ANIMA_29B_TO_BASE) if base == source)
    for source in range(ANIMA_BASE_BLOCKS)
)


# Kohya, Forge generic, PEFT-style, and bare Comfy-style block names.
_ANIMA_BLOCK_KEY_PATTERNS = (
    re.compile(r"^(?P<prefix>lora_unet_(?:model_)?(?:diffusion_model_)?blocks_)(?P<block>\d+)(?P<suffix>_.+)$"),
    re.compile(r"^(?P<prefix>(?:.+\.)?diffusion_model\.blocks\.)(?P<block>\d+)(?P<suffix>\..+)$"),
    re.compile(r"^(?P<prefix>blocks\.)(?P<block>\d+)(?P<suffix>\..+)$"),
)


@dataclass(frozen=True)
class AnimaLoraConversionReport:
    source_blocks: int | None
    target_blocks: int
    source_block_indices: tuple[int, ...]
    direction: str
    duplicated_entries: int = 0
    dropped_entries: int = 0

    @property
    def converted(self) -> bool:
        return self.direction in {"28_to_40", "40_to_28"}


_Value = TypeVar("_Value")


def _parse_anima_block_key(key: str) -> tuple[int, str, str] | None:
    for pattern in _ANIMA_BLOCK_KEY_PATTERNS:
        match = pattern.match(key)
        if match is not None:
            return int(match.group("block")), match.group("prefix"), match.group("suffix")
    return None


def anima_lora_block_indices(lora: dict[str, _Value]) -> tuple[int, ...]:
    """Return sorted transformer-block indices recognized in a LoRA state dict."""

    indices = set()
    for key in lora:
        parsed = _parse_anima_block_key(key)
        if parsed is not None:
            indices.add(parsed[0])
    return tuple(sorted(indices))


def detect_anima_lora_block_count(lora: dict[str, _Value]) -> int | None:
    """Detect only complete 28- or 40-block layouts to avoid unsafe guessing."""

    indices = anima_lora_block_indices(lora)
    if indices == tuple(range(ANIMA_BASE_BLOCKS)):
        return ANIMA_BASE_BLOCKS
    if indices == tuple(range(ANIMA_29B_BLOCKS)):
        return ANIMA_29B_BLOCKS
    return None


def convert_anima_lora_layout(
    lora: dict[str, _Value], target_blocks: int
) -> tuple[dict[str, _Value], AnimaLoraConversionReport]:
    """Convert a complete Anima LoRA between the 28- and 40-block layouts.

    Conversion is deliberately skipped for sparse or otherwise ambiguous block
    coverage.  Expanding reuses tensor objects for inserted blocks, avoiding a
    second allocation of large LoRA tensors.  Collapsing a native 40-block LoRA
    drops entries for the 12 inserted blocks and is therefore lossy.
    """

    source_indices = anima_lora_block_indices(lora)
    source_blocks = detect_anima_lora_block_count(lora)

    if target_blocks not in {ANIMA_BASE_BLOCKS, ANIMA_29B_BLOCKS}:
        return lora, AnimaLoraConversionReport(source_blocks, target_blocks, source_indices, "unsupported_target")

    if source_blocks is None:
        return lora, AnimaLoraConversionReport(None, target_blocks, source_indices, "ambiguous_source")

    if source_blocks == target_blocks:
        return lora, AnimaLoraConversionReport(source_blocks, target_blocks, source_indices, "native")

    converted: dict[str, _Value] = {}
    source_for_output: dict[str, str] = {}
    duplicated_entries = 0
    dropped_entries = 0

    def add(output_key: str, value: _Value, source_key: str) -> None:
        if output_key in converted:
            previous = source_for_output[output_key]
            raise ValueError(f"Anima LoRA conversion collision for {output_key!r}: {previous!r} and {source_key!r}")
        converted[output_key] = value
        source_for_output[output_key] = source_key

    for key, value in lora.items():
        parsed = _parse_anima_block_key(key)
        if parsed is None:
            add(key, value, key)
            continue

        source_block, prefix, suffix = parsed
        if source_blocks == ANIMA_BASE_BLOCKS and target_blocks == ANIMA_29B_BLOCKS:
            targets = _ANIMA_BASE_TO_29B_TARGETS[source_block]
            for target_block in targets:
                add(f"{prefix}{target_block}{suffix}", value, key)
            duplicated_entries += len(targets) - 1
            continue

        base_block = _ANIMA_29B_ORIGINAL_TO_BASE.get(source_block)
        if base_block is None:
            dropped_entries += 1
            continue
        add(f"{prefix}{base_block}{suffix}", value, key)

    direction = "28_to_40" if source_blocks == ANIMA_BASE_BLOCKS else "40_to_28"
    return converted, AnimaLoraConversionReport(
        source_blocks,
        target_blocks,
        source_indices,
        direction,
        duplicated_entries=duplicated_entries,
        dropped_entries=dropped_entries,
    )
