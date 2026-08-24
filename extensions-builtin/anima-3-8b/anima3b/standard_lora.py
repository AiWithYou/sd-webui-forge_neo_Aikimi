from __future__ import annotations

from collections.abc import Mapping
import math
from pathlib import Path
import re
from typing import Any


NO_STANDARD_LORA = "None"


def _append_lora_tag(prompt: str, name: str, strength: float) -> str:
    existing = re.compile(
        rf"<lora:\s*{re.escape(name)}\s*:[^>]+>",
        flags=re.IGNORECASE,
    )
    if existing.search(prompt):
        return prompt
    separator = "" if not prompt or prompt[-1].isspace() else " "
    return f"{prompt}{separator}<lora:{name}:{strength:.12g}>"


def _append_to_prompt_value(value: str | list[str], name: str, strength: float):
    if isinstance(value, str):
        return _append_lora_tag(value, name, strength)
    if isinstance(value, list):
        return [_append_lora_tag(prompt, name, strength) for prompt in value]
    raise TypeError(f"Unsupported prompt type for Anima LoRA selection: {type(value)!r}")


def apply_standard_lora_selection(
    processing: Any,
    name: str | None,
    strength: float,
    available: Mapping[str, str],
) -> bool:
    """Inject one UI-selected LoRA into standard Forge prompt processing."""

    if name in {None, "", NO_STANDARD_LORA}:
        return False
    if name not in available:
        raise FileNotFoundError(
            f"Standard Anima LoRA '{name}' is unavailable. Refresh Forge and select it again."
        )
    path = Path(available[name])
    if not path.is_file():
        raise FileNotFoundError(
            f"Standard Anima LoRA '{name}' no longer exists at {path}."
        )

    strength = float(strength)
    if not math.isfinite(strength):
        raise ValueError("Standard Anima LoRA strength must be finite.")

    processing.prompt = _append_to_prompt_value(processing.prompt, name, strength)
    processing.all_prompts = _append_to_prompt_value(
        processing.all_prompts,
        name,
        strength,
    )
    processing.extra_generation_params["Anima 3.8B standard LoRA"] = (
        f"{name}:{strength:.12g}"
    )
    return True
