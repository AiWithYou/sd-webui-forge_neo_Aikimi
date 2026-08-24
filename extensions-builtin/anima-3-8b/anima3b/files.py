from __future__ import annotations

import os
from pathlib import Path

from safetensors import SafetensorError, safe_open

from modules_forge.anima_lora import (
    detect_anima_lora_block_count_from_keys,
    has_anima_lora_signature,
)

ARCHITECTURE = "anima_progressive_qwen35_cross_adapter_v1"


def forge_root() -> Path:
    return Path(__file__).resolve().parents[3]


def text_encoder_roots() -> list[Path]:
    roots = [forge_root() / "models" / "text_encoder"]
    try:
        from modules_forge.main_entry import module_list

        roots.extend(Path(path).parent for path in module_list.values())
    except Exception:
        pass
    return list(dict.fromkeys(path.resolve() for path in roots if path.is_dir()))


def lora_roots() -> list[Path]:
    roots = [forge_root() / "models" / "Lora"]
    try:
        from modules import shared

        roots = [
            Path(shared.cmd_opts.lora_dir),
            *(Path(path) for path in shared.cmd_opts.lora_dirs),
        ]
    except Exception:
        pass
    return list(dict.fromkeys(path.resolve() for path in roots if path.is_dir()))


def qwen35_models() -> dict[str, str]:
    found: dict[str, str] = {}
    markers = ("qwen35_4b", "qwen3.5-4b", "qwen3_5_4b")
    for root in text_encoder_roots():
        for path in root.rglob("*.safetensors"):
            if any(marker in path.name.lower() for marker in markers):
                found.setdefault(path.name, str(path))
    return dict(sorted(found.items()))


def adapters() -> dict[str, str]:
    found: dict[str, str] = {}
    for root in text_encoder_roots():
        for path in root.rglob("*.safetensors"):
            try:
                with safe_open(path, framework="pt", device="cpu") as checkpoint:
                    metadata = checkpoint.metadata() or {}
                    keys = set(checkpoint.keys())
                if metadata.get("architecture") != ARCHITECTURE:
                    continue
                if any(key.startswith(("timestep_gates.", "anchor_deviation")) for key in keys):
                    continue
            except (OSError, ValueError, SafetensorError):
                continue
            label = os.path.relpath(path, root).replace("\\", "/")
            found.setdefault(label, str(path))
    return dict(sorted(found.items()))


def standard_anima_loras() -> dict[str, str]:
    """Find complete 28-, 40-, or 52-block Anima LoRAs from their headers."""

    candidates: dict[str, Path] = {}
    for root in lora_roots():
        for path in sorted(root.rglob("*.safetensors")):
            # Forge's standard registry resolves duplicate stems to the last
            # configured path. Mirror that behavior before compatibility checks.
            candidates[path.stem] = path

    found: dict[str, str] = {}
    for name, path in candidates.items():
        try:
            with safe_open(path, framework="pt", device="cpu") as checkpoint:
                metadata = checkpoint.metadata() or {}
                keys = checkpoint.keys()
                block_count = detect_anima_lora_block_count_from_keys(keys)
                is_anima = has_anima_lora_signature(keys, metadata)
            if block_count is None or not is_anima:
                continue
        except (OSError, ValueError, SafetensorError):
            continue
        found[name] = str(path)
    return dict(sorted(found.items()))


def tokenizer_dir() -> Path:
    bundled = Path(__file__).resolve().parents[1] / "qwen35_tokenizer"
    if (bundled / "tokenizer.json").is_file():
        return bundled
    for candidate in (
        root / "qwen35_tokenizer" for root in text_encoder_roots()
    ):
        if (candidate / "tokenizer.json").is_file():
            return candidate
    raise FileNotFoundError(
        "Qwen3.5 tokenizer files are missing from the extension's "
        "qwen35_tokenizer directory. Reinstall the extension."
    )
