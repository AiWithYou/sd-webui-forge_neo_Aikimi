"""Isolated SenseNova U1.5 inference worker used by Forge Neo.

The model has its own sampling loop and cannot be routed through Forge's
KSampler.  Keeping it in a child process makes cancellation reliable and
returns all GPU/CPU model memory to the OS when a job finishes.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import torch
from PIL import Image, ImageOps


EVENT_PREFIX = "SENSENOVA_EVENT "
NORM_MEAN = (0.5, 0.5, 0.5)
NORM_STD = (0.5, 0.5, 0.5)
IMAGE_GRID_FACTOR = 32
DEFAULT_TARGET_PIXELS = 2048 * 2048
MIN_INPUT_PIXELS = 512 * 512
MAX_INPUT_PIXELS = 2048 * 2048


def emit(stage: str, message: str, progress: float, **extra: Any) -> None:
    payload = {
        "stage": stage,
        "message": message,
        "progress": max(0.0, min(1.0, float(progress))),
        **extra,
    }
    sys.stdout.write(EVENT_PREFIX + json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    partial.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(partial, path)


def _atomic_png(path: Path, image: Image.Image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    image.save(partial, format="PNG", optimize=False)
    os.replace(partial, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _to_pil(batch: torch.Tensor) -> list[Image.Image]:
    mean = torch.tensor(NORM_MEAN, device=batch.device, dtype=batch.dtype).view(
        1, 3, 1, 1
    )
    std = torch.tensor(NORM_STD, device=batch.device, dtype=batch.dtype).view(
        1, 3, 1, 1
    )
    values = (batch * std + mean).clamp(0, 1).float().permute(0, 2, 3, 1).cpu().numpy()
    values = (values * 255.0).round().astype(np.uint8)
    return [Image.fromarray(value, mode="RGB") for value in values]


def _auto_input_max_pixels(image_count: int) -> int:
    if image_count <= 2:
        return MAX_INPUT_PIXELS
    total_budget = 2 * MAX_INPUT_PIXELS
    return max(MIN_INPUT_PIXELS, total_budget // max(1, image_count))


def _flatten_rgb(image: Image.Image) -> Image.Image:
    image = ImageOps.exif_transpose(image)
    if image.mode in {"RGBA", "LA"} or "transparency" in image.info:
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        background.alpha_composite(rgba)
        return background.convert("RGB")
    return image.convert("RGB")


def _load_images(
    paths: Sequence[str],
    *,
    smart_resize,
    input_max_pixels: int | str,
) -> list[Image.Image]:
    if input_max_pixels == "auto":
        budget = _auto_input_max_pixels(len(paths))
    else:
        budget = int(input_max_pixels)
    images: list[Image.Image] = []
    for raw_path in paths:
        with Image.open(raw_path) as opened:
            image = _flatten_rgb(opened)
        resized_height, resized_width = smart_resize(
            height=image.height,
            width=image.width,
            factor=IMAGE_GRID_FACTOR,
            min_pixels=budget,
            max_pixels=budget,
        )
        if image.size != (resized_width, resized_height):
            image = image.resize(
                (resized_width, resized_height), Image.Resampling.LANCZOS
            )
        images.append(image)
    return images


def _resolve_output_size(
    payload: dict[str, Any], images: Sequence[Image.Image], smart_resize
) -> tuple[int, int]:
    width = payload.get("width")
    height = payload.get("height")
    if width is not None and height is not None:
        return int(width), int(height)
    if not images:
        raise RuntimeError("Automatic output size requires at least one input image.")
    target_pixels = int(payload.get("target_pixels", DEFAULT_TARGET_PIXELS))
    source_width, source_height = images[0].size
    resized_height, resized_width = smart_resize(
        height=source_height,
        width=source_width,
        factor=IMAGE_GRID_FACTOR,
        min_pixels=target_pixels,
        max_pixels=target_pixels,
    )
    return resized_width, resized_height


@contextlib.contextmanager
def _sampling_progress(model: Any, total_steps: int) -> Iterator[None]:
    if not hasattr(model, "unpatchify"):
        yield
        return

    original = model.unpatchify
    completed = 0

    def wrapped(*args, **kwargs):
        nonlocal completed
        result = original(*args, **kwargs)
        completed += 1
        ratio = min(1.0, completed / max(1, total_steps))
        emit(
            "sampling",
            f"Sampling {min(completed, total_steps)} / {total_steps}",
            0.35 + ratio * 0.58,
            step=min(completed, total_steps),
            total_steps=total_steps,
        )
        return result

    model.unpatchify = wrapped
    try:
        yield
    finally:
        try:
            del model.unpatchify
        except AttributeError:
            model.unpatchify = original


def _validate_payload(payload: dict[str, Any]) -> None:
    mode = payload.get("mode")
    if mode not in {"text", "edit"}:
        raise RuntimeError(f"Unsupported mode: {mode!r}")
    if not str(payload.get("prompt", "")).strip():
        raise RuntimeError("Prompt is empty.")
    image_paths = payload.get("input_images", [])
    if mode == "edit" and not image_paths:
        raise RuntimeError("Image editing requires at least one reference image.")
    if mode == "text" and image_paths:
        raise RuntimeError("Text-to-image does not accept reference images.")
    for image_path in image_paths:
        if not Path(image_path).is_file():
            raise RuntimeError(f"Input image does not exist: {image_path}")

    width = payload.get("width")
    height = payload.get("height")
    if (width is None) != (height is None):
        raise RuntimeError("width and height must both be set or both be automatic.")
    if width is None and mode != "edit":
        raise RuntimeError("Text-to-image requires an explicit output resolution.")
    if width is not None:
        for name, value in (("width", width), ("height", height)):
            value = int(value)
            if value < 512 or value > 4096 or value % IMAGE_GRID_FACTOR:
                raise RuntimeError(
                    f"{name} must be a multiple of 32 from 512 through 4096."
                )


def _load_runtime(payload: dict[str, Any]):
    source_path = Path(payload["source_path"]).resolve()
    if not (source_path / "sensenova_u1" / "__init__.py").is_file():
        raise RuntimeError(f"SenseNova runtime source is incomplete: {source_path}")
    sys.path.insert(0, os.fspath(source_path))

    import sensenova_u1
    from sensenova_u1.models.neo_unify.utils import smart_resize
    from sensenova_u1.utils import (
        load_model_and_tokenizer,
        make_offload_ctx,
        vram_mode_keeps_generation_resident,
        vram_mode_to_prefetch_count,
    )

    attention_backend = str(payload.get("attn_backend", "auto"))
    sensenova_u1.set_attn_backend(attention_backend)
    dtype_name = str(payload.get("dtype", "bfloat16"))
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[dtype_name]
    vram_mode = str(payload.get("vram_mode", "low"))
    prefetch_count = vram_mode_to_prefetch_count(vram_mode)
    gguf_checkpoint = payload.get("gguf_checkpoint") or None

    emit("loading", "Loading model and tokenizer", 0.08)
    started = time.monotonic()
    model, tokenizer = load_model_and_tokenizer(
        str(payload["model_path"]),
        dtype=dtype,
        device=str(payload.get("device", "cuda")),
        gguf_checkpoint=str(gguf_checkpoint) if gguf_checkpoint else None,
        for_offload=prefetch_count > 0,
        device_map=None,
        max_memory=None,
    )
    emit(
        "loaded",
        "Model is ready",
        0.28,
        load_seconds=round(time.monotonic() - started, 3),
        effective_attn_backend=sensenova_u1.effective_attn_backend(),
    )

    offload_context = lambda: make_offload_ctx(
        model,
        prefetch_count,
        str(payload.get("device", "cuda")),
        keep_generation_resident=vram_mode_keeps_generation_resident(vram_mode),
        fast_vram_fraction=float(payload.get("fast_vram_fraction", 0.90)),
        fast_vram_headroom_gib=float(payload.get("fast_vram_headroom_gib", 2.0)),
        fast_activation_reserve_gib=float(
            payload.get("fast_activation_reserve_gib", 4.0)
        ),
        fast_vram_budget_gib=(
            float(payload["fast_vram_budget_gib"])
            if float(payload.get("fast_vram_budget_gib", 0.0)) > 0
            else None
        ),
    )
    return sensenova_u1, smart_resize, model, tokenizer, offload_context


def run_request(payload: dict[str, Any]) -> dict[str, Any]:
    _validate_payload(payload)
    started = time.monotonic()
    source_revision_file = (
        Path(payload["source_path"]).resolve().parent / ".sensenova_revision"
    )
    source_revision = (
        source_revision_file.read_text(encoding="utf-8").strip()
        if source_revision_file.is_file()
        else ""
    )

    sensenova_u1, smart_resize, model, tokenizer, offload_context = _load_runtime(
        payload
    )
    mode = str(payload["mode"])
    image_paths = [str(value) for value in payload.get("input_images", [])]
    images = _load_images(
        image_paths,
        smart_resize=smart_resize,
        input_max_pixels=payload.get("input_max_pixels", "auto"),
    )
    width, height = _resolve_output_size(payload, images, smart_resize)
    steps = int(payload.get("steps", 50))
    prompt = str(payload["prompt"]).strip()

    emit("preparing", f"Preparing {width} x {height} generation", 0.31)
    torch.backends.cuda.matmul.allow_tf32 = True
    with (
        torch.inference_mode(),
        offload_context() as offloaded,
        _sampling_progress(model, steps),
    ):
        if mode == "text":
            tensor = offloaded.t2i_generate(
                tokenizer,
                prompt,
                image_size=(width, height),
                cfg_scale=float(payload.get("cfg_scale", 4.0)),
                cfg_norm=str(payload.get("cfg_norm", "none")),
                timestep_shift=float(payload.get("timestep_shift", 3.0)),
                cfg_interval=(0.0, 1.0),
                num_steps=steps,
                batch_size=1,
                seed=int(payload.get("seed", 42)),
                think_mode=False,
            )
        else:
            tensor = offloaded.it2i_generate(
                tokenizer,
                prompt,
                images,
                image_size=(width, height),
                cfg_scale=float(payload.get("cfg_scale", 4.0)),
                img_cfg_scale=float(payload.get("img_cfg_scale", 1.0)),
                cfg_norm=str(payload.get("cfg_norm", "none")),
                timestep_shift=float(payload.get("timestep_shift", 3.0)),
                cfg_interval=(0.0, 1.0),
                num_steps=steps,
                batch_size=1,
                seed=int(payload.get("seed", 42)),
                think_mode=False,
            )

    emit("decoding", "Saving generated image", 0.95)
    output_images = _to_pil(tensor)
    if len(output_images) != 1:
        raise RuntimeError(f"Expected one output image, received {len(output_images)}.")
    output_path = Path(payload["output_path"]).resolve()
    metadata_path = Path(payload["metadata_path"]).resolve()
    _atomic_png(output_path, output_images[0])

    metadata = {
        "schema_version": 1,
        "model": str(payload["model_path"]),
        "mode": mode,
        "prompt": prompt,
        "quantization": str(payload["quantization"]),
        "gguf_checkpoint": str(payload.get("gguf_checkpoint") or ""),
        "source_revision": source_revision,
        "sensenova_u1_version": getattr(sensenova_u1, "__version__", "unknown"),
        "torch_version": torch.__version__,
        "width": width,
        "height": height,
        "steps": steps,
        "cfg_scale": float(payload.get("cfg_scale", 4.0)),
        "img_cfg_scale": float(payload.get("img_cfg_scale", 1.0)),
        "timestep_shift": float(payload.get("timestep_shift", 3.0)),
        "seed": int(payload.get("seed", 42)),
        "vram_mode": str(payload.get("vram_mode", "low")),
        "attn_backend": str(payload.get("attn_backend", "auto")),
        "dtype": str(payload.get("dtype", "bfloat16")),
        "input_image_count": len(images),
        "input_image_names": [Path(path).name for path in image_paths],
        "output_sha256": _sha256(output_path),
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }
    _atomic_json(metadata_path, metadata)
    result = {
        "output_path": os.fspath(output_path),
        "metadata_path": os.fspath(metadata_path),
        "metadata": metadata,
    }
    emit("complete", "Generation completed", 1.0, **result)
    return result


def self_test(source_path: str) -> None:
    source = Path(source_path).resolve()
    if not (source / "sensenova_u1" / "__init__.py").is_file():
        raise RuntimeError(f"SenseNova source was not found: {source}")
    sys.path.insert(0, os.fspath(source))
    import sentencepiece
    import sensenova_u1
    import transformers

    sys.stdout.write(
        json.dumps(
            {
                "ok": True,
                "sensenova_u1": getattr(sensenova_u1, "__version__", "unknown"),
                "sentencepiece": getattr(sentencepiece, "__version__", "unknown"),
                "torch": torch.__version__,
                "transformers": transformers.__version__,
            },
            ensure_ascii=False,
        )
        + "\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Forge Neo SenseNova U1.5 worker")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--request", help="Path to a validated job JSON file.")
    group.add_argument(
        "--self-test",
        dest="self_test_source",
        help="Runtime source directory to import-check.",
    )
    args = parser.parse_args()

    try:
        if args.self_test_source:
            self_test(args.self_test_source)
            return 0
        request_path = Path(args.request).resolve()
        payload = json.loads(request_path.read_text(encoding="utf-8"))
        run_request(payload)
        return 0
    except Exception as exc:
        emit("error", str(exc), 0.0, error_type=type(exc).__name__)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
