"""Isolated SenseNova U1.5 inference worker used by Forge Neo.

The model has its own sampling loop and cannot be routed through Forge's
KSampler.  Keeping it in a child process makes cancellation reliable and
returns all GPU/CPU model memory to the OS when a job finishes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Sequence

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
TOTAL_INPUT_PIXELS = 4096 * 4096
MAX_REFERENCE_IMAGES = 64
EXPECTED_CONVROT_LAYERS = 588
LOW_VRAM_MAX_OUTPUT_PIXELS = 2048 * 2048
LOW_VRAM_MAX_INPUT_PIXELS = 512 * 512
LOW_VRAM_MAX_REFERENCE_IMAGES = 2
FINAL_MODEL_ID = "sensenova/SenseNova-U1.5-8B-MoT"
CHECKPOINT_REVISION = "57de22ad4e2fc24c77f56dfe45dbb87a60dfebee"
RUNTIME_REVISION = "e6dfd45762eb46f805067fe079c14bcb643ccccd"


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
    if batch.ndim != 4:
        raise RuntimeError(f"Expected a four-dimensional image batch, got {batch.ndim}D.")
    if batch.shape[-1] in {3, 4}:
        values = batch[..., :3].clamp(0, 1).float().cpu().numpy()
    elif batch.shape[1] == 3:
        mean = torch.tensor(NORM_MEAN, device=batch.device, dtype=batch.dtype).view(
            1, 3, 1, 1
        )
        std = torch.tensor(NORM_STD, device=batch.device, dtype=batch.dtype).view(
            1, 3, 1, 1
        )
        values = (
            (batch * std + mean)
            .clamp(0, 1)
            .float()
            .permute(0, 2, 3, 1)
            .cpu()
            .numpy()
        )
    else:
        raise RuntimeError(f"Unsupported image tensor shape: {tuple(batch.shape)}")
    values = (values * 255.0).round().astype(np.uint8)
    return [Image.fromarray(value, mode="RGB") for value in values]


def _auto_input_max_pixels(image_count: int) -> int:
    return max(
        MIN_INPUT_PIXELS,
        min(MAX_INPUT_PIXELS, TOTAL_INPUT_PIXELS // max(1, image_count)),
    )


def _effective_input_max_pixels(image_count: int, requested: int | str) -> int:
    automatic_limit = _auto_input_max_pixels(image_count)
    if requested == "auto":
        return automatic_limit
    return min(int(requested), automatic_limit)


def _flatten_rgb(image: Image.Image) -> Image.Image:
    image = ImageOps.exif_transpose(image)
    if image.mode in {"RGBA", "LA"} or "transparency" in image.info:
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        background.alpha_composite(rgba)
        return background.convert("RGB")
    return image.convert("RGB")


def _aspect_fit_size(
    source_width: int,
    source_height: int,
    canvas_width: int,
    canvas_height: int,
) -> tuple[int, int]:
    if min(source_width, source_height, canvas_width, canvas_height) <= 0:
        raise RuntimeError("Image and canvas dimensions must be positive.")
    scale = min(canvas_width / source_width, canvas_height / source_height)
    content_width = max(1, min(canvas_width, round(source_width * scale)))
    content_height = max(1, min(canvas_height, round(source_height * scale)))
    return content_width, content_height


def _resize_with_edge_padding(
    image: Image.Image, canvas_width: int, canvas_height: int
) -> Image.Image:
    content_width, content_height = _aspect_fit_size(
        image.width,
        image.height,
        canvas_width,
        canvas_height,
    )
    if image.size != (content_width, content_height):
        image = image.resize(
            (content_width, content_height), Image.Resampling.LANCZOS
        )

    pad_left = (canvas_width - content_width) // 2
    pad_right = canvas_width - content_width - pad_left
    pad_top = (canvas_height - content_height) // 2
    pad_bottom = canvas_height - content_height - pad_top
    if not any((pad_left, pad_right, pad_top, pad_bottom)):
        return image

    values = np.asarray(image)
    padded = np.pad(
        values,
        ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
        mode="edge",
    )
    return Image.fromarray(padded, mode="RGB")


def _load_images(
    paths: Sequence[str],
    *,
    smart_resize,
    input_max_pixels: int | str,
) -> tuple[list[Image.Image], list[tuple[int, int]]]:
    budget = _effective_input_max_pixels(len(paths), input_max_pixels)
    images: list[Image.Image] = []
    original_sizes: list[tuple[int, int]] = []
    for raw_path in paths:
        with Image.open(raw_path) as opened:
            image = _flatten_rgb(opened)
        original_sizes.append(image.size)
        resized_height, resized_width = smart_resize(
            height=image.height,
            width=image.width,
            factor=IMAGE_GRID_FACTOR,
            min_pixels=budget,
            max_pixels=budget,
        )
        images.append(
            _resize_with_edge_padding(image, resized_width, resized_height)
        )
    return images, original_sizes


def _resolve_output_size(
    payload: dict[str, Any], source_sizes: Sequence[tuple[int, int]], smart_resize
) -> tuple[int, int]:
    width = payload.get("width")
    height = payload.get("height")
    if width is not None and height is not None:
        return int(width), int(height)
    if not source_sizes:
        raise RuntimeError("Automatic output size requires at least one input image.")
    target_pixels = int(payload.get("target_pixels", DEFAULT_TARGET_PIXELS))
    source_width, source_height = source_sizes[0]
    resized_height, resized_width = smart_resize(
        height=source_height,
        width=source_width,
        factor=IMAGE_GRID_FACTOR,
        min_pixels=target_pixels,
        max_pixels=target_pixels,
    )
    return resized_width, resized_height


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
    if len(image_paths) > MAX_REFERENCE_IMAGES:
        raise RuntimeError(
            f"At most {MAX_REFERENCE_IMAGES} ordered reference images are supported."
        )
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
    if payload.get("quantization") != "int8_convrot":
        raise RuntimeError("This worker only accepts the final INT8 ConvRot checkpoint.")
    if payload.get("model_path") != FINAL_MODEL_ID:
        raise RuntimeError(f"This worker is fixed to the final model config: {FINAL_MODEL_ID}")
    if payload.get("checkpoint_revision") != CHECKPOINT_REVISION:
        raise RuntimeError("The pinned INT8 ConvRot checkpoint revision does not match.")
    checkpoint = Path(str(payload.get("checkpoint", "")))
    if checkpoint.suffix.lower() != ".safetensors" or not checkpoint.is_file():
        raise RuntimeError(f"INT8 ConvRot checkpoint does not exist: {checkpoint}")
    vram_mode = str(payload.get("vram_mode", "low"))
    if vram_mode not in {"low", "unrestricted", "full"}:
        raise RuntimeError(f"Unsupported VRAM mode: {vram_mode}")
    if vram_mode == "low":
        output_pixels = (
            int(payload.get("target_pixels", DEFAULT_TARGET_PIXELS))
            if width is None
            else int(width) * int(height)
        )
        requested_input = payload.get("input_max_pixels", "auto")
        safe_input = (
            None if requested_input == "auto" else int(requested_input)
        )
        if (
            output_pixels > LOW_VRAM_MAX_OUTPUT_PIXELS
            or len(image_paths) > LOW_VRAM_MAX_REFERENCE_IMAGES
            or (
                mode == "edit"
                and (
                    safe_input is None
                    or safe_input > LOW_VRAM_MAX_INPUT_PIXELS
                )
            )
        ):
            raise RuntimeError(
                "The 24 GB safe profile requires output <= 2048^2, at most 2 references, "
                "and input_max_pixels <= 512^2 per reference. Use unrestricted mode explicitly "
                "for a larger workload."
            )


def _load_runtime(payload: dict[str, Any]):
    source_path = Path(payload["source_path"]).resolve()
    package_path = source_path / "SenseNova" / "src" / "sensenova_u1" / "__init__.py"
    inference_path = source_path / "SenseNova" / "examples" / "editing" / "inference.py"
    config_repo = source_path / "SenseNova-U1.5-8B-MoT"
    if not package_path.is_file() or not inference_path.is_file() or not config_repo.is_dir():
        raise RuntimeError(f"SenseNova runtime source is incomplete: {source_path}")
    revision_file = source_path / ".sensenova_runtime_revision"
    revision = (
        revision_file.read_text(encoding="utf-8").strip()
        if revision_file.is_file()
        else ""
    )
    if revision != RUNTIME_REVISION:
        raise RuntimeError("SenseNova runtime revision does not match the pinned loader.")
    sys.path.insert(0, os.fspath(source_path))

    from SenseNova.examples.editing.inference import load_sensenova_model
    from SenseNova.src import sensenova_u1
    from SenseNova.src.sensenova_u1.models.neo_unify.utils import smart_resize

    attention_backend = str(payload.get("attn_backend", "auto"))
    dtype_name = str(payload.get("dtype", "bfloat16"))
    if dtype_name != "bfloat16":
        raise RuntimeError("INT8 ConvRot inference is fixed to bfloat16 compute.")
    dtype = torch.bfloat16
    vram_mode = str(payload.get("vram_mode", "low"))
    if vram_mode not in {"low", "unrestricted", "full"}:
        raise RuntimeError(f"Unsupported VRAM mode: {vram_mode}")
    prefetch_count = 1 if vram_mode in {"low", "unrestricted"} else None
    checkpoint = Path(payload["checkpoint"]).resolve()
    device = torch.device(str(payload.get("device", "cuda")))
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("SenseNova U1.5 INT8 ConvRot requires an NVIDIA CUDA GPU.")

    emit("loading", "Loading final INT8 ConvRot model and tokenizer", 0.08)
    started = time.monotonic()
    engine = load_sensenova_model(
        os.fspath(checkpoint),
        device,
        attention_backend,
        os.fspath(config_repo),
        dtype,
    )
    load_info = engine.quant_load_info
    loaded_layers = int(load_info.get("int8", 0))
    missing = list(load_info.get("missing_keys", []))
    unexpected = list(load_info.get("unexpected_keys", []))
    leftover = list(load_info.get("meta_materialized", []))
    if engine.quantization_format != "int8_convrot":
        raise RuntimeError(
            f"Expected int8_convrot, loaded {engine.quantization_format or 'unknown'}."
        )
    if loaded_layers != EXPECTED_CONVROT_LAYERS:
        raise RuntimeError(
            f"Expected {EXPECTED_CONVROT_LAYERS} INT8 ConvRot layers, loaded {loaded_layers}."
        )
    if missing or unexpected or leftover:
        raise RuntimeError(
            "Checkpoint/config mismatch: "
            f"{len(missing)} missing, {len(unexpected)} unexpected, "
            f"{len(leftover)} uninitialized tensors."
        )
    if engine.release_variant != "final" or not engine.pruned_lm_head:
        raise RuntimeError(
            "The checkpoint is not the pruned final SenseNova U1.5 T2I/edit release."
        )
    emit(
        "loaded",
        "Final INT8 ConvRot model is ready",
        0.28,
        load_seconds=round(time.monotonic() - started, 3),
        effective_attn_backend=sensenova_u1.effective_attn_backend(),
        loaded_int8_layers=loaded_layers,
        release_variant=engine.release_variant,
    )
    return sensenova_u1, smart_resize, engine, prefetch_count, loaded_layers


def run_request(payload: dict[str, Any]) -> dict[str, Any]:
    _validate_payload(payload)
    started = time.monotonic()
    source_revision_file = Path(payload["source_path"]).resolve() / ".sensenova_runtime_revision"
    source_revision = (
        source_revision_file.read_text(encoding="utf-8").strip()
        if source_revision_file.is_file()
        else ""
    )

    sensenova_u1, smart_resize, engine, prefetch_count, loaded_layers = _load_runtime(payload)
    mode = str(payload["mode"])
    image_paths = [str(value) for value in payload.get("input_images", [])]
    images, original_sizes = _load_images(
        image_paths,
        smart_resize=smart_resize,
        input_max_pixels=payload.get("input_max_pixels", "auto"),
    )
    effective_input_max_pixels = (
        _effective_input_max_pixels(
            len(image_paths), payload.get("input_max_pixels", "auto")
        )
        if image_paths
        else 0
    )
    width, height = _resolve_output_size(payload, original_sizes, smart_resize)
    steps = int(payload.get("steps", 50))
    prompt = str(payload["prompt"]).strip()

    emit("preparing", f"Preparing {width} x {height} generation", 0.31)
    torch.backends.cuda.matmul.allow_tf32 = True
    full_model = prefetch_count is None

    def progress_callback(value: int, total: int) -> None:
        ratio = min(1.0, value / max(1, total))
        emit(
            "sampling",
            f"Sampling {min(value, total)} / {total}",
            0.35 + ratio * 0.58,
            step=min(value, total),
            total_steps=total,
        )

    if full_model:
        engine.model.to(engine.device)
    try:
        with torch.inference_mode():
            if mode == "text":
                tensor = engine.generate(
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
                    streaming_prefetch_count=prefetch_count,
                    progress_callback=progress_callback,
                )
            else:
                tensor = engine.edit(
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
                    think_mode=False,
                    seed=int(payload.get("seed", 42)),
                    streaming_prefetch_count=prefetch_count,
                    progress_callback=progress_callback,
                )
    finally:
        if full_model:
            engine.model.to("cpu")
            torch.cuda.empty_cache()

    emit("decoding", "Saving generated image", 0.95)
    output_images = _to_pil(tensor)
    if len(output_images) != 1:
        raise RuntimeError(f"Expected one output image, received {len(output_images)}.")
    output_path = Path(payload["output_path"]).resolve()
    metadata_path = Path(payload["metadata_path"]).resolve()
    _atomic_png(output_path, output_images[0])

    metadata = {
        "schema_version": 2,
        "model": str(payload["model_path"]),
        "checkpoint_revision": str(payload["checkpoint_revision"]),
        "mode": mode,
        "prompt": prompt,
        "quantization": str(payload["quantization"]),
        "checkpoint": str(payload["checkpoint"]),
        "runtime_revision": source_revision,
        "release_variant": engine.release_variant,
        "pruned_lm_head": engine.pruned_lm_head,
        "loaded_int8_layers": loaded_layers,
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
        "input_preprocessing": "aspect_fit_edge_pad_32",
        "input_original_sizes": [
            {"width": size[0], "height": size[1]} for size in original_sizes
        ],
        "input_prepared_sizes": [
            {"width": image.width, "height": image.height} for image in images
        ],
        "output_aspect_source": (
            "original_input_1"
            if payload.get("width") is None and payload.get("height") is None
            else "explicit"
        ),
        "requested_input_max_pixels": payload.get("input_max_pixels", "auto"),
        "effective_input_max_pixels": effective_input_max_pixels,
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
    package = source / "SenseNova" / "src" / "sensenova_u1" / "__init__.py"
    inference = source / "SenseNova" / "examples" / "editing" / "inference.py"
    config = source / "SenseNova-U1.5-8B-MoT" / "config.json"
    if not package.is_file() or not inference.is_file() or not config.is_file():
        raise RuntimeError(f"SenseNova source was not found: {source}")
    sys.path.insert(0, os.fspath(source))
    import comfy_kitchen
    import safetensors
    import transformers
    from SenseNova.examples.editing.inference import load_sensenova_model
    from SenseNova.src import sensenova_u1

    sys.stdout.write(
        json.dumps(
            {
                "ok": True,
                "sensenova_u1": getattr(sensenova_u1, "__version__", "unknown"),
                "loader": load_sensenova_model.__name__,
                "comfy_kitchen": getattr(comfy_kitchen, "__version__", "unknown"),
                "safetensors": getattr(safetensors, "__version__", "unknown"),
                "torch": torch.__version__,
                "transformers": transformers.__version__,
            },
            ensure_ascii=False,
        )
        + "\n"
    )


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")
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
