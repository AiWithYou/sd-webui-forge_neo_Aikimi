import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image

from modules_forge.krea2_upscale import require_positive_int, target_size
from tools.krea2_8k_img2img import (
    decode_b64_image,
    image_to_b64_png,
    post_img2img,
    prompt_from_png,
)


DEFAULT_API = "http://127.0.0.1:7861"
DEFAULT_OUTPUT_ROOT = "output/krea2_tiled_refine"
DEFAULT_MAX_OUTPUT_PIXELS = 24_000_000
DEFAULT_MAX_TILE_PIXELS = 1_638_400


def emit(message: str):
    sys.stdout.write(f"{message}\n")
    sys.stdout.flush()


def require_positive_float(value: float, name: str):
    if value <= 0:
        raise ValueError(f"{name} must be > 0.")


def validate_args(args):
    if (args.width is None) != (args.height is None):
        raise ValueError("--width and --height must be passed together.")

    for value, name in (
        (args.long_edge, "--long-edge"),
        (args.width, "--width"),
        (args.height, "--height"),
        (args.tile_size, "--tile-size"),
        (args.steps, "--steps"),
        (args.timeout, "--timeout"),
        (args.max_output_pixels, "--max-output-pixels"),
        (args.max_tile_pixels, "--max-tile-pixels"),
    ):
        require_positive_int(value, name)

    require_positive_float(args.scale, "--scale")
    if args.overlap < 0:
        raise ValueError("--overlap must be >= 0.")
    if args.overlap >= args.tile_size:
        raise ValueError("--overlap must be < --tile-size.")
    if args.tile_size * args.tile_size > args.max_tile_pixels:
        raise ValueError("--tile-size exceeds --max-tile-pixels.")
    if not 0 <= args.denoise <= 1:
        raise ValueError("--denoise must be between 0 and 1.")
    if args.cfg < 0:
        raise ValueError("--cfg must be >= 0.")
    if args.distilled_cfg < 0:
        raise ValueError("--distilled-cfg must be >= 0.")
    if args.progress_interval < 0:
        raise ValueError("--progress-interval must be >= 0.")
    if args.no_progress_timeout < 0:
        raise ValueError("--no-progress-timeout must be >= 0.")
    if args.no_progress_timeout > 0 and args.progress_interval <= 0:
        raise ValueError("--no-progress-timeout requires --progress-interval > 0.")


def target_size_from_args(width: int, height: int, args) -> tuple[int, int]:
    if args.width is not None and args.height is not None:
        target_w, target_h = target_size(
            width, height, args.long_edge, args.width, args.height
        )
    elif args.long_edge is not None:
        target_w, target_h = target_size(width, height, args.long_edge, None, None)
    else:
        target_w = max(1, round(width * args.scale / 64) * 64)
        target_h = max(1, round(height * args.scale / 64) * 64)

    pixels = target_w * target_h
    if pixels > args.max_output_pixels:
        raise ValueError(
            f"target size {target_w}x{target_h} is {pixels:,} pixels, exceeding --max-output-pixels."
        )
    return target_w, target_h


def tile_positions(length: int, tile_size: int, overlap: int) -> list[int]:
    require_positive_int(length, "length")
    require_positive_int(tile_size, "tile size")
    if overlap < 0:
        raise ValueError("overlap must be >= 0.")
    if overlap >= tile_size:
        raise ValueError("overlap must be < tile size.")
    if length <= tile_size:
        return [0]

    stride = tile_size - overlap
    positions = list(range(0, length - tile_size + 1, stride))
    end_position = length - tile_size
    if positions[-1] != end_position:
        positions.append(end_position)
    return positions


def axis_weights(
    tile_length: int, previous_overlap: int, next_overlap: int
) -> np.ndarray:
    require_positive_int(tile_length, "tile length")
    weights = np.ones(tile_length, dtype=np.float32)

    previous_overlap = min(max(0, previous_overlap), tile_length)
    next_overlap = min(max(0, next_overlap), tile_length)
    if previous_overlap > 0:
        weights[:previous_overlap] = np.minimum(
            weights[:previous_overlap],
            np.linspace(0.0, 1.0, previous_overlap, endpoint=True, dtype=np.float32),
        )
    if next_overlap > 0:
        weights[-next_overlap:] = np.minimum(
            weights[-next_overlap:],
            np.linspace(1.0, 0.0, next_overlap, endpoint=True, dtype=np.float32),
        )
    return weights


def tile_weight_mask(
    x_positions: list[int],
    y_positions: list[int],
    x_index: int,
    y_index: int,
    tile_w: int,
    tile_h: int,
) -> np.ndarray:
    x = x_positions[x_index]
    y = y_positions[y_index]
    previous_x_overlap = (
        0 if x_index == 0 else max(0, x_positions[x_index - 1] + tile_w - x)
    )
    next_x_overlap = (
        0
        if x_index == len(x_positions) - 1
        else max(0, x + tile_w - x_positions[x_index + 1])
    )
    previous_y_overlap = (
        0 if y_index == 0 else max(0, y_positions[y_index - 1] + tile_h - y)
    )
    next_y_overlap = (
        0
        if y_index == len(y_positions) - 1
        else max(0, y + tile_h - y_positions[y_index + 1])
    )

    xw = axis_weights(tile_w, previous_x_overlap, next_x_overlap)
    yw = axis_weights(tile_h, previous_y_overlap, next_y_overlap)
    return np.outer(yw, xw).astype(np.float32)


def build_tile_payload(
    args, tile: Image.Image, prompt: str, negative_prompt: str
) -> dict:
    return {
        "init_images": [image_to_b64_png(tile)],
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "seed": args.seed,
        "sampler_name": args.sampler,
        "scheduler": args.scheduler,
        "steps": args.steps,
        "cfg_scale": args.cfg,
        "distilled_cfg_scale": args.distilled_cfg,
        "width": tile.width,
        "height": tile.height,
        "resize_mode": 0,
        "denoising_strength": args.denoise,
        "n_iter": 1,
        "batch_size": 1,
        "restore_faces": False,
        "tiling": False,
        "send_images": True,
        "save_images": False,
        "include_init_images": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Krea2 tiled img2img refine with feather blending."
    )
    parser.add_argument("--input", required=True, help="Input image path.")
    parser.add_argument("--api", default=DEFAULT_API, help="Forge API base URL.")
    parser.add_argument(
        "--output-root", default=DEFAULT_OUTPUT_ROOT, help="Output root directory."
    )
    parser.add_argument(
        "--prompt", default=None, help="Prompt. Defaults to PNG infotext prompt."
    )
    parser.add_argument(
        "--negative-prompt",
        default=None,
        help="Negative prompt. Defaults to PNG infotext negative prompt.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.5,
        help="Target scale when --long-edge/--width/--height are not set.",
    )
    parser.add_argument("--long-edge", type=int, default=None, help="Target long edge.")
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Explicit target width. Must be passed with --height.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Explicit target height. Must be passed with --width.",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=1024,
        help="Maximum tile width/height sent to img2img.",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=192,
        help="Tile overlap used for feather blending.",
    )
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--sampler", default="DPM++ SDE")
    parser.add_argument("--scheduler", default="Simple")
    parser.add_argument("--cfg", type=float, default=1.0)
    parser.add_argument("--distilled-cfg", type=float, default=1.15)
    parser.add_argument("--denoise", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument(
        "--timeout", type=int, default=1800, help="Per-tile HTTP timeout in seconds."
    )
    parser.add_argument("--progress-interval", type=float, default=20.0)
    parser.add_argument("--no-progress-timeout", type=float, default=600.0)
    parser.add_argument(
        "--max-output-pixels", type=int, default=DEFAULT_MAX_OUTPUT_PIXELS
    )
    parser.add_argument("--max-tile-pixels", type=int, default=DEFAULT_MAX_TILE_PIXELS)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Prepare resized base and manifest without calling Forge.",
    )
    args = parser.parse_args()
    validate_args(args)

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    with Image.open(input_path) as image:
        source = image.convert("RGB")

    target_w, target_h = target_size_from_args(source.width, source.height, args)
    base = source.resize((target_w, target_h), Image.Resampling.LANCZOS)

    png_prompt, png_negative = prompt_from_png(input_path)
    prompt = args.prompt if args.prompt is not None else png_prompt
    negative_prompt = (
        args.negative_prompt if args.negative_prompt is not None else png_negative
    )
    if not prompt:
        raise ValueError(
            "Prompt is empty. Pass --prompt or use a PNG with Forge infotext."
        )

    x_positions = tile_positions(target_w, args.tile_size, args.overlap)
    y_positions = tile_positions(target_h, args.tile_size, args.overlap)
    tile_count = len(x_positions) * len(y_positions)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / f"krea2_tiled_refine_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    base_path = output_dir / "base_lanczos.png"
    base.save(base_path)

    manifest = {
        "input": str(input_path),
        "source_size": f"{source.width}x{source.height}",
        "target_size": f"{target_w}x{target_h}",
        "tile_size": args.tile_size,
        "overlap": args.overlap,
        "tile_count": tile_count,
        "x_positions": x_positions,
        "y_positions": y_positions,
        "steps": args.steps,
        "denoise": args.denoise,
        "sampler": args.sampler,
        "scheduler": args.scheduler,
        "cfg": args.cfg,
        "distilled_cfg": args.distilled_cfg,
        "base_lanczos": str(base_path),
    }
    manifest_path = output_dir / "tile_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    emit(f"OUTPUT_DIR={output_dir}")
    emit(f"INPUT={input_path}")
    emit(f"SOURCE={source.width}x{source.height}")
    emit(f"TARGET={target_w}x{target_h} ({target_w * target_h / 1_000_000:.1f} MP)")
    emit(f"TILE_SIZE={args.tile_size}")
    emit(f"OVERLAP={args.overlap}")
    emit(f"TILE_COUNT={tile_count}")
    emit(f"BASE_LANCZOS={base_path}")
    emit(f"MANIFEST={manifest_path}")

    if args.dry_run:
        emit("DRY_RUN=1")
        return 0

    accumulator = np.zeros((target_h, target_w, 3), dtype=np.float32)
    weight_sum = np.zeros((target_h, target_w), dtype=np.float32)

    tile_index = 0
    for y_index, y in enumerate(y_positions):
        for x_index, x in enumerate(x_positions):
            tile_index += 1
            tile = base.crop(
                (
                    x,
                    y,
                    min(x + args.tile_size, target_w),
                    min(y + args.tile_size, target_h),
                )
            )
            emit(
                f"TILE={tile_index}/{tile_count} X={x} Y={y} SIZE={tile.width}x{tile.height}"
            )
            payload = build_tile_payload(args, tile, prompt, negative_prompt)
            preview_payload = dict(payload)
            preview_payload["init_images"] = [
                f"<base64 PNG omitted: {tile.width}x{tile.height}>"
            ]
            (output_dir / f"tile_{tile_index:04d}_request_preview.json").write_text(
                json.dumps(preview_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            data = post_img2img(args, payload, f"TILE_{tile_index:04d}")
            images = data.get("images") or []
            if not images:
                raise RuntimeError(f"Tile {tile_index} returned no image.")

            refined_tile = decode_b64_image(images[0])
            if refined_tile.size != tile.size:
                raise RuntimeError(
                    f"Tile {tile_index} returned {refined_tile.size}, expected {tile.size}."
                )
            tile_path = output_dir / f"tile_{tile_index:04d}.png"
            refined_tile.save(tile_path)

            tile_array = np.asarray(refined_tile, dtype=np.float32)
            mask = tile_weight_mask(
                x_positions,
                y_positions,
                x_index,
                y_index,
                refined_tile.width,
                refined_tile.height,
            )
            accumulator[y : y + refined_tile.height, x : x + refined_tile.width, :] += (
                tile_array * mask[:, :, None]
            )
            weight_sum[y : y + refined_tile.height, x : x + refined_tile.width] += mask

    if np.any(weight_sum <= 0):
        raise RuntimeError("Tile blending left uncovered pixels.")

    result = np.clip(accumulator / weight_sum[:, :, None], 0, 255).astype(np.uint8)
    result_image = Image.fromarray(result, mode="RGB")
    output_path = output_dir / "krea2_tiled_refine.png"
    result_image.save(output_path)
    emit(f"IMAGE={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
