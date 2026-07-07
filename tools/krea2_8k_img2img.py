import argparse
import base64
import io
import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import requests
from PIL import Image

from modules_forge.krea2_upscale import require_positive_int, target_size, two_stage_sizes


DEFAULT_API = "http://127.0.0.1:7861"
DEFAULT_OUTPUT_ROOT = "output"


def emit(message: str):
    sys.stdout.write(f"{message}\n")


def validate_args(args):
    if (args.width is None) != (args.height is None):
        raise ValueError("--width and --height must be passed together.")

    for value, name in (
        (args.long_edge, "--long-edge"),
        (args.width, "--width"),
        (args.height, "--height"),
        (args.steps, "--steps"),
        (args.tile_width, "--tile-width"),
        (args.tile_height, "--tile-height"),
        (args.tile_batch_size, "--tile-batch-size"),
        (args.timeout, "--timeout"),
    ):
        require_positive_int(value, name)

    if args.first_pass_long_edge < 0:
        raise ValueError("--first-pass-long-edge must be >= 0.")
    if args.tile_overlap < 0:
        raise ValueError("--tile-overlap must be >= 0.")
    if not 0 <= args.denoise <= 1:
        raise ValueError("--denoise must be between 0 and 1.")
    if args.cfg < 0:
        raise ValueError("--cfg must be >= 0.")
    if args.distilled_cfg < 0:
        raise ValueError("--distilled-cfg must be >= 0.")
    if not 0 <= args.first_pass_denoise <= 1:
        raise ValueError("--first-pass-denoise must be between 0 and 1.")


def image_to_b64_png(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def decode_b64_image(data: str) -> Image.Image:
    if "," in data:
        data = data.split(",", 1)[1]
    with Image.open(io.BytesIO(base64.b64decode(data))) as image:
        return image.convert("RGB")


def prompt_from_png(path: Path) -> tuple[str, str]:
    with Image.open(path) as image:
        params = image.info.get("parameters", "")
    if not params:
        return "", ""
    negative = ""
    prompt_block = params
    if "\nNegative prompt:" in params:
        prompt_block, rest = params.split("\nNegative prompt:", 1)
        negative = rest.split("\nSteps:", 1)[0].strip()
    elif "\nSteps:" in params:
        prompt_block = params.split("\nSteps:", 1)[0]
    return prompt_block.strip(), negative


def recent_saved_images(root: Path, started_at: float) -> list[Path]:
    candidates: list[Path] = []
    for subdir in [root / "img2img-images", root]:
        if not subdir.exists():
            continue
        for pattern in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
            for path in subdir.rglob(pattern):
                try:
                    if path.stat().st_mtime >= started_at - 2:
                        candidates.append(path)
                except OSError:
                    pass
    return sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)


def build_img2img_payload(args, image: Image.Image, prompt: str, negative_prompt: str, width: int, height: int, denoise: float, send_images: bool, save_images: bool) -> dict:
    return {
        "init_images": [image_to_b64_png(image)],
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "seed": args.seed,
        "sampler_name": args.sampler,
        "scheduler": args.scheduler,
        "steps": args.steps,
        "cfg_scale": args.cfg,
        "distilled_cfg_scale": args.distilled_cfg,
        "width": width,
        "height": height,
        "resize_mode": 0,
        "denoising_strength": denoise,
        "n_iter": 1,
        "batch_size": 1,
        "restore_faces": False,
        "tiling": False,
        "send_images": send_images,
        "save_images": save_images,
        "include_init_images": False,
        "alwayson_scripts": {
            "multidiffusion integrated": {
                "args": [
                    True,
                    args.method,
                    args.tile_width,
                    args.tile_height,
                    args.tile_overlap,
                    args.tile_batch_size,
                ]
            }
        },
    }


def write_payload_preview(output_dir: Path, name: str, payload: dict, init_image_description: str, extra: dict | None = None) -> Path:
    preview = dict(payload)
    preview["init_images"] = [init_image_description]
    if extra:
        preview.update(extra)
    preview_path = output_dir / name
    preview_path.write_text(json.dumps(preview, ensure_ascii=False, indent=2), encoding="utf-8")
    return preview_path


def post_img2img(args, payload: dict, stage_name: str) -> dict:
    response = requests.post(f"{args.api}/sdapi/v1/img2img", json=payload, timeout=args.timeout)
    emit(f"{stage_name}_HTTP={response.status_code}")
    if response.status_code != 200:
        emit(response.text[:4000])
        raise RuntimeError(f"{stage_name} img2img failed: HTTP {response.status_code}")
    return response.json()


def main() -> int:
    parser = argparse.ArgumentParser(description="Krea2 8K img2img upscale via Forge MultiDiffusion.")
    parser.add_argument("--input", required=True, help="Input image path.")
    parser.add_argument("--api", default=DEFAULT_API, help="Forge API base URL.")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT, help="Output root directory.")
    parser.add_argument("--prompt", default=None, help="Prompt. Defaults to PNG infotext prompt.")
    parser.add_argument("--negative-prompt", default=None, help="Negative prompt. Defaults to PNG infotext negative prompt.")
    parser.add_argument("--upscale-mode", default="two-stage", choices=["single-stage", "two-stage"], help="Upscale flow to run.")
    parser.add_argument("--long-edge", type=int, default=8192, help="Target long edge when width/height are not set.")
    parser.add_argument("--width", type=int, default=None, help="Explicit target width. Must be passed with --height.")
    parser.add_argument("--height", type=int, default=None, help="Explicit target height. Must be passed with --width.")
    parser.add_argument("--first-pass-long-edge", type=int, default=0, help="Intermediate long edge for --upscale-mode two-stage. 0 selects an automatic size.")
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--sampler", default="DPM++ SDE")
    parser.add_argument("--scheduler", default="Simple")
    parser.add_argument("--cfg", type=float, default=1.0)
    parser.add_argument("--distilled-cfg", type=float, default=1.15)
    parser.add_argument("--first-pass-denoise", type=float, default=0.22)
    parser.add_argument("--denoise", type=float, default=0.28)
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--tile-width", type=int, default=768)
    parser.add_argument("--tile-height", type=int, default=768)
    parser.add_argument("--tile-overlap", type=int, default=96)
    parser.add_argument("--tile-batch-size", type=int, default=1)
    parser.add_argument("--method", default="Mixture of Diffusers", choices=["MultiDiffusion", "Mixture of Diffusers"])
    parser.add_argument("--timeout", type=int, default=43200, help="HTTP timeout in seconds.")
    parser.add_argument("--return-image", action="store_true", help="Return image over API instead of relying on Forge save.")
    parser.add_argument("--dry-run", action="store_true", help="Write resized input and request preview, but do not call Forge.")
    args = parser.parse_args()
    validate_args(args)

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    with Image.open(input_path) as image:
        source = image.convert("RGB")
    target_w, target_h = target_size(source.width, source.height, args.long_edge, args.width, args.height)

    png_prompt, png_negative = prompt_from_png(input_path)
    prompt = args.prompt if args.prompt is not None else png_prompt
    negative_prompt = args.negative_prompt if args.negative_prompt is not None else png_negative
    if not prompt:
        raise ValueError("Prompt is empty. Pass --prompt or use a PNG with Forge infotext.")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / f"krea2_8k_img2img_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    emit(f"OUTPUT_DIR={output_dir}")
    emit(f"INPUT={input_path}")
    emit(f"UPSCALE_MODE={args.upscale_mode}")
    emit(f"TARGET={target_w}x{target_h} ({target_w * target_h / 1_000_000:.1f} MP)")

    if args.upscale_mode == "single-stage":
        resized = source.resize((target_w, target_h), Image.Resampling.LANCZOS)
        resized_path = output_dir / "resized_input.png"
        resized.save(resized_path)

        payload = build_img2img_payload(args, resized, prompt, negative_prompt, target_w, target_h, args.denoise, bool(args.return_image), True)
        preview_path = write_payload_preview(
            output_dir,
            "request_preview.json",
            payload,
            f"<base64 PNG omitted: {target_w}x{target_h}>",
            {"input_path": str(input_path), "resized_input": str(resized_path), "upscale_mode": args.upscale_mode},
        )

        emit(f"RESIZED_INPUT={resized_path}")
        emit(f"REQUEST_PREVIEW={preview_path}")

        if args.dry_run:
            emit("DRY_RUN=1")
            return 0

        started_at = time.time()
        data = post_img2img(args, payload, "FINAL")
    else:
        (stage1_w, stage1_h), _ = two_stage_sizes(source.width, source.height, target_w, target_h, args.first_pass_long_edge)
        emit(f"STAGE1_TARGET={stage1_w}x{stage1_h} ({stage1_w * stage1_h / 1_000_000:.1f} MP)")
        if args.first_pass_long_edge == 0:
            emit(f"STAGE1_LONG_EDGE=auto ({max(stage1_w, stage1_h)})")

        stage1_input = source.resize((stage1_w, stage1_h), Image.Resampling.LANCZOS)
        stage1_input_path = output_dir / "stage1_resized_input.png"
        stage1_input.save(stage1_input_path)

        stage1_payload = build_img2img_payload(args, stage1_input, prompt, negative_prompt, stage1_w, stage1_h, args.first_pass_denoise, True, False)
        stage1_preview_path = write_payload_preview(
            output_dir,
            "stage1_request_preview.json",
            stage1_payload,
            f"<base64 PNG omitted: {stage1_w}x{stage1_h}>",
            {
                "input_path": str(input_path),
                "stage1_resized_input": str(stage1_input_path),
                "upscale_mode": args.upscale_mode,
            },
        )

        stage2_preview_payload = build_img2img_payload(args, Image.new("RGB", (64, 64)), prompt, negative_prompt, target_w, target_h, args.denoise, bool(args.return_image), True)
        stage2_preview_path = write_payload_preview(
            output_dir,
            "stage2_request_preview.json",
            stage2_preview_payload,
            f"<stage1 result resized to {target_w}x{target_h}>",
            {"upscale_mode": args.upscale_mode},
        )

        emit(f"STAGE1_RESIZED_INPUT={stage1_input_path}")
        emit(f"STAGE1_REQUEST_PREVIEW={stage1_preview_path}")
        emit(f"STAGE2_REQUEST_PREVIEW={stage2_preview_path}")

        if args.dry_run:
            emit("DRY_RUN=1")
            return 0

        stage1_data = post_img2img(args, stage1_payload, "STAGE1")
        if not stage1_data.get("images"):
            raise RuntimeError("STAGE1 img2img returned no image.")

        stage1_image = decode_b64_image(stage1_data["images"][0])
        stage1_output_path = output_dir / "stage1_img2img.png"
        stage1_image.save(stage1_output_path)
        emit(f"STAGE1_IMAGE={stage1_output_path}")

        stage2_input = stage1_image.resize((target_w, target_h), Image.Resampling.LANCZOS)
        stage2_input_path = output_dir / "stage2_resized_input.png"
        stage2_input.save(stage2_input_path)
        emit(f"STAGE2_RESIZED_INPUT={stage2_input_path}")

        stage2_payload = build_img2img_payload(args, stage2_input, prompt, negative_prompt, target_w, target_h, args.denoise, bool(args.return_image), True)
        write_payload_preview(
            output_dir,
            "stage2_request_preview.json",
            stage2_payload,
            f"<base64 PNG omitted: {target_w}x{target_h}>",
            {"upscale_mode": args.upscale_mode, "stage2_resized_input": str(stage2_input_path)},
        )

        started_at = time.time()
        data = post_img2img(args, stage2_payload, "FINAL")

    info_path = output_dir / "response_info.txt"
    info_path.write_text(str(data.get("info", "")), encoding="utf-8")
    emit(f"INFO={info_path}")

    if args.return_image and data.get("images"):
        image = decode_b64_image(data["images"][0])
        output_path = output_dir / "krea2_8k_img2img.png"
        image.save(output_path)
        emit(f"IMAGE={output_path}")
    else:
        saved = recent_saved_images(Path(args.output_root), started_at)
        manifest = {
            "started_at": started_at,
            "latest_saved_images": [str(p) for p in saved[:10]],
            "output_dir": str(output_dir),
        }
        manifest_path = output_dir / "saved_images_manifest.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        emit(f"SAVED_IMAGES_MANIFEST={manifest_path}")
        if saved:
            emit(f"LATEST_SAVED_IMAGE={saved[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
