import argparse
import base64
import io
import json
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import requests
from PIL import Image

from modules_forge.krea2_upscale import (
    require_positive_int,
    size_from_long_edge,
    target_size,
    two_stage_sizes,
)


DEFAULT_API = "http://127.0.0.1:7861"
DEFAULT_OUTPUT_ROOT = "output"


def emit(message: str):
    sys.stdout.write(f"{message}\n")
    sys.stdout.flush()


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

    if args.diffusion_long_edge_cap < 0:
        raise ValueError("--diffusion-long-edge-cap must be >= 0.")
    if args.first_pass_long_edge < 0:
        raise ValueError("--first-pass-long-edge must be >= 0.")
    if args.tile_overlap < 0:
        raise ValueError("--tile-overlap must be >= 0.")
    if args.progress_interval < 0:
        raise ValueError("--progress-interval must be >= 0.")
    if args.no_progress_timeout < 0:
        raise ValueError("--no-progress-timeout must be >= 0.")
    if args.no_progress_timeout > 0 and args.progress_interval <= 0:
        raise ValueError("--no-progress-timeout requires --progress-interval > 0.")
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


def capped_diffusion_size(
    source_width: int,
    source_height: int,
    target_width: int,
    target_height: int,
    cap_long_edge: int,
) -> tuple[int, int]:
    if cap_long_edge == 0 or max(target_width, target_height) <= cap_long_edge:
        return target_width, target_height

    source_long_edge = max(source_width, source_height)
    if cap_long_edge < source_long_edge:
        raise ValueError(
            "--diffusion-long-edge-cap must be >= source long edge when it caps the diffusion pass."
        )
    return size_from_long_edge(target_width, target_height, cap_long_edge)


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


def build_img2img_payload(
    args,
    image: Image.Image,
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    denoise: float,
    send_images: bool,
    save_images: bool,
) -> dict:
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


def write_payload_preview(
    output_dir: Path,
    name: str,
    payload: dict,
    init_image_description: str,
    extra: dict | None = None,
) -> Path:
    preview = dict(payload)
    preview["init_images"] = [init_image_description]
    if extra:
        preview.update(extra)
    preview_path = output_dir / name
    preview_path.write_text(
        json.dumps(preview, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return preview_path


def poll_progress(args, stage_name: str, stop_event: threading.Event):
    last_error = ""
    last_progress_key = None
    last_progress_at = time.monotonic()
    while not stop_event.wait(args.progress_interval):
        try:
            response = requests.get(
                f"{args.api}/sdapi/v1/progress",
                params={"skip_current_image": "true"},
                timeout=5,
            )
            response.raise_for_status()
            data = response.json()
        except (requests.RequestException, ValueError) as exc:
            error = f"{type(exc).__name__}: {exc}"
            if error != last_error:
                emit(f"{stage_name}_PROGRESS_ERROR={error}")
                last_error = error
            continue

        state = data.get("state") or {}
        progress = data.get("progress")
        progress_key = (
            progress,
            state.get("sampling_step"),
            state.get("sampling_steps"),
            state.get("job_timestamp"),
        )
        if progress_key != last_progress_key:
            last_progress_key = progress_key
            last_progress_at = time.monotonic()
        elif (
            args.no_progress_timeout > 0
            and time.monotonic() - last_progress_at >= args.no_progress_timeout
        ):
            emit(f"{stage_name}_NO_PROGRESS_TIMEOUT={args.no_progress_timeout}")
            try:
                requests.post(f"{args.api}/sdapi/v1/interrupt", timeout=5)
            except requests.RequestException as exc:
                emit(f"{stage_name}_INTERRUPT_ERROR={type(exc).__name__}: {exc}")
            stop_event.set()
            return

        if isinstance(progress, (int, float)):
            progress_text = f"{progress:.4f}"
        else:
            progress_text = str(progress)
        emit(
            f"{stage_name}_PROGRESS={progress_text} "
            f"STEP={state.get('sampling_step')}/{state.get('sampling_steps')} "
            f"JOB={state.get('job')} ETA={data.get('eta_relative')}"
        )


def post_img2img(args, payload: dict, stage_name: str) -> dict:
    stop_event = None
    progress_thread = None
    if args.progress_interval > 0:
        stop_event = threading.Event()
        progress_thread = threading.Thread(
            target=poll_progress, args=(args, stage_name, stop_event), daemon=True
        )
        progress_thread.start()
    try:
        response = requests.post(
            f"{args.api}/sdapi/v1/img2img", json=payload, timeout=args.timeout
        )
    finally:
        if stop_event is not None:
            stop_event.set()
        if progress_thread is not None:
            progress_thread.join(timeout=1)
    emit(f"{stage_name}_HTTP={response.status_code}")
    if response.status_code != 200:
        emit(response.text[:4000])
        raise RuntimeError(f"{stage_name} img2img failed: HTTP {response.status_code}")
    return response.json()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Krea2 8K img2img upscale via Forge MultiDiffusion."
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
        "--upscale-mode",
        default="two-stage",
        choices=["single-stage", "two-stage"],
        help="Upscale flow to run.",
    )
    parser.add_argument(
        "--long-edge",
        type=int,
        default=8192,
        help="Target long edge when width/height are not set.",
    )
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
        "--first-pass-long-edge",
        type=int,
        default=0,
        help="Intermediate long edge for --upscale-mode two-stage. 0 selects an automatic size.",
    )
    parser.add_argument(
        "--diffusion-long-edge-cap",
        type=int,
        default=0,
        help="Maximum long edge used for the final diffusion pass. 0 disables the cap. When capped below the output target, the returned diffusion image is resized to the requested target locally.",
    )
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
    parser.add_argument(
        "--method",
        default="Mixture of Diffusers",
        choices=["MultiDiffusion", "Mixture of Diffusers"],
    )
    parser.add_argument(
        "--timeout", type=int, default=43200, help="HTTP timeout in seconds."
    )
    parser.add_argument(
        "--progress-interval",
        type=float,
        default=30.0,
        help="Seconds between Forge progress polling messages during API calls. 0 disables progress polling.",
    )
    parser.add_argument(
        "--no-progress-timeout",
        type=float,
        default=0.0,
        help="Interrupt Forge if progress does not change for this many seconds. 0 disables this watchdog.",
    )
    parser.add_argument(
        "--return-image",
        action="store_true",
        help="Return image over API instead of relying on Forge save.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write resized input and request preview, but do not call Forge.",
    )
    args = parser.parse_args()
    validate_args(args)

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    with Image.open(input_path) as image:
        source = image.convert("RGB")
    target_w, target_h = target_size(
        source.width, source.height, args.long_edge, args.width, args.height
    )
    diffusion_w, diffusion_h = capped_diffusion_size(
        source.width, source.height, target_w, target_h, args.diffusion_long_edge_cap
    )
    needs_final_resize = (diffusion_w, diffusion_h) != (target_w, target_h)

    png_prompt, png_negative = prompt_from_png(input_path)
    prompt = args.prompt if args.prompt is not None else png_prompt
    negative_prompt = (
        args.negative_prompt if args.negative_prompt is not None else png_negative
    )
    if not prompt:
        raise ValueError(
            "Prompt is empty. Pass --prompt or use a PNG with Forge infotext."
        )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / f"krea2_8k_img2img_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    emit(f"OUTPUT_DIR={output_dir}")
    emit(f"INPUT={input_path}")
    emit(f"UPSCALE_MODE={args.upscale_mode}")
    emit(f"TARGET={target_w}x{target_h} ({target_w * target_h / 1_000_000:.1f} MP)")
    if needs_final_resize:
        emit(
            f"DIFFUSION_TARGET={diffusion_w}x{diffusion_h} ({diffusion_w * diffusion_h / 1_000_000:.1f} MP)"
        )
        emit(f"FINAL_RESIZE_TARGET={target_w}x{target_h}")

    if args.upscale_mode == "single-stage":
        resized = source.resize((diffusion_w, diffusion_h), Image.Resampling.LANCZOS)
        resized_path = output_dir / "resized_input.png"
        resized.save(resized_path)

        payload = build_img2img_payload(
            args,
            resized,
            prompt,
            negative_prompt,
            diffusion_w,
            diffusion_h,
            args.denoise,
            bool(args.return_image or needs_final_resize),
            True,
        )
        preview_path = write_payload_preview(
            output_dir,
            "request_preview.json",
            payload,
            f"<base64 PNG omitted: {diffusion_w}x{diffusion_h}>",
            {
                "input_path": str(input_path),
                "resized_input": str(resized_path),
                "upscale_mode": args.upscale_mode,
                "target_output": f"{target_w}x{target_h}",
                "diffusion_target": f"{diffusion_w}x{diffusion_h}",
                "final_resize": needs_final_resize,
            },
        )

        emit(f"RESIZED_INPUT={resized_path}")
        emit(f"REQUEST_PREVIEW={preview_path}")

        if args.dry_run:
            emit("DRY_RUN=1")
            return 0

        started_at = time.time()
        data = post_img2img(args, payload, "FINAL")
    else:
        (stage1_w, stage1_h), _ = two_stage_sizes(
            source.width,
            source.height,
            diffusion_w,
            diffusion_h,
            args.first_pass_long_edge,
        )
        emit(
            f"STAGE1_TARGET={stage1_w}x{stage1_h} ({stage1_w * stage1_h / 1_000_000:.1f} MP)"
        )
        if args.first_pass_long_edge == 0:
            emit(f"STAGE1_LONG_EDGE=auto ({max(stage1_w, stage1_h)})")

        stage1_input = source.resize((stage1_w, stage1_h), Image.Resampling.LANCZOS)
        stage1_input_path = output_dir / "stage1_resized_input.png"
        stage1_input.save(stage1_input_path)

        stage1_payload = build_img2img_payload(
            args,
            stage1_input,
            prompt,
            negative_prompt,
            stage1_w,
            stage1_h,
            args.first_pass_denoise,
            True,
            False,
        )
        stage1_preview_path = write_payload_preview(
            output_dir,
            "stage1_request_preview.json",
            stage1_payload,
            f"<base64 PNG omitted: {stage1_w}x{stage1_h}>",
            {
                "input_path": str(input_path),
                "stage1_resized_input": str(stage1_input_path),
                "upscale_mode": args.upscale_mode,
                "target_output": f"{target_w}x{target_h}",
                "diffusion_target": f"{diffusion_w}x{diffusion_h}",
                "final_resize": needs_final_resize,
            },
        )

        stage2_preview_payload = build_img2img_payload(
            args,
            Image.new("RGB", (64, 64)),
            prompt,
            negative_prompt,
            diffusion_w,
            diffusion_h,
            args.denoise,
            bool(args.return_image or needs_final_resize),
            True,
        )
        stage2_preview_path = write_payload_preview(
            output_dir,
            "stage2_request_preview.json",
            stage2_preview_payload,
            f"<stage1 result resized to {diffusion_w}x{diffusion_h}>",
            {
                "upscale_mode": args.upscale_mode,
                "target_output": f"{target_w}x{target_h}",
                "diffusion_target": f"{diffusion_w}x{diffusion_h}",
                "final_resize": needs_final_resize,
            },
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

        stage2_input = stage1_image.resize(
            (diffusion_w, diffusion_h), Image.Resampling.LANCZOS
        )
        stage2_input_path = output_dir / "stage2_resized_input.png"
        stage2_input.save(stage2_input_path)
        emit(f"STAGE2_RESIZED_INPUT={stage2_input_path}")

        stage2_payload = build_img2img_payload(
            args,
            stage2_input,
            prompt,
            negative_prompt,
            diffusion_w,
            diffusion_h,
            args.denoise,
            bool(args.return_image or needs_final_resize),
            True,
        )
        write_payload_preview(
            output_dir,
            "stage2_request_preview.json",
            stage2_payload,
            f"<base64 PNG omitted: {diffusion_w}x{diffusion_h}>",
            {
                "upscale_mode": args.upscale_mode,
                "stage2_resized_input": str(stage2_input_path),
                "target_output": f"{target_w}x{target_h}",
                "diffusion_target": f"{diffusion_w}x{diffusion_h}",
                "final_resize": needs_final_resize,
            },
        )

        started_at = time.time()
        data = post_img2img(args, stage2_payload, "FINAL")

    info_path = output_dir / "response_info.txt"
    info_path.write_text(str(data.get("info", "")), encoding="utf-8")
    emit(f"INFO={info_path}")

    returned_images = data.get("images") or []
    if needs_final_resize:
        if not returned_images:
            raise RuntimeError(
                "FINAL img2img returned no image. --diffusion-long-edge-cap requires a returned image for local final resize."
            )
        image = decode_b64_image(returned_images[0])
        diffusion_output_path = output_dir / "final_diffusion_img2img.png"
        image.save(diffusion_output_path)
        output_path = output_dir / "krea2_8k_img2img.png"
        image.resize((target_w, target_h), Image.Resampling.LANCZOS).save(output_path)
        emit(f"FINAL_DIFFUSION_IMAGE={diffusion_output_path}")
        emit(f"IMAGE={output_path}")
    elif args.return_image and returned_images:
        image = decode_b64_image(returned_images[0])
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
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        emit(f"SAVED_IMAGES_MANIFEST={manifest_path}")
        if saved:
            emit(f"LATEST_SAVED_IMAGE={saved[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
