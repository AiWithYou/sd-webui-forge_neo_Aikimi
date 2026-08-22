"""Opt-in real-file and generation smoke for Anima 3.8B Qwen3.5."""

from __future__ import annotations

import base64
import io
import json
import os
from pathlib import Path
from urllib import request as urlrequest
import unittest

from PIL import Image, ImageStat
from PIL.PngImagePlugin import PngInfo
from safetensors import safe_open


LIVE = os.environ.get("ANIMA_38B_LIVE_API_TEST") == "1"
API = os.environ.get("ANIMA_38B_LIVE_API", "http://127.0.0.1:7862").rstrip(
    "/"
)
TIMEOUT = int(os.environ.get("ANIMA_38B_LIVE_TIMEOUT", "1800"))
ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = (
    ROOT
    / "models"
    / "Stable-diffusion"
    / "Anima-3.8B-int8-convrot.safetensors"
)
NATIVE_TEXT_ENCODER = (
    ROOT / "models" / "text_encoder" / "qwen_3_06b_base.safetensors"
)
QWEN35 = ROOT / "models" / "text_encoder" / "qwen35_4b.safetensors"
ADAPTER = (
    ROOT
    / "models"
    / "text_encoder"
    / "Anima-3.8B-expanded_adapter.safetensors"
)
VAE = ROOT / "models" / "VAE" / "qwen_image_vae.safetensors"
OUTPUT = Path(
    os.environ.get(
        "ANIMA_38B_LIVE_OUTPUT",
        str(ROOT / "outputs" / "anima_38b_smoke" / "anima_38b_smoke.png"),
    )
)


def post_json(endpoint: str, payload: dict) -> dict:
    body = json.dumps(payload).encode("utf-8")
    request = urlrequest.Request(
        f"{API}{endpoint}",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlrequest.urlopen(request, timeout=TIMEOUT) as response:
        return json.loads(response.read().decode("utf-8"))


def get_json(endpoint: str) -> dict:
    with urlrequest.urlopen(f"{API}{endpoint}", timeout=TIMEOUT) as response:
        return json.loads(response.read().decode("utf-8"))


@unittest.skipUnless(LIVE, "set ANIMA_38B_LIVE_API_TEST=1 for the live smoke")
class Anima38LiveApiTests(unittest.TestCase):
    def test_downloaded_assets_have_the_expected_mixed_precision_layouts(self):
        expected_sizes = {
            NATIVE_TEXT_ENCODER: 1_192_135_096,
            QWEN35: 4_779_016_600,
            ADAPTER: 88_131_712,
            VAE: 253_806_246,
        }
        for path, expected_size in expected_sizes.items():
            self.assertTrue(path.is_file(), path)
            self.assertEqual(path.stat().st_size, expected_size, path)
        self.assertTrue(CHECKPOINT.is_file(), CHECKPOINT)

        with safe_open(CHECKPOINT, framework="pt", device="cpu") as handle:
            metadata = handle.metadata() or {}
            keys = list(handle.keys())
            quantized = [key for key in keys if key.endswith(".comfy_quant")]
            blocks = {
                int(key.split(".")[2])
                for key in keys
                if key.startswith("net.blocks.")
            }
        self.assertEqual(metadata["forge.quantization.profile"], "anima38_main_attention_mlp_v1")
        self.assertEqual(metadata["forge.quantization.quantized_layers"], "520")
        self.assertEqual(len(quantized), 520)
        self.assertEqual(blocks, set(range(52)))

        dtype_tensors: dict[str, int] = {}
        dtype_numel: dict[str, int] = {}
        with safe_open(QWEN35, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                tensor = handle.get_slice(key)
                dtype = str(tensor.get_dtype())
                dtype_tensors[dtype] = dtype_tensors.get(dtype, 0) + 1
                numel = 1
                for dimension in tensor.get_shape():
                    numel *= dimension
                dtype_numel[dtype] = dtype_numel.get(dtype, 0) + numel
        self.assertEqual(dtype_tensors, {"BF16": 181, "F8_E4M3": 245})
        self.assertEqual(
            dtype_numel, {"BF16": 640_328_704, "F8_E4M3": 3_498_311_680}
        )

        with safe_open(ADAPTER, framework="pt", device="cpu") as handle:
            metadata = handle.metadata() or {}
        self.assertEqual(
            metadata["architecture"],
            "anima_progressive_qwen35_cross_adapter_v1",
        )

    def test_txt2img_uses_52_blocks_convrot_and_qwen35_adapter(self):
        width = int(os.environ.get("ANIMA_38B_LIVE_WIDTH", "512"))
        height = int(os.environ.get("ANIMA_38B_LIVE_HEIGHT", "512"))
        response = post_json(
            "/sdapi/v1/txt2img",
            {
                "prompt": (
                    "masterpiece, best quality, high quality, newest, "
                    "Description:\nA silver-haired astronomer stands on a moonlit "
                    "observatory terrace and points toward a bright comet."
                ),
                "negative_prompt": "worst quality, low quality, blurry",
                "seed": 20260823,
                "sampler_name": "Res Multistep",
                "scheduler": "Beta",
                "steps": int(os.environ.get("ANIMA_38B_LIVE_STEPS", "4")),
                "cfg_scale": 7.0,
                "distilled_cfg_scale": 3.0,
                "width": width,
                "height": height,
                "n_iter": 1,
                "batch_size": 1,
                "send_images": True,
                "save_images": False,
                "override_settings_restore_afterwards": False,
                "override_settings": {
                    "sd_model_checkpoint": CHECKPOINT.name,
                    "forge_additional_modules": [
                        VAE.name,
                        NATIVE_TEXT_ENCODER.name,
                    ],
                    "forge_unet_storage_dtype": "Automatic",
                },
                "alwayson_scripts": {
                    "Anima 3.8B": {
                        "args": [True, ADAPTER.name, 1.0, False, 1.0]
                    }
                },
            },
        )

        images = response.get("images") or []
        self.assertEqual(len(images), 1)
        encoded = images[0].split(",", 1)[-1]
        with Image.open(io.BytesIO(base64.b64decode(encoded))) as opened:
            image = opened.convert("RGB")
        self.assertEqual(image.size, (width, height))
        self.assertGreater(sum(ImageStat.Stat(image).var) / 3.0, 1.0)

        info = str(response.get("info", ""))
        self.assertIn("Anima 3.8B adapter", info)
        self.assertIn("Sampler: Res Multistep", info)
        self.assertIn("Schedule type: Beta", info)
        self.assertIn("Shift: 3.0", info)
        output_info = PngInfo()
        output_info.add_text("parameters", info)
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        image.save(OUTPUT, pnginfo=output_info)

        status = get_json("/sdapi/v1/forge-model-status")
        self.assertTrue(status["loaded"])
        self.assertEqual(
            status["architecture"], "backend.diffusion_engine.anima.Anima"
        )
        quantization = status["quantization"]
        self.assertEqual(quantization["convrot_layer_count"], 520)
        self.assertEqual(quantization["convrot_group_sizes"], [256])
        self.assertEqual(status["inspection_errors"], [])


if __name__ == "__main__":
    unittest.main()
