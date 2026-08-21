import os
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from modules_forge.sensenova_u15_bridge import (
    DEFAULT_Q8_PATH,
    DEFAULT_SOURCE_PATH,
    MODE_EDIT,
    QUANT_Q8,
    SenseNovaRequest,
    inspect_runtime,
    run_generation,
)


ROOT = Path(__file__).resolve().parents[2]


@unittest.skipUnless(
    os.getenv("SENSENOVA_U15_LIVE_TEST") == "1", "opt-in real SenseNova Q8 GPU test"
)
class SenseNovaU15LiveTest(unittest.TestCase):
    def test_q8_two_image_edit_generates_nonuniform_png(self):
        status = inspect_runtime(
            DEFAULT_SOURCE_PATH, quantization=QUANT_Q8, gguf_checkpoint=DEFAULT_Q8_PATH
        )
        self.assertTrue(status.ready, " / ".join(status.messages))

        first = Image.new("RGB", (512, 512), (235, 242, 250))
        first_draw = ImageDraw.Draw(first)
        first_draw.ellipse((100, 80, 412, 392), fill=(30, 90, 190))
        second = Image.new("RGB", (512, 512), (250, 235, 220))
        second_draw = ImageDraw.Draw(second)
        second_draw.rectangle((112, 112, 400, 400), fill=(220, 70, 55))

        request = SenseNovaRequest(
            mode=MODE_EDIT,
            prompt=(
                "Use the centered blue circle from the first image as the main subject. "
                "Apply the warm red color palette from the second image while preserving a clean simple background."
            ),
            quantization=QUANT_Q8,
            gguf_checkpoint=str(DEFAULT_Q8_PATH),
            source_path=str(DEFAULT_SOURCE_PATH),
            input_images=(first, second),
            width=512,
            height=512,
            input_max_pixels=str(512 * 512),
            steps=1,
            cfg_scale=4.0,
            img_cfg_scale=1.0,
            timestep_shift=3.0,
            seed=42,
            vram_mode="low",
            attn_backend="sdpa",
            dtype="bfloat16",
        )
        updates = list(
            run_generation(
                request,
                output_directory=ROOT / "outputs" / "sensenova_u15_live_smoke",
                cache_directory=ROOT / "cache" / "sensenova_u15_live_smoke",
                log_directory=ROOT / "logs" / "sensenova_u15_live_smoke",
            )
        )
        complete = [update for update in updates if update["stage"] == "complete"]
        self.assertEqual(len(complete), 1)
        output_path = Path(complete[0]["path"])
        self.assertTrue(output_path.is_file())
        with Image.open(output_path) as image:
            self.assertEqual(image.size, (512, 512))
            values = np.asarray(image.convert("RGB"), dtype=np.float32)
        self.assertGreater(float(values.var()), 1.0)
        self.assertEqual(complete[0]["metadata"]["input_image_count"], 2)
        self.assertEqual(complete[0]["metadata"]["quantization"], QUANT_Q8)


if __name__ == "__main__":
    unittest.main()
