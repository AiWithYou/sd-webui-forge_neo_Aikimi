import importlib.util
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
WORKER_PATH = ROOT / "tools" / "sensenova_u15_worker.py"
SPEC = importlib.util.spec_from_file_location("sensenova_u15_worker", WORKER_PATH)
worker = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(worker)


class SenseNovaWorkerTests(unittest.TestCase):
    def test_auto_input_budget_obeys_final_multi_reference_contract(self):
        self.assertEqual(worker._auto_input_max_pixels(1), 2048 * 2048)
        self.assertEqual(worker._auto_input_max_pixels(2), 2048 * 2048)
        self.assertEqual(worker._auto_input_max_pixels(4), 2048 * 2048)
        self.assertEqual(worker._auto_input_max_pixels(16), 1024 * 1024)
        self.assertEqual(worker._auto_input_max_pixels(64), 512 * 512)
        self.assertEqual(worker._auto_input_max_pixels(100), 512 * 512)
        self.assertEqual(
            worker._effective_input_max_pixels(64, 2048 * 2048), 512 * 512
        )
        self.assertEqual(
            worker._effective_input_max_pixels(2, 1024 * 1024), 1024 * 1024
        )

    def test_explicit_output_size_does_not_call_resize(self):
        payload = {"width": 1024, "height": 1536}
        size = worker._resolve_output_size(
            payload, [], lambda **_: self.fail("unexpected resize")
        )
        self.assertEqual(size, (1024, 1536))

    def test_automatic_output_size_uses_original_first_reference_ratio(self):
        calls = []

        def smart_resize(**kwargs):
            calls.append(kwargs)
            return 1536, 2752

        size = worker._resolve_output_size(
            {"target_pixels": 2048 * 2048},
            [(1920, 1080)],
            smart_resize,
        )

        self.assertEqual(size, (2752, 1536))
        self.assertEqual(calls[0]["width"], 1920)
        self.assertEqual(calls[0]["height"], 1080)

    def test_aspect_fit_keeps_subject_geometry_inside_model_canvas(self):
        content_size = worker._aspect_fit_size(1600, 1200, 576, 416)
        self.assertEqual(content_size, (555, 416))
        self.assertAlmostEqual(
            content_size[0] / content_size[1], 4 / 3, delta=0.001
        )

        source = Image.new("RGB", (4, 2))
        for y in range(source.height):
            for x in range(source.width):
                source.putpixel((x, y), (x * 60, y * 120, 20))
        prepared = worker._resize_with_edge_padding(source, 4, 4)

        self.assertEqual(prepared.size, (4, 4))
        self.assertEqual(
            [prepared.getpixel((x, 0)) for x in range(prepared.width)],
            [prepared.getpixel((x, 1)) for x in range(prepared.width)],
        )
        self.assertEqual(
            [prepared.getpixel((x, 3)) for x in range(prepared.width)],
            [prepared.getpixel((x, 2)) for x in range(prepared.width)],
        )

    def test_load_images_returns_original_sizes_before_padding(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "reference.png"
            Image.new("RGB", (160, 120), (20, 40, 60)).save(path)
            images, original_sizes = worker._load_images(
                [str(path)],
                smart_resize=lambda **_: (416, 576),
                input_max_pixels=512 * 512,
            )

        self.assertEqual(original_sizes, [(160, 120)])
        self.assertEqual([image.size for image in images], [(576, 416)])

    def test_edit_payload_accepts_multiple_existing_inputs(self):
        with tempfile.TemporaryDirectory() as temp:
            first = Path(temp) / "first.png"
            second = Path(temp) / "second.png"
            checkpoint = Path(temp) / "model.safetensors"
            Image.new("RGB", (16, 16), (255, 0, 0)).save(first)
            Image.new("RGB", (16, 16), (0, 0, 255)).save(second)
            checkpoint.write_bytes(b"test")
            worker._validate_payload(
                {
                    "mode": "edit",
                    "prompt": "combine",
                    "input_images": [str(first), str(second)],
                    "quantization": "int8_convrot",
                    "model_path": worker.FINAL_MODEL_ID,
                    "checkpoint_revision": worker.CHECKPOINT_REVISION,
                    "checkpoint": str(checkpoint),
                    "width": 1024,
                    "height": 1024,
                    "input_max_pixels": 512 * 512,
                    "vram_mode": "low",
                }
            )

    def test_low_vram_payload_rejects_uncapped_four_megapixel_edit(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            image = root / "reference.png"
            checkpoint = root / "model.safetensors"
            Image.new("RGB", (16, 16)).save(image)
            checkpoint.write_bytes(b"test")
            payload = {
                "mode": "edit",
                "prompt": "combine",
                "input_images": [str(image), str(image)],
                "quantization": "int8_convrot",
                "model_path": worker.FINAL_MODEL_ID,
                "checkpoint_revision": worker.CHECKPOINT_REVISION,
                "checkpoint": str(checkpoint),
                "width": 1664,
                "height": 2496,
                "input_max_pixels": "auto",
                "vram_mode": "low",
            }
            with self.assertRaisesRegex(RuntimeError, "24 GB safe profile"):
                worker._validate_payload(payload)

            payload["vram_mode"] = "unrestricted"
            worker._validate_payload(payload)

            payload.update(
                {
                    "vram_mode": "low",
                    "width": 2048,
                    "height": 2048,
                    "input_max_pixels": 512 * 512,
                    "input_images": [str(image), str(image)],
                }
            )
            worker._validate_payload(payload)

            payload.update(
                {
                    "vram_mode": "low",
                    "width": 1024,
                    "height": 1024,
                    "input_max_pixels": 512 * 512,
                    "input_images": [str(image), str(image), str(image)],
                }
            )
            with self.assertRaisesRegex(RuntimeError, "at most 2 references"):
                worker._validate_payload(payload)

    def test_normalized_tensor_is_converted_to_rgb_image(self):
        batch = torch.tensor([[[[-1.0]], [[0.0]], [[1.0]]]])
        image = worker._to_pil(batch)[0]
        self.assertEqual(image.mode, "RGB")
        self.assertEqual(image.getpixel((0, 0)), (0, 128, 255))

    def test_unit_range_nhwc_tensor_is_converted_to_rgb_image(self):
        batch = torch.tensor([[[[0.0, 0.5, 1.0]]]])
        image = worker._to_pil(batch)[0]
        self.assertEqual(image.mode, "RGB")
        self.assertEqual(image.getpixel((0, 0)), (0, 128, 255))


if __name__ == "__main__":
    unittest.main()
