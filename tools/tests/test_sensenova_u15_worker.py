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
                }
            )

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
