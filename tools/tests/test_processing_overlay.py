import sys
import unittest
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
PACKAGES = ROOT / "modules_forge" / "packages"
if str(PACKAGES) not in sys.path:
    sys.path.insert(0, str(PACKAGES))

_ORIGINAL_ARGV = sys.argv[:]
sys.argv = [sys.argv[0]]

import modules.shared_init as shared_init

shared_init.initialize()
sys.argv = _ORIGINAL_ARGV

from modules.processing import _INPAINT_FULL_RES_OVERLAY_MASK, apply_overlay


class InpaintOverlayTests(unittest.TestCase):
    def test_crop_overlay_keeps_original_pixels_outside_mask(self):
        generated = Image.new("RGB", (4, 4), (255, 0, 0))
        overlay = Image.new("RGB", (8, 8), (0, 0, 255))

        crop_mask = Image.new("L", (4, 4), 0)
        crop_mask.putpixel((1, 1), 255)
        overlay.info[_INPAINT_FULL_RES_OVERLAY_MASK] = crop_mask

        result, original_denoised = apply_overlay(generated, (2, 2, 4, 4), overlay)

        self.assertEqual(result.size, (8, 8))
        self.assertEqual(original_denoised.size, (8, 8))
        self.assertEqual(result.getpixel((0, 0)), (0, 0, 255))
        self.assertEqual(result.getpixel((2, 2)), (0, 0, 255))
        self.assertEqual(result.getpixel((3, 3)), (255, 0, 0))

    def test_crop_overlay_rejects_mask_size_mismatch(self):
        generated = Image.new("RGB", (4, 4), (255, 0, 0))
        overlay = Image.new("RGB", (8, 8), (0, 0, 255))
        overlay.info[_INPAINT_FULL_RES_OVERLAY_MASK] = Image.new("L", (3, 4), 255)

        with self.assertRaisesRegex(ValueError, "does not match paste size"):
            apply_overlay(generated, (2, 2, 4, 4), overlay)


if __name__ == "__main__":
    unittest.main()
