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

from modules.color_flatten import FAST_MODE
from modules.scripts_postprocessing import PostprocessedImage
from scripts.postprocessing_color_flatten import ScriptPostprocessingColorFlatten
from scripts.postprocessing_color_mura_checker import ScriptPostprocessingColorMuraChecker


class ColorPostprocessingDefaultsTests(unittest.TestCase):
    def test_color_flatten_runs_by_default_with_stronger_fast_settings(self):
        pp = PostprocessedImage(Image.new("RGB", (16, 16), (120, 100, 90)))

        ScriptPostprocessingColorFlatten().process(pp)

        self.assertEqual(pp.info["Color Flatten"], FAST_MODE)
        self.assertEqual(pp.info["Color Flatten strength"], 0.8)
        self.assertTrue(pp.info["Color Flatten edge protect"])

    def test_color_mura_checker_runs_by_default_with_review_outputs(self):
        pp = PostprocessedImage(Image.new("RGB", (16, 16), (120, 100, 90)))

        ScriptPostprocessingColorMuraChecker().process(pp)

        self.assertIn("Color mura summary", pp.info)
        self.assertEqual(len(pp.extra_images), 2)
        self.assertEqual(pp.extra_images[0].nametags[-1], "mura-overlay")
        self.assertEqual(pp.extra_images[1].nametags[-1], "mura-heatmap")


if __name__ == "__main__":
    unittest.main()
