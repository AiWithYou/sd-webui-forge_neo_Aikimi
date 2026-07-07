from types import SimpleNamespace
import unittest

from modules_forge.krea2_upscale import target_size, two_stage_sizes
from tools.krea2_8k_img2img import validate_args


def valid_args(**overrides):
    values = {
        "long_edge": 8192,
        "width": None,
        "height": None,
        "first_pass_long_edge": 4096,
        "steps": 12,
        "tile_width": 768,
        "tile_height": 768,
        "tile_overlap": 96,
        "tile_batch_size": 1,
        "timeout": 43200,
        "denoise": 0.28,
        "first_pass_denoise": 0.22,
        "cfg": 1.0,
        "distilled_cfg": 1.15,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class TargetSizeTests(unittest.TestCase):
    def test_preserves_landscape_ratio_at_long_edge(self):
        self.assertEqual(target_size(1000, 500, 8192, None, None), (8192, 4096))

    def test_preserves_portrait_ratio_at_long_edge(self):
        self.assertEqual(target_size(500, 1000, 8192, None, None), (4096, 8192))

    def test_rounds_explicit_size_to_multiple_of_64(self):
        self.assertEqual(target_size(3000, 2000, 8192, 8010, 6001), (8000, 6016))

    def test_rejects_single_explicit_dimension(self):
        with self.assertRaisesRegex(ValueError, "--width and --height"):
            target_size(3000, 2000, 8192, 8192, None)


class ArgumentValidationTests(unittest.TestCase):
    def test_accepts_default_arguments(self):
        validate_args(valid_args())

    def test_rejects_non_positive_steps(self):
        with self.assertRaisesRegex(ValueError, "--steps"):
            validate_args(valid_args(steps=0))

    def test_rejects_negative_tile_overlap(self):
        with self.assertRaisesRegex(ValueError, "--tile-overlap"):
            validate_args(valid_args(tile_overlap=-1))

    def test_rejects_denoise_outside_unit_interval(self):
        with self.assertRaisesRegex(ValueError, "--denoise"):
            validate_args(valid_args(denoise=1.01))

    def test_rejects_first_pass_denoise_outside_unit_interval(self):
        with self.assertRaisesRegex(ValueError, "--first-pass-denoise"):
            validate_args(valid_args(first_pass_denoise=-0.01))

    def test_rejects_single_explicit_dimension(self):
        with self.assertRaisesRegex(ValueError, "--width and --height"):
            validate_args(valid_args(width=8192))


class TwoStageSizeTests(unittest.TestCase):
    def test_uses_final_aspect_ratio_for_intermediate_size(self):
        self.assertEqual(two_stage_sizes(1000, 500, 8192, 8192, 4096), ((4096, 4096), (8192, 8192)))

    def test_rejects_first_pass_smaller_than_source(self):
        with self.assertRaisesRegex(ValueError, "first pass long edge"):
            two_stage_sizes(5000, 3000, 8192, 4928, 4096)

    def test_rejects_first_pass_at_final_size(self):
        with self.assertRaisesRegex(ValueError, "first pass long edge"):
            two_stage_sizes(1000, 500, 8192, 4096, 8192)


if __name__ == "__main__":
    unittest.main()
