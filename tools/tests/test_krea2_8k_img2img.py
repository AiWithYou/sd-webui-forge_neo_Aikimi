from types import SimpleNamespace
import unittest

from modules_forge.krea2_upscale import (
    auto_first_pass_long_edge,
    target_size,
    two_stage_sizes,
)
from tools.krea2_8k_img2img import capped_diffusion_size, validate_args


def valid_args(**overrides):
    values = {
        "long_edge": 8192,
        "width": None,
        "height": None,
        "first_pass_long_edge": 0,
        "diffusion_long_edge_cap": 0,
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
        "progress_interval": 30.0,
        "no_progress_timeout": 0.0,
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

    def test_rejects_negative_first_pass_long_edge(self):
        with self.assertRaisesRegex(ValueError, "--first-pass-long-edge"):
            validate_args(valid_args(first_pass_long_edge=-64))

    def test_rejects_negative_diffusion_long_edge_cap(self):
        with self.assertRaisesRegex(ValueError, "--diffusion-long-edge-cap"):
            validate_args(valid_args(diffusion_long_edge_cap=-1))

    def test_rejects_negative_progress_interval(self):
        with self.assertRaisesRegex(ValueError, "--progress-interval"):
            validate_args(valid_args(progress_interval=-0.1))

    def test_rejects_negative_no_progress_timeout(self):
        with self.assertRaisesRegex(ValueError, "--no-progress-timeout"):
            validate_args(valid_args(no_progress_timeout=-0.1))

    def test_rejects_no_progress_timeout_without_progress_polling(self):
        with self.assertRaisesRegex(ValueError, "--no-progress-timeout requires"):
            validate_args(valid_args(no_progress_timeout=120, progress_interval=0))


class DiffusionCapTests(unittest.TestCase):
    def test_returns_target_when_cap_is_disabled(self):
        self.assertEqual(capped_diffusion_size(1254, 1254, 6144, 6144, 0), (6144, 6144))

    def test_returns_target_when_target_is_below_cap(self):
        self.assertEqual(
            capped_diffusion_size(1254, 1254, 4096, 4096, 6144), (4096, 4096)
        )

    def test_caps_diffusion_size_to_requested_long_edge(self):
        self.assertEqual(
            capped_diffusion_size(1254, 1254, 6144, 6144, 4096), (4096, 4096)
        )

    def test_rejects_cap_below_source_long_edge(self):
        with self.assertRaisesRegex(ValueError, "--diffusion-long-edge-cap"):
            capped_diffusion_size(1254, 1254, 6144, 6144, 1024)


class TwoStageSizeTests(unittest.TestCase):
    def test_uses_final_aspect_ratio_for_intermediate_size(self):
        self.assertEqual(
            two_stage_sizes(1000, 500, 8192, 8192, 4096), ((4096, 4096), (8192, 8192))
        )

    def test_uses_auto_intermediate_size(self):
        self.assertEqual(auto_first_pass_long_edge(1000, 500, 8192, 4096), 2880)
        self.assertEqual(
            two_stage_sizes(1000, 500, 8192, 4096, 0), ((2880, 1408), (8192, 4096))
        )

    def test_rejects_auto_when_no_intermediate_multiple_exists(self):
        with self.assertRaisesRegex(ValueError, "too close"):
            two_stage_sizes(8150, 4000, 8192, 4032, 0)

    def test_rejects_first_pass_smaller_than_source(self):
        with self.assertRaisesRegex(ValueError, "first pass long edge"):
            two_stage_sizes(5000, 3000, 8192, 4928, 4096)

    def test_rejects_first_pass_at_final_size(self):
        with self.assertRaisesRegex(ValueError, "first pass long edge"):
            two_stage_sizes(1000, 500, 8192, 4096, 8192)


if __name__ == "__main__":
    unittest.main()
