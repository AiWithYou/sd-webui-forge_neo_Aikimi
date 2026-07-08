from types import SimpleNamespace
import unittest

import numpy as np

from tools.krea2_tiled_refine import (
    axis_weights,
    target_size_from_args,
    tile_positions,
    tile_weight_mask,
    validate_args,
)


def valid_args(**overrides):
    values = {
        "width": None,
        "height": None,
        "long_edge": 5760,
        "scale": 1.5,
        "tile_size": 1024,
        "overlap": 192,
        "steps": 4,
        "timeout": 1800,
        "max_output_pixels": 24_000_000,
        "max_tile_pixels": 1_638_400,
        "denoise": 0.10,
        "cfg": 1.0,
        "distilled_cfg": 1.15,
        "progress_interval": 20.0,
        "no_progress_timeout": 600.0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class ValidationTests(unittest.TestCase):
    def test_accepts_default_arguments(self):
        validate_args(valid_args())

    def test_rejects_single_explicit_dimension(self):
        with self.assertRaisesRegex(ValueError, "--width and --height"):
            validate_args(valid_args(width=5760))

    def test_rejects_overlap_at_tile_size(self):
        with self.assertRaisesRegex(ValueError, "--overlap must be < --tile-size"):
            validate_args(valid_args(overlap=1024))

    def test_rejects_large_tile(self):
        with self.assertRaisesRegex(ValueError, "--tile-size exceeds"):
            validate_args(valid_args(tile_size=2048))

    def test_rejects_no_progress_timeout_without_progress_polling(self):
        with self.assertRaisesRegex(ValueError, "--no-progress-timeout requires"):
            validate_args(valid_args(no_progress_timeout=120, progress_interval=0))


class TargetSizeTests(unittest.TestCase):
    def test_uses_long_edge_when_set(self):
        self.assertEqual(
            target_size_from_args(3840, 1664, valid_args(long_edge=5760)), (5760, 2496)
        )

    def test_uses_scale_when_long_edge_is_not_set(self):
        self.assertEqual(
            target_size_from_args(3840, 1664, valid_args(long_edge=None, scale=1.5)),
            (5760, 2496),
        )

    def test_rejects_target_above_pixel_limit(self):
        with self.assertRaisesRegex(ValueError, "exceeding --max-output-pixels"):
            target_size_from_args(
                3840, 1664, valid_args(long_edge=7680, max_output_pixels=10_000_000)
            )


class TileLayoutTests(unittest.TestCase):
    def test_single_tile_when_length_fits(self):
        self.assertEqual(tile_positions(900, 1024, 192), [0])

    def test_positions_include_end(self):
        self.assertEqual(tile_positions(2496, 1024, 192), [0, 832, 1472])

    def test_axis_weights_feather_edges(self):
        weights = axis_weights(8, 3, 3)
        self.assertEqual(weights[0], 0.0)
        self.assertEqual(weights[-1], 0.0)
        self.assertEqual(weights[3], 1.0)

    def test_weight_mask_shape(self):
        x_positions = [0, 832, 1472]
        y_positions = [0, 832]
        mask = tile_weight_mask(x_positions, y_positions, 1, 0, 1024, 1024)
        self.assertEqual(mask.shape, (1024, 1024))
        self.assertTrue(np.all(mask >= 0))
        self.assertTrue(np.all(mask <= 1))


if __name__ == "__main__":
    unittest.main()
