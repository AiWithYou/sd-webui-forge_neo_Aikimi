import unittest

from PIL import Image, ImageDraw

from tools.build_aikimi_motion_asset import encoded_indices, extract_components


class AikimiMotionBuilderTests(unittest.TestCase):
    def green_strip(self, centers):
        strip = Image.new("RGBA", (400, 160), (0, 255, 0, 255))
        draw = ImageDraw.Draw(strip)
        for center in centers:
            draw.rounded_rectangle(
                (center - 24, 28, center + 24, 142),
                radius=12,
                fill=(240, 244, 250, 255),
            )
        return strip

    def test_ping_pong_indices_do_not_duplicate_endpoints(self):
        self.assertEqual(encoded_indices(4, "ping-pong"), [0, 1, 2, 3, 2, 1])
        self.assertEqual(encoded_indices(5, "once"), [0, 1, 2, 3, 4])

    def test_component_extraction_accepts_one_pose_per_slot(self):
        entries, _, diagnostics = extract_components(
            self.green_strip((50, 150, 250, 350)),
            frame_count=4,
            alpha_cutoff=16 / 255,
        )

        self.assertEqual(len(entries), 4)
        self.assertEqual(len(diagnostics["principal_centers_x"]), 4)

    def test_component_extraction_rejects_missing_nominal_slot(self):
        with self.assertRaisesRegex(ValueError, "outside its nominal slot"):
            extract_components(
                self.green_strip((25, 75, 250, 350)),
                frame_count=4,
                alpha_cutoff=16 / 255,
            )

    def test_component_extraction_rejects_connected_poses(self):
        strip = self.green_strip((50, 150, 250, 350))
        draw = ImageDraw.Draw(strip)
        draw.rectangle((50, 80, 350, 90), fill=(240, 244, 250, 255))

        with self.assertRaisesRegex(ValueError, "foreground components"):
            extract_components(strip, frame_count=4, alpha_cutoff=16 / 255)


if __name__ == "__main__":
    unittest.main()
