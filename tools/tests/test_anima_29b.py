from types import SimpleNamespace
import unittest

from modules_forge.packages.huggingface_guess.detection import detect_unet_config


def anima_state_dict(block_count: int, key_prefix: str = "") -> dict:
    state_dict = {
        f"{key_prefix}llm_adapter.blocks.0.cross_attn.q_proj.weight": SimpleNamespace(
            shape=(1024, 1024)
        ),
        f"{key_prefix}x_embedder.proj.1.weight": SimpleNamespace(shape=(2048, 68)),
    }
    for block in range(block_count):
        state_dict[f"{key_prefix}blocks.{block}.mlp.layer1.weight"] = SimpleNamespace(
            shape=(8192, 2048)
        )
    return state_dict


class Anima29BDetectionTests(unittest.TestCase):
    def test_detects_original_anima_as_28_blocks(self):
        config = detect_unet_config(anima_state_dict(28), "")

        self.assertEqual(config["image_model"], "anima")
        self.assertEqual(config["num_blocks"], 28)

    def test_detects_anima_29b_as_40_blocks(self):
        config = detect_unet_config(anima_state_dict(40), "")

        self.assertEqual(config["image_model"], "anima")
        self.assertEqual(config["num_blocks"], 40)
        self.assertEqual(config["model_channels"], 2048)
        self.assertEqual(config["in_channels"], 16)

    def test_detects_anima_38b_as_52_blocks(self):
        config = detect_unet_config(anima_state_dict(52), "")

        self.assertEqual(config["image_model"], "anima")
        self.assertEqual(config["num_blocks"], 52)
        self.assertEqual(config["model_channels"], 2048)
        self.assertEqual(config["in_channels"], 16)

    def test_counts_blocks_below_a_checkpoint_prefix(self):
        prefix = "model.diffusion_model."

        config = detect_unet_config(anima_state_dict(40, prefix), prefix)

        self.assertEqual(config["num_blocks"], 40)


if __name__ == "__main__":
    unittest.main()
