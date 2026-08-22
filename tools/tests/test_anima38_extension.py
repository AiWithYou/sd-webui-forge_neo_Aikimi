import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch

from safetensors.torch import save_file
import torch


ROOT = Path(__file__).resolve().parents[2]
EXTENSION = ROOT / "extensions-builtin" / "anima-3-8b"
if str(EXTENSION) not in sys.path:
    sys.path.insert(0, str(EXTENSION))

from anima3b.files import ARCHITECTURE, adapters, qwen35_models, tokenizer_dir
from anima3b.runtime import ANIMA38_BLOCK_COUNT, Anima3BRuntime


def fake_anima(block_count: int):
    clip = object()
    diffusion_model = SimpleNamespace(blocks=[object()] * block_count)
    unet = SimpleNamespace(model=SimpleNamespace(diffusion_model=diffusion_model))
    return SimpleNamespace(
        text_processing_engine_anima=object(),
        forge_objects=SimpleNamespace(clip=clip, unet=unet),
    )


class Anima38ExtensionTests(unittest.TestCase):
    def test_bundled_qwen35_tokenizer_is_available_offline(self):
        directory = tokenizer_dir()

        self.assertEqual(directory, EXTENSION / "qwen35_tokenizer")
        config = json.loads(
            (directory / "tokenizer_config.json").read_text(encoding="utf-8")
        )
        self.assertIn("tokenizer_class", config)

    def test_adapter_discovery_uses_architecture_metadata(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            valid = root / "adapter.safetensors"
            unrelated = root / "other.safetensors"
            save_file(
                {"query_norms.0.weight": torch.ones(2)},
                valid,
                metadata={"architecture": ARCHITECTURE},
            )
            save_file(
                {"query_norms.0.weight": torch.ones(2)},
                unrelated,
                metadata={"architecture": "different"},
            )

            with patch("anima3b.files.text_encoder_roots", return_value=[root]):
                found = adapters()

        self.assertEqual(found, {"adapter.safetensors": str(valid)})

    def test_qwen_discovery_accepts_the_paired_filename(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            paired = root / "qwen35_4b.safetensors"
            paired.touch()
            (root / "qwen3_06b_base.safetensors").touch()

            with patch("anima3b.files.text_encoder_roots", return_value=[root]):
                found = qwen35_models()

        self.assertEqual(found, {"qwen35_4b.safetensors": str(paired)})

    def test_runtime_requires_the_paired_52_block_checkpoint(self):
        engine, clip = Anima3BRuntime._require_anima(
            fake_anima(ANIMA38_BLOCK_COUNT)
        )

        self.assertIsNotNone(engine)
        self.assertIsNotNone(clip)
        with self.assertRaisesRegex(RuntimeError, "52-block checkpoint"):
            Anima3BRuntime._require_anima(fake_anima(40))


if __name__ == "__main__":
    unittest.main()
