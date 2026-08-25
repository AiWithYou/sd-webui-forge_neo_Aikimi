import importlib.metadata
import inspect
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import gradio as gr
import gradio.processing_utils
from packaging.requirements import Requirement
from PIL import Image

from modules import gradio_compat

ROOT = Path(__file__).resolve().parents[2]


def import_ui_tempdir():
    with patch.object(sys, "argv", ["gradio-compatibility-test"]):
        from modules import (
            shared,  # noqa: F401
            ui_tempdir,
        )

    return ui_tempdir


class GradioDependencyContractTests(unittest.TestCase):
    def test_runtime_matches_the_audited_direct_dependency_set(self):
        expected = {
            "gradio": gradio_compat.SUPPORTED_GRADIO_VERSION,
            "gradio-client": "2.5.0",
            "fastapi": "0.141.1",
            "starlette": "1.6.0",
            "huggingface-hub": "0.36.2",
            "transformers": "4.57.6",
        }
        self.assertEqual(
            {name: importlib.metadata.version(name) for name in expected},
            expected,
        )

    def test_requirements_are_the_single_source_for_gradio_installation(self):
        requirements = {}
        for raw_line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if " #" in line:
                line = line.split(" #", 1)[0].rstrip()
            requirement = Requirement(line)
            requirements[requirement.name] = requirement

        self.assertEqual(
            str(requirements["gradio"].specifier),
            f"=={gradio_compat.SUPPORTED_GRADIO_VERSION}",
        )
        self.assertEqual(str(requirements["gradio-client"].specifier), "==2.5.0")
        self.assertNotIn("gradio-rangeslider", requirements)
        self.assertNotIn("torch", requirements)

        launch_source = (ROOT / "modules" / "launch_utils.py").read_text(encoding="utf-8")
        self.assertNotIn("GRADIO_PACKAGE", launch_source)
        self.assertNotIn('is_installed("gradio")', launch_source)


class GradioFileHandlingTests(unittest.TestCase):
    def test_temp_override_keeps_gradio_official_async_path_validator(self):
        ui_tempdir = import_ui_tempdir()
        official_validator = gradio.processing_utils.async_move_files_to_cache

        ui_tempdir.install_ui_tempdir_override()

        self.assertIs(
            gradio.processing_utils.async_move_files_to_cache,
            official_validator,
        )
        self.assertEqual(official_validator.__module__, "gradio.processing_utils")

    def test_png_cache_override_accepts_gradio6_signature_and_keeps_metadata(self):
        ui_tempdir = import_ui_tempdir()
        image = Image.new("RGB", (4, 4), "white")
        image.info["parameters"] = "seed=123"

        with (
            tempfile.TemporaryDirectory() as temp_dir,
            patch.object(
                ui_tempdir,
                "shared",
                SimpleNamespace(opts=SimpleNamespace(temp_dir=""), demo=None),
            ),
        ):
            output = Path(
                ui_tempdir.save_pil_to_file(
                    image,
                    cache_dir=temp_dir,
                    name="preview",
                    format="png",
                )
            )
            with Image.open(output) as restored:
                self.assertEqual(restored.info["parameters"], "seed=123")

            self.assertTrue(output.name.startswith(ui_tempdir.MANAGED_TEMP_PREFIX))
            self.assertEqual(output.suffix, ".png")


class GradioUiContractTests(unittest.TestCase):
    def test_gradio6_launch_keeps_security_and_app_level_parameters(self):
        parameters = inspect.signature(gr.Blocks.launch).parameters
        for name in ("auth", "allowed_paths", "blocked_paths", "theme", "head"):
            self.assertIn(name, parameters)

    def test_representative_upload_gallery_and_file_components_build(self):
        import_ui_tempdir()
        with gr.Blocks() as demo:
            image = gr.Image(sources="upload", type="pil", buttons=[])
            gallery = gr.Gallery(type="pil", buttons=[])
            upload = gr.File(file_count="multiple", file_types=["image"])

        config = demo.get_config_file()
        component_types = {component["type"] for component in config["components"]}
        self.assertTrue({"image", "gallery", "file"}.issubset(component_types))
        self.assertEqual(image.sources, ["upload"])
        self.assertEqual(gallery.buttons, [])
        self.assertEqual(upload.file_count, "multiple")

    def test_hidden_components_stay_mounted_for_gradio617_visibility_updates(self):
        import_ui_tempdir()
        with gr.Blocks():
            slider = gr.Slider(visible=False)
            with gr.Column(visible=False) as column:
                gr.Textbox()

        self.assertEqual(slider.visible, "hidden")
        self.assertEqual(column.visible, "hidden")
        self.assertEqual(gr.update(visible=False)["visible"], "hidden")

    def test_runtime_import_does_not_generate_type_stubs_in_checkout(self):
        with patch.object(sys, "argv", ["gradio-compatibility-test"]):
            import modules.ui_components  # noqa: F401

        self.assertFalse((ROOT / "modules" / "ui_components.pyi").exists())

    def test_single_style_editor_rejects_gradio6_list_values_safely(self):
        self.assertEqual(gradio_compat.normalize_single_selection("style"), "style")
        self.assertEqual(gradio_compat.normalize_single_selection([]), "")

    def test_controlnet_range_replacement_preserves_order_and_bounds(self):
        self.assertEqual(gradio_compat.normalize_unit_interval(0.2, 0.8), (0.2, 0.8))
        self.assertEqual(gradio_compat.normalize_unit_interval(0.9, 0.1), (0.1, 0.9))
        self.assertEqual(gradio_compat.normalize_unit_interval(-1, 2), (0.0, 1.0))


if __name__ == "__main__":
    unittest.main()
