import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from unittest import mock

from PIL import Image

from modules_forge.sensenova_u15_bridge import RuntimeStatus


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = (
    ROOT
    / "extensions-builtin"
    / "sensenova-u15-studio"
    / "scripts"
    / "sensenova_u15_studio.py"
)
STYLE = ROOT / "extensions-builtin" / "sensenova-u15-studio" / "style.css"
JAVASCRIPT = (
    ROOT
    / "extensions-builtin"
    / "sensenova-u15-studio"
    / "javascript"
    / "sensenova_u15_studio.js"
)
BRIDGE = ROOT / "modules_forge" / "sensenova_u15_bridge.py"


def load_studio_module():
    modules_package = ModuleType("modules")
    callbacks_module = ModuleType("modules.script_callbacks")
    callbacks_module.on_ui_tabs = mock.Mock()
    paths_module = ModuleType("modules.paths")
    paths_module.data_path = str(ROOT)
    modules_package.script_callbacks = callbacks_module

    spec = importlib.util.spec_from_file_location("_test_sensenova_u15_studio", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    stubs = {
        "modules": modules_package,
        "modules.script_callbacks": callbacks_module,
        "modules.paths": paths_module,
    }
    with mock.patch.dict(sys.modules, stubs):
        assert spec.loader is not None
        spec.loader.exec_module(module)
    return module


class SenseNovaStudioSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.script = SCRIPT.read_text(encoding="utf-8")
        cls.style = STYLE.read_text(encoding="utf-8")
        cls.javascript = JAVASCRIPT.read_text(encoding="utf-8")
        cls.bridge = BRIDGE.read_text(encoding="utf-8")
        cls.studio = load_studio_module()

    def test_dedicated_tab_is_registered(self):
        self.assertIn('"SenseNova U1.5"', self.script)
        self.assertIn(
            'script_callbacks.on_ui_tabs(_build_ui, name="sensenova_u15_studio")',
            self.script,
        )

    def test_multi_image_order_controls_are_present(self):
        self.assertIn("gr.Gallery", self.script)
        self.assertIn("末尾へ追加", self.script)
        self.assertIn("選択ファイルを一括追加", self.script)
        self.assertIn('file_count="multiple"', self.script)
        self.assertIn('elem_id="sn-bulk-add"', self.script)
        self.assertIn(".sn-bulk-row", self.style)
        self.assertIn("選択画像を差し替え", self.script)
        self.assertIn("_move_reference", self.script)
        self.assertIn(
            'label=f"参照画像（最大{MAX_REFERENCE_IMAGES}枚）"', self.script
        )

    def test_final_int8_convrot_is_the_fixed_model(self):
        self.assertIn("gr.State(QUANT_INT8_CONVROT)", self.script)
        self.assertIn("正式版 · INT8 ConvRot", self.script)
        self.assertNotIn("QUANT_BF16", self.script)
        self.assertNotIn("Q8_0 GGUF", self.script)
        self.assertIn(
            "community-maintained",
            (ROOT / "download_sensenova_u15_int8.ps1").read_text(encoding="utf-8"),
        )
        self.assertIn("interactive=False", self.script)

    def test_generation_and_cancel_share_job_state(self):
        self.assertIn("run_generation(", self.script)
        self.assertIn("cancel_generation(job_id or None)", self.script)
        self.assertIn("job_id = gr.State", self.script)
        self.assertNotIn("def _generate(*values)", self.script)
        self.assertNotIn("def _summary_from_ui(*values)", self.script)

    def test_measured_24gb_safe_defaults_are_visible(self):
        self.assertIn('value="2048x2048"', self.script)
        self.assertIn("24GB Safe · 2K出力優先", self.script)
        self.assertIn("各約0.26MP · 比率保護", self.script)
        self.assertIn("元の入力1枚目を基準", self.bridge)
        self.assertIn("Uncapped streaming", self.script)
        updates = self.studio._mode_updates(self.studio.MODE_EDIT, "2048x2048")
        self.assertEqual(updates[1]["value"], "auto")
        self.assertEqual(updates[3]["value"], str(512 * 512))

    def test_responsive_and_accessible_ui_contracts(self):
        self.assertIn("@media (max-width: 640px)", self.style)
        self.assertIn("prefers-reduced-motion", self.style)
        self.assertIn(":focus-visible", self.style)
        self.assertIn('role="status"', self.bridge)
        self.assertIn("Ctrl / ⌘ + Enter", self.script)
        self.assertIn("onUiLoaded(setupStudio)", self.javascript)
        self.assertIn("window.localStorage", self.javascript)
        self.assertNotIn("function restoreDraft", self.javascript)

    def test_real_gradio_ui_builds_with_unique_controls(self):
        status = RuntimeStatus(
            ready=False,
            source_ready=True,
            dependencies_ready=True,
            checkpoint_ready=False,
            source_path=ROOT / "runtime-final",
            checkpoint_path=ROOT / "model.safetensors",
            messages=("test",),
        )
        with mock.patch.object(self.studio, "inspect_runtime", return_value=status):
            interface = self.studio._build_ui()[0][0]
        config = interface.get_config_file()
        element_ids = [
            component["props"].get("elem_id")
            for component in config["components"]
            if component.get("props", {}).get("elem_id")
        ]
        self.assertEqual(len(element_ids), len(set(element_ids)))
        self.assertIn("sn-reference-gallery", element_ids)
        self.assertIn("sn-reference-bulk-upload", element_ids)
        self.assertIn("sn-generate", element_ids)

        prompt_id = next(
            component["id"]
            for component in config["components"]
            if component.get("props", {}).get("elem_id") == "sn-prompt"
        )
        draft_loads = [
            dependency
            for dependency in config["dependencies"]
            if dependency.get("js")
            and dependency.get("outputs") == [prompt_id]
            and "localStorage.getItem" in dependency["js"]
        ]
        self.assertEqual(len(draft_loads), 1)
        self.assertFalse(draft_loads[0]["backend_fn"])
        self.assertLess(
            self.script.index('elem_id="sn-progress"'),
            self.script.index('elem_id="sn-validation"'),
        )

    def test_bulk_reference_files_keep_selection_order(self):
        with tempfile.TemporaryDirectory() as temp:
            first = Path(temp) / "first.png"
            second = Path(temp) / "second.png"
            Image.new("RGB", (8, 8), (255, 0, 0)).save(first)
            Image.new("RGB", (8, 8), (0, 0, 255)).save(second)
            gallery, uploads, order, selected = self.studio._append_reference_files(
                [], [str(first), str(second)]
            )
        values = gallery["value"]
        self.assertEqual(len(values), 2)
        self.assertEqual(values[0][0].getpixel((0, 0)), (255, 0, 0))
        self.assertEqual(values[1][0].getpixel((0, 0)), (0, 0, 255))
        self.assertIsNone(uploads["value"])
        self.assertIn("Image 1", order)
        self.assertEqual(selected, 1)

    def test_gallery_selection_event_keeps_zero_based_index(self):
        self.assertEqual(self.studio._select_reference(mock.Mock(index=3)), 3)
        self.assertEqual(self.studio._select_reference(None), -1)


if __name__ == "__main__":
    unittest.main()
