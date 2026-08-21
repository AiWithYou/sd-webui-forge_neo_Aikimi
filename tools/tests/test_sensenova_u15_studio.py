import unittest
from pathlib import Path


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


class SenseNovaStudioSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.script = SCRIPT.read_text(encoding="utf-8")
        cls.style = STYLE.read_text(encoding="utf-8")
        cls.javascript = JAVASCRIPT.read_text(encoding="utf-8")
        cls.bridge = BRIDGE.read_text(encoding="utf-8")

    def test_dedicated_tab_is_registered(self):
        self.assertIn('"SenseNova U1.5"', self.script)
        self.assertIn(
            'script_callbacks.on_ui_tabs(_build_ui, name="sensenova_u15_studio")',
            self.script,
        )

    def test_multi_image_order_controls_are_present(self):
        self.assertIn("gr.Gallery", self.script)
        self.assertIn("末尾へ追加", self.script)
        self.assertIn("選択画像を差し替え", self.script)
        self.assertIn("_move_reference", self.script)
        self.assertIn("最大8枚", self.script)

    def test_q8_and_bf16_are_explicit_choices(self):
        self.assertIn('("INT8 · Q8_0 GGUF", QUANT_Q8)', self.script)
        self.assertIn('("公式 BF16", QUANT_BF16)', self.script)
        self.assertIn(
            "community-maintained",
            (ROOT / "download_sensenova_u15_int8.ps1").read_text(encoding="utf-8"),
        )
        self.assertIn("interactive=False", self.script)
        self.assertIn("interactive=not q8", self.script)

    def test_generation_and_cancel_share_job_state(self):
        self.assertIn("run_generation(", self.script)
        self.assertIn("cancel_generation(job_id or None)", self.script)
        self.assertIn("job_id = gr.State", self.script)
        self.assertNotIn("def _generate(*values)", self.script)
        self.assertNotIn("def _summary_from_ui(*values)", self.script)

    def test_responsive_and_accessible_ui_contracts(self):
        self.assertIn("@media (max-width: 640px)", self.style)
        self.assertIn("prefers-reduced-motion", self.style)
        self.assertIn(":focus-visible", self.style)
        self.assertIn('role="status"', self.bridge)
        self.assertIn("Ctrl / ⌘ + Enter", self.script)
        self.assertIn("onUiLoaded(setupStudio)", self.javascript)
        self.assertIn("window.localStorage", self.javascript)


if __name__ == "__main__":
    unittest.main()
