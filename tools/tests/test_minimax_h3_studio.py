import importlib.util
import sys
import unittest
from pathlib import Path
from types import ModuleType
from unittest import mock

from modules_forge.minimax_h3_bridge import RuntimeReadiness


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
STUDIO_PATH = (
    REPOSITORY_ROOT
    / "extensions-builtin"
    / "minimax-h3-studio"
    / "scripts"
    / "minimax_h3_studio.py"
)
JAVASCRIPT_PATH = STUDIO_PATH.parents[1] / "javascript" / "minimax_h3_studio.js"


def load_studio_module():
    modules_package = ModuleType("modules")
    callbacks_module = ModuleType("modules.script_callbacks")
    callbacks_module.on_ui_tabs = mock.Mock()
    paths_module = ModuleType("modules.paths")
    paths_module.data_path = str(REPOSITORY_ROOT)
    paths_module.script_path = str(REPOSITORY_ROOT)
    modules_package.script_callbacks = callbacks_module

    spec = importlib.util.spec_from_file_location("_test_minimax_h3_studio", STUDIO_PATH)
    module = importlib.util.module_from_spec(spec)
    stubs = {
        "modules": modules_package,
        "modules.script_callbacks": callbacks_module,
        "modules.paths": paths_module,
    }
    with mock.patch.dict(sys.modules, stubs):
        spec.loader.exec_module(module)
    return module


class MiniMaxH3StudioCallbackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.studio = load_studio_module()

    def test_manual_setting_update_refreshes_summary_and_marks_custom(self):
        summary, state, validation = self.studio._custom_settings_updates(
            "16:9",
            "balanced",
            5.0,
            20,
            "simple",
            "match",
        )
        self.assertIn("960 × 544", summary)
        self.assertIn('data-h3-preset="custom"', state)
        self.assertEqual(validation["__type__"], "update")

    def test_mode_update_resets_hidden_reference_max_and_summary(self):
        updates = self.studio._mode_updates(
            self.studio.MODE_TEXT,
            "16:9",
            "preview",
            5.0,
            20,
            "simple",
            "max",
        )
        self.assertEqual(len(updates), 6)
        self.assertFalse(updates[1]["visible"])
        self.assertEqual(updates[3]["value"], "match")
        self.assertNotIn("Reference Max", updates[4])
        self.assertEqual(updates[5]["__type__"], "update")

    def test_generate_callback_disables_and_restores_actions(self):
        generation_updates = [
            {"stage": "prepare", "message": "prepare", "progress": 0.06, "prompt_id": ""},
            {"stage": "queued", "message": "queued", "progress": 0.12, "prompt_id": "job-1"},
            {
                "stage": "complete",
                "message": "complete",
                "progress": 1.0,
                "prompt_id": "job-1",
                "path": "result.mp4",
            },
        ]
        with mock.patch.object(self.studio, "resolve_runtime_root", return_value=Path("runtime")), mock.patch.object(
            self.studio,
            "run_generation",
            return_value=iter(generation_updates),
        ), mock.patch.object(
            self.studio,
            "_history_state",
            return_value=([], "history", [("latest", "result.mp4")]),
        ):
            results = list(
                self.studio._generate(
                    "runtime",
                    "http://127.0.0.1:8188",
                    "fast",
                    "text",
                    "prompt",
                    None,
                    None,
                    None,
                    None,
                    None,
                    "16:9",
                    "preview",
                    5.0,
                    20,
                    -1,
                    "simple",
                    "match",
                )
            )

        self.assertTrue(all(len(result) == 8 for result in results))
        self.assertFalse(results[0][5]["interactive"])
        self.assertFalse(results[0][6]["interactive"])
        self.assertEqual(results[0][6]["value"], "生成を準備中…")
        self.assertTrue(results[2][5]["interactive"])
        self.assertFalse(results[2][6]["interactive"])
        self.assertFalse(results[-1][5]["interactive"])
        self.assertTrue(results[-1][6]["interactive"])
        self.assertEqual(results[-1][6]["value"], "映像＋音声を生成")
        self.assertEqual(results[0][7], "")

    def test_generate_callback_restores_button_after_validation_error(self):
        with mock.patch.object(
            self.studio,
            "resolve_runtime_root",
            side_effect=self.studio.H3BridgeError("invalid runtime"),
        ):
            results = list(
                self.studio._generate(
                    "runtime",
                    "http://127.0.0.1:8188",
                    "fast",
                    "text",
                    "prompt",
                    None,
                    None,
                    None,
                    None,
                    None,
                    "16:9",
                    "preview",
                    5.0,
                    20,
                    -1,
                    "simple",
                    "match",
                )
            )
        self.assertEqual(len(results), 2)
        self.assertIn("invalid runtime", results[-1][0])
        self.assertTrue(results[-1][6]["interactive"])

    def test_empty_prompt_returns_inline_alert_and_focus_target(self):
        results = list(
            self.studio._generate(
                "runtime",
                "http://127.0.0.1:8188",
                "fast",
                "text",
                "",
                None,
                None,
                None,
                None,
                None,
                "16:9",
                "preview",
                5.0,
                20,
                -1,
                "simple",
                "match",
            )
        )
        self.assertEqual(len(results), 2)
        self.assertIn('data-h3-invalid="prompt"', results[-1][7])
        self.assertIn('role="alert"', results[-1][7])
        self.assertTrue(results[-1][6]["interactive"])

    def test_prompt_action_clears_input_validation(self):
        prompt, validation = self.studio._prompt_camera_updates(
            "",
            '<div data-h3-invalid="prompt">prompt error</div>',
        )
        self.assertIn("Camera:", prompt)
        self.assertEqual(validation, "")

    def test_settings_error_targets_settings_and_clears_on_seed_input(self):
        results = list(
            self.studio._generate(
                "runtime",
                "http://127.0.0.1:8188",
                "fast",
                "keyframes",
                "prompt",
                None,
                None,
                None,
                None,
                None,
                "16:9",
                "preview",
                5.0,
                20,
                -2,
                "simple",
                "match",
            )
        )
        validation = results[-1][7]
        self.assertIn('data-h3-invalid="settings"', validation)
        self.assertEqual(self.studio._clear_settings_validation(validation), "")
        unchanged = self.studio._clear_settings_validation(
            '<div data-h3-invalid="keyframes">keyframe error</div>'
        )
        self.assertEqual(unchanged["__type__"], "update")

    def test_ui_dependencies_match_fast_manual_update_contract(self):
        readiness = RuntimeReadiness(
            runtime_root=None,
            server_url=self.studio.H3_SERVER_URL,
            connected=False,
        )
        with mock.patch.object(self.studio, "_initial_runtime", return_value=None), mock.patch.object(
            self.studio,
            "inspect_readiness",
            return_value=readiness,
        ), mock.patch.object(
            self.studio,
            "_history_state",
            return_value=([], self.studio.history_html([]), []),
        ):
            interface = self.studio._build_ui()[0][0]
        config = interface.get_config_file()
        element_ids = [
            component["props"].get("elem_id")
            for component in config["components"]
            if component.get("props", {}).get("elem_id")
        ]
        self.assertEqual(len(element_ids), len(set(element_ids)))
        component_ids = {
            component["props"].get("elem_id"): component["id"]
            for component in config["components"]
            if component.get("props", {}).get("elem_id")
        }

        dependencies = config["dependencies"]
        generate_id = component_ids["h3-generate"]
        generate = next(
            dependency
            for dependency in dependencies
            if any(target[0] == generate_id and target[1] == "click" for target in dependency["targets"])
        )
        self.assertEqual(len(generate["inputs"]), 17)
        self.assertEqual(len(generate["outputs"]), 8)
        self.assertEqual(generate["trigger_mode"], "once")

        quick_id = component_ids["h3-preset-quick"]
        quick_preset = next(
            dependency
            for dependency in dependencies
            if any(target[0] == quick_id and target[1] == "click" for target in dependency["targets"])
        )
        self.assertEqual(quick_preset["api_name"], "h3_apply_quick_preset")
        self.assertEqual(len(quick_preset["inputs"]), 1)
        self.assertEqual(len(quick_preset["outputs"]), 7)

        quality_id = component_ids["h3-quality"]
        quality_input = next(
            dependency
            for dependency in dependencies
            if any(target[0] == quality_id and target[1] == "input" for target in dependency["targets"])
        )
        self.assertEqual(len(quality_input["inputs"]), 7)
        self.assertEqual(len(quality_input["outputs"]), 3)
        self.assertFalse(quality_input["queue"])

        mode_id = component_ids["h3-mode"]
        mode_change = next(
            dependency
            for dependency in dependencies
            if any(target[0] == mode_id and target[1] == "change" for target in dependency["targets"])
        )
        self.assertEqual(len(mode_change["inputs"]), 8)
        self.assertEqual(len(mode_change["outputs"]), 6)

        seed_id = component_ids["h3-seed"]
        seed_input = next(
            dependency
            for dependency in dependencies
            if any(target[0] == seed_id and target[1] == "input" for target in dependency["targets"])
        )
        self.assertEqual(len(seed_input["inputs"]), 1)
        self.assertEqual(len(seed_input["outputs"]), 1)
        self.assertFalse(seed_input["queue"])

    def test_javascript_exposes_keyboard_and_busy_accessibility_contract(self):
        source = JAVASCRIPT_PATH.read_text(encoding="utf-8")
        self.assertIn('input.name = "h3-generation-mode"', source)
        self.assertIn('event.key === "ArrowRight"', source)
        self.assertIn('event.key === "Home"', source)
        self.assertIn('generate.setAttribute("aria-keyshortcuts"', source)
        self.assertIn('studio.setAttribute("aria-busy"', source)


if __name__ == "__main__":
    unittest.main()
