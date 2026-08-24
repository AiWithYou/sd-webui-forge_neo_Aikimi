import ast
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

from PIL import Image

from modules import aikimi_status


ROOT = Path(__file__).resolve().parents[2]
ASSET_ROOT = ROOT / "assets" / "aikimi"
EXPECTED_STATES = {
    "idle",
    "loading_model",
    "generating",
    "completed",
    "queued",
    "warning",
    "error",
    "out_of_memory",
    "updating",
}


class AikimiStatusTests(unittest.TestCase):
    def test_manifest_covers_states_and_references_local_assets(self):
        manifest = json.loads((ASSET_ROOT / "manifest.json").read_text(encoding="utf-8"))

        self.assertEqual(set(manifest["states"]), EXPECTED_STATES)
        self.assertIn(manifest["default_state"], manifest["states"])
        for state in manifest["states"].values():
            asset_key = state["asset"]
            self.assertIn(asset_key, manifest["assets"])
            filename = manifest["assets"][asset_key]
            self.assertEqual(Path(filename).name, filename)
            self.assertTrue((ASSET_ROOT / filename).is_file())

    def test_character_assets_have_consistent_transparent_canvas(self):
        for name in ("idle", "working", "happy", "troubled"):
            with self.subTest(name=name):
                with Image.open(ASSET_ROOT / f"{name}.webp") as image:
                    self.assertEqual(image.size, (512, 640))
                    self.assertEqual(image.mode, "RGBA")
                    self.assertEqual(image.getchannel("A").getextrema(), (0, 255))

        with Image.open(ASSET_ROOT / "favicon.png") as favicon:
            self.assertEqual(favicon.size, (128, 128))
            self.assertEqual(favicon.mode, "RGBA")

    def test_model_snapshot_uses_shallow_loader_state(self):
        model_data = SimpleNamespace(
            sd_model=SimpleNamespace(
                forge_objects=object(),
                filename=r"C:\models\loaded-aikimi.safetensors",
                sd_checkpoint_info=SimpleNamespace(name="folder-a/loaded-aikimi.safetensors"),
            ),
            forge_loading_parameters={
                "checkpoint_info": SimpleNamespace(
                    filename=r"C:\models\folder-b\aikimi.safetensors",
                    name="folder-b/aikimi.safetensors",
                )
            },
            forge_hash="old-hash",
            forge_loading=True,
            last_load_seconds=8.21,
            last_load_error=None,
        )

        status = aikimi_status._model_snapshot(model_data)

        self.assertTrue(status["loaded"])
        self.assertTrue(status["loading"])
        self.assertEqual(status["selected_name"], "folder-b/aikimi.safetensors")
        self.assertEqual(status["loaded_name"], "folder-a/loaded-aikimi.safetensors")
        self.assertTrue(status["reload_pending"])
        self.assertNotIn("path", status)
        self.assertEqual(status["last_load_seconds"], 8.21)

    def test_status_auth_matches_gradio_cookie_boundary(self):
        open_app = SimpleNamespace(auth=None, auth_dependency=None)
        request = SimpleNamespace(cookies={})
        self.assertTrue(aikimi_status.request_is_authorized(open_app, request))

        protected_app = SimpleNamespace(
            auth={"user": "hash"},
            auth_dependency=None,
            cookie_id="cookie",
            tokens={"valid-token": "user"},
        )
        self.assertFalse(aikimi_status.request_is_authorized(protected_app, request))
        request.cookies["access-token-unsecure-cookie"] = "valid-token"
        self.assertTrue(aikimi_status.request_is_authorized(protected_app, request))

    def test_generation_snapshot_reports_pending_queue_separately(self):
        state = SimpleNamespace(
            job_count=2,
            job_no=1,
            sampling_steps=10,
            sampling_step=5,
            time_start=None,
            job="task(test)",
            textinfo="Sampling",
        )

        status = aikimi_status._generation_snapshot(
            state,
            pending_tasks={"task(waiting-1)": 1.0, "task(waiting-2)": 2.0},
        )

        self.assertTrue(status["active"])
        self.assertEqual(status["progress"], 0.75)
        self.assertEqual(status["queue_size"], 2)
        self.assertNotIn("task", status)
        self.assertNotIn("job", status)
        self.assertNotIn("queue_tasks", status)

    def test_loader_status_defaults_do_not_change_model_contract(self):
        tree = ast.parse((ROOT / "modules" / "sd_models.py").read_text(encoding="utf-8"))
        model_data_class = next(
            node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "SdModelData"
        )
        constructor = next(
            node for node in model_data_class.body if isinstance(node, ast.FunctionDef) and node.name == "__init__"
        )
        assigned_attributes = {
            target.attr
            for node in ast.walk(constructor)
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Name)
            and target.value.id == "self"
        }

        self.assertIn("forge_loading", assigned_attributes)
        self.assertIn("last_load_seconds", assigned_attributes)
        self.assertIn("last_load_error", assigned_attributes)

    def test_model_loading_flag_is_cleared_by_outer_finally(self):
        tree = ast.parse((ROOT / "modules" / "sd_models.py").read_text(encoding="utf-8"))
        reload_function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "forge_model_reload"
        )
        tracking_try = next(node for node in reload_function.body if isinstance(node, ast.Try))
        final_assignments = [
            node
            for node in ast.walk(ast.Module(body=tracking_try.finalbody, type_ignores=[]))
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Attribute)
                and target.attr == "forge_loading"
                for target in node.targets
            )
        ]

        self.assertEqual(len(final_assignments), 1)
        self.assertIs(final_assignments[0].value.value, False)

    def test_frontend_subscribes_without_replacing_progress_api(self):
        progress_source = (ROOT / "javascript" / "progressbar.js").read_text(encoding="utf-8")
        assistant_source = (ROOT / "javascript" / "aikimiStatus.js").read_text(encoding="utf-8")

        for event_name in (
            "webui:task-start",
            "webui:task-progress",
            "webui:task-error",
            "webui:task-end",
        ):
            self.assertIn(event_name, progress_source)
            self.assertIn(event_name, assistant_source)
        self.assertIn("window.AikimiStatus", assistant_source)
        self.assertNotIn("idle.webp", assistant_source)
        self.assertNotIn("working.webp", assistant_source)


if __name__ == "__main__":
    unittest.main()
