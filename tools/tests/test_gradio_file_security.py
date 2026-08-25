import ast
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from modules.aikimi_security.paths import (
    UnsafeAllowedPathError,
    build_gradio_allowed_paths,
    build_gradio_blocked_paths,
)

ROOT = Path(__file__).resolve().parents[2]


def _attribute_name(node):
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


class GradioFileSecurityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.script_root = self.base / "repo"
        self.data_root = self.base / "data"
        self.canvas_root = self.script_root / "modules_forge" / "forge_canvas"
        for path in (
            self.script_root,
            self.data_root,
            self.script_root / "output",
            self.data_root / "outputs",
            self.data_root / "tmp",
            self.canvas_root,
        ):
            path.mkdir(parents=True, exist_ok=True)
        for name in ("canvas.js", "canvas.css", "canvas.py"):
            (self.canvas_root / name).write_text(name, encoding="utf-8")

    def test_allowed_paths_contain_only_managed_directories_and_exact_canvas_assets(self):
        result = {
            Path(path)
            for path in build_gradio_allowed_paths(
                self.script_root,
                self.data_root,
                canvas_root=self.canvas_root,
            )
        }

        self.assertIn((self.script_root / "output").resolve(), result)
        self.assertIn((self.data_root / "outputs").resolve(), result)
        self.assertIn((self.data_root / "tmp").resolve(), result)
        self.assertIn((self.canvas_root / "canvas.js").resolve(), result)
        self.assertIn((self.canvas_root / "canvas.css").resolve(), result)
        self.assertNotIn(self.script_root.resolve(), result)
        self.assertNotIn(self.data_root.resolve(), result)
        self.assertNotIn((self.canvas_root / "canvas.py").resolve(), result)

    def test_arbitrary_requested_path_and_symlink_escape_are_rejected(self):
        outside = self.base / "private"
        outside.mkdir()
        with self.assertRaises(UnsafeAllowedPathError):
            build_gradio_allowed_paths(
                self.script_root,
                self.data_root,
                requested_paths=[outside],
            )

        link = self.data_root / "outputs" / "escape"
        try:
            link.symlink_to(outside, target_is_directory=True)
        except OSError as exc:
            self.skipTest(f"symlink creation is unavailable: {exc}")
        with self.assertRaises(UnsafeAllowedPathError):
            build_gradio_allowed_paths(
                self.script_root,
                self.data_root,
                requested_paths=[link],
            )

    def test_managed_root_and_canvas_asset_symlink_escapes_are_rejected(self):
        outside = self.base / "outside"
        outside.mkdir()
        output_link = self.script_root / "outputs"
        try:
            output_link.symlink_to(outside, target_is_directory=True)
        except OSError as exc:
            self.skipTest(f"symlink creation is unavailable: {exc}")

        with self.assertRaises(UnsafeAllowedPathError):
            build_gradio_allowed_paths(
                self.script_root,
                self.data_root,
                canvas_root=self.canvas_root,
            )

        output_link.unlink()
        secret = outside / "secret.txt"
        secret.write_text("secret", encoding="utf-8")
        canvas_asset = self.canvas_root / "canvas.js"
        canvas_asset.unlink()
        canvas_asset.symlink_to(secret)
        with self.assertRaises(UnsafeAllowedPathError):
            build_gradio_allowed_paths(
                self.script_root,
                self.data_root,
                canvas_root=self.canvas_root,
            )

    def test_blocked_paths_cover_state_credentials_logs_models_and_dynamic_secrets(self):
        dynamic = (
            self.data_root / ".env.local",
            self.data_root / "sysinfo-2026.json",
            self.data_root / "server.key",
            self.script_root / "client.p12",
        )
        for path in dynamic:
            path.write_text("secret", encoding="utf-8")

        result = {
            Path(path)
            for path in build_gradio_blocked_paths(
                self.script_root,
                self.data_root,
            )
        }

        for relative in (
            ".git",
            "config.json",
            "ui-config.json",
            "forge_neo_model_paths.yaml",
            "models",
            "extensions",
            "repositories",
            "venv",
            "logs",
            "secrets",
            "api-auth.txt",
            "gradio-auth.txt",
            "params.txt",
            "webui-user.bat",
        ):
            with self.subTest(relative=relative):
                self.assertIn((self.data_root / relative).resolve(), result)
        for path in dynamic:
            self.assertIn(path.resolve(), result)

    def test_gradio_async_file_validator_is_not_monkeypatched(self):
        source_path = ROOT / "modules" / "ui_tempdir.py"
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        assignments = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                assignments.extend(_attribute_name(target) for target in targets)

        self.assertNotIn(
            "gradio.processing_utils.async_move_files_to_cache",
            assignments,
            "A copied validator must not replace Gradio's maintained async file checks.",
        )
        self.assertIn("gradio.processing_utils.save_pil_to_cache", assignments)

    def test_gradio_allowed_path_cli_default_is_empty(self):
        tree = ast.parse((ROOT / "modules" / "cmd_args.py").read_text(encoding="utf-8"))
        matches = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not node.args or not isinstance(node.args[0], ast.Constant):
                continue
            if node.args[0].value != "--gradio-allowed-path":
                continue
            defaults = [keyword.value for keyword in node.keywords if keyword.arg == "default"]
            matches.extend(defaults)

        self.assertEqual(len(matches), 1)
        self.assertIsInstance(matches[0], ast.List)
        self.assertEqual(matches[0].elts, [])

    def test_gradio_file_policy_blocks_state_but_serves_managed_output(self):
        from gradio.routes import file_fetch
        from starlette.exceptions import HTTPException

        config = self.data_root / "config.json"
        git_config = self.data_root / ".git" / "config"
        output = self.data_root / "outputs" / "result.txt"
        config.write_text("{}", encoding="utf-8")
        git_config.parent.mkdir(parents=True)
        git_config.write_text("private", encoding="utf-8")
        output.write_text("result", encoding="utf-8")

        allowed = build_gradio_allowed_paths(
            self.script_root,
            self.data_root,
            canvas_root=self.canvas_root,
        )
        blocked = build_gradio_blocked_paths(self.script_root, self.data_root)
        policy = SimpleNamespace(
            allowed_paths=allowed,
            blocked_paths=blocked,
        )
        request = SimpleNamespace(headers={})
        upload_dir = self.base / "uploads"
        upload_dir.mkdir()

        for path in (config, git_config, self.canvas_root / "canvas.py"):
            with self.subTest(path=path), self.assertRaises(HTTPException) as raised:
                file_fetch(str(path), request, policy, upload_dir)
            self.assertEqual(raised.exception.status_code, 403)

        for path in (output, self.canvas_root / "canvas.js"):
            with self.subTest(path=path):
                response = file_fetch(str(path), request, policy, upload_dir)
                self.assertEqual(response.status_code, 200)


if __name__ == "__main__":
    unittest.main()
