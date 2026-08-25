import ast
import importlib
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import modules
import modules_forge
from modules.aikimi_security.api_policy import (
    PUBLIC_CMD_FLAG_ALLOWLIST,
    public_cmd_flags,
    public_options,
)
from modules.aikimi_security.auth import (
    AuthenticationConfigError,
    validate_auth_configuration,
)
from modules.aikimi_security.remote_access import (
    RemoteAccessError,
    exposure_reasons,
    server_bind_name,
    validate_remote_access,
)

ROOT = Path(__file__).resolve().parents[2]
SENTINEL = "unit-test-sensitive-value-7fb4"
PRIVATE_WINDOWS_PATH = "Q:/private/models/secret-model.safetensors"


def fake_module(name: str, **attributes) -> ModuleType:
    module = ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    return module


def load_isolated_options_module():
    """Load the production Options classes without importing GPU/UI modules."""

    module_name = "aikimi_test_options_module"
    spec = importlib.util.spec_from_file_location(module_name, ROOT / "modules" / "options.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load modules/options.py")
    module = importlib.util.module_from_spec(spec)
    command_options = SimpleNamespace(
        freeze_settings=False,
        freeze_settings_in_sections=None,
        freeze_specific_settings=None,
        hide_ui_dir_config=False,
    )
    replacements = {
        module_name: module,
        "gradio": fake_module("gradio", HTML=lambda **_kwargs: None),
        "modules.errors": fake_module("modules.errors", display=lambda *_args, **_kwargs: None),
        "modules.paths_internal": fake_module("modules.paths_internal", script_path=str(ROOT)),
        "modules.shared_cmd_options": fake_module("modules.shared_cmd_options", cmd_opts=command_options),
        "modules.ui_components": fake_module("modules.ui_components", FormRow=object),
    }
    with patch.dict(sys.modules, replacements):
        spec.loader.exec_module(module)
    return module


def load_sysinfo_module():
    previous = os.environ.get("IGNORE_CMD_ARGS_ERRORS")
    os.environ["IGNORE_CMD_ARGS_ERRORS"] = "1"
    try:
        return importlib.import_module("modules.sysinfo")
    finally:
        if previous is None:
            os.environ.pop("IGNORE_CMD_ARGS_ERRORS", None)
        else:
            os.environ["IGNORE_CMD_ARGS_ERRORS"] = previous


def launch_options(**overrides) -> SimpleNamespace:
    values = {
        "aikimi_remote": False,
        "api": False,
        "api_auth": None,
        "api_auth_path": None,
        "gradio_auth": None,
        "gradio_auth_path": None,
        "listen": False,
        "ngrok": None,
        "nowebui": False,
        "server_name": None,
        "share": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class PublicCommandFlagTests(unittest.TestCase):
    def test_cmd_flags_are_allowlisted_and_future_secrets_and_paths_are_absent(self):
        options = Namespace(
            aikimi_remote=False,
            api=True,
            api_auth=SENTINEL,
            api_auth_path=PRIVATE_WINDOWS_PATH,
            disable_all_extensions=False,
            disable_extra_extensions=False,
            freeze_settings=False,
            gradio_auth=SENTINEL,
            gradio_auth_path=PRIVATE_WINDOWS_PATH,
            gradio_debug=False,
            hide_ui_dir_config=False,
            listen=False,
            lowvram=False,
            medvram=False,
            ngrok=SENTINEL,
            ngrok_options={"basic_auth": SENTINEL},
            no_gradio_queue=False,
            nowebui=False,
            port=7861,
            share=False,
            theme="dark",
            tls_keyfile=PRIVATE_WINDOWS_PATH,
            ui_debug_mode=False,
            ui_settings_file=PRIVATE_WINDOWS_PATH,
            future_secret_token=SENTINEL,
        )

        result = public_cmd_flags(options)
        serialized = json.dumps(result, sort_keys=True)
        computed = {
            "api_auth_enabled",
            "bind_scope",
            "gradio_auth_enabled",
        }

        self.assertLessEqual(set(result), PUBLIC_CMD_FLAG_ALLOWLIST | computed)
        self.assertEqual(result["port"], 7861)
        self.assertEqual(result["bind_scope"], "remote")
        self.assertTrue(result["api_auth_enabled"])
        self.assertTrue(result["gradio_auth_enabled"])
        self.assertNotIn("future_secret_token", result)
        self.assertNotIn(SENTINEL, serialized)
        self.assertNotIn(PRIVATE_WINDOWS_PATH, serialized)


class PublicOptionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.options_module = load_isolated_options_module()

    def build_options(self):
        OptionInfo = self.options_module.OptionInfo
        Options = self.options_module.Options
        options = Options(
            {
                "samples_format": OptionInfo("png"),
                "public_count": OptionInfo(1),
                "api_enable_requests": OptionInfo(True, restrict_api=True),
                "outdir_samples": OptionInfo(PRIVATE_WINDOWS_PATH),
                "forge_additional_modules": OptionInfo([PRIVATE_WINDOWS_PATH]),
            },
            restricted_opts={"outdir_samples"},
        )
        options.add_option("future_extension_default", OptionInfo("hidden-by-default"))
        options.add_option("future_api_token", OptionInfo(SENTINEL, api_access="read-write"))
        options.add_option("future_extension_path", OptionInfo(PRIVATE_WINDOWS_PATH, api_access="read-write"))
        options.add_option("reviewed_extension_option", OptionInfo("visible", api_access="read"))
        return options

    def test_options_get_hides_restricted_paths_and_unreviewed_extensions(self):
        options = self.build_options()

        result = public_options(options)
        serialized = json.dumps(result, sort_keys=True)

        self.assertEqual(result["samples_format"], "png")
        self.assertEqual(result["reviewed_extension_option"], "visible")
        self.assertNotIn("api_enable_requests", result)
        self.assertNotIn("outdir_samples", result)
        self.assertNotIn("forge_additional_modules", result)
        self.assertNotIn("future_extension_default", result)
        self.assertNotIn("future_api_token", result)
        self.assertNotIn("future_extension_path", result)
        self.assertNotIn(SENTINEL, serialized)
        self.assertNotIn(PRIVATE_WINDOWS_PATH, serialized)
        self.assertTrue(options.api_accessible("reviewed_extension_option", write=False))
        self.assertFalse(options.api_accessible("reviewed_extension_option", write=True))

    def test_legacy_local_request_toggle_is_forced_safe_during_initialization(self):
        source = (ROOT / "modules" / "shared_init.py").read_text(encoding="utf-8")
        self.assertIn('shared.opts.data["api_forbid_local_requests"] = True', source)


class OptionUpdateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.options_module = load_isolated_options_module()
        cls.sysinfo = load_sysinfo_module()

    def setUp(self):
        OptionInfo = self.options_module.OptionInfo
        Options = self.options_module.Options
        self.options = Options(
            {
                "public_count": OptionInfo(1),
                "public_toggle": OptionInfo(False),
                "api_enable_requests": OptionInfo(True, restrict_api=True),
                "outdir_samples": OptionInfo(PRIVATE_WINDOWS_PATH),
            },
            restricted_opts={"outdir_samples"},
        )
        self.options.save = MagicMock()
        self.fake_shared = fake_module(
            "modules.shared",
            opts=self.options,
            config_filename="config.json",
        )
        self.fake_sd_models = fake_module(
            "modules.sd_models",
            checkpoint_aliases={},
        )
        self.fake_main_entry = fake_module(
            "modules_forge.main_entry",
            checkpoint_change=MagicMock(return_value=False),
            modules_change=MagicMock(return_value=False),
            refresh_model_loading_parameters=MagicMock(),
        )

    def runtime_patches(self):
        return (
            patch.dict(
                sys.modules,
                {
                    "modules.shared": self.fake_shared,
                    "modules.sd_models": self.fake_sd_models,
                    "modules_forge.main_entry": self.fake_main_entry,
                },
            ),
            patch.object(modules, "shared", self.fake_shared, create=True),
            patch.object(modules, "sd_models", self.fake_sd_models, create=True),
            patch.object(modules_forge, "main_entry", self.fake_main_entry, create=True),
        )

    def call_set_config(self, request):
        patches = self.runtime_patches()
        with patches[0], patches[1], patches[2], patches[3]:
            return self.sysinfo.set_config(request, is_api=True)

    def test_valid_api_update_changes_and_saves_reviewed_option(self):
        changed = self.call_set_config({"public_count": 2})

        self.assertEqual(changed, ["public_count"])
        self.assertEqual(self.options.data["public_count"], 2)
        self.options.save.assert_called_once_with("config.json")

    def test_restricted_option_is_rejected_without_change_or_save(self):
        with self.assertRaises(self.sysinfo.RestrictedOptionError):
            self.call_set_config({"api_enable_requests": False})

        self.assertTrue(self.options.data["api_enable_requests"])
        self.options.save.assert_not_called()

    def test_unknown_option_is_422_class_for_null_and_non_null_values(self):
        for value in (None, "unknown"):
            with self.subTest(value=value):
                with self.assertRaises(self.sysinfo.UnknownOptionError):
                    self.call_set_config({"future_unknown": value})
        self.options.save.assert_not_called()

    def test_wrong_type_is_rejected_without_change(self):
        for value in ("two", None):
            with self.subTest(value=value):
                with self.assertRaises(self.sysinfo.InvalidOptionTypeError):
                    self.call_set_config({"public_count": value})

        self.assertEqual(self.options.data["public_count"], 1)
        self.options.save.assert_not_called()

    def test_invalid_later_key_does_not_partially_apply_valid_earlier_key(self):
        with self.assertRaises(self.sysinfo.RestrictedOptionError):
            self.call_set_config(
                {
                    "public_count": 2,
                    "api_enable_requests": False,
                }
            )

        self.assertEqual(self.options.data["public_count"], 1)
        self.assertTrue(self.options.data["api_enable_requests"])
        self.options.save.assert_not_called()

    def test_api_handler_maps_restricted_to_403_and_validation_to_422(self):
        tree = ast.parse((ROOT / "modules" / "api" / "api.py").read_text(encoding="utf-8"))
        api_class = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Api")
        handler = next(
            node for node in api_class.body if isinstance(node, ast.FunctionDef) and node.name == "set_config"
        )
        status_codes = {
            keyword.value.value
            for node in ast.walk(handler)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "HTTPException"
            for keyword in node.keywords
            if keyword.arg == "status_code" and isinstance(keyword.value, ast.Constant)
        }
        api_calls = [
            node
            for node in ast.walk(handler)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "set_config"
        ]

        self.assertEqual(status_codes, {403, 422})
        self.assertTrue(
            any(
                keyword.arg == "is_api" and isinstance(keyword.value, ast.Constant) and keyword.value.value is True
                for call in api_calls
                for keyword in call.keywords
            )
        )


class LaunchPolicyTests(unittest.TestCase):
    def test_default_and_explicit_loopback_bind_are_local(self):
        defaults = launch_options()
        loopback = launch_options(server_name="127.0.0.1")

        self.assertEqual(server_bind_name(defaults), "127.0.0.1")
        self.assertEqual(exposure_reasons(defaults), ())
        self.assertEqual(exposure_reasons(loopback), ())
        self.assertEqual(validate_remote_access(defaults), ())

    def test_each_remote_surface_requires_explicit_opt_in(self):
        cases = (
            launch_options(listen=True),
            launch_options(server_name="0.0.0.0"),  # noqa: S104 - rejection case
            launch_options(share=True),
            launch_options(ngrok="from-environment"),
        )

        for options in cases:
            with self.subTest(options=options):
                with self.assertRaises(RemoteAccessError):
                    validate_remote_access(options)

    def test_remote_webui_and_api_require_both_auth_sources(self):
        base = {
            "aikimi_remote": True,
            "api": True,
            "listen": True,
        }

        with self.assertRaisesRegex(RemoteAccessError, "Remote WebUI requires"):
            validate_remote_access(launch_options(**base, api_auth="api-user:api-pass"))
        with self.assertRaisesRegex(RemoteAccessError, "Remote API requires"):
            validate_remote_access(launch_options(**base, gradio_auth="web-user:web-pass"))

        valid = launch_options(
            **base,
            api_auth="api-user:api-pass",
            gradio_auth="web-user:web-pass",
        )
        self.assertEqual(validate_remote_access(valid), ("--listen",))
        validate_auth_configuration(valid)
        self.assertEqual(server_bind_name(valid), "0.0.0.0")  # noqa: S104 - explicit remote mode

    def test_remote_api_only_accepts_valid_auth_path_and_rejects_missing_file(self):
        with tempfile.TemporaryDirectory() as directory:
            auth_file = Path(directory) / "api-auth.txt"
            auth_file.write_text("api-user:api-pass\n", encoding="utf-8")
            valid = launch_options(
                aikimi_remote=True,
                api=True,
                api_auth_path=str(auth_file),
                listen=True,
                nowebui=True,
            )

            self.assertEqual(validate_remote_access(valid), ("--listen",))
            validate_auth_configuration(valid)

            missing = launch_options(
                aikimi_remote=True,
                api=True,
                api_auth_path=str(Path(directory) / "missing.txt"),
                listen=True,
                nowebui=True,
            )
            with self.assertRaises(AuthenticationConfigError):
                validate_auth_configuration(missing)


if __name__ == "__main__":
    unittest.main()
