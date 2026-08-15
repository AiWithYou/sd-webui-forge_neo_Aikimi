import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from modules import launch_utils


class PrepareEnvironmentTests(unittest.TestCase):
    @staticmethod
    def _args(**overrides):
        values = {
            "skip_python_version_check": True,
            "reinstall_torch": False,
            "skip_torch_cuda_test": True,
            "xformers": False,
            "reinstall_xformers": False,
            "sage": False,
            "flash": False,
            "nunchaku": False,
            "bnb": False,
            "ngrok": None,
            "onnxruntime_gpu": False,
            "skip_install": True,
            "update_all_extensions": False,
            "ui_settings_file": "config.json",
        }
        values.update(overrides)
        return SimpleNamespace(**values)

    def test_default_packaging_and_bnb_packages_are_preserved(self):
        installed = []

        def is_installed(package):
            return package not in {"packaging", "bitsandbytes"}

        with (
            patch.object(launch_utils, "args", self._args(bnb=True)),
            patch.object(launch_utils, "is_installed", side_effect=is_installed),
            patch.object(launch_utils, "requirements_met", return_value=True),
            patch.object(launch_utils, "_torch_version", return_value=("2.11.0", "cu130")),
            patch.object(launch_utils, "run_pip", side_effect=lambda command, _label: installed.append(command)),
            patch.object(launch_utils, "git_tag", return_value="test"),
            patch.object(launch_utils.os.path, "isfile", return_value=True),
            patch.object(launch_utils.os, "remove", side_effect=OSError),
            patch.object(launch_utils.startup_timer, "record"),
            patch.dict(os.environ, {}, clear=False),
        ):
            os.environ.pop("PACKAGING_PACKAGE", None)
            os.environ.pop("BNB_PACKAGE", None)
            launch_utils.prepare_environment()

        self.assertIn("install packaging==26.2", installed)
        self.assertIn("install bitsandbytes==0.49.2", installed)

    def test_gpu_probe_reports_a_generic_compute_device_error(self):
        with (
            patch.object(launch_utils, "args", self._args(skip_torch_cuda_test=False)),
            patch.object(launch_utils, "is_installed", return_value=True),
            patch.object(launch_utils, "check_run_python", return_value=(False, "probe failed")),
            patch.object(launch_utils, "git_tag", return_value="test"),
            patch.object(launch_utils.os, "remove", side_effect=OSError),
            patch.object(launch_utils.startup_timer, "record"),
        ):
            with self.assertRaisesRegex(RuntimeError, "any compute device"):
                launch_utils.prepare_environment()

    def test_old_driver_probe_keeps_the_manual_pytorch_fallback(self):
        with (
            patch.object(launch_utils, "args", self._args(skip_torch_cuda_test=False)),
            patch.object(launch_utils, "is_installed", return_value=True),
            patch.object(launch_utils, "check_run_python", return_value=(False, "older driver")),
            patch.object(launch_utils, "git_tag", return_value="test"),
            patch.object(launch_utils.os, "remove", side_effect=OSError),
            patch.object(launch_utils.startup_timer, "record"),
        ):
            with self.assertRaisesRegex(SystemError, "manually install older version of PyTorch"):
                launch_utils.prepare_environment()


if __name__ == "__main__":
    unittest.main()
