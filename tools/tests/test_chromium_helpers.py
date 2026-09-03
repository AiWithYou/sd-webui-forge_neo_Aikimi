from __future__ import annotations

import subprocess
import unittest
from unittest.mock import patch

from websockets.exceptions import ConnectionClosedOK

from tools.tests import chromium_helpers


class FakeProcess:
    def __init__(self) -> None:
        self.pid = 4242
        self.returncode: int | None = None
        self.killed = False
        self.terminated = False

    def poll(self):
        return self.returncode

    def wait(self, timeout):
        if self.returncode is None:
            raise subprocess.TimeoutExpired("chromium", timeout)
        return self.returncode

    def kill(self):
        self.killed = True
        self.returncode = -1

    def terminate(self):
        self.terminated = True
        self.returncode = -1


class FakeWebSocket:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.closed = False

    def close(self):
        self.closed = True
        if self.error is not None:
            raise self.error


class ChromiumCleanupTests(unittest.TestCase):
    def windows_taskkill(self, process: FakeProcess):
        def complete(*_args, **_kwargs):
            process.returncode = 0
            return subprocess.CompletedProcess([], 0)

        return patch.object(chromium_helpers.subprocess, "run", side_effect=complete)

    def test_connection_closed_during_browser_close_still_closes_socket_and_process_tree(self):
        process = FakeProcess()
        websocket = FakeWebSocket()

        def close_browser():
            raise ConnectionClosedOK(None, None)

        with (
            patch.object(chromium_helpers.os, "name", "nt"),
            patch.object(chromium_helpers, "_find_windows_taskkill", return_value=r"C:\Windows\System32\taskkill.exe"),
            self.windows_taskkill(process) as taskkill,
        ):
            chromium_helpers.close_chromium(
                process,
                websocket=websocket,
                close_browser=close_browser,
                profile_flush_delay=0,
            )

        self.assertTrue(websocket.closed)
        self.assertEqual(process.returncode, 0)
        self.assertEqual(taskkill.call_args.args[0][1:], ["/PID", "4242", "/T", "/F"])

    def test_unexpected_socket_close_error_does_not_skip_process_tree_cleanup(self):
        process = FakeProcess()
        websocket = FakeWebSocket(ValueError("unexpected close failure"))

        with (
            patch.object(chromium_helpers.os, "name", "nt"),
            patch.object(chromium_helpers, "_find_windows_taskkill", return_value=r"C:\Windows\System32\taskkill.exe"),
            self.windows_taskkill(process) as taskkill,
            self.assertRaisesRegex(ValueError, "unexpected close failure"),
        ):
            chromium_helpers.close_chromium(
                process,
                websocket=websocket,
                profile_flush_delay=0,
            )

        self.assertEqual(process.returncode, 0)
        taskkill.assert_called_once()

    def test_already_exited_process_needs_no_tree_termination(self):
        process = FakeProcess()
        process.returncode = 0

        with patch.object(chromium_helpers.subprocess, "run") as taskkill:
            chromium_helpers.close_chromium(process, profile_flush_delay=0)

        taskkill.assert_not_called()

    def test_failed_taskkill_is_reported_after_stopping_the_root_process(self):
        process = FakeProcess()

        with (
            patch.object(chromium_helpers.os, "name", "nt"),
            patch.object(chromium_helpers, "_find_windows_taskkill", return_value=r"C:\Windows\System32\taskkill.exe"),
            patch.object(
                chromium_helpers.subprocess,
                "run",
                return_value=subprocess.CompletedProcess([], 1),
            ),
            self.assertRaisesRegex(RuntimeError, "taskkill failed with exit code 1"),
        ):
            chromium_helpers.close_chromium(process, profile_flush_delay=0)

        self.assertTrue(process.killed)
        self.assertEqual(process.returncode, -1)

    def test_missing_taskkill_is_reported_after_stopping_the_root_process(self):
        process = FakeProcess()

        with (
            patch.object(chromium_helpers.os, "name", "nt"),
            patch.object(chromium_helpers, "_find_windows_taskkill", return_value=None),
            self.assertRaisesRegex(RuntimeError, "taskkill is unavailable"),
        ):
            chromium_helpers.close_chromium(process, profile_flush_delay=0)

        self.assertTrue(process.terminated)
        self.assertEqual(process.returncode, -1)

    def test_nonzero_taskkill_needs_root_exit_and_owned_port_release(self):
        process = FakeProcess()

        def root_exits_during_taskkill(*_args, **_kwargs):
            process.returncode = 0
            return subprocess.CompletedProcess([], 128)

        with (
            patch.object(chromium_helpers.os, "name", "nt"),
            patch.object(chromium_helpers, "_find_windows_taskkill", return_value=r"C:\Windows\System32\taskkill.exe"),
            patch.object(chromium_helpers.subprocess, "run", side_effect=root_exits_during_taskkill),
            patch.object(chromium_helpers, "_wait_for_port_release", return_value=False) as port_release,
            self.assertRaisesRegex(RuntimeError, "taskkill failed with exit code 128"),
        ):
            chromium_helpers.close_chromium(
                process,
                owned_port=9222,
                profile_flush_delay=0,
            )

        port_release.assert_called_once_with(9222)
        self.assertFalse(process.killed)

    def test_nonzero_taskkill_is_normalized_only_after_owned_port_release(self):
        process = FakeProcess()

        def root_exits_during_taskkill(*_args, **_kwargs):
            process.returncode = 0
            return subprocess.CompletedProcess([], 128)

        with (
            patch.object(chromium_helpers.os, "name", "nt"),
            patch.object(chromium_helpers, "_find_windows_taskkill", return_value=r"C:\Windows\System32\taskkill.exe"),
            patch.object(chromium_helpers.subprocess, "run", side_effect=root_exits_during_taskkill),
            patch.object(chromium_helpers, "_wait_for_port_release", return_value=True) as port_release,
        ):
            chromium_helpers.close_chromium(
                process,
                owned_port=9222,
                profile_flush_delay=0,
            )

        port_release.assert_called_once_with(9222)
        self.assertFalse(process.killed)

    def test_already_exited_root_still_requires_owned_port_release(self):
        process = FakeProcess()
        process.returncode = 0

        with (
            patch.object(chromium_helpers, "_wait_for_port_release", return_value=False),
            self.assertRaisesRegex(RuntimeError, "CDP port is still in use"),
        ):
            chromium_helpers.close_chromium(
                process,
                owned_port=9222,
                profile_flush_delay=0,
            )


if __name__ == "__main__":
    unittest.main()
