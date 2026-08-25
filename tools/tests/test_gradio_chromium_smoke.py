import os
import shutil
import socket
import subprocess
import tempfile
import unittest
from pathlib import Path

import gradio as gr

from modules import gradio_compat


def find_chromium():
    candidates = [
        os.environ.get("CHROME_BIN"),
        shutil.which("google-chrome"),
        shutil.which("chromium"),
        Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
        Path(r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return str(Path(candidate))
    return None


def reserve_local_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


VISIBILITY_PROBE = """
<script>
(() => {
  let attempts = 0;
  let clicked = false;
  const probe = () => {
    const button = document.querySelector(
      "#visibility-show button, button#visibility-show"
    );
    const sliders = [
      document.querySelector("#visibility-slider-a"),
      document.querySelector("#visibility-slider-b"),
    ];
    if (!clicked && button) {
      clicked = true;
      button.click();
    }
    if (clicked && sliders.every((node) => node?.getBoundingClientRect().width > 0)) {
      requestAnimationFrame(() => {
        document.documentElement.dataset.gradioVisibilitySmoke = "pass";
      });
      return;
    }
    attempts += 1;
    if (attempts >= 160) {
      document.documentElement.dataset.gradioVisibilitySmoke = "fail";
      return;
    }
    setTimeout(probe, 50);
  };
  if (document.readyState === "loading") {
    window.addEventListener("DOMContentLoaded", probe, { once: true });
  } else {
    probe();
  }
})();
</script>
"""


class GradioChromiumVisibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.chromium = find_chromium()
        if not cls.chromium:
            raise unittest.SkipTest("Chrome or Chromium is not installed")

        hidden = gradio_compat.keep_hidden_component_mounted(False)
        with gr.Blocks() as cls.demo:
            show = gr.Button("Show controls", elem_id="visibility-show")
            first = gr.Slider(visible=hidden, elem_id="visibility-slider-a")
            second = gr.Slider(visible=hidden, elem_id="visibility-slider-b")
            show.click(
                lambda: (gr.update(visible=True), gr.update(visible=True)),
                outputs=[first, second],
                queue=False,
            )

        cls.port = reserve_local_port()
        cls.demo.launch(
            server_name="127.0.0.1",
            server_port=cls.port,
            prevent_thread_lock=True,
            quiet=True,
            head=VISIBILITY_PROBE,
        )

    @classmethod
    def tearDownClass(cls):
        cls.demo.close()

    def test_multiple_hidden_sliders_become_visible_without_freezing_chromium(self):
        with tempfile.TemporaryDirectory(prefix="gradio617-chrome-") as profile:
            command = [
                self.chromium,
                "--headless=new",
                "--disable-gpu",
                "--disable-extensions",
                "--no-first-run",
                "--no-default-browser-check",
                "--window-size=1280,900",
                f"--user-data-dir={profile}",
                "--virtual-time-budget=12000",
                "--dump-dom",
                f"http://127.0.0.1:{self.port}/",
            ]
            # The executable is an existing local Chrome binary and arguments
            # are passed as a fixed list without a shell.
            completed = subprocess.run(  # noqa: S603
                command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=30,
                check=False,
            )

        self.assertEqual(completed.returncode, 0, completed.stderr[-2000:])
        self.assertIn('data-gradio-visibility-smoke="pass"', completed.stdout)
        self.assertNotIn('data-gradio-visibility-smoke="fail"', completed.stdout)


if __name__ == "__main__":
    unittest.main()
