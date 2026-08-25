import html
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse
import unittest


ROOT = Path(__file__).resolve().parents[2]


def find_chromium():
    configured = os.environ.get("CHROME_BIN")
    candidates = [
        configured,
        shutil.which("google-chrome"),
        shutil.which("google-chrome-stable"),
        shutil.which("chromium"),
        shutil.which("chromium-browser"),
        shutil.which("chrome"),
        Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
        Path(r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return str(Path(candidate))
    return None


class AikimiFixtureHandler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        return

    def send_bytes(self, data, content_type, status=200):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(data)

    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self.send_fixture(parsed.query)
            return
        if parsed.path == "/javascript/aikimiStatus.js":
            self.send_bytes(
                (ROOT / "javascript" / "aikimiStatus.js").read_bytes(),
                "text/javascript; charset=utf-8",
            )
            return
        if parsed.path == "/style.css":
            self.send_bytes(
                (ROOT / "style.css").read_bytes(),
                "text/css; charset=utf-8",
            )
            return
        if parsed.path == "/aikimi-assets/manifest.json":
            self.send_bytes(
                (ROOT / "assets" / "aikimi" / "manifest.json").read_bytes(),
                "application/json",
            )
            return
        if parsed.path.startswith("/aikimi-assets/"):
            filename = Path(parsed.path).name
            asset = ROOT / "assets" / "aikimi" / filename
            if not asset.is_file():
                self.send_bytes(b"not found", "text/plain", 404)
                return
            content_type = "image/webp" if asset.suffix == ".webp" else "image/png"
            self.send_bytes(asset.read_bytes(), content_type)
            return
        if parsed.path == "/internal/aikimi-status":
            payload = {
                "generation": {"active": False, "progress": 0, "queue_size": 0},
                "model": {"loading": False, "loaded": False, "reload_pending": False},
                "memory": {"available": False},
                "backend": {"ready": True, "uptime_seconds": 1},
            }
            self.send_bytes(json.dumps(payload).encode(), "application/json")
            return
        self.send_bytes(b"not found", "text/plain", 404)

    def send_fixture(self, query):
        values = parse_qs(query)
        size = values.get("size", ["medium"])[0]
        position = values.get("position", ["bottom-right"])[0]
        dialogue = values.get("dialogue", ["1"])[0] != "0"
        animation = values.get("animation", ["1"])[0] != "0"
        options = {
            "aikimi_assistant_enabled": True,
            "aikimi_assistant_size": size,
            "aikimi_assistant_position": position,
            "aikimi_assistant_dialogue_enabled": dialogue,
            "aikimi_assistant_animation_enabled": animation,
        }
        document = f"""<!doctype html>
<html><head><meta charset="utf-8"><link rel="stylesheet" href="/style.css"></head>
<body><div id="tabs"></div><pre id="result">pending</pre>
<script>
window.opts = {json.dumps(options)};
window.gradioApp = () => document;
window.onUiLoaded = (callback) => setTimeout(callback, 0);
window.onOptionsAvailable = (callback) => setTimeout(callback, 5);
window.onOptionsChanged = () => {{}};
window.onAfterUiUpdate = () => {{}};
</script>
<script src="/javascript/aikimiStatus.js"></script>
<script>
(function report(attempt) {{
    const panel = document.querySelector("#aikimi-status");
    const portrait = panel?.querySelector(".aikimi-status__portrait");
    if ((!portrait?.currentSrc || !portrait.naturalWidth) && attempt < 120) {{
        setTimeout(() => report(attempt + 1), 50);
        return;
    }}
    const message = panel?.querySelector(".aikimi-status__message");
    const portraitWrap = panel?.querySelector(".aikimi-status__portrait-wrap");
    const result = document.querySelector("#result");
    result.dataset.ready = "true";
    result.textContent = JSON.stringify({{
        reduced: matchMedia("(prefers-reduced-motion: reduce)").matches,
        src: portrait?.currentSrc || "",
        size: panel?.dataset.size,
        position: panel?.dataset.position,
        dialogue: panel?.dataset.dialogue,
        motion: panel?.dataset.motion,
        messageHidden: Boolean(message?.hidden),
        characterWidth: Math.round(portraitWrap?.getBoundingClientRect().width || 0)
    }});
}})(0);
</script></body></html>"""
        self.send_bytes(document.encode(), "text/html; charset=utf-8")


class AikimiChromiumTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.chromium = find_chromium()
        if not cls.chromium:
            raise unittest.SkipTest("Chrome or Chromium is not installed")
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), AikimiFixtureHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base_url = f"http://127.0.0.1:{cls.server.server_port}/"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)

    def render_fixture(self, query="", reduced_motion=False):
        url = self.base_url + (f"?{query}" if query else "")
        with tempfile.TemporaryDirectory(prefix="aikimi-chrome-") as profile:
            command = [
                self.chromium,
                "--headless=new",
                "--disable-gpu",
                "--disable-extensions",
                "--no-first-run",
                "--no-default-browser-check",
                "--window-size=1280,900",
                f"--user-data-dir={profile}",
                "--virtual-time-budget=7000",
                "--dump-dom",
            ]
            if reduced_motion:
                command.append("--force-prefers-reduced-motion")
            command.append(url)
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=30,
                check=False,
            )
        self.assertEqual(completed.returncode, 0, completed.stderr[-2000:])
        match = re.search(
            r'<pre id="result"[^>]*>(.*?)</pre>', completed.stdout, re.DOTALL
        )
        self.assertIsNotNone(match, completed.stdout[-2000:])
        self.assertIn('data-ready="true"', match.group(0))
        return json.loads(html.unescape(match.group(1)))

    def test_os_reduced_motion_uses_still_asset_in_chromium(self):
        result = self.render_fixture(reduced_motion=True)

        self.assertTrue(result["reduced"])
        self.assertEqual(urlparse(result["src"]).path, "/aikimi-assets/idle-still.webp")
        self.assertEqual(result["motion"], "reduced")

    def test_default_preferences_use_animated_asset_in_chromium(self):
        result = self.render_fixture()

        self.assertFalse(result["reduced"])
        self.assertEqual(urlparse(result["src"]).path, "/aikimi-assets/idle.png")
        self.assertEqual(result["size"], "medium")
        self.assertEqual(result["position"], "bottom-right")
        self.assertEqual(result["dialogue"], "on")
        self.assertEqual(result["motion"], "animated")
        self.assertFalse(result["messageHidden"])
        self.assertEqual(result["characterWidth"], 150)

    def test_user_preferences_apply_in_chromium(self):
        result = self.render_fixture(
            "size=large&position=bottom-left&dialogue=0&animation=0"
        )

        self.assertFalse(result["reduced"])
        self.assertEqual(urlparse(result["src"]).path, "/aikimi-assets/idle-still.webp")
        self.assertEqual(result["size"], "large")
        self.assertEqual(result["position"], "bottom-left")
        self.assertEqual(result["dialogue"], "off")
        self.assertEqual(result["motion"], "disabled")
        self.assertTrue(result["messageHidden"])
        self.assertEqual(result["characterWidth"], 180)


if __name__ == "__main__":
    unittest.main()
