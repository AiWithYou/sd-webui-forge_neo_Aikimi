from __future__ import annotations

import hashlib
import unittest
from pathlib import Path
from unittest.mock import patch

import gradio
from fastapi import FastAPI
from fastapi.testclient import TestClient

from modules import gradio_frontend_compat


class GradioFrontendCompatibilityTests(unittest.TestCase):
    def test_webui_validates_patch_before_starting_gradio_listener(self):
        root = Path(__file__).resolve().parents[2]
        source = (root / "webui.py").read_text(encoding="utf-8")

        self.assertLess(
            source.index("patched_tabs = build_patched_tabs_asset()"),
            source.index("app, local_url, share_url = shared.demo.launch("),
        )

    def test_exact_pinned_asset_is_patched_without_modifying_site_packages(self):
        asset_path = (
            Path(gradio.__file__).resolve().parent
            / "templates"
            / "frontend"
            / "assets"
            / gradio_frontend_compat.TABS_ASSET_NAME
        )
        before = asset_path.read_bytes()

        patched = gradio_frontend_compat.build_patched_tabs_asset()

        self.assertEqual(hashlib.sha256(before).hexdigest(), patched.original_sha256)
        self.assertEqual(before, asset_path.read_bytes())
        self.assertNotEqual(before, patched.content)
        self.assertNotIn(gradio_frontend_compat._ORIGINAL_OVERFLOW_FUNCTION, patched.content)
        self.assertIn(gradio_frontend_compat._NO_OVERFLOW_FUNCTION, patched.content)
        self.assertNotIn(gradio_frontend_compat._ORIGINAL_INITIAL_TAB_SYNC, patched.content)
        self.assertIn(gradio_frontend_compat._STATIC_INITIAL_TAB_SYNC, patched.content)
        self.assertNotIn(gradio_frontend_compat._ORIGINAL_TAB_REGISTRATION, patched.content)
        self.assertIn(gradio_frontend_compat._STATIC_TAB_REGISTRATION, patched.content)
        self.assertEqual(hashlib.sha256(patched.content).hexdigest(), patched.patched_sha256)
        self.assertEqual(patched.patched_sha256, gradio_frontend_compat.PATCHED_ASSET_SHA256)

    def test_version_mismatch_fails_closed(self):
        with patch.object(gradio, "__version__", "6.18.0"):
            with self.assertRaisesRegex(
                gradio_frontend_compat.GradioFrontendCompatibilityError,
                "not supported",
            ):
                gradio_frontend_compat.build_patched_tabs_asset()

    def test_hash_mismatch_fails_closed(self):
        with patch.object(gradio_frontend_compat, "ORIGINAL_ASSET_SHA256", "0" * 64):
            with self.assertRaisesRegex(
                gradio_frontend_compat.GradioFrontendCompatibilityError,
                "does not match",
            ):
                gradio_frontend_compat.build_patched_tabs_asset()

    def test_tampered_prebuilt_asset_fails_closed(self):
        asset = gradio_frontend_compat.build_patched_tabs_asset()
        tampered = gradio_frontend_compat.PatchedFrontendAsset(
            filename=asset.filename,
            content=asset.content + b" ",
            original_sha256=asset.original_sha256,
            patched_sha256=asset.patched_sha256,
        )

        with self.assertRaisesRegex(
            gradio_frontend_compat.GradioFrontendCompatibilityError,
            "integrity validation",
        ):
            gradio_frontend_compat.install_gradio_tabs_compatibility_route(
                FastAPI(),
                tampered,
            )

    def test_exact_route_precedes_generic_assets_and_supports_etag(self):
        app = FastAPI()

        @app.get("/assets/{path:path}")
        async def generic_asset(path: str):
            return {"generic": path}

        asset = gradio_frontend_compat.build_patched_tabs_asset()
        installed = gradio_frontend_compat.install_gradio_tabs_compatibility_route(app, asset)
        self.assertIs(installed, asset)
        client = TestClient(app)

        response = client.get(f"/assets/{asset.filename}")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, asset.content)
        self.assertEqual(response.headers["content-length"], str(len(asset.content)))
        self.assertEqual(response.headers["etag"], f'"{asset.patched_sha256}"')
        self.assertEqual(response.headers["x-aikimi-gradio-compat"], "6.17.3")
        self.assertEqual(response.headers["x-content-type-options"], "nosniff")

        head = client.head(f"/assets/{asset.filename}")
        self.assertEqual(head.status_code, 200)
        self.assertEqual(head.content, b"")
        self.assertEqual(head.headers["content-length"], str(len(asset.content)))
        self.assertEqual(head.headers["etag"], f'"{asset.patched_sha256}"')

        cached = client.get(
            f"/assets/{asset.filename}",
            headers={"If-None-Match": f'"{asset.patched_sha256}"'},
        )
        self.assertEqual(cached.status_code, 304)
        self.assertEqual(client.get("/assets/other.js").json(), {"generic": "other.js"})

    def test_route_is_idempotent_and_works_with_asgi_root_path(self):
        app = FastAPI(root_path="/aikimi")
        app.router.routes.insert(0, object())
        first = gradio_frontend_compat.install_gradio_tabs_compatibility_route(app)
        second = gradio_frontend_compat.install_gradio_tabs_compatibility_route(app)

        self.assertIs(first, second)
        self.assertEqual(
            sum(getattr(route, "name", None) == gradio_frontend_compat.PATCH_ROUTE_NAME for route in app.router.routes),
            1,
        )
        response = TestClient(app, root_path="/aikimi").get(f"/assets/{first.filename}")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, first.content)


if __name__ == "__main__":
    unittest.main()
