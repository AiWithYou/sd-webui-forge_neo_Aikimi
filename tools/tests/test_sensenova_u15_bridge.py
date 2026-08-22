import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
from PIL import Image
from safetensors.torch import save_file

from modules_forge import sensenova_u15_bridge as bridge


class SenseNovaRequestTests(unittest.TestCase):
    def test_official_resolution_and_edit_auto_resolution(self):
        self.assertEqual(
            bridge.parse_resolution("2720x1536", bridge.MODE_TEXT), (2720, 1536)
        )
        self.assertEqual(
            bridge.parse_resolution("auto", bridge.MODE_EDIT), (None, None)
        )
        with self.assertRaisesRegex(bridge.SenseNovaBridgeError, "画像編集"):
            bridge.parse_resolution("auto", bridge.MODE_TEXT)

    def test_multiple_gallery_images_keep_order_and_flatten_alpha(self):
        red = Image.new("RGBA", (16, 16), (255, 0, 0, 128))
        blue = Image.new("RGB", (12, 10), (0, 0, 255))
        images = bridge.normalize_gallery_images([(red, None), (blue, "second")])
        self.assertEqual([image.size for image in images], [(16, 16), (12, 10)])
        self.assertEqual([image.mode for image in images], ["RGB", "RGB"])
        self.assertGreater(images[0].getpixel((0, 0))[0], images[0].getpixel((0, 0))[1])
        self.assertEqual(images[1].getpixel((0, 0)), (0, 0, 255))

    def test_edit_accepts_sixty_four_ordered_images(self):
        images = tuple(
            Image.new("RGB", (32, 32), (index, 0, 0))
            for index in range(bridge.MAX_REFERENCE_IMAGES)
        )
        request = bridge.SenseNovaRequest(
            mode=bridge.MODE_EDIT,
            prompt="Use every image in order.",
            input_images=images,
            width=None,
            height=None,
            vram_mode="unrestricted",
        )
        bridge.validate_request(request)

        with self.assertRaisesRegex(bridge.SenseNovaBridgeError, "最大64枚"):
            bridge.validate_request(
                bridge.SenseNovaRequest(
                    mode=bridge.MODE_EDIT,
                    prompt="Too many references",
                    input_images=images + (Image.new("RGB", (32, 32)),),
                )
            )

    def test_low_vram_profile_rejects_the_failed_four_megapixel_workload(self):
        images = (
            Image.new("RGB", (32, 32), (255, 0, 0)),
            Image.new("RGB", (32, 32), (0, 0, 255)),
        )
        unsafe = bridge.SenseNovaRequest(
            mode=bridge.MODE_EDIT,
            prompt="Combine two references",
            input_images=images,
            width=1664,
            height=2496,
            input_max_pixels="auto",
            vram_mode="low",
        )
        with self.assertRaisesRegex(bridge.SenseNovaBridgeError, "24GB安全"):
            bridge.validate_request(unsafe)

        unrestricted = bridge.SenseNovaRequest(
            **{**unsafe.__dict__, "vram_mode": "unrestricted"}
        )
        bridge.validate_request(unrestricted)

    def test_low_vram_defaults_match_the_measured_3090_profile(self):
        request = bridge.SenseNovaRequest(
            mode=bridge.MODE_EDIT,
            prompt="Combine two references",
            input_images=(
                Image.new("RGB", (32, 32)),
                Image.new("RGB", (32, 32)),
            ),
            width=None,
            height=None,
        )
        bridge.validate_request(request)
        self.assertEqual(request.target_pixels, 2048 * 2048)
        self.assertEqual(request.input_max_pixels, 512 * 512)
        self.assertEqual(request.vram_mode, "low")

        three_references = bridge.SenseNovaRequest(
            **{
                **request.__dict__,
                "input_images": request.input_images
                + (Image.new("RGB", (32, 32)),),
            }
        )
        with self.assertRaisesRegex(bridge.SenseNovaBridgeError, "参照3枚"):
            bridge.validate_request(three_references)

        two_k = bridge.SenseNovaRequest(
            mode=bridge.MODE_EDIT,
            prompt="2K edit",
            input_images=request.input_images,
            width=2048,
            height=2048,
        )
        bridge.validate_request(two_k)

    def test_text_mode_rejects_hidden_reference_data(self):
        request = bridge.SenseNovaRequest(
            mode=bridge.MODE_TEXT,
            prompt="A lighthouse",
            input_images=(Image.new("RGB", (32, 32)),),
        )
        with self.assertRaisesRegex(bridge.SenseNovaBridgeError, "画像編集"):
            bridge.validate_request(request)

    def test_worker_payload_pins_final_checkpoint_revision(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            request = bridge.SenseNovaRequest(
                mode=bridge.MODE_TEXT,
                prompt="A lighthouse",
                checkpoint=str(root / "model.safetensors"),
            )
            payload = bridge._request_payload(
                request,
                input_paths=[],
                output_path=root / "output.png",
                metadata_path=root / "output.json",
            )
        self.assertEqual(payload["model_path"], bridge.DEFAULT_MODEL_ID)
        self.assertEqual(
            payload["checkpoint_revision"], bridge.CHECKPOINT_REVISION
        )
        self.assertEqual(payload["quantization"], bridge.QUANT_INT8_CONVROT)

    def test_convrot_requires_safetensors_extension(self):
        request = bridge.SenseNovaRequest(
            mode=bridge.MODE_TEXT,
            prompt="A lighthouse",
            checkpoint="weights.gguf",
        )
        with self.assertRaisesRegex(bridge.SenseNovaBridgeError, r"\.safetensors"):
            bridge.validate_request(request)

    def test_convrot_rejects_preview_model_config(self):
        request = bridge.SenseNovaRequest(
            mode=bridge.MODE_TEXT,
            prompt="A lighthouse",
            model_path="sensenova/SenseNova-U1.5-8B-MoT-Preview",
        )
        with self.assertRaisesRegex(bridge.SenseNovaBridgeError, "正式版"):
            bridge.validate_request(request)


class SenseNovaRuntimeTests(unittest.TestCase):
    @staticmethod
    def _make_runtime(root: Path) -> Path:
        source = root / "runtime-final"
        package = source / "SenseNova" / "src" / "sensenova_u1"
        inference = source / "SenseNova" / "examples" / "editing"
        config = source / "SenseNova-U1.5-8B-MoT"
        package.mkdir(parents=True)
        inference.mkdir(parents=True)
        config.mkdir(parents=True)
        (package / "__init__.py").write_text("", encoding="utf-8")
        (inference / "inference.py").write_text("", encoding="utf-8")
        (config / "config.json").write_text("{}", encoding="utf-8")
        (source / ".sensenova_runtime_revision").write_text(
            bridge.SOURCE_REVISION + "\n", encoding="utf-8"
        )
        return source

    def test_runtime_status_checks_revision_size_and_convrot_header(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = self._make_runtime(root)
            checkpoint = root / "model.safetensors"
            marker = torch.tensor(
                list(json.dumps({"format": "int8_tensorwise"}).encode("utf-8")),
                dtype=torch.uint8,
            )
            save_file(
                {
                    "layer.comfy_quant": marker,
                    "fm_modules.vision_model_mot_gen.embeddings.patch_embedding.weight": torch.ones(1),
                },
                checkpoint,
            )
            Path(str(checkpoint) + ".sha256").write_text(
                bridge.CONVROT_SHA256 + "  model.safetensors\n", encoding="utf-8"
            )

            with (
                mock.patch.object(
                    bridge, "CONVROT_EXPECTED_BYTES", checkpoint.stat().st_size
                ),
                mock.patch.object(bridge, "EXPECTED_CONVROT_LAYERS", 1),
            ):
                status = bridge.inspect_runtime(source, checkpoint=checkpoint)

            self.assertTrue(status.ready)
            self.assertTrue(status.source_ready)
            self.assertTrue(status.checkpoint_ready)

    def test_partial_convrot_is_reported_without_being_ready(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = self._make_runtime(root)
            checkpoint = root / "model.safetensors"
            Path(str(checkpoint) + ".part").write_bytes(b"partial")
            status = bridge.inspect_runtime(source, checkpoint=checkpoint)
            self.assertFalse(status.ready)
            self.assertEqual(status.partial_bytes, 7)
            self.assertTrue(
                any("ダウンロード中" in message for message in status.messages)
            )

    def test_parallel_convrot_chunks_are_counted_in_download_progress(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = self._make_runtime(root)
            checkpoint = root / "model.safetensors"
            chunks = Path(str(checkpoint) + ".part.chunks")
            chunks.mkdir()
            (chunks / "chunk-001.bin").write_bytes(b"1234")
            (chunks / "chunk-002.bin").write_bytes(b"56789")
            (chunks / "chunk-002.bin.resume").write_bytes(b"abc")
            status = bridge.inspect_runtime(source, checkpoint=checkpoint)
            self.assertFalse(status.ready)
            self.assertEqual(status.partial_bytes, 12)


class SenseNovaWorkerBridgeTests(unittest.TestCase):
    def test_worker_events_are_parsed_strictly(self):
        event = bridge._parse_event(
            'SENSENOVA_EVENT {"stage":"sampling","progress":0.5}'
        )
        self.assertEqual(event["stage"], "sampling")
        self.assertIsNone(bridge._parse_event("ordinary log line"))
        self.assertIsNone(bridge._parse_event("SENSENOVA_EVENT not-json"))

    def test_job_cleanup_rejects_paths_outside_cache_jobs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cache = root / "cache"
            outside = root / "outside"
            outside.mkdir()
            with self.assertRaisesRegex(bridge.SenseNovaBridgeError, "許可範囲外"):
                bridge._cleanup_job_directory(outside, cache)
            self.assertTrue(outside.is_dir())

    def test_run_generation_stages_ordered_inputs_and_verifies_outputs(self):
        fake_worker = r"""
import json
import pathlib
import sys

request_path = pathlib.Path(sys.argv[sys.argv.index("--request") + 1])
payload = json.loads(request_path.read_text(encoding="utf-8"))
print('SENSENOVA_EVENT ' + json.dumps({"stage": "loading", "message": "fake load", "progress": 0.2}), flush=True)
names = [pathlib.Path(path).name for path in payload["input_images"]]
pathlib.Path(payload["output_path"]).write_bytes(b"fake-png")
metadata = {"input_image_names": names, "input_image_count": len(names)}
pathlib.Path(payload["metadata_path"]).write_text(json.dumps(metadata), encoding="utf-8")
print('SENSENOVA_EVENT ' + json.dumps({"stage": "complete", "message": "fake done", "progress": 1.0}), flush=True)
"""
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            worker = root / "worker.py"
            worker.write_text(fake_worker, encoding="utf-8")
            output = root / "outputs"
            cache = root / "cache"
            logs = root / "logs"
            request = bridge.SenseNovaRequest(
                mode=bridge.MODE_EDIT,
                prompt="Combine in order",
                quantization=bridge.QUANT_INT8_CONVROT,
                checkpoint="weights.safetensors",
                input_images=(
                    Image.new("RGB", (20, 20), (255, 0, 0)),
                    Image.new("RGB", (20, 20), (0, 0, 255)),
                ),
                width=1024,
                height=1024,
            )
            ready = bridge.RuntimeStatus(
                ready=True,
                source_ready=True,
                dependencies_ready=True,
                checkpoint_ready=True,
                source_path=root,
                checkpoint_path=root / "weights.safetensors",
                messages=(),
            )
            with (
                mock.patch.object(bridge, "inspect_runtime", return_value=ready),
                mock.patch.object(bridge, "_release_forge_vram"),
            ):
                updates = list(
                    bridge.run_generation(
                        request,
                        output_directory=output,
                        cache_directory=cache,
                        log_directory=logs,
                        worker_path=worker,
                    )
                )

            completed = [update for update in updates if update["stage"] == "complete"]
            self.assertEqual(len(completed), 1)
            self.assertEqual(completed[0]["metadata"]["input_image_count"], 2)
            self.assertEqual(
                completed[0]["metadata"]["input_image_names"],
                ["reference_01.png", "reference_02.png"],
            )
            self.assertTrue(Path(completed[0]["path"]).is_file())
            self.assertEqual(list((cache / "jobs").glob("*")), [])

    def test_cancel_terminates_the_active_worker_and_cleans_the_job(self):
        fake_worker = r"""
import json
import pathlib
import sys
import time

request_path = pathlib.Path(sys.argv[sys.argv.index("--request") + 1])
json.loads(request_path.read_text(encoding="utf-8"))
print('SENSENOVA_EVENT ' + json.dumps({"stage": "loading", "message": "waiting", "progress": 0.1}), flush=True)
time.sleep(60)
"""
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            worker = root / "worker.py"
            worker.write_text(fake_worker, encoding="utf-8")
            cache = root / "cache"
            request = bridge.SenseNovaRequest(
                mode=bridge.MODE_TEXT,
                prompt="A lighthouse",
                quantization=bridge.QUANT_INT8_CONVROT,
                checkpoint="weights.safetensors",
                width=1024,
                height=1024,
            )
            ready = bridge.RuntimeStatus(
                ready=True,
                source_ready=True,
                dependencies_ready=True,
                checkpoint_ready=True,
                source_path=root,
                checkpoint_path=root / "weights.safetensors",
                messages=(),
            )
            with (
                mock.patch.object(bridge, "inspect_runtime", return_value=ready),
                mock.patch.object(bridge, "_release_forge_vram"),
            ):
                updates = bridge.run_generation(
                    request,
                    output_directory=root / "outputs",
                    cache_directory=cache,
                    log_directory=root / "logs",
                    worker_path=worker,
                )
                loading = None
                for update in updates:
                    if update["stage"] == "loading":
                        loading = update
                        break
                self.assertIsNotNone(loading)
                message = bridge.cancel_generation(loading["job_id"])
                self.assertIn("キャンセル", message)
                with self.assertRaises(bridge.SenseNovaGenerationCancelled):
                    list(updates)
            self.assertEqual(list((cache / "jobs").glob("*")), [])


if __name__ == "__main__":
    unittest.main()
