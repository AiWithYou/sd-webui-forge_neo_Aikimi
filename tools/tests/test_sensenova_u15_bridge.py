import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

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

    def test_edit_accepts_eight_ordered_images(self):
        images = tuple(Image.new("RGB", (32, 32), (index, 0, 0)) for index in range(8))
        request = bridge.SenseNovaRequest(
            mode=bridge.MODE_EDIT,
            prompt="Use every image in order.",
            input_images=images,
            width=None,
            height=None,
        )
        bridge.validate_request(request)

    def test_text_mode_rejects_hidden_reference_data(self):
        request = bridge.SenseNovaRequest(
            mode=bridge.MODE_TEXT,
            prompt="A lighthouse",
            input_images=(Image.new("RGB", (32, 32)),),
        )
        with self.assertRaisesRegex(bridge.SenseNovaBridgeError, "画像編集"):
            bridge.validate_request(request)

    def test_q8_requires_gguf_extension(self):
        request = bridge.SenseNovaRequest(
            mode=bridge.MODE_TEXT,
            prompt="A lighthouse",
            gguf_checkpoint="weights.safetensors",
        )
        with self.assertRaisesRegex(bridge.SenseNovaBridgeError, r"\.gguf"):
            bridge.validate_request(request)

    def test_q8_rejects_a_different_model_config(self):
        request = bridge.SenseNovaRequest(
            mode=bridge.MODE_TEXT,
            prompt="A lighthouse",
            model_path="sensenova/SenseNova-U1.5-8B-MoT",
        )
        with self.assertRaisesRegex(bridge.SenseNovaBridgeError, "Preview"):
            bridge.validate_request(request)


class SenseNovaRuntimeTests(unittest.TestCase):
    def test_runtime_status_checks_revision_size_and_gguf_header(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "runtime" / "src"
            package = source / "sensenova_u1"
            package.mkdir(parents=True)
            (package / "__init__.py").write_text("", encoding="utf-8")
            (source.parent / ".sensenova_revision").write_text(
                bridge.SOURCE_REVISION + "\n", encoding="utf-8"
            )
            gguf = root / "model.gguf"
            gguf.write_bytes(b"GGUF" + (3).to_bytes(4, "little"))
            Path(str(gguf) + ".sha256").write_text(
                bridge.Q8_SHA256 + "  model.gguf\n", encoding="utf-8"
            )

            with mock.patch.object(bridge, "Q8_EXPECTED_BYTES", 8):
                status = bridge.inspect_runtime(
                    source, quantization=bridge.QUANT_Q8, gguf_checkpoint=gguf
                )

            self.assertTrue(status.ready)
            self.assertTrue(status.source_ready)
            self.assertTrue(status.quantization_ready)

    def test_partial_q8_is_reported_without_being_ready(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "runtime" / "src"
            package = source / "sensenova_u1"
            package.mkdir(parents=True)
            (package / "__init__.py").write_text("", encoding="utf-8")
            (source.parent / ".sensenova_revision").write_text(
                bridge.SOURCE_REVISION + "\n", encoding="utf-8"
            )
            gguf = root / "model.gguf"
            Path(str(gguf) + ".part").write_bytes(b"partial")
            status = bridge.inspect_runtime(
                source, quantization=bridge.QUANT_Q8, gguf_checkpoint=gguf
            )
            self.assertFalse(status.ready)
            self.assertEqual(status.partial_bytes, 7)
            self.assertTrue(
                any("ダウンロード中" in message for message in status.messages)
            )

    def test_parallel_q8_chunks_are_counted_in_download_progress(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "runtime" / "src"
            package = source / "sensenova_u1"
            package.mkdir(parents=True)
            (package / "__init__.py").write_text("", encoding="utf-8")
            (source.parent / ".sensenova_revision").write_text(
                bridge.SOURCE_REVISION + "\n", encoding="utf-8"
            )
            gguf = root / "model.gguf"
            chunks = Path(str(gguf) + ".part.chunks")
            chunks.mkdir()
            (chunks / "chunk-001.bin").write_bytes(b"1234")
            (chunks / "chunk-002.bin").write_bytes(b"56789")
            status = bridge.inspect_runtime(
                source, quantization=bridge.QUANT_Q8, gguf_checkpoint=gguf
            )
            self.assertFalse(status.ready)
            self.assertEqual(status.partial_bytes, 9)


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
                quantization=bridge.QUANT_BF16,
                gguf_checkpoint="",
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
                quantization_ready=True,
                source_path=root,
                gguf_path=None,
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
                quantization=bridge.QUANT_BF16,
                gguf_checkpoint="",
                width=1024,
                height=1024,
            )
            ready = bridge.RuntimeStatus(
                ready=True,
                source_ready=True,
                dependencies_ready=True,
                quantization_ready=True,
                source_path=root,
                gguf_path=None,
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
