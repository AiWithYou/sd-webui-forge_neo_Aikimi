from __future__ import annotations

import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from modules_forge.minimax_h3_bridge import (
    H3BridgeError,
    H3JobNotFound,
    H3Request,
    H3_AUDIO_VAE,
    H3_FL_MODEL,
    H3_REF_MODEL,
    H3_TEXT_ENCODER,
    H3_VIDEO_VAE,
    MODE_KEYFRAMES,
    MODE_REFERENCES,
    MODE_TEXT,
    _cleanup_after_terminal,
    _copy_to_comfy_input,
    _is_cancelled_job,
    _mark_cancelled_job,
    _resolve_local_path,
    append_prompt_section,
    build_workflow,
    cache_history_video,
    cleanup_prepared_media,
    cleanup_stale_prepared_media,
    dimensions_for,
    discover_runtime_root,
    extract_history_video,
    history_html,
    inspect_readiness,
    list_history,
    normalize_file_list,
    normalize_loopback_url,
    prepare_media,
    progress_html,
    prompt_template,
    reference_guide_html,
    resolve_runtime_root,
    run_generation,
    server_model_file_status,
    settings_summary_html,
    snap_h3_frames,
    validate_request,
)


class MiniMaxH3GeometryTests(unittest.TestCase):
    def test_duration_snaps_to_17k_plus_5_grid(self):
        self.assertEqual(snap_h3_frames(5), 124)
        self.assertEqual(snap_h3_frames(15), 362)
        for seconds in (5, 5.5, 8, 12.5, 15):
            self.assertEqual(snap_h3_frames(seconds) % 17, 5)

    def test_duration_rejects_untrained_ui_range(self):
        for seconds in (4.5, 15.5):
            with self.subTest(seconds=seconds), self.assertRaises(H3BridgeError):
                snap_h3_frames(seconds)

    def test_every_quality_dimension_is_multiple_of_32(self):
        for quality in ("draft", "preview", "balanced", "native"):
            for aspect in ("21:9", "16:9", "4:3", "1:1", "3:4", "9:16"):
                with self.subTest(quality=quality, aspect=aspect):
                    width, height = dimensions_for(aspect, quality)
                    self.assertEqual(width % 32, 0)
                    self.assertEqual(height % 32, 0)

    def test_default_summary_shows_exact_effective_output(self):
        rendered = settings_summary_html("16:9", "preview", 5, 20)
        self.assertIn("864 × 480", rendered)
        self.assertIn("124 frames", rendered)
        self.assertIn("5.17 sec", rendered)
        self.assertIn("24fps stereo", rendered)

    def test_progress_exposes_accessible_numeric_value(self):
        rendered = progress_html("running", "生成中", 0.42)
        self.assertIn('role="progressbar"', rendered)
        self.assertIn('aria-valuenow="42"', rendered)


class MiniMaxH3ValidationTests(unittest.TestCase):
    def test_text_request_is_valid(self):
        validate_request(H3Request(mode=MODE_TEXT, prompt="A quiet room with natural stereo ambience."))

    def test_keyframe_request_requires_an_image(self):
        with self.assertRaisesRegex(H3BridgeError, "開始画像または終了画像"):
            validate_request(H3Request(mode=MODE_KEYFRAMES, prompt="A slow transition."))

    def test_reference_request_rejects_audio_only(self):
        request = H3Request(
            mode=MODE_REFERENCES,
            prompt="Use <Audio 1>.",
            reference_audios=("voice.wav",),
        )
        with self.assertRaisesRegex(H3BridgeError, "音声だけ"):
            validate_request(request)

    def test_reference_request_enforces_each_limit_and_total(self):
        cases = (
            H3Request(mode=MODE_REFERENCES, prompt="x", reference_images=tuple(f"{i}.png" for i in range(10))),
            H3Request(
                mode=MODE_REFERENCES,
                prompt="x",
                reference_images=tuple(f"{i}.png" for i in range(9)),
                reference_videos=tuple(f"{i}.mp4" for i in range(3)),
                reference_audios=("voice.wav",),
            ),
        )
        for request in cases:
            with self.subTest(count=len(request.reference_images)), self.assertRaises(H3BridgeError):
                validate_request(request)

    def test_backend_url_is_strictly_loopback(self):
        self.assertEqual(normalize_loopback_url("http://localhost:8188/"), "http://localhost:8188")
        self.assertEqual(normalize_loopback_url("http://127.0.0.1:8188"), "http://127.0.0.1:8188")
        for url in (
            "https://127.0.0.1:8188",
            "http://example.com:8188",
            "http://127.0.0.1:8188/api",
            "http://[::1]:8188",
        ):
            with self.subTest(url=url), self.assertRaises(H3BridgeError):
                normalize_loopback_url(url)

    def test_runtime_rejects_unc_path_before_filesystem_access(self):
        with self.assertRaisesRegex(H3BridgeError, "UNC"):
            resolve_runtime_root(r"\\server\share\ComfyUI")

    @unittest.skipUnless(os.name == "nt", "Windows drive classification")
    @mock.patch("modules_forge.minimax_h3_bridge.ctypes.windll.kernel32.GetDriveTypeW", return_value=4)
    def test_runtime_rejects_mapped_network_drive(self, _drive_type):
        with self.assertRaisesRegex(H3BridgeError, "ネットワークドライブ"):
            resolve_runtime_root(r"Z:\ComfyUI")

    def test_file_normalization_accepts_gradio_style_values(self):
        class Named:
            name = "b.png"

        self.assertEqual(normalize_file_list([{"path": "a.png"}, Named(), None]), ("a.png", "b.png"))


class MiniMaxH3WorkflowTests(unittest.TestCase):
    def test_t2v_graph_matches_native_h3_contract(self):
        request = H3Request(mode=MODE_TEXT, prompt="A scene with synchronized rain audio.", seed=42)
        workflow = build_workflow(request, {}, seed=42)

        self.assertEqual(workflow["1"]["inputs"]["unet_name"], H3_FL_MODEL)
        self.assertEqual(workflow["2"]["inputs"], {"clip_name": H3_TEXT_ENCODER, "type": "minimax", "device": "default"})
        self.assertEqual(workflow["3"]["inputs"]["vae_name"], H3_VIDEO_VAE)
        self.assertEqual(workflow["4"]["inputs"]["vae_name"], H3_AUDIO_VAE)
        self.assertEqual(workflow["5"]["class_type"], "MiniMaxH3ImageToVideo")
        self.assertEqual(workflow["5"]["inputs"]["length"], 124)
        self.assertEqual(workflow["10"]["inputs"]["latent_image"], ["5", 1])
        self.assertEqual(workflow["11"]["inputs"]["samples"], ["10", 0])
        self.assertEqual(workflow["12"]["class_type"], "VAEDecodeAudio")
        self.assertEqual(workflow["13"]["inputs"]["audio"], ["12", 0])
        self.assertEqual(workflow["14"]["inputs"]["codec"], "auto")

    def test_keyframe_graph_omits_missing_optional_input(self):
        request = H3Request(mode=MODE_KEYFRAMES, prompt="Move from day to night.", first_frame="first.png")
        workflow = build_workflow(request, {"first_frame": "forge_h3/first.png", "last_frame": None}, seed=7)
        inputs = workflow["5"]["inputs"]
        self.assertEqual(inputs["first_frame"], ["20", 0])
        self.assertNotIn("last_frame", inputs)

    def test_reference_graph_uses_ref_model_dynamic_paths_and_video_audio(self):
        request = H3Request(
            mode=MODE_REFERENCES,
            prompt="Use <Picture 1>, <Audio 1>, <Video 1>, and <Audio 2>.",
            reference_images=("ref.png",),
            reference_videos=("ref.mp4",),
            reference_audios=("voice.wav",),
        )
        media = {
            "images": ["forge_h3/ref.png"],
            "videos": [{"name": "forge_h3/ref.mp4", "has_audio": True, "duration": 4.0}],
            "audios": ["forge_h3/voice.wav"],
        }
        workflow = build_workflow(request, media, seed=9)
        conditioning = workflow["5"]
        self.assertEqual(workflow["1"]["inputs"]["unet_name"], H3_REF_MODEL)
        self.assertEqual(conditioning["class_type"], "MiniMaxH3ReferenceToVideo")
        self.assertIn("ref_images.ref_image_0", conditioning["inputs"])
        self.assertIn("ref_videos.ref_video_0", conditioning["inputs"])
        self.assertIn("ref_video_audios.ref_video_audio_0", conditioning["inputs"])
        self.assertIn("ref_audios.ref_audio_0", conditioning["inputs"])
        self.assertEqual(workflow["8"]["inputs"]["scheduler"], "simple")

    def test_reference_graph_does_not_connect_absent_video_audio(self):
        request = H3Request(
            mode=MODE_REFERENCES,
            prompt="Use <Video 1>.",
            reference_videos=("ref.mp4",),
        )
        media = {"images": [], "videos": [{"name": "forge_h3/ref.mp4", "has_audio": False}], "audios": []}
        workflow = build_workflow(request, media, seed=9)
        inputs = workflow["5"]["inputs"]
        self.assertIn("ref_videos.ref_video_0", inputs)
        self.assertNotIn("ref_video_audios.ref_video_audio_0", inputs)

    def test_reference_graph_rejects_missing_material_tag(self):
        request = H3Request(
            mode=MODE_REFERENCES,
            prompt="Use <Picture 1>.",
            reference_images=("ref.png",),
            reference_videos=("ref.mp4",),
        )
        media = {
            "images": ["forge_h3/ref.png"],
            "videos": [{"name": "forge_h3/ref.mp4", "has_audio": True}],
            "audios": [],
        }
        with self.assertRaisesRegex(H3BridgeError, "<Audio 1>.*<Video 1>"):
            build_workflow(request, media, seed=9)

    def test_reference_graph_rejects_unknown_or_malformed_tag(self):
        media = {"images": ["forge_h3/ref.png"], "videos": [], "audios": []}
        cases = (
            ("Use <Picture 1> and <Video 1>.", "存在しない参照タグ"),
            ("Use <picture 1>.", "形式が不正"),
        )
        for prompt, message in cases:
            request = H3Request(
                mode=MODE_REFERENCES,
                prompt=prompt,
                reference_images=("ref.png",),
            )
            with self.subTest(prompt=prompt), self.assertRaisesRegex(H3BridgeError, message):
                build_workflow(request, media, seed=9)

    @mock.patch("modules_forge.minimax_h3_bridge._copy_to_comfy_input", return_value="forge_h3/ref.mp4")
    @mock.patch("modules_forge.minimax_h3_bridge._validate_media_path", return_value=Path("ref.mp4"))
    @mock.patch("modules_forge.minimax_h3_bridge._probe_media", return_value=(4.0, True, 30.0))
    def test_reference_media_rejects_video_that_is_not_24fps(self, _probe, _validate, _copy):
        request = H3Request(
            mode=MODE_REFERENCES,
            prompt="Use <Audio 1> and <Video 1>.",
            reference_videos=("ref.mp4",),
        )
        with tempfile.TemporaryDirectory() as temporary, self.assertRaisesRegex(H3BridgeError, "24fps"):
            prepare_media(request, Path(temporary))


class MiniMaxH3RuntimeTests(unittest.TestCase):
    def test_server_model_status_uses_exact_loader_choices(self):
        nodes = {
            "UNETLoader": {"input": {"required": {"unet_name": [[H3_FL_MODEL, H3_REF_MODEL]]}}},
            "CLIPLoader": {"input": {"required": {"clip_name": [[H3_TEXT_ENCODER]]}}},
            "VAELoader": {
                "input": {"required": {"vae_name": [[H3_VIDEO_VAE, H3_AUDIO_VAE]]}}
            },
        }
        self.assertTrue(all(server_model_file_status(nodes).values()))
        nodes["CLIPLoader"]["input"]["required"]["clip_name"] = [["other.safetensors"]]
        self.assertFalse(server_model_file_status(nodes)["Qwen3-VL 32B"])

    @mock.patch("modules_forge.minimax_h3_bridge.server_runtime_root")
    @mock.patch("modules_forge.minimax_h3_bridge.ComfyH3Client")
    def test_readiness_rejects_different_runtime_on_same_port(self, client_class, connected_root):
        with tempfile.TemporaryDirectory() as temporary:
            selected = Path(temporary) / "selected"
            other = Path(temporary) / "other"
            for root in (selected, other):
                (root / "models").mkdir(parents=True)
                (root / "main.py").write_text("# test\n", encoding="utf-8")
            client_class.return_value.system_stats.return_value = {"system": {}, "devices": []}
            connected_root.return_value = other.resolve()
            readiness = inspect_readiness(selected, "http://127.0.0.1:8188")
            self.assertFalse(readiness.connected)
            self.assertIn("別のComfyUI", readiness.error or "")

    @mock.patch("modules_forge.minimax_h3_bridge.cleanup_prepared_media")
    @mock.patch("modules_forge.minimax_h3_bridge.ComfyH3Client")
    @mock.patch("modules_forge.minimax_h3_bridge.build_workflow", return_value={"graph": {}})
    @mock.patch("modules_forge.minimax_h3_bridge.prepare_media", return_value={"images": []})
    @mock.patch("modules_forge.minimax_h3_bridge.ensure_ready")
    def test_in_progress_job_is_running_and_terminal_cancel_cleans_inputs(
        self,
        _ready,
        _prepare,
        _workflow,
        client_class,
        cleanup,
    ):
        client = client_class.return_value
        client.submit.return_value = "job-id"
        client.job.return_value = {"status": "in_progress"}
        request = H3Request(mode=MODE_TEXT, prompt="A scene with stereo ambience.")
        updates = run_generation(
            request,
            Path("runtime"),
            "http://127.0.0.1:8188",
            Path("logs"),
            Path("output"),
            poll_seconds=0.5,
        )
        self.assertEqual(next(updates)["stage"], "prepare")
        self.assertEqual(next(updates)["stage"], "queued")
        running = next(updates)
        self.assertEqual(running["stage"], "running")
        self.assertIn("生成中", running["message"])
        client.job.return_value = {"status": "cancelled"}
        with self.assertRaisesRegex(H3BridgeError, "停止"):
            next(updates)
        cleanup.assert_called_once_with({"images": []}, Path("runtime"))

    @mock.patch("modules_forge.minimax_h3_bridge.cleanup_prepared_media")
    @mock.patch("modules_forge.minimax_h3_bridge.ComfyH3Client")
    @mock.patch("modules_forge.minimax_h3_bridge.build_workflow", return_value={"graph": {}})
    @mock.patch("modules_forge.minimax_h3_bridge.prepare_media", return_value={"images": []})
    @mock.patch("modules_forge.minimax_h3_bridge.ensure_ready")
    def test_cancelled_pending_job_404_is_terminal_and_cleans_inputs(
        self,
        _ready,
        _prepare,
        _workflow,
        client_class,
        cleanup,
    ):
        client = client_class.return_value
        client.submit.return_value = "pending-job-id"
        client.job.side_effect = H3JobNotFound("HTTP 404")
        _mark_cancelled_job("pending-job-id")
        updates = run_generation(
            H3Request(mode=MODE_TEXT, prompt="A scene with stereo ambience."),
            Path("runtime"),
            "http://127.0.0.1:8188",
            Path("logs"),
            Path("output"),
        )
        self.assertEqual(next(updates)["stage"], "prepare")
        self.assertEqual(next(updates)["stage"], "queued")
        with self.assertRaisesRegex(H3BridgeError, "停止"):
            next(updates)
        cleanup.assert_called_once_with({"images": []}, Path("runtime"))
        self.assertFalse(_is_cancelled_job("pending-job-id"))

    @mock.patch("modules_forge.minimax_h3_bridge.cleanup_prepared_media")
    @mock.patch("modules_forge.minimax_h3_bridge._schedule_deferred_cleanup")
    @mock.patch("modules_forge.minimax_h3_bridge.ComfyH3Client")
    @mock.patch("modules_forge.minimax_h3_bridge.build_workflow", return_value={"graph": {}})
    @mock.patch("modules_forge.minimax_h3_bridge.prepare_media", return_value={"images": []})
    @mock.patch("modules_forge.minimax_h3_bridge.ensure_ready")
    def test_abandoned_job_requests_cancel_without_deleting_live_inputs(
        self,
        _ready,
        _prepare,
        _workflow,
        client_class,
        schedule_cleanup,
        cleanup,
    ):
        client = client_class.return_value
        client.submit.return_value = "job-id"
        client.job.return_value = {"status": "in_progress"}
        updates = run_generation(
            H3Request(mode=MODE_TEXT, prompt="A scene with stereo ambience."),
            Path("runtime"),
            "http://127.0.0.1:8188",
            Path("logs"),
            Path("output"),
        )
        next(updates)
        next(updates)
        next(updates)
        updates.close()
        client.cancel.assert_called_once_with("job-id")
        schedule_cleanup.assert_called_once_with(client, "job-id", {"images": []}, Path("runtime"))
        cleanup.assert_not_called()

    def test_runtime_is_discovered_from_model_path_yaml(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "ComfyUI"
            model_dir = root / "models" / "diffusion_models"
            model_dir.mkdir(parents=True)
            (root / "main.py").write_text("# test\n", encoding="utf-8")
            (model_dir / H3_FL_MODEL).write_bytes(b"test")
            config = Path(temporary) / "paths.yaml"
            config.write_text(
                "h3:\n  base_path: '" + (root / "models").as_posix() + "'\n",
                encoding="utf-8",
            )
            self.assertEqual(discover_runtime_root(config), root.resolve())

    def test_resolved_unc_path_is_rejected(self):
        with mock.patch.object(Path, "resolve", return_value=Path(r"\\server\share\ComfyUI")):
            with self.assertRaisesRegex(H3BridgeError, "UNC"):
                _resolve_local_path(Path("C:/local/ComfyUI"), "ComfyUI runtime")

    def test_history_result_is_contained_in_runtime_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = root / "output" / "video" / "MiniMax_H3_00001_.mp4"
            result.parent.mkdir(parents=True)
            result.write_bytes(b"video")
            history = {
                "job": {
                    "outputs": {
                        "14": {
                            "images": [
                                {"filename": result.name, "subfolder": "video", "type": "output"}
                            ]
                        }
                    }
                }
            }
            self.assertEqual(extract_history_video(history, "job", root), result.resolve())

    def test_history_result_rejects_path_escape(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            outside = root / "outside.mp4"
            outside.write_bytes(b"video")
            history = {
                "job": {
                    "outputs": {
                        "14": {
                            "images": [
                                {"filename": "outside.mp4", "subfolder": "../..", "type": "output"}
                            ]
                        }
                    }
                }
            }
            with self.assertRaises(H3BridgeError):
                extract_history_video(history, "job", root)

    def test_history_html_escapes_filename(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            video = output / "MiniMax_H3_unsafe&name.mp4"
            video.write_bytes(b"video")
            items = list_history(None, output)
            rendered = history_html(items)
            self.assertNotIn("unsafe&name", rendered)
            self.assertIn("unsafe&amp;name", rendered)

    def test_cache_history_rejects_unlisted_selection(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "output"
            outside = Path(temporary) / "outside.mp4"
            outside.write_bytes(b"video")
            with self.assertRaises(H3BridgeError):
                cache_history_video(str(outside), [], output)


class MiniMaxH3PromptHelperTests(unittest.TestCase):
    def test_prompt_template_preserves_user_text_and_adds_av_sections(self):
        rendered = prompt_template("A cat crosses a quiet kitchen.")
        self.assertIn("A cat crosses a quiet kitchen.", rendered)
        self.assertIn("Camera:", rendered)
        self.assertIn("Audio:", rendered)

    def test_prompt_chip_is_idempotent(self):
        first = append_prompt_section("Scene", "sfx")
        second = append_prompt_section(first, "sfx")
        self.assertEqual(first, second)

    @mock.patch("modules_forge.minimax_h3_bridge._validate_media_path", side_effect=lambda value, expected: Path(value))
    @mock.patch("modules_forge.minimax_h3_bridge._probe_media")
    def test_reference_guide_uses_paired_audio_presentation_order(self, probe, _validate):
        probe.side_effect = [(4.0, True, 24.0), (3.0, True, None)]
        rendered = reference_guide_html(["hero.png"], ["motion.mp4"], ["voice.wav"])
        picture = rendered.index("&lt;Picture 1&gt;")
        paired_audio = rendered.index("&lt;Audio 1&gt;")
        video = rendered.index("&lt;Video 1&gt;")
        standalone_audio = rendered.index("&lt;Audio 2&gt;")
        self.assertLess(picture, paired_audio)
        self.assertLess(paired_audio, video)
        self.assertLess(video, standalone_audio)
        self.assertIn("合計 3/12", rendered)

    def test_reference_guide_reports_audio_only_before_generate(self):
        with mock.patch("modules_forge.minimax_h3_bridge._validate_media_path", return_value=Path("voice.wav")), mock.patch(
            "modules_forge.minimax_h3_bridge._probe_media", return_value=(3.0, True, None)
        ):
            rendered = reference_guide_html(None, None, ["voice.wav"])
        self.assertIn("音声だけでは生成できません", rendered)
        self.assertIn('data-tone="error"', rendered)

    def test_cleanup_only_removes_managed_copies(self):
        with tempfile.TemporaryDirectory() as temporary:
            runtime = Path(temporary)
            managed = runtime / "input" / "forge_h3" / "copy.png"
            outside = runtime / "input" / "keep.png"
            managed.parent.mkdir(parents=True)
            managed.write_bytes(b"copy")
            outside.write_bytes(b"original")
            cleanup_prepared_media(
                {"images": ["forge_h3/copy.png", "keep.png"], "videos": [], "audios": []},
                runtime,
            )
            self.assertFalse(managed.exists())
            self.assertTrue(outside.exists())

    def test_stale_cleanup_only_reaps_owned_old_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            runtime = Path(temporary)
            managed_root = runtime / "input" / "forge_h3"
            managed_root.mkdir(parents=True)
            stale = managed_root / "0123456789ab_old.png"
            recent = managed_root / "abcdef012345_recent.png"
            foreign = managed_root / "keep.png"
            for path in (stale, recent, foreign):
                path.write_bytes(b"test")
            with mock.patch("modules_forge.minimax_h3_bridge.time.time", return_value=10_000.0):
                os.utime(stale, (1.0, 1.0))
                os.utime(recent, (9_999.0, 9_999.0))
                cleanup_stale_prepared_media(runtime, max_age_seconds=60)
            self.assertFalse(stale.exists())
            self.assertTrue(recent.exists())
            self.assertTrue(foreign.exists())

    def test_stale_cleanup_rejects_managed_root_resolved_to_unc(self):
        with tempfile.TemporaryDirectory() as temporary:
            runtime = Path(temporary)
            input_root = runtime / "input"
            input_root.mkdir()
            with mock.patch.object(
                Path,
                "resolve",
                side_effect=[input_root, Path(r"\\server\share\forge_h3")],
            ):
                with self.assertRaisesRegex(H3BridgeError, "UNC"):
                    cleanup_stale_prepared_media(runtime)

    def test_stale_cleanup_rejects_managed_root_outside_input(self):
        with tempfile.TemporaryDirectory() as temporary:
            runtime = Path(temporary)
            input_root = runtime / "input"
            outside = runtime / "outside"
            input_root.mkdir()
            outside.mkdir()
            with mock.patch.object(Path, "resolve", side_effect=[input_root, outside]):
                with self.assertRaisesRegex(H3BridgeError, "外"):
                    cleanup_stale_prepared_media(runtime)

    def test_managed_copy_gets_current_mtime_instead_of_source_mtime(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "old-reference.png"
            runtime = root / "ComfyUI"
            (runtime / "input").mkdir(parents=True)
            source.write_bytes(b"reference")
            old_time = time.time() - 30 * 24 * 60 * 60
            os.utime(source, (old_time, old_time))
            copied_at = time.time()
            relative_name = _copy_to_comfy_input(source, runtime)
            copied = runtime / "input" / relative_name
            self.assertGreaterEqual(copied.stat().st_mtime, copied_at - 1.0)

    def test_deferred_cleanup_treats_known_cancel_404_as_terminal(self):
        with tempfile.TemporaryDirectory() as temporary:
            runtime = Path(temporary)
            managed = runtime / "input" / "forge_h3" / "copy.png"
            managed.parent.mkdir(parents=True)
            managed.write_bytes(b"copy")
            client = mock.Mock()
            client.job.side_effect = H3JobNotFound("HTTP 404")
            _mark_cancelled_job("cancelled-job")
            _cleanup_after_terminal(
                client,
                "cancelled-job",
                {"images": ["forge_h3/copy.png"]},
                runtime,
                wait_seconds=0.1,
            )
            self.assertFalse(managed.exists())
            self.assertFalse(_is_cancelled_job("cancelled-job"))


if __name__ == "__main__":
    unittest.main()
