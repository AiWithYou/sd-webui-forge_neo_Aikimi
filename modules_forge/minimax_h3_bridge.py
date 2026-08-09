from __future__ import annotations

import atexit
import ctypes
import html
import json
import logging
import os
import re
import secrets
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, Sequence

import yaml


H3_FPS = 24
H3_MIN_SECONDS = 5.0
H3_MAX_SECONDS = 15.0
H3_SERVER_URL = "http://127.0.0.1:8188"
H3_FL_MODEL = "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
H3_REF_MODEL = "minimax_h3_ref2va_pruned_int8_convrot.safetensors"
H3_TEXT_ENCODER = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
H3_VIDEO_VAE = "minimax_h3_video_vae_fp16.safetensors"
H3_AUDIO_VAE = "minimax_h3_audio_vae_fp32.safetensors"

MODE_TEXT = "text"
MODE_KEYFRAMES = "keyframes"
MODE_REFERENCES = "references"
MODES = {MODE_TEXT, MODE_KEYFRAMES, MODE_REFERENCES}

ASPECTS = ("21:9", "16:9", "4:3", "1:1", "3:4", "9:16")
QUALITY_DIMENSIONS: dict[str, dict[str, tuple[int, int]]] = {
    "draft": {
        "21:9": (672, 288),
        "16:9": (608, 352),
        "4:3": (512, 384),
        "1:1": (448, 448),
        "3:4": (384, 512),
        "9:16": (352, 608),
    },
    "preview": {
        "21:9": (960, 416),
        "16:9": (864, 480),
        "4:3": (736, 544),
        "1:1": (640, 640),
        "3:4": (544, 736),
        "9:16": (480, 864),
    },
    "balanced": {
        "21:9": (1120, 480),
        "16:9": (960, 544),
        "4:3": (832, 640),
        "1:1": (736, 736),
        "3:4": (640, 832),
        "9:16": (544, 960),
    },
    "native": {
        "21:9": (1344, 576),
        "16:9": (1344, 768),
        "4:3": (1024, 768),
        "1:1": (768, 768),
        "3:4": (768, 1024),
        "9:16": (768, 1344),
    },
}

REQUIRED_NODE_TYPES = {
    "BasicGuider",
    "BasicScheduler",
    "CLIPLoader",
    "CreateVideo",
    "GetVideoComponents",
    "KSamplerSelect",
    "LoadAudio",
    "LoadImage",
    "LoadVideo",
    "MiniMaxH3ImageToVideo",
    "MiniMaxH3ReferenceToVideo",
    "RandomNoise",
    "SamplerCustomAdvanced",
    "SaveVideo",
    "UNETLoader",
    "VAEDecode",
    "VAEDecodeAudio",
    "VAELoader",
}

MODEL_FILES = {
    "FL2VA": ("diffusion_models", H3_FL_MODEL),
    "Ref2VA": ("diffusion_models", H3_REF_MODEL),
    "Qwen3-VL 32B": ("text_encoders", H3_TEXT_ENCODER),
    "Video VAE": ("vae", H3_VIDEO_VAE),
    "Audio VAE": ("vae", H3_AUDIO_VAE),
}

_MANAGED_PROCESS: subprocess.Popen | None = None
_MANAGED_PROCESS_IDENTITY: tuple[Path, str] | None = None
_PROCESS_LOCK = threading.Lock()
_CANCELLED_JOB_LOCK = threading.Lock()
_CANCELLED_JOB_IDS: set[str] = set()
_LOG = logging.getLogger(__name__)
_TERMINAL_JOB_STATUSES = {"completed", "success", "failed", "error", "cancelled", "canceled"}


class H3BridgeError(RuntimeError):
    """A user-actionable MiniMax H3 bridge error."""


class H3JobNotFound(H3BridgeError):
    """A ComfyUI job endpoint no longer knows the requested prompt ID."""


def _mark_cancelled_job(prompt_id: str) -> None:
    with _CANCELLED_JOB_LOCK:
        _CANCELLED_JOB_IDS.add(prompt_id)


def _is_cancelled_job(prompt_id: str) -> bool:
    with _CANCELLED_JOB_LOCK:
        return prompt_id in _CANCELLED_JOB_IDS


def _clear_cancelled_job(prompt_id: str) -> None:
    with _CANCELLED_JOB_LOCK:
        _CANCELLED_JOB_IDS.discard(prompt_id)


@dataclass(frozen=True)
class H3Request:
    mode: str
    prompt: str
    aspect: str = "16:9"
    quality: str = "preview"
    duration_seconds: float = H3_MIN_SECONDS
    steps: int = 20
    seed: int = -1
    scheduler: str = "simple"
    ref_image_size: str = "match"
    first_frame: str | None = None
    last_frame: str | None = None
    reference_images: tuple[str, ...] = ()
    reference_videos: tuple[str, ...] = ()
    reference_audios: tuple[str, ...] = ()

    @property
    def dimensions(self) -> tuple[int, int]:
        return dimensions_for(self.aspect, self.quality)

    @property
    def frame_count(self) -> int:
        return snap_h3_frames(self.duration_seconds)

    @property
    def effective_seconds(self) -> float:
        return self.frame_count / H3_FPS

    @property
    def resolved_seed(self) -> int:
        if self.seed == -1:
            return secrets.randbelow(2**53)
        return self.seed


@dataclass(frozen=True)
class RuntimeReadiness:
    runtime_root: Path | None
    server_url: str
    connected: bool
    comfy_version: str | None = None
    gpu_name: str | None = None
    vram_gib: float | None = None
    model_files: dict[str, bool] = field(default_factory=dict)
    server_model_files: dict[str, bool] = field(default_factory=dict)
    missing_nodes: tuple[str, ...] = ()
    error: str | None = None

    @property
    def ready_for_fl2va(self) -> bool:
        required = ("FL2VA", "Qwen3-VL 32B", "Video VAE", "Audio VAE")
        return (
            self.connected
            and not self.missing_nodes
            and all(self.model_files.get(name) for name in required)
            and all(self.server_model_files.get(name) for name in required)
        )

    @property
    def ready_for_ref2va(self) -> bool:
        return (
            self.ready_for_fl2va
            and bool(self.model_files.get("Ref2VA"))
            and bool(self.server_model_files.get("Ref2VA"))
        )


@dataclass(frozen=True)
class HistoryItem:
    path: Path
    modified_at: float
    source: str

    @property
    def label(self) -> str:
        stamp = datetime.fromtimestamp(self.modified_at).strftime("%m/%d %H:%M")
        return f"{stamp} · {self.path.name}"


def snap_h3_frames(seconds: float) -> int:
    try:
        seconds = float(seconds)
    except (TypeError, ValueError) as exc:
        raise H3BridgeError("長さは秒数で指定してください。") from exc
    if not H3_MIN_SECONDS <= seconds <= H3_MAX_SECONDS:
        raise H3BridgeError("MiniMax H3 の長さは 5〜15 秒で指定してください。")
    requested = max(5, int(seconds * H3_FPS + 0.5))
    return requested + (5 - requested % 17) % 17


def dimensions_for(aspect: str, quality: str) -> tuple[int, int]:
    if quality not in QUALITY_DIMENSIONS:
        raise H3BridgeError(f"未対応の品質プリセットです: {quality}")
    try:
        width, height = QUALITY_DIMENSIONS[quality][aspect]
    except KeyError as exc:
        raise H3BridgeError(f"未対応のアスペクト比です: {aspect}") from exc
    if width % 32 or height % 32:
        raise AssertionError("MiniMax H3 dimensions must be multiples of 32")
    return width, height


def normalize_file_list(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, os.PathLike)):
        return (os.fspath(value),)
    result: list[str] = []
    for item in value:
        if item is None:
            continue
        if isinstance(item, dict):
            item = item.get("path") or item.get("name")
        elif hasattr(item, "name"):
            item = item.name
        if item:
            result.append(os.fspath(item))
    return tuple(result)


def validate_request(request: H3Request) -> None:
    if request.mode not in MODES:
        raise H3BridgeError("生成モードを選択してください。")
    if not request.prompt or not request.prompt.strip():
        raise H3BridgeError("映像と音のプロンプトを入力してください。")
    if len(request.prompt) > 20_000:
        raise H3BridgeError("プロンプトが長すぎます。20,000文字以内にしてください。")
    dimensions_for(request.aspect, request.quality)
    snap_h3_frames(request.duration_seconds)
    if not 1 <= int(request.steps) <= 100:
        raise H3BridgeError("Steps は 1〜100 で指定してください。")
    if int(request.seed) < -1 or int(request.seed) >= 2**63:
        raise H3BridgeError("Seed は -1、または 0〜2^63-1 で指定してください。")
    if request.scheduler not in {"simple", "beta", "normal"}:
        raise H3BridgeError("Scheduler は simple / beta / normal から選択してください。")
    if request.ref_image_size not in {"match", "max"}:
        raise H3BridgeError("参照画像サイズは match または max を選択してください。")

    if request.mode == MODE_KEYFRAMES and not (request.first_frame or request.last_frame):
        raise H3BridgeError("キーフレームモードでは開始画像または終了画像を追加してください。")

    if request.mode == MODE_REFERENCES:
        image_count = len(request.reference_images)
        video_count = len(request.reference_videos)
        audio_count = len(request.reference_audios)
        if image_count > 9:
            raise H3BridgeError("参照画像は最大9枚です。")
        if video_count > 3:
            raise H3BridgeError("参照動画は最大3本です。")
        if audio_count > 3:
            raise H3BridgeError("参照音声は最大3本です。")
        if image_count + video_count + audio_count > 12:
            raise H3BridgeError("参照素材は合計12個までです。")
        if not image_count and not video_count:
            raise H3BridgeError("参照モードには画像または動画が必要です。音声だけでは生成できません。")


def normalize_loopback_url(url: str) -> str:
    parsed = urllib.parse.urlsplit((url or "").strip())
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost"}:
        raise H3BridgeError("H3 backend は安全のため 127.0.0.1 / localhost の HTTP URL のみ利用できます。")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise H3BridgeError("H3 backend URL に認証情報、query、fragmentは指定できません。")
    if parsed.path not in {"", "/"}:
        raise H3BridgeError("H3 backend URL にはパスを付けないでください。")
    try:
        port = parsed.port or 80
    except ValueError as exc:
        raise H3BridgeError("H3 backend のポート番号が不正です。") from exc
    return f"http://{parsed.hostname}:{port}"


def _local_filesystem_path(value: str | os.PathLike[str], label: str) -> Path:
    expanded = os.path.expandvars(os.path.expanduser(os.fspath(value)))
    normalized = expanded.replace("/", "\\")
    if normalized.startswith("\\\\"):
        raise H3BridgeError(f"{label}にUNCネットワークパスは使用できません。ローカルディスクを指定してください。")
    path = Path(expanded)
    if os.name == "nt" and path.drive:
        drive_root = path.drive.rstrip("\\/") + "\\"
        drive_type = int(ctypes.windll.kernel32.GetDriveTypeW(drive_root))
        if drive_type == 4:
            raise H3BridgeError(
                f"{label}にネットワークドライブ {path.drive} は使用できません。ローカルディスクを指定してください。"
            )
        if drive_type in {0, 1}:
            raise H3BridgeError(f"{label}のドライブ種別を確認できません: {path.drive}")
    return path


def _resolve_local_path(value: str | os.PathLike[str], label: str) -> Path:
    unresolved = _local_filesystem_path(value, label)
    resolved = unresolved.resolve()
    _local_filesystem_path(resolved, label)
    return resolved


def _runtime_from_models_path(models_path: Path) -> Path:
    return _resolve_local_path(models_path, "モデルパス").parent


def discover_runtime_root(config_path: str | os.PathLike[str]) -> Path | None:
    config = Path(config_path)
    if not config.is_file():
        return None
    try:
        data = yaml.safe_load(config.read_text(encoding="utf-8")) or {}
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise H3BridgeError(f"モデルパス設定を読めません: {exc}") from exc
    if not isinstance(data, dict):
        raise H3BridgeError("モデルパス設定の形式が不正です。")

    candidates: list[Path] = []
    for entry in data.values():
        if not isinstance(entry, dict) or not entry.get("base_path"):
            continue
        models_path = _local_filesystem_path(str(entry["base_path"]), "モデルパス")
        runtime = _runtime_from_models_path(models_path)
        if runtime not in candidates:
            candidates.append(runtime)

    for runtime in candidates:
        if (runtime / "main.py").is_file() and (runtime / "models" / "diffusion_models" / H3_FL_MODEL).is_file():
            return runtime
    return None


def resolve_runtime_root(value: str | os.PathLike[str] | None) -> Path:
    if not value:
        raise H3BridgeError("ComfyUI のフォルダーを指定してください。")
    root = _resolve_local_path(value, "ComfyUI runtime")
    if not (root / "main.py").is_file():
        raise H3BridgeError(f"ComfyUI main.py が見つかりません: {root}")
    _resolve_local_path(root / "main.py", "ComfyUI main.py")
    if not (root / "models").is_dir():
        raise H3BridgeError(f"ComfyUI models フォルダーが見つかりません: {root}")
    return root


def model_file_status(runtime_root: Path | None) -> dict[str, bool]:
    if runtime_root is None:
        return {name: False for name in MODEL_FILES}
    return {
        name: (runtime_root / "models" / directory / filename).is_file()
        for name, (directory, filename) in MODEL_FILES.items()
    }


def server_model_file_status(nodes: dict[str, Any]) -> dict[str, bool]:
    loader_inputs = {
        "UNETLoader": "unet_name",
        "CLIPLoader": "clip_name",
        "VAELoader": "vae_name",
    }
    choices: dict[str, set[str]] = {}
    for node_name, input_name in loader_inputs.items():
        try:
            values = nodes[node_name]["input"]["required"][input_name][0]
        except (KeyError, IndexError, TypeError):
            values = []
        choices[node_name] = {str(value) for value in values if isinstance(value, str)}
    return {
        "FL2VA": H3_FL_MODEL in choices["UNETLoader"],
        "Ref2VA": H3_REF_MODEL in choices["UNETLoader"],
        "Qwen3-VL 32B": H3_TEXT_ENCODER in choices["CLIPLoader"],
        "Video VAE": H3_VIDEO_VAE in choices["VAELoader"],
        "Audio VAE": H3_AUDIO_VAE in choices["VAELoader"],
    }


def _version_tuple(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in re.findall(r"\d+", version)[:3])


def server_runtime_root(server_url: str) -> Path | None:
    normalized_url = normalize_loopback_url(server_url)
    port = urllib.parse.urlsplit(normalized_url).port or 80
    try:
        import psutil

        listeners = [
            connection
            for connection in psutil.net_connections(kind="tcp")
            if connection.status == psutil.CONN_LISTEN
            and connection.laddr
            and int(connection.laddr.port) == port
        ]
    except (ImportError, OSError) as exc:
        raise H3BridgeError(f"H3 backend processを確認できません: {exc}") from exc
    if not listeners:
        return None

    loopback = [
        connection
        for connection in listeners
        if str(connection.laddr.ip).split("%", 1)[0] in {"127.0.0.1", "::1"}
    ]
    if not loopback:
        raise H3BridgeError(
            f"port {port} のprocessはloopback専用ではありません。H3 backendは127.0.0.1だけで起動してください。"
        )
    connection = loopback[0]
    if connection.pid is None:
        raise H3BridgeError(f"port {port} を使用中のprocess IDを確認できません。")
    try:
        process = psutil.Process(connection.pid)
        cwd = Path(process.cwd())
        for argument in process.cmdline()[1:]:
            candidate = Path(argument)
            if candidate.name.lower() != "main.py":
                continue
            main_path = candidate if candidate.is_absolute() else cwd / candidate
            if main_path.is_file():
                return _resolve_local_path(main_path.parent, "接続中のComfyUI runtime")
        if (cwd / "main.py").is_file():
            return _resolve_local_path(cwd, "接続中のComfyUI runtime")
    except (OSError, psutil.Error) as exc:
        raise H3BridgeError(f"port {port} のComfyUI processを確認できません: {exc}") from exc
    raise H3BridgeError(f"port {port} を使用中のprocessがComfyUI main.pyから起動されたことを確認できません。")


def _same_local_path(left: Path, right: Path) -> bool:
    return os.path.normcase(os.fspath(left.resolve())) == os.path.normcase(os.fspath(right.resolve()))


class ComfyH3Client:
    def __init__(self, server_url: str = H3_SERVER_URL, timeout: float = 15.0):
        self.server_url = normalize_loopback_url(server_url)
        self.timeout = timeout
        self.client_id = str(uuid.uuid4())
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def _request_json(self, path: str, payload: dict[str, Any] | None = None, timeout: float | None = None) -> Any:
        url = f"{self.server_url}{path}"
        body = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(url, data=body, headers=headers, method="POST" if body is not None else "GET")
        try:
            with self._opener.open(request, timeout=timeout or self.timeout) as response:
                raw = response.read()
        except urllib.error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            try:
                parsed = json.loads(details)
                details = _format_comfy_error(parsed)
            except json.JSONDecodeError:
                pass
            error_type = H3JobNotFound if exc.code == 404 else H3BridgeError
            raise error_type(f"ComfyUI が要求を拒否しました (HTTP {exc.code}): {details[:1200]}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise H3BridgeError(f"ComfyUI に接続できません: {exc}") from exc
        if not raw:
            return {}
        try:
            return json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise H3BridgeError("ComfyUI から不正な応答を受信しました。") from exc

    def system_stats(self) -> dict[str, Any]:
        value = self._request_json("/system_stats")
        if not isinstance(value, dict):
            raise H3BridgeError("ComfyUI system_stats の形式が不正です。")
        return value

    def object_info(self, timeout: float = 8.0) -> dict[str, Any]:
        value = self._request_json("/object_info", timeout=timeout)
        if not isinstance(value, dict):
            raise H3BridgeError("ComfyUI object_info の形式が不正です。")
        return value

    def submit(self, workflow: dict[str, Any]) -> str:
        prompt_id = str(uuid.uuid4())
        response = self._request_json(
            "/prompt",
            {
                "prompt": workflow,
                "client_id": self.client_id,
                "prompt_id": prompt_id,
                "extra_data": {"workflow_id": "forge-neo-minimax-h3-studio"},
            },
            timeout=90,
        )
        result = response.get("prompt_id") if isinstance(response, dict) else None
        if not result:
            raise H3BridgeError("ComfyUI が prompt_id を返しませんでした。")
        return str(result)

    def job(self, prompt_id: str) -> dict[str, Any]:
        value = self._request_json(f"/api/jobs/{urllib.parse.quote(prompt_id)}")
        if not isinstance(value, dict):
            raise H3BridgeError("ComfyUI job status の形式が不正です。")
        return value

    def history(self, prompt_id: str) -> dict[str, Any]:
        value = self._request_json(f"/history/{urllib.parse.quote(prompt_id)}")
        if not isinstance(value, dict):
            raise H3BridgeError("ComfyUI history の形式が不正です。")
        return value

    def cancel(self, prompt_id: str) -> None:
        if prompt_id:
            self._request_json(f"/api/jobs/{urllib.parse.quote(prompt_id)}/cancel", {})
            _mark_cancelled_job(prompt_id)


def _format_comfy_error(value: Any) -> str:
    if not isinstance(value, dict):
        return str(value)
    error = value.get("error")
    if isinstance(error, dict):
        parts = [error.get("message"), error.get("details")]
        node_errors = value.get("node_errors")
        if node_errors:
            parts.append(json.dumps(node_errors, ensure_ascii=False))
        return " / ".join(str(part) for part in parts if part)
    return str(error or value)


def inspect_readiness(
    runtime_root: Path | None,
    server_url: str = H3_SERVER_URL,
    object_timeout: float = 8.0,
) -> RuntimeReadiness:
    files = model_file_status(runtime_root)
    try:
        client = ComfyH3Client(server_url, timeout=2.0)
        stats = client.system_stats()
        connected_root = server_runtime_root(client.server_url)
        if connected_root is None:
            raise H3BridgeError("ComfyUI APIは応答しましたが、対応するloopback processを確認できません。")
        if runtime_root is None:
            raise H3BridgeError(f"接続中のComfyUI runtimeを選択してください: {connected_root}")
        selected_root = resolve_runtime_root(runtime_root)
        if not _same_local_path(connected_root, selected_root):
            raise H3BridgeError(
                "別のComfyUIがこのportを使用中です。"
                f" 選択: {selected_root} / 接続中: {connected_root}"
            )
        system = stats.get("system") or {}
        devices = stats.get("devices") or []
        version = str(system.get("comfyui_version") or "")
        if _version_tuple(version) < (0, 30, 0):
            raise H3BridgeError(f"MiniMax H3 には ComfyUI 0.30.0 以降が必要です（現在 {version or '不明'}）。")
        nodes = client.object_info(timeout=object_timeout)
        missing_nodes = tuple(sorted(REQUIRED_NODE_TYPES - set(nodes)))
        server_files = server_model_file_status(nodes)
        device = devices[0] if devices else {}
        return RuntimeReadiness(
            runtime_root=runtime_root,
            server_url=client.server_url,
            connected=True,
            comfy_version=version or None,
            gpu_name=str(device.get("name") or "") or None,
            vram_gib=(float(device["vram_total"]) / 1024**3) if device.get("vram_total") else None,
            model_files=files,
            server_model_files=server_files,
            missing_nodes=missing_nodes,
        )
    except H3BridgeError as exc:
        return RuntimeReadiness(
            runtime_root=runtime_root,
            server_url=server_url,
            connected=False,
            model_files=files,
            error=str(exc),
        )


def _python_for_runtime(runtime_root: Path) -> Path:
    candidates = (
        runtime_root.parent / ".venv" / "Scripts" / "python.exe",
        runtime_root / ".venv" / "Scripts" / "python.exe",
        runtime_root.parent / "python_embeded" / "python.exe",
    )
    for candidate in candidates:
        if candidate.is_file():
            return _resolve_local_path(candidate, "ComfyUI Python")
    raise H3BridgeError(
        "ComfyUI 用 Python が見つかりません。runtimeの親またはruntime内に .venv を用意してください。"
    )


def start_runtime(runtime_root: Path, server_url: str, log_directory: Path, wait_seconds: float = 120.0) -> RuntimeReadiness:
    global _MANAGED_PROCESS, _MANAGED_PROCESS_IDENTITY
    runtime_root = resolve_runtime_root(runtime_root)
    normalized_url = normalize_loopback_url(server_url)
    parsed = urllib.parse.urlsplit(normalized_url)
    if parsed.hostname not in {"127.0.0.1", "localhost"}:
        raise H3BridgeError("ローカル以外のComfyUIは起動できません。")
    port = parsed.port or 80
    identity = (runtime_root, normalized_url)

    current = inspect_readiness(runtime_root, normalized_url)
    if current.connected:
        return current
    listening_root = server_runtime_root(normalized_url)
    if listening_root is not None and not _same_local_path(listening_root, runtime_root):
        raise H3BridgeError(
            f"別のComfyUIがport {port}を使用中です。選択: {runtime_root} / 接続中: {listening_root}"
        )
    if listening_root is not None and current.error and "接続できません" not in current.error:
        raise H3BridgeError(current.error)

    python = _python_for_runtime(runtime_root)
    log_directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    stdout_path = log_directory / f"minimax-h3-{stamp}.stdout.log"
    stderr_path = log_directory / f"minimax-h3-{stamp}.stderr.log"
    command = [
        os.fspath(python),
        "main.py",
        "--listen",
        "127.0.0.1",
        "--port",
        str(port),
        "--disable-all-custom-nodes",
        "--reserve-vram",
        "2",
        "--vram-headroom",
        "12",
        "--preview-method",
        "none",
        "--fast-disk",
        "--disable-pinned-memory",
        "--disable-async-offload",
    ]

    with _PROCESS_LOCK:
        if _MANAGED_PROCESS is not None and _MANAGED_PROCESS.poll() is None:
            if _MANAGED_PROCESS_IDENTITY != identity:
                managed_root, managed_url = _MANAGED_PROCESS_IDENTITY or (Path("不明"), "不明")
                raise H3BridgeError(
                    "H3 Studioが別のruntimeを管理中です。"
                    f" 実行中: {managed_root} ({managed_url}) / 要求: {runtime_root} ({normalized_url})"
                )
        elif listening_root is None:
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            with stdout_path.open("ab") as stdout, stderr_path.open("ab") as stderr:
                _MANAGED_PROCESS = subprocess.Popen(
                    command,
                    cwd=runtime_root,
                    stdin=subprocess.DEVNULL,
                    stdout=stdout,
                    stderr=stderr,
                    creationflags=creationflags,
                )
                _MANAGED_PROCESS_IDENTITY = identity

    deadline = time.monotonic() + wait_seconds
    last = current
    while time.monotonic() < deadline:
        if _MANAGED_PROCESS is not None and _MANAGED_PROCESS.poll() is not None:
            raise H3BridgeError(
                f"ComfyUI が起動直後に終了しました。ログを確認してください: {stderr_path}"
            )
        time.sleep(1.0)
        last = inspect_readiness(runtime_root, normalized_url)
        if last.connected:
            return last
    raise H3BridgeError(
        f"ComfyUI の起動を {int(wait_seconds)} 秒待ちましたが接続できません。ログ: {stderr_path}"
    )


def _stop_managed_runtime() -> None:
    global _MANAGED_PROCESS, _MANAGED_PROCESS_IDENTITY
    with _PROCESS_LOCK:
        process = _MANAGED_PROCESS
        _MANAGED_PROCESS = None
        _MANAGED_PROCESS_IDENTITY = None
    if process is not None and process.poll() is None:
        try:
            process.terminate()
            process.wait(timeout=10)
        except (OSError, subprocess.TimeoutExpired):
            pass


atexit.register(_stop_managed_runtime)


def ensure_ready(runtime_root: Path, server_url: str, log_directory: Path) -> RuntimeReadiness:
    readiness = inspect_readiness(runtime_root, server_url)
    if not readiness.connected:
        readiness = start_runtime(runtime_root, server_url, log_directory)
    if readiness.missing_nodes:
        raise H3BridgeError("ComfyUI に H3 必須ノードがありません: " + ", ".join(readiness.missing_nodes))
    if not readiness.ready_for_fl2va:
        required = ("FL2VA", "Qwen3-VL 32B", "Video VAE", "Audio VAE")
        missing_local = [name for name in required if not readiness.model_files.get(name)]
        missing_server = [name for name in required if not readiness.server_model_files.get(name)]
        details = []
        if missing_local:
            details.append("filesystem: " + ", ".join(missing_local))
        if missing_server:
            details.append("接続先のmodel一覧: " + ", ".join(missing_server))
        raise H3BridgeError("H3 FL2VA の必須モデルが不足しています: " + " / ".join(details))
    return readiness


def _validate_media_path(path_value: str, expected: str) -> Path:
    path = Path(path_value).resolve()
    if not path.is_file():
        raise H3BridgeError(f"参照ファイルが見つかりません: {path}")
    allowed = {
        "image": {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"},
        "video": {".mp4", ".mov", ".mkv", ".webm", ".avi"},
        "audio": {".wav", ".flac", ".mp3", ".m4a", ".aac", ".ogg", ".opus"},
    }[expected]
    if path.suffix.lower() not in allowed:
        raise H3BridgeError(f"{expected}として未対応の形式です: {path.suffix or '(拡張子なし)'}")
    return path


def _probe_media(path: Path) -> tuple[float, bool, float | None]:
    try:
        import av

        with av.open(os.fspath(path)) as container:
            duration = float(container.duration or 0) / float(av.time_base)
            has_audio = bool(container.streams.audio)
            video_stream = container.streams.video[0] if container.streams.video else None
            video_rate = None
            if video_stream is not None:
                video_rate = video_stream.average_rate or video_stream.base_rate
            video_fps = float(video_rate) if video_rate else None
    except Exception as exc:
        raise H3BridgeError(f"参照メディアを解析できません: {path.name}: {exc}") from exc
    return duration, has_audio, video_fps


def _managed_input_root(runtime_root: Path, create: bool = False) -> tuple[Path, Path]:
    input_root = _resolve_local_path(runtime_root / "input", "ComfyUI input")
    managed_candidate = input_root / "forge_h3"
    if create:
        managed_candidate.mkdir(parents=True, exist_ok=True)
    managed_root = _resolve_local_path(managed_candidate, "MiniMax H3 input")
    if input_root not in managed_root.parents:
        raise H3BridgeError("MiniMax H3 input フォルダーが ComfyUI input の外を指しています。")
    return input_root, managed_root


def _copy_to_comfy_input(source: Path, runtime_root: Path) -> str:
    input_root, target_dir = _managed_input_root(runtime_root, create=True)
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", source.name).strip("._") or "reference"
    target_name = f"{uuid.uuid4().hex[:12]}_{safe_name}"
    target = _resolve_local_path(target_dir / target_name, "MiniMax H3 input copy")
    if input_root not in target.parents:
        raise H3BridgeError("ComfyUI input フォルダー外への書き込みを拒否しました。")
    shutil.copy2(source, target)
    os.utime(target, None)
    return f"forge_h3/{target_name}"


def _prepared_media_names(prepared: dict[str, Any]) -> Iterator[str]:
    for key in ("first_frame", "last_frame"):
        if prepared.get(key):
            yield str(prepared[key])
    for name in prepared.get("images") or []:
        yield str(name)
    for video in prepared.get("videos") or []:
        if isinstance(video, dict) and video.get("name"):
            yield str(video["name"])
    for name in prepared.get("audios") or []:
        yield str(name)


def cleanup_prepared_media(prepared: dict[str, Any], runtime_root: Path) -> None:
    input_root, managed_root = _managed_input_root(runtime_root)
    for relative_name in set(_prepared_media_names(prepared)):
        candidate = _resolve_local_path(input_root / relative_name, "MiniMax H3 input copy")
        if managed_root not in candidate.parents:
            continue
        last_error: OSError | None = None
        for attempt in range(3):
            try:
                candidate.unlink(missing_ok=True)
                last_error = None
                break
            except OSError as exc:
                last_error = exc
                if attempt < 2:
                    time.sleep(0.1 * (attempt + 1))
        if last_error is not None:
            _LOG.warning("MiniMax H3 input copy could not be removed: %s: %s", candidate, last_error)


def cleanup_stale_prepared_media(
    runtime_root: Path,
    max_age_seconds: float = 7 * 24 * 60 * 60,
) -> None:
    _, managed_root = _managed_input_root(runtime_root)
    if not managed_root.is_dir():
        return
    cutoff = time.time() - max(60.0, float(max_age_seconds))
    stale_pattern = re.compile(r"^[0-9a-f]{12}_.+")
    for candidate in managed_root.iterdir():
        try:
            if not candidate.is_file() or not stale_pattern.fullmatch(candidate.name):
                continue
            if candidate.stat().st_mtime >= cutoff:
                continue
        except OSError:
            continue
        cleanup_prepared_media({"images": [f"forge_h3/{candidate.name}"]}, runtime_root)


def _cleanup_after_terminal(
    client: ComfyH3Client,
    prompt_id: str,
    prepared: dict[str, Any],
    runtime_root: Path,
    wait_seconds: float = 600.0,
) -> None:
    deadline = time.monotonic() + wait_seconds
    while time.monotonic() < deadline:
        try:
            status = str(client.job(prompt_id).get("status") or "").lower()
            if status in _TERMINAL_JOB_STATUSES:
                try:
                    cleanup_prepared_media(prepared, runtime_root)
                finally:
                    _clear_cancelled_job(prompt_id)
                return
        except H3JobNotFound:
            if _is_cancelled_job(prompt_id):
                try:
                    cleanup_prepared_media(prepared, runtime_root)
                finally:
                    _clear_cancelled_job(prompt_id)
                return
        except H3BridgeError:
            pass
        time.sleep(1.0)
    _LOG.warning("MiniMax H3 deferred input cleanup timed out for job %s", prompt_id)


def _schedule_deferred_cleanup(
    client: ComfyH3Client,
    prompt_id: str,
    prepared: dict[str, Any],
    runtime_root: Path,
) -> None:
    worker = threading.Thread(
        target=_cleanup_after_terminal,
        args=(client, prompt_id, prepared, runtime_root),
        name=f"minimax-h3-cleanup-{prompt_id[:8]}",
        daemon=True,
    )
    worker.start()


def prepare_media(request: H3Request, runtime_root: Path) -> dict[str, Any]:
    validate_request(request)
    prepared: dict[str, Any] = {
        "first_frame": None,
        "last_frame": None,
        "images": [],
        "videos": [],
        "audios": [],
    }
    try:
        if request.mode == MODE_KEYFRAMES:
            if request.first_frame:
                source = _validate_media_path(request.first_frame, "image")
                prepared["first_frame"] = _copy_to_comfy_input(source, runtime_root)
            if request.last_frame:
                source = _validate_media_path(request.last_frame, "image")
                prepared["last_frame"] = _copy_to_comfy_input(source, runtime_root)
            return prepared

        if request.mode != MODE_REFERENCES:
            return prepared

        total_timed_seconds = 0.0
        for value in request.reference_images:
            source = _validate_media_path(value, "image")
            prepared["images"].append(_copy_to_comfy_input(source, runtime_root))
        for value in request.reference_videos:
            source = _validate_media_path(value, "video")
            duration, has_audio, video_fps = _probe_media(source)
            if not 2.0 <= duration <= 15.0:
                raise H3BridgeError(f"参照動画 {source.name} は 2〜15 秒にしてください（{duration:.1f}秒）。")
            if video_fps is None or abs(video_fps - H3_FPS) > 0.05:
                shown_fps = "不明" if video_fps is None else f"{video_fps:.3f}"
                raise H3BridgeError(
                    f"参照動画 {source.name} は24fpsに変換してください（現在 {shown_fps}fps）。"
                )
            total_timed_seconds += duration
            prepared["videos"].append(
                {
                    "name": _copy_to_comfy_input(source, runtime_root),
                    "has_audio": has_audio,
                    "duration": duration,
                    "fps": video_fps,
                }
            )
        for value in request.reference_audios:
            source = _validate_media_path(value, "audio")
            duration, _, _ = _probe_media(source)
            if not 2.0 <= duration <= 15.0:
                raise H3BridgeError(f"参照音声 {source.name} は 2〜15 秒にしてください（{duration:.1f}秒）。")
            total_timed_seconds += duration
            prepared["audios"].append(_copy_to_comfy_input(source, runtime_root))
        if total_timed_seconds > 15.0 + 1e-6:
            raise H3BridgeError(f"参照動画と音声の合計は15秒までです（現在 {total_timed_seconds:.1f}秒）。")
        return prepared
    except Exception:
        cleanup_prepared_media(prepared, runtime_root)
        raise


def _node(class_type: str, **inputs: Any) -> dict[str, Any]:
    return {"class_type": class_type, "inputs": inputs}


def expected_reference_tags(prepared_media: dict[str, Any]) -> tuple[str, ...]:
    tags = [f"<Picture {index}>" for index, _ in enumerate(prepared_media.get("images") or [], 1)]
    audio_index = 1
    for video_index, video in enumerate(prepared_media.get("videos") or [], 1):
        if video.get("has_audio"):
            tags.append(f"<Audio {audio_index}>")
            audio_index += 1
        tags.append(f"<Video {video_index}>")
    for _ in prepared_media.get("audios") or []:
        tags.append(f"<Audio {audio_index}>")
        audio_index += 1
    return tuple(tags)


def validate_reference_prompt_tags(prompt: str, prepared_media: dict[str, Any]) -> None:
    expected = expected_reference_tags(prepared_media)
    related_tags = re.findall(r"<(?:Picture|Video|Audio)\b[^<>]*>", prompt, flags=re.IGNORECASE)
    exact_tags = re.findall(r"<(?:Picture|Video|Audio) [1-9][0-9]*>", prompt)
    malformed = sorted(set(related_tags) - set(exact_tags))
    if malformed:
        raise H3BridgeError(
            "参照タグの形式が不正です: "
            + ", ".join(malformed)
            + "。大文字小文字と半角スペースを含め、<Picture 1> の形式で入力してください。"
        )

    expected_set = set(expected)
    supplied_set = set(exact_tags)
    unknown = [tag for tag in exact_tags if tag not in expected_set]
    if unknown:
        raise H3BridgeError(
            "選択した素材に存在しない参照タグです: " + ", ".join(dict.fromkeys(unknown))
        )
    missing = [tag for tag in expected if tag not in supplied_set]
    if missing:
        raise H3BridgeError(
            "Prompt に未使用の参照素材があります。次のタグを追加してください: " + ", ".join(missing)
        )


def build_workflow(request: H3Request, prepared_media: dict[str, Any], seed: int | None = None) -> dict[str, Any]:
    validate_request(request)
    width, height = request.dimensions
    seed = request.resolved_seed if seed is None else int(seed)
    model_name = H3_REF_MODEL if request.mode == MODE_REFERENCES else H3_FL_MODEL

    workflow: dict[str, Any] = {
        "1": _node("UNETLoader", unet_name=model_name, weight_dtype="default"),
        "2": _node("CLIPLoader", clip_name=H3_TEXT_ENCODER, type="minimax", device="default"),
        "3": _node("VAELoader", vae_name=H3_VIDEO_VAE),
        "4": _node("VAELoader", vae_name=H3_AUDIO_VAE),
        "6": _node("RandomNoise", noise_seed=seed),
        "7": _node("KSamplerSelect", sampler_name="res_multistep"),
        "8": _node(
            "BasicScheduler",
            model=["1", 0],
            scheduler=request.scheduler,
            steps=int(request.steps),
            denoise=1.0,
        ),
        "9": _node("BasicGuider", model=["1", 0], conditioning=["5", 0]),
        "10": _node(
            "SamplerCustomAdvanced",
            noise=["6", 0],
            guider=["9", 0],
            sampler=["7", 0],
            sigmas=["8", 0],
            latent_image=["5", 1],
        ),
        "11": _node("VAEDecode", samples=["10", 0], vae=["3", 0]),
        "12": _node("VAEDecodeAudio", samples=["10", 0], vae=["4", 0]),
        "13": _node("CreateVideo", images=["11", 0], audio=["12", 0], fps=H3_FPS, bit_depth=8),
        "14": _node(
            "SaveVideo",
            video=["13", 0],
            filename_prefix="video/Forge_Neo_MiniMax_H3",
            format="auto",
            codec="auto",
        ),
    }

    conditioning_inputs: dict[str, Any]
    if request.mode == MODE_REFERENCES:
        validate_reference_prompt_tags(request.prompt, prepared_media)
        conditioning_inputs = {
            "clip": ["2", 0],
            "vae": ["3", 0],
            "audio_vae": ["4", 0],
            "prompt": request.prompt.strip(),
            "width": width,
            "height": height,
            "length": request.frame_count,
            "ref_image_size": request.ref_image_size,
        }
        next_id = 20
        for index, image_name in enumerate(prepared_media.get("images") or []):
            node_id = str(next_id)
            next_id += 1
            workflow[node_id] = _node("LoadImage", image=image_name)
            conditioning_inputs[f"ref_images.ref_image_{index}"] = [node_id, 0]
        for index, video in enumerate(prepared_media.get("videos") or []):
            load_id = str(next_id)
            components_id = str(next_id + 1)
            next_id += 2
            workflow[load_id] = _node("LoadVideo", file=video["name"])
            workflow[components_id] = _node("GetVideoComponents", video=[load_id, 0])
            conditioning_inputs[f"ref_videos.ref_video_{index}"] = [components_id, 0]
            if video.get("has_audio"):
                conditioning_inputs[f"ref_video_audios.ref_video_audio_{index}"] = [components_id, 1]
        for index, audio_name in enumerate(prepared_media.get("audios") or []):
            node_id = str(next_id)
            next_id += 1
            workflow[node_id] = _node("LoadAudio", audio=audio_name)
            conditioning_inputs[f"ref_audios.ref_audio_{index}"] = [node_id, 0]
        workflow["5"] = _node("MiniMaxH3ReferenceToVideo", **conditioning_inputs)
    else:
        conditioning_inputs = {
            "clip": ["2", 0],
            "vae": ["3", 0],
            "prompt": request.prompt.strip(),
            "width": width,
            "height": height,
            "length": request.frame_count,
        }
        if request.mode == MODE_KEYFRAMES:
            if prepared_media.get("first_frame"):
                workflow["20"] = _node("LoadImage", image=prepared_media["first_frame"])
                conditioning_inputs["first_frame"] = ["20", 0]
            if prepared_media.get("last_frame"):
                workflow["21"] = _node("LoadImage", image=prepared_media["last_frame"])
                conditioning_inputs["last_frame"] = ["21", 0]
        workflow["5"] = _node("MiniMaxH3ImageToVideo", **conditioning_inputs)

    return workflow


def extract_history_video(history: dict[str, Any], prompt_id: str, runtime_root: Path) -> Path:
    item = history.get(prompt_id)
    if not isinstance(item, dict):
        raise H3BridgeError("完了履歴に生成結果がありません。")
    outputs = item.get("outputs") or {}
    candidates: list[dict[str, Any]] = []
    for output in outputs.values():
        if not isinstance(output, dict):
            continue
        for key in ("images", "videos"):
            values = output.get(key) or []
            if isinstance(values, list):
                candidates.extend(value for value in values if isinstance(value, dict))
    for candidate in reversed(candidates):
        filename = str(candidate.get("filename") or "")
        if Path(filename).suffix.lower() not in {".mp4", ".webm", ".mov", ".mkv"}:
            continue
        folder_type = candidate.get("type") or "output"
        base = runtime_root / ("output" if folder_type == "output" else "temp")
        path = (base / str(candidate.get("subfolder") or "") / filename).resolve()
        if base.resolve() not in path.parents or not path.is_file():
            continue
        return path
    raise H3BridgeError("ComfyUI の完了履歴から動画ファイルを特定できません。")


def _execution_error(job: dict[str, Any]) -> str:
    error = job.get("execution_error")
    if isinstance(error, dict):
        return str(error.get("exception_message") or error.get("message") or error)
    return str(error or "生成処理が失敗しました。ComfyUIログを確認してください。")


def mirror_result(source: Path, output_directory: Path, request: H3Request, prompt_id: str, seed: int) -> Path:
    output_directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    target = output_directory / f"MiniMax_H3_{stamp}_{prompt_id[:8]}{source.suffix.lower()}"
    shutil.copy2(source, target)
    metadata = {
        "model": "MiniMax H3",
        "prompt_id": prompt_id,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "mode": request.mode,
        "prompt": request.prompt,
        "aspect": request.aspect,
        "quality": request.quality,
        "dimensions": list(request.dimensions),
        "requested_seconds": request.duration_seconds,
        "frames": request.frame_count,
        "steps": request.steps,
        "seed": seed,
        "scheduler": request.scheduler,
        "source": os.fspath(source),
    }
    target.with_suffix(".json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return target


def run_generation(
    request: H3Request,
    runtime_root: Path,
    server_url: str,
    log_directory: Path,
    output_directory: Path,
    poll_seconds: float = 2.0,
) -> Iterator[dict[str, Any]]:
    validate_request(request)
    readiness = ensure_ready(runtime_root, server_url, log_directory)
    if request.mode == MODE_REFERENCES and not readiness.ready_for_ref2va:
        raise H3BridgeError("参照モード用 Ref2VA モデルがありません。")
    cleanup_stale_prepared_media(runtime_root)
    yield {"stage": "prepare", "message": "入力素材を検証しています", "progress": 0.06, "prompt_id": ""}
    prepared = prepare_media(request, runtime_root)
    client: ComfyH3Client | None = None
    prompt_id = ""
    terminal = False
    try:
        seed = request.resolved_seed
        workflow = build_workflow(request, prepared, seed=seed)
        client = ComfyH3Client(server_url)
        prompt_id = client.submit(workflow)
        yield {
            "stage": "queued",
            "message": "ComfyUI のキューに追加しました",
            "progress": 0.12,
            "prompt_id": prompt_id,
            "seed": seed,
        }
        started = time.monotonic()
        while True:
            try:
                job = client.job(prompt_id)
            except H3JobNotFound:
                if _is_cancelled_job(prompt_id):
                    terminal = True
                    _clear_cancelled_job(prompt_id)
                    raise H3BridgeError("生成を停止しました。") from None
                raise
            status = str(job.get("status") or "pending").lower()
            elapsed = time.monotonic() - started
            if status in {"completed", "success"}:
                terminal = True
                _clear_cancelled_job(prompt_id)
                history = client.history(prompt_id)
                source = extract_history_video(history, prompt_id, runtime_root)
                target = mirror_result(source, output_directory, request, prompt_id, seed)
                yield {
                    "stage": "complete",
                    "message": "音声付き動画を保存しました",
                    "progress": 1.0,
                    "prompt_id": prompt_id,
                    "seed": seed,
                    "path": os.fspath(target),
                    "elapsed": elapsed,
                }
                return
            if status in {"failed", "error"}:
                terminal = True
                _clear_cancelled_job(prompt_id)
                raise H3BridgeError(_execution_error(job))
            if status in {"cancelled", "canceled"}:
                terminal = True
                _clear_cancelled_job(prompt_id)
                raise H3BridgeError("生成を停止しました。")
            running = status in {"running", "in_progress"}
            message = "生成中 — RTX 3090 では長時間かかる場合があります" if running else "キューで待機しています"
            progress = min(0.88, 0.16 + elapsed / 7200.0) if running else 0.13
            yield {
                "stage": "running" if running else "queued",
                "message": message,
                "progress": progress,
                "prompt_id": prompt_id,
                "seed": seed,
                "elapsed": elapsed,
            }
            time.sleep(max(0.5, poll_seconds))
    finally:
        if prompt_id and not terminal and client is not None:
            try:
                client.cancel(prompt_id)
            except H3BridgeError as exc:
                _LOG.warning("MiniMax H3 job %s could not be cancelled during cleanup: %s", prompt_id, exc)
            else:
                _schedule_deferred_cleanup(client, prompt_id, prepared, runtime_root)
        if not prompt_id or terminal:
            cleanup_prepared_media(prepared, runtime_root)


def cancel_generation(prompt_id: str, server_url: str) -> None:
    if prompt_id:
        ComfyH3Client(server_url).cancel(prompt_id)


def list_history(runtime_root: Path | None, output_directory: Path, limit: int = 12) -> list[HistoryItem]:
    found: dict[Path, HistoryItem] = {}
    if output_directory.is_dir():
        for path in output_directory.glob("MiniMax_H3_*.mp4"):
            resolved = path.resolve()
            found[resolved] = HistoryItem(resolved, path.stat().st_mtime, "Forge Neo")
    if runtime_root is not None:
        runtime_output = runtime_root / "output" / "video"
        if runtime_output.is_dir():
            for pattern in ("*MiniMax*H3*.mp4", "*minimax*h3*.mp4"):
                for path in runtime_output.glob(pattern):
                    resolved = path.resolve()
                    found.setdefault(resolved, HistoryItem(resolved, path.stat().st_mtime, "ComfyUI"))
    return sorted(found.values(), key=lambda item: item.modified_at, reverse=True)[:limit]


def history_html(items: Sequence[HistoryItem]) -> str:
    if not items:
        return (
            '<div class="h3-history-empty"><span>履歴はまだありません</span>'
            '<small>生成が完了すると、音声付きMP4がここに並びます。</small></div>'
        )
    rows = []
    for item in items[:6]:
        size_mib = item.path.stat().st_size / 1024**2
        rows.append(
            '<div class="h3-history-row">'
            '<span class="h3-history-thumb" aria-hidden="true">▶</span>'
            '<span class="h3-history-copy">'
            f'<strong>{html.escape(item.path.stem)}</strong>'
            f'<small>{html.escape(item.label.split(" · ", 1)[0])} · {size_mib:.1f} MiB · {html.escape(item.source)}</small>'
            "</span></div>"
        )
    return '<div class="h3-history-list">' + "".join(rows) + "</div>"


def history_choices(items: Sequence[HistoryItem]) -> list[tuple[str, str]]:
    return [(item.label, os.fspath(item.path)) for item in items]


def reference_guide_html(image_values: Any, video_values: Any, audio_values: Any) -> str:
    images = normalize_file_list(image_values)
    videos = normalize_file_list(video_values)
    audios = normalize_file_list(audio_values)
    total = len(images) + len(videos) + len(audios)
    issues: list[str] = []
    if len(images) > 9:
        issues.append("画像は9枚まで")
    if len(videos) > 3:
        issues.append("動画は3本まで")
    if len(audios) > 3:
        issues.append("単独音声は3本まで")
    if total > 12:
        issues.append("参照素材は合計12個まで")
    if (audios or videos) and not (images or videos):
        issues.append("音声だけでは生成できません")

    chips: list[str] = []
    for index, value in enumerate(images, 1):
        title = html.escape(Path(value).name, quote=True)
        chips.append(f'<code title="{title}">&lt;Picture {index}&gt;</code>')

    audio_index = 1
    total_timed_seconds = 0.0
    for video_index, value in enumerate(videos, 1):
        path = Path(value)
        has_audio = False
        try:
            path = _validate_media_path(value, "video")
            duration, has_audio, video_fps = _probe_media(path)
            total_timed_seconds += duration
            if not 2.0 <= duration <= 15.0:
                issues.append(f"{path.name}: 動画は2〜15秒")
            if video_fps is None or abs(video_fps - H3_FPS) > 0.05:
                shown_fps = "不明" if video_fps is None else f"{video_fps:.3f}"
                issues.append(f"{path.name}: 24fpsへ変換（現在 {shown_fps}fps）")
        except H3BridgeError as exc:
            issues.append(str(exc))
        title = html.escape(path.name, quote=True)
        if has_audio:
            chips.append(f'<code title="{title} · paired audio">&lt;Audio {audio_index}&gt;</code>')
            audio_index += 1
        chips.append(f'<code title="{title}">&lt;Video {video_index}&gt;</code>')

    for value in audios:
        path = Path(value)
        try:
            path = _validate_media_path(value, "audio")
            duration, _, _ = _probe_media(path)
            total_timed_seconds += duration
            if not 2.0 <= duration <= 15.0:
                issues.append(f"{path.name}: 音声は2〜15秒")
        except H3BridgeError as exc:
            issues.append(str(exc))
        title = html.escape(path.name, quote=True)
        chips.append(f'<code title="{title}">&lt;Audio {audio_index}&gt;</code>')
        audio_index += 1

    if total_timed_seconds > 15.0 + 1e-6:
        issues.append(f"動画＋音声は合計15秒まで（現在 {total_timed_seconds:.1f}秒）")

    tone = "error" if issues else "ready"
    issue_html = ""
    if issues:
        issue_html = '<span class="h3-reference-issue">' + html.escape(" / ".join(issues)) + "</span>"
    elif not chips:
        issue_html = "<span>素材を追加すると、実際の presentation 順でタグを表示します。</span>"
    else:
        issue_html = "<span>このタグをクリックせず、そのままPromptへ記述してください。</span>"
    chips_html = "".join(chips) or "<code>&lt;Picture 1&gt;</code><code>&lt;Video 1&gt;</code><code>&lt;Audio 1&gt;</code>"
    return (
        f'<div class="h3-reference-note" data-tone="{tone}" role="status" aria-live="polite">'
        f'<strong>画像 {len(images)}/9 · 動画 {len(videos)}/3 · 音声 {len(audios)}/3 · 合計 {total}/12</strong>'
        f"{chips_html}{issue_html}</div>"
    )


def cache_history_video(selected: str, items: Sequence[HistoryItem], output_directory: Path) -> str:
    selected_path = Path(selected).resolve()
    allowed = {item.path.resolve() for item in items}
    if selected_path not in allowed:
        raise H3BridgeError("選択された履歴ファイルは現在の一覧にありません。")
    if selected_path.parent == output_directory.resolve():
        return os.fspath(selected_path)
    output_directory.mkdir(parents=True, exist_ok=True)
    target = output_directory / f"Imported_{selected_path.name}"
    if not target.exists() or target.stat().st_mtime < selected_path.stat().st_mtime:
        shutil.copy2(selected_path, target)
    return os.fspath(target)


def readiness_html(readiness: RuntimeReadiness) -> str:
    connected_tone = "ready" if readiness.connected else "warn"
    connected_text = "ComfyUI 接続済み" if readiness.connected else "ComfyUI 未接続"
    files_ready = sum(1 for present in readiness.model_files.values() if present)
    total_files = len(readiness.model_files)
    server_files_ready = sum(1 for present in readiness.server_model_files.values() if present)
    server_files_total = len(MODEL_FILES)
    gpu = readiness.gpu_name or "GPU 未確認"
    if readiness.vram_gib:
        gpu += f" · {readiness.vram_gib:.0f} GiB"
    node_text = "H3 nodes OK" if readiness.connected and not readiness.missing_nodes else "H3 nodes 未確認"
    missing_server_files = [
        name for name in MODEL_FILES if not readiness.server_model_files.get(name)
    ] if readiness.connected else []
    details = readiness.error or (
        "不足ノード: " + ", ".join(readiness.missing_nodes)
        if readiness.missing_nodes
        else "接続先で未検出のmodel: " + ", ".join(missing_server_files)
        if missing_server_files
        else "ローカル音声付き生成の準備ができています。"
    )
    root = os.fspath(readiness.runtime_root) if readiness.runtime_root else "runtime未検出"
    return (
        '<div class="h3-runtime-card" role="status" aria-live="polite" aria-atomic="true">'
        '<div class="h3-runtime-badges">'
        f'<span data-tone="{connected_tone}"><i></i>{html.escape(connected_text)}</span>'
        f'<span><i></i>{html.escape(gpu)}</span>'
        f'<span data-tone="{"ready" if files_ready == total_files else "warn"}"><i></i>Files {files_ready}/{total_files}</span>'
        f'<span data-tone="{"ready" if server_files_ready == server_files_total else "warn"}"><i></i>Backend {server_files_ready}/{server_files_total}</span>'
        f'<span data-tone="{"ready" if readiness.connected and not readiness.missing_nodes else "warn"}"><i></i>{html.escape(node_text)}</span>'
        "</div>"
        f'<p>{html.escape(details)}</p><code title="{html.escape(root, quote=True)}">{html.escape(root)}</code>'
        "</div>"
    )


def settings_summary_html(aspect: str, quality: str, duration: float, steps: int) -> str:
    try:
        width, height = dimensions_for(aspect, quality)
        frames = snap_h3_frames(duration)
        effective = frames / H3_FPS
        native_warning = quality == "native"
        tone = "warn" if native_warning else "ready"
        note = (
            "Nativeは最高品質です。RTX 3090では生成時間が大幅に伸びます。"
            if native_warning
            else "入力値はH3の32pxキャンバスと17-frameグリッドへ整列済みです。"
        )
        return (
            f'<div class="h3-settings-summary" data-tone="{tone}" role="status" aria-live="polite">'
            f'<strong>{width} × {height}</strong><span>{frames} frames · {effective:.2f} sec · {int(steps)} steps · 24fps stereo</span>'
            f'<small>{html.escape(note)}</small></div>'
        )
    except (H3BridgeError, TypeError, ValueError) as exc:
        return f'<div class="h3-settings-summary" data-tone="error" role="alert">{html.escape(str(exc))}</div>'


def progress_html(stage: str, message: str, progress: float = 0.0, elapsed: float | None = None) -> str:
    progress = min(1.0, max(0.0, float(progress)))
    percent = int(progress * 100)
    elapsed_text = ""
    if elapsed is not None:
        minutes, seconds = divmod(int(elapsed), 60)
        elapsed_text = f" · {minutes:02d}:{seconds:02d}"
    tone = "ready" if stage == "complete" else "error" if stage == "error" else "active"
    return (
        f'<div class="h3-progress" data-tone="{tone}" role="status" aria-live="polite" aria-atomic="true">'
        '<div class="h3-progress-copy">'
        f'<strong>{html.escape(message)}</strong><span>{html.escape(stage.upper())}{elapsed_text}</span></div>'
        f'<div class="h3-progress-track" role="progressbar" aria-label="MiniMax H3 progress" '
        f'aria-valuemin="0" aria-valuemax="100" aria-valuenow="{percent}">'
        f'<i style="width:{percent}%"></i></div></div>'
    )


def append_prompt_section(prompt: str, section: str) -> str:
    templates = {
        "camera": "Camera: slow controlled push-in, stable composition, natural motion.",
        "dialogue": 'Dialogue: 「」 spoken naturally with accurate lip sync.',
        "sfx": "SFX: spatially precise environmental sound and tactile movement details.",
        "music": "Music: restrained cinematic score that follows the edit and resolves cleanly.",
    }
    addition = templates.get(section)
    if addition is None:
        return prompt
    prompt = (prompt or "").rstrip()
    if addition in prompt:
        return prompt
    return f"{prompt}\n\n{addition}".lstrip()


def prompt_template(prompt: str) -> str:
    prompt = (prompt or "").strip()
    if any(marker in prompt for marker in ("Scene:", "Camera:", "Audio:")):
        return prompt
    scene = prompt or "Describe the subject, environment, look, and intended action."
    return (
        f"Scene: {scene}\n\n"
        "Shot plan:\n"
        "[0s-2s] Establish the subject and environment.\n"
        "[2s-4s] Develop the primary action with clear continuity.\n"
        "[4s-end] Resolve on a deliberate final image.\n\n"
        "Camera: describe framing, lens, movement, and cuts.\n"
        "Audio: describe dialogue, ambience, sound effects, and music in sync with the shots."
    )
