"""Minimal file-serving paths for Gradio."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path


class UnsafeAllowedPathError(ValueError):
    """Raised when a CLI path would expose data outside managed directories."""


def _resolved(path: str | Path) -> Path:
    return Path(path).expanduser().resolve(strict=False)


def _within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _deduplicate(paths: Iterable[Path]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path).casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append(str(path))
    return result


def build_gradio_allowed_paths(
    script_path: str | Path,
    data_path: str | Path,
    *,
    canvas_root: str | Path | None = None,
    requested_paths: Iterable[str | Path] = (),
) -> list[str]:
    """Return only managed output/temp directories and exact Canvas assets."""

    script_root = _resolved(script_path)
    data_root = _resolved(data_path)
    directory_roots: set[Path] = set()
    for root in (script_root, data_root):
        for name in ("output", "outputs", "tmp"):
            candidate = _resolved(root / name)
            if not _within(candidate, root):
                raise UnsafeAllowedPathError(f"The managed {name} path resolves outside its data root.")
            directory_roots.add(candidate)
    exact_files: set[Path] = set()
    if canvas_root is not None:
        root = _resolved(canvas_root)
        for name in ("canvas.js", "canvas.css"):
            candidate = _resolved(root / name)
            if not _within(candidate, root):
                raise UnsafeAllowedPathError(f"The Forge Canvas {name} asset resolves outside its asset root.")
            exact_files.add(candidate)

    for requested in requested_paths:
        candidate = _resolved(requested)
        if candidate in exact_files or any(candidate == root or _within(candidate, root) for root in directory_roots):
            continue
        raise UnsafeAllowedPathError("--gradio-allowed-path may only select a managed output or temporary path.")

    existing_directories = sorted((path for path in directory_roots if path.is_dir()), key=lambda item: str(item))
    existing_files = sorted((path for path in exact_files if path.is_file()), key=lambda item: str(item))
    return _deduplicate([*existing_directories, *existing_files])


def build_gradio_blocked_paths(script_path: str | Path, data_path: str | Path) -> list[str]:
    """Add defense-in-depth blocks for credentials, code, models, and state."""

    script_root = _resolved(script_path)
    data_root = _resolved(data_path)
    relative_targets = (
        ".git",
        ".env",
        ".credentials.json",
        "config.json",
        "config_states",
        "ui-config.json",
        "forge_neo_model_paths.yaml",
        "cache",
        "models",
        "extensions",
        "extensions-builtin",
        "repositories",
        "venv",
        ".venv",
        "logs",
        "secrets",
        "api-auth.txt",
        "gradio-auth.txt",
        "params.txt",
        "styles.csv",
        "sysinfo.json",
        "webui-user.bat",
        "webui-user.local.bat",
    )
    targets = {_resolved(root / relative) for root in (script_root, data_root) for relative in relative_targets}
    # Block existing variable-name credential/support files too. Gradio does not
    # interpret globs in ``blocked_paths``, so expand only the narrow root-level
    # patterns instead of scanning model/output trees recursively.
    for root in (script_root, data_root):
        for pattern in (".env.*", "sysinfo*.json", "*.pem", "*.key", "*.p12", "*.pfx"):
            targets.update(_resolved(path) for path in root.glob(pattern))
    return _deduplicate(sorted(targets, key=lambda item: str(item)))
