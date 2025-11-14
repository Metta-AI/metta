"""Utilities to ensure the compiled C++ extension matches Python expectations."""

from __future__ import annotations

import importlib
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Final

_EXTENSION_PATTERNS: Final[tuple[str, ...]] = (
    "cpp/mettagrid_c.so",
    "cpp/mettagrid_c.pyd",
    "cpp/mettagrid_c.dylib",
    "mettagrid_c.so",
    "mettagrid_c.pyd",
    "mettagrid_c.dylib",
)

_PROTOCOL_MIN_AGENTS_READY = False


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[6]


def _mettagrid_root() -> Path:
    return _repo_root() / "packages" / "mettagrid"


def _find_extension_file(bazel_bin: Path) -> Path:
    for pattern in _EXTENSION_PATTERNS:
        candidate = bazel_bin / pattern
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "mettagrid_c build completed but no shared library was found inside bazel-bin; "
        "try running `bazel clean` and re-building manually."
    )


def _copy_extension_to_package(extension: Path) -> None:
    dest_dir = _mettagrid_root() / "python" / "src" / "mettagrid"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_file = dest_dir / extension.name
    if dest_file.exists():
        dest_file.chmod(0o766)
    shutil.copy2(extension, dest_file)


def _build_extension(logger: logging.Logger) -> None:
    bazel_root = _mettagrid_root()
    env = os.environ.copy()
    env.setdefault("METTAGRID_BAZEL_PYTHON_VERSION", f"{sys.version_info.major}.{sys.version_info.minor}")
    output_root = env.setdefault("METTAGRID_BAZEL_OUTPUT_ROOT", str(bazel_root / ".bazel_output"))
    Path(output_root).mkdir(parents=True, exist_ok=True)

    cmd = [
        "bazel",
        "--batch",
        f"--output_user_root={output_root}",
        "build",
        "--jobs=auto",
        "--verbose_failures",
        "//cpp:mettagrid_c",
    ]

    logger.info("Building mettagrid_c via Bazel to pick up Protocol.min_agents support…")
    try:
        subprocess.run(cmd, cwd=bazel_root, check=True, env=env)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "Bazel is required to rebuild mettagrid_c but was not found on PATH. "
            "Install it via ./devops/tools/install-system.sh or reinstall mettagrid from a wheel."
        ) from exc

    extension = _find_extension_file(bazel_root / "bazel-bin")
    _copy_extension_to_package(extension)


def ensure_protocol_min_agents(logger: logging.Logger | None = None) -> None:
    """Ensure the C++ Protocol binding exposes the new min_agents attribute."""

    global _PROTOCOL_MIN_AGENTS_READY
    if _PROTOCOL_MIN_AGENTS_READY:
        return

    if logger is None:
        logger = logging.getLogger(__name__)

    mettagrid_c = importlib.import_module("mettagrid.mettagrid_c")
    if hasattr(mettagrid_c.Protocol, "min_agents"):
        _PROTOCOL_MIN_AGENTS_READY = True
        return

    logger.warning(
        "Detected an out-of-date mettagrid_c build (Protocol.min_agents missing); attempting automatic rebuild."
    )

    _build_extension(logger)

    importlib.reload(mettagrid_c)
    if not hasattr(mettagrid_c.Protocol, "min_agents"):
        raise RuntimeError(
            "mettagrid_c was rebuilt but still lacks Protocol.min_agents. "
            "Run `bazel clean` and `bazel --batch build //cpp:mettagrid_c` from packages/mettagrid/, "
            "or reinstall mettagrid to refresh the extension."
        )

    _PROTOCOL_MIN_AGENTS_READY = True


__all__ = ["ensure_protocol_min_agents"]
