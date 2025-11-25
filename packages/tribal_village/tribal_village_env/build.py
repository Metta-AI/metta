from __future__ import annotations

import os
import platform
import shutil
import stat
import subprocess
import tempfile
import urllib.request
from pathlib import Path
from typing import Iterable, Optional

DEFAULT_NIM_VERSION = os.environ.get("TRIBAL_VILLAGE_NIM_VERSION", "2.2.6")
DEFAULT_NIMBY_VERSION = os.environ.get("TRIBAL_VILLAGE_NIMBY_VERSION", "0.1.11")


def _target_library_name() -> str:
    system = platform.system()
    if system == "Darwin":
        return "libtribal_village.dylib"
    if system == "Windows":
        return "libtribal_village.dll"
    return "libtribal_village.so"


def _collect_source_files(project_root: Path) -> list[Path]:
    nim_sources = list(project_root.rglob("*.nim"))
    return nim_sources + [
        project_root / "tribal_village.nim",
        project_root / "tribal_village.nimble",
    ]


def _latest_mtime(paths: Iterable[Path]) -> Optional[float]:
    mtimes = [path.stat().st_mtime for path in paths if path.exists()]
    if not mtimes:
        return None
    return max(mtimes)


def _build_library(project_root: Path) -> Path:
    _ensure_nim_toolchain()
    _install_nim_deps(project_root)

    build_script = project_root / "build_lib.sh"
    if build_script.exists():
        result = subprocess.run(["bash", str(build_script)], cwd=project_root, capture_output=True, text=True)
    else:
        ext = Path(_target_library_name()).suffix
        cmd = [
            "nim",
            "c",
            "--app:lib",
            "--mm:arc",
            "--opt:speed",
            "-d:danger",
            f"--out:libtribal_village{ext}",
            "src/tribal_village_interface.nim",
        ]
        result = subprocess.run(cmd, cwd=project_root, capture_output=True, text=True)

    if result.returncode != 0:
        stdout = result.stdout.strip()
        stderr = result.stderr.strip()
        raise RuntimeError(f"Failed to build Nim library (exit {result.returncode}). stdout: {stdout} stderr: {stderr}")

    for ext in (".dylib", ".dll", ".so"):
        candidate = project_root / f"libtribal_village{ext}"
        if candidate.exists():
            return candidate

    raise RuntimeError("Build completed but libtribal_village.{so,dylib,dll} not found.")


def ensure_nim_library_current(verbose: bool = True) -> Path:
    """Rebuild libtribal_village if missing or stale."""

    package_dir = Path(__file__).resolve().parent
    project_root = package_dir.parent
    target_name = _target_library_name()
    target_path = package_dir / target_name

    source_files = _collect_source_files(project_root)
    latest_source_mtime = _latest_mtime(source_files)
    lib_mtime: Optional[float] = target_path.stat().st_mtime if target_path.exists() else None

    needs_rebuild = lib_mtime is None or (latest_source_mtime is not None and lib_mtime < latest_source_mtime)

    if not needs_rebuild:
        return target_path

    if verbose:
        print("Building Tribal Village Nim library to keep bindings current...")

    built_lib = _build_library(project_root)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(built_lib, target_path)

    if verbose:
        print(f"Copied {built_lib} to {target_path}")

    return target_path


def _ensure_nim_toolchain() -> None:
    """Ensure nim/nimble exist, bootstrap via nimby if missing."""

    if shutil.which("nim") and shutil.which("nimble"):
        return

    system = platform.system()
    arch = platform.machine().lower()
    if system == "Linux":
        url = f"https://github.com/treeform/nimby/releases/download/{DEFAULT_NIMBY_VERSION}/nimby-Linux-X64"
    elif system == "Darwin":
        suffix = "ARM64" if "arm" in arch else "X64"
        url = f"https://github.com/treeform/nimby/releases/download/{DEFAULT_NIMBY_VERSION}/nimby-macOS-{suffix}"
    else:
        raise RuntimeError(f"Unsupported OS for nimby bootstrap: {system}")

    dst = Path.home() / ".nimby" / "nim" / "bin" / "nimby"
    with tempfile.TemporaryDirectory() as tmp:
        nimby_path = Path(tmp) / "nimby"
        urllib.request.urlretrieve(url, nimby_path)
        nimby_path.chmod(nimby_path.stat().st_mode | stat.S_IEXEC)
        subprocess.check_call([str(nimby_path), "use", DEFAULT_NIM_VERSION])

        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(nimby_path, dst)

    nim_bin_dir = Path.home() / ".nimby" / "nim" / "bin"
    os.environ["PATH"] = f"{nim_bin_dir}{os.pathsep}" + os.environ.get("PATH", "")

    if not shutil.which("nim") or not shutil.which("nimble"):
        raise RuntimeError("Failed to provision nim/nimble via nimby.")


def _install_nim_deps(project_root: Path) -> None:
    """Install Nim deps via nimble."""

    nimble = shutil.which("nimble")
    if nimble is None:
        raise RuntimeError("nimble not found after nimby setup.")

    # Install dependencies only; this is idempotent and faster than full install.
    result = subprocess.run(
        [nimble, "install", "-y", "--depsOnly"],
        cwd=project_root,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        stdout = result.stdout.strip()
        stderr = result.stderr.strip()
        raise RuntimeError(f"nimble install failed (exit {result.returncode}). stdout: {stdout} stderr: {stderr}")
