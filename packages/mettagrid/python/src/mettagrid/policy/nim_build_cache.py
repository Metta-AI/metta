"""Utility for caching Nim compilation in submission archives."""

from __future__ import annotations

import fcntl
import hashlib
import os
import platform
import re
import shutil
import stat
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Optional

logger = None


def _get_logger():
    """Lazy import logger to avoid circular dependencies."""
    global logger
    if logger is None:
        import logging

        logger = logging.getLogger(__name__)
    return logger


def _ensure_nim_toolchain() -> None:
    """Ensure nim/nimby exist, bootstrap via nimby if missing."""
    if shutil.which("nim") and shutil.which("nimby"):
        return

    DEFAULT_NIM_VERSION = "2.2.6"
    DEFAULT_NIMBY_VERSION = "0.1.11"

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

    if not shutil.which("nim") or not shutil.which("nimby"):
        raise RuntimeError("Failed to provision nim/nimby via nimby.")


def _get_compiled_library_path(nim_dir: Path, nim_file: Optional[Path] = None) -> Optional[Path]:
    """Get the path to the compiled Nim library if it exists.

    Args:
        nim_dir: Directory containing the Nim source files
        nim_file: Optional Path to the Nim source file. If provided, will look for
                  lib{stem}.{ext} where {stem} is the filename without extension.
                  If not provided, will scan for common library names.
    """
    # If we know the Nim file, derive the library name from it
    if nim_file is not None:
        lib_stem = nim_file.stem  # e.g., "dinky_agents" from "dinky_agents.nim"
        for ext in [".so", ".dylib", ".dll"]:
            # Check bindings/generated first (preferred location)
            lib_path = nim_dir / "bindings" / "generated" / f"lib{lib_stem}{ext}"
            if lib_path.exists():
                return lib_path
            # Also check in the directory itself
            lib_path = nim_dir / f"lib{lib_stem}{ext}"
            if lib_path.exists():
                return lib_path

    # Fallback: Check for common library names (backward compatibility)
    for ext in [".so", ".dylib", ".dll"]:
        # Look for libnim_agents.{ext} or nim_agents.{ext} in bindings/generated
        for name in ["libnim_agents", "nim_agents"]:
            lib_path = nim_dir / "bindings" / "generated" / f"{name}{ext}"
            if lib_path.exists():
                return lib_path

    # Also check in the directory itself
    for ext in [".so", ".dylib", ".dll"]:
        for name in ["libnim_agents", "nim_agents"]:
            lib_path = nim_dir / f"{name}{ext}"
            if lib_path.exists():
                return lib_path

    return None


def _verify_library_file(lib_path: Path) -> bool:
    """Verify that a library file exists and is valid (not corrupted/truncated).

    Args:
        lib_path: Path to the library file to verify

    Returns:
        True if the library file is valid, False otherwise
    """
    try:
        if not lib_path.exists():
            return False

        # Check file size - must be > 0
        lib_size = lib_path.stat().st_size
        if lib_size == 0:
            return False

        # Try to read the first few bytes to ensure file is readable
        # This catches cases where the file is being written
        with open(lib_path, "rb") as f:
            header = f.read(4)
            if len(header) < 4:
                return False

        return True
    except Exception:
        return False


def _get_nim_source_files(nim_dir: Path) -> list[Path]:
    """Get all Nim source files that need to be compiled."""
    source_files = []
    for pattern in ["*.nim", "*.nims"]:
        source_files.extend(nim_dir.rglob(pattern))
    return sorted(source_files)


def _compute_source_hash(nim_dir: Path) -> str:
    """Compute a hash of all Nim source files to detect changes."""
    source_files = _get_nim_source_files(nim_dir)
    if not source_files:
        return ""

    hasher = hashlib.sha256()
    for source_file in source_files:
        hasher.update(str(source_file.relative_to(nim_dir)).encode())
        hasher.update(b":")
        try:
            hasher.update(source_file.read_bytes())
        except Exception:
            # If we can't read the file, include its mtime
            hasher.update(str(source_file.stat().st_mtime).encode())
        hasher.update(b"\n")

    return hasher.hexdigest()


def ensure_nim_compiled(nim_dir: Path, force_rebuild: bool = False) -> Path:
    """Ensure Nim code is synced and compiled, using cache when possible.

    Args:
        nim_dir: Directory containing Nim source files (e.g., containing agents.py)
        force_rebuild: If True, force rebuild even if cache exists

    Returns:
        Path to the compiled library

    Raises:
        RuntimeError: If compilation fails
    """
    _get_logger().debug(f"Ensuring Nim compilation for {nim_dir}")

    _ensure_nim_toolchain()

    # Find the main Nim file first (needed to determine library name)
    nim_file = None
    for name in ["nim_agents.nim", "agents.nim", "dinky_agents.nim"]:
        candidate = nim_dir / name
        if candidate.exists():
            nim_file = candidate
            break

    # If no specific file found, look for any .nim file
    if nim_file is None:
        nim_files = list(nim_dir.glob("*.nim"))
        if len(nim_files) == 1:
            nim_file = nim_files[0]
        elif len(nim_files) > 1:
            # Prefer files that look like main entry points
            for name in ["nim_agents", "agents", "main"]:
                for f in nim_files:
                    if f.stem == name:
                        nim_file = f
                        break
                if nim_file:
                    break
            if nim_file is None:
                # Just use the first one
                nim_file = nim_files[0]

    # Quick cache check without lock - if library exists and is valid, return immediately
    # This avoids unnecessary waiting when library is already compiled
    lock_file = nim_dir / ".nim_compile.lock"
    cache_marker = nim_dir / ".nim_compiled"

    if not force_rebuild:
        compiled_lib = None
        if nim_file:
            compiled_lib = _get_compiled_library_path(nim_dir, nim_file)
        else:
            compiled_lib = _get_compiled_library_path(nim_dir)

        if compiled_lib and _verify_library_file(compiled_lib) and cache_marker.exists():
            try:
                cached_hash = cache_marker.read_text().strip()
                current_hash = _compute_source_hash(nim_dir)
                if cached_hash == current_hash:
                    _get_logger().debug(f"Using cached Nim library (quick check): {compiled_lib}")
                    return compiled_lib
            except Exception:
                pass

    # Ensure lock file directory exists
    lock_file.parent.mkdir(parents=True, exist_ok=True)

    # Open lock file in append mode and acquire exclusive lock
    # Use "a+" mode to create file if it doesn't exist, but we need to ensure
    # we can actually lock it. On some filesystems (NFS), file locking may not work.
    lock_fd = None
    lock_acquired = False
    try:
        lock_fd = open(lock_file, "a+")
        # Try to acquire exclusive lock (non-blocking first, then blocking)
        try:
            fcntl.flock(lock_fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            lock_acquired = True
            _get_logger().debug(f"Acquired exclusive lock on {lock_file}")
        except BlockingIOError:
            # Another process is compiling or checking cache, wait for it to finish
            _get_logger().debug("Another process is checking/compiling Nim code, waiting...")
            fcntl.flock(lock_fd.fileno(), fcntl.LOCK_EX)
            lock_acquired = True
            _get_logger().debug(f"Acquired exclusive lock on {lock_file} after waiting")
        except OSError as e:
            # File locking may not be supported on this filesystem (e.g., NFS)
            _get_logger().warning(f"File locking not available ({e}), proceeding without lock")
            lock_acquired = False

        # Now we have exclusive access - re-check cache (library might have been compiled while waiting)
        compiled_lib = None
        if nim_file:
            compiled_lib = _get_compiled_library_path(nim_dir, nim_file)
        else:
            compiled_lib = _get_compiled_library_path(nim_dir)

        # Always verify library file is valid before using cache
        if not force_rebuild and compiled_lib:
            if not _verify_library_file(compiled_lib):
                _get_logger().debug(f"Library file {compiled_lib} is invalid, will rebuild")
                compiled_lib = None

        if not force_rebuild and compiled_lib and cache_marker.exists():
            # Check if source files have changed
            try:
                cached_hash = cache_marker.read_text().strip()
                current_hash = _compute_source_hash(nim_dir)
                if cached_hash == current_hash:
                    # Re-verify library file is still valid (file-based, no memory cache)
                    if _verify_library_file(compiled_lib):
                        _get_logger().debug(f"Using cached Nim library: {compiled_lib}")
                        return compiled_lib
                    else:
                        _get_logger().debug(f"Cached library {compiled_lib} is invalid, rebuilding")
                else:
                    _get_logger().debug("Source files changed, rebuilding")
            except Exception as e:
                _get_logger().warning(f"Failed to check cache hash: {e}, rebuilding")

        # Also check if library exists and is newer than source files (simple mtime check)
        if not force_rebuild and compiled_lib:
            try:
                # Re-read from disk (no memory cache)
                if nim_file:
                    compiled_lib = _get_compiled_library_path(nim_dir, nim_file)
                else:
                    compiled_lib = _get_compiled_library_path(nim_dir)
                if compiled_lib and _verify_library_file(compiled_lib):
                    lib_mtime = compiled_lib.stat().st_mtime
                    source_files = _get_nim_source_files(nim_dir)
                    if source_files:
                        latest_source_mtime = max(f.stat().st_mtime for f in source_files)
                        if lib_mtime >= latest_source_mtime:
                            _get_logger().debug(f"Using existing Nim library (newer than sources): {compiled_lib}")
                            # Update cache marker
                            try:
                                source_hash = _compute_source_hash(nim_dir)
                                cache_marker.write_text(source_hash)
                            except Exception:
                                pass
                            return compiled_lib
            except Exception as e:
                _get_logger().debug(f"Failed to check mtimes: {e}, will rebuild")

        # Need to sync and compile (we still hold the lock)
        _get_logger().info(f"Syncing and compiling Nim code in {nim_dir}")

        # Find nimby.lock file
        nimby_lock = nim_dir / "nimby.lock"
        if nimby_lock.exists():
            # Retry sync with lock cleanup for concurrent access issues
            max_retries = 3
            for attempt in range(max_retries):
                result = subprocess.run(
                    ["nimby", "sync", "-g", str(nimby_lock)],
                    cwd=nim_dir,
                    capture_output=True,
                    text=True,
                )
                if result.returncode == 0:
                    break

                # Check if error is due to git lock conflicts
                error_output = (result.stderr or "") + (result.stdout or "")
                is_git_lock_error = (
                    "shallow.lock" in error_output
                    or "File exists" in error_output
                    or "Bad file descriptor" in error_output
                    or "shallow file has changed" in error_output
                )

                if is_git_lock_error and attempt < max_retries - 1:
                    _get_logger().warning(
                        f"nimby sync failed due to git lock conflict (attempt {attempt + 1}/{max_retries}), "
                        "cleaning up stale locks and retrying..."
                    )
                    # Clean up stale git lock files in nimby package cache
                    nimby_pkgs_dir = Path.home() / ".nimby" / "pkgs"
                    if nimby_pkgs_dir.exists():
                        for pkg_dir in nimby_pkgs_dir.iterdir():
                            if pkg_dir.is_dir():
                                git_lock = pkg_dir / ".git" / "shallow.lock"
                                if git_lock.exists():
                                    try:
                                        git_lock.unlink()
                                        _get_logger().debug(f"Removed stale git lock: {git_lock}")
                                    except Exception as e:
                                        _get_logger().debug(f"Failed to remove git lock {git_lock}: {e}")
                    # Wait before retrying (exponential backoff)
                    time.sleep(0.5 * (2**attempt))
                else:
                    # Not a retryable error or out of retries
                    raise RuntimeError(
                        f"Failed to sync Nim code: {result.returncode}\n"
                        f"stderr: {result.stderr}\n"
                        f"stdout: {result.stdout}"
                    )

        if nim_file is None:
            raise FileNotFoundError(f"No Nim source file found in {nim_dir}")

        # Compile the Nim file
        # Suppress UnusedImport warnings (common in generated/submission code)
        result = subprocess.run(
            ["nim", "c", "--warning[UnusedImport]:off", str(nim_file)],
            cwd=nim_dir,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            # Check if the error is due to missing packages
            error_output = (result.stderr or "") + (result.stdout or "")
            missing_packages = []

            # Look for "cannot open file: <package>" errors
            pattern = r"cannot open file:\s*(\w+)"
            matches = re.findall(pattern, error_output)
            if matches:
                missing_packages = list(set(matches))
                _get_logger().info(f"Detected missing Nim packages: {missing_packages}")

                # Try to install missing packages using nimble
                nimble = shutil.which("nimble")
                if nimble:
                    for pkg in missing_packages:
                        _get_logger().info(f"Installing missing Nim package: {pkg}")
                        install_result = subprocess.run(
                            [nimble, "install", "-y", pkg],
                            cwd=nim_dir,
                            capture_output=True,
                            text=True,
                        )
                        if install_result.returncode != 0:
                            _get_logger().warning(
                                f"Failed to install {pkg}: {install_result.stderr or install_result.stdout}"
                            )

                    # Retry compilation after installing packages
                    _get_logger().info("Retrying compilation after installing missing packages")
                    result = subprocess.run(
                        ["nim", "c", "--warning[UnusedImport]:off", str(nim_file)],
                        cwd=nim_dir,
                        capture_output=True,
                        text=True,
                    )

            if result.returncode != 0:
                error_msg = f"Failed to compile Nim code: {result.returncode}"
                if result.stderr:
                    error_msg += f"\n{result.stderr}"
                if result.stdout:
                    error_msg += f"\n{result.stdout}"
                raise RuntimeError(error_msg)

        # Wait for file system to sync and file to be fully written
        # Check multiple times to ensure file size is stable
        max_wait_attempts = 10
        wait_interval = 0.2
        compiled_lib = None

        for _attempt in range(max_wait_attempts):
            time.sleep(wait_interval)

            # Find the compiled library (always read from disk)
            if nim_file:
                compiled_lib = _get_compiled_library_path(nim_dir, nim_file)
            else:
                compiled_lib = _get_compiled_library_path(nim_dir)

            if compiled_lib and compiled_lib.exists():
                # Check if file size is stable (not still being written)
                try:
                    size1 = compiled_lib.stat().st_size
                    time.sleep(0.1)
                    size2 = compiled_lib.stat().st_size
                    if size1 > 0 and size1 == size2:
                        # File size is stable, verify it's complete
                        if _verify_library_file(compiled_lib):
                            # Sync to disk to ensure it's fully written
                            try:
                                with open(compiled_lib, "rb") as f:
                                    os.fsync(f.fileno())
                            except Exception:
                                pass  # fsync may not be available on all systems
                            break
                except Exception:
                    pass

        if compiled_lib is None or not compiled_lib.exists():
            expected_lib = f"lib{nim_file.stem}.so"
            raise RuntimeError(
                f"Compilation succeeded but library not found in {nim_dir} "
                f"(expected {expected_lib} based on {nim_file.name})"
            )

        # Final verification that library file is complete and valid
        if not _verify_library_file(compiled_lib):
            raise RuntimeError(
                f"Compiled library {compiled_lib} is invalid or corrupted (file may be truncated or incomplete)"
            )

        # Write cache marker with source hash
        try:
            source_hash = _compute_source_hash(nim_dir)
            cache_marker.write_text(source_hash)
        except Exception as e:
            _get_logger().warning(f"Failed to write cache marker: {e}")

        # Write a ready marker file that agents.py can check
        # This helps prevent agents.py from trying to compile while we're still writing
        ready_marker = nim_dir / ".nim_ready"
        try:
            ready_marker.write_text(str(compiled_lib))
        except Exception as e:
            _get_logger().debug(f"Failed to write ready marker: {e}")

        _get_logger().info(f"Successfully compiled Nim library: {compiled_lib}")
        return compiled_lib
    finally:
        # Release lock if we acquired it
        if lock_fd is not None:
            try:
                if lock_acquired:
                    fcntl.flock(lock_fd.fileno(), fcntl.LOCK_UN)
            except Exception:
                pass
            lock_fd.close()
