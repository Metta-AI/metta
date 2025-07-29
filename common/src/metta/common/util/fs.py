import hashlib
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar

T = TypeVar("T")


def get_repo_root() -> Path:
    """
    Get the repository root directory.

    Returns:
        Path to the repository root

    Raises:
        SystemExit: If repository root cannot be found
    """
    current = Path.cwd().resolve()
    search_paths = [current] + list(current.parents)

    for parent in search_paths:
        if (parent / ".git").exists():
            return parent

    # If we get here, no .git directory was found
    raise SystemExit("Repository root not found - no .git directory in current path or parent directories")


def cd_repo_root():
    """
    Ensure we're running in the repository root.

    Raises:
        SystemExit: If repository root cannot be found
    """
    repo_root = get_repo_root()
    os.chdir(repo_root)


def atomic_write(
    write_func: Callable[[Path], Any],
    target_path: Path | str,
    suffix: str = ".tmp",
) -> None:
    """
    Write a file atomically by writing to a temporary file and then moving it.

    This ensures that the target file is either fully written or not written at all,
    preventing corruption from partial writes or concurrent access.

    Args:
        write_func: A function that takes a path and saves to it
        target_path: The final destination path for the file
        suffix: Suffix for the temporary file (default: ".tmp")

    Example:
        def save_model(path):
            torch.save(model.state_dict(), path)

        atomic_write(save_model, "model.pt")
    """
    target_path = Path(target_path)
    target_dir = target_path.parent
    target_dir.mkdir(parents=True, exist_ok=True)

    # Create temporary file in the same directory to ensure atomic move
    with tempfile.NamedTemporaryFile(mode="wb", dir=target_dir, suffix=suffix, delete=False) as tmp_file:
        tmp_path = Path(tmp_file.name)

    try:
        write_func(tmp_path)
        # Atomic move (on same filesystem)
        shutil.move(str(tmp_path), str(target_path))
    except Exception:
        # Clean up temporary file on error
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def wait_for_file(
    file_path: Path | str,
    timeout: float = 300.0,
    check_interval: float = 0.1,
    stability_duration: float = 0.5,
    min_size: int = 1,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> bool:
    """
    Wait for a file to be created and fully written.

    This function waits for a file to exist and ensures it's fully written by
    checking that the file size remains stable for a specified duration.

    Args:
        file_path: Path to the file to wait for
        timeout: Maximum time to wait in seconds (default: 300)
        check_interval: How often to check in seconds (default: 0.1)
        stability_duration: How long the file size must be stable in seconds (default: 0.5)
        min_size: Minimum file size in bytes to consider valid (default: 1)
        progress_callback: Optional callback called on each check with (elapsed_time, status)
                          where status is one of: "waiting", "found", "stabilizing", "stable"

    Returns:
        True if file was found and is stable, False if timeout

    Example:
        def progress(elapsed, status):
            logger.info(f"Rank {rank}: {status} ({elapsed:.1f}s)")

        if wait_for_file("model.pt", timeout=60, progress_callback=progress):
            model = torch.load("model.pt")
        else:
            raise TimeoutError("Model file not found")
    """
    file_path = Path(file_path)
    start_time = time.time()

    # Wait for file to exist
    while not file_path.exists():
        elapsed = time.time() - start_time
        if progress_callback:
            progress_callback(elapsed, "waiting")

        if elapsed > timeout:
            return False

        time.sleep(check_interval)

    # File exists, wait for it to be fully written
    if progress_callback:
        progress_callback(time.time() - start_time, "found")

    stable_checks = 0
    required_stable_checks = int(stability_duration / check_interval)
    last_size = -1

    while stable_checks < required_stable_checks:
        elapsed = time.time() - start_time
        if elapsed > timeout:
            return False

        try:
            current_size = file_path.stat().st_size

            if current_size == last_size and current_size >= min_size:
                stable_checks += 1
                if progress_callback:
                    progress_callback(elapsed, "stabilizing")
            else:
                stable_checks = 0
                last_size = current_size

        except OSError:
            # File might be in the process of being renamed/moved
            stable_checks = 0

        time.sleep(check_interval)

    if progress_callback:
        progress_callback(time.time() - start_time, "stable")

    # Small delay to ensure filesystem propagation
    time.sleep(0.1)

    return True


def get_file_hash(filepath: Path | str, hash_func: Callable[[], Any] = hashlib.sha256) -> str:
    hash_obj = hash_func()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            hash_obj.update(chunk)
    return hash_obj.hexdigest()
