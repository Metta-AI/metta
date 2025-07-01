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
    stability_check_interval: float = 0.05,
    min_size: int = 1,
    logger: Optional[Any] = None,
    rank: Optional[int] = None,
) -> bool:
    """
    Wait for a file to be created and fully written.

    This function waits for a file to exist and ensures it's fully written by
    checking that the file size remains stable for a specified duration.

    Args:
        file_path: Path to the file to wait for
        timeout: Maximum time to wait in seconds (default: 300)
        check_interval: How often to check for file existence in seconds (default: 0.1)
        stability_duration: How long the file size must be stable in seconds (default: 0.5)
        stability_check_interval: How often to check file size stability in seconds (default: 0.05)
        min_size: Minimum file size in bytes to consider valid (default: 1)
        logger: Optional logger for status messages
        rank: Optional rank identifier for distributed systems

    Returns:
        True if file was found and is stable, False if timeout

    Example:
        if wait_for_file("model.pt", timeout=60):
            model = torch.load("model.pt")
        else:
            raise TimeoutError("Model file not found")
    """
    file_path = Path(file_path)
    rank_prefix = f"Rank {rank}: " if rank is not None else ""

    def log(message: str, level: str = "info"):
        if logger:
            getattr(logger, level)(f"{rank_prefix}{message}")

    log(f"Waiting for file at {file_path}")
    start_time = time.time()

    # Wait for file to exist
    while not file_path.exists():
        time.sleep(check_interval)
        elapsed = time.time() - start_time

        if elapsed > timeout:
            log(f"Timeout after {timeout}s waiting for file at {file_path}", "error")
            return False

        if int(elapsed) % 10 == 0 and elapsed > 0:
            log(f"Still waiting for file... ({elapsed:.0f}s elapsed)")

    # File exists, wait for it to be fully written
    log("File found, waiting for write to complete...")

    stable_duration_elapsed = 0.0
    last_size = -1

    while stable_duration_elapsed < stability_duration:
        try:
            current_size = file_path.stat().st_size

            if current_size == last_size and current_size >= min_size:
                stable_duration_elapsed += stability_check_interval
            else:
                stable_duration_elapsed = 0.0
                last_size = current_size

            time.sleep(stability_check_interval)

            # Check total timeout
            if time.time() - start_time > timeout:
                log(f"Timeout after {timeout}s waiting for file stability", "error")
                return False

        except OSError:
            # File might be in the process of being renamed/moved
            stable_duration_elapsed = 0.0
            time.sleep(stability_check_interval)

    elapsed = time.time() - start_time
    log(f"File stable after {elapsed:.1f}s")

    # Small delay to ensure filesystem propagation
    time.sleep(0.1)

    return True
