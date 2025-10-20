import subprocess

from metta.common.util.fs import get_repo_root


def test_only_root_uv_lock_exists() -> None:
    """Ensure only a single uv.lock exists at the repository root."""

    repo_root = get_repo_root()

    cmd = ["git", "ls-files", "*/uv.lock"]

    result = subprocess.run(
        cmd,
        cwd=repo_root,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise AssertionError(f"git ls-files command failed: {result.stderr}")

    lock_files = [repo_root / f for f in result.stdout.strip().split("\n") if f]
    assert len(lock_files) == 0, "Found extra uv.lock files: " + ", ".join(
        str(f.relative_to(repo_root)) for f in lock_files
    )
