import subprocess
from pathlib import Path

from metta.common.util.fs import get_repo_root


def test_no_xcxc_debug():
    """
    Some of our developers use 'xcxc' as a special marker (similar to TODO or FIXME)
    to mark code sections that should never make it to production.
    """
    project_root = get_repo_root()

    # Use git to get all files (tracked + untracked), respecting .gitignore
    cmd = ["git", "ls-files", "--cached", "--others", "--exclude-standard", "*.py"]

    try:
        result = subprocess.run(
            cmd,
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=15,
        )

        if result.returncode != 0:
            raise AssertionError(f"git ls-files command failed: {result.stderr}")

        py_files = [f for f in result.stdout.strip().split("\n") if f]

        relative_this_file = str(Path(__file__).relative_to(project_root))

        offenders = []
        for py_file in py_files:
            # Filter out this test file
            if py_file == relative_this_file:
                continue

            full_path = project_root / py_file
            try:
                content = full_path.read_text(encoding="utf-8", errors="ignore")
                if "xcxc" in content:
                    offenders.append(py_file)
            except (OSError, UnicodeDecodeError):
                # Skip files we can't read
                continue

        if offenders:
            message = f"'xcxc' found in {len(offenders)} file(s):\n"
            message += "\n".join(f"  {line}" for line in offenders[:20])  # Show up to 20 matches
            if len(offenders) > 20:
                message += f"\n  ... and {len(offenders) - 20} more"
            raise AssertionError(message)

    except subprocess.TimeoutExpired as e:
        raise AssertionError("xcxc search timed out after 15 seconds") from e

    except FileNotFoundError as e:
        raise AssertionError("git command not found - this test requires git to be available") from e
