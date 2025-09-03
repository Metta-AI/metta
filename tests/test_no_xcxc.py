import subprocess
from pathlib import Path

from metta.common.util.fs import get_repo_root


def test_no_xcxc_debug():
    """
    Some of our developers use 'xcxc' as a special marker (similar to TODO or FIXME)
    to mark code that should never make it to production.
    """
    project_root = get_repo_root()

    # Use grep for efficiency and to avoid Python file handling issues
    cmd = [
        "grep",
        "-r",
        "--include=*.py",
        "xcxc",
        "--exclude-dir=.venv",
        "--exclude-dir=build",
        "--exclude-dir=.git",
        "--exclude-dir=__pycache__",
        "--exclude-dir=.pytest_cache",
        "--exclude-dir=metta.egg-info",
        "--exclude-dir=node_modules",
        "--exclude-dir=.tox",
        str(project_root),
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=15,  # 15 second timeout
        )

        if result.returncode == 0:
            # Found matches
            lines = result.stdout.strip().split("\n")

            # Filter out this test file
            this_file = str(Path(__file__).resolve())
            offenders = []

            for line in lines:
                if line and this_file not in line:
                    # Make paths relative to repo root for cleaner output
                    if str(project_root) in line:
                        relative_line = line.replace(str(project_root) + "/", "")
                        offenders.append(relative_line)
                    else:
                        offenders.append(line)

            if offenders:
                # Format the assertion message nicely
                message = f"'xcxc' found in {len(offenders)} file(s):\n"
                message += "\n".join(
                    f"  {line}" for line in offenders[:20]
                )  # Show up to 20 matches
                if len(offenders) > 20:
                    message += f"\n  ... and {len(offenders) - 20} more"
                assert False, message

        elif result.returncode == 2:
            # grep returns 2 for errors
            assert False, f"grep command failed: {result.stderr}"

    except subprocess.TimeoutExpired:
        # If grep times out, fail the test with a clear message
        assert False, "xcxc search timed out after 15 seconds"

    except FileNotFoundError:
        # grep not available (e.g., on Windows without Git Bash)
        assert False, "grep command not found - this test requires grep to be available"
