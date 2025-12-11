"""Bootstrap dependencies installer.

Handles installation of critical bootstrap dependencies needed before core.py can run:
- bazel (via bazelisk)
- nimby + nim
- git, g++ (via system package manager)
"""

import os
import platform
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

# Bootstrap dependency versions
REQUIRED_NIM_VERSION = "2.2.6"
REQUIRED_NIMBY_VERSION = "0.1.13"
MIN_BAZEL_VERSION = "7.0.0"
DEFAULT_BAZEL_VERSION = "latest"
BAZELISKVERSION_FILE = Path(__file__).parent.parent.parent.parent.parent / ".bazeliskversion"


TARGET_INSTALL_DIRS = [
    "/usr/local/bin",
    "/usr/bin",
    "/opt/bin",
    str(Path.home() / ".local" / "bin"),
    str(Path.home() / "bin"),
]


def _log(level: str, message: str) -> None:
    """Minimal logger that works before project dependencies install."""
    stream = sys.stderr if level == "ERROR" else sys.stdout
    print(f"[bootstrap:{level}] {message}", file=stream, flush=True)


def info(message: str) -> None:
    _log("INFO", message)


def warning(message: str) -> None:
    _log("WARN", message)


def error(message: str) -> None:
    _log("ERROR", message)


def _safe_unlink(path: Path) -> None:
    """Safely remove a file/symlink, logging warnings on failure."""
    try:
        path.unlink(missing_ok=True)
    except PermissionError:
        warning(f"Unable to remove {path}: permission denied")
    except Exception as exc:
        warning(f"Unable to remove {path}: {exc}")


def get_install_dir() -> Path | None:
    path_dirs = os.environ.get("PATH", "").split(os.pathsep)
    for dir_str in TARGET_INSTALL_DIRS:
        if dir_str in path_dirs:
            dir_path = Path(dir_str)
            if dir_path.exists() and (os.access(dir_path, os.W_OK) or (hasattr(os, "geteuid") and os.geteuid() == 0)):
                return dir_path
    return None


def ensure_bazel_version_file(version: str) -> None:
    """Ensure a workspace-level .bazelversion exists to pin Bazelisk."""
    try:
        workspace = Path.cwd()
        version_file = workspace / ".bazelversion"
        if version_file.exists():
            return
        # Check if directory is writable (skip in Docker builds or read-only contexts)
        if not os.access(workspace, os.W_OK):
            return  # Not writable, skip silently
        version_file.write_text(f"{version}\n")
        info(f"Created {version_file} to request Bazel '{version}'.")
    except Exception:  # pragma: no cover - silently fail in Docker builds or other non-writable contexts
        pass


def bazel_env() -> dict[str, str]:
    """Environment dict ensuring Bazelisk uses the desired default version."""
    env = os.environ.copy()
    env.setdefault("USE_BAZEL_VERSION", DEFAULT_BAZEL_VERSION)
    return env


def _parse_version(v: str) -> tuple[int, ...]:
    parts = []
    for part in v.split("."):
        digits = ""
        for c in part:
            if c.isdigit():
                digits += c
            else:
                break
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


def version_ge(current: str | None, required: str) -> bool:
    if not current:
        return False
    cur = _parse_version(current)
    req = _parse_version(required)
    max_len = max(len(cur), len(req))
    cur = cur + (0,) * (max_len - len(cur))
    req = req + (0,) * (max_len - len(req))
    return cur >= req


def check_bootstrap_deps() -> bool:
    """Check bootstrap deps using the current PATH (no temporary additions)."""
    # Check git
    if not shutil.which("git"):
        return False

    # Check g++
    if not shutil.which("g++"):
        return False

    # Check bazel (with version)
    if not shutil.which("bazel"):
        return False
    try:
        result = subprocess.run(
            ["bazel", "--version"],
            check=True,
            capture_output=True,
            text=True,
            env=bazel_env(),
        )
        version_line = result.stdout.strip()
        version_raw = version_line.split()[1] if len(version_line.split()) > 1 else ""
        version_match = re.match(r"^(\d+(?:\.\d+)*)", version_raw)
        version = version_match.group(1) if version_match else ""
        if version and not version_ge(version, MIN_BAZEL_VERSION):
            return False
    except (subprocess.CalledProcessError, IndexError, ValueError):
        return False

    # Check nimby (with version)
    if not shutil.which("nimby"):
        return False
    try:
        result = subprocess.run(["nimby", "--version"], check=True, capture_output=True, text=True)
        version_output = result.stdout.strip()
        version = version_output.split()[-1].replace("v", "")
        if version and not version_ge(version, REQUIRED_NIMBY_VERSION):
            return False
    except (subprocess.CalledProcessError, IndexError, ValueError):
        return False

    # Check nim (with version)
    if not shutil.which("nim"):
        return False
    try:
        result = subprocess.run(["nim", "--version"], check=True, capture_output=True, text=True)
        version_line = result.stdout.split("\n")[0]
        version = version_line.split()[3] if len(version_line.split()) > 3 else ""
        if version and not version_ge(version, REQUIRED_NIM_VERSION):
            return False
    except (subprocess.CalledProcessError, IndexError, ValueError):
        return False

    return True


def _simple_run_command(
    cmd: list[str],
    cwd: Path | None = None,
    check: bool = True,
    capture_output: bool = True,
    env: dict[str, str] | None = None,
    non_interactive: bool = False,
) -> subprocess.CompletedProcess[str]:
    """Simple run_command wrapper for standalone CLI usage."""
    if cwd is None:
        cwd = Path.cwd()

    # Set up environment for non-interactive mode
    if env is None:
        env = {}
    full_env = os.environ.copy()
    full_env.update(env)

    if non_interactive:
        full_env["DEBIAN_FRONTEND"] = "noninteractive"
        full_env["NEEDRESTART_MODE"] = "a"
        full_env["UCF_FORCE_CONFFNEW"] = "1"
        stdin = subprocess.DEVNULL
    else:
        stdin = None

    return subprocess.run(
        cmd,
        cwd=cwd,
        check=check,
        capture_output=capture_output,
        stdin=stdin,
        env=full_env,
        text=True,
    )


def install_system_packages(run_command=None, non_interactive: bool = False) -> None:
    """Install git and g++ via system package manager."""
    if run_command is None:
        run_command = _simple_run_command

    if platform.system() == "Darwin":
        if shutil.which("brew"):
            for pkg in ["git", "gcc"]:
                if not shutil.which(pkg):
                    info(f"Installing {pkg} via Homebrew...")
                    run_command(["brew", "install", pkg], non_interactive=non_interactive)
    elif platform.system() == "Linux":
        if shutil.which("apt-get"):
            packages = []
            if not shutil.which("git"):
                packages.append("git")
            if not shutil.which("g++"):
                packages.append("build-essential")
            if packages:
                info(f"Installing {', '.join(packages)} via apt...")
                run_command(["sudo", "apt-get", "update"], non_interactive=non_interactive, capture_output=False)
                run_command(
                    ["sudo", "apt-get", "install", "-y"] + packages,
                    non_interactive=non_interactive,
                    capture_output=False,
                )


def get_bazelisk_url() -> str:
    """Get bazelisk download URL for current platform."""
    system = platform.system().lower()
    machine = platform.machine().lower()

    version = BAZELISKVERSION_FILE.read_text().strip()
    base = f"https://github.com/bazelbuild/bazelisk/releases/download/v{version}/"

    if system == "linux":
        if machine in ("aarch64", "arm64"):
            return f"{base}bazelisk-linux-arm64"
        else:
            return f"{base}bazelisk-linux-amd64"
    elif system == "darwin":
        if machine == "arm64":
            return f"{base}bazelisk-darwin-arm64"
        else:
            return f"{base}bazelisk-darwin-amd64"
    elif system in ("mingw", "msys", "windows"):
        return f"{base}bazelisk-windows-amd64.exe"
    else:
        raise RuntimeError(f"Unsupported platform: {system}")


def install_bazel(run_command=None, non_interactive: bool = False) -> None:
    """Install bazel via bazelisk."""
    ensure_bazel_version_file(DEFAULT_BAZEL_VERSION)

    if shutil.which("bazel"):
        try:
            result = subprocess.run(
                ["bazel", "--version"],
                check=True,
                capture_output=True,
                text=True,
                env=bazel_env(),
            )
            version_line = result.stdout.strip()
            version_raw = version_line.split()[1] if len(version_line.split()) > 1 else ""
            version_match = re.match(r"^(\d+(?:\.\d+)*)", version_raw)
            version = version_match.group(1) if version_match else ""
            if version and version_ge(version, MIN_BAZEL_VERSION):
                return  # Already installed with correct version
        except (subprocess.CalledProcessError, IndexError, ValueError):
            pass

    info("Installing bazel via bazelisk...")

    install_dir = get_install_dir()
    if install_dir:
        dest = install_dir / "bazel"
        info(f"Installing bazel to {dest} (detected from PATH)")
    else:
        dest = Path.home() / ".local" / "bin" / "bazel"
        info(f"Installing bazel to default location: {dest}")

    url = get_bazelisk_url()

    if platform.system() in ("Windows", "MSYS", "CYGWIN"):
        dest = dest.with_suffix(".exe")

    dest.parent.mkdir(parents=True, exist_ok=True)

    try:
        urllib.request.urlretrieve(url, dest)
        dest.chmod(0o755)
    except Exception as e:
        error(f"Failed to download bazelisk: {e}")
        raise

    if not shutil.which("bazel"):
        error("Failed to install bazelisk. Please install it manually from https://github.com/bazelbuild/bazelisk")
        raise RuntimeError("Bazel installation failed")

    # Check version
    try:
        result = subprocess.run(
            ["bazel", "--version"],
            check=True,
            capture_output=True,
            text=True,
            env=bazel_env(),
        )
        version_line = result.stdout.strip()
        version_raw = version_line.split()[1] if len(version_line.split()) > 1 else ""
        version_match = re.match(r"^(\d+(?:\.\d+)*)", version_raw)
        version = version_match.group(1) if version_match else ""
        if version and not version_ge(version, MIN_BAZEL_VERSION):
            error(f"Bazel version {version} is too old. Minimum required: {MIN_BAZEL_VERSION}. Please upgrade bazel.")
            raise RuntimeError(f"Bazel version {version} is too old")
    except (subprocess.CalledProcessError, IndexError, ValueError) as e:
        error(f"Failed to determine bazel version: {e}")
        raise


def _get_nim_version() -> str | None:
    """Get the current Nim version from the system PATH."""
    if shutil.which("nim"):
        try:
            result = subprocess.run(["nim", "--version"], check=True, capture_output=True, text=True)
            version_line = result.stdout.split("\n")[0]
            return version_line.split()[3] if len(version_line.split()) > 3 else ""
        except (subprocess.CalledProcessError, IndexError):
            return None
    return None


def _get_nimby_version() -> str | None:
    """Get the current Nimby version from the system PATH."""
    if shutil.which("nimby"):
        try:
            result = subprocess.run(["nimby", "--version"], check=True, capture_output=True, text=True)
            return result.stdout.strip().split()[-1].replace("v", "")
        except (subprocess.CalledProcessError, IndexError):
            return None
    return None


def install_nim_via_nimby(run_command=None, non_interactive: bool = False) -> None:
    """Install nim via nimby, checking versions before touching PATH."""
    # Check versions using current PATH (before we modify it)
    # This ensures we check what the user actually has, not what we've added to PATH
    current_nim_version = _get_nim_version()
    nim_up_to_date = version_ge(current_nim_version, REQUIRED_NIM_VERSION)
    current_nimby_version = _get_nimby_version()
    nimby_up_to_date = version_ge(current_nimby_version, REQUIRED_NIMBY_VERSION)

    # If both are up to date, exit early
    if nimby_up_to_date and nim_up_to_date:
        return

    install_dir = get_install_dir()
    if not install_dir:
        error(f"No dir to install it into identified. Consider adding {TARGET_INSTALL_DIRS[0]} to your PATH")
        raise RuntimeError("No dir to install it into identified")
    target_nimby_path = install_dir / "nimby"

    # 3. Install nimby if missing or out of date
    if not nimby_up_to_date:
        info(f"Nimby is {'out of date' if current_nimby_version else 'not found'}. Installing at {target_nimby_path}")
        _install_nimby(target_nimby_path)

    # 4. Install nim if missing or out of date
    if not nim_up_to_date:
        info(f"Nim is {'out of date' if current_nim_version else 'not found'}. Installing...")
        result = subprocess.run(
            [str(target_nimby_path), "use", REQUIRED_NIM_VERSION],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            error(f"Failed to install Nim version {REQUIRED_NIM_VERSION}: {result.stderr}")
            raise RuntimeError("Nim installation failed")
        # This is where nimby use installs
        target_nim_path = Path.home() / ".nimby" / "nim" / "bin" / "nim"
        if not target_nim_path.exists():
            error(f"Nim not found in {target_nim_path}")
            raise RuntimeError(f"Nim not found in {target_nim_path}")
        if not shutil.which("nim"):
            error(f"Nim at {target_nim_path} not found in PATH after installation. Symlinking into {install_dir}")
            dest = install_dir / "nim"
            if dest.exists():
                dest.unlink()
            dest.symlink_to(target_nim_path)
            info(f"Linked Nim to {dest}")


def _install_nimby(target_nimby_path: Path) -> None:
    # Download and install nimby
    machine = platform.machine()
    if platform.system() == "Linux":
        os_name = "Linux"
    elif platform.system() == "Darwin":
        os_name = "macOS"
    else:
        error(f"Unsupported OS: {platform.system()}")
        raise RuntimeError(f"Unsupported OS: {platform.system()}")

    if machine in ("x86_64", "amd64"):
        arch = "X64"
    elif machine in ("arm64", "aarch64"):
        arch = "ARM64"
    else:
        error(f"Unsupported architecture: {machine}")
        raise RuntimeError(f"Unsupported architecture: {machine}")

    url = f"https://github.com/treeform/nimby/releases/download/{REQUIRED_NIMBY_VERSION}/nimby-{os_name}-{arch}"
    info(f"Downloading Nimby from {url} to {target_nimby_path}")
    target_nimby_path.parent.mkdir(parents=True, exist_ok=True)
    if target_nimby_path.exists() or target_nimby_path.is_symlink():
        target_nimby_path.unlink()
    try:
        urllib.request.urlretrieve(url, str(target_nimby_path))
    except urllib.error.HTTPError as e:
        if e.code == 404:
            error(
                f"Nimby {REQUIRED_NIMBY_VERSION} does not have a binary for {os_name} {arch}. "
                f"Available binaries: Linux-X64, macOS-ARM64, macOS-X64. "
                f"For Docker builds on ARM64 Mac, use: docker build --platform=linux/amd64 ..."
            )
        raise RuntimeError(f"Nimby binary not available for {os_name} {arch}") from e
    except Exception as e:
        error(f"Failed to download Nimby: {e}")
        raise
    target_nimby_path.chmod(0o755)
    info(f"Nimby installed to {target_nimby_path}")


def install_bootstrap_deps(run_command=None, non_interactive: bool = False) -> None:
    """Install all bootstrap dependencies: bazel, nimby, nim, git, g++."""
    # Install git and g++ via package manager
    install_system_packages(run_command, non_interactive=non_interactive)

    # Install bazel (via bazelisk)
    install_bazel(run_command, non_interactive=non_interactive)

    # Install nimby and nim
    install_nim_via_nimby(run_command, non_interactive=non_interactive)


def main() -> None:
    """CLI entrypoint for bootstrap installation."""
    # Use argparse here instead of typer so that this script runs before any project
    # dependencies (typer/rich/etc.) are installed. argparse is in the stdlib.
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Install bootstrap dependencies (bazel, nimby, nim, git, g++)")
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Run in non-interactive mode (no prompts)",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check if bootstrap deps are installed (don't install)",
    )
    args = parser.parse_args()

    if args.check_only:
        if check_bootstrap_deps():
            print("All bootstrap dependencies are installed.")
            sys.exit(0)
        else:
            print("Some bootstrap dependencies are missing.")
            sys.exit(1)

    try:
        install_bootstrap_deps(_simple_run_command, non_interactive=args.non_interactive)
        print("Bootstrap dependencies installed successfully.")
    except Exception as e:
        error(f"Failed to install bootstrap dependencies: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
