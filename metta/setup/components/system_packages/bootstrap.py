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
import tempfile
import urllib.request
from pathlib import Path

from metta.setup.utils import error, info, warning

# Bootstrap dependency versions
REQUIRED_NIM_VERSION = "2.2.6"
REQUIRED_NIMBY_VERSION = "0.1.6"
REQUIRED_BAZEL_VERSION = "7.0.0"
BAZELISK_VERSION = "v1.19.0"

# Common install directories in order of preference
COMMON_INSTALL_DIRS = [
    "/usr/local/bin",
    "/usr/bin",
    "/opt/bin",
    str(Path.home() / ".local" / "bin"),
    str(Path.home() / "bin"),
    str(Path.home() / ".nimby" / "nim" / "bin"),
    str(Path.home() / ".cargo" / "bin"),
    "/opt/homebrew/bin",
]


def get_install_dir() -> Path | None:
    """Return first dir in COMMON_INSTALL_DIRS that is in PATH and writable."""
    path_dirs = os.environ.get("PATH", "").split(os.pathsep)
    for dir_str in COMMON_INSTALL_DIRS:
        if dir_str in path_dirs:
            dir_path = Path(dir_str)
            if dir_path.exists() and (os.access(dir_path, os.W_OK) or (hasattr(os, "geteuid") and os.geteuid() == 0)):
                return dir_path
    return None


def ensure_paths() -> None:
    """Add common directories to PATH if not already present."""
    path_dirs = os.environ.get("PATH", "").split(os.pathsep)
    for dir_str in COMMON_INSTALL_DIRS:
        dir_path = Path(dir_str)
        if dir_path.exists() and dir_str not in path_dirs:
            os.environ["PATH"] = f"{dir_str}:{os.environ.get('PATH', '')}"

    # Add cargo bin if it exists
    cargo_bin = Path.home() / ".cargo" / "bin"
    if cargo_bin.exists() and str(cargo_bin) not in path_dirs:
        os.environ["PATH"] = f"{cargo_bin}:{os.environ.get('PATH', '')}"


def version_ge(current: str, required: str) -> bool:
    """Check if current version >= required version."""
    try:
        from packaging import version

        return version.parse(current) >= version.parse(required)
    except Exception:
        # Fallback: numeric segment compare (handles X.Y.Z)
        current_parts = current.split(".")
        required_parts = required.split(".")

        max_len = max(len(current_parts), len(required_parts))
        current_parts.extend(["0"] * (max_len - len(current_parts)))
        required_parts.extend(["0"] * (max_len - len(required_parts)))

        if len(current_parts) != len(required_parts):
            raise ValueError("Mismatched lengths: current_parts and required_parts must have the same length") from None
        for c, r in zip(current_parts, required_parts, strict=True):
            c_int = int(c) if c.isdigit() else 0
            r_int = int(r) if r.isdigit() else 0
            if c_int > r_int:
                return True
            if c_int < r_int:
                return False
        return True


def check_bootstrap_deps() -> bool:
    """Check if bootstrap deps (bazel, nimby, nim, git, g++) are installed."""
    ensure_paths()

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
        result = subprocess.run(["bazel", "--version"], check=True, capture_output=True, text=True)
        version_line = result.stdout.strip()
        version_raw = version_line.split()[1] if len(version_line.split()) > 1 else ""
        version_match = re.match(r"^(\d+(?:\.\d+)*)", version_raw)
        version = version_match.group(1) if version_match else ""
        if version and not version_ge(version, REQUIRED_BAZEL_VERSION):
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
    """Simple run_command wrapper for standalone CLI usage.

    This provides a minimal run_command implementation for use outside of SetupModule context.
    """
    from metta.common.util.fs import get_repo_root

    if cwd is None:
        cwd = get_repo_root()

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

    base = f"https://github.com/bazelbuild/bazelisk/releases/download/{BAZELISK_VERSION}/"

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
    ensure_paths()

    if shutil.which("bazel"):
        try:
            result = subprocess.run(["bazel", "--version"], check=True, capture_output=True, text=True)
            version_line = result.stdout.strip()
            version_raw = version_line.split()[1] if len(version_line.split()) > 1 else ""
            version_match = re.match(r"^(\d+(?:\.\d+)*)", version_raw)
            version = version_match.group(1) if version_match else ""
            if version and version_ge(version, REQUIRED_BAZEL_VERSION):
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

    ensure_paths()

    if not shutil.which("bazel"):
        error("Failed to install bazelisk. Please install it manually from https://github.com/bazelbuild/bazelisk")
        raise RuntimeError("Bazel installation failed")

    # Check version
    try:
        result = subprocess.run(["bazel", "--version"], check=True, capture_output=True, text=True)
        version_line = result.stdout.strip()
        version_raw = version_line.split()[1] if len(version_line.split()) > 1 else ""
        version_match = re.match(r"^(\d+(?:\.\d+)*)", version_raw)
        version = version_match.group(1) if version_match else ""
        if version and not version_ge(version, REQUIRED_BAZEL_VERSION):
            error(
                f"Bazel version {version} is too old. Minimum required: {REQUIRED_BAZEL_VERSION}. Please upgrade bazel."
            )
            raise RuntimeError(f"Bazel version {version} is too old")
    except (subprocess.CalledProcessError, IndexError, ValueError) as e:
        error(f"Failed to determine bazel version: {e}")
        raise


def remove_legacy_nim_installations() -> None:
    """Remove legacy Nim installations."""
    if shutil.which("brew"):
        try:
            result = subprocess.run(["brew", "list", "--versions", "nim"], check=False, capture_output=True, text=True)
            if result.returncode == 0:
                info("Removing legacy Homebrew Nim installation...")
                subprocess.run(["brew", "uninstall", "--force", "nim"], check=False, capture_output=True)
        except Exception:
            pass

    # Remove legacy nimble bin directory
    nimble_bin = Path.home() / ".nimble" / "bin"
    if nimble_bin.exists():
        shutil.rmtree(nimble_bin, ignore_errors=True)

    # Remove legacy choosenim directory
    choosenim_dir = Path.home() / ".choosenim"
    if choosenim_dir.exists():
        shutil.rmtree(choosenim_dir, ignore_errors=True)

    # Remove legacy symlinks
    local_bin = Path.home() / ".local" / "bin"
    for tool in ["nim", "nimble"]:
        symlink = local_bin / tool
        if symlink.is_symlink():
            try:
                target = symlink.readlink()
                if ".nimble" in str(target):
                    symlink.unlink()
            except Exception:
                pass


def link_nim_bins(src_dir: Path) -> None:
    """Link nim and nimby binaries to install directory."""
    if not src_dir.exists():
        return

    install_dir = get_install_dir()
    if not install_dir:
        info(f'Nim is installed in {src_dir}. Add it to your PATH (e.g. export PATH="{src_dir}:$PATH").')
        return

    install_dir.mkdir(parents=True, exist_ok=True)

    linked_any = False
    for tool in ["nim", "nimby"]:
        src = src_dir / tool
        dest = install_dir / tool

        if src.exists() and src.is_file() and os.access(src, os.X_OK):
            if dest.is_symlink():
                try:
                    current_target = dest.readlink()
                    if current_target == src:
                        continue
                except Exception:
                    pass
            try:
                if dest.exists():
                    dest.unlink()
                dest.symlink_to(src)
                linked_any = True
            except Exception as e:
                warning(f"Failed to link {tool}: {e}")

    if linked_any:
        info(f"Linked Nim binaries into {install_dir}. Ensure this directory is in your PATH.")


def install_nim_via_nimby(run_command=None, non_interactive: bool = False) -> None:
    """Install nim via nimby."""
    remove_legacy_nim_installations()
    ensure_paths()

    # Check current versions
    current_nim_version = ""
    if shutil.which("nim"):
        try:
            result = subprocess.run(["nim", "--version"], check=True, capture_output=True, text=True)
            version_line = result.stdout.split("\n")[0]
            current_nim_version = version_line.split()[3] if len(version_line.split()) > 3 else ""
        except (subprocess.CalledProcessError, IndexError):
            pass

    current_nimby_version = ""
    if shutil.which("nimby"):
        try:
            result = subprocess.run(["nimby", "--version"], check=True, capture_output=True, text=True)
            current_nimby_version = result.stdout.strip().split()[-1].replace("v", "")
        except (subprocess.CalledProcessError, IndexError):
            pass

    nim_bin_dir = Path.home() / ".nimby" / "nim" / "bin"

    # Check if already installed with correct versions
    if (
        current_nim_version
        and version_ge(current_nim_version, REQUIRED_NIM_VERSION)
        and current_nimby_version
        and version_ge(current_nimby_version, REQUIRED_NIMBY_VERSION)
    ):
        link_nim_bins(nim_bin_dir)
        return

    if current_nim_version:
        info(f"Found Nim {current_nim_version} but require >= {REQUIRED_NIM_VERSION}. Installing via Nimby...")
    else:
        info("Nim not found. Installing via Nimby...")

    # Download and install nimby
    system = platform.system()
    machine = platform.machine()

    if system == "Linux":
        os_name = "Linux"
    elif system == "Darwin":
        os_name = "macOS"
    else:
        error(f"Unsupported OS: {system}")
        raise RuntimeError(f"Unsupported OS: {system}")

    if machine in ("x86_64", "amd64"):
        arch = "X64"
    elif machine in ("arm64", "aarch64"):
        arch = "ARM64"
    else:
        error(f"Unsupported architecture: {machine}")
        raise RuntimeError(f"Unsupported architecture: {machine}")

    url = f"https://github.com/treeform/nimby/releases/download/{REQUIRED_NIMBY_VERSION}/nimby-{os_name}-{arch}"
    info(f"Downloading Nimby from {url}")

    with tempfile.TemporaryDirectory() as tmpdir:
        nimby_path = Path(tmpdir) / "nimby"
        try:
            urllib.request.urlretrieve(url, nimby_path)
            nimby_path.chmod(0o755)
        except Exception as e:
            error(f"Failed to download Nimby: {e}")
            raise

        # Move nimby to bin directory
        nim_bin_dir.mkdir(parents=True, exist_ok=True)
        final_nimby_path = nim_bin_dir / "nimby"
        if final_nimby_path.exists():
            final_nimby_path.unlink()
        nimby_path.rename(final_nimby_path)

        # Verify nimby installation
        if not final_nimby_path.exists() or not os.access(final_nimby_path, os.X_OK):
            error("Failed to install nimby: binary not found or not executable")
            raise RuntimeError("Nimby installation failed")

        try:
            verify_result = subprocess.run(
                [str(final_nimby_path), "--version"],
                check=True,
                capture_output=True,
                text=True,
            )
            installed_nimby_version = verify_result.stdout.strip().split()[-1].replace("v", "")
            info(f"Successfully installed nimby version {installed_nimby_version}")
        except subprocess.CalledProcessError as e:
            error(f"Failed to verify nimby installation: {e}")
            raise RuntimeError("Nimby installation verification failed") from e

        # Run nimby to install nim
        info(f"Installing Nim version {REQUIRED_NIM_VERSION}...")
        result = subprocess.run(
            [str(final_nimby_path), "use", REQUIRED_NIM_VERSION],
            cwd=tmpdir,
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            error(f"Failed to install Nim version {REQUIRED_NIM_VERSION}: {result.stderr}")
            raise RuntimeError("Nim installation failed")

    link_nim_bins(nim_bin_dir)


def install_bootstrap_deps(run_command=None, non_interactive: bool = False) -> None:
    """Install all bootstrap dependencies: bazel, nimby, nim, git, g++."""
    ensure_paths()

    # Install git and g++ via package manager
    install_system_packages(run_command, non_interactive=non_interactive)

    # Install bazel (via bazelisk)
    install_bazel(run_command, non_interactive=non_interactive)

    # Install nimby and nim
    install_nim_via_nimby(run_command, non_interactive=non_interactive)

    ensure_paths()


def main() -> None:
    """CLI entrypoint for bootstrap installation."""
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
