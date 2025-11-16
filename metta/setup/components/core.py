import os
import shutil
import subprocess
import sys
from pathlib import Path

from metta.setup.components.base import SetupModule
from metta.setup.registry import register_module
from metta.setup.utils import error, success


@register_module
class CoreSetup(SetupModule):
    always_required = True

    @property
    def description(self) -> str:
        return "Core Python dependencies and virtual environment"

    def check_installed(self) -> bool:
        # TODO: cooling: remove partial redundancy with install.sh system dep existence checks
        # TODO: check versions
        # TODO: move some of this logic into components/system.py, and ideally have components/system.py
        # and have core.py and system.py checks run before requiring a full uv sync
        for system_dep in ["uv", "bazel", "git", "g++", "nimby", "nim"]:
            if not shutil.which(system_dep):
                error(f"{system_dep} is not installed. Please install it using `./install.sh`")
                sys.exit(1)
        try:
            subprocess.run(["uv", "--version"], check=True, capture_output=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            error("uv is not installed. Please install it using `./install.sh`")
            sys.exit(1)

    def install(self, non_interactive: bool = False, force: bool = False) -> None:
        repo_root = Path(__file__).resolve().parents[3]
        lock_path = repo_root / "uv.lock"
        constraint_path = repo_root / ".uv-build-constraints.txt"

        torch_ver = None
        with lock_path.open("rb") as f:
            import tomllib

            data = tomllib.load(f)
            for pkg in data.get("package", []):
                if pkg.get("name") == "torch":
                    torch_ver = pkg.get("version")
                    break
        if not torch_ver:
            raise RuntimeError("torch not found in uv.lock")
        constraint_path.write_text(f"torch=={torch_ver}\n")

        cmd = ["uv", "sync", "--build-constraints", str(constraint_path)]
        cmd.extend(["--force-reinstall", "--no-cache"] if force else [])
        env = os.environ.copy()
        env["METTAGRID_FORCE_NIM_BUILD"] = "1"
        self.run_command(cmd, non_interactive=non_interactive, env=env, capture_output=False)

        # Ensure torch is installed with the correct backend for the host
        torch_cmd = [
            "uv",
            "pip",
            "install",
            "--python",
            str(Path.cwd() / ".venv/bin/python"),
            "torch>=2.9.1",
        ]
        env.setdefault("UV_TORCH_BACKEND", "auto")
        self.run_command(torch_cmd, non_interactive=non_interactive, env=env, capture_output=False)
        success("Core dependencies installed")
