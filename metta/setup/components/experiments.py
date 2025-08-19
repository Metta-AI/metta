import os
import shutil
from pathlib import Path

from metta.setup.components.base import SetupModule
from metta.setup.registry import register_module
from metta.setup.utils import info, success


@register_module
class ExperimentsSetup(SetupModule):
    @property
    def description(self) -> str:
        return "Experiments"

    @property
    def user_experiments_dir(self) -> Path:
        return self.repo_root / "experiments" / "user"

    @property
    def _personal_experiments_path(self) -> Path:
        username = os.getenv("USER", "user")
        return self.user_experiments_dir / f"{username}.py"

    def install(self) -> None:
        info(f"Setting up personal experiments file under {self._personal_experiments_path}...")
        shutil.copyfile(self.user_experiments_dir / "example.py", self._personal_experiments_path)
        success("Experiments installed")

    def check_installed(self) -> bool:
        return self._personal_experiments_path.exists()
