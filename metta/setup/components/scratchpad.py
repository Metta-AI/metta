import os
from pathlib import Path

from metta.setup.components.base import SetupModule
from metta.setup.registry import register_module
from metta.setup.utils import info, success


@register_module
class ScratchpadSetup(SetupModule):
    install_once = True

    @property
    def description(self) -> str:
        return "Scratchpad for experiments"

    @property
    def user_experiments_dir(self) -> Path:
        return self.repo_root / "experiments" / "recipes" / "scratchpad"

    @property
    def _personal_experiments_path(self) -> Path:
        username = os.getenv("USER", "user")
        return self.user_experiments_dir / f"{username}.py"

    def install(self, non_interactive: bool = False, force: bool = False) -> None:
        info(f"Setting up personal experiments file under {self._personal_experiments_path}...")
        username = os.getenv("USER", "user")

        template_path = self.user_experiments_dir / "template.py"
        with open(template_path, "r") as f:
            template_content = f.read()

        processed_content = template_content.replace("{{ USER }}", username)

        with open(self._personal_experiments_path, "w") as f:
            f.write(processed_content)

        success("Experiments installed")

    def check_installed(self) -> bool:
        return self._personal_experiments_path.exists()
