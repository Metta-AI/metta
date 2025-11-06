import os
import pathlib

import metta.setup.components.base
import metta.setup.registry
import metta.setup.utils


@metta.setup.registry.register_module
class ScratchpadSetup(metta.setup.components.base.SetupModule):
    install_once = True

    @property
    def description(self) -> str:
        return "Scratchpad for experiments"

    @property
    def user_experiments_dir(self) -> pathlib.Path:
        return self.repo_root / "experiments" / "recipes" / "scratchpad"

    @property
    def _personal_experiments_path(self) -> pathlib.Path:
        username = os.getenv("USER", "user")
        return self.user_experiments_dir / f"{username}.py"

    def install(self, non_interactive: bool = False, force: bool = False) -> None:
        metta.setup.utils.info(f"Setting up personal experiments file under {self._personal_experiments_path}...")
        username = os.getenv("USER", "user")

        template_path = self.user_experiments_dir / "template.py"
        with open(template_path, "r") as f:
            template_content = f.read()

        processed_content = template_content.replace("{{ USER }}", username)

        with open(self._personal_experiments_path, "w") as f:
            f.write(processed_content)

        metta.setup.utils.success("Experiments installed")

    def check_installed(self) -> bool:
        return self._personal_experiments_path.exists()
