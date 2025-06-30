import subprocess
from pathlib import Path

from metta.setup.components.base import SetupModule
from metta.setup.registry import register_module
from metta.setup.utils import info, success, warning


@register_module
class ObservatoryCliSetup(SetupModule):
    @property
    def name(self) -> str:
        return "observatory-cli"

    @property
    def description(self) -> str:
        return "Observatory CLI authentication"

    def is_applicable(self) -> bool:
        return self.config.is_component_enabled("observatory-cli")

    def get_token(self) -> str | None:
        token_file = Path.home() / ".metta" / "observatory_token"
        if not token_file.exists():
            return None
        with open(token_file, "r") as f:
            return f.read().strip()

    def check_installed(self) -> bool:
        return self.get_token() is not None

    def install(self) -> None:
        info("Setting up Observatory authentication...")
        login_script = self.repo_root / "devops" / "observatory_login.py"

        if login_script.exists():
            try:
                self.run_command(["uv", "run", "python", str(login_script)])
                success("Observatory authentication configured")
            except subprocess.CalledProcessError:
                warning("Observatory login failed.")
        else:
            warning("Observatory login script not found at devops/observatory_login.py")

    def check_connected_as(self) -> str | None:
        token = self.get_token()
        if not token:
            return None

        # NOTE: We should do the api/whoami check once it is working
        return "@stem.ai"
