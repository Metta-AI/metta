import json
import platform
import shutil
from pathlib import Path
from typing import Any

from metta.setup.components.base import SetupModule
from metta.setup.registry import register_module
from metta.setup.utils import info, warning


@register_module
class DockerComposeSetup(SetupModule):
    """Configure Docker CLI to find docker-compose plugin installed via Homebrew."""

    install_once = True

    DOCKER_CONFIG_PATH = Path.home() / ".docker" / "config.json"
    PLUGINS_DIR = "/opt/homebrew/lib/docker/cli-plugins"

    @property
    def name(self) -> str:
        return "docker-compose"

    @property
    def description(self) -> str:
        return "Docker Compose CLI plugin configuration"

    def dependencies(self) -> list[str]:
        return ["system"]

    def _is_applicable(self) -> bool:
        return platform.system() == "Darwin"

    def _config_has_plugins_dir(self) -> bool:
        """Check if config.json already has our plugins directory."""
        if not self.DOCKER_CONFIG_PATH.exists():
            return False

        try:
            config_data = json.loads(self.DOCKER_CONFIG_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return False

        extra_dirs = config_data.get("cliPluginsExtraDirs", [])
        return self.PLUGINS_DIR in extra_dirs

    def check_installed(self) -> bool:
        if not shutil.which("docker-compose"):
            return False
        return self._config_has_plugins_dir()

    def install(self, non_interactive: bool = False, force: bool = False) -> None:
        if not shutil.which("docker-compose"):
            warning("docker-compose not found. Run 'metta install' to install it via brew.")
            return

        if self._config_has_plugins_dir() and not force:
            return

        info("Configuring Docker CLI to find docker-compose plugin...")
        self._ensure_config_has_plugins_dir()

    def _ensure_config_has_plugins_dir(self) -> None:
        """Add Homebrew plugin directory to Docker CLI config.

        Adds the following to ~/.docker/config.json:
            "cliPluginsExtraDirs": ["/opt/homebrew/lib/docker/cli-plugins"]

        This allows `docker compose` to find the docker-compose plugin installed via Homebrew.
        """
        config_data: dict[str, Any] = {}

        if self.DOCKER_CONFIG_PATH.exists():
            try:
                raw_config = self.DOCKER_CONFIG_PATH.read_text(encoding="utf-8")
                if raw_config.strip():
                    parsed = json.loads(raw_config)
                    if isinstance(parsed, dict):
                        config_data = parsed
                    else:
                        warning("Docker config has unexpected structure; leaving it unchanged.")
                        return
            except json.JSONDecodeError as error:
                warning(f"Docker config is not valid JSON: {error}")
                return
            except OSError as error:
                warning(f"Unable to read Docker config: {error}")
                return

        extra_dirs = config_data.get("cliPluginsExtraDirs", [])
        if not isinstance(extra_dirs, list):
            extra_dirs = []

        if self.PLUGINS_DIR not in extra_dirs:
            extra_dirs.append(self.PLUGINS_DIR)

        config_data["cliPluginsExtraDirs"] = extra_dirs

        self.DOCKER_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)

        try:
            self.DOCKER_CONFIG_PATH.write_text(
                json.dumps(config_data, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            info(f"Added {self.PLUGINS_DIR} to Docker CLI plugin search path.")
        except OSError as error:
            warning(f"Failed to write Docker config: {error}")
