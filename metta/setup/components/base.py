import subprocess
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, TypeVar

from metta.setup.config import SetupConfig
from metta.setup.utils import error

T = TypeVar("T")


class SetupModule(ABC):
    install_once: bool = False

    def __init__(self, config: SetupConfig):
        self.config: SetupConfig = config
        self.repo_root: Path = Path(__file__).parent.parent.parent.parent

    @property
    def name(self) -> str:
        return self.__class__.__name__.replace("Setup", "").lower()

    @property
    @abstractmethod
    def description(self) -> str:
        pass

    @property
    def setup_script_location(self) -> str | None:
        return None

    @abstractmethod
    def is_applicable(self) -> bool:
        pass

    @abstractmethod
    def check_installed(self) -> bool:
        pass

    def dependencies(self) -> list[str]:
        # Other components that must be installed before this one
        # It is assumed that `core` and `system` are always installed first
        return []

    def install(self) -> None:
        if self.setup_script_location:
            _ = self.run_script(self.setup_script_location)
        else:
            raise NotImplementedError(
                f"{self.__class__.__name__} must implement install() or define setup_script_location"
            )

    def run_command(
        self,
        cmd: list[str],
        cwd: Path | None = None,
        check: bool = True,
        capture_output: bool = True,
        input: str | None = None,
        env: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        if cwd is None:
            cwd = self.repo_root

        params: dict[str, str | bool | Path | None | dict[str, str]] = dict(
            cwd=cwd, check=check, capture_output=capture_output, text=True, input=input
        )
        if env is not None:
            params["env"] = env

        return subprocess.run(cmd, **params)  # type: ignore

    def run_script(self, script_path: str, args: list[str] | None = None) -> subprocess.CompletedProcess[str]:
        script = self.repo_root / script_path
        if not script.exists():
            raise FileNotFoundError(f"Script not found: {script}")

        cmd = ["bash", str(script)]
        if args:
            cmd.extend(args)

        return self.run_command(cmd)

    def check_connected_as(self) -> str | None:
        """Check what account/profile/org we're connected as.

        Returns:
            Current account/profile/org or None if not connected
        """
        return None

    def get_configuration_options(self) -> dict[str, tuple[Any, str]]:
        """
        Dict of {setting_name: (default_value, description)}
        """
        return {}

    def configure(self) -> None:
        """This method is called by 'metta configure <component>'.
        Override this to provide custom configuration logic.
        """
        error(f"Component {self.name} does not support configure commands.")

    def run(self, args: list[str]) -> None:
        """Run a component-specific command.

        This method is called by 'metta run <component> <args>'.
        Override this to provide component-specific commands.

        Args:
            args: Command arguments passed after the component name
        """
        error(f"Component {self.name} does not support running commands.")

    def get_setting(self, key: str, default: T) -> T:
        """Get a module-specific setting from the configuration.

        Args:
            key: The setting key (will be prefixed with module name)
            default: Default value if setting not found

        Returns:
            The setting value or default
        """
        full_key = f"module_settings.{self.name}.{key}"
        value = self.config.get(full_key, None)
        # Only return saved value if it differs from default
        return value if value is not None else default

    def set_setting(self, key: str, value: Any) -> None:
        """Save a module-specific setting to the configuration.

        Only saves if value differs from the default defined in get_configuration_options().

        Args:
            key: The setting key (will be prefixed with module name)
            value: The value to save
        """
        # Check if this is the default value
        options = self.get_configuration_options()
        if key in options:
            default_value, _ = options[key]
            if value == default_value:
                # Don't save default values, remove if exists
                full_key = f"module_settings.{self.name}.{key}"
                self._remove_setting(full_key)
                return

        full_key = f"module_settings.{self.name}.{key}"
        self.config.set(full_key, value)

    def _remove_setting(self, full_key: str) -> None:
        """Remove a setting from the configuration."""
        keys = full_key.split(".")
        config = self.config._config
        for k in keys[:-1]:
            if k not in config:
                return  # Key doesn't exist
            config = config[k]

        # Remove the key if it exists
        if keys[-1] in config:
            del config[keys[-1]]
            self.config.save()
            self._cleanup_empty_dicts(self.config._config, keys[:-1])

    def _cleanup_empty_dicts(self, config: dict, keys: list[str]) -> None:
        if not keys:
            return

        # Navigate to the parent
        parent = config
        for k in keys[:-1]:
            if k not in parent:
                return
            parent = parent[k]

        # Check if the target dict is empty
        if keys[-1] in parent and isinstance(parent[keys[-1]], dict) and not parent[keys[-1]]:
            del parent[keys[-1]]
            # Recursively clean up parent
            self._cleanup_empty_dicts(config, keys[:-1])
