import os
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel

from metta.common.util.fs import get_repo_root
from metta.setup.saved_settings import get_saved_settings
from metta.setup.utils import error

T = TypeVar("T")


class SetupModuleStatus(BaseModel):
    installed: bool
    connected_as: str | None
    expected: str | None


class SetupModule(ABC):
    install_once: bool = False
    repo_root: Path = get_repo_root()

    def __init__(self):
        self._non_interactive = False

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

    def _is_applicable(self) -> bool:
        """Check if this module applies to the current environment (OS)."""
        return True

    @abstractmethod
    def check_installed(self) -> bool:
        """Check if this module is already installed and configured and no changes are needed."""
        pass

    def is_enabled(self) -> bool:
        """Check if this module should be installed based on applicability and settings."""
        return self._is_applicable() and all(
            get_saved_settings().is_component_enabled(dep) for dep in ([self.name] + self.dependencies())
        )

    def dependencies(self) -> list[str]:
        # Other components that must be installed before this one
        # It is assumed that `core` and `system` are always installed first
        return []

    def install(self, non_interactive: bool = False, force: bool = False) -> None:
        """Install this component.

        This is called during a metta install if any of:
        - the component is not installed
        - the component is installed and SetupModule.install_once is False
        - the component is installed and the user is running a force-install

        Force-installs are likely called when the user is trying to repair an issue, so ideally this function
        should self-doctor when force is True.

        Args:
            non_interactive: If True, run in non-interactive mode without prompts
        """
        self._non_interactive = non_interactive
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
        non_interactive: bool | None = None,
    ) -> subprocess.CompletedProcess[str]:
        """Execute a command with proper environment setup and non-interactive support.

        This method handles command execution with automatic environment inheritance,
        non-interactive mode configuration, and proper error handling. It ensures
        commands run correctly in both interactive and CI/Docker environments.

        Args:
            cmd: Command and arguments as a list of strings
            cwd: Working directory for the command (defaults to repo_root)
            check: Whether to raise CalledProcessError on non-zero exit codes
            capture_output: Whether to capture stdout/stderr
            input: Input to send to the command's stdin
            env: Additional environment variables (merged with os.environ)
            non_interactive: Force non-interactive mode (defaults to instance setting)

        Returns:
            CompletedProcess object containing execution results

        Raises:
            CalledProcessError: If check=True and command returns non-zero exit code
            FileNotFoundError: If the command executable is not found
            OSError: For other system-level execution errors

        Note:
            In non-interactive mode, stdin is redirected to /dev/null and environment
            variables are set to prevent interactive prompts (DEBIAN_FRONTEND, etc.).
        """
        if cwd is None:
            cwd = self.repo_root

        # Use instance non_interactive setting if not explicitly provided
        if non_interactive is None:
            non_interactive = self._non_interactive

        # Set up environment for non-interactive mode
        if env is None:
            env = {}
        # Ensure we inherit the current environment (including PATH) and then add our overrides
        env = {**os.environ, **env}

        if non_interactive:
            # Set environment variables for non-interactive operation
            env.update(
                {
                    "DEBIAN_FRONTEND": "noninteractive",
                    "NEEDRESTART_MODE": "a",  # Automatically restart services
                    "UCF_FORCE_CONFFNEW": "1",  # Use new config files without prompting
                }
            )

        params: dict[str, str | bool | Path | None | dict[str, str] | int] = dict(
            cwd=cwd, check=check, capture_output=capture_output, text=True, input=input, env=env
        )

        # In non-interactive mode, redirect stdin to prevent hanging
        if non_interactive and input is None:
            params["stdin"] = subprocess.DEVNULL

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
        """
        Current account/profile/org the user is authenticated as, or None if not connected.
        """
        return None

    @property
    def can_remediate_connected_status_with_install(self) -> bool:
        """
        If force-installing should be recommended to re-authenticate users when:
        - check_installed is True and
        - check_connected_as does not match the expected value
        """
        return False

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
        value = get_saved_settings().get(full_key, None)
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
        get_saved_settings().set(full_key, value)

    def _remove_setting(self, full_key: str) -> None:
        """Remove a setting from the configuration."""
        keys = full_key.split(".")
        saved_settings = get_saved_settings()
        config = saved_settings._config
        for k in keys[:-1]:
            if k not in config:
                return  # Key doesn't exist
            config = config[k]

        # Remove the key if it exists
        if keys[-1] in config:
            del config[keys[-1]]
            saved_settings.save()
            self._cleanup_empty_dicts(saved_settings._config, keys[:-1])

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

    def get_status(self) -> SetupModuleStatus:
        """Get the status of this module. Does not check if the module is enabled."""
        installed = self.check_installed()
        connected_as = self.check_connected_as() if installed else None
        expected = get_saved_settings().get_expected_connection(self.name)

        return SetupModuleStatus(installed=installed, connected_as=connected_as, expected=expected)
