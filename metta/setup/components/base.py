import subprocess
from abc import ABC, abstractmethod
from pathlib import Path

from metta.setup.config import SetupConfig


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
    ) -> subprocess.CompletedProcess[str]:
        if cwd is None:
            cwd = self.repo_root

        return subprocess.run(cmd, cwd=cwd, check=check, capture_output=capture_output, text=True, input=input)

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
