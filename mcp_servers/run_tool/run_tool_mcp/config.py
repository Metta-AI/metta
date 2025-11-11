"""Configuration for the Run Tool MCP Server."""

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class RunToolMCPConfig:
    """Configuration for the Run Tool MCP Server."""

    repo_root: Path
    run_script_path: Path
    timeout: int = 3600  # 1 hour default timeout
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    @classmethod
    def from_env(cls) -> "RunToolMCPConfig":
        """Load configuration from environment variables."""
        # First priority: use environment variable if set
        env_repo_root = os.getenv("METTA_REPO_ROOT")
        if env_repo_root:
            repo_root = Path(env_repo_root).resolve()
            if (repo_root / "tools" / "run.py").exists():
                run_script_path = repo_root / "tools" / "run.py"
                return cls(
                    repo_root=repo_root,
                    run_script_path=run_script_path,
                    timeout=int(os.getenv("METTA_RUN_TOOL_TIMEOUT", "3600")),
                    log_level=os.getenv("LOG_LEVEL", "INFO"),
                    log_format=os.getenv("LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"),
                )

        # Second priority: try to find from __file__ location
        # __file__ is: mcp_servers/run_tool/run_tool_mcp/config.py
        # So parent.parent.parent.parent should be repo root
        repo_root = Path(__file__).parent.parent.parent.parent.resolve()
        if (repo_root / "tools" / "run.py").exists():
            run_script_path = repo_root / "tools" / "run.py"
            return cls(
                repo_root=repo_root,
                run_script_path=run_script_path,
                timeout=int(os.getenv("METTA_RUN_TOOL_TIMEOUT", "3600")),
                log_level=os.getenv("LOG_LEVEL", "INFO"),
                log_format=os.getenv("LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"),
            )

<<<<<<< Updated upstream
=======
        # Third priority: try current working directory
>>>>>>> Stashed changes
        cwd = Path.cwd().resolve()
        if (cwd / "tools" / "run.py").exists():
            repo_root = cwd
            run_script_path = repo_root / "tools" / "run.py"
            return cls(
                repo_root=repo_root,
                run_script_path=run_script_path,
                timeout=int(os.getenv("METTA_RUN_TOOL_TIMEOUT", "3600")),
                log_level=os.getenv("LOG_LEVEL", "INFO"),
                log_format=os.getenv("LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"),
            )

        # If all else fails, raise an error
        raise FileNotFoundError(
            f"Could not find Metta repository root. "
            f"Set METTA_REPO_ROOT environment variable to point to the Metta repository root. "
            f"Tried: {Path(__file__).parent.parent.parent.parent}, {cwd}"
        )

    def validate(self) -> list[str]:
        """Validate configuration and return list of errors."""
        errors = []
        if not self.repo_root.exists():
            errors.append(f"Repository root does not exist: {self.repo_root}")
        if not self.run_script_path.exists():
            errors.append(f"run.py script does not exist: {self.run_script_path}")
        if self.timeout <= 0:
            errors.append(f"Timeout must be positive, got: {self.timeout}")
        return errors
