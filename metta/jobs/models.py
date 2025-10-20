"""Core job specification models shared across Metta job systems."""

from typing import Any, Literal

from pydantic import Field

from metta.common.config import Config


class JobConfig(Config):
    """Universal job configuration for all Metta job systems.

    This is our serialization format - everything needed to launch a job
    via tools/run.py or RemoteJob.

    Serialization example:
        module="experiments.recipes.arena.train"
        args={"run": "my_run"}
        overrides={"trainer.total_timesteps": 1000000}

    Becomes command:
        ./tools/run.py experiments.recipes.arena.train run=my_run trainer.total_timesteps=1000000

    This replaces:
    - Old PR's TrainingJobConfig
    - adaptive.JobDefinition
    - Arguments to release.Task constructors
    """

    # Job identification
    name: str

    # Tool maker specification (module path)
    module: str  # e.g., "experiments.recipes.arena.train"

    # Arguments to tool maker function
    args: dict[str, Any] = Field(default_factory=dict)

    # Config overrides (dotted paths like "trainer.total_timesteps")
    overrides: dict[str, Any] = Field(default_factory=dict)

    # Execution settings
    execution: Literal["remote", "local"] = "remote"

    # Infrastructure settings (for remote execution)
    gpus: int = 1
    nodes: int = 1
    spot: bool = True
    timeout_s: int = 7200

    # Job type hint (for specialized behavior)
    job_type: Literal["train", "eval", "task"] = "train"

    # Metadata (experiment_id, tags, etc.)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_remote_job_args(self, log_dir: str) -> dict[str, Any]:
        """Convert to RemoteJob constructor arguments.

        Args:
            log_dir: Directory for job logs

        Returns:
            Dict with keys: name, module, args, timeout_s, log_dir, base_args
        """
        # Flatten args and overrides into single list
        arg_list = []
        for k, v in self.args.items():
            arg_list.append(f"{k}={v}")
        for k, v in self.overrides.items():
            arg_list.append(f"{k}={v}")

        # Build skypilot base args
        base_args = [f"--gpus={self.gpus}", f"--nodes={self.nodes}"]
        if not self.spot:
            base_args.insert(0, "--no-spot")

        return {
            "name": self.name,
            "module": self.module,
            "args": arg_list,
            "timeout_s": self.timeout_s,
            "log_dir": log_dir,
            "base_args": base_args,
        }

    def to_local_job_args(self, log_dir: str) -> dict[str, Any]:
        """Convert to LocalJob constructor arguments.

        Args:
            log_dir: Directory for job logs

        Returns:
            Dict with keys: name, cmd, timeout_s, log_dir
        """
        # Build command: uv run ./tools/run.py <module> <args> <overrides>
        cmd = ["uv", "run", "./tools/run.py", self.module]

        # Add args
        for k, v in self.args.items():
            cmd.append(f"{k}={v}")

        # Add overrides
        for k, v in self.overrides.items():
            cmd.append(f"{k}={v}")

        return {
            "name": self.name,
            "cmd": cmd,
            "timeout_s": self.timeout_s,
            "log_dir": log_dir,
        }


# Backwards compatibility alias
JobSpec = JobConfig
