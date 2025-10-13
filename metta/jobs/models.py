"""Core job specification models shared across Metta job systems."""

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class JobSpec:
    """Universal job specification for all Metta job systems.

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
    args: dict[str, Any] = field(default_factory=dict)

    # Config overrides (dotted paths like "trainer.total_timesteps")
    overrides: dict[str, Any] = field(default_factory=dict)

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
    metadata: dict[str, Any] = field(default_factory=dict)

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

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible dict."""
        return {
            "name": self.name,
            "module": self.module,
            "args": self.args,
            "overrides": self.overrides,
            "execution": self.execution,
            "gpus": self.gpus,
            "nodes": self.nodes,
            "spot": self.spot,
            "timeout_s": self.timeout_s,
            "job_type": self.job_type,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "JobSpec":
        """Deserialize from JSON-compatible dict."""
        return cls(**data)
