"""CoGames train tool with optional S3 upload support."""

import subprocess
import sys
from pathlib import Path

from pydantic import Field

from metta.common.tool import Tool
from metta.common.util.file import write_file


def find_latest_checkpoint(checkpoints_dir: Path, mission_name: str) -> Path | None:
    """Find the most recent checkpoint for the mission.

    PufferLib may save checkpoints in either:
    - checkpoints_dir/mission_name/*.pt (when env_name is set)
    - checkpoints_dir/*.pt (direct save to data_dir)

    This matches the behavior of mettagrid.policy.loader.find_policy_checkpoints().
    """
    # First try mission subdirectory (PufferLib standard)
    mission_dir = checkpoints_dir / mission_name
    if mission_dir.exists():
        checkpoints = list(mission_dir.glob("*.pt"))
        if checkpoints:
            checkpoints.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            return checkpoints[0]

    # Fallback: check root checkpoints directory
    if checkpoints_dir.exists():
        checkpoints = list(checkpoints_dir.glob("*.pt"))
        if checkpoints:
            checkpoints.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            return checkpoints[0]

    return None


class CogamesTrainTool(Tool):
    """Train a cogames policy and optionally upload to S3.

    Example usage:
        # Local training (no S3)
        uv run ./tools/run.py cogames_train mission=training_facility.harvest steps=1000

        # Remote training with S3 upload
        uv run ./tools/run.py cogames_train mission=training_facility.harvest \\
            steps=100000 s3_uri=s3://bucket/path/checkpoint.pt
    """

    run: str | None = Field(default=None, description="Run name for tracking (optional)")
    mission: str = Field(description="Mission name (e.g., training_facility.harvest)")
    steps: int = Field(default=10000, description="Number of training steps")
    checkpoints: str = Field(default="./train_dir", description="Local checkpoints directory")
    variant: list[str] = Field(default_factory=list, description="Mission variants")
    policy: str = Field(default="class=lstm", description="Policy specification (class=CLS[,data=PATH])")
    seed: int = Field(default=42, description="Random seed")
    cogs: int | None = Field(default=None, description="Number of cogs (agents)")
    device: str = Field(default="cuda", description="Device to train on (cuda, cpu, or auto)")
    s3_uri: str | None = Field(
        default=None, description="S3 URI to upload checkpoint (e.g., s3://bucket/path/checkpoint.pt)"
    )

    def invoke(self, args: dict[str, str]) -> int | None:
        checkpoints_dir = Path(self.checkpoints)
        checkpoints_dir.mkdir(parents=True, exist_ok=True)

        # Build cogames train command
        train_cmd = [
            "cogames",
            "train",
            "--mission",
            self.mission,
            "--steps",
            str(self.steps),
            "--checkpoints",
            str(checkpoints_dir),
            "--policy",
            self.policy,
            "--seed",
            str(self.seed),
            "--device",
            self.device,
        ]

        for variant in self.variant:
            train_cmd.extend(["--variant", variant])

        if self.cogs:
            train_cmd.extend(["--cogs", str(self.cogs)])

        print(f"Training: {' '.join(train_cmd)}", flush=True)

        # Run training
        result = subprocess.run(train_cmd, capture_output=False, text=True)
        if result.returncode != 0:
            print(f"Training failed with exit code {result.returncode}", file=sys.stderr, flush=True)
            return result.returncode

        print("Training complete", flush=True)

        # If S3 URI provided, upload the checkpoint
        if self.s3_uri:
            checkpoint_path = find_latest_checkpoint(checkpoints_dir, self.mission)
            if not checkpoint_path:
                print(f"No checkpoint found in {checkpoints_dir}/{self.mission}", file=sys.stderr, flush=True)
                return 1

            print(f"Found checkpoint: {checkpoint_path}", flush=True)
            print(f"Uploading to {self.s3_uri}", flush=True)

            try:
                write_file(self.s3_uri, str(checkpoint_path), content_type="application/octet-stream")
                print(f"Successfully uploaded checkpoint to {self.s3_uri}", flush=True)
            except Exception as e:
                print(f"S3 upload failed: {e}", file=sys.stderr, flush=True)
                return 1

        return 0


def cogames_train(**overrides) -> CogamesTrainTool:
    """Create a cogames training tool.

    Args:
        **overrides: Override default configuration values
    """
    return CogamesTrainTool(**overrides)
