"""CoGames train tool with optional S3 upload support."""

import subprocess
import sys
from pathlib import Path

from pydantic import Field

from metta.common.tool import Tool


def find_latest_checkpoint(checkpoints_dir: Path, mission_name: str) -> Path | None:
    """Find the most recent checkpoint for the mission.

    Checkpoints are in: checkpoints_dir/mission_name/*.pt
    """
    mission_dir = checkpoints_dir / mission_name
    if not mission_dir.exists():
        return None

    checkpoints = list(mission_dir.glob("*.pt"))
    if not checkpoints:
        return None

    # Sort by modification time, return most recent
    checkpoints.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return checkpoints[0]


class CogamesTrainTool(Tool):
    """Train a cogames policy and optionally upload to S3.

    Example usage:
        # Local training (no S3)
        uv run ./tools/run.py cogames_train mission=training_facility.harvest steps=1000

        # Remote training with S3 upload
        uv run ./tools/run.py cogames_train mission=training_facility.harvest \\
            steps=100000 s3_uri=s3://bucket/path/checkpoint.pt
    """

    mission: str = Field(description="Mission name (e.g., training_facility.harvest)")
    steps: int = Field(default=10000, description="Number of training steps")
    checkpoints: str = Field(default="./train_dir", description="Local checkpoints directory")
    variant: list[str] = Field(default_factory=list, description="Mission variants")
    policy: str = Field(default="lstm", description="Policy type")
    seed: int = Field(default=42, description="Random seed")
    cogs: int | None = Field(default=None, description="Number of cogs (agents)")
    s3_uri: str | None = Field(default=None, description="S3 URI to upload checkpoint (e.g., s3://bucket/path/checkpoint.pt)")

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

            upload_cmd = ["aws", "s3", "cp", str(checkpoint_path), self.s3_uri]
            upload_result = subprocess.run(upload_cmd, capture_output=True, text=True, timeout=60)

            if upload_result.returncode != 0:
                print(f"S3 upload failed: {upload_result.stderr}", file=sys.stderr, flush=True)
                return upload_result.returncode

            print(f"Successfully uploaded checkpoint to {self.s3_uri}", flush=True)

        return 0


def cogames_train(**overrides) -> CogamesTrainTool:
    """Create a cogames training tool.

    Args:
        **overrides: Override default configuration values
    """
    return CogamesTrainTool(**overrides)
