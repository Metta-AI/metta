"""CoGames eval tool with optional S3 download support."""

import subprocess
import sys
from pathlib import Path

from pydantic import Field

from metta.common.tool import Tool


class CogamesEvalTool(Tool):
    """Evaluate a cogames policy with optional S3 download.

    Example usage:
        # Local evaluation (no S3)
        uv run ./tools/run.py cogames_eval mission=training_facility.harvest \\
            policy_uri=file://./train_dir/checkpoint.pt

        # Remote evaluation with S3 download
        uv run ./tools/run.py cogames_eval mission=training_facility.harvest \\
            policy_uri=s3://bucket/path/checkpoint.pt
    """

    mission: str = Field(description="Mission name (e.g., training_facility.harvest)")
    policy_uri: str = Field(description="Policy URI (file://path or s3://path)")
    policy_class: str = Field(default="lstm", description="Policy class (e.g., lstm, stateless)")
    episodes: int = Field(default=10, description="Evaluation episodes")
    variant: list[str] = Field(default_factory=list, description="Mission variants")
    cogs: int | None = Field(default=None, description="Number of cogs (agents)")
    format: str = Field(default="json", description="Output format (json or yaml)")

    def invoke(self, args: dict[str, str]) -> int | None:
        # Determine if we need to download from S3
        if self.policy_uri.startswith("s3://"):
            # Download from S3 to temp location
            checkpoint_dir = Path("/tmp/cogames_eval_checkpoints")
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_file = checkpoint_dir / "checkpoint.pt"

            print(f"Downloading from {self.policy_uri}", flush=True)
            download_cmd = ["aws", "s3", "cp", self.policy_uri, str(checkpoint_file)]
            download_result = subprocess.run(download_cmd, capture_output=True, text=True, timeout=60)

            if download_result.returncode != 0:
                print(f"S3 download failed: {download_result.stderr}", file=sys.stderr, flush=True)
                return download_result.returncode

            print(f"Downloaded checkpoint to {checkpoint_file}", flush=True)
            local_path = checkpoint_file
        elif self.policy_uri.startswith("file://"):
            # Use local file directly
            local_path = Path(self.policy_uri[7:])  # Remove "file://" prefix
            if not local_path.exists():
                print(f"Policy file not found: {local_path}", file=sys.stderr, flush=True)
                return 1
        else:
            # Assume it's a direct local path
            local_path = Path(self.policy_uri)
            if not local_path.exists():
                print(f"Policy file not found: {local_path}", file=sys.stderr, flush=True)
                return 1

        # Build cogames eval command
        policy_arg = f"{self.policy_class}:{local_path}"
        eval_cmd = [
            "cogames",
            "eval",
            "--mission",
            self.mission,
            "--policy",
            policy_arg,
            "--episodes",
            str(self.episodes),
            "--format",
            self.format,
        ]

        for variant in self.variant:
            eval_cmd.extend(["--variant", variant])

        if self.cogs:
            eval_cmd.extend(["--cogs", str(self.cogs)])

        print(f"Evaluating: {' '.join(eval_cmd)}", flush=True)

        # Run evaluation
        result = subprocess.run(eval_cmd, capture_output=False, text=True)

        # Clean up temp file if we downloaded from S3
        if self.policy_uri.startswith("s3://"):
            local_path.unlink(missing_ok=True)

        return result.returncode


def cogames_eval(**overrides) -> CogamesEvalTool:
    """Create a cogames evaluation tool.

    Args:
        **overrides: Override default configuration values
    """
    return CogamesEvalTool(**overrides)
