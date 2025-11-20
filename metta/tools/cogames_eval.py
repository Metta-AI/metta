"""CoGames eval tool with optional S3 download support."""

import subprocess
import sys

from pydantic import Field

from metta.common.tool import Tool
from metta.common.util.file import local_copy


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
        # Use local_copy context manager to handle both local and S3 URIs
        # For S3, it downloads to temp and cleans up automatically
        # For local files, it just yields the path
        try:
            with local_copy(self.policy_uri) as local_path:
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
                return result.returncode

        except Exception as e:
            print(f"Failed to access policy: {e}", file=sys.stderr, flush=True)
            return 1


def cogames_eval(**overrides) -> CogamesEvalTool:
    """Create a cogames evaluation tool.

    Args:
        **overrides: Override default configuration values
    """
    return CogamesEvalTool(**overrides)
