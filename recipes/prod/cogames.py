"""CoGames recipe - STABLE
This recipe is automatically validated in CI and release processes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from metta.tools.cogames_eval import CogamesEvalTool
from metta.tools.cogames_train import CogamesTrainTool

# ============================================================================
# Shared Test Configurations
# ============================================================================
# These configurations define the canonical parameters for CI and stable tests.
#
# Usage:
# - CI suite (recipes/validation/ci_suite.py): Imports these functions and uses
#   build_train_command() to create raw cogames commands for local testing
# - Stable suite (recipes/validation/stable_suite.py): Uses recipe tool_makers
#   directly with inline args that match these configs (see comments in stable_suite.py)
#
# This ensures CI tests (raw commands, local) and stable tests (tools, remote)
# use the same underlying parameters and stay in sync.


def get_ci_train_config(name: str) -> dict[str, Any]:
    """Configuration for CI training test (fast, local, no S3).

    Used by: ci_suite.py (imported and converted to command)
    """
    return {
        "mission": "training_facility.harvest",
        "variant": ["small_50"],
        "steps": 1000,
        "checkpoints": f"./train_dir/{name}",
    }


def get_ci_eval_config(train_name: str) -> dict[str, Any]:
    """Configuration for CI evaluation test (fast, local).

    Used by: ci_suite.py (for documentation; eval uses evaluate_latest_in_dir tool)
    """
    return {
        "mission": "training_facility.harvest",
        "variant": ["small_50"],
        "episodes": 5,
        "checkpoint_dir": f"./train_dir/{train_name}",  # For evaluate_latest_in_dir
    }


def get_stable_train_config(name: str) -> dict[str, Any]:
    """Configuration for stable release training test (remote, with S3).

    Used by: stable_suite.py (as reference; inline args should match these values)
    Returns dict matching recipes.prod.cogames.train() parameters.
    """
    s3_uri = f"s3://softmax-public/cogames/{name}/checkpoint.pt"
    return {
        "mission": "training_facility.harvest",
        "variant": ["standard"],
        "steps": 100000,
        "checkpoints": f"/tmp/{name}",
        "s3_uri": s3_uri,
    }


def get_stable_eval_config(train_name: str) -> dict[str, Any]:
    """Configuration for stable release evaluation test (remote, from S3).

    Used by: stable_suite.py (as reference; inline args should match these values)
    Returns dict matching recipes.prod.cogames.evaluate() parameters.
    """
    s3_uri = f"s3://softmax-public/cogames/{train_name}/checkpoint.pt"
    return {
        "mission": "training_facility.harvest",
        "variant": ["standard"],
        "episodes": 20,
        "policy_uri": s3_uri,
    }


def build_train_command(config: dict[str, Any]) -> str:
    """Build cogames train command from configuration dictionary.

    Used by CI suite to create raw commands from shared configs.
    """
    parts = ["cogames", "train"]
    parts.extend(["--mission", config["mission"]])

    if config.get("variant"):
        # For commands, use first variant (CI uses single variant)
        parts.extend(["--variant", config["variant"][0]])

    parts.extend(["--steps", str(config["steps"])])
    parts.extend(["--checkpoints", config["checkpoints"]])

    return " ".join(parts)


def build_eval_command(config: dict[str, Any], checkpoint_path: str) -> str:
    """Build cogames eval command from configuration dictionary.

    Used by CI suite to create raw commands from shared configs.

    Args:
        config: Eval configuration dictionary
        checkpoint_path: Full path to checkpoint file to evaluate
    """
    parts = ["cogames", "eval"]
    parts.extend(["--mission", config["mission"]])

    if config.get("variant"):
        # For commands, use first variant (CI uses single variant)
        parts.extend(["--variant", config["variant"][0]])

    parts.extend(["--policy", f"lstm:{checkpoint_path}"])
    parts.extend(["--episodes", str(config["episodes"])])
    parts.append("--format json")

    return " ".join(parts)


def train(
    mission: str = "training_facility.harvest",
    steps: int = 10000,
    variant: list[str] | None = None,
    checkpoints: str = "./train_dir",
    s3_uri: str | None = None,
) -> CogamesTrainTool:
    """Train a cogames policy.

    Args:
        mission: Mission name (e.g., training_facility.harvest)
        steps: Number of training steps
        variant: List of mission variants to apply
        checkpoints: Local checkpoints directory
        s3_uri: Optional S3 URI to upload checkpoint after training
    """
    return CogamesTrainTool(
        mission=mission,
        steps=steps,
        variant=variant or [],
        checkpoints=checkpoints,
        s3_uri=s3_uri,
    )


def evaluate(
    mission: str = "training_facility.harvest",
    policy_uri: str = "",
    variant: list[str] | None = None,
    episodes: int = 10,
) -> CogamesEvalTool:
    """Evaluate a cogames policy.

    Args:
        mission: Mission name (e.g., training_facility.harvest)
        policy_uri: Policy URI (file://path or s3://path)
        variant: List of mission variants to apply
        episodes: Number of evaluation episodes
    """
    return CogamesEvalTool(
        mission=mission,
        policy_uri=policy_uri,
        variant=variant or [],
        episodes=episodes,
    )


def evaluate_latest_in_dir(dir_path: Path) -> CogamesEvalTool:
    """Evaluate the latest checkpoint in a directory.

    Args:
        dir_path: Directory containing checkpoints
    """
    # Find latest .pt file
    checkpoints = list(dir_path.glob("**/*.pt"))
    if not checkpoints:
        raise ValueError(f"No checkpoints found in {dir_path}")

    latest = max(checkpoints, key=lambda p: p.stat().st_mtime)

    # Infer mission from directory structure (checkpoints/mission_name/*.pt)
    mission = latest.parent.name if latest.parent != dir_path else "training_facility.harvest"

    return CogamesEvalTool(
        mission=mission,
        policy_uri=f"file://{latest}",
        episodes=10,
    )
