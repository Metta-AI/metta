"""CoGames recipe - STABLE
This recipe is automatically validated in CI and release processes.
"""

from __future__ import annotations

from pathlib import Path

from metta.tools.cogames_eval import CogamesEvalTool
from metta.tools.cogames_train import CogamesTrainTool


def train(
    mission: str = "training_facility.harvest",
    steps: int = 10000,
    variant: str | list[str] | None = None,
    checkpoints: str = "./train_dir",
    device: str = "cuda",
    s3_uri: str | None = None,
) -> CogamesTrainTool:
    """Train a cogames policy.

    Args:
        mission: Mission name (e.g., training_facility.harvest)
        steps: Number of training steps
        variant: Mission variant(s) - single string or list of strings
        checkpoints: Local checkpoints directory
        device: Device to train on (cuda, cpu, or auto)
        s3_uri: Optional S3 URI to upload checkpoint after training
    """
    # Convert variant to list if single string provided
    if isinstance(variant, str):
        variant_list = [variant]
    elif variant is None:
        variant_list = []
    else:
        variant_list = variant

    return CogamesTrainTool(
        mission=mission,
        steps=steps,
        variant=variant_list,
        checkpoints=checkpoints,
        device=device,
        s3_uri=s3_uri,
    )


def evaluate(
    mission: str = "training_facility.harvest",
    policy_uri: str = "",
    variant: str | list[str] | None = None,
    episodes: int = 10,
) -> CogamesEvalTool:
    """Evaluate a cogames policy.

    Args:
        mission: Mission name (e.g., training_facility.harvest)
        policy_uri: Policy URI (file://path or s3://path)
        variant: Mission variant(s) - single string or list of strings
        episodes: Number of evaluation episodes
    """
    # Convert variant to list if single string provided
    if isinstance(variant, str):
        variant_list = [variant]
    elif variant is None:
        variant_list = []
    else:
        variant_list = variant

    return CogamesEvalTool(
        mission=mission,
        policy_uri=policy_uri,
        variant=variant_list,
        episodes=episodes,
    )


def evaluate_latest_in_dir(dir_path: Path | str) -> CogamesEvalTool:
    """Evaluate the latest checkpoint in a directory.

    Args:
        dir_path: Directory containing checkpoints (Path or string)
    """
    # Convert to Path if string (when called via job args)
    dir_path = Path(dir_path) if isinstance(dir_path, str) else dir_path

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
