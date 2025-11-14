"""CoGames recipe - STABLE
This recipe is automatically validated in CI and release processes.
"""

from pathlib import Path

from metta.tools.cogames_eval import CogamesEvalTool
from metta.tools.cogames_train import CogamesTrainTool


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
