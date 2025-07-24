#!/usr/bin/env python3
"""
Function to get the most recent policy name and ID from a wandb run name.
"""

import os
from typing import Optional

import wandb

# Initialize wandb API
api = wandb.Api()


def get_recent_policy_info(run_name: str, entity: Optional[str] = None, project: Optional[str] = None) -> dict:
    """
    Get detailed information about the most recent policy from a wandb run name.

    Args:
        run_name: The wandb run name
        entity: W&B entity name (defaults to environment variable WANDB_ENTITY)
        project: W&B project name (defaults to environment variable WANDB_PROJECT)

    Returns:
        Dictionary containing policy information:
        - policy_name: The human-readable policy name
        - policy_version: The wandb artifact ID or version
        - wandb_uri: The full wandb URI for the policy
        - metadata: Policy metadata (epoch, agent_step, etc.)
    """
    # Set defaults from environment variables
    entity = entity or os.environ.get("WANDB_ENTITY", "metta-research")
    project = project or os.environ.get("WANDB_PROJECT", "metta")

    try:
        # Get the run
        run_path = f"{entity}/{project}/{run_name}"
        run = api.run(run_path)

        if not run:
            raise ValueError(f"No run found: {run_path}")

        # Get the latest model artifact from this run
        artifacts = run.logged_artifacts()
        model_artifacts = [a for a in artifacts if a.type == "model"]

        if not model_artifacts:
            raise ValueError(f"No model artifacts found for run: {run_name}")

        # Get the most recent model artifact
        latest_artifact = model_artifacts[-1]

        # Construct the wandb URI
        wandb_uri = f"wandb://{entity}/{project}/model/{run_name}:{latest_artifact.version}"

        return {
            "id": latest_artifact.id,
            "policy_name": run_name,
            "policy_version": latest_artifact.version,
            "run_id": run.id,
            "wandb_uri": wandb_uri,
            "url": latest_artifact.url,
            "metadata": latest_artifact.metadata or {},
            "artifact_name": latest_artifact.name,
            "artifact_type": latest_artifact.type,
            "created_at": latest_artifact.created_at,
        }

    except Exception as e:
        raise Exception(f"Error getting policy info from run '{run_name}': {e}") from e


# Example usage
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python get_recent_policy.py <run_name> [entity] [project]")
        print("Example: python get_recent_policy.py alex.local.1950")
        sys.exit(1)

    run_name = sys.argv[1]
    entity = sys.argv[2] if len(sys.argv) > 2 else None
    project = sys.argv[3] if len(sys.argv) > 3 else None

    try:
        info = get_recent_policy_info(run_name, entity, project)
        print(f"Policy: {info['policy_name']}:{info['policy_version']}")
        print(f"ID: {info['id']}")
        print(f"Wandb URI: {info['wandb_uri']}")
        print(f"URL: {info['url']}")
        print(f"Policy type: {info['artifact_type']}")
        print(f"Created at: {info['created_at']}")

        # Get detailed info
        # print(f"Policy info: {json.dumps(info, indent=4)}")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
