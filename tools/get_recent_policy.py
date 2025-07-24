#!/usr/bin/env python3
"""
Function to get the most recent policy name and ID from a wandb run name.
"""

import os
from typing import Optional

import wandb

# Initialize wandb API
api = wandb.Api()


def get_run_artifacts(
    run_name: str, entity: Optional[str] = None, project: Optional[str] = None
) -> list[wandb.Artifact]:
    """
    Get detailed information about the most recent policy from a wandb run name.

    Args:
        run_name: The wandb run name
        entity: W&B entity name (defaults to environment variable WANDB_ENTITY)
        project: W&B project name (defaults to environment variable WANDB_PROJECT)
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
            raise ValueError(f"No policies found for run: {run_name}")

        # Get the most recent model artifact
        return model_artifacts

    except Exception as e:
        raise Exception(f"Error getting run policies from run '{run_name}': {e}") from e


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
    try:
        latest_artifact = get_run_artifacts(run_name, entity, project)[-1]
        return get_policy_info(run_name, latest_artifact, entity, project)
    except Exception as e:
        raise Exception(f"Error getting latest policy info from run '{run_name}': {e}") from e


def get_policy_info(
    run_name: str, policy: wandb.Artifact, entity: Optional[str] = None, project: Optional[str] = None
) -> dict:
    """
    Get detailed information about the most recent policy from a wandb run name.
    """
    entity = entity or os.environ.get("WANDB_ENTITY", "metta-research")
    project = project or os.environ.get("WANDB_PROJECT", "metta")

    # Construct the wandb URI
    wandb_uri = f"wandb://{entity}/{project}/model/{run_name}:{policy.version}"

    return {
        "id": policy.id,
        "policy_name": run_name,
        "policy_version": policy.version,
        "run_id": policy.id,
        "wandb_uri": wandb_uri,
        "url": policy.url,
        "metadata": policy.metadata or {},
        "artifact_name": policy.name,
        "artifact_type": policy.type,
        "created_at": policy.created_at,
    }


# Example usage
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python get_recent_policy.py <run_name> [latest] [entity] [project]\n")
        print("\tTo get all policies for a run:\n\t\tpython tools/get_recent_policy.py zfogg.1753326530")
        print("\tFor just the latest policy:\n\t\tpython tools/get_recent_policy.py zfogg.1753326530 1")
        sys.exit(1)

    run_name = sys.argv[1]
    latest = bool(int(sys.argv[2])) if len(sys.argv) > 2 else False
    entity = sys.argv[3] if len(sys.argv) > 3 else None
    project = sys.argv[4] if len(sys.argv) > 4 else None

    try:
        if latest:
            info = get_recent_policy_info(run_name, entity, project)
            print(f"Policy: {info['policy_name']}:{info['policy_version']}")
            print(f"ID: {info['id']}")
            print(f"Wandb URI: {info['wandb_uri']}")
            print(f"URL: {info['url']}")
            print(f"Policy type: {info['artifact_type']}")
            print(f"Created at: {info['created_at']}")
        else:
            policies = get_run_artifacts(run_name, entity, project)
            for policy in policies:
                print(get_policy_info(run_name, policy, entity, project)["artifact_name"])

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
