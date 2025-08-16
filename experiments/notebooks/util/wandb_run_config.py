from typing import Any

import wandb

from metta.common.util.constants import METTA_WANDB_ENTITY, METTA_WANDB_PROJECT


def config_for_run(
    *,
    run_name: str | None = None,
    policy_name: str | None = None,
    entity: str = METTA_WANDB_ENTITY,
    project: str = METTA_WANDB_PROJECT,
) -> dict[str, Any]:
    """Fetch a W&B run configuration by run or policy name.

    Exactly one of ``run_name`` or ``policy_name`` must be provided. The function
    searches the given W&B project for a run matching the provided identifier and
    returns its configuration dictionary.

    Args:
        run_name: Display name of the W&B run to fetch.
        policy_name: Policy name stored in the run configuration under the
            ``policy_name`` key.
        entity: W&B entity (user or team) to search within.
        project: W&B project name.

    Returns:
        The configuration dictionary for the matching run.

    Raises:
        ValueError: If neither or both ``run_name`` and ``policy_name`` are given.
        LookupError: If no run matching the criteria is found.
    """
    if (run_name is None) == (policy_name is None):
        msg = "Provide exactly one of run_name or policy_name"
        raise ValueError(msg)

    api = wandb.Api()
    filters: dict[str, Any]
    if run_name is not None:
        filters = {"displayName": run_name}
    else:
        filters = {"config.policy_name": policy_name}

    runs = api.runs(f"{entity}/{project}", filters=filters)
    for run in runs:
        config = dict(run.config)
        if run_name is not None and run.name == run_name:
            return config
        if policy_name is not None and config.get("policy_name") == policy_name:
            return config

    identifier = run_name if run_name is not None else policy_name
    raise LookupError(f"No W&B run found for '{identifier}'")
