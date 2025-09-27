"""Helpers for registering policies with the stats service."""

from __future__ import annotations

import importlib
import uuid
from typing import Any, Iterable, Tuple

from bidict import bidict

from metta.app_backend.clients.stats_client import StatsClient


def _resolve_policy_metadata(uri: str) -> dict[str, Any]:
    checkpoint_module = importlib.import_module("metta.rl.checkpoint_manager")
    metadata = checkpoint_module.CheckpointManager.get_policy_metadata(uri)
    return metadata


def get_or_create_policy_ids(
    stats_client: StatsClient,
    policies: Iterable[Tuple[str, str | None]],
    epoch_id: uuid.UUID | None = None,
    create: bool = True,
) -> bidict[str, uuid.UUID]:
    """Get or create policy IDs in the stats database."""

    processed_policies = []
    for uri, description in policies:
        metadata = _resolve_policy_metadata(uri)
        run_name = metadata["run_name"]
        epoch = metadata.get("epoch", 0)
        name = f"{run_name}:v{epoch}"
        processed_policies.append((uri, name, description))

    policy_names = [name for _, name, _ in processed_policies]
    policy_ids_response = stats_client.get_policy_ids(policy_names)
    name_to_id = policy_ids_response.policy_ids

    policy_ids = bidict()
    for uri, name, _ in processed_policies:
        if name in name_to_id:
            policy_ids[uri] = name_to_id[name]

    if create:
        for uri, name, description in processed_policies:
            if uri not in policy_ids:
                policy_response = stats_client.create_policy(
                    name=name,
                    description=description,
                    url=uri,
                    epoch_id=epoch_id,
                )
                policy_ids[uri] = policy_response.id

    return policy_ids


__all__ = ["get_or_create_policy_ids"]
