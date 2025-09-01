import uuid

from bidict import bidict

from metta.app_backend.clients.stats_client import StatsClient


def get_or_create_policy_ids(
    stats_client: StatsClient,
    policies: list[tuple[str, str | None]],
    epoch_id: uuid.UUID | None = None,
    create: bool = True,
) -> bidict[str, uuid.UUID]:
    """Get or create policy IDs in the stats database.

    Args:
        stats_client: Client for stats database
        policies: List of (uri, description) tuples
        epoch_id: Optional epoch ID for policy creation
        create: Whether to create policies that don't exist

    Returns:
        Bidirectional mapping of run_name to policy UUID
    """
    from metta.rl.checkpoint_manager import CheckpointManager

    # Process policies - extract run_name from URI
    processed_policies = []
    for uri, description in policies:
        # Extract run_name from URI
        metadata = CheckpointManager.get_policy_metadata(uri)
        name = metadata["run_name"]
        processed_policies.append((name, uri, description))

    policy_names = [name for (name, _, __) in processed_policies]
    policy_ids_response = stats_client.get_policy_ids(policy_names)
    policy_ids = bidict(policy_ids_response.policy_ids)

    if create:
        for name, uri, description in processed_policies:
            if name not in policy_ids:
                policy_response = stats_client.create_policy(
                    name=name, description=description, url=uri, epoch_id=epoch_id
                )
                policy_ids[name] = policy_response.id
    return policy_ids
