import uuid

import bidict

import metta.app_backend.clients.stats_client
import metta.rl.checkpoint_manager


def get_or_create_policy_ids(
    stats_client: metta.app_backend.clients.stats_client.StatsClient,
    policies: list[tuple[str, str | None]],
    epoch_id: uuid.UUID | None = None,
    create: bool = True,
) -> bidict.bidict[str, uuid.UUID]:
    """Get or create policy IDs in the stats database.

    Args:
        stats_client: Client for stats database
        policies: List of (uri, description) tuples
        epoch_id: Optional epoch ID for policy creation
        create: Whether to create policies that don't exist

    Returns:
        Bidirectional mapping of URI to policy UUID
    """
    # Process policies - using URIs as primary identifier
    processed_policies = []
    for uri, description in policies:
        # Extract run_name from URI metadata
        metadata = metta.rl.checkpoint_manager.CheckpointManager.get_policy_metadata(uri)
        run_name = metadata["run_name"]
        epoch = metadata.get("epoch", 0)
        name = f"{run_name}:v{epoch}"
        processed_policies.append((uri, name, description))

    # Get existing policy IDs from stats server (still uses names for now)
    policy_names = [name for (_, name, __) in processed_policies]
    policy_ids_response = stats_client.get_policy_ids(policy_names)
    name_to_id = policy_ids_response.policy_ids

    # Build URI-based bidict
    policy_ids = bidict.bidict()
    for uri, name, _ in processed_policies:
        if name in name_to_id:
            policy_ids[uri] = name_to_id[name]

    if create:
        for uri, name, description in processed_policies:
            if uri not in policy_ids:
                policy_response = stats_client.create_policy(
                    name=name, description=description, url=uri, epoch_id=epoch_id
                )
                policy_ids[uri] = policy_response.id
    return policy_ids
