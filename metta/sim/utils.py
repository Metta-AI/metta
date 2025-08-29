import uuid

from bidict import bidict

from metta.app_backend.clients.stats_client import StatsClient


def get_or_create_policy_ids(
    stats_client: StatsClient,
    policies: list[tuple[str, str, str | None]],
    epoch_id: uuid.UUID | None = None,
    create: bool = True,
) -> bidict[str, uuid.UUID]:
    policy_names = [name for (name, _, __) in policies]
    policy_ids_response = stats_client.get_policy_ids(policy_names)
    policy_ids = bidict(policy_ids_response.policy_ids)

    if create:
        for name, uri, description in policies:
            if name not in policy_ids:
                policy_response = stats_client.create_policy(
                    name=name, description=description, url=uri, epoch_id=epoch_id
                )
                policy_ids[name] = policy_response.id
    return policy_ids


def wandb_policy_name_to_uri(wandb_policy_name: str) -> tuple[str, str]:
    """Convert wandb qualified name like 'entity/project/artifact:version' to URI format."""
    parts = wandb_policy_name.split("/")
    if len(parts) != 3:
        raise ValueError(f"Invalid wandb policy name format: {wandb_policy_name}")

    entity, project, artifact_with_version = parts
    internal_wandb_policy_name = artifact_with_version
    wandb_uri = f"wandb://{project}/{artifact_with_version}"
    return (internal_wandb_policy_name, wandb_uri)
