import uuid

from bidict import bidict

from metta.app_backend.clients.stats_client import StatsClient


def get_or_create_policy_ids(
    stats_client: StatsClient, policies: list[tuple[str, str, str | None]], epoch_id: uuid.UUID | None = None
) -> bidict[str, uuid.UUID]:
    """
    policies is a list of tuples of (policy_name, policy_uri)
    """
    policy_names = [name for (name, _, __) in policies]
    policy_ids_response = stats_client.get_policy_ids(policy_names)
    policy_ids = bidict(policy_ids_response.policy_ids)

    for name, uri, description in policies:
        if name not in policy_ids:
            policy_response = stats_client.create_policy(name=name, description=description, url=uri, epoch_id=epoch_id)
            policy_ids[name] = policy_response.id
    return policy_ids


def wandb_policy_name_to_uri(wandb_policy_name: str) -> tuple[str, str]:
    # wandb_policy_name is a qualified name like 'entity/project/artifact:version'
    # we store the uris as 'wandb://project/artifact:version', so need to strip 'entity'
    arr = wandb_policy_name.split("/")
    internal_wandb_policy_name = arr[2]
    wandb_uri = "wandb://" + arr[1] + "/" + arr[2]
    return (internal_wandb_policy_name, wandb_uri)
