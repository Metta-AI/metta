import uuid

from bidict import bidict

from metta.app_backend.stats_client import StatsClient


def get_or_create_policy_ids(
    stats_client: StatsClient, policies: list[tuple[str, str]], epoch_id: uuid.UUID | None = None
) -> bidict[str, uuid.UUID]:
    """
    policies is a list of tuples of (policy_name, policy_uri)
    """
    policy_names = [name for (name, _) in policies]
    policy_ids_response = stats_client.get_policy_ids(policy_names)
    policy_ids = bidict(policy_ids_response.policy_ids)

    for name, uri in policies:
        if name not in policy_ids:
            policy_response = stats_client.create_policy(name=name, description=None, url=uri, epoch_id=epoch_id)
            policy_ids[name] = policy_response.id
    return policy_ids
