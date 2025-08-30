import uuid

from bidict import bidict

from metta.agent.policy_record import PolicyRecord
from metta.app_backend.clients.stats_client import StatsClient
from metta.app_backend.routes.score_routes import PolicyScoresData, PolicyScoresRequest


def get_or_create_policy_ids(
    stats_client: StatsClient,
    policies: list[tuple[str, str, str | None]],
    epoch_id: uuid.UUID | None = None,
    create: bool = True,
) -> bidict[str, uuid.UUID]:
    """policies is a list of tuples of (policy_name, policy_uri)"""
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
    # wandb_policy_name is a qualified name like 'entity/project/artifact:version'
    # we store the uris as 'wandb://project/artifact:version', so need to strip 'entity'
    arr = wandb_policy_name.split("/")
    internal_wandb_policy_name = arr[2]
    wandb_uri = "wandb://" + arr[1] + "/" + arr[2]
    return (internal_wandb_policy_name, wandb_uri)


def get_pr_scores_from_stats_server(
    stats_client: StatsClient,
    policy_records: list[PolicyRecord],
    eval_name: str,
    metric: str,
) -> dict[PolicyRecord, float]:
    prs_by_name = {pr.run_name: pr for pr in policy_records if pr.uri and pr.run_name}
    policy_ids = get_or_create_policy_ids(
        stats_client,
        policies=[(name, pr.uri, None) for name, pr in prs_by_name.items() if pr.uri],
        create=False,
    )

    response: PolicyScoresData = stats_client.get_policy_scores(
        PolicyScoresRequest(
            policy_ids=[pid for pid in policy_ids.values()],
            eval_names=[eval_name],
            metrics=[metric],
        )
    )
    scores_by_policy: dict[PolicyRecord, float] = {}
    for policy_name, pr in prs_by_name.items():
        pid = policy_ids.get(policy_name)
        if not pid:
            continue
        if (metric_stats := response.scores.get(pid, {}).get(eval_name, {}).get(metric)) is not None:
            scores_by_policy[pr] = metric_stats.avg

    return scores_by_policy
