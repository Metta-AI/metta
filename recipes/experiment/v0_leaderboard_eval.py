from metta.tools.eval import EvaluateTool
from recipes.experiment.v0_leaderboard import evaluate


def run(
    policy_version_id: str,
    stats_server_uri: str | None = None,
    seed: int = 50,
) -> EvaluateTool:
    return evaluate(policy_version_id=policy_version_id, stats_server_uri=stats_server_uri, seed=seed)
