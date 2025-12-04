from metta.tools.multi_versioned_policy_eval import MultiPolicyVersionEvalTool
from recipes.experiment.v0_leaderboard import evaluate


# Here temporarily for backwards compatibility
def run(
    policy_version_id: str,
    result_file_path: str | None = None,
    stats_server_uri: str | None = None,
    seed: int = 50,
) -> MultiPolicyVersionEvalTool:
    return evaluate(
        policy_version_id=policy_version_id,
        result_file_path=result_file_path,
        stats_server_uri=stats_server_uri,
        seed=seed,
    )
