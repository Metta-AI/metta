from cogames.cogs_vs_clips.missions import Machina1OpenWorldMission
from metta.common.tool import Tool
from metta.sim.pure_single_episode_runner import PureSingleEpisodeJob, run_pure_single_episode


class PureSingleEpisodeTool(Tool):
    job: PureSingleEpisodeJob

    def invoke(self, args: dict[str, str]) -> int:
        run_pure_single_episode(self.job)
        return 0


def run_example(policy_uri: str, results_uri: str, replay_uri: str) -> PureSingleEpisodeTool:
    """
    ./tools/run.py recipes.experiment.pure_single_episode.run_example \
        policy_uri=metta://policy/relh.machina1_bc_dinky_sliced.hc.1209.12-neginf:v3 \
        results_uri=file://./results.json \
        replay_uri=file://./replay.json.z
    """
    env = Machina1OpenWorldMission.model_copy(deep=True).make_env()
    return PureSingleEpisodeTool(
        job=PureSingleEpisodeJob(
            policy_uris=[policy_uri],
            assignments=[0] * env.game.num_agents,
            env=env,
            results_uri=results_uri,
            replay_uri=replay_uri,
        )
    )
