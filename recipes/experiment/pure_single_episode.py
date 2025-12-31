import json
import os
import subprocess
import tempfile
import uuid

from cogames.cogs_vs_clips.missions import Machina1OpenWorldMission
from metta.app_backend.clients.base_client import get_machine_token
from metta.app_backend.clients.stats_client import StatsClient
from metta.common.tool import Tool
from metta.sim.pure_single_episode_runner import PureSingleEpisodeJob
from metta.tools.utils.auto_config import auto_stats_server_uri


class PureSingleEpisodeTool(Tool):
    job: PureSingleEpisodeJob

    def invoke(self, args: dict[str, str]) -> int:
        with tempfile.NamedTemporaryFile(delete=True) as temp_file:
            temp_file.write(
                json.dumps({"job": self.job.model_dump(), "device": "cpu", "allow_network": True}).encode("utf-8")
            )
            temp_file.flush()
            subprocess.run(["python", "-m", "metta.sim.pure_single_episode_runner", temp_file.name], check=True)
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


class SingleEpisodeTool(Tool):
    stats_server_uri: str | None = auto_stats_server_uri()
    job_id: uuid.UUID

    def invoke(self, args: dict[str, str]) -> int:
        if not self.stats_server_uri:
            raise ValueError("Stats server URI is not set")
        machine_token = get_machine_token(self.stats_server_uri)
        StatsClient(backend_url=self.stats_server_uri, machine_token=machine_token or "")._validate_authenticated()

        env = os.environ.copy()
        env["BACKEND_URL"] = self.stats_server_uri
        env["MACHINE_TOKEN"] = machine_token or ""
        subprocess.run(["python", "-m", "metta.sim.single_episode_runner", str(self.job_id)], check=True, env=env)
        return 0


def run_single_episode_example(job_id: str) -> SingleEpisodeTool:
    """
    ./tools/run.py recipes.experiment.pure_single_episode.run_single_episode_example job_id=my-uuid
    """
    return SingleEpisodeTool(job_id=uuid.UUID(job_id))
