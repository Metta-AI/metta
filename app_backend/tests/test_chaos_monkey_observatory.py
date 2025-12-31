import uuid

import pytest

from cogames.policy.chaos_monkey import ChaosMonkeyPolicy
from metta.app_backend.clients.base_client import get_machine_token
from metta.app_backend.clients.stats_client import StatsClient
from metta.app_backend.routes.stats_routes import EpisodeQueryRequest
from metta.sim.handle_results import write_eval_results_to_observatory
from metta.sim.runner import SimulationRunConfig, SimulationRunResult
from metta.tools.utils.auto_config import auto_stats_server_uri
from mettagrid.config.mettagrid_config import MettaGridConfig
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.simulator.multi_episode.rollout import multi_episode_rollout


def _get_stats_client() -> StatsClient:
    stats_server_uri = auto_stats_server_uri()
    if not stats_server_uri:
        pytest.skip("Stats server not configured; skipping observatory smoke test.")
    machine_token = get_machine_token(stats_server_uri)
    if not machine_token:
        pytest.skip("No observatory auth token available; skipping observatory smoke test.")

    stats_client = StatsClient(backend_url=stats_server_uri, machine_token=machine_token)
    response = stats_client._http_client.get("/whoami", headers={"X-Auth-Token": machine_token})
    if response.status_code != 200 or response.json().get("user_email") in {"unknown", None}:
        pytest.skip("No observatory auth token available; skipping observatory smoke test.")
    return stats_client


def test_chaos_monkey_observatory_roundtrip() -> None:
    stats_client = _get_stats_client()

    policy_name = f"chaos-monkey-smoke-{uuid.uuid4()}"
    policy_id = stats_client.create_policy(name=policy_name).id
    policy_spec = {
        "class_path": "cogames.policy.chaos_monkey.ChaosMonkeyPolicy",
        "init_kwargs": {"fail_step": 10, "fail_probability": 1.0},
    }
    policy_version_id = stats_client.create_policy_version(policy_id=policy_id, policy_spec=policy_spec).id

    cfg = MettaGridConfig.EmptyRoom(num_agents=2, width=4, height=4, with_walls=True)
    cfg.game.max_steps = 12

    policy_env_info = PolicyEnvInterface.from_mg_cfg(cfg)
    policy = ChaosMonkeyPolicy(policy_env_info, fail_step=10, fail_probability=1.0)
    rollout_result = multi_episode_rollout(env_cfg=cfg, policies=[policy], episodes=1, seed=0)

    tag_value = str(uuid.uuid4())
    sim_cfg = SimulationRunConfig(env=cfg, num_episodes=1, episode_tags={"chaos_monkey_smoke": tag_value})
    sim_result = SimulationRunResult(run=sim_cfg, results=rollout_result)

    write_eval_results_to_observatory(
        policy_version_ids=[str(policy_version_id)],
        rollout_results=[sim_result],
        stats_client=stats_client,
        primary_policy_version_id=str(policy_version_id),
    )

    response = stats_client.query_episodes(
        EpisodeQueryRequest(
            primary_policy_version_ids=[policy_version_id],
            tag_filters={"chaos_monkey_smoke": [tag_value]},
            limit=1,
        )
    )

    assert response.episodes, "No episodes found for chaos monkey smoke test."
    episode = response.episodes[0]
    metrics = episode.policy_metrics.get(str(policy_version_id))
    if not metrics:
        pytest.skip("Policy metrics not returned by stats server; skipping observatory policy metrics assertion.")
    assert metrics.get("exception_flag") == 1.0
    assert metrics.get("exception_step") == 10.0
