"""Smoke tests for scripted policies implemented in Nim and Python.

These tests ensure scripted agents can 1) act as supervisor teachers inside the
training environment and 2) run through a short `cogames.play` rollout. This
prevents regressions like missing bindings or policy registration mistakes.
"""

from __future__ import annotations

import io
from typing import Iterable

import numpy as np
import pytest
from rich.console import Console

from cogames.play import play as play_episode
from experiments.recipes.cogs_v_clips import make_training_env
from metta.common.tool.run_tool import init_mettagrid_system_environment
from mettagrid.config.mettagrid_config import EnvSupervisorConfig
from mettagrid.envs.mettagrid_puffer_env import MettaGridPufferEnv
from mettagrid.policy.loader import discover_and_register_policies, resolve_policy_class_path
from mettagrid.policy.policy import PolicySpec
from mettagrid.simulator import Simulator

init_mettagrid_system_environment()
discover_and_register_policies("cogames.policy")


SCRIPTED_POLICY_REFS: tuple[str, ...] = (
    "nim_thinky",
    "nim_random",
    "nim_race_car",
    "scripted_baseline",
    "scripted_unclipping",
    "cogames.policy.nim_agents.agents.ThinkyAgentsMultiPolicy",
    "cogames.policy.scripted_agent.baseline_agent.BaselinePolicy",
)


@pytest.fixture(scope="module")
def simulator() -> Simulator:
    return Simulator()


@pytest.fixture(scope="module")
def base_env_config():
    env_cfg = make_training_env(num_cogs=2, mission="extractor_hub_30", variants=("lonely_heart",))
    env_cfg.game.max_steps = 8
    return env_cfg


@pytest.fixture
def env_config(base_env_config):
    # Provide an isolated copy so tests can adjust fields safely.
    return base_env_config.model_copy(deep=True)


def _policy_ids(policies: Iterable[str]) -> list[str]:
    return [policy.replace("cogames.policy.", "").replace(".", "_") for policy in policies]


@pytest.mark.parametrize("policy_ref", SCRIPTED_POLICY_REFS, ids=_policy_ids(SCRIPTED_POLICY_REFS))
def test_scripted_policies_work_as_supervisors(policy_ref: str, simulator: Simulator, env_config) -> None:
    """Supervisor policies must load and generate teacher actions for training."""

    env = MettaGridPufferEnv(simulator, env_config, EnvSupervisorConfig(policy=policy_ref))
    try:
        observations, _ = env.reset(seed=123)
        assert observations.shape[0] == env_config.game.num_agents

        teacher_actions = env.teacher_actions
        assert teacher_actions.shape == (env_config.game.num_agents,)

        noop_idx = env._sim.action_names.index("noop")
        noop_actions = np.full(env_config.game.num_agents, noop_idx, dtype=np.int32)

        next_obs, rewards, terminals, truncations, _ = env.step(noop_actions)
        assert next_obs.shape == observations.shape
        assert rewards.shape == (env_config.game.num_agents,)
        assert terminals.shape == (env_config.game.num_agents,)
        assert truncations.shape == (env_config.game.num_agents,)
    finally:
        env.close()


@pytest.mark.parametrize("policy_ref", SCRIPTED_POLICY_REFS, ids=_policy_ids(SCRIPTED_POLICY_REFS))
def test_scripted_policies_can_play_short_episode(policy_ref: str, env_config) -> None:
    """Policies should run through a short cogames.play session."""

    console = Console(file=io.StringIO(), force_terminal=False, soft_wrap=True, width=80)
    policy_class_path = resolve_policy_class_path(policy_ref)
    policy_spec = PolicySpec(policy_class_path=policy_class_path, policy_data_path=None)

    play_episode(
        console=console,
        env_cfg=env_config,
        policy_spec=policy_spec,
        game_name="extractor_hub_30",
        seed=42,
        render_mode="none",
    )
