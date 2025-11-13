"""Smoke tests for scripted policies implemented in Nim and Python.

These tests ensure scripted agents can 1) act as supervisor teachers inside the
training environment and 2) run through a short `cogames.play` rollout. This
prevents regressions like missing bindings or policy registration mistakes.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from functools import cache
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


@dataclass(frozen=True)
class PolicyUnderTest:
    reference: str
    requires_nim: bool = False
    supports_supervisor: bool = False


@cache
def _nim_bindings_available() -> bool:
    try:
        import cogames.policy.nim_agents.agents as _  # noqa: F401
    except ModuleNotFoundError:
        return False
    return True


POLICIES_UNDER_TEST: tuple[PolicyUnderTest, ...] = (
    PolicyUnderTest("nim_thinky", requires_nim=True, supports_supervisor=True),
    PolicyUnderTest("nim_random", requires_nim=True, supports_supervisor=True),
    PolicyUnderTest("nim_race_car", requires_nim=True, supports_supervisor=True),
    PolicyUnderTest("scripted_baseline"),
    PolicyUnderTest("scripted_unclipping"),
    PolicyUnderTest(
        "cogames.policy.nim_agents.agents.ThinkyAgentsMultiPolicy",
        requires_nim=True,
        supports_supervisor=True,
    ),
    PolicyUnderTest("cogames.policy.scripted_agent.baseline_agent.BaselinePolicy"),
)

SUPERVISOR_POLICIES: tuple[PolicyUnderTest, ...] = tuple(p for p in POLICIES_UNDER_TEST if p.supports_supervisor)


@pytest.fixture(scope="module")
def simulator() -> Simulator:
    return Simulator()


@pytest.fixture
def env_config():
    env_cfg = make_training_env(num_cogs=2, mission="extractor_hub_30", variants=("lonely_heart",))
    env_cfg.game.max_steps = 8
    return env_cfg


def _policy_ids(policies: Iterable[PolicyUnderTest]) -> list[str]:
    return [policy.reference.replace("cogames.policy.", "").replace(".", "_") for policy in policies]


@pytest.mark.parametrize("policy", SUPERVISOR_POLICIES, ids=_policy_ids(SUPERVISOR_POLICIES))
def test_scripted_policies_work_as_supervisors(policy: PolicyUnderTest, simulator: Simulator, env_config) -> None:
    """Supervisor policies must load and generate teacher actions for training."""

    if policy.requires_nim and not _nim_bindings_available():
        pytest.skip("Nim bindings are missing. Run nim c nim_agents.nim to build them.")

    env = MettaGridPufferEnv(simulator, env_config, EnvSupervisorConfig(policy=policy.reference))
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


@pytest.mark.parametrize("policy", POLICIES_UNDER_TEST, ids=_policy_ids(POLICIES_UNDER_TEST))
def test_scripted_policies_can_play_short_episode(policy: PolicyUnderTest, env_config) -> None:
    """Policies should run through a short cogames.play session."""

    if policy.requires_nim and not _nim_bindings_available():
        pytest.skip("Nim bindings are missing. Run nim c nim_agents.nim to build them.")

    console = Console(file=io.StringIO(), force_terminal=False, soft_wrap=True, width=80)
    policy_class_path = resolve_policy_class_path(policy.reference)
    policy_spec = PolicySpec(policy_class_path=policy_class_path, policy_data_path=None)

    play_episode(
        console=console,
        env_cfg=env_config,
        policy_spec=policy_spec,
        game_name="extractor_hub_30",
        seed=42,
        render_mode="none",
    )
