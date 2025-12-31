"""Smoke tests for scripted policies implemented in Nim and Python.

These tests ensure scripted agents can 1) act as supervisor teachers inside the
training environment and 2) run through a short `cogames.play` rollout. This
prevents regressions like missing bindings or policy registration mistakes.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from functools import cache

import numpy as np
import pytest
from rich.console import Console

from cogames.cli.mission import get_mission
from cogames.play import play as play_episode
from metta.common.util.log_config import init_mettagrid_system_environment
from mettagrid.envs.mettagrid_puffer_env import MettaGridPufferEnv
from mettagrid.policy.loader import discover_and_register_policies
from mettagrid.policy.policy import PolicySpec
from mettagrid.simulator import Simulator

discover_and_register_policies("cogames.policy")
init_mettagrid_system_environment()


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
    PolicyUnderTest("thinky", requires_nim=True, supports_supervisor=True),
    PolicyUnderTest("nim_random", requires_nim=True, supports_supervisor=True),
    PolicyUnderTest("race_car", requires_nim=True, supports_supervisor=True),
    PolicyUnderTest("scripted_baseline"),
    PolicyUnderTest("ladybug"),
    PolicyUnderTest(
        "cogames.policy.nim_agents.agents.ThinkyAgentsMultiPolicy",
        requires_nim=True,
        supports_supervisor=True,
    ),
    PolicyUnderTest("cogames.policy.scripted_agent.baseline_agent.BaselinePolicy"),
    PolicyUnderTest("cogames.policy.scripted_agent.starter_agent.StarterPolicy"),
)

SUPERVISOR_POLICIES: tuple[PolicyUnderTest, ...] = tuple(p for p in POLICIES_UNDER_TEST if p.supports_supervisor)


def _policy_param(policy: PolicyUnderTest):  # -> pytest.ParameterSet
    marks = ()
    if policy.requires_nim and not _nim_bindings_available():
        marks = pytest.mark.skip("Nim bindings missing. Run `nim c nim_agents.nim` to build them.")
    policy_id = policy.reference.replace("cogames.policy.", "").replace(".", "_")
    return pytest.param(policy, id=policy_id, marks=marks)


POLICY_PARAMS = tuple(_policy_param(policy) for policy in POLICIES_UNDER_TEST)
SUPERVISOR_PARAMS = tuple(_policy_param(policy) for policy in SUPERVISOR_POLICIES)


@pytest.fixture(scope="module")
def simulator() -> Simulator:
    return Simulator()


@pytest.fixture
def env_config():
    _, env_cfg, _ = get_mission("evals.diagnostic_chest_navigation1", variants_arg=None, cogs=2)
    env_cfg.game.max_steps = 8
    return env_cfg


@pytest.mark.parametrize("policy", SUPERVISOR_PARAMS)
def test_scripted_policies_work_as_supervisors(policy: PolicyUnderTest, simulator: Simulator, env_config) -> None:
    """Supervisor policies must load and generate teacher actions for training."""

    env = MettaGridPufferEnv(simulator, env_config, supervisor_policy_spec=PolicySpec(class_path=policy.reference))
    try:
        observations, _ = env.reset(seed=123)
        assert observations.shape[0] == env_config.game.num_agents

        teacher_actions = env.teacher_actions
        assert teacher_actions.shape == (env_config.game.num_agents,)

        assert env._sim is not None
        noop_idx = env._sim.action_names.index("noop")
        noop_actions = np.full(env_config.game.num_agents, noop_idx, dtype=np.int32)

        next_obs, rewards, terminals, truncations, _ = env.step(noop_actions)
        assert next_obs.shape == observations.shape
        assert rewards.shape == (env_config.game.num_agents,)
        assert terminals.shape == (env_config.game.num_agents,)
        assert truncations.shape == (env_config.game.num_agents,)
    finally:
        env.close()


@pytest.mark.parametrize("policy", POLICY_PARAMS)
def test_scripted_policies_can_play_short_episode(policy: PolicyUnderTest, env_config) -> None:
    """Policies should run through a short cogames.play session."""

    console = Console(file=io.StringIO(), force_terminal=False, soft_wrap=True, width=80)
    policy_spec = PolicySpec(class_path=policy.reference, data_path=None)

    play_episode(
        console=console,
        env_cfg=env_config,
        policy_spec=policy_spec,
        game_name="diagnostic_chest_navigation1",
        seed=42,
        render_mode="none",
    )
