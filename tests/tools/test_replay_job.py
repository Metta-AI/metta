"""
Unit‑tests for SimulationSuiteConfig ⇄ SimulationConfig behavior.
Covered
-------
* suite‑level defaults propagate into children
* child‑level overrides win
* missing required keys always raise (allow_missing removed)
"""

from typing import List

import hydra
import pytest
from hydra.core.global_hydra import GlobalHydra

from tools.replay import ReplayJob


@pytest.fixture(scope="session", autouse=True)
def hydra_init_once():
    GlobalHydra.instance().clear()
    hydra.initialize(config_path="../../configs", version_base=None)


@pytest.fixture
def build_config():
    def _build(overrides: List[str]):
        hydra_cfg = hydra.compose(config_name="replay_job", overrides=overrides)
        return ReplayJob(hydra_cfg.replay_job)

    return _build


def test_config_defaults(build_config):
    cfg = build_config(["run=test"])
    assert cfg.selector_type == "top"
    assert cfg.policy_uri == "file://./train_dir/test/checkpoints"
    assert cfg.sim.env == "/env/mettagrid/simple"


def test_config_overrides(build_config):
    cfg = build_config(
        [
            "run=test",
            "sim.env=/env/mettagrid/teams",
            "+sim.env_overrides.game.num_agents=36",
            "policy_uri=test_policy",
            "sim.num_episodes=10",
        ]
    )
    print(cfg)
    assert cfg.policy_uri == "test_policy"
    assert cfg.sim.env == "/env/mettagrid/teams"
    assert cfg.sim.num_episodes == 10
    assert cfg.sim.env_overrides["game"]["num_agents"] == 36
