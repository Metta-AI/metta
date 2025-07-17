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
    assert cfg.policy_uri.endswith("test/checkpoints")
    assert cfg.sim.env == "/env/mettagrid/arena/advanced"


def test_config_overrides(build_config):
    cfg = build_config(
        [
            "run=test",
            "policy_uri=test_policy",
            "replay_job.sim.env=/env/mettagrid/arena/advanced",
            "+replay_job.sim.env_overrides.game.num_agents=36",
            "replay_job.sim.num_episodes=10",
        ]
    )
    print(cfg)
    assert cfg.policy_uri == "test_policy"
    assert cfg.sim.env == "/env/mettagrid/arena/advanced"
    assert cfg.sim.num_episodes == 10
    assert cfg.sim.env_overrides["game"]["num_agents"] == 36
