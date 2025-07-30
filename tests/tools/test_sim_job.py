from typing import List

import hydra
import pytest
from hydra.core.global_hydra import GlobalHydra

from tools.sim import SimJob


@pytest.fixture(scope="session", autouse=True)
def hydra_init_once():
    GlobalHydra.instance().clear()
    hydra.initialize(config_path="../../configs", version_base=None)


@pytest.fixture
def build_config():
    def _build(overrides: List[str]):
        hydra_cfg = hydra.compose(config_name="sim_job", overrides=overrides)
        return SimJob(hydra_cfg.sim_job)

    return _build


def test_config_defaults(build_config):
    cfg = build_config(["run=test"])
    assert cfg.selector_type == "top"
    assert len(cfg.policy_uris) == 1
    assert cfg.policy_uris[0].endswith("test/checkpoints")
    assert cfg.simulation_suite.name == "all"


def test_config_overrides(build_config):
    cfg = build_config(
        [
            "run=test",
            "sim=smoke_test",
            "policy_uri=test_policy",
            "sim.num_episodes=10",
        ]
    )
    print(cfg)
    assert cfg.policy_uris == ["test_policy"]
    assert cfg.simulation_suite.name == "smoke_test"
    assert cfg.simulation_suite.num_episodes == 10
    assert cfg.simulation_suite.simulations["emptyspace_withinsight"].num_episodes == 10
