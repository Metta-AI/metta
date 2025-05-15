"""
Unit‑tests for SimulationSuiteConfig ⇄ SimulationConfig behavior.
Covered
-------
* suite‑level defaults propagate into children
* child‑level overrides win
* missing required keys always raise (allow_missing removed)
"""

from typing import Dict

import pytest
from omegaconf import OmegaConf
from pydantic import ValidationError

from metta.sim.simulation_config import SimulationSuiteConfig

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------
ROOT_ENV, CHILD_A, CHILD_B = "env/root", "env/a", "env/b"
DEVICE = "cpu"


@pytest.fixture
def build_simulation_suite_config():
    def _build(cfg: Dict):
        # First create the OmegaConf object
        dict_config = OmegaConf.create(cfg)

        # Convert to a Python dictionary
        regular_dict = OmegaConf.to_container(dict_config, resolve=True)

        # Now create the SimulationSuiteConfig using the model_validate method
        return SimulationSuiteConfig.model_validate(regular_dict)

    return _build


# ---------------------------------------------------------------------------
# propagation & overrides
# ---------------------------------------------------------------------------
def test_propagate_defaults_and_overrides(build_simulation_suite_config):
    cfg = {
        "name": "test",
        "num_episodes": 4,
        "simulations": {
            "a": {"env": CHILD_A},  # inherits device, num_envs is default (50)
            "b": {"env": CHILD_B, "num_episodes": 8},  # overrides num_envs
        },
    }
    suite = build_simulation_suite_config(cfg)
    a, b = suite.simulations["a"], suite.simulations["b"]
    # device and num_envs both propagated, even though num_envs has a default
    assert a.num_episodes == 4
    assert b.num_episodes == 8


# ---------------------------------------------------------------------------
# allow_extra – child nodes
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "has_extra, should_pass",
    [
        (False, True),
        (True, False),
    ],
)
def test_allow_extra_child_keys(build_simulation_suite_config, has_extra, should_pass):
    child_node = {"env": CHILD_A}
    if has_extra:
        child_node["foo"] = "bar"  # <- unknown key
    cfg = {
        "name": "test",
        "num_episodes": 4,
        "simulations": {"sim": child_node},
    }
    if should_pass:
        suite = build_simulation_suite_config(cfg)
        assert suite.simulations["sim"].env == CHILD_A
    else:
        with pytest.raises(ValueError):
            build_simulation_suite_config(cfg)


# ---------------------------------------------------------------------------
# missing required keys should always error
# ---------------------------------------------------------------------------
def test_missing_device_always_errors(build_simulation_suite_config):
    cfg = {
        "num_episodes": 4,
        "simulations": {"sim": {}},
    }
    with pytest.raises(ValidationError):
        build_simulation_suite_config(cfg)


def test_missing_suite_env_is_allowed(build_simulation_suite_config):
    cfg = {
        "name": "test",
        "num_episodes": 4,
        "simulations": {
            "sim": {
                "env": CHILD_A,
            }
        },
    }
    suite = build_simulation_suite_config(cfg)
    assert suite.simulations["sim"].env == CHILD_A
