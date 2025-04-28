"""
Unit‑tests for SimulationSuiteConfig  ⇄  SimulationConfig behaviour.

Covered
-------
* suite‑level defaults propagate into children
* child‑level overrides win
* missing required keys always raise  (allow_missing removed)
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
DEVICE, RUN_DIR = "cpu", "./runs/test"


def _build(cfg: Dict):
    """Helper around ``dictconfig_to_dataclass``."""
    return SimulationSuiteConfig(OmegaConf.create(cfg))


# ---------------------------------------------------------------------------
# propagation & overrides
# ---------------------------------------------------------------------------


def test_propogate_defaults_and_overrides():
    cfg = {
        "env": ROOT_ENV,
        "num_envs": 4,
        "num_episodes": 4,
        "device": DEVICE,
        "run_dir": RUN_DIR,
        "simulations": {
            "a": {"env": CHILD_A},  # inherits device, num_envs is default (50)
            "b": {"env": CHILD_B, "num_envs": 8},  # overrides num_envs
        },
    }
    suite = _build(cfg)
    a, b = suite.simulations["a"], suite.simulations["b"]

    # device and num_envs both propagated, even though num_envs has a default
    assert (a.device, a.num_envs) == (DEVICE, 4)
    assert (b.device, b.num_envs) == (DEVICE, 8)


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
def test_allow_extra_child_keys(has_extra, should_pass):
    child_node = {"env": CHILD_A}
    if has_extra:
        child_node["foo"] = "bar"  # <- unknown key

    cfg = {
        "env": ROOT_ENV,
        "num_envs": 4,
        "num_episodes": 4,
        "device": DEVICE,
        "run_dir": RUN_DIR,
        "simulations": {"sim": child_node},
    }

    if should_pass:
        suite = _build(cfg)
        assert suite.simulations["sim"].device == DEVICE
    else:
        with pytest.raises(ValueError):
            _build(cfg)


# ---------------------------------------------------------------------------
# missing required keys should always error
# ---------------------------------------------------------------------------


def test_missing_device_always_errors():
    cfg = {
        "env": ROOT_ENV,
        "num_envs": 4,
        "num_episodes": 4,
        "run_dir": RUN_DIR,
        "simulations": {"sim": {}},  # required 'device' omitted
    }
    with pytest.raises(ValidationError):
        _build(cfg)


def test_missing_suite_env_is_allowed():
    cfg = {
        "run_dir": RUN_DIR,
        "device": DEVICE,
        "num_envs": 4,
        "num_episodes": 4,
        "simulations": {
            "sim": {
                "env": CHILD_A,
            }
        },
    }
    suite = _build(cfg)
    assert suite.simulations["sim"].env == CHILD_A
