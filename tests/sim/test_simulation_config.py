"""
Unit‑tests for SimulationSuiteConfig  ⇄  SimulationConfig behaviour.

Covered
-------
* suite‑level defaults propagate into children
* child‑level overrides win
* four permutations of (extra‑key × allow_extra_keys)
* unknown key on the *suite* node raises unless allow_extra_keys=True
* missing required keys always raise  (allow_missing removed)
"""

from typing import Dict

import pytest
from omegaconf import OmegaConf

from metta.sim.simulation_config import SimulationSuiteConfig
from metta.util.config import dictconfig_to_dataclass

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------

ROOT_ENV, CHILD_A, CHILD_B = "env/root", "env/a", "env/b"
DEVICE, RUN_DIR = "cpu", "./runs/test"


def _build(cfg: Dict, *, allow_extra_keys: bool = False):
    """Helper around ``dictconfig_to_dataclass``."""
    return dictconfig_to_dataclass(
        SimulationSuiteConfig,
        OmegaConf.create(cfg),
        allow_extra_keys=allow_extra_keys,
    )


# ---------------------------------------------------------------------------
# propagation & overrides
# ---------------------------------------------------------------------------


def test_propogate_defaults_and_overrides():
    cfg = {
        "env": ROOT_ENV,
        "device": DEVICE,
        "run_dir": RUN_DIR,
        "num_envs": 4,  # suite‑wide default
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
    "has_extra, allow_extra, should_pass",
    [
        (False, False, True),
        (False, True, True),
        (True, True, True),  # extra tolerated
        (True, False, False),  # extra rejected
    ],
)
def test_allow_extra_child_keys(has_extra, allow_extra, should_pass):
    child_node = {"env": CHILD_A}
    if has_extra:
        child_node["foo"] = "bar"  # <- unknown key

    cfg = {
        "env": ROOT_ENV,
        "device": DEVICE,
        "run_dir": RUN_DIR,
        "simulations": {"sim": child_node},
    }

    if should_pass:
        suite = _build(cfg, allow_extra_keys=allow_extra)
        assert suite.simulations["sim"].device == DEVICE
    else:
        with pytest.raises(ValueError):
            _build(cfg, allow_extra_keys=allow_extra)


# ---------------------------------------------------------------------------
# allow_extra – suite node
# ---------------------------------------------------------------------------


def test_extra_key_on_suite_rejected_unless_allowed():
    cfg = {
        "env": ROOT_ENV,
        "device": DEVICE,
        "run_dir": RUN_DIR,
        "suite_only": "secret",
        "simulations": {"sim": {"env": CHILD_A}},
    }

    # default → error
    with pytest.raises(ValueError):
        _build(cfg)

    # allowed
    suite = _build(cfg, allow_extra_keys=True)
    assert suite.simulations["sim"].env == CHILD_A


# ---------------------------------------------------------------------------
# missing required keys should always error
# ---------------------------------------------------------------------------


def test_missing_device_always_errors():
    cfg = {
        "env": ROOT_ENV,
        "run_dir": RUN_DIR,
        "simulations": {"sim": {}},  # required 'device' omitted
    }
    with pytest.raises(TypeError):
        _build(cfg)
