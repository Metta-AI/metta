"""
Test suite for mettagrid-specific curriculum functionality.

This module tests mettagrid-specific curriculum features. 
General curriculum tests are in tests/rl/curriculum/.
"""

import random

import numpy as np
import pytest
from omegaconf import OmegaConf

from metta.mettagrid.curriculum.core import SingleTaskCurriculum


@pytest.fixture(autouse=True)
def set_random_seeds():
    """Set all random seeds for deterministic test behavior."""
    random.seed(42)
    np.random.seed(42)
    yield
    # Reset after test
    random.seed()
    np.random.seed()


@pytest.fixture
def env_cfg():
    return OmegaConf.create({"sampling": 0, "game": {"num_agents": 1, "map": {"width": 10, "height": 10}}})


def test_mettagrid_single_task_curriculum_integration(env_cfg):
    """Test SingleTaskCurriculum with mettagrid-specific configuration."""
    curr = SingleTaskCurriculum("mettagrid_task", env_cfg)
    task = curr.get_task()
    
    # Test mettagrid-specific attributes
    assert task.id() == "mettagrid_task"
    assert task.env_cfg() == env_cfg
    assert hasattr(task.env_cfg(), "game")
    assert hasattr(task.env_cfg().game, "num_agents")
    assert hasattr(task.env_cfg().game.map, "width")
    assert hasattr(task.env_cfg().game.map, "height")
    
    # Test completion
    assert not task.is_complete()
    task.complete(0.5)
    assert task.is_complete()


# Note: Most curriculum tests have been moved to tests/rl/curriculum/
# This file now contains only mettagrid-specific integration tests.
