import hydra
import numpy as np

# Make sure all modules import without errors:
import mettagrid

import mettagrid.mettagrid_env
import mettagrid.objects
import mettagrid.observation_encoder

import mettagrid.actions.actions
import mettagrid.actions.attack
import mettagrid.actions.move
import mettagrid.actions.noop
import mettagrid.actions.rotate
import mettagrid.actions.swap

import mettagrid.action
import mettagrid.event
import mettagrid.grid_env
import mettagrid.grid_object
import mettagrid.observation_encoder

# Make sure all dependencies are installed:
import hydra
import jmespath
import matplotlib
import pettingzoo
import pynvml
import pytest
import yaml
import raylib
import rich
import scipy
import tabulate
import tensordict
import torchrl
import termcolor
import wandb
import wandb_core
import pandas
import tqdm

@hydra.main(version_base=None, config_path="../configs", config_name="test_basic")
def main(cfg):

    # Create the environment:
    mettaGridEnv = mettagrid.mettagrid_env.MettaGridEnv(cfg, render_mode=None)

    # Make sure the environment was created correctly:
    print("mettaGridEnv._renderer: ", mettaGridEnv._renderer)
    assert mettaGridEnv._renderer is None
    print("mettaGridEnv._c_env: ", mettaGridEnv._c_env)
    assert mettaGridEnv._c_env is not None
    print("mettaGridEnv._grid_env: ", mettaGridEnv._grid_env)
    assert mettaGridEnv._grid_env is not None
    assert mettaGridEnv._c_env == mettaGridEnv._grid_env
    print("mettaGridEnv.done: ", mettaGridEnv.done)
    assert mettaGridEnv.done == False

    # Make sure reset works:
    mettaGridEnv.reset()

    # Run a single step:
    print("current_timestep: ", mettaGridEnv._c_env.current_timestep())
    assert mettaGridEnv._c_env.current_timestep() == 0
    (obs, rewards, terminated, truncated, infos) = mettaGridEnv.step([[0,0]]*5)
    assert mettaGridEnv._c_env.current_timestep() == 1
    print("obs: ", obs)
    # We have 5 agents, ~22 channels, 11x11 grid
    # We expect the number of channels to be updated more regularly, so we give that a range.
    # Feel free to make this less fragile if you're updating this.
    [num_agents, grid_width, grid_height, num_channels] = obs.shape
    assert num_agents == 5
    assert grid_width == 11
    assert grid_height == 11
    assert 20 <= num_channels <= 50
    print("rewards: ", rewards)
    assert rewards.shape == (5,)
    print("terminated: ", terminated)
    assert np.array_equal(terminated, [0, 0, 0, 0, 0])
    print("truncated: ", truncated)
    assert np.array_equal(truncated, [0, 0, 0, 0, 0])
    print("infos: ", infos)

    print(mettaGridEnv._c_env.render())

    print("grid_objects: ")
    for grid_object in mettaGridEnv._c_env.grid_objects().values():
      print(f"* {grid_object}")

    infos = {}
    mettaGridEnv.process_episode_stats(infos)
    print("process_episode_stats infos: ", infos)

    # Print some environment info:
    print("mettaGridEnv._max_steps: ", mettaGridEnv._max_steps)
    assert mettaGridEnv._max_steps == 5000
    print("mettaGridEnv.single_observation_space: ", mettaGridEnv.single_observation_space)
    assert mettaGridEnv.single_observation_space.shape == (grid_width, grid_height, num_channels)
    print("mettaGridEnv.single_action_space: ", mettaGridEnv.single_action_space)
    [num_actions, max_arg] = mettaGridEnv.single_action_space.nvec.tolist()
    # We don't want to hard-code the number of actions to expect (we might add more), so
    # we do some loose testing of "is this a reasonable number of actions?"
    assert 5 <= num_actions <= 20, f"num_actions: {num_actions}"
    # Same this for max_arg. No reason the it's "reasonable range" should be the same as num_actions,
    # but it happens to be.
    assert 5 <= max_arg <= 20, f"max_arg: {max_arg}"
    print("mettaGridEnv.action_names(): ", mettaGridEnv.action_names())
    print("mettaGridEnv.grid_features: ", mettaGridEnv.grid_features)
    print("mettaGridEnv.global_features: ", mettaGridEnv.global_features)
    print("mettaGridEnv.render_mode: ", mettaGridEnv.render_mode)
    assert mettaGridEnv.render_mode == None

    print("mettaGridEnv._c_env.map_width(): ", mettaGridEnv._c_env.map_width())
    assert mettaGridEnv._c_env.map_width() == 25
    print("mettaGridEnv._c_env.map_height(): ", mettaGridEnv._c_env.map_height())
    assert mettaGridEnv._c_env.map_height() == 25

    print("mettaGridEnv._c_env.num_agents(): ", mettaGridEnv._c_env.num_agents())
    assert mettaGridEnv._c_env.num_agents() == 5

    # Test action success:
    print("mettaGridEnv.action_success: ", mettaGridEnv.action_success)
    assert mettaGridEnv.action_success.shape == (5,)

if __name__ == "__main__":
    main()
