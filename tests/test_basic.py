import hydra
import numpy as np

# Make sure all modules import without errors:
import mettagrid

import mettagrid.mettagrid_env
import mettagrid.objects
import mettagrid.observation_encoder

import mettagrid.actions.actions
import mettagrid.actions.attack
import mettagrid.actions.gift
import mettagrid.actions.move
import mettagrid.actions.noop
import mettagrid.actions.rotate
import mettagrid.actions.shield
import mettagrid.actions.swap
import mettagrid.actions.use

import mettagrid.config.game_builder
import mettagrid.config.sample_config

import puffergrid
import puffergrid.action
import puffergrid.event
import puffergrid.grid_env
import puffergrid.grid_object
import puffergrid.observation_encoder
import puffergrid.stats_tracker

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
    mettaGridEnv = mettagrid.mettagrid_env.MettaGridEnv(render_mode=None, **cfg)

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
    assert obs.shape == (5, 24, 11, 11)
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
    assert mettaGridEnv.single_observation_space.shape == (24, 11, 11)
    print("mettaGridEnv.single_action_space: ", mettaGridEnv.single_action_space)
    assert mettaGridEnv.single_action_space.nvec.tolist() == [8, 9]
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

if __name__ == "__main__":
    main()
