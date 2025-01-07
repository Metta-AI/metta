import hydra
import numpy as np
import mettagrid
import mettagrid.mettagrid_env
from regen_env_trace import render_obs_to_string, render_to_string, header

@hydra.main(version_base=None, config_path="../configs", config_name="test_basic")
def main(cfg):
    output = ""

    output += header("Last Action Tracker:")
    np.random.seed(123)
    cfg.track_last_action = True
    env = mettagrid.mettagrid_env.MettaGridEnv(render_mode=None, **cfg)
    env.reset()

    actions = [
        [0,5],
        [1,6],
        [2,7],
        [3,8],
        [4,9],
    ]
    output += render_to_string(env)

    (obs, rewards, terminated, truncated, infos) = env.step(actions)
    output += header("Observations:")
    output += render_obs_to_string(env, obs, match="last_action")

    output += f"grid_features: {env.grid_features}\n"

    assert "last_action" in env.grid_features
    assert "last_action_argument" in env.grid_features

    output += f"rewards: {rewards}\n"
    output += f"terminated: {terminated}\n"
    output += f"truncated: {truncated}\n"
    output += f"infos: {infos}\n"
    output += f"obs.shape: {obs.shape}\n"

    output += header("# No Last Action Tracker:")

    cfg.track_last_action = False
    env = mettagrid.mettagrid_env.MettaGridEnv(render_mode=None, **cfg)
    output += f"grid_features: {env.grid_features}\n"

    assert "last_action" not in env.grid_features
    assert "last_action_argument" not in env.grid_features

    (obs, rewards, terminated, truncated, infos) = env.step([[1,2]]*5)
    output += f"rewards: {rewards}\n"
    output += f"terminated: {terminated}\n"
    output += f"truncated: {truncated}\n"
    output += f"infos: {infos}\n"
    output += f"obs.shape: {obs.shape}\n"

    with open("tests/gold/track_last_action_trace.txt", "w") as f:
        f.write(output)

if __name__ == "__main__":
    main()
