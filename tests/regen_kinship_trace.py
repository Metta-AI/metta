import hydra
import numpy as np
import mettagrid
import mettagrid.mettagrid_env
from regen_env_trace import render_obs_to_string, render_to_string, header, dump_agents

@hydra.main(version_base=None, config_path="../configs", config_name="test_basic")
def main(cfg):
    output = ""

    output += header("Kinship:")
    np.random.seed(123)
    cfg.kinship.enabled = True
    cfg.kinship.team_size = 3
    cfg.kinship.team_reward = 1.0
    cfg.game.map.room.num_agents = 20
    cfg.game.map.room.objects.agent = 20
    cfg.game.num_agents = 20
    env = mettagrid.mettagrid_env.MettaGridEnv(render_mode=None, **cfg)
    env.reset()

    actions = [
        [0,5],
        [1,6],
        [2,7],
        [3,8],
        [4,9],
    ] * 4
    output += render_to_string(env, show_team=True)
    output += dump_agents(env)

    (obs, rewards, terminated, truncated, infos) = env.step(actions)
    output += header("Observations:")
    output += render_obs_to_string(env, obs, match="kinship")

    output += f"grid_features: {env.grid_features}\n"

    # assert "kinship" in env.grid_features

    output += f"rewards: {rewards}\n"
    output += f"terminated: {terminated}\n"
    output += f"truncated: {truncated}\n"
    output += f"infos: {infos}\n"
    output += f"obs.shape: {obs.shape}\n"

    output += header("Kinship reward sharing")
    actions = [[0, 0]] * 20
    # Rotate agent 12 to face the altar
    actions[12] = [0, 3]
    (obs, rewards, terms, truncs, infos) = env.step(actions)
    output += render_to_string(env, show_team=True)
    output += dump_agents(env, show_team=True)
    output += f"rewards: {rewards}\n"
    # Move agent 12 to the altar
    actions[12] = [1, 1]
    (obs, rewards, terms, truncs, infos) = env.step(actions)
    output += render_to_string(env, show_team=True)
    output += dump_agents(env, show_team=True)
    output += f"rewards: {rewards}\n"
    # Make agent 12 use the the altar
    actions[12] = [3, 0]
    (obs, rewards, terms, truncs, infos) = env.step(actions)
    output += render_to_string(env, show_team=True)
    output += dump_agents(env, show_team=True)
    output += f"rewards: {rewards}\n"

    #output += header("# No Last Action Tracker:")

    # cfg.kinship.enabled = False
    # env = mettagrid.mettagrid_env.MettaGridEnv(render_mode=None, **cfg)
    # output += f"grid_features: {env.grid_features}\n"

    # assert "last_action" not in env.grid_features
    # assert "last_action_argument" not in env.grid_features

    # (obs, rewards, terminated, truncated, infos) = env.step([[1,2]]*5)
    # output += f"rewards: {rewards}\n"
    # output += f"terminated: {terminated}\n"
    # output += f"truncated: {truncated}\n"
    # output += f"infos: {infos}\n"
    # output += f"obs.shape: {obs.shape}\n"

    with open("tests/gold/track_kinship_trace.txt", "w") as f:
        f.write(output)

if __name__ == "__main__":
    main()
