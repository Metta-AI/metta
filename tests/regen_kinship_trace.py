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
    cfg.game.map_builder.agents = 20
    cfg.game.map_builder.objects.altar = 10
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

    assert "agent:group" in env.grid_features

    output += f"rewards: {rewards}\n"
    output += f"terminated: {terminated}\n"
    output += f"truncated: {truncated}\n"
    output += f"infos: {infos}\n"
    output += f"obs.shape: {obs.shape}\n"

    output += header("Kinship reward sharing")
    # Here we will have agent 12 move and use the altar, and it should show
    # the sharing of the reward with the other agents in the team.
    actions = [[0, 0]] * 20
    # Rotate agent 12 to face the altar
    actions[12] = [2, 2]
    (obs, rewards, terms, truncs, infos) = env.step(actions)
    output += render_to_string(env, show_team=True)
    output += dump_agents(env, show_team=True, agent_id=12)
    output += f"rewards: {rewards}\n"
    # Move agent 12 to the altar
    actions[12] = [1, 0]
    (obs, rewards, terms, truncs, infos) = env.step(actions)
    output += render_to_string(env, show_team=True)
    output += dump_agents(env, show_team=True, agent_id=12)
    output += f"rewards: {rewards}\n"
    # Make agent 12 use the the altar
    actions[12] = [3, 0]
    (obs, rewards, terms, truncs, infos) = env.step(actions)
    output += render_to_string(env, show_team=True)
    output += dump_agents(env, show_team=True, agent_id=12)
    output += f"rewards: {rewards}\n"

    output += header("# No Kinship:")

    env = mettagrid.mettagrid_env.MettaGridEnv(render_mode=None, **cfg)
    output += f"grid_features: {env.grid_features}\n"

    assert "kinship" not in env.grid_features

    (obs, rewards, terminated, truncated, infos) = env.step([[0,0]]*20)
    output += render_obs_to_string(env, obs, match="kinship")

    with open("tests/gold/track_kinship_trace.txt", "w") as f:
        f.write(output)

if __name__ == "__main__":
    main()
