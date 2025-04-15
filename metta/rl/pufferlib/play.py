import numpy as np
import torch
from omegaconf import OmegaConf

from metta.agent.policy_store import PolicyStore
from metta.rl.pufferlib.vecenv import make_vecenv
from mettagrid.renderer.raylib.raylib_renderer import MettaGridRaylibRenderer

def play(cfg: OmegaConf, policy_store: PolicyStore):
    device = cfg.device
    vecenv = make_vecenv(cfg.eval.env, cfg.vectorization, num_envs=1, render_mode="human")

    obs, _ = vecenv.reset()
    env = vecenv.envs[0]
    policy_record = policy_store.policy(cfg.policy_uri)

    assert policy_record.metadata["action_names"] == env._c_env.action_names(), (
        f"Action names do not match: {policy_record.metadata['action_names']} != {env._c_env.action_names()}"
    )
    policy = policy_record.policy()

    renderer = MettaGridRaylibRenderer(env._c_env, env._env_cfg.game)
    policy_rnn_state = None

    rewards = np.zeros(vecenv.num_agents)
    total_rewards = np.zeros(vecenv.num_agents)

    while True:
        with torch.no_grad():
            obs = torch.as_tensor(obs).to(device=device)

            # Parallelize across opponents
            actions, _, _, _, policy_rnn_state, _, _, _ = policy(obs, policy_rnn_state)
            if actions.dim() == 0:  # scalar tensor like tensor(2)
                actions = torch.tensor([actions.item()])

        renderer.update(
            actions.cpu().numpy(),
            obs,
            rewards,
            total_rewards,
            env._c_env.current_timestep(),
        )
        renderer.render_and_wait()
        actions = renderer.get_actions()

        obs, rewards, dones, truncated, infos = vecenv.step(actions)
        total_rewards += rewards
        if any(dones) or any(truncated):
            print(f"Total rewards: {total_rewards}")
            break

    import pandas as pd

    agent_df = pd.DataFrame([infos[0]["agent"]]).T.reset_index()
    agent_df.columns = ["stat", "value"]
    agent_df = agent_df.sort_values("stat")
    game_df = pd.DataFrame([infos[0]["game"]]).T.reset_index()
    game_df.columns = ["stat", "value"]
    game_df = game_df.sort_values("stat")
    print("\nAgent stats:")
    print(agent_df.to_string(index=False))
    print("\nGame stats:")
    print(game_df.to_string(index=False))
