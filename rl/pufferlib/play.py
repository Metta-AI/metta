from omegaconf import OmegaConf
import torch
import numpy as np
from mettagrid.renderer.raylib.raylib_renderer import MettaGridRaylibRenderer
from rl.pufferlib.vecenv import make_vecenv
from mettagrid.config.sample_config import sample_config
from agent.policy_store import PolicyRecord

def play(cfg: OmegaConf, policy_record: PolicyRecord):
    device = cfg.device
    vecenv = make_vecenv(cfg, num_envs=1, render_mode="human")

    obs, _ = vecenv.reset()
    env = vecenv.envs[0]
    policy = policy_record.policy()

    assert policy_record.metadata["action_names"] == env._c_env.action_names(), \
        f"Action names do not match: {policy_record.metadata['action_names']} != {env._c_env.action_names()}"

    game_cfg = OmegaConf.create(sample_config(cfg.env.game, cfg.env.sampling))
    renderer = MettaGridRaylibRenderer(env._c_env, game_cfg)
    policy_rnn_state = None

    rewards = np.zeros(vecenv.num_agents)
    total_rewards = np.zeros(vecenv.num_agents)

    while True:
        with torch.no_grad():
            obs = torch.as_tensor(obs).to(device=device)

            # Parallelize across opponents
            if hasattr(policy, 'lstm'):
                actions, _, _, _, policy_rnn_state = policy(obs, policy_rnn_state)
            else:
                actions, _, _, _ = policy(obs)

        renderer.update(
            env._c_env.unflatten_actions(actions.cpu().numpy()),
            obs,
            rewards,
            total_rewards,
            env._c_env.current_timestep(),
        )
        renderer.render_and_wait()
        actions = env._c_env.flatten_actions(renderer.get_actions())

        obs, rewards, dones, truncated, infos = vecenv.step(actions)
        total_rewards += rewards
        if any(dones) or any(truncated):
            print(f"Total rewards: {total_rewards}")
            break
