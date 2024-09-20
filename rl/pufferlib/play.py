from omegaconf import OmegaConf
import torch
import numpy as np
from rl.pufferlib.policy import load_policy_from_uri
from mettagrid.renderer.raylib.raylib_renderer import MettaGridRaylibRenderer
from rl.pufferlib.vecenv import make_vecenv

def play(cfg: OmegaConf):
    device = cfg.framework.pufferlib.device
    vecenv = make_vecenv(cfg, num_envs=1, render_mode="human")
    policy = load_policy_from_uri(cfg.eval.policy_uri, cfg)
    agents_count = cfg.env.game.num_agents

    obs, _ = vecenv.reset()
    env = vecenv.envs[0]
    renderer = MettaGridRaylibRenderer(env._c_env, cfg.env)
    policy_rnn_state = None

    total_rewards = np.zeros(vecenv.num_agents)
    while True:
        with torch.no_grad():
            obs = torch.as_tensor(obs).to(device=device)

            # Parallelize across opponents
            if hasattr(policy, 'lstm'):
                actions, _, _, _, policy_rnn_state = policy(obs, policy_rnn_state)
            else:
                actions, _, _, _ = policy(obs)

        render_result = renderer.render(
            env._c_env.current_timestep(),
            env._c_env.grid_objects(),
            actions,
            obs
        )

        obs, rewards, dones, truncated, infos = vecenv.step(render_result["actions"].cpu().numpy())
        total_rewards += rewards
        if any(dones) or any(truncated):
            print(f"Total rewards: {total_rewards}")
            break
