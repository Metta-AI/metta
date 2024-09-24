from omegaconf import OmegaConf
import torch
import numpy as np
from rl.pufferlib.policy import load_policy_from_uri
from mettagrid.renderer.raylib.raylib_renderer import MettaGridRaylibRenderer
from rl.pufferlib.vecenv import make_vecenv
from rl.pufferlib.dashboard.dashboard import Dashboard
def play(cfg: OmegaConf, dashboard: Dashboard):
    device = cfg.device
    vecenv = make_vecenv(cfg, num_envs=1, render_mode="human")

    policy = load_policy_from_uri(cfg.eval.policy_uri, cfg)
    dashboard.set_policy(policy)

    obs, _ = vecenv.reset()
    env = vecenv.envs[0]

    assert policy._action_names == env._c_env.action_names(), \
        f"Action names do not match: {policy._action_names} != {env._c_env.action_names()}"

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

        renderer.update(
            actions,
            obs,
            env._c_env.current_timestep()
        )
        renderer.render_and_wait()
        actions = renderer.get_actions()

        obs, rewards, dones, truncated, infos = vecenv.step(actions.cpu().numpy())
        total_rewards += rewards
        if any(dones) or any(truncated):
            print(f"Total rewards: {total_rewards}")
            break
