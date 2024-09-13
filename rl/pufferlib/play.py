from re import T
from omegaconf import OmegaConf
import torch
import time
import numpy as np
from rl.pufferlib.policy import load_policy_from_uri
from util.stats import print_policy_stats

def play(cfg: OmegaConf, vecenv):
    device = cfg.framework.pufferlib.device

    policy = load_policy_from_uri(cfg.eval.policy_uri, cfg)
    agents_count = cfg.env.game.num_agents

    obs, _ = vecenv.reset()
    policy_rnn_state = None

    start = time.time()
    total_rewards = np.zeros(vecenv.num_agents)
    agent_stats = [{} for a in range(vecenv.num_agents)]
    while True:
        with torch.no_grad():
            obs = torch.as_tensor(obs).to(device=device)

            # Parallelize across opponents
            if hasattr(policy, 'lstm'):
                actions, _, _, _, policy_rnn_state = policy(obs, policy_rnn_state)
            else:
                actions, _, _, _ = policy(obs)

        actions = actions.view(agents_count, -1)
        render_result = vecenv.envs[0].render()
        if render_result["action"] is not None and render_result["selected_agent_idx"] is not None:
            actions[render_result["selected_agent_idx"]] = torch.tensor(render_result["action"], device=actions.device)

        obs, rewards, dones, truncated, infos = vecenv.step(actions.cpu().numpy())
        total_rewards += rewards
