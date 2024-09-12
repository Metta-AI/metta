from omegaconf import OmegaConf
import torch
import time
import numpy as np
from rl.pufferlib.policy import load_policy_from_uri
from util.stats import print_policy_stats

def evaluate(cfg: OmegaConf, vecenv):
    num_envs = cfg.eval.num_envs
    device = cfg.framework.pufferlib.device

    policy = load_policy_from_uri(cfg.eval.policy_uri, device)
    baselines = [load_policy_from_uri(b, device) for b in cfg.eval.baseline_uris]
    policy_agent_pct = cfg.eval.policy_agents_pct

    if len(baselines) == 0:
        policy_agent_pct = 1.0

    num_baselines = len(baselines)

    agents_count = cfg.env.game.num_agents
    policy_agents_count = max(1, int(cfg.env.game.num_agents * policy_agent_pct))
    baseline_agents_count = agents_count - policy_agents_count

    print(f'Policy Agents: {policy_agents_count}, Baseline Agents: {baseline_agents_count}')

    slice_idxs = torch.arange(vecenv.num_agents).reshape(num_envs, agents_count).to(device=device)
    policy_idxs = slice_idxs[:, :policy_agents_count].reshape(policy_agents_count * num_envs)

    baseline_idxs = []
    if num_baselines > 0:
        envs_per_opponent = num_envs // num_baselines
        baseline_idxs = slice_idxs[:, policy_agents_count:].reshape(num_envs*baseline_agents_count).split(baseline_agents_count*envs_per_opponent)

    obs, _ = vecenv.reset()
    policy_rnn_state = None
    baselines_rnn_state = [None for _ in range(num_baselines)]

    start = time.time()
    episodes = 0
    step = 0
    total_rewards = np.zeros(vecenv.num_agents)
    agent_stats = [{} for a in range(vecenv.num_agents)]
    while episodes < cfg.eval.num_episodes and time.time() - start < cfg.eval.max_time_s:
        step += 1
        baseline_actions = []
        with torch.no_grad():
            obs = torch.as_tensor(obs).to(device=device)
            my_obs = obs[policy_idxs]

            # Parallelize across opponents
            if hasattr(policy, 'lstm'):
                policy_actions, _, _, _, policy_rnn_state = policy(my_obs, policy_rnn_state)
            else:
                policy_actions, _, _, _ = policy(my_obs)

            # Iterate opponent policies
            for i in range(num_baselines):
                baseline_obs = obs[baseline_idxs[i]]
                baseline_rnn_state = baselines_rnn_state[i]

                baseline = baselines[i]
                if hasattr(policy, 'lstm'):
                    baseline_action, _, _, _, baselines_rnn_state[i] = baseline(baseline_obs, baseline_rnn_state)
                else:
                    baseline_action, _, _, _ = baseline(baseline_obs)

                baseline_actions.append(baseline_action)


        if num_baselines > 0:
            actions = torch.cat([
                policy_actions.view(num_envs, policy_agents_count, -1),
                torch.cat(baseline_actions, dim=1).view(num_envs, baseline_agents_count, -1),
            ], dim=1)
        else:
            actions = policy_actions

        actions = actions.view(num_envs*agents_count, -1)

        obs, rewards, dones, truncated, infos = vecenv.step(actions.cpu().numpy())
        total_rewards += rewards
        episodes += sum([e.done for e in vecenv.envs])
        agent_infos = []
        for info in infos:
            if "agent" in info:
                agent_infos.extend(info["agent"])

        for idx, info in enumerate(agent_infos):
            for k, v in info.items():
                if k not in agent_stats[idx]:
                    agent_stats[idx][k] = 0
                agent_stats[idx][k] += v

    policy_stats = [{} for a in range(num_baselines + 1)]
    policy_idxs = [policy_idxs] + list(baseline_idxs)
    for policy_idx, stats in enumerate(policy_stats):
        num_policy_agents = len(policy_idxs[policy_idx])
        for agent_idx in policy_idxs[policy_idx]:
            for k, v in agent_stats[agent_idx.item()].items():
                if k not in stats:
                    stats[k] = {"sum": 0, "count": num_policy_agents}
                stats[k]["sum"] += v

    print_policy_stats(policy_stats)

    return policy_stats
