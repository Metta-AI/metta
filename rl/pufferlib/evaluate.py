from omegaconf import OmegaConf
import torch
import time
import numpy as np
import os
from rl.pufferlib.policy import load_policy_from_uri
# from util.stats import print_policy_stats
from util.eval_analyzer import print_policy_stats

def evaluate(cfg: OmegaConf, vecenv):
    device = cfg.framework.pufferlib.device
    num_envs = len(vecenv.envs)

    policy = load_policy_from_uri(cfg.eval.policy_uri, cfg)
    baselines = [load_policy_from_uri(b, cfg) for b in cfg.eval.baseline_uris]
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

    #---NEW---
    # Extract policy names from URIs
    policy_name = os.path.basename(cfg.eval.policy_uri)
    baseline_names = [os.path.basename(uri) for uri in cfg.eval.baseline_uris]

    # Create mapping from agent index to policy name
    agent_idx_to_policy_name = {}
    for agent_idx in policy_idxs:
        agent_idx_to_policy_name[agent_idx.item()] = policy_name
    for i, baseline_agent_idxs in enumerate(baseline_idxs):
        for agent_idx in baseline_agent_idxs:
            agent_idx_to_policy_name[agent_idx.item()] = baseline_names[i]
    #---END NEW---

    # Print the eval setup
    print(f"Policy is called {policy_name} and baselines are:")
    for i, name in enumerate(baseline_names):
        print(f"Baseline {i+1}: {name}")
    print(f'No. of parallel envs: {num_envs}')
    print(f"eval.num_episodes: {cfg.eval.num_episodes}")
    print(f'Policy agnts/env: {policy_agents_count}, Baseline agnts/env: {baseline_agents_count}')

    obs, _ = vecenv.reset()
    policy_rnn_state = None
    baselines_rnn_state = [None for _ in range(num_baselines)]

    start = time.time()
    episodes = 0
    step_count = 0 
    game_stats = []
    total_rewards = np.zeros(vecenv.num_agents)
    while episodes < cfg.eval.num_episodes and time.time() - start < cfg.eval.max_time_s:
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
        #---NEW---
        if len(infos) > 0:
            # Loop through each environment in infos
            for n in range(len(infos)):
                # Loop through each agent in the environment
                for m in range(len(infos[n]["agent"])):
                    # Get the agent's index (assuming agent indices correspond to m)
                    agent_idx = m + n * agents_count
                    # Add the policy name to the agent's dictionary if the index exists
                    if agent_idx in agent_idx_to_policy_name:
                        infos[n]["agent"][m]['policy_name'] = agent_idx_to_policy_name[agent_idx]
                    else:
                        infos[n]["agent"][m]['policy_name'] = "No Name Found"
                game_stats.append(infos[n]["agent"])
        #---END NEW---                        
        total_rewards += rewards
        step_count += 1
        episodes += sum([e.done for e in vecenv.envs])

    print(f"Step count: {step_count}")
    print(f"Final episode count: {episodes}")
    #---NEW---
    print_policy_stats(game_stats, '1v1', 'altar')
    print_policy_stats(game_stats, 'elo_1v1', 'altar')
    #---END NEW---
    return game_stats