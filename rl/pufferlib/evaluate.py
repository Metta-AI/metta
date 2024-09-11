from omegaconf import OmegaConf
import os
import torch
import time
import numpy as np


def evaluate(cfg: OmegaConf, vecenv):
    num_envs = cfg.eval.num_envs
    device = cfg.framework.pufferlib.device

    run_path = os.path.join(cfg.framework.pufferlib.train_dir, cfg.experiment)
    trainer_state = torch.load(os.path.join(run_path, 'trainer_state.pt'))
    model_path = os.path.join(run_path, trainer_state["model_name"])
    print(f'Loaded model from {model_path}')
    policy = torch.load(model_path, map_location=device)
    opponents = [policy]
    policy_names = [model_path.split('/')[-1], "opponent"]

    # paths = glob.glob(f'{checkpoint_dir}/model_*.pt', recursive=True)
    # names = [path.split('/')[-1] for path in paths]
    # print(f'Loaded {len(paths)} models')
    # paths.remove(f'{checkpoint_dir}/{checkpoint}')
    # print(f'Removed {checkpoint} from paths')
    # elos[checkpoint] = 1000

    # Sample with replacement if not enough models
    # print(f'Sampling {num_opponents} opponents')
    # n_models = len(paths)
    # if n_models < num_opponents:
    #     idxs = random.choices(range(n_models), k=num_opponents)
    # else:
    #     idxs = random.sample(range(n_models), num_opponents)
    # print(f'Sampled {num_opponents} opponents')

    # opponent_names = [names[i] for i in idxs]
    # opponents = [torch.load(paths[i], map_location='cuda') for i in idxs]
    # print(f'Loaded {num_opponents} opponents')
    obs, _ = vecenv.reset()

    num_opponents = len(opponents)
    envs_per_opponent = num_envs // num_opponents
    my_state = None
    opp_states = [None for _ in range(num_opponents)]

    num_agents = cfg.env.game.num_agents
    num_my_agents = max(1, int(cfg.env.game.num_agents * cfg.eval.policy_agents_pct))
    num_opponent_agents = num_agents - num_my_agents
    print(f'Policy Agents: {num_my_agents}, Opponent Agents: {num_opponent_agents}')
    slice_idxs = torch.arange(vecenv.num_agents).reshape(num_envs, num_agents).to(device=device)
    my_idxs = slice_idxs[:, :num_my_agents].reshape(vecenv.num_agents//2)
    opp_idxs = slice_idxs[:, num_my_agents:].reshape(num_envs*num_opponent_agents).split(num_opponent_agents*envs_per_opponent)

    start = time.time()
    episodes = 0
    step = 0
    total_rewards = np.zeros(vecenv.num_agents)
    agent_stats = [{} for a in range(vecenv.num_agents)]
    while episodes < cfg.eval.num_episodes and time.time() - start < cfg.eval.max_time_s:
        step += 1
        opp_actions = []
        with torch.no_grad():
            obs = torch.as_tensor(obs).to(device=device)
            my_obs = obs[my_idxs]

            # Parallelize across opponents
            if hasattr(policy, 'lstm'):
                my_actions, _, _, _, my_state = policy(my_obs, my_state)
            else:
                my_actions, _, _, _ = policy(my_obs)

            # Iterate opponent policies
            for i in range(num_opponents):
                opp_obs = obs[opp_idxs[i]]
                opp_state = opp_states[i]

                opponent = opponents[i]
                if hasattr(policy, 'lstm'):
                    opp_atn, _, _, _, opp_states[i] = opponent(opp_obs, opp_state)
                else:
                    opp_atn, _, _, _ = opponent(opp_obs)

                opp_actions.append(opp_atn)

        opp_actions = torch.cat(opp_actions)
        actions = torch.cat([
            my_actions.view(num_envs, num_my_agents, -1),
            opp_actions.view(num_envs, num_opponent_agents, -1),
        ], dim=1).view(num_envs*num_agents, -1)

        obs, rewards, dones, truncated, infos = vecenv.step(actions.cpu().numpy())
        total_rewards += rewards
        episodes += sum([e.done for e in vecenv.envs])
        agent_infos = []
        for info in infos:
            agent_infos.extend(info["agent_stats"])

        for idx, info in enumerate(agent_infos):
            for k, v in info.items():
                if k not in agent_stats[idx]:
                    agent_stats[idx][k] = 0
                agent_stats[idx][k] += v

        # for i in range(num_envs):
        #     c = envs.c_envs[i]
        #     opp_idx = i // envs_per_opponent
        #     if c.radiant_victories > prev_radiant_victories[i]:
        #         prev_radiant_victories[i] = c.radiant_victories
        #         scores.append((opp_idx, 1))
        #         games_played += 1
        #         print('Radiant Victory')
        #     elif c.dire_victories > prev_dire_victories[i]:
        #         prev_dire_victories[i] = c.dire_victories
        #         scores.append((opp_idx, 0))
        #         games_played += 1
        #         print('Dire Victory')

    policy_stats = [{} for a in range(len(policy_names))]
    policy_idxs = [my_idxs] + list(opp_idxs)
    for policy_idx, stats in enumerate(policy_stats):
        num_policy_agents = len(policy_idxs[policy_idx])
        for agent_idx in policy_idxs[policy_idx]:
            for k, v in agent_stats[agent_idx.item()].items():
                if k not in stats:
                    stats[k] = {"sum": 0, "count": num_policy_agents}
                stats[k]["sum"] += v

    return policy_stats
