# Generate a graphical trace of multiple runs.

import os
import hydra
import json
from omegaconf import OmegaConf
from rich import traceback
import torch
import numpy as np
from rl.pufferlib.vecenv import make_vecenv
from agent.policy_store import PolicyRecord
from rl.wandb.wandb_context import WandbContext
from util.seeding import seed_everything
from agent.policy_store import PolicyStore

import signal  # Aggressively exit on ctrl+c
signal.signal(signal.SIGINT, lambda sig, frame: os._exit(0))


def nice_orientation(orientation):
    """ Convert an orientation into a human-readable string """
    """ Convert an orientation into a human-readable string """
    return ["north", "south", "west", "east"][orientation]


def nice_actions(action):
    """ Convert a un-flattened action into a human-readable string """
    if action[0] == 0:
        return "noop"
    elif action[0] == 1:
        if action[1] == 0:
            return "move_back"
        elif action[1] == 1:
            return "move_forward"
    elif action[0] == 2:
        return "rotate_" + nice_orientation(action[1])
    elif action[0] == 3:
        return "use"
    elif action[0] == 4:
        return "attack_" + str(action[1] // 3) + "_" + str(action[1] % 3)
    elif action[0] == 6:
        return "shield"
    elif action[0] == 7:
        return "gift"
    elif action[0] == 8:
        return "swap"
    else:
        return "unknown"


def trace(cfg: OmegaConf, policy_record: PolicyRecord):
    """ Trace a policy and generate a jsonl file """
    device = cfg.device
    vecenv = make_vecenv(cfg, num_envs=1, render_mode="human")
    obs, _ = vecenv.reset()
    env = vecenv.envs[0]
    policy = policy_record.policy()
    policy_rnn_state = None
    rewards = np.zeros(vecenv.num_agents)
    total_rewards = np.zeros(vecenv.num_agents)

    output = open("output_1.jsonl", "w")

    while True:
        output.write(json.dumps({"step": env._c_env.current_timestep()}) + "\n")

        with torch.no_grad():
            obs = torch.as_tensor(obs).to(device=device)
            actions, _, _, _, policy_rnn_state = policy(obs, policy_rnn_state)

        actions_array = env._c_env.unflatten_actions(actions.cpu().numpy())
        for id, action in enumerate(actions_array):
            for grid_object in env.grid_objects.values():
                if "agent_id" in grid_object and grid_object["agent_id"] == id:
                    agent = grid_object
                    break
            output.write(json.dumps({
                "agent": id,
                "action": action.tolist(),
                "action_name": nice_actions(action),
                "reward": rewards[id].item(),
                "total_reward": total_rewards[id].item(),
                "position": [agent["c"], agent["r"]],
                "energy": agent["agent:energy"],
                "hp": agent["agent:hp"],
                "frozen": agent["agent:frozen"],
                "orientation": nice_orientation(agent["agent:orientation"]),
                "shield": agent["agent:shield"],
                "inventory": agent["agent:inv:r1"]
            }) + "\n")

        obs, rewards, dones, trunc, infos = vecenv.step(actions.cpu().numpy())
        total_rewards += rewards
        if any(dones) or any(trunc):
            break

    output.close()


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg):

    traceback.install(show_locals=False)

    print(OmegaConf.to_yaml(cfg))

    seed_everything(cfg.seed, cfg.torch_deterministic)
    os.makedirs(cfg.run_dir, exist_ok=True)
    with open(os.path.join(cfg.run_dir, "config.yaml"), "w") as f:
        OmegaConf.save(cfg, f)

    with WandbContext(cfg) as wandb_run:
        policy_store = PolicyStore(cfg, wandb_run)
        policy = policy_store.policy(cfg.evaluator.policy)
        trace(cfg, policy)


if __name__ == "__main__":
    main()
