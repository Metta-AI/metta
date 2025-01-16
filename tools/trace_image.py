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
import pixie

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

WHITE = pixie.Color(1, 1, 1, 1)
BLACK = pixie.Color(0, 0, 0, 1)
# Using flat UI colors:
TURQUOISE = pixie.Color(0x1a/255, 0xbc/255, 0x9c/255, 1)
GREEN_SEA = pixie.Color(0x16/255, 0xa0/255, 0x85/255, 1)
EMERALD = pixie.Color(0x2e/255, 0xcc/255, 0x71/255, 1)
NEPHRITIS = pixie.Color(0x27/255, 0xae/255, 0x60/255, 1)
PETER_RIVER = pixie.Color(0x34/255, 0x98/255, 0xdb/255, 1)
BELIZE_HOLE = pixie.Color(0x29/255, 0x80/255, 0xb9/255, 1)
AMETHYST = pixie.Color(0x9b/255, 0x59/255, 0xb6/255, 1)
WISTERIA = pixie.Color(0x8e/255, 0x44/255, 0xad/255, 1)
WET_ASPHALT = pixie.Color(0x34/255, 0x49/255, 0x5e/255, 1)
MIDNIGHT_BLUE = pixie.Color(0x2c/255, 0x3e/255, 0x50/255, 1)
SUN_FLOWER = pixie.Color(0xf1/255, 0xc4/255, 0x0f/255, 1)
ORANGE = pixie.Color(0xf3/255, 0x9c/255, 0x12/255, 1)
CARROT = pixie.Color(0xe6/255, 0x7e/255, 0x22/255, 1)
PUMPKIN = pixie.Color(0xd3/255, 0x54/255, 0x00/255, 1)
ALIZARIN = pixie.Color(0xe7/255, 0x4c/255, 0x3c/255, 1)
POMEGRANATE = pixie.Color(0xc0/255, 0x39/255, 0x2b/255, 1)
CLOUDS = pixie.Color(0xec/255, 0xf0/255, 0xf1/255, 1)
SILVER = pixie.Color(0xbd/255, 0xc3/255, 0xc7/255, 1)
CONCRETE = pixie.Color(0x95/255, 0xa5/255, 0xa6/255, 1)
ASBESTOS = pixie.Color(0x7f/255, 0x8c/255, 0x8d/255, 1)

BG_COLOR = MIDNIGHT_BLUE
NOOP_COLOR = SILVER
MOVE_COLOR = EMERALD
ROTATE_COLOR = NEPHRITIS
USE_COLOR = ORANGE
ATTACK_COLOR = ALIZARIN
SHIELD_COLOR = PETER_RIVER
GIFT_COLOR = PUMPKIN
SWAP_COLOR = POMEGRANATE

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
    num_agents = vecenv.num_agents
    num_steps = 500

    #output = open("output_1.jsonl", "w")
    image = pixie.Image(52 + num_steps*2 + 50, 10 + 60 * num_agents + 10)
    image.fill(pixie.Color(44/255, 62/255, 80/255, 1))
    ctx = image.new_context()

    steps = []

    while True:
        #output.write(json.dumps({"step": env._c_env.current_timestep()}) + "\n")

        with torch.no_grad():
            obs = torch.as_tensor(obs).to(device=device)
            actions, _, _, _, policy_rnn_state = policy(obs, policy_rnn_state)

        actions_array = env._c_env.unflatten_actions(actions.cpu().numpy())
        agents = []
        for id, action in enumerate(actions_array):
            for grid_object in env.grid_objects.values():
                if "agent_id" in grid_object and grid_object["agent_id"] == id:
                    agent = grid_object
                    break
            agents.append({
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
            })
        steps.append(agents)

        obs, rewards, dones, trunc, infos = vecenv.step(actions.cpu().numpy())
        total_rewards += rewards
        if any(dones) or any(trunc):
            break

    font = pixie.read_font("deps/mettagrid/mettagrid/renderer/assets/Inter-Regular.ttf")
    font.size = 20
    font.paint.color = WHITE

    for id in range(vecenv.num_agents):
        # Draw the agent ID
        image.fill_text(
            font,
            f"{id}",
            bounds = pixie.Vector2(100, 100),
            transform = pixie.translate(10, 10 + 60 * id)
        )

        # Draw the start bar
        paint = pixie.Paint(pixie.SOLID_PAINT)
        paint.color = WHITE
        ctx.fill_style = paint
        ctx.fill_rect(40, 10 + 60 * id, 2, 50)

        # Draw the actions:
        for step in range(len(steps)):
            agent = steps[step][id]
            change = True
            if step > 0:
                change = steps[step][id] != steps[step-1][id]
            x = 40 + step * 2
            y = 10 + 60 * id + 29



            paint.color = pixie.Color(1, 1, 1, 0.05)
            yo = 20 * agent["energy"]/256
            ctx.fill_rect(x, y - yo, 2, yo*2+2)

            if agent["shield"]:
                paint.color = SHIELD_COLOR
                ctx.fill_rect(x, y - 16, 2, 16*2+2)

            if agent["frozen"]:
                pass
            elif not change:
                paint.color = CONCRETE
                ctx.fill_rect(x, y, 2, 2)
            elif agent["action_name"] == "noop":
                paint.color = NOOP_COLOR
                ctx.fill_rect(x, y, 2, 2)
            elif agent["action_name"] == "move_forward":
                paint.color = MOVE_COLOR
                ctx.fill_rect(x, y, 2, 2)
            elif agent["action_name"] == "move_back":
                paint.color = MOVE_COLOR
                ctx.fill_rect(x, y, 2, 2)
            elif "rotate" in agent["action_name"]:
                paint.color = ROTATE_COLOR
                ctx.fill_rect(x, y, 2, 2)
            elif agent["action_name"] == "use":
                paint.color = USE_COLOR
                ctx.fill_rect(x, y-4, 2, 4*2+2)
            elif "attack" in agent["action_name"]:
                paint.color = ATTACK_COLOR
                ctx.fill_rect(x, y-6, 2, 6*2+2)
            elif agent["action_name"] == "shield":
                paint.color = SHIELD_COLOR
                ctx.fill_rect(x, y, 2, 2)
                ctx.fill_rect(x, y-2, 2, 2*2+2)
            elif agent["action_name"] == "gift":
                paint.color = GIFT_COLOR
                ctx.fill_rect(x, y, 2, 2)
            elif agent["action_name"] == "swap":
                paint.color = SWAP_COLOR
                ctx.fill_rect(x, y, 2, 2)

            if agent["reward"] > 0:
                paint.color = AMETHYST
                ctx.fill_rect(x, y - 16, 2, 6)

    #output.close()
    image.write_file("output.png")


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
