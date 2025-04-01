# Generate a graphical trace of multiple runs.

import os
import json
import zlib
import torch
import pixie
from omegaconf import OmegaConf
from agent.policy_store import PolicyRecord
from rl.pufferlib.simulator import Simulator


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
WHITE = pixie.Color(1, 1, 1, 1)
BLACK = pixie.Color(0, 0, 0, 1)


# Action Colors:
BG_COLOR = MIDNIGHT_BLUE
NOOP_COLOR = SILVER
MOVE_COLOR = EMERALD
ROTATE_COLOR = NEPHRITIS
PUT_COLOR = ORANGE
GET_COLOR = CARROT
ATTACK_COLOR = ALIZARIN
CHANGE_COLOR_COLOR = PETER_RIVER
SWAP_COLOR = POMEGRANATE


def nice_orientation(orientation):
    """ Convert an orientation into a human-readable string """
    return ["north", "south", "west", "east"][orientation % 4]


def nice_actions(env, action):
    """ Convert a un-flattened action into a human-readable string """
    name = env.action_names()[action[0]]
    if name == "move":
        return name + ("_back", "_forward")[action[1] % 2]
    elif name == "rotate":
        return "rotate_" + nice_orientation(action[1])
    elif name == "attack":
        return "attack_" + str(action[1] // 3) + "_" + str(action[1] % 3)
    else:
        return name


def save_trace_image(
    cfg: OmegaConf,
    policy_record: PolicyRecord,
    output_path: str
):
    """ Trace a policy and generate a jsonl file """

    simulator = Simulator(cfg, policy_record)

    steps = []
    actions_names = simulator.env.action_names()
    while not simulator.done():
        actions = simulator.actions()

        actions_array = actions.cpu().numpy()
        step_info = []
        for id, action in enumerate(actions_array):
            for grid_object in simulator.grid_objects():
                if "agent_id" in grid_object and grid_object["agent_id"] == id:
                    agent = grid_object
                    break

            step_info.append({
                "agent": id,
                "action": action.tolist(),
                "action_name": actions_names[action[0]],
                "action_nice_name": nice_actions(simulator.env, action),
                "reward": simulator.rewards[id].item(),
                "total_reward": simulator.total_rewards[id].item(),
                "position": [agent["c"], agent["r"]],
                "frozen": agent["agent:frozen"],
                "orientation": nice_orientation(agent["agent:orientation"]),
            })

        simulator.step(actions)

        for i in range(len(simulator.env.action_success)):
            step_info[i]["action_success"] = simulator.env.action_success[i]

        steps.append(step_info)

    image = pixie.Image(52 + simulator.num_steps*2 + 50, 10 + 60 * simulator.num_agents + 10)
    image.fill(pixie.Color(44/255, 62/255, 80/255, 1))
    ctx = image.new_context()

    font = pixie.read_font("deps/mettagrid/mettagrid/renderer/assets/Inter-Regular.ttf")
    font.size = 20
    font.paint.color = WHITE

    for id in range(simulator.num_agents):
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

            x = 40 + step * 2
            y = 10 + 60 * id + 29

            if agent["frozen"]:
                pass
            elif not agent["action_success"]:
                paint.color = CONCRETE
                ctx.fill_rect(x, y, 2, 2)
            elif agent["action_name"] == "noop":
                paint.color = NOOP_COLOR
                ctx.fill_rect(x, y, 2, 2)
            elif agent["action_name"] == "move":
                paint.color = MOVE_COLOR
                ctx.fill_rect(x, y, 2, 2)
            elif agent["action_name"] == "rotate":
                paint.color = ROTATE_COLOR
                ctx.fill_rect(x, y, 2, 2)
            elif agent["action_name"] == "put_items":
                paint.color = PUT_COLOR
                ctx.fill_rect(x, y-4, 2, 4*2+2)
            elif agent["action_name"] == "get_items":
                paint.color = GET_COLOR
                ctx.fill_rect(x, y-4, 2, 4*2+2)
            elif "attack" in agent["action_name"]:
                paint.color = ATTACK_COLOR
                ctx.fill_rect(x, y-6, 2, 6*2+2)
            elif agent["action_name"] == "swap":
                paint.color = SWAP_COLOR
                ctx.fill_rect(x, y, 2, 2)
            elif agent["action_name"] == "change_color":
                paint.color = CHANGE_COLOR_COLOR
                ctx.fill_rect(x, y, 2, 2)
            elif agent["action_name"] == "put_recipe_items":
                paint.color = PUT_COLOR
                ctx.fill_rect(x, y, 2, 2)
            elif agent["action_name"] == "get_output":
                paint.color = GET_COLOR
                ctx.fill_rect(x, y, 2, 2)
            else:
                print("Unknown action:", agent["action_name"])

            if agent["reward"] > 0:
                paint.color = AMETHYST
                ctx.fill_rect(x-1, y - 16, 1, 16*2+2)

    # Make sure the directory exists:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    image.write_file(output_path)


def add_sequence_key(grid_object, key, step, value):
    """ Add a key to the replay that is a sequence of values. """
    if key not in grid_object:
        # Add new key.
        grid_object[key] = [[step, value]]
    else:
        # Only add new entry if it has changed:
        if grid_object[key][-1][1] != value:
            grid_object[key].append([step, value])


def save_replay(
    cfg: OmegaConf,
    policy_record: PolicyRecord,
    output_path: str
):
    """ Trace a policy and generate a replay file """

    simulator = Simulator(cfg, policy_record)

    grid_objects = []

    replay = {
        "version": 1,
        "action_names": simulator.env.action_names(),
        "object_types": [],
        "map_size": [simulator.env.map_width, simulator.env.map_height],
        "num_agents": simulator.num_agents,
        "max_steps": simulator.num_steps,
        "grid_objects": grid_objects,
    }

    replay["object_types"] = simulator.env.object_type_names()

    step = 0
    while not simulator.done():
        actions = simulator.actions()

        actions_array = actions.cpu().numpy()
        step_info = []

        for i, grid_object in enumerate(simulator.grid_objects()):
            if len(grid_objects) <= i:
                # Add new grid object.
                grid_objects.append({})
            for key, value in grid_object.items():
                add_sequence_key(grid_objects[i], key, step, value)

            if "agent_id" in grid_object:
                agent_id = grid_object["agent_id"]
                add_sequence_key(grid_objects[i], "action", step, actions_array[agent_id].tolist())
                add_sequence_key(grid_objects[i], "action_success", step, bool(simulator.env.action_success[agent_id]))
                add_sequence_key(grid_objects[i], "reward", step, simulator.rewards[agent_id].item())
                add_sequence_key(grid_objects[i], "total_reward", step, simulator.total_rewards[agent_id].item())

        simulator.step(actions)

        step += 1

    replay["max_steps"] = step

    # Trim value changes to make them more compact.
    for grid_object in grid_objects:
        for key, changes in grid_object.items():
            if len(changes) == 1:
                grid_object[key] = changes[0][1]

    # Make sure the directory exists:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        f.write(json.dumps(replay))

    # Compress it with deflate.
    replay_data = json.dumps(replay)  # Convert to JSON string
    replay_bytes = replay_data.encode('utf-8')  # Encode to bytes
    compressed_data = zlib.compress(replay_bytes)  # Compress the bytes
    # Write the compressed data to a file
    with open(output_path + '.z', 'wb') as f:
        f.write(compressed_data)
