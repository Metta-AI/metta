#!/usr/bin/env -S uv run

from metta.interface.environment import _get_default_env_config
from omegaconf import OmegaConf
from tools.renderer import get_policy, setup_environment

# Define a simple hallway map with more ore
hallway_map = """###########
#@m.......#
###########"""

# Build full environment config using interface helper
env_cfg = _get_default_env_config(num_agents=1, width=11, height=3)
# Override map builder for inline ASCII
env_cfg["game"]["map_builder"] = {
    "_target_": "metta.map.mapgen.MapGen",
    "border_width": 0,
    "root": {
        "type": "metta.map.scenes.inline_ascii.InlineAscii",
        "params": {"data": hallway_map},
    },
}
# Start with a single ore and regenerate every 4 ticks
env_cfg["game"]["objects"]["mine_red"]["initial_resource_count"] = 1
env_cfg["game"]["objects"]["mine_red"]["conversion_ticks"] = (
    4  # Time to produce new ore
)
env_cfg["game"]["objects"]["mine_red"]["cooldown"] = 0

# Add explicit reward for ore collection
env_cfg["game"]["agent"]["rewards"]["inventory"]["ore_red"] = 1.0

# Debug: Print the final mine configuration
print("Final mine_red configuration:")
print(env_cfg["game"]["objects"]["mine_red"])
print()

# Full config for setup_environment
config = OmegaConf.create(
    {
        "renderer_job": {
            "policy_type": "opportunistic",
            "num_steps": 20,
            "num_agents": 1,
            "max_steps": 20,
            "sleep_time": 0.0,  # No delay for faster simulation
        },
        "env": env_cfg,
    }
)

env, _ = setup_environment(config)
policy = get_policy("opportunistic", env, config)

# Simulation
total_reward = 0.0
obs, info = env.reset()
for step in range(config.renderer_job.num_steps):
    actions = policy.predict(obs)
    obs, rewards, terminals, truncations, info = env.step(actions)
    # Update score to reflect current agent inventory of ore
    # Inspect grid objects to find agent's inventory (info dict does not include inventory)
    grid_objs = env.grid_objects
    # Find primary agent (agent_id == 0)
    agent_obj = next(o for o in grid_objs.values() if o.get("agent_id") == 0)
    raw_inv = agent_obj.get("inventory", {})  # maps inventory index -> count
    # Map indices to names
    inv_names = env.inventory_item_names
    inventory = {
        inv_names[idx]: count for idx, count in raw_inv.items() if idx < len(inv_names)
    }
    total_reward = inventory.get("ore_red", 0)

    # Clear screen and show real-time info
    # os.system("clear" if os.name == "posix" else "cls")  # Comment out screen clearing
    # Show score, current ore inventory, and step
    print(
        f"Score: {total_reward:.1f} | Inventory: {inventory.get('ore_red', 0)} ore(s) | Step: {step + 1}/{config.renderer_job.num_steps}"
    )

    # Show mine's inventory
    # mine_inventory = {}
    # if env._c_env_instance is not None:
    #     grid_objects = env._c_env_instance.grid_objects()
    #     print(f"All grid objects: {grid_objects}")
    #     for obj_id, obj_data in grid_objects.items():
    #         if obj_data.get("type_id") == 2:  # mine_red has type_id 2
    #             mine_inventory = obj_data.get("inventory", {})
    #             print(f"Found mine_red object: {obj_data}")
    #             break
    # print(f"Mine inventory: {mine_inventory}")

    print()

    rendered = env.render()
    if rendered:
        print(rendered)

    # No sleep for faster simulation
    # time.sleep(config.renderer_job.sleep_time)

env.close()
print(f"\nFinal total reward: {total_reward}")
