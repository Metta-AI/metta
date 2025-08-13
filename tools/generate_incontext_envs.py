# %%
import itertools
from itertools import permutations

import numpy as np

converter_types = [
    "mine_red",
    "mine_blue",
    "mine_green",
    "generator_red",
    "generator_blue",
    "generator_green",
    "altar",
    "lab",
    "factory",
    "temple",
]

resource_types = [
    "ore_red",
    "ore_blue",
    "ore_green",
    "battery_red",
    "battery_blue",
    "battery_green",
    "laser",
    "blueprint",
    "armor",
]

env_cfgs = []

# CURRICULUM PARAMS
num_resources = 4
# for now, initial inventory means you are initialized with the last resource that you need  to get the heart
# but we can extend this to allow for initializing with any resource along the chain
maximum_chain_length = 4

assert num_resources >= maximum_chain_length, "num_resources must be >= maximum_chain_length"

env_cfg_template = {"game.map_builder.root.params.objects": {}, "game.objects": {}}
# GENERATE ENV CFGS

# Generate all possible combinations of num_converters converter types
resources = resource_types[:num_resources]
#
# %%
all_resource_permutations = []
for length in range(1, min(len(resources), maximum_chain_length) + 1):
    all_resource_permutations.extend(list(permutations(resources, length)))
# %%


def get_env_cfg(resource_chain, sink_per_level, initial_inventory):
    print(f"Initial inventory: {initial_inventory}")

    resource_chain = ["nothing"] + list(resource_chain) + ["heart"]
    sink_per_level = [0] + list(sink_per_level) + [0]
    chain_length = len(resource_chain)

    print(f"Resource chain: {resource_chain}")
    print(f"Sink per level: {sink_per_level}")

    used_objects = []

    env_cfg_template["game.map_builder.root.params.objects"] = {}
    env_cfg_template["game.objects"] = {}
    env_cfg_template["game.agent.initial_inventory"] = {}

    # for every pair along the way of our resource chain
    for i in range(chain_length - 1):
        # first with no sink
        pair = resource_chain[i], resource_chain[i + 1]
        converter = np.random.choice([c for c in converter_types if c not in used_objects])
        used_objects.append(converter)

        print(f"{converter} will convert {pair[0]} to {pair[1]}")

        env_cfg_template["game.map_builder.root.params.objects"][converter] = 1

        if pair[0] == "nothing":
            env_cfg_template["game.objects"][converter] = {"output_resources": {pair[1]: 1}}
        else:
            env_cfg_template["game.objects"][converter] = {
                "input_resources": {pair[0]: 1},
                "output_resources": {pair[1]: 1},
            }

        if sink_per_level[i] == 1:
            sink = np.random.choice([c for c in converter_types if c not in used_objects])
            used_objects.append(sink)

            print(f"Sink {sink} will sink {pair[0]} at level {i}")

            env_cfg_template["game.map_builder.root.params.objects"][sink] = 1

            env_cfg_template["game.objects"][sink] = {
                "input_resources": {pair[0]: 1},
            }

    if initial_inventory:
        env_cfg_template["game.agent.initial_inventory"][resource_chain[-2]] = 1

    return env_cfg_template


# %%
# %%

env_cfgs = []
for resource_chain in all_resource_permutations:
    chain_length = len(resource_chain)
    sink_per_level_options = list(itertools.product([0, 1], repeat=len(resource_chain)))
    for sink_per_level in sink_per_level_options:
        if sum(sink_per_level) > 2:
            continue
        for initial_inventory in [True]:
            yaml_file_path = (
                f"configs/env/mettagrid/operant_conditioning/in_context_learning/"
                f"chain_length_{chain_length + 1}/{'-'.join(str(x) for x in resource_chain)}"
                f"_{''.join(str(x) for x in sink_per_level)}.yaml"
            )

            f = open(yaml_file_path, "w")
            f.write("defaults:" + "\n")
            f.write("   - /env/mettagrid/operant_conditioning/in_context_learning/defaults@" + "\n")
            f.write("   - _self_game:" + "\n")

            env_cfg = get_env_cfg(resource_chain, sink_per_level, initial_inventory)
            f.write("game:" + "\n")
            f.write("  map_builder:" + "\n")
            f.write("    root:" + "\n")
            f.write("      params:" + "\n")
            f.write("        objects:" + "\n")
            for obj, count in env_cfg["game.map_builder.root.params.objects"].items():
                f.write(f"          {obj}: {count}" + "\n")
            f.write("  objects:" + "\n")
            for obj, resource_dict in env_cfg["game.objects"].items():
                f.write(f"    {obj}:" + "\n")
                for key, value in resource_dict.items():
                    f.write(f"      {key}:" + "\n")
                    for resource, count in value.items():
                        f.write(f"        {resource}: {count}" + "\n")
            f.write("  agent:" + "\n")
            f.write("    initial_inventory:" + "\n")
            for resource in env_cfg["game.agent.initial_inventory"]:
                f.write(f"      {resource}: 1" + "\n")
            f.close()

            # print(f"Environment cfg: {env_cfg}")
            # env_cfgs.append(env_cfg)
            # print("\n")

print(f"Have {len(env_cfgs)} environments total" + "\n")

# %%
