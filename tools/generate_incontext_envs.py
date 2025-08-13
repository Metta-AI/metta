# %%
import itertools
import os

import numpy as np
import yaml

CONVERTER_TYPES = [
    "mine_red",
    "mine_blue",
    "mine_green",
    "generator_red",
    "generator_blue",
    "generator_green",
    "altar",
    "lab",
    # leave out some objects for evals
    # "factory",
    # "temple",
    # lasery
]

RESOURCE_TYPES = [
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

DEFAULT_ENV_CFG = {
    "defaults": [
        "/env/mettagrid/operant_conditioning/in_context_learning/defaults@",
        "_self_:",
    ],
    "game": {
        "map_builder": {
            "root": {
                "params": {
                    "objects": {},
                }
            }
        },
        "objects": {},
    },
}


class InContextEnv:
    def __init__(self, resource_types, converter_types, resource_chain, num_sinks: int):
        self.env_cfg = DEFAULT_ENV_CFG.copy()
        self.resource_types = resource_types
        self.converter_types = converter_types
        self.resource_chain = resource_chain
        self.num_sinks = num_sinks
        self.used_objects = []
        self.all_input_resources = []

    def set_converter(self, input_resource, output_resource):
        """
        Get a converter, add it to the environment config, and return the converter, input, and output
        """
        converter = np.random.choice([c for c in self.converter_types if c not in self.used_objects])
        self.used_objects.append(converter)

        self.env_cfg["game"]["map_builder"]["root"]["params"]["objects"][converter] = 1

        if input_resource == "nothing":
            self.env_cfg["game"]["objects"][converter] = {"output_resources": {output_resource: 1}}
        else:
            self.env_cfg["game"]["objects"][converter] = {
                "input_resources": {input_resource: 1},
                "output_resources": {output_resource: 1},
            }
            self.all_input_resources.append(input_resource)

    def set_sink(self):
        """
        Get a sink, add it to the environment config, and return the sink, input, and output
        """
        sink = np.random.choice([c for c in self.converter_types if c not in self.used_objects])
        self.used_objects.append(sink)

        self.env_cfg["game"]["map_builder"]["root"]["params"]["objects"][sink] = 1

        for input_resource in self.all_input_resources:
            self.env_cfg["game"]["objects"][sink] = {
                "input_resources": {input_resource: 1},
            }

    def get_env_cfg(self):
        # first resource is always nothing, last resource is always heart
        resource_chain = ["nothing"] + list(self.resource_chain) + ["heart"]

        chain_length = len(resource_chain)

        # for every i/o pair along the way of our resource chain
        for i in range(chain_length - 1):
            input_resource, output_resource = resource_chain[i], resource_chain[i + 1]

            self.set_converter(input_resource, output_resource)

        self.set_sink()

        return self.env_cfg


class InContextEnvGenerator:
    def __init__(self, maximum_chain_length: int, maximum_num_sinks: int):
        self.maximum_chain_length = maximum_chain_length
        self.maximum_num_sinks = maximum_num_sinks

        self.resource_types = RESOURCE_TYPES[: self.maximum_chain_length]
        self.converter_types = CONVERTER_TYPES[: self.maximum_chain_length]

        all_resource_permutations = []
        for length in range(1, min(len(self.resource_types), self.maximum_chain_length) + 1):
            all_resource_permutations.extend(list(itertools.permutations(self.resource_types, length)))

        self.all_resource_permutations = all_resource_permutations

    def generate_yaml_cfgs(self):
        num_envs = 0
        for resource_chain in self.all_resource_permutations:
            chain_length = len(resource_chain)
            for num_sinks in range(self.maximum_num_sinks):
                env = InContextEnv(self.resource_types, self.converter_types, resource_chain, num_sinks)
                env_cfg = env.get_env_cfg()

                yaml_file_path = (
                    f"configs/env/mettagrid/operant_conditioning/in_context_learning/"
                    f"chain_length_{chain_length}/{num_sinks}/{'-'.join(str(x) for x in resource_chain)}.yaml"
                )

                os.makedirs(yaml_file_path, exist_ok=True)

                f = open(yaml_file_path, "w")

                yaml.dump(env_cfg, f)

                num_envs += 1

        print(f"Generated {num_envs} envs")


if __name__ == "__main__":
    generator = InContextEnvGenerator(maximum_chain_length=4, maximum_num_sinks=2)
    generator.generate_yaml_cfgs()

# %%
