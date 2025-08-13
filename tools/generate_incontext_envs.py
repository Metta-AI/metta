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
        "_self_",
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


def to_builtin(value):
    """Recursively convert numpy scalars/arrays and keys to plain Python types."""
    if isinstance(value, dict):
        return {str(to_builtin(k)): to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_builtin(v) for v in value]
    if isinstance(value, np.generic):  # numpy scalar types (including numpy.str_)
        try:
            return value.item()
        except Exception:
            return str(value)
    return value


class IndentDumper(yaml.SafeDumper):
    def increase_indent(self, flow=False, indentless=False):
        return super().increase_indent(flow, False)


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
        converter = str(np.random.choice([c for c in self.converter_types if c not in self.used_objects]))
        print(f"Converter: {converter} will convert {input_resource} to {output_resource}")
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
        sink = str(np.random.choice([c for c in self.converter_types if c not in self.used_objects]))
        self.used_objects.append(sink)

        self.env_cfg["game"]["map_builder"]["root"]["params"]["objects"][sink] = 1

        for input_resource in self.all_input_resources:
            self.env_cfg["game"]["objects"][sink] = {
                "input_resources": {input_resource: 1},
            }
        print(f"Sink: {sink} will convert {self.all_input_resources} to {sink}")

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

                yaml_file_dir = (
                    f"configs/env/mettagrid/operant_conditioning/in_context_learning/"
                    f"chain_length_{chain_length}/{num_sinks}_sinks"
                )

                os.makedirs(yaml_file_dir, exist_ok=True)

                yaml_file_path = os.path.join(yaml_file_dir, f"{'-'.join(str(x) for x in resource_chain)}.yaml")

                # Sanitize to pure Python types and dump safely
                clean_cfg = to_builtin(env_cfg)
                with open(yaml_file_path, "w") as f:
                    yaml.safe_dump(
                        clean_cfg,
                        f,
                        sort_keys=False,
                        allow_unicode=True,
                        default_flow_style=False,
                        Dumper=IndentDumper,
                        indent=2,
                    )
                num_envs += 1

        print(f"Generated {num_envs} envs")


if __name__ == "__main__":
    generator = InContextEnvGenerator(maximum_chain_length=4, maximum_num_sinks=2)
    generator.generate_yaml_cfgs()

# %%
