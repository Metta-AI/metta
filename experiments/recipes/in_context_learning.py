import numpy as np
from metta.mettagrid.config import building

CONVERTER_TYPES = {
    "mine_red": building.mine_red,
    "mine_blue": building.mine_blue,
    "mine_green": building.mine_green,
    "generator_red": building.generator_red,
    "generator_blue": building.generator_blue,
    "generator_green": building.generator_green,
    "altar": building.altar,
    "lab": building.lab,
    "factory": building.factory,
    "temple": building.temple,
    "lasery": building.lasery,
}

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


class InContextResourceChain:
    def __init__(self, resource_chain, num_sinks):
        self.resource_types = RESOURCE_TYPES.copy()
        self.converter_types = CONVERTER_TYPES.copy()
        self.num_sinks = num_sinks
        self.resource_chain = resource_chain
        self.converters = []
        self.used_objects = []
        self.all_input_resources = []

        self.map_builder_objects = {}

        self.game_objects = {}

        self.max_steps = 256

    def set_converter(self, input_resource, output_resource):
        converter_name = str(
            np.random.choice(
                [c for c in self.converter_types if c not in self.used_objects]
            )
        )
        self.used_objects.append(converter_name)
        self.converters.append(converter_name)

        converter = self.converter_types[converter_name]
        converter.output_resources = {output_resource: 1}

        if input_resource != "nothing":
            converter.input_resources = {input_resource: 1}

            self.all_input_resources.append(input_resource)

        self.game_objects[converter_name] = converter
        self.map_builder_objects[converter_name] = 1

    def set_sink(self):
        sink_name = str(
            np.random.choice(
                [c for c in self.converter_types if c not in self.used_objects]
            )
        )
        self.used_objects.append(sink_name)
        sink = self.converter_types[sink_name]

        for input_resource in self.all_input_resources:
            sink.input_resources[input_resource] = 1

        self.game_objects[sink_name] = sink
        self.map_builder_objects[sink_name] = 1

    def make_env(self):
        self.used_objects = []
        self.all_input_resources = []

        resource_chain = ["nothing"] + list(self.resource_chain) + ["heart"]

        chain_length = len(resource_chain)

        for i in range(chain_length - 1):
            input_resource, output_resource = resource_chain[i], resource_chain[i + 1]
            self.set_converter(input_resource, output_resource)

        for _ in range(self.num_sinks):
            self.set_sink()

        # longer episodes for longer chains
        if len(self.used_objects) > 4:
            self.max_steps = 512

        cooldown = 6 * (chain_length - 1)

        for obj in self.converters:
            self.game_objects[obj].cooldown = cooldown

        return self.game_objects, self.map_builder_objects


# class InContextTaskGenerator(SingleTaskGenerator):
#     def __init__(self, maximum_chain_length: int = 5, maximum_num_sinks: int = 2):
#         self.

#     def generate_task(self, task_id: int, rng: random.Random) -> EnvConfig:

#         return InContextResourceChain(
#             resource_chain=["ore_red", "ore_blue"], num_sinks=0
#         ).make_env()
