from typing import Dict, List
import numpy as np
import yaml
from omegaconf import OmegaConf

class NoAliasDumper(yaml.Dumper):
    def ignore_aliases(self, data):
        return True

class MettaGridGameBuilder():
    def __init__(
            self,
            obs_width: int,
            obs_height: int,
            tile_size: int,
            max_steps: int,
            num_agents: int,
            no_energy_steps: int,
            objects,
            actions,
            map):

        self.obs_width = obs_width
        self.obs_height = obs_height
        self.tile_size = tile_size
        self.num_agents = num_agents
        self.max_steps = max_steps

        self._symbols = {
            "agent": "A",
            "altar": "a",
            "converter": "c",
            "generator": "g",
            "wall": "w",
            "empty": " ",
        }

        self.no_energy_steps = no_energy_steps
        objects = OmegaConf.create(objects)
        self.object_configs = objects
        actions = OmegaConf.create(actions)
        self.action_configs = actions
        self.map_config = OmegaConf.create(map)


    def level(self):
        layout = self.map_config.layout

        if "rooms" in layout:
            return self.build_map(layout.rooms)
        else:
            return self.build_map(
                [["room"] * layout.rooms_x] * layout.rooms_y)

    def build_map(self, rooms):
        num_agents = 0
        layers = []
        for layer in rooms:
            rooms = []
            for room_name in layer:
                room_config = self.map_config[room_name]
                rooms.append(self.build_room(room_config, num_agents + 1))
                num_agents += room_config.objects.agent
            layers.append(np.concatenate(rooms, axis=1))
        level = np.concatenate(layers, axis=0)
        assert num_agents == self.num_agents, f"Number of agents in map ({num_agents}) does not match num_agents ({self.num_agents})"

        footer = np.full((1, level.shape[1]), "W", dtype="U6")
        footer[0, 0] = "q"

        level = np.concatenate([level, footer], axis=0)
        return level

    def build_room(self, room_config, starting_agent=1):
        symbols = []
        content_width = room_config.width - 2*room_config.border
        content_height = room_config.height - 2*room_config.border
        area = content_width * content_height

        for obj_name, count in room_config.objects.items():
            symbol = self._symbols[obj_name]
            if obj_name == "agent":
                symbols.extend([f"{symbol}{i+starting_agent}" for i in range(count)])
            else:
                symbols.extend([symbol] * count)

        assert(len(symbols) <= area), f"Too many objects in room: {len(symbols)} > {area}"
        symbols.extend(["."] * (area - len(symbols)))
        symbols = np.array(symbols).astype("U8")
        np.random.shuffle(symbols)
        content = symbols.reshape(content_height, content_width)
        room = np.full((room_config.height, room_config.width), "W", dtype="U6")
        room[room_config.border:room_config.border+content_height,
             room_config.border:room_config.border+content_width] = content

        return room

    def termination_conditions(self):
        return {
            "Win": [ {"lt": ["game:max_steps", "game:step"]} ],
        }
