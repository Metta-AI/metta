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
            map,
            kinship):

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
            "wall": "W",
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

        if self.map_config.uri:
            return self.build_map_from_ascii(self.map_config.uri)
        
        elif self.map_config.rooms: #target is a list of room classes and their params?
            return self.build_map_from_target(self.map_config.rooms)

        elif "rooms" in layout:
            return self.build_map(layout.rooms)
        else:
            return self.build_map(
                [["room"] * layout.rooms_x] * layout.rooms_y)
        
    def build_map_from_target(self, room_configs):

        rooms = []

        max_height, max_width = 0, 0

        for room in room_configs:
            room_array = room.build_room()
            max_height = max(max_height, room_array.shape[0])
            max_width = max(max_width, room_array.shape[1])
            rooms.append(room_array)

        # Determine grid dimensions based on number of rooms
        n_rooms = len(rooms)

        grid_rows = int(np.ceil(np.sqrt(n_rooms)))
        grid_cols = int(np.ceil(n_rooms / grid_rows))

        # Create empty grid to hold all rooms
        level = np.full((grid_rows * max_height, grid_cols * max_width), 
                      self._symbols["wall"], dtype="U6")
        
        # Place rooms into grid
        for i in range(n_rooms):
            row = i // grid_cols
            col = i % grid_cols
            room_height = rooms[i].shape[0]
            room_width = rooms[i].shape[1]
            
            # Calculate starting position to center the room in its grid cell
            start_row = row * max_height + (max_height - room_height) // 2
            start_col = col * max_width + (max_width - room_width) // 2
            
            # Place room in centered position
            level[start_row:start_row + room_height,
                 start_col:start_col + room_width] = rooms[i]
            
        b = self.map_config.border
        h, w = level.shape
        final_level = np.full((h + b * 2, w + b * 2), self._symbols["wall"], dtype="U6")
        final_level[b:b + h, b:b + w] = level

        return final_level
        
    def build_map_from_ascii(self, ascii_map_uri):
        """Currently this is only for a single room"""


        with open(ascii_map_uri, "r") as f:
            ascii_map = f.read()
        # Convert ASCII map string to numpy array
        lines = ascii_map.strip().splitlines()
        level = np.array([list(line) for line in lines])

        return level

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

        # Add map border around the level.
        b = self.map_config.border
        h, w = level.shape
        final_level = np.full((h + b * 2, w + b * 2), self._symbols["wall"], dtype="U6")
        final_level[b:b + h, b:b + w] = level

        return final_level

    def build_room(self, room_config, starting_agent=1):
        symbols = []
        content_width = room_config.width - 2*room_config.border
        content_height = room_config.height - 2*room_config.border
        area = content_width * content_height

        # Check if total objects exceed room size and halve counts if needed
        total_objects = sum(count for count in room_config.objects.values())
        while total_objects > 2*area / 3:
            for obj_name in room_config.objects:
                if obj_name != "agent":
                    room_config.objects[obj_name] = max(1, room_config.objects[obj_name] // 2)
                total_objects = sum(count for count in room_config.objects.values())

        # Add all objects in the proper amounts to a single large array.
        for obj_name, count in room_config.objects.items():
            symbol = self._symbols[obj_name]
            if obj_name == "agent":
                symbols.extend([f"{symbol}{i+starting_agent}" for i in range(count)])
            else:
                symbols.extend([symbol] * count)
        assert(len(symbols) <= area), f"Too many objects in room: {len(symbols)} > {area}"
        symbols.extend(["."] * (area - len(symbols)))

        # Shuffle and reshape the array into a room.
        symbols = np.array(symbols).astype("U8")
        np.random.shuffle(symbols)
        content = symbols.reshape(content_height, content_width)

        # Add room border.
        room = np.full((room_config.height, room_config.width), self._symbols["wall"], dtype="U6")
        room[room_config.border:room_config.border+content_height,
             room_config.border:room_config.border+content_width] = content

        return room

    def termination_conditions(self):
        return {
            "Win": [ {"lt": ["game:max_steps", "game:step"]} ],
        }
