#%%
import numpy as np
import math
import yaml
import re
from typing import Dict, List, Tuple, Union, Set
from dataclasses import dataclass

###
# This script calculates the maximum reward for a gridworld environment, defined by the total inventory contents at the end of the max_timesteps.
# It does this by finding all enclosed spaces in the map and then calculating the maximum reward for each space.
# It then returns the minimum of either:
# - the maximum reward for a single agent multiplied by the number of agents in each enclosed space, summed over all enclosed spaces.
# - the maximum reward assuming that the resources are flowing at their maximum possible rate for each enclosed space, summed over all enclosed spaces.

# To run:   
# gridworld = GridWorld(mettagrid_yaml_path, map_array)
# est_max_reward = gridworld.estimate_max_reward()
# print(gridworld.simulation_summary())

# Current limitations:
# - Needs further testing to ensure robustness to all map configurations.
# - Only provides an estimate of the maximum reward - will need to refine to get the exact maximum reward if we want this in the future
# - Assumes no conversion ticks.
# - Assumes walls are indestructible.
###

# Constants for resource types
class ResourceType:
    ORE = 'ore'  # For uncolored resources
    ORE_RED = 'ore.red'
    ORE_BLUE = 'ore.blue'
    ORE_GREEN = 'ore.green'
    BATTERY = 'battery'
    HEART = 'heart'

@dataclass
class ResourceConfig:
    """Configuration for a resource in the environment."""
    cooldown: int
    max_output: int
    input_battery: int = 0  # Altar conversion ratio
    input_ore: int = 0     # Generator conversion ratio

@dataclass
class ColoredResourceConfig:
    """Configuration for colored resources (mines and generators)."""
    red: ResourceConfig
    blue: ResourceConfig
    green: ResourceConfig

@dataclass
class EnclosedSpace:
    """Represents an enclosed section in the map."""
    width: int
    height: int
    mines: Dict[str, List[Tuple[int, int]]]  # dict with color keys
    generators: Dict[str, List[Tuple[int, int]]]  # dict with color keys
    altars: List[Tuple[int, int]]
    agents: List[Tuple[int, int]]
    walls: List[Tuple[int, int]]

class GridWorld:
    """A class to calculate maximum possible reward in a gridworld environment.
    
    This class finds all enclosed spaces in the map and calculates
    the maximum reward using two methods:
    1. Single agent reward multiplied by number of agents
    2. Maximum resource flow rate calculation
    
    Attributes:
        inventory_limit (int): Maximum inventory size for agents
        max_timesteps (int): Maximum number of steps in the simulation
        mine_config (ResourceConfig): Configuration for mines
        generator_config (ResourceConfig): Configuration for generators
        altar_config (ResourceConfig): Configuration for altars
        enclosed_spaces (List[EnclosedSpace]): List of enclosed spaces in the map
        rewards (Dict[str, float]): Reward values for each resource type
    """

    def __init__(self, mettagrid_yaml_path: str, map_array: np.ndarray):
        
        def parse_yaml(param: Union[str, int]) -> int:
            """Parse YAML parameter, handling uniform distribution syntax."""
            if isinstance(param, str):
                match = re.match(r'\$\{uniform:[^,]+,[^,]+,([^\}]+)\}', param)
                return int(match.group(1)) if match else int(param)
            return param

        # Load mettagrid configuration
        with open(mettagrid_yaml_path, 'r') as file:
            env_config = yaml.safe_load(file)

        # Extract configurations
        game_env = env_config['game']
        agent = game_env['agent']
        objects = game_env['objects']

        # Check if using colored or uncolored resources
        self.using_colors = 'mine.red' in objects
        
        if self.using_colors:
            # Initialize colored resource configurations
            self.mine_config = ColoredResourceConfig(
                red=ResourceConfig(
                    cooldown=parse_yaml(objects['mine.red']['cooldown']),
                    max_output=parse_yaml(objects['mine.red']['max_output'])
                ),
                blue=ResourceConfig(
                    cooldown=parse_yaml(objects['mine.blue']['cooldown']),
                    max_output=parse_yaml(objects['mine.blue']['max_output'])
                ),
                green=ResourceConfig(
                    cooldown=parse_yaml(objects['mine.green']['cooldown']),
                    max_output=parse_yaml(objects['mine.green']['max_output'])
                )
            )

            self.generator_config = ColoredResourceConfig(
                red=ResourceConfig(
                    cooldown=parse_yaml(objects['generator.red']['cooldown']),
                    max_output=parse_yaml(objects['generator.red']['max_output']),
                    input_ore=parse_yaml(objects['generator.red']['input_ore.red'])
                ),
                blue=ResourceConfig(
                    cooldown=parse_yaml(objects['generator.blue']['cooldown']),
                    max_output=parse_yaml(objects['generator.blue']['max_output']),
                    input_ore=parse_yaml(objects['generator.blue']['input_ore.blue'])
                ),
                green=ResourceConfig(
                    cooldown=parse_yaml(objects['generator.green']['cooldown']),
                    max_output=parse_yaml(objects['generator.green']['max_output']),
                    input_ore=parse_yaml(objects['generator.green']['input_ore.green'])
                )
            )
        else:
            # Initialize uncolored resource configurations
            self.mine_config = ResourceConfig(
                cooldown=parse_yaml(objects['mine']['cooldown']),
                max_output=parse_yaml(objects['mine']['max_output'])
            )
            
            self.generator_config = ResourceConfig(
                cooldown=parse_yaml(objects['generator']['cooldown']),
                max_output=parse_yaml(objects['generator']['max_output']),
                input_ore=parse_yaml(objects['generator']['input_ore'])
            )

        self.altar_config = ResourceConfig(
            cooldown=parse_yaml(objects['altar']['cooldown']),
            max_output=parse_yaml(objects['altar']['max_output']),
            input_battery=parse_yaml(objects['altar']['input_battery'])
        )

        # Load environment parameters
        self.inventory_limit = parse_yaml(agent['max_inventory'])
        self.max_timesteps = parse_yaml(game_env['max_steps'])
        
        # Set up rewards based on whether using colors
        if self.using_colors:
            self.rewards = {
                ResourceType.ORE_RED: parse_yaml(agent['rewards']['ore.red']),
                ResourceType.ORE_BLUE: parse_yaml(agent['rewards']['ore.blue']),
                ResourceType.ORE_GREEN: parse_yaml(agent['rewards']['ore.green']),
                ResourceType.BATTERY: parse_yaml(agent['rewards']['battery']),
                ResourceType.HEART: parse_yaml(agent['rewards']['heart'])
            }
        else:
            self.rewards = {
                ResourceType.ORE: parse_yaml(agent['rewards']['ore']),  # Use single ore reward
                ResourceType.BATTERY: parse_yaml(agent['rewards']['battery']),
                ResourceType.HEART: parse_yaml(agent['rewards']['heart'])
            }

        # Find enclosed spaces in the map array
        self.enclosed_spaces = self.find_enclosed_spaces(map_array)

    def find_enclosed_spaces(self, map_array: np.ndarray) -> List[EnclosedSpace]:
        """Find enclosed spaces in a numpy array map."""
        height, width = map_array.shape
        visited = np.zeros((height, width), dtype=bool)
        enclosed_spaces = []
        
        def flood_fill(x: int, y: int, space: Set[Tuple[int, int]]) -> None:
            """Flood fill to identify connected empty spaces."""
            if x < 0 or x >= width or y < 0 or y >= height:
                return
            if visited[y, x]:
                return
            if map_array[y, x] == "wall":
                return
                
            visited[y, x] = True
            space.add((x, y))
            
            flood_fill(x + 1, y, space)
            flood_fill(x - 1, y, space)
            flood_fill(x, y + 1, space)
            flood_fill(x, y - 1, space)
        
        # Find all enclosed spaces
        for y in range(height):
            for x in range(width):
                if not visited[y, x] and map_array[y, x] != "wall":
                    space = set()
                    flood_fill(x, y, space)
                    
                    # Get space boundaries
                    x_coords = [pos[0] for pos in space]
                    y_coords = [pos[1] for pos in space]
                    space_width = max(x_coords) - min(x_coords) + 1
                    space_height = max(y_coords) - min(y_coords) + 1
                    
                    # Get objects in space, handling both colored and uncolored cases
                    if self.using_colors:
                        mines = {
                            'red': [(x, y) for x, y in space if map_array[y, x] == "mine.red"],
                            'blue': [(x, y) for x, y in space if map_array[y, x] == "mine.blue"],
                            'green': [(x, y) for x, y in space if map_array[y, x] == "mine.green"]
                        }
                        
                        generators = {
                            'red': [(x, y) for x, y in space if map_array[y, x] == "generator.red"],
                            'blue': [(x, y) for x, y in space if map_array[y, x] == "generator.blue"],
                            'green': [(x, y) for x, y in space if map_array[y, x] == "generator.green"]
                        }
                    else:
                        mines = {
                            'default': [(x, y) for x, y in space if map_array[y, x] == "mine"]
                        }
                        
                        generators = {
                            'default': [(x, y) for x, y in space if map_array[y, x] == "generator"]
                        }
                    
                    altars = [(x, y) for x, y in space if map_array[y, x] == "altar"]
                    agents = [(x, y) for x, y in space if map_array[y, x] == "agent.agent"]
                    walls = [(x, y) for x, y in space if map_array[y, x] == "wall"]
                    
                    enclosed_spaces.append(EnclosedSpace(
                        width=space_width,
                        height=space_height,
                        mines=mines,
                        generators=generators,
                        altars=altars,
                        agents=agents,
                        walls=walls
                    ))
        
        return enclosed_spaces

    def manhattan_distance(self, pos_a: Tuple[int, int], pos_b: Tuple[int, int]) -> int:
        """Calculate Manhattan distance between two positions."""
        return abs(pos_a[0] - pos_b[0]) + abs(pos_a[1] - pos_b[1])

    def get_minimal_distances(self, space: EnclosedSpace) -> Dict[str, Tuple[int, int, int]]:
        """Calculate the minimal distances between objects in an enclosed space."""
        distances = {}
        colors = ['red', 'blue', 'green'] if self.using_colors else ['default']
        
        for color in colors:
            # Find closest mine to generator for each color
            mine_to_generator = float('inf')
            for mine_pos in space.mines[color]:
                for gen_pos in space.generators[color]:
                    dist = self.manhattan_distance(mine_pos, gen_pos)
                    mine_to_generator = min(mine_to_generator, dist)
            
            # Find closest generator to altar
            generator_to_altar = float('inf')
            for gen_pos in space.generators[color]:
                for altar_pos in space.altars:
                    dist = self.manhattan_distance(gen_pos, altar_pos)
                    generator_to_altar = min(generator_to_altar, dist)
            
            # Find closest altar to mine
            altar_to_mine = float('inf')
            for altar_pos in space.altars:
                for mine_pos in space.mines[color]:
                    dist = self.manhattan_distance(altar_pos, mine_pos)
                    altar_to_mine = min(altar_to_mine, dist)
            
            distances[color] = (mine_to_generator, generator_to_altar, altar_to_mine)
        
        return distances

    def estimate_max_reward_simple(self, space: EnclosedSpace) -> float:
        """Calculate max reward for a single space by calculating the max reward for a single agent 
        and multiplying by number of agents. Returns the maximum reward given the best color strategy."""
        inventory_space = self.inventory_limit
        time_remaining = self.max_timesteps
        max_reward = 0

        # Calculate rewards for each color strategy
        colors = ['red', 'blue', 'green'] if self.using_colors else ['default']
        for color in colors:
            ore, batteries, hearts = 0, 0, 0
            time_left = time_remaining
            
            # Get configurations for this color
            if self.using_colors:
                mine_config = getattr(self.mine_config, color)
                generator_config = getattr(self.generator_config, color)
            else:
                mine_config = self.mine_config
                generator_config = self.generator_config
            
            # Get distances for this color
            mine_to_generator, generator_to_altar, altar_to_mine = self.get_minimal_distances(space)[color]
            
            while time_left > 0 and inventory_space > 0:
                # Travel to Mine
                if time_left < altar_to_mine:
                    break
                time_left -= altar_to_mine

                # Mining phase
                mine_time = inventory_space * (mine_config.cooldown + 1)
                if time_left < mine_time:
                    ore += time_left // (mine_config.cooldown + 1)
                    break
                time_left -= mine_time
                ore += inventory_space

                # Travel to Generator
                if time_left < mine_to_generator:
                    break
                time_left -= mine_to_generator

                # Generator put phase
                generator_put_time = ore
                if time_left < generator_put_time:
                    ore -= time_left
                    break
                time_left -= generator_put_time

                # Generator get phase
                generator_get_actions = math.ceil(ore / generator_config.max_output)
                generator_get_time = generator_get_actions + (generator_get_actions - 1) * generator_config.cooldown
                if time_left < generator_get_time:
                    completed_gets = (time_left + generator_config.cooldown) // (1 + generator_config.cooldown)
                    batteries += min(ore, completed_gets * generator_config.max_output)
                    ore -= batteries
                    break
                batteries += ore
                ore = 0
                time_left -= generator_get_time

                # Travel to Altar
                if time_left < generator_to_altar:
                    break
                time_left -= generator_to_altar

                # Altar put phase
                altar_put_time = batteries
                if time_left < altar_put_time:
                    batteries -= time_left
                    break
                time_left -= altar_put_time

                # Altar get phase
                hearts_generated = batteries // self.altar_config.input_battery
                altar_get_actions = math.ceil(hearts_generated / self.altar_config.max_output)
                altar_get_time = altar_get_actions + (altar_get_actions - 1) * self.altar_config.cooldown
                if time_left < altar_get_time:
                    completed_gets = (time_left + self.altar_config.cooldown) // (1 + self.altar_config.cooldown)
                    hearts += min(hearts_generated, completed_gets * self.altar_config.max_output)
                    batteries -= hearts * self.altar_config.input_battery
                    break
                hearts += hearts_generated
                batteries = 0
                inventory_space -= hearts_generated
                time_left -= altar_get_time

            # Calculate reward for this color
            if self.using_colors:
                color_reward = (
                    ore * self.rewards[f'ore.{color}'] + 
                    batteries * self.rewards[ResourceType.BATTERY] + 
                    hearts * self.rewards[ResourceType.HEART]
                ) * len(space.agents)
            else:
                color_reward = (
                    ore * self.rewards[ResourceType.ORE] +  # Use single ore reward
                    batteries * self.rewards[ResourceType.BATTERY] + 
                    hearts * self.rewards[ResourceType.HEART]
                ) * len(space.agents)
            
            # Update maximum reward if this color's reward is higher
            max_reward = max(max_reward, color_reward)
        
        return max_reward
    
    def estimate_max_reward_max_flow(self, space: EnclosedSpace) -> float:
        """Calculate max reward for a single space assuming maximum resource flow rate."""
        total_reward = 0
        colors = ['red', 'blue', 'green'] if self.using_colors else ['default']
        
        # Calculate maximum flow for each color
        for color in colors:
            # Get configurations for this color
            if self.using_colors:
                mine_config = getattr(self.mine_config, color)
                generator_config = getattr(self.generator_config, color)
            else:
                mine_config = self.mine_config
                generator_config = self.generator_config
            
            # 1. Mine throughput (ore per timestep)
            time_per_mine_op = mine_config.cooldown + 1
            ore_per_timestep_per_mine = 1 / time_per_mine_op
            max_ore_rate = ore_per_timestep_per_mine * len(space.mines[color])
            
            # 2. Generator throughput (batteries per timestep)
            ore_needed = generator_config.max_output * generator_config.input_ore
            time_per_generator_cycle = (
                ore_needed +                    # Time to put ore
                generator_config.cooldown +      # Cooldown period
                1                               # Action to get batteries
            )
            batteries_per_timestep_per_generator = generator_config.max_output / time_per_generator_cycle
            max_battery_rate = batteries_per_timestep_per_generator * len(space.generators[color])
            
            # 3. Altar throughput (hearts per timestep)
            batteries_needed = self.altar_config.max_output * self.altar_config.input_battery
            time_per_altar_cycle = batteries_needed + self.altar_config.cooldown + 1
            hearts_per_timestep_per_altar = self.altar_config.max_output / time_per_altar_cycle
            max_heart_rate = hearts_per_timestep_per_altar * len(space.altars)
            
            # Convert all rates to the same unit (hearts per timestep) for comparison
            max_heart_rate_from_ore = max_ore_rate / self.altar_config.input_battery
            max_heart_rate_from_batteries = max_battery_rate / self.altar_config.input_battery
            
            # Identify the bottleneck (lowest potential heart production rate)
            max_possible_heart_rate = min(
                max_heart_rate_from_ore, 
                max_heart_rate_from_batteries, 
                max_heart_rate
            )
            
            # Calculate total maximum heart production over the entire simulation
            total_hearts = max_possible_heart_rate * self.max_timesteps
            total_reward += total_hearts * self.rewards[ResourceType.HEART]
        
        return total_reward

    def estimate_max_reward(self) -> float:
        """Calculate the estimated maximum reward using both methods."""
        total_reward = 0
        for space in self.enclosed_spaces:
            space_reward = min(
                self.estimate_max_reward_simple(space),
                self.estimate_max_reward_max_flow(space)
            )
            total_reward += space_reward
        return total_reward
    
    def simulation_summary(self) -> str:
        """Generate a summary of the simulation parameters and results."""
        estimated_reward = self.estimate_max_reward()
        
        summary = (
            f"The estimated maximum reward is {estimated_reward:.2f}. \n"
            f"Key parameters are:\n"
            f"Number of Enclosed Spaces: {len(self.enclosed_spaces)}\n"
            f"Inventory Limit: {self.inventory_limit}\n"
            f"Max Timesteps: {self.max_timesteps}\n"
            f"Using Colored Resources: {self.using_colors}\n"
            f"Rewards:\n"
        )
        
        if self.using_colors:
            summary += (
                f"  - Red Ore: {self.rewards[ResourceType.ORE_RED]}\n"
                f"  - Blue Ore: {self.rewards[ResourceType.ORE_BLUE]}\n"
                f"  - Green Ore: {self.rewards[ResourceType.ORE_GREEN]}\n"
            )
        else:
            summary += f"  - Ore: {self.rewards[ResourceType.ORE]}\n"
            
        summary += (
            f"  - Battery: {self.rewards[ResourceType.BATTERY]}\n"
            f"  - Heart: {self.rewards[ResourceType.HEART]}\n"
        )
        
        # Add space-specific information
        for i, space in enumerate(self.enclosed_spaces):
            summary += f"\nEnclosed Space {i+1}:\n"
            summary += f"Size: {space.width}x{space.height}\n"
            if self.using_colors:
                for color in ['red', 'blue', 'green']:
                    summary += f"Number of {color.capitalize()} Mines: {len(space.mines[color])}\n"
                    summary += f"Number of {color.capitalize()} Generators: {len(space.generators[color])}\n"
            else:
                summary += f"Number of Mines: {len(space.mines['default'])}\n"
                summary += f"Number of Generators: {len(space.generators['default'])}\n"
            summary += f"Number of Altars: {len(space.altars)}\n"
            summary += f"Number of Agents: {len(space.agents)}\n"
        
        return summary