import numpy as np
import math
import yaml
import re
from typing import Dict, List, Tuple, Union
from dataclasses import dataclass

###
# This script calculates the maximum reward for a gridworld environment, defined as the total inventory contents at the end of the max_timesteps.
# It does this by simulating the environment with random object placements
# then calculating the maximum reward for a single agent multiplied by the number of agents and rooms.
# It also calculates the maximum reward assuming that the resources are flowing at their maximum possible rate
# and returns the minimum of the two as the estimated maximum reward.

# To run:   
# gridworld = GridWorld(mettagrid_yaml_path, map_config_yaml_path)
# est_max_reward = gridworld.estimate_max_reward()
# print(gridworld.simulation_summary())

# Current limitations:
# - Only works for simple gridworlds - will need to adapt to more complex environments. Currenlty optimised for simple.yaml
# - Only provides an estimate of the maximum reward - will need to refine to get the exact maximum reward if we want this in the future
# - Assumes no conversion ticks.
# - Assumes no generator conversion rate.
###

# Constants for resource types
class ResourceType:
    ORE = 'ore'
    BATTERY = 'battery'
    HEART = 'heart'

@dataclass
class ResourceConfig:
    """Configuration for a resource in the environment."""
    cooldown: int
    max_output: int
    input_battery: int = 0  # Altar conversion ratio
    input_ore: int = 0     # Generator conversion ratio

class GridWorld:
    """A class to calculate maximum possible reward in a gridworld environment.
    
    This class simulates the environment with random object placements and calculates
    the maximum reward using two methods:
    1. Single agent reward multiplied by number of agents and rooms
    2. Maximum resource flow rate calculation
    
    Attributes:
        n_simulations (int): Number of simulations to run for distance estimation
        inventory_limit (int): Maximum inventory size for agents
        max_timesteps (int): Maximum number of steps in the simulation
        mine_config (ResourceConfig): Configuration for mines
        generator_config (ResourceConfig): Configuration for generators
        altar_config (ResourceConfig): Configuration for altars
        width (int): Width of each room
        height (int): Height of each room
        num_agents_per_room (int): Number of agents per room
        num_rooms (int): Total number of rooms
        n_mines (int): Number of mines per room
        n_generators (int): Number of generators per room
        n_altars (int): Number of altars per room
        rewards (Dict[str, float]): Reward values for each resource type
    """

    def __init__(self, mettagrid_yaml_path: str, map_config_yaml_path: str, n_simulations: int = 1000):
        self.n_simulations = n_simulations

        def parse_yaml(param: Union[str, int]) -> int:
            """Parse YAML parameter, handling uniform distribution syntax.
            
            Args:
                param: Parameter value from YAML
                
            Returns:
                Parsed integer value
            """
            if isinstance(param, str):
                match = re.match(r'\$\{uniform:[^,]+,[^,]+,([^\}]+)\}', param)
                return int(match.group(1)) if match else int(param)
            return param

        # Load configurations
        with open(mettagrid_yaml_path, 'r') as file:
            env_config = yaml.safe_load(file)
        with open(map_config_yaml_path, 'r') as file:
            map_config = yaml.safe_load(file)

        # Extract configurations
        game_env = env_config['game']
        agent = game_env['agent']
        objects = game_env['objects']
        game = map_config['game']
        map_builder = game['map_builder']
        room = map_builder['room']
        object_counts = room['objects']

        # Initialize resource configurations
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
        self.rewards = {
            ResourceType.ORE: parse_yaml(agent['rewards']['ore']),
            ResourceType.BATTERY: parse_yaml(agent['rewards']['battery']),
            ResourceType.HEART: parse_yaml(agent['rewards']['heart'])
        }

        # Load map parameters
        self.width = parse_yaml(room['width'])
        self.height = parse_yaml(room['height'])
        self.num_agents_per_room = parse_yaml(room['agents'])
        self.num_rooms = parse_yaml(game['num_agents']) / self.num_agents_per_room
        self.n_mines = parse_yaml(object_counts['mine'])
        self.n_generators = parse_yaml(object_counts['generator'])
        self.n_altars = parse_yaml(object_counts['altar'])

    def generate_positions(self, n_objects: int) -> np.ndarray:
        """Generate random positions for objects in the grid.
        
        Args:
            n_objects: Number of positions to generate
            
        Returns:
            Array of (x, y) positions
        """
        positions = np.random.choice(self.width * self.height, n_objects, replace=False)
        return np.array([(pos % self.width, pos // self.width) for pos in positions])

    def manhattan_distance(self, pos_a: Tuple[int, int], pos_b: Tuple[int, int]) -> int:
        """Calculate Manhattan distance between two positions."""
        return abs(pos_a[0] - pos_b[0]) + abs(pos_a[1] - pos_b[1])

    def average_minimal_distance(self, set_a: np.ndarray, set_b: np.ndarray) -> float:
        """Calculate average minimal distance between two sets of positions."""
        distances = [min(self.manhattan_distance(pos_a, pos_b) for pos_b in set_b) 
                    for pos_a in set_a]
        return np.mean(distances)

    def simulate_distances(self) -> Dict[str, float]:
        """Simulate average distances between resource types."""
        avg_mine_gen, avg_gen_altar, avg_altar_mine = [], [], []

        for _ in range(self.n_simulations):
            mines = self.generate_positions(self.n_mines)
            generators = self.generate_positions(self.n_generators)
            altars = self.generate_positions(self.n_altars)

            avg_mine_gen.append(self.average_minimal_distance(mines, generators))
            avg_gen_altar.append(self.average_minimal_distance(generators, altars))
            avg_altar_mine.append(self.average_minimal_distance(altars, mines))

        return {
            'mine_to_generator': np.mean(avg_mine_gen),
            'generator_to_altar': np.mean(avg_gen_altar),
            'altar_to_mine': np.mean(avg_altar_mine)
        }

    def estimate_max_reward_simple(self) -> float:
        """Calculate max reward by calculating the max reward for a single agent 
        and multiplying by number of agents and rooms. 
        
        Simulates a single agent's optimal path through the environment,
        considering travel time, resource collection, and conversion rates.
        
        Returns:
            float: Estimated maximum total reward
        """
        distances = self.simulate_distances()
        inventory_space = self.inventory_limit
        time_remaining = self.max_timesteps
        ore, batteries, hearts = 0, 0, 0

        while time_remaining > 0 and inventory_space > 0:
            # Travel to Mine
            if time_remaining < distances['altar_to_mine']:
                break
            time_remaining -= distances['altar_to_mine']

            # Mining phase
            mine_time = inventory_space * (self.mine_config.cooldown + 1)
            if time_remaining < mine_time:
                ore += time_remaining // (self.mine_config.cooldown + 1)
                break
            time_remaining -= mine_time
            ore += inventory_space

            # Travel to Generator
            if time_remaining < distances['mine_to_generator']:
                break
            time_remaining -= distances['mine_to_generator']

            # Generator put phase
            generator_put_time = ore
            if time_remaining < generator_put_time:
                ore -= time_remaining
                break
            time_remaining -= generator_put_time

            # Generator get phase
            generator_get_actions = math.ceil(ore / self.generator_config.max_output)
            generator_get_time = generator_get_actions + (generator_get_actions - 1) * self.generator_config.cooldown
            if time_remaining < generator_get_time:
                completed_gets = (time_remaining + self.generator_config.cooldown) // (1 + self.generator_config.cooldown)
                batteries += min(ore, completed_gets * self.generator_config.max_output)
                ore -= batteries
                break
            batteries += ore
            ore = 0
            time_remaining -= generator_get_time

            # Travel to Altar
            if time_remaining < distances['generator_to_altar']:
                break
            time_remaining -= distances['generator_to_altar']

            # Altar put phase
            altar_put_time = batteries
            if time_remaining < altar_put_time:
                batteries -= time_remaining
                break
            time_remaining -= altar_put_time

            # Altar get phase
            hearts_generated = batteries // self.altar_config.input_battery
            altar_get_actions = math.ceil(hearts_generated / self.altar_config.max_output)
            altar_get_time = altar_get_actions + (altar_get_actions - 1) * self.altar_config.cooldown
            if time_remaining < altar_get_time:
                completed_gets = (time_remaining + self.altar_config.cooldown) // (1 + self.altar_config.cooldown)
                hearts += min(hearts_generated, completed_gets * self.altar_config.max_output)
                batteries -= hearts * self.altar_config.input_battery
                break
            hearts += hearts_generated
            batteries = 0
            inventory_space -= hearts_generated
            time_remaining -= altar_get_time

        total_reward_per_agent = (
            ore * self.rewards[ResourceType.ORE] + 
            batteries * self.rewards[ResourceType.BATTERY] + 
            hearts * self.rewards[ResourceType.HEART]
        )
        total_reward_per_room = total_reward_per_agent * self.num_agents_per_room
        total_reward = total_reward_per_room * self.num_rooms
        return total_reward
    
    def estimate_max_reward_max_flow(self) -> float:
        """Calculate max reward assuming maximum resource flow rate.
        
        This method calculates the theoretical maximum reward assuming:
        1. Resources flow at their maximum possible rate
        2. There are enough agents to keep all resources operating at throughput limits
        3. No travel time between resources
        
        Returns:
            float: Estimated maximum total reward based on resource flow rates
        """
        # Calculate maximum throughput for each resource type
        
        # 1. Mine throughput (ore per timestep)
        time_per_mine_op = self.mine_config.cooldown + 1
        ore_per_timestep_per_mine = 1 / time_per_mine_op
        max_ore_rate = ore_per_timestep_per_mine * self.n_mines
        
        # 2. Generator throughput (batteries per timestep)
        time_per_generator_cycle = self.generator_config.max_output + self.generator_config.cooldown + 1
        batteries_per_timestep_per_generator = self.generator_config.max_output / time_per_generator_cycle
        max_battery_rate = batteries_per_timestep_per_generator * self.n_generators
        
        # 3. Altar throughput (hearts per timestep)
        batteries_needed = self.altar_config.max_output * self.altar_config.input_battery
        time_per_altar_cycle = batteries_needed + self.altar_config.cooldown + 1
        hearts_per_timestep_per_altar = self.altar_config.max_output / time_per_altar_cycle
        max_heart_rate = hearts_per_timestep_per_altar * self.n_altars
        
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
        total_reward_per_room = total_hearts * self.rewards[ResourceType.HEART]
        total_reward = total_reward_per_room * self.num_rooms
        
        return total_reward

    def estimate_max_reward(self) -> float:
        """Calculate the estimated maximum reward using both methods.
        
        Returns:
            float: The minimum of the two estimation methods, which provides
                  a more conservative estimate of the maximum possible reward
        """
        return min(self.estimate_max_reward_simple(), self.estimate_max_reward_max_flow())
    
    def simulation_summary(self) -> str:
        """Generate a summary of the simulation parameters and results.
        
        Returns:
            str: Formatted string containing all relevant parameters and the
                 estimated maximum reward
        """
        estimated_reward = self.estimate_max_reward()
        estimation_method = (
            "The estimate is based on the assumption that the resources are flowing at their maximum possible rate."
            if min(self.estimate_max_reward_simple(), self.estimate_max_reward_max_flow()) == self.estimate_max_reward_max_flow()
            else "The estimate is based on the maximum reward for a single agent multiplied by the number of agents and rooms."
        )
        
        summary = (
            f"The estimated maximum reward is {estimated_reward:.2f}. \n"
            f"{estimation_method} \n"
            f"Key parameters are:\n"
            f"Number of Rooms: {self.num_rooms}\n"
            f"Number of Agents per Room: {self.num_agents_per_room}\n"
            f"Room Size: {self.width}x{self.height}\n"
            f"Number of Mines (per room): {self.n_mines}\n"
            f"Number of Generators (per room): {self.n_generators}\n"
            f"Number of Altars (per room): {self.n_altars}\n"         
            f"Inventory Limit: {self.inventory_limit}\n"
            f"Max Timesteps: {self.max_timesteps}\n"
            f"Mine Cooldown: {self.mine_config.cooldown}\n"
            f"Generator Cooldown: {self.generator_config.cooldown}\n"
            f"Altar Cooldown: {self.altar_config.cooldown}\n"
            f"Generator Max Output: {self.generator_config.max_output}\n"
            f"Altar Max Output: {self.altar_config.max_output}\n"
            f"Altar Conversion Ratio: {self.altar_config.input_battery}\n"
            f"Ore Reward: {self.rewards[ResourceType.ORE]}\n"
            f"Battery Reward: {self.rewards[ResourceType.BATTERY]}\n"
            f"Heart Reward: {self.rewards[ResourceType.HEART]}\n"
        )
        return summary
