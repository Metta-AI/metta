# Simulator API Reference

This document describes the Simulator, Simulation, and SimulationAgent APIs in MettaGrid. These components provide a
clean, framework-agnostic interface for running multi-agent simulations.

## Overview

The simulator API is organized into three main classes:

- **`Simulator`**: Factory for creating simulations with consistent configuration
- **`Simulation`**: Represents a single running simulation episode
- **`SimulationAgent`**: Provides per-agent operations and observations

## Core Classes

### Simulator

The `Simulator` class is a factory that ensures all simulations maintain consistent configuration invariants (action
names, number of agents, etc.). It manages event handlers and enforces that only one simulation runs at a time.

#### Creating a Simulator

```python
from mettagrid.simulator.simulator import Simulator
from mettagrid.config.mettagrid_config import MettaGridConfig

simulator = Simulator()
```

#### Methods

- **`add_event_handler(handler: SimulatorEventHandler)`**: Register an event handler that will be attached to all
  simulations
- **`new_simulation(config: MettaGridConfig, seed: int = 0) -> Simulation`**: Create a new simulation with the given
  config and seed. Throws an error if a simulation is already running or if config invariants don't match previous
  simulations.
- **`close()`**: Shut down the simulator

#### Configuration Invariants

The simulator enforces that the following configuration properties remain constant across all simulations:

- Number of agents
- Action names
- Object type names
- Resource names
- Vibe names

Attempting to create a simulation with different invariants will raise a `ValueError`.

#### Example: Basic Usage

```python
simulator = Simulator()

# First simulation
config1 = load_config("arena")
sim1 = simulator.new_simulation(config1, seed=42)
# ... run simulation ...
simulator.close()

# Second simulation with same invariants
config2 = load_config("arena")
sim2 = simulator.new_simulation(config2, seed=100)
```

### Simulation

The `Simulation` class represents a single running episode of the game. It provides access to agents, observations,
rewards, and environment state.

#### Properties

##### Configuration and Setup

- **`config: MettaGridConfig`**: The configuration used for this simulation
- **`seed: int`**: Random seed for this simulation
- **`num_agents: int`**: Number of agents in the simulation
- **`current_step: int`**: Current timestep in the episode

##### Map Properties

- **`map_width: int`**: Width of the game map
- **`map_height: int`**: Height of the game map

##### Action Space

- **`action_ids: dict[str, int]`**: Mapping of action names to indices
- **`action_names: list[str]`**: List of available action names
- **`action_success: list[bool]`**: Whether each agent's last action succeeded

##### Observation Space

- **`features: Sequence[ObservationFeatureSpec]`**: Available observation features
- **`num_observation_tokens: int`**: Maximum number of observation tokens per agent
- **`observation_shape: tuple`**: Shape of observation array (num_tokens, 3)

##### Game Elements

- **`object_type_names: list[str]`**: Names of object types in the game
- **`resource_names: list[str]`**: Names of resources in the game

##### Rewards and Stats

- **`episode_rewards: np.ndarray`**: Cumulative rewards for each agent
- **`episode_stats: EpisodeStats`**: Detailed episode statistics

#### Methods

##### Agent Access

- **`agents() -> list[SimulationAgent]`**: Get all agents in the simulation
- **`agent(agent_id: int) -> SimulationAgent`**: Get a specific agent by ID
- **`observations() -> list[AgentObservation]`**: Get observations for all agents

##### Simulation Control

- **`step()`**: Execute one timestep. Actions must be set beforehand via `agent.set_action()`.
- **`is_done() -> bool`**: Check if the episode has ended (all agents truncated or terminated)
- **`end_episode()`**: Force the episode to end by setting all agents to truncated state
- **`close()`**: Clean up and close the simulation

##### Environment Inspection

- **`get_feature(feature_id: int) -> ObservationFeatureSpec`**: Get an observation feature by ID
- **`grid_objects(bbox: Optional[BoundingBox] = None, ignore_types: Optional[List[str]] = None) -> Dict[int, Dict[str, Any]]`**:
  Get objects on the grid, optionally filtered by bounding box and type

#### Example: Running a Simulation

```python
from mettagrid.simulator.types import Action

# Create simulation
sim = simulator.new_simulation(config, seed=42)

# Get agents
agents = sim.agents()

# Game loop
while not sim.is_done():
    # Set actions for each agent
    for agent in agents:
        action = Action(name="move_forward")
        agent.set_action(action)

    # Execute timestep
    sim.step()

    # Check rewards
    for agent in agents:
        print(f"Agent {agent.id} reward: {agent.step_reward}")

# Get final statistics
print(f"Episode rewards: {sim.episode_rewards}")
print(f"Episode stats: {sim.episode_stats}")

sim.close()
```

### SimulationAgent

The `SimulationAgent` class provides per-agent operations and state access. Agents are accessed via
`Simulation.agent(agent_id)` or `Simulation.agents()`.

#### Properties

- **`id: int`**: The agent's unique identifier (0-indexed)
- **`observation: AgentObservation`**: The agent's current observation
- **`step_reward: float`**: Reward received in the last step
- **`episode_reward: float`**: Cumulative reward for the current episode
- **`last_action_success: bool`**: Whether the agent's last action succeeded
- **`inventory: Dict[str, int]`**: The agent's current inventory (resource name -> quantity)
- **`global_observations: Dict[str, int]`**: Global observation features (e.g., episode_completion_pct, last_action)

#### Methods

- **`set_action(action: Action)`**: Set the action for this agent to execute on the next step
- **`set_inventory(inventory: Dict[str, int])`**: Set the agent's inventory. Resources not mentioned will be cleared.

#### Example: Agent Operations

```python
# Get an agent
agent = sim.agent(0)

# Read observation
obs = agent.observation
for token in obs.tokens:
    print(f"Feature: {token.feature.name}, Location: {token.location}, Value: {token.value}")

# Check inventory
inventory = agent.inventory
print(f"Food: {inventory.get('food', 0)}")

# Set inventory
agent.set_inventory({"food": 10, "wood": 5})

# Set action
from mettagrid.simulator.types import Action
agent.set_action(Action(name="harvest"))

# Check if last action succeeded
if not agent.last_action_success:
    print("Action failed!")
```

## Observation System

> **Note**: For detailed technical information about the observation format, including binary structure, coordinate
> encoding, feature IDs, and how to use `IdMap` to discover features, see [observations.md](observations.md). This
> section focuses on using observations via the Python API.

### ObservationFeatureSpec

Represents metadata about an observation feature type.

```python
class ObservationFeatureSpec(BaseModel):
    id: int                # Unique identifier
    name: str              # Human-readable name (e.g., "inv:food", "object_type")
    normalization: float   # Normalization factor for the feature
```

### ObservationToken

Represents a single observation value at a specific location.

```python
@dataclass
class ObservationToken:
    feature: ObservationFeatureSpec  # What is being observed
    location: tuple[int, int]        # (col, row) in observation window
    value: int                       # The observed value

    def row() -> int  # Extract row from location
    def col() -> int  # Extract column from location
```

### AgentObservation

Represents the complete observation for one agent.

```python
@dataclass
class AgentObservation:
    agent_id: int                      # Which agent this is for
    tokens: Sequence[ObservationToken] # The observation tokens
```

#### Understanding Observations

Observations are token-based, where each token represents a feature at a specific location:

- **Spatial features**: Object types, terrain, other agents (vary by location)
- **Inventory features**: Resources held by the agent (appear at agent's center position)
- **Global features**: Episode progress, last action/reward (appear at agent's center position)

Example:

```python
agent = sim.agent(0)
obs = agent.observation

for token in obs.tokens:
    if token.feature.name == "object_type":
        print(f"Object at ({token.col()}, {token.row()}): {token.value}")

# For inventory, use the agent.inventory property which handles the encoding
inventory = agent.inventory
for resource, amount in inventory.items():
    print(f"Inventory {resource}: {amount}")
```

## Event Handling

Event handlers allow you to hook into the simulation lifecycle for custom behavior like rendering, logging, or
statistics collection.

### SimulatorEventHandler

Base class for event handlers.

```python
class SimulatorEventHandler:
    def set_simulation(self, simulation: Simulation) -> None:
        """Called when the handler is attached to a simulation."""
        pass

    def on_episode_start(self) -> None:
        """Called at the start of each episode."""
        pass

    def on_step(self) -> None:
        """Called after each simulation step."""
        pass

    def on_episode_end(self) -> None:
        """Called when an episode ends."""
        pass

    def on_close(self) -> None:
        """Called when the simulation is closed."""
        pass
```

### Example: Custom Event Handler

```python
class RewardLogger(SimulatorEventHandler):
    def __init__(self):
        super().__init__()
        self.episode_rewards = []

    def on_episode_start(self):
        self.step_rewards = []

    def on_step(self):
        # Access simulation via self._sim
        rewards = [agent.step_reward for agent in self._sim.agents()]
        self.step_rewards.append(rewards)

    def on_episode_end(self):
        total = sum(sum(r) for r in self.step_rewards)
        self.episode_rewards.append(total)
        print(f"Episode total reward: {total}")

# Use the handler
simulator = Simulator()
logger = RewardLogger()
simulator.add_event_handler(logger)

sim = simulator.new_simulation(config, seed=42)
# ... run simulation ...
```

## Grid Inspection

### BoundingBox

Used to specify a region of the map for inspection.

```python
@dataclass
class BoundingBox:
    min_row: int
    max_row: int
    min_col: int
    max_col: int
```

### Getting Grid Objects

```python
from mettagrid.simulator.simulator import BoundingBox

# Get all objects
all_objects = sim.grid_objects()

# Get objects in a region
bbox = BoundingBox(min_row=0, max_row=10, min_col=0, max_col=10)
region_objects = sim.grid_objects(bbox=bbox)

# Get objects excluding walls
non_walls = sim.grid_objects(ignore_types=["wall"])

# Inspect an object
for obj_id, obj_data in all_objects.items():
    print(f"Object {obj_id}: {obj_data}")
```

## Complete Example: Multi-Agent Rollout

```python
from mettagrid.simulator.simulator import Simulator
from mettagrid.simulator.types import Action
from mettagrid.config.mettagrid_config import MettaGridConfig

# Setup
config = MettaGridConfig.load("arena")
simulator = Simulator()

# Optional: Add event handlers
from mettagrid.renderer.renderer import create_renderer, RenderMode
renderer = create_renderer(RenderMode.ASCII)
simulator.add_event_handler(renderer)

# Create simulation
sim = simulator.new_simulation(config, seed=42)

# Define policies (example: random actions)
import random
def random_policy(observation):
    return Action(name=random.choice(sim.action_names))

# Run episode
step_count = 0
while not sim.is_done():
    # Set actions for all agents
    for agent in sim.agents():
        action = random_policy(agent.observation)
        agent.set_action(action)

    # Step simulation
    sim.step()
    step_count += 1

    # Optional: Print per-step info
    if step_count % 100 == 0:
        avg_reward = sum(a.episode_reward for a in sim.agents()) / sim.num_agents
        print(f"Step {step_count}, Avg reward: {avg_reward:.2f}")

# Report results
print(f"Episode ended after {step_count} steps")
print(f"Final rewards: {sim.episode_rewards}")
print(f"Episode stats: {sim.episode_stats}")

# Cleanup
sim.close()
```

## Best Practices

1. **Always close simulations**: Call `sim.close()` or `simulator.close()` when done
2. **Set actions before stepping**: Use `agent.set_action()` before calling `sim.step()`
3. **Reuse simulators**: Create one `Simulator` and use it for multiple simulations to enforce consistent configuration
4. **Use event handlers**: For rendering, logging, or statistics, implement `SimulatorEventHandler` rather than
   polluting the main loop
5. **Check action success**: Use `agent.last_action_success` to detect invalid actions
6. **Access inventory via properties**: Use `agent.inventory` instead of parsing observations manually
7. **Understand token semantics**: Spatial features use location, while inventory/global features appear at the agent's
   center position

## Common Patterns

### Testing Different Seeds

```python
simulator = Simulator()
for seed in range(10):
    sim = simulator.new_simulation(config, seed=seed)
    # ... run simulation ...
    print(f"Seed {seed} reward: {sim.episode_rewards.mean()}")
    simulator.close()
```

### Collecting Episode Statistics

```python
episode_data = []
for episode in range(100):
    sim = simulator.new_simulation(config, seed=episode)
    # ... run simulation ...
    episode_data.append({
        'seed': episode,
        'steps': sim.current_step,
        'rewards': sim.episode_rewards.copy(),
        'stats': sim.episode_stats
    })
    simulator.close()
```

### Debugging Actions

```python
sim.step()
for agent in sim.agents():
    if not agent.last_action_success:
        print(f"Agent {agent.id} action failed!")
        print(f"Observation: {agent.observation}")
        print(f"Inventory: {agent.inventory}")
```

## See Also

- **Rollout class** (`mettagrid.simulator.rollout`): Higher-level API for running policy rollouts
- **PettingZoo environment** (`mettagrid.envs.pettingzoo_env`): Framework-specific wrapper for training
- **Policy interface** (`mettagrid.policy.policy`): Abstract interface for agent policies
- **Renderer** (`mettagrid.renderer.renderer`): Visualization event handlers
