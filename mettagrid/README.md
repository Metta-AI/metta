# MettaGrid Environment

MettaGrid is a multi-agent gridworld environment for studying the emergence of cooperation and social behaviors in
reinforcement learning agents. The environment features a variety of objects and actions that agents can interact with
to manage resources, engage in combat, share with others, and optimize their rewards.

## Overview

In MettaGrid, agents navigate a gridworld and interact with various objects to manage their energy, harvest resources,
engage in combat, and cooperate with other agents. The key dynamics include:

- **Energy Management**: Agents must efficiently manage their energy, which is required for all actions. They can
  harvest resources and convert them to energy at charger stations.
- **Resource Gathering**: Agents can gather resources from generator objects scattered throughout the environment.
- **Cooperation and Sharing**: Agents have the ability to share resources with other agents and use energy to power the
  heart altar, which provides rewards.
- **Combat**: Agents can attack other agents to temporarily freeze them and steal their resources. They can also use
  shields to defend against attacks.

The environment is highly configurable, allowing for experimentation with different world layouts, object placements,
and agent capabilities.

## Objects

### Agent

<img src="https://github.com/daveey/Griddly/blob/develop/resources/images/oryx/oryx_tiny_galaxy/tg_sliced/tg_monsters/tg_monsters_astronaut_u1.png?raw=true" width="32"/>

The `Agent` object represents an individual agent in the environment. Agents can move, rotate, attack, and interact with
other objects. Each agent has energy, resources, and shield properties that govern its abilities and interactions.

### Altar

<img src="https://github.com/daveey/Griddly/blob/develop/resources/images/oryx/oryx_tiny_galaxy/tg_sliced/tg_items/tg_items_heart_full.png?raw=true" width="32"/>

The `Altar` object allows agents to spend energy to gain rewards. Agents can power the altar by using the `use` action
when near it. The altar has a cooldown period between uses.

- Using the heart altar costs `altar.use_cost energy`. So, no matter how much energy you have, you are always dumping
  the same amount of energy in it and getting the same amount of reward.
- After the heart altar is used, it is unable to be used for altar.cooldown timesteps.
- A single use of the heart altar gives you a single unit of reward: if
  `target._type_id == ObjectType.AltarT: self.env._rewards[actor_id] += 1`

### Converter

<img src="https://github.com/daveey/Griddly/blob/develop/resources/images/oryx/oryx_tiny_galaxy/tg_sliced/tg_items/tg_items_pda_A.png?raw=true" width="32"/>

The `Converter` object allows agents to convert their harvested resources into energy. Agents can use converters by
moving to them and taking the `use` action. Each use of a converter provides a specified amount of energy and has a
cooldown period.

- Using the converter does not cost any energy.
- Using the converter outputs `converter.energy_output.r1` energy
  - see `this.output_energy = cfg[b"energy_output.r1"]` in the Converter cppclass
- Using the converter increments resource 2 by one and decrements resource 1 by 1
- There is currently no use for `converter.energy_output.r2` and `converter.energy_output.r3`
- After the converter is used, it is unable to be used for `converter.cooldown` timesteps

### Generator

<img src="https://github.com/daveey/Griddly/blob/develop/resources/images/oryx/oryx_fantasy/ore-0.png?raw=true" width="32"/>

The `Generator` object produces resources that agents can harvest. Agents can gather resources from generators by moving
to them and taking the `use` action. Generators have a specified capacity and replenish resources over time.

- Using the generator once gives one resource 1
- After the generator is used, it is unable to be used for `generator.cooldown` timesteps

### Wall

<img src="https://github.com/daveey/Griddly/blob/develop/resources/images/oryx/oryx_fantasy/wall2-0.png?raw=true" width="32"/>

The `Wall` object acts as an impassable barrier in the environment, restricting agent movement.

### Cooldown

The `cooldown` property determines how long before objects can be used again.

## Actions

### Move / Rotate

The `move` action allows agents to move to an adjacent cell in the gridworld. The action has two modes: moving forward
and moving backward relative to the agent's current orientation.

The `rotate` action enables agents to change their orientation within the gridworld. Agents can rotate to face in four
directions: down, left, right, and up.

### Attack

The `attack` action allows agents to attack other agents or objects within their attack range. Successful attacks freeze
the target for `freeze_duration` timesteps and allow the attacker to steal resources. Further, the attacked agent's
energy is set to `0`. Attacks have a cost and inflict a damage value. The agent selects from one of nine coordinates
within its attack range.

### Shield (Toggle)

The `shield` action turns on a shield. When the shield is active, the agent is protected from attacks by other agents.
The shield consumes energy defined by `upkeep.shield` while active. Attack damage is subtracted from the agent's energy,
rather than freezing the agent.

### Transfer

The `transfer` action enables agents to share resources with other agents. Agents can choose to transfer specific
resources to another agent in an adjacent cell. It is currently not implemented.

### Use

The `use` action allows agents to interact with objects such as altars, converters, and generators. The specific effects
of the `use` action depend on the target object and can include converting resources to energy, powering the altar for
rewards, or harvesting resources from generators.

### Swap

The `swap` action allows agents to swap positions with other agents. It is currently not implemented.

## Configuration

The MettaGrid environment is highly configurable through the use of YAML configuration files. These files specify the
layout of the gridworld, the placement of objects, and various properties of the objects and agents.

**Current settings:**

1. Ore   - Base resource obtained from mines. Mines produce one ore when used. No resource requirements for use.   -
   Reward value: 0.005 per unit (max 2)   - Used to create batteries and lasers
2. Battery   - Intermediate resource created from ore at a generator. Generator turns one ore into one battery.   -
   Reward value: 0.01 per unit (max 2)   - Used to create hearts and lasers
3. Heart / heart altar   - High value reward, requires 3 batteries to be converted into a heart at a heart altar.
4. Laser   - Weapon resource created from ore and batteries. Requires 1 ore and 2 batteries. Created at the lasery.

- Consumed on use. When hitting an unarmored agent: freezes them and steals their whole inventory. When hitting an
  armoured agent, destroys their armor. **Inventory System**
- Agents have limited inventory space (default max: 50 items)
- Resources provide rewards just by being in inventory (up to their max reward value)
- Resources can be stolen through attacks Objects Various buildings: Mine, Generator, Armory, Lasery, Altar, Lab,
  Factory, Temple.
- HP — hitpoints, the number of times something can be hit before destruction.
- Cooldown between uses (varies by building)
- Can be damaged and destroyed by attacks

## Building and testing

For local development, refer to the top-level [README.md](../README.md) in this repository.

### CMake

By default, `uv sync` will run the CMake build in an isolated environment, so if you need to run C++ tests and
benchmarks, you'll need to invoke `cmake` directly.

Build C++ tests and benchmarks in debug mode:

```sh
# Generate `./build-debug` dir. Presets are described in `./CMakePresets.json`.
cmake --preset debug
# Build `./build-debug` dir
cmake --build build-debug
# Run all tests
ctest --test-dir build-debug
```

For benchmarks you might prefer to use the release build:

```sh
cmake --preset release
cmake --build build-release

# Run some benchmark
./build-release/benchmarks/grid_object_benchmark
```
