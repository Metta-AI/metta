# MettaGrid Environment

MettaGrid is a multi-agent gridworld environment for studying the emergence of cooperation and social behaviors in reinforcement learning agents. The environment features a variety of objects and actions that agents can interact with to manage resources, engage in combat, share with others, and optimize their rewards.

## Overview

In MettaGrid, agents navigate a gridworld and interact with various objects to manage their energy, harvest resources, engage in combat, and cooperate with other agents. The key dynamics include:

- **Energy Management**: Agents must efficiently manage their energy, which is required for all actions. They can harvest resources and convert them to energy at charger stations.
- **Resource Gathering**: Agents can gather resources from generator objects scattered throughout the environment.
- **Cooperation and Sharing**: Agents have the ability to share resources with other agents and use energy to power the heart altar, which provides rewards.
- **Combat**: Agents can attack other agents to temporarily freeze them and steal their resources. They can also use shields to defend against attacks.

The environment is highly configurable, allowing for experimentation with different world layouts, object placements, and agent capabilities.

## Objects

### Agent

<img src="https://github.com/daveey/Griddly/blob/develop/resources/images/oryx/oryx_tiny_galaxy/tg_sliced/tg_monsters/tg_monsters_astronaut_u1.png?raw=true" width="32"/>

The `Agent` object represents an individual agent in the environment. Agents can move, rotate, attack, and interact with other objects. Each agent has energy, resources, and shield properties that govern its abilities and interactions.

### Altar

<img src="https://github.com/daveey/Griddly/blob/develop/resources/images/oryx/oryx_tiny_galaxy/tg_sliced/tg_items/tg_items_heart_full.png?raw=true" width="32"/>

The `Altar` object allows agents to spend energy to gain rewards. Agents can power the altar by using the `use` action when near it. The altar has a cooldown period between uses.

### Converter

<img src="https://github.com/daveey/Griddly/blob/develop/resources/images/oryx/oryx_tiny_galaxy/tg_sliced/tg_items/tg_items_pda_A.png?raw=true" width="32"/>

The `Converter` object allows agents to convert their harvested resources into energy. Agents can use converters by moving to them and taking the `use` action. Each use of a converter provides a specified amount of energy and has a cooldown period.

### Generator

<img src="https://github.com/daveey/Griddly/blob/develop/resources/images/oryx/oryx_fantasy/ore-0.png?raw=true" width="32"/>

The `Generator` object produces resources that agents can harvest. Agents can gather resources from generators by moving to them and taking the `use` action. Generators have a specified capacity and replenish resources over time.

### Wall

<img src="https://github.com/daveey/Griddly/blob/develop/resources/images/oryx/oryx_fantasy/wall2-0.png?raw=true" width="32"/>

The `Wall` object acts as an impassable barrier in the environment, restricting agent movement.

## Actions

### Move / Rotate

The `move` action allows agents to move to an adjacent cell in the gridworld. The action has two modes: moving forward and moving backward relative to the agent's current orientation.

The `rotate` action enables agents to change their orientation within the gridworld. Agents can rotate to face in four directions: down, left, right, and up.

### Attack

The `attack` action allows agents to attack other agents or objects within their attack range. Successful attacks temporarily freeze the target and may allow the attacker to steal resources.

### Shield (Toggle)

The `shield` action turns on a shield. When the shield is active, the agent is protected from attacks by other agents. The shield consumes energy while active. Attack damage is subtracted from the agent's energy, rather than freezing the agent.

### Transfer

The `transfer` action enables agents to share resources with other agents. Agents can choose to transfer specific resources to another agent in an adjacent cell.

### Use

The `use` action allows agents to interact with objects such as altars, converters, and generators. The specific effects of the `use` action depend on the target object and can include converting resources to energy, powering the altar for rewards, or harvesting resources from generators.

## Configuration

The MettaGrid environment is highly configurable through the use of YAML configuration files. These files specify the layout of the gridworld, the placement of objects, and various properties of the objects and agents.

