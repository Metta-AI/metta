# MettaGrid Environment

MettaGrid is a multi-agent gridworld environment for studying cooperation and resource economies. Agents collect and transform resources, craft equipment in buildings, and interact through a small set of actions. This document supplements the project [README](../README.md) with detailed mechanics and configuration.

## Overview

- **Recipe-based resources** – Agents keep an inventory of items and interact with converter buildings that transform inputs to outputs after a conversion delay and cooldown. Mines generate ore, generators refine ore into batteries, and higher-tier buildings craft armor, lasers, blueprints, and hearts.
- **Combat** – Attacking consumes a laser and targets an agent in a 3×3 grid in front of the attacker. Successful hits freeze the target and steal its inventory, while armor can be consumed to block the attack.
- **Team reward sharing** – Per step, a configurable portion of each agent’s reward is withheld and evenly redistributed to all members of the agent’s group.
- **Building interaction** – Agents deposit required inputs (`put_items`) and retrieve produced outputs (`get_items`).

## Objects

### Agent

The `Agent` object represents an individual in the environment. Agents track orientation, inventory, group membership, and a freeze counter that is set when they are attacked (see [`agent.hpp`](src/metta/mettagrid/objects/agent.hpp) and [`mettagrid_test_args_env_cfg.json`](tests/mettagrid_test_args_env_cfg.json)).

### Converter buildings

All buildings are instances of a generic converter that consumes recipe inputs, waits `conversion_ticks`, and produces outputs before entering a `cooldown` period (see [`converter.hpp`](src/metta/mettagrid/objects/converter.hpp)). The default test configuration defines the following recipes (see [`mettagrid_test_args_env_cfg.json`](tests/mettagrid_test_args_env_cfg.json)):

- **Mines** output one unit of ore.
- **Generators** transform ore into matching batteries.
- **Armory** turns ore into armor.
- **Lasery** converts ore and batteries into lasers.
- **Lab** consumes ore and batteries to create blueprints.
- **Factory** uses a blueprint plus ore and batteries to mass-produce armor and lasers.
- **Altar** exchanges batteries for hearts (which are the reward items for RL-trained agents).
- **Temple** multiplies hearts when supplied with a heart and a blueprint.
- **Walls** are impassable and not swappable, while **blocks** are swappable obstacles.

Note: Mines, ore, generators, and batteries are color coded (can be red, blue, green), but other converters and resources are currently not color coded as of now and consume only red ore and/or batteries.

## Actions

- `move` and `rotate` move the agent or change its facing direction.
- `put_items` deposits recipe inputs into the adjacent converter (see [`put_recipe_items.hpp`](src/metta/mettagrid/actions/put_recipe_items.hpp)).
- `get_items` withdraws available outputs from the adjacent converter (see [`get_output.hpp`](src/metta/mettagrid/actions/get_output.hpp)).
- `attack` consumes a laser, freezes the target, and steals its resources; armor can block the hit (see [`attack.hpp`](src/metta/mettagrid/actions/attack.hpp) and [`mettagrid_test_args_env_cfg.json`](tests/mettagrid_test_args_env_cfg.json)).
- `swap` exchanges positions with another agent.
- `change_color`, `change_glyph`, and `noop` provide cosmetic or no-op actions.

## Configuration

Environment instances are defined by YAML/JSON configuration files. These files specify agent limits, object recipes, available actions, and room generation. Group definitions include identifiers and a `group_reward_pct` that controls how much reward is shared within a team (see [`mettagrid_test_args_env_cfg.json`](tests/mettagrid_test_args_env_cfg.json)).

## Groups and cooperation

During each step, a portion of every agent's reward equal to `group_reward_pct` is pooled and evenly redistributed among members of the same group (see [`mettagrid_c.cpp`](src/metta/mettagrid/mettagrid_c.cpp)). This mechanism enables experiments on cooperation and collective strategies.

## Future plans

Metabolism, energy costs, richer object interactions, population-level dynamics, and marker-based communication tools (see [`roadmap.md`](../roadmap.md)).

## Environment Architecture

MettaGrid uses a modular architecture designed primarily for the Softmax Studio ML project, with lightweight adapters to maintain compatibility with external RL frameworks:

### Primary Training Environment

**`MettaGridEnv`** – The main environment actively developed for Softmax Studio training systems
- Full-featured environment with comprehensive stats collection, replay recording, and curriculum support
- Inherits from `MettaGridCore` for C++ environment implementation
- **Exclusively used** by `metta.rl.trainer` and `metta.sim.simulation`
- Continuously developed and optimized for Softmax Studio use cases
- Backward compatible with existing training code

### Core Infrastructure

**`MettaGridCore`** – Low-level C++ environment wrapper
- Foundation that provides the core game mechanics and performance
- **Not used directly** for training – serves as implementation detail for `MettaGridEnv`
- Provides the base functionality that external adapters wrap

### External Framework Compatibility Adapters

Lightweight wrappers around `MettaGridCore` to maintain compatibility with other training systems:

- **`MettaGridGymEnv`** – Gymnasium compatibility for research workflows
- **`MettaGridPettingZooEnv`** – PettingZoo compatibility for multi-agent research
- **`MettaGridPufferEnv`** – PufferLib compatibility for high-performance external training

**Important**: These adapters are **only used with their respective training systems**, not with the Metta trainer.

### Design Philosophy

- **Primary Focus**: `MettaGridEnv` receives active development and new features for Softmax Studio
- **Compatibility Maintenance**: External adapters ensure other frameworks continue working as the core evolves
- **Testing for Compatibility**: Demos verify external frameworks remain functional during core development
- **Clear Separation**: Each environment type serves its specific training system – no mixing between systems

### Compatibility Testing Demos

These demos ensure external framework adapters remain functional as the core environment evolves:

```bash
# Verify PettingZoo compatibility
python -m mettagrid.demos.demo_train_pettingzoo

# Verify PufferLib compatibility
python -m mettagrid.demos.demo_train_puffer

# Verify Gymnasium compatibility
python -m mettagrid.demos.demo_train_gym
```

The demos serve as regression tests to catch compatibility issues during core development, ensuring external users can continue using their preferred frameworks.

## Building and testing

For local development, refer to the top-level [README.md](../README.md) in this repository.

### CMake

By default, `uv sync` will run the CMake build in an isolated environment, so if you need to run C++ tests and benchmarks, you'll need to invoke `cmake` directly.

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
