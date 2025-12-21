# MettaGrid Environment

MettaGrid is a multi-agent gridworld environment for studying the emergence of cooperation and social behaviors in
reinforcement learning agents. The environment features a variety of objects and actions that agents can interact with
to manage resources, engage in combat, share with others, and optimize their rewards.

## Requirements

- Bazel 7.0.0 or newer (the project uses Bzlmod and modern Bazel features)
- Python 3.11 or newer
- C++ compiler with C++20 support

## Overview

In MettaGrid, agents navigate a gridworld and interact with various objects to manage their energy, harvest resources,
engage in combat, and cooperate with other agents. The key dynamics include:

- **Energy Management**: Agents must efficiently manage their energy, which is required for all actions. They can
  harvest resources and convert them to energy at charger stations.
- **Resource Gathering**: Agents can gather resources from generator objects scattered throughout the environment.
- **Cooperation and Sharing**: Agents have the ability to share resources with other agents and use energy to power the
  heart assembler, which provides rewards.
- **Combat**: Agents can attack other agents to temporarily freeze them and steal their resources. They can also use
  shields to defend against attacks.

The environment is highly configurable, allowing for experimentation with different world layouts, object placements,
and agent capabilities.

## Objects

### Agent

<img src="https://github.com/daveey/Griddly/blob/develop/resources/images/oryx/oryx_tiny_galaxy/tg_sliced/tg_monsters/tg_monsters_astronaut_u1.png?raw=true" width="32"/>

The `Agent` object represents an individual agent in the environment. Agents can move, rotate, attack, and interact with
other objects. Each agent has energy, resources, and shield properties that govern its abilities and interactions.

### assembler

<img src="https://github.com/daveey/Griddly/blob/develop/resources/images/oryx/oryx_tiny_galaxy/tg_sliced/tg_items/tg_items_heart_full.png?raw=true" width="32"/>

The `assembler` object allows agents to spend energy to gain rewards. Agents can power the assembler by using the `use`
action when near it. The assembler has a cooldown period between uses.

- Using the heart assembler costs `assembler.use_cost energy`. So, no matter how much energy you have, you are always
  dumping the same amount of energy in it and getting the same amount of reward.
- After the heart assembler is used, it is unable to be used for the next value in `assembler.cooldown` (defaults to a
  single delay) timesteps. The cooldown field accepts either an integer or a list of integers; when a list is provided
  the assembler cycles through those delays.
- A single use of the heart assembler gives you a single unit of reward: if
  `target._type_id == ObjectType.assemblerT: self.env._rewards[actor_id] += 1`

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
- After the converter is used, it waits for the next value in `converter.cooldown` before it can be used again.
  Supplying a list of integers causes the converter to cycle through the provided schedule.

### Generator

<img src="https://github.com/daveey/Griddly/blob/develop/resources/images/oryx/oryx_fantasy/ore-0.png?raw=true" width="32"/>

The `Generator` object produces resources that agents can harvest. Agents can gather resources from generators by moving
to them and taking the `use` action. Generators have a specified capacity and replenish resources over time.

- Using the generator once gives one resource 1
- After the generator is used, it is unable to be used for the next value in `generator.cooldown` timesteps

### Wall

<img src="https://github.com/daveey/Griddly/blob/develop/resources/images/oryx/oryx_fantasy/wall2-0.png?raw=true" width="32"/>

The `Wall` object acts as an impassable barrier in the environment, restricting agent movement.

### Cooldown

The `cooldown` property holds one or more delays that determine how long objects wait before they can be used again.
When provided as a list, the delays are applied cyclically.

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

The `use` action allows agents to interact with objects such as assemblers, converters, and generators. The specific
effects of the `use` action depend on the target object and can include converting resources to energy, powering the
assembler for rewards, or harvesting resources from generators.

## Configuration

The MettaGrid environment is highly configurable through the use of YAML configuration files. These files specify the
layout of the gridworld, the placement of objects, and various properties of the objects and agents.

**Current settings:**

1. Ore   - Base resource obtained from mines. Mines produce one ore when used. No resource requirements for use.   -
   Reward value: 0.005 per unit (max 2)   - Used to create batteries and lasers
2. Battery   - Intermediate resource created from ore at a generator. Generator turns one ore into one battery.   -
   Reward value: 0.01 per unit (max 2)   - Used to create hearts and lasers
3. Heart / heart assembler   - High value reward, requires 3 batteries to be converted into a heart at a heart
   assembler.
4. Laser   - Weapon resource created from ore and batteries. Requires 1 ore and 2 batteries. Created at the lasery.

- Consumed on use. When hitting an unarmored agent: freezes them and steals their whole inventory. When hitting an
  armoured agent, destroys their armor. **Inventory System**
- Agents have limited inventory space (default max: 50 items)
- Resources provide rewards just by being in inventory (up to their max reward value)
- Resources can be stolen through attacks Objects Various buildings: Mine, Generator, Armory, Lasery, assembler, Lab,
  Factory, Temple.
- HP — hitpoints, the number of times something can be hit before destruction.
- Cooldown between uses (varies by building)
- Can be damaged and destroyed by attacks

## Environment Architecture

MettaGrid uses a modular architecture designed primarily for the Softmax Studio ML project, with lightweight adapters to
maintain compatibility with external RL frameworks:

### Primary Training Environment

**`PufferMettaGridEnv`** - The main environment actively developed for Softmax Studio training systems

- Full-featured environment with comprehensive stats collection, replay recording, and curriculum support
- PufferLib-compatible environment wrapper that provides reset/step API
- **Exclusively used** by `metta.rl.trainer` and `metta.sim.simulation`
- Continuously developed and optimized for Softmax Studio use cases
- Backward compatible with existing training code

### Core Infrastructure

**`Simulation`** - Core simulation class for running MettaGrid simulations

- Foundation that provides the core game mechanics and performance
- Direct simulation access without environment API (no reset/step)
- Use when you need fine-grained control over simulation steps

### External Framework Compatibility Adapters

Lightweight wrappers around `MettaGridCore` to maintain compatibility with other training systems:

- **`MettaGridGymEnv`** - Gymnasium compatibility for research workflows
- **`MettaGridPettingZooEnv`** - PettingZoo compatibility for multi-agent research
- **`MettaGridPufferEnv`** - PufferLib compatibility for high-performance external training

**Important**: These adapters are **only used with their respective training systems**, not with the Metta trainer.

### Design Philosophy

- **Primary Focus**: `PufferMettaGridEnv` receives active development and new features for Softmax Studio
- **Compatibility Maintenance**: External adapters ensure other frameworks continue working as the core evolves
- **Testing for Compatibility**: Demos verify external frameworks remain functional during core development
- **Clear Separation**: Each environment type serves its specific training system - no mixing between systems

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

The demos serve as regression tests to catch compatibility issues during core development, ensuring external users can
continue using their preferred frameworks.

## Building and testing

For local development, refer to the top-level [README.md](../README.md) in this repository.

### Bazel

By default, `uv sync` will run the Bazel build automatically via the custom build backend. If you need to run C++ tests
and benchmarks directly, you'll need to invoke `bazel` directly.

Build C++ tests and benchmarks in debug mode:

```sh
# Build with debug flags
bazel build --config=dbg //:mettagrid_c
# Run all tests
bazel test //...
```

For benchmarks you might prefer to use the optimized build:

```sh
# Build with optimizations
bazel build --config=opt //:mettagrid_c

# Run benchmarks
./build-release/benchmarks/grid_object_benchmark
```

For a quick benchmark of MettaGrid performance on a single CPU core, use the `test_perf.sh` script:

```bash
bash test_perf.sh
```

## Debugging C++ Code

MettaGrid is written in C++ with Python bindings via pybind11. You can debug C++ code directly in VSCode/Cursor by
setting breakpoints in the C++ source files.

### Prerequisites

1. **VSCode Extension**: Install the
   [Python C++ Debugger](https://marketplace.visualstudio.com/items?itemName=benjamin-simmonds.pythoncpp-debug)
   extension (`pythoncpp`)
2. **Debug Build**: Always build with `DEBUG=1` to enable debug symbols and dSYM generation

### Setup

The repository includes pre-configured launch configurations in `.vscode/launch.json`:

- **MettaGrid Demo** and other pythoncpp configurations - Combined Python + C++ debugging session for the demo script
  (requires the pythoncpp extension)
- **\_C++ Attach** - Attach C++ debugger to any running Python process (shared by all configurations but can be ran
  manually).

### Quick Start

1. **Build with debug symbols**:
   - Clean everything up

     ```sh
     cd packages/mettagrid # (from root of the repository)
     bazel clean --expunge
     ```

   - Rebuild with debug flags

     ```sh
     bazel build --config=dbg //:mettagrid_c
     ```

   - Or Reinstall with DEBUG=1 to trigger dSYM generation

     ```sh
     cd ../..
     export DEBUG=1
     uv sync --reinstall-package mettagrid
     ```

2. **Set breakpoints** in both Python and C++ files (e.g., `packages/mettagrid/cpp/bindings/mettagrid_c.cpp`,
   `packages/mettagrid/demos/demo_train_pettingzoo.py`)

3. **Launch debugger** using the "MettaGrid Demo" or any other pythoncpp configuration from the VSCode Run panel.

4. **Alternatively**, you can use the "\_C++ Attach" configuration to attach the debugger to any running Python process.
   It will ask you to select a process - type "metta" or "python" to filter the list.

### Testing C++ Debugging

To verify that C++ breakpoints are working correctly, use a simple test that calls from Python into C++:

#### Quick Test Method

1. **Add a test call** to any Python entrypoint that uses mettagrid:

   ```python
   def test_cpp_debugging() -> None:
       """Test function to trigger C++ code for debugging."""
       try:
           from mettagrid.mettagrid_c import PackedCoordinate

           # Call a simple C++ function
           packed = PackedCoordinate.pack(5, 10)
           print(f"C++ test: PackedCoordinate.pack(5, 10) = {packed}")

           # Unpack it back
           r, c = PackedCoordinate.unpack(packed)
           print(f"C++ test: PackedCoordinate.unpack({packed}) = ({r}, {c})")
       except Exception as e:
           print(f"C++ debugging test failed: {e}")

   # Call at module level or early in your script
   test_cpp_debugging()
   ```

2. **Set a C++ breakpoint** in the corresponding C++ implementation:
   - Open `packages/mettagrid/cpp/include/mettagrid/systems/packed_coordinate.hpp`
   - Find the `pack()` or `unpack()` function implementation
   - Set a breakpoint inside the function body (e.g., on the return statement)

3. **Launch your debug configuration** (e.g., "MettaGrid Demo" or any pythoncpp configuration)

4. **Verify the breakpoint hits** when the Python code calls `PackedCoordinate.pack()`

#### Where to Add the Test

Add the test call early in any Python entrypoint that uses mettagrid:

- Demo scripts (e.g., `packages/mettagrid/demos/demo_train_*.py`)
- CLI entrypoints (e.g., `packages/cogames/src/cogames/main.py`)
- Tool runners (e.g., `common/src/metta/common/tool/run_tool.py`)
- Training scripts (e.g., `metta/tools/train.py`)

**Note**: This test is only for verifying your debugging setup. Remove it before committing.

### Configuration Files

- **`.bazelrc`** - Defines the `--config=dbg` build mode with debug flags (`-g`, `-O0`, `--apple_generate_dsym`)
- **`.vscode/launch.json`** - Contains launch configurations for combined Python/C++ debugging

### Important Notes

- **Always use `DEBUG=1`**: Without this environment variable, dSYM files won't be generated and C++ breakpoints won't
  work.
- **Source maps**: The launch config includes source maps to correctly locate C++ files in the packages/mettagrid's
  workspace.
