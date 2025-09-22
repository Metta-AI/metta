I'll help you catalogue the variables and environment structures in Mettagrid for your artificial chemistry-based
environment design project. Let me analyze the provided files systematically. Ran tool

## 1. EXPLICIT VARIABLES CATALOG

### Core Game Variables (`mettagrid/src/metta/mettagrid/mettagrid_config.py`)

#### Inventory & Resources

- **`inventory_item_names`**: List[str] - Default: ["ore_red", "ore_blue", "ore_green", "battery_red", "battery_blue",
  "battery_green", "heart", "armor", "laser", "blueprint"]
- **`default_resource_limit`**: int (≥0, default: 255) - Global cap for each resource type
- **`resource_limits`**: dict[str, int] - Per-resource specific limit
- **`initial_inventory`**: dict[str, int] - Starting resources for agents

#### Agent Configuration

- **`num_agents`**: int (≥1, default: 24)
- **`freeze_duration`**: int (≥-1, default: 10) - Ticks agent is frozen after certain actions
- **`action_failure_penalty`**: float (≥0, default: 0) - Reward penalty for failed actions

#### Episode & Time

- **`max_steps`**: int (≥0, default: 1000) - 0 means no limit
- **`episode_truncates`**: bool (default: False) - Whether episodes truncate vs terminate

#### Observation Space

- **`obs_width`**: Literal[3,5,7,9,11,13,15] (default: 11)
- **`obs_height`**: Literal[3,5,7,9,11,13,15] (default: 11)
- **`num_observation_tokens`**: int (≥1, default: 200)

#### Rewards (Per Agent Group) (`mettagrid/src/metta/mettagrid/mettagrid_config.py`)

Inventory rewards with caps:

- **`ore_red`**, **`ore_blue`**, **`ore_green`**: float (default: 0)
- **`ore_red_max`**, **`ore_blue_max`**, **`ore_green_max`**: int (default: 255)
- **`battery_red`**, **`battery_blue`**, **`battery_green`**: float (default: 0)
- **`battery_red_max`**, **`battery_blue_max`**, **`battery_green_max`**: int (default: 255)
- **`heart`**: float (default: 1) / **`heart_max`**: int (default: 255)
- **`armor`**: float (default: 0) / **`armor_max`**: int (default: 255)
- **`laser`**: float (default: 0) / **`laser_max`**: int (default: 255)
- **`blueprint`**: float (default: 0) / **`blueprint_max`**: int (default: 255)

Stats rewards (dynamic):

- Any stat name → reward_per_unit
- Any stat_name_max → cumulative cap

#### Group Variables (`mettagrid/src/metta/mettagrid/mettagrid_config.py`)

- **`group_reward_pct`**: float (0.0-1.0) - Percentage of group rewards shared
- **`sprite`**: Optional[int] - Visual representation ID

#### Action Configuration (`mettagrid/src/metta/mettagrid/mettagrid_config.py`)

Each action has:

- **`enabled`**: bool
- **`required_resources`**: dict[str, int] - Resources needed to attempt (mapped in
  `mettagrid/src/metta/mettagrid/mettagrid_c_config.py`)
- **`consumed_resources`**: dict[str, int] - Resources consumed on success (mapped in
  `mettagrid/src/metta/mettagrid/mettagrid_c_config.py`)

Attack-specific:

- **`defense_resources`**: dict[str, int] - Resources that provide defense (used in
  `mettagrid/src/metta/mettagrid/actions/attack.hpp`)

Change glyph-specific:

- **`number_of_glyphs`**: int (0-255) - Available glyph variations (used in
  `mettagrid/src/metta/mettagrid/actions/change_glyph.hpp`)

#### Converter Objects (`mettagrid/src/metta/mettagrid/mettagrid_config.py`)

- **`input_resources`**: dict[str, int] - Required inputs
- **`output_resources`**: dict[str, int] - Produced outputs
- **`max_output`**: int (≥-1, default: 5) - Output cap per conversion
- **`max_conversions`**: int (default: -1) - Total conversion limit
- **`conversion_ticks`**: int (≥0, default: 1) - Time per conversion
- **`cooldown`**: int (≥0) - Ticks between conversions
- **`initial_resource_count`**: int (≥0, default: 0) - Starting resources
- **`color`**: int (0-255, default: 0) - Visual identifier

#### Box Objects (`mettagrid/src/metta/mettagrid/mettagrid_config.py`)

- **`resources_to_create`**: dict[str, int] - Resources spawned when opened

#### Wall Objects (`mettagrid/src/metta/mettagrid/mettagrid_config.py`)

- **`swappable`**: bool (default: False) - Can agents swap positions with it

#### Global Mechanics (`mettagrid/src/metta/mettagrid/mettagrid_config.py`)

- **`resource_loss_prob`**: float (default: 0.0) - Per-step resource decay probability (mapped in
  `mettagrid/src/metta/mettagrid/mettagrid_c_config.py`)
- **`desync_episodes`**: bool (default: True) - Asynchronous episode resets

#### Feature Flags (`mettagrid/src/metta/mettagrid/mettagrid_config.py`)

- **`track_movement_metrics`**: bool (default: True) - Track sequential rotations
- **`no_agent_interference`**: bool (default: False) - Agents can pass through each other (used in
  `mettagrid/src/metta/mettagrid/actions/move.hpp`)
- **`recipe_details_obs`**: bool (default: False) - Show converter recipes in observations (mapped in
  `mettagrid/src/metta/mettagrid/mettagrid_c_config.py`)
- **`allow_diagonals`**: bool (default: False) - Enable diagonal actions (used in
  `mettagrid/src/metta/mettagrid/actions/move.hpp`, `mettagrid/src/metta/mettagrid/actions/rotate.hpp`)

### Global Observation Features (`mettagrid/src/metta/mettagrid/mettagrid_config.py`)

- **`episode_completion_pct`**: bool (default: True) - Include progress % (converted in
  `mettagrid/src/metta/mettagrid/mettagrid_c_config.py`)
- **`last_action`**: bool (default: True) - Include previous action (converted in
  `mettagrid/src/metta/mettagrid/mettagrid_c_config.py`)
- **`last_reward`**: bool (default: True) - Include previous reward (converted in
  `mettagrid/src/metta/mettagrid/mettagrid_c_config.py`)
- **`resource_rewards`**: bool (default: False) - Include reward values (converted in
  `mettagrid/src/metta/mettagrid/mettagrid_c_config.py`)
- **`visitation_counts`**: bool (default: False) - Include cell visit counts Ran tool (converted in
  `mettagrid/src/metta/mettagrid/mettagrid_c_config.py`)

## 2. ENVIRONMENT CONSTRUCTION FUNCTIONS

### MapGenAscii (`mettagrid/src/metta/map/mapgen_ascii.py`)

Simple ASCII map loader with shortcut syntax:

- **`uri`**: str - Path to .map file
- **`border_width`**: int (default: 0) - Empty border around map

Provides streamlined YAML syntax for single ASCII maps without nested configuration.

### VariedTerrain (`mettagrid/src/metta/map/scenes/varied_terrain.py`)

Complex procedural terrain generator with multiple feature types:

#### Style Presets

- **`style`**: str - One of ["all-sparse", "balanced", "dense", "maze"]
  - Controls density scaling of all features
  - Automatically scales to room size (baseline: 60×60)

#### Object Placement Parameters (auto-scaled by style and room area)

**Large Obstacles**:

- **`size_range`**: [10, 25] cells
- **`count`**: Style-dependent (sparse: 0-2, balanced: 3-7, dense: 8-15)
- Randomly grown connected shapes

**Small Obstacles**:

- **`size_range`**: [3, 6] cells
- **`count`**: Style-dependent (sparse: 0-2, balanced: 3-7, dense: 8-15)
- Randomly grown connected shapes

**Cross Obstacles**:

- **`count`**: Style-dependent (sparse: 0-2, balanced: 3-7, dense: 7-15)
- Width: 1-9 cells, Height: 1-9 cells
- Cross-shaped wall patterns

**Mini Labyrinths**:

- **`count`**: Style-dependent (sparse: 0-2, balanced: 3-7, maze: 10-20)
- Size: 11×11 to 25×25 (forced odd dimensions)
- Features:
  - Maze generation via recursive backtracking
  - Border gaps: Minimum 2 contiguous empty cells per edge
  - Thickening probability: 0.0-0.7 (random per maze)
  - **Altar spawn**: 3% chance per empty cell in maze

**Scattered Walls**:

- **`count`**: Style-dependent (sparse: 0-2, balanced: 3-7, dense: 40-60)
- Single wall cells placed randomly

**Blocks** (Rectangular obstacles):

- **`count`**: Style-dependent (sparse: 0-2, balanced: 3-7, dense: 5-15)
- Width: 2-14 cells (uniform random)
- Height: 2-14 cells (uniform random)

#### VariedTerrain Variables

- **`agents`**: int (default: 1) - Number of agents to place
- **`objects`**: dict[str, int] - Additional objects to place (e.g., altars, converters)
- **`clumpiness`**: [int, int] range - Biases object placement (not fully implemented in shown code)

#### Build Order

1. Mini labyrinths
2. Large obstacles → Small obstacles → Cross obstacles
3. Scattered walls
4. Blocks
5. Altars (from objects dict)
6. Agents

All features respect 1-cell clearance between different object types.

### Built-in Environment Factories

**`EnvConfig.EmptyRoom`** (from `mettagrid/src/metta/mettagrid/mettagrid_config.py`):

- **`num_agents`**: int
- **`width`**: int (default: 10)
- **`height`**: int (default: 10)
- **`border_width`**: int (default: 1)
- **`with_walls`**: bool (default: False) Creates minimal environment with move/rotate actions only.

### Map Builder Types

- **`RandomMapBuilder`**: Basic random placement (`mettagrid/src/metta/mettagrid/map_builder/random.py`)
  - `agents`: int - Number of agents
  - `width`, `height`: Map dimensions
  - `border_width`: Wall border thickness

- **`AsciiMapBuilder`**: Direct ASCII map specification (`mettagrid/src/metta/mettagrid/map_builder/ascii.py`)
  - `map_data`: list[list[str]] - 2D grid of cell types

## 3. TRAINER-LEVEL VARIABLES (`mettagrid/src/metta/mettagrid/trainer.py`)

### Batch & Parallelization

- **`target_batch_size`**: Calculated from `forward_pass_minibatch_target_size`
- **`batch_size`**: int - Actual batch size used
- **`num_envs`**: int - Number of parallel environments
- **`rollout_workers`**: int - Number of parallel rollout workers
- **`async_factor`**: int - Asynchronous stepping multiplier
- **`zero_copy`**: bool - Memory optimization flag

### Training Schedule

- **`total_timesteps`**: int - Total training steps
- **`agent_step`**: int - Current training step (checkpointed)
- **`epoch`**: int - Current epoch number (checkpointed)
- **`update_epochs`**: int - PPO epochs per rollout

### PPO Hyperparameters

- **`gamma`**: float - Discount factor
- **`gae_lambda`**: float - GAE lambda for advantage estimation
- **`target_kl`**: Optional[float] - KL divergence threshold for early stopping
- **`max_grad_norm`**: float - Gradient clipping threshold

### V-trace Parameters

- **`vtrace_rho_clip`**: float - Importance sampling ratio clipping
- **`vtrace_c_clip`**: float - V-trace clipping parameter

### Experience Buffer

- **`bptt_horizon`**: int - Backpropagation through time horizon
- **`minibatch_size`**: int - Size of training minibatches
- **`cpu_offload`**: bool - Offload experience to CPU memory

### Prioritized Experience Replay

- **`prio_alpha`**: float - Prioritization exponent
- **`prio_beta0`**: float - Initial importance sampling correction
- **`anneal_beta`**: float - Calculated annealing for beta

### Optimizer Configuration

- **`type`**: "adam" or "muon"
- **`learning_rate`**: float
- **`beta1`**, **`beta2`**: float - Adam/Muon momentum parameters
- **`eps`**: float - Epsilon for numerical stability
- **`weight_decay`**: float/int - L2 regularization

### Evaluation Variables

- **`evaluate_interval`**: int - Epochs between evaluations
- **`num_training_tasks`**: int - Training tasks to evaluate
- **`evaluate_local`**: bool - Run evaluation locally
- **`evaluate_remote`**: bool - Submit remote evaluation
- **`replay_dir`**: str - Directory for replay storage

### Checkpointing

- **`checkpoint_dir`**: str - Save directory
- **`checkpoint_interval`**: int - Steps between checkpoints
- Various checkpoint triggers and force flags

### Monitoring Intervals

- **`grad_mean_variance_interval`**: int - Gradient statistics frequency
- **`profiler`**: Configuration for PyTorch profiling

## 4. C++ CONFIGURATION MAPPING (`mettagrid_c_config.py`)

The C++ converter maps Python configs to C++ with key transformations:

### Resource Mapping

- String resource names → Integer IDs (0-indexed)
- Creates `resource_name_to_id` mapping dictionary

### Group Processing

- Merges default `AgentConfig` with group-specific `props`
- Maps inventory/stats rewards to integer IDs
- Preserves `group_reward_pct` for shared rewards

### Object Type Mapping

- `ConverterConfig` → `CppConverterConfig`
  - Maps all resource dictionaries to integer IDs
  - Preserves timing parameters (ticks, cooldown)
- `WallConfig` → `CppWallConfig`
- `BoxConfig` → `CppBoxConfig`
  - Maps `resources_to_create` to integer IDs

### Action Validation

- Validates `consumed_resources` exist in `inventory_item_names`
- Maps action resource requirements to integer IDs
- Special handling for `AttackActionConfig` (defense resources)
- Special handling for `ChangeGlyphActionConfig` (glyph count)

## 5. KEY CONSTRAINTS & RELATIONSHIPS

### Resource System

- All resources must be declared in `inventory_item_names`
- Resource IDs are array indices (0-based)
- Resource limits cap both inventory and rewards
- `resource_loss_prob` creates decay dynamics

### Temporal Constraints

- `max_steps` bounds episode length (0 = unbounded)
- `conversion_ticks` delays converter outputs
- `cooldown` spaces converter activations
- `freeze_duration` penalizes certain actions

### Spatial Constraints

- Observation window fixed to odd squares (3×3 to 15×15)
- All VariedTerrain objects maintain 1-cell clearance
- Labyrinth borders guarantee 2-cell gaps for connectivity

### Reward Structure

- Inventory rewards: per-item with individual caps
- Stats rewards: arbitrary stat tracking with caps
- Group rewards: percentage-based sharing mechanism
- Action failure penalty: negative reinforcement

## 6. Movement and Actions

### Movement styles (`mettagrid/src/metta/mettagrid/mettagrid_config.py`)

- **allow_diagonals (GameConfig)**: When false (default), movement/orientation uses 4-way (N,S,W,E). When true, uses
  8-way (adds NW, NE, SW, SE). (used by `mettagrid/src/metta/mettagrid/actions/move.hpp`,
  `mettagrid/src/metta/mettagrid/actions/rotate.hpp`)
- **Auto-facing on move**: `move` always updates the agent's orientation to the chosen move direction, even if the move
  fails.
- **Collision mode**:
  - **no_agent_interference=false**: standard collision; moves use `move_object` and fail into occupied/invalid cells.
  - **no_agent_interference=true**: ghosting; moves use `ghost_move_object` and ignore agent collisions.
    (`mettagrid/src/metta/mettagrid/actions/move.hpp`; flag in `mettagrid/src/metta/mettagrid/mettagrid_config.py`)
- **Orientation indices** (action args for `move`/`rotate`): 0=N, 1=S, 2=W, 3=E; with diagonals enabled also 4=NW, 5=NE,
  6=SW, 7=SE. Max arg is 3 or 7 accordingly. (`mettagrid/src/metta/mettagrid/actions/orientation.hpp`)

### Available actions and argument semantics

- **noop**
  - **arg**: 0
  - Does nothing. (`mettagrid/src/metta/mettagrid/actions/noop.hpp`)

- **move**
  - **arg**: orientation index (see above). Max 3 (4-way) or 7 (8-way).
  - Sets orientation to arg; attempts to move one cell in that direction. Respects `no_agent_interference`.
    (`mettagrid/src/metta/mettagrid/actions/move.hpp`)

- **rotate**
  - **arg**: orientation index (same index set and max as `move`).
  - Sets orientation only; when `track_movement_metrics` is enabled, tracks rotation stats and sequential rotations.
    (`mettagrid/src/metta/mettagrid/actions/rotate.hpp`)

- **attack**
  - **arg**: 0..8 (selects among potential targets scanned ahead of the agent).
  - Scan pattern depends on diagonals:
    - 4-way: scans a forward 3×3 frustum (three rows ahead; center and side offsets), picks the arg-th seen target.
    - 8-way: uses a diagonal-aware pattern (enabled when `allow_diagonals=true`).
  - Uses configured `consumed_resources` and considers `defense_resources` (via group/action configs).
    (`mettagrid/src/metta/mettagrid/actions/attack.hpp`)

- **get_items**
  - **arg**: 0
  - Interacts with the object in front:
    - If facing a Converter: pulls available output items into the agent inventory; success only if at least one item
      taken.
    - If facing a Box: non-creator can “open” it; creator is refunded the creation resources; box is teleported away and
      `box.opened` is logged. (`mettagrid/src/metta/mettagrid/actions/get_output.hpp`)

- **put_items**
  - **arg**: 0
  - If facing a Converter: transfers as many required input resources from agent to the converter; logs `<resource>.put`
    stats; success if any transferred. (`mettagrid/src/metta/mettagrid/actions/put_recipe_items.hpp`)

- **place_box**
  - **arg**: 0
  - If the front cell is empty and the agent has the required creation resources (from the configured
    `BoxConfig.resources_to_create`), consumes them and places/moves the agent’s box there; logs `box.created`.
    (`mettagrid/src/metta/mettagrid/actions/place_box.hpp`)

- **swap**
  - **arg**: 0
  - Swaps positions with a swappable target in front (checks object layer first, then agent layer); logs
    `action.swap.<type_name>`. (`mettagrid/src/metta/mettagrid/actions/swap.hpp`)

- **change_color**
  - **arg**: 0..3
  - 0=+1, 1=−1, 2=+large step, 3=−large step (step≈255/4). Wraps in uint8.
    (`mettagrid/src/metta/mettagrid/actions/change_color.hpp`)

- **change_glyph**
  - **arg**: 0..(number_of_glyphs−1)
  - Sets the agent’s glyph to the given index. (`mettagrid/src/metta/mettagrid/actions/change_glyph.hpp`)

### Action registration and indices

- The active action set and their order come from `GameConfig.actions`. The order determines the action type indices
  exposed to policies.
- Each action also exposes `max_arg` (0 for arg-less actions). The environment provides `action_names()` and
  `max_action_args()` for runtime introspection. (registration: `mettagrid/src/metta/mettagrid/mettagrid_c.cpp`; API:
  `mettagrid/src/metta/mettagrid/mettagrid_c.hpp`)

## 7. Groups and Reward Sharing

### Group definitions (configuration) (`mettagrid/src/metta/mettagrid/mettagrid_config.py`)

- `GameConfig.groups`: dict[str, GroupConfig] — Team map; at least one group is required.
- `GroupConfig.id`: int — Numeric team id used at runtime.
- `GroupConfig.sprite`: Optional[int] — Visual identifier.
- `GroupConfig.group_reward_pct`: float in [0, 1] — Fraction of each agent’s per-step reward shared evenly with
  teammates.
- `GroupConfig.props`: AgentConfig — Per-group overrides merged over `GameConfig.agent` (freeze_duration,
  action_failure_penalty, resource_limits, rewards, initial_inventory). (merge:
  `mettagrid/src/metta/mettagrid/mettagrid_c_config.py`)

Prop merge and conversion details: (`mettagrid/src/metta/mettagrid/mettagrid_c_config.py`)

- Group `props` are merged into defaults with nested updates (dict fields updated key-wise).
- Rewards are split into `inventory` and `stats` with optional per-item/per-stat caps; names are converted to resource
  ids.

Agent types per group: (`mettagrid/src/metta/mettagrid/mettagrid_c_config.py`)

- For each group, a C++ `AgentConfig` is constructed with `group_id`, `group_name`, merged props, and
  `group_reward_pct`, and registered as object type `agent.<group_name>`.
- Maps that place `agent.<group_name>` spawn agents belonging to that team. (creation in
  `mettagrid/src/metta/mettagrid/mettagrid_c.cpp`)

### Reward sharing algorithm (runtime) (`mettagrid/src/metta/mettagrid/mettagrid_c.cpp`)

- Group percentages are captured per group during environment construction:
  `_group_reward_pct[group_id] = agent_config->group_reward_pct`.
- Group sizes are initialized from registered agent configs and incremented as agents spawn; used for equal split.
- On each step, after action processing:
  1. Zero `_group_rewards` accumulators.
  2. For each agent with non-zero reward `R`:
     - `group_reward = R * _group_reward_pct[group_id]`
     - Agent reward becomes `R - group_reward`
     - Accumulate `_group_rewards[group_id] += group_reward / group_size[group_id]`
  3. If any agent had non-zero reward, add `_group_rewards[agent.group]` to each agent’s reward.

### Reward sources (pre-sharing)

- Inventory-based rewards: Applied on inventory changes with per-item caps; directly accumulates into agent reward.
  (`mettagrid/src/metta/mettagrid/objects/agent.hpp`)
- Stats-based rewards: Computed from tracked stats each step with optional caps; deltas added to agent reward.
  (`mettagrid/src/metta/mettagrid/objects/agent.hpp`)
- Action failure penalty: Subtracted according to agent config.
  (`mettagrid/src/metta/mettagrid/objects/agent_config.hpp`; configured via
  `mettagrid/src/metta/mettagrid/mettagrid_config.py`)
- After these per-agent updates, group sharing is applied as above. (`mettagrid/src/metta/mettagrid/mettagrid_c.cpp`)

File references:

- Config models: `mettagrid/src/metta/mettagrid/mettagrid_config.py`
- Python→C++ conversion and prop merge: `mettagrid/src/metta/mettagrid/mettagrid_c_config.py`
- Runtime reward sharing and group sizes: `mettagrid/src/metta/mettagrid/mettagrid_c.cpp`
- Reward accumulation from inventory/stats: `mettagrid/src/metta/mettagrid/objects/agent.hpp`

## 8. Legacy environments variable mappings (from old_data.md)

This section maps the cataloged variables to the legacy environments and defaults described in old_data.md. Only values
explicitly shown there are listed; others are unspecified and thus use defaults from `mettagrid_config.py`.

### Memory Evaluations (defaults)

- `game.num_agents`: 1
- `game.agent.rewards.inventory`: heart=1, ore_red=0, battery_red=0
- `game.map_builder`: `_target_ = metta.map.mapgen_ascii.MapGenAscii`, `border_width = 3`
- `game.objects.altar` (ConverterConfig): input_resources: {battery_red: 1}; output_resources: {heart: 1}; max_output:
  1; conversion_ticks: 1; cooldown: 255; initial_resource_count: 0
- `game.objects.mine_red` (ConverterConfig): output_resources: {ore_red: 1}; color: 0; max_output: 1; conversion_ticks:
  1; cooldown: 10; initial_resource_count: 1
- `game.objects.generator_red` (ConverterConfig): input_resources: {ore_red: 1}; output_resources: {battery_red: 1};
  color: 0; max_output: 1; conversion_ticks: 1; cooldown: 10; initial_resource_count: 0

### Navigation Evaluations (defaults)

- `game.num_agents`: 1
- `game.max_steps`: unspecified
- `game.global_obs`: episode_completion_pct=true; last_action=true; last_reward=true; resource_rewards=false
- `game.actions`: put_items.enabled=false; attack.enabled=false; swap.enabled=false; change_color.enabled=false
- `game.agent.rewards.inventory`: heart=0.333
- `game.objects.altar`: initial_resource_count=1
- `game.map_builder`: `_target_ = metta.map.mapgen_ascii.MapGenAscii`, `border_width = 1`

### Navigation Sequence Evaluations (defaults)

- `game.num_agents`: 1
- `game.max_steps`: 700
- `game.map_builder`: `_target_ = metta.map.mapgen_ascii.MapGenAscii`, `border_width = 1`
- `game.agent.rewards.inventory`: heart=1.0
- `game.agent.default_resource_limit`: 100
- `game.agent.freeze_duration`: 0
- `game.objects.altar`: input_resources: {battery_red: 1}; output_resources: {heart: 1}; max_output: 1;
  conversion_ticks: 1; cooldown: 255; initial_resource_count: 0
- `game.objects.mine_red`: output_resources: {ore_red: 1}; color: 0; max_output: 1; conversion_ticks: 1; cooldown: 1;
  initial_resource_count: 1
- `game.objects.generator_red`: input_resources: {ore_red: 1}; output_resources: {battery_red: 3}; color: 0; max_output:
  3; conversion_ticks: 1; cooldown: 10; initial_resource_count: 0

### Object Use Evaluations (defaults)

- `game.max_steps`: unspecified
- `game.agent.rewards.inventory`: heart=1
- `game.map_builder`: `_target_ = metta.map.mapgen.MapGen`; width=11; height=11; border_width=3
- `game.map_builder.root`: type=`metta.map.scenes.mean_distance.MeanDistance`; params: mean_distance=6; objects={}
- `game.objects.altar`: input_resources: {battery_red: 1}; output_resources: {heart: 1}; conversion_ticks: 1;
  initial_resource_count: 1
- `game.objects.mine_red`: output_resources: {ore_red: 1}; conversion_ticks: 1; initial_resource_count: 1
- `game.objects.generator_red`: input_resources: {ore_red: 1}; output_resources: {battery_red: 1}; conversion_ticks: 1;
  initial_resource_count: 1

### Systematic Exploration & Memory Evaluations (defaults)

- `game.num_agents`: 1
- `game.max_steps`: unspecified
- `game.agent.rewards.inventory`: heart=0.333
- `game.objects.altar`: initial_resource_count=1
- `game.map_builder`: `_target_ = metta.map.mapgen_ascii.MapGenAscii`; border_width=1

### Base and training environments

- `mettagrid.yaml` (base): environment-wide defaults; no explicit variable overrides listed in old_data.md.

- `arena/advanced.yaml`:
  - `game.map_builder.root.params.objects`: lab=1, factory=1, temple=1 (adds advanced crafting objects)

- `memory/training/easy.yaml`:
  - `game.num_agents`: 16
  - `game.max_steps`: 45
  - `game.agent.rewards.inventory.heart`: 0.333
  - `game.objects.altar.cooldown`: 255
  - `game.map_builder.instances`: 2

- `navigation/training/cylinder_world.yaml`:
  - `game.map_builder.instance_map.params.dir`: cylinder-world_large/medium/small
  - `game.map_builder.instance_map.params.objects.altar`: range [3, 50]

- `object_use/training/easy_all_objects.yaml`:
  - `game.num_agents`: 4
  - `game.agent.default_resource_limit`: 5
  - `game.agent.rewards.inventory`: ore_red=0.01, battery_red=0.1, laser=0.1, armor=0.1, blueprint=0.1, heart=1
  - Map builder parameters left unspecified (filled by curriculum)
