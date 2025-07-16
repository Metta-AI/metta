# C++ Changes Needed for Global Observation Configuration

To complete the implementation of configurable global observation tokens, the following changes need to be made to the C++ code:

## 1. Update GameConfig struct in `mettagrid_c.hpp`

Add the following fields to the GameConfig struct (around line 46):

```cpp
struct GameConfig {
  int num_agents;
  unsigned int max_steps;
  bool episode_truncates;
  ObservationCoord obs_width;
  ObservationCoord obs_height;
  std::vector<std::string> inventory_item_names;
  unsigned int num_observation_tokens;
  std::map<std::string, std::shared_ptr<ActionConfig>> actions;
  std::map<std::string, std::shared_ptr<GridObjectConfig>> objects;
  
  // Add these new fields for global observation configuration
  bool include_episode_completion_pct = true;
  bool include_last_action = true;
  bool include_last_reward = true;
};
```

## 2. Update GameConfig pybind11 bindings in `mettagrid_c.cpp`

Update the GameConfig constructor and add new fields (around line 915):

```cpp
py::class_<GameConfig>(m, "GameConfig")
    .def(py::init<int,
                  unsigned int,
                  bool,
                  unsigned short,
                  unsigned short,
                  const std::vector<std::string>&,
                  unsigned int,
                  const std::map<std::string, std::shared_ptr<ActionConfig>>&,
                  const std::map<std::string, std::shared_ptr<GridObjectConfig>>&,
                  bool,  // include_episode_completion_pct
                  bool,  // include_last_action
                  bool>(),  // include_last_reward
         py::arg("num_agents"),
         py::arg("max_steps"),
         py::arg("episode_truncates"),
         py::arg("obs_width"),
         py::arg("obs_height"),
         py::arg("inventory_item_names"),
         py::arg("num_observation_tokens"),
         py::arg("actions"),
         py::arg("objects"),
         py::arg("include_episode_completion_pct") = true,
         py::arg("include_last_action") = true,
         py::arg("include_last_reward") = true)
    .def_readwrite("num_agents", &GameConfig::num_agents)
    .def_readwrite("max_steps", &GameConfig::max_steps)
    .def_readwrite("episode_truncates", &GameConfig::episode_truncates)
    .def_readwrite("obs_width", &GameConfig::obs_width)
    .def_readwrite("obs_height", &GameConfig::obs_height)
    .def_readwrite("inventory_item_names", &GameConfig::inventory_item_names)
    .def_readwrite("num_observation_tokens", &GameConfig::num_observation_tokens)
    .def_readwrite("include_episode_completion_pct", &GameConfig::include_episode_completion_pct)
    .def_readwrite("include_last_action", &GameConfig::include_last_action)
    .def_readwrite("include_last_reward", &GameConfig::include_last_reward);
```

## 3. Store configuration in MettaGrid class

Add member variables to MettaGrid class in `mettagrid_c.hpp` (around line 130):

```cpp
private:
  // ... existing members ...
  
  // Global observation configuration
  bool _include_episode_completion_pct;
  bool _include_last_action;
  bool _include_last_reward;
```

Initialize these in the MettaGrid constructor in `mettagrid_c.cpp` (around line 36):

```cpp
MettaGrid::MettaGrid(const GameConfig& cfg, py::list map, unsigned int seed)
    : max_steps(cfg.max_steps),
      episode_truncates(cfg.episode_truncates),
      obs_width(cfg.obs_width),
      obs_height(cfg.obs_height),
      inventory_item_names(cfg.inventory_item_names),
      _num_observation_tokens(cfg.num_observation_tokens),
      _include_episode_completion_pct(cfg.include_episode_completion_pct),
      _include_last_action(cfg.include_last_action),
      _include_last_reward(cfg.include_last_reward) {
```

## 4. Update _compute_observation method

Modify the global tokens section in `mettagrid_c.cpp` (around line 271):

```cpp
// Global tokens
ObservationToken* agent_obs_ptr = reinterpret_cast<ObservationToken*>(observation_view.mutable_data(agent_idx, 0, 0));
ObservationTokens agent_obs_tokens(agent_obs_ptr, observation_view.shape(1) - tokens_written);

std::vector<PartialObservationToken> global_tokens;

if (_include_episode_completion_pct) {
  ObservationType episode_completion_pct = 0;
  if (max_steps > 0) {
    episode_completion_pct = static_cast<ObservationType>(
        std::round((static_cast<float>(current_step) / max_steps) * std::numeric_limits<ObservationType>::max()));
  }
  global_tokens.push_back({ObservationFeature::EpisodeCompletionPct, episode_completion_pct});
}

if (_include_last_action) {
  global_tokens.push_back({ObservationFeature::LastAction, static_cast<ObservationType>(action)});
  global_tokens.push_back({ObservationFeature::LastActionArg, static_cast<ObservationType>(action_arg)});
}

if (_include_last_reward) {
  ObservationType reward_int = static_cast<ObservationType>(std::round(rewards_view(agent_idx) * 100.0f));
  global_tokens.push_back({ObservationFeature::LastReward, reward_int});
}

// Only add global tokens if we have any
if (!global_tokens.empty()) {
  // Global tokens are always at the center of the observation.
  uint8_t global_location =
      PackedCoordinate::pack(static_cast<uint8_t>(obs_height_radius), static_cast<uint8_t>(obs_width_radius));

  attempted_tokens_written +=
      _obs_encoder->append_tokens_if_room_available(agent_obs_tokens, global_tokens, global_location);
  tokens_written = std::min(attempted_tokens_written, static_cast<size_t>(observation_view.shape(1)));
}
```

## 5. Update mettagrid_c.pyi

Add the new fields to the GameConfig class in `mettagrid_c.pyi`:

```python
class GameConfig:
    num_agents: int
    max_steps: int
    episode_truncates: bool
    obs_width: int
    obs_height: int
    inventory_item_names: list[str]
    num_observation_tokens: int
    include_episode_completion_pct: bool
    include_last_action: bool
    include_last_reward: bool
```

## Building the Changes

After making these C++ changes, run:

```bash
uv sync
```

This will rebuild the C++ extension with the new global observation configuration support.