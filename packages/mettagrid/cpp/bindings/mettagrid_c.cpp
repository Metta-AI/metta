#include "bindings/mettagrid_c.hpp"

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <unordered_set>
#include <vector>

#include "actions/action_handler.hpp"
#include "actions/action_handler_factory.hpp"
#include "actions/attack.hpp"
#include "actions/change_vibe.hpp"
#include "actions/move_config.hpp"
#include "actions/transfer.hpp"
#include "core/grid_object_factory.hpp"
#include "handler/handler_bindings.hpp"
#include "config/observation_features.hpp"
#include "core/aoe_bindings.hpp"
#include "core/grid.hpp"
#include "core/types.hpp"
#include "objects/agent.hpp"
#include "objects/alignable.hpp"
#include "objects/assembler.hpp"
#include "objects/assembler_config.hpp"
#include "objects/chest.hpp"
#include "objects/collective.hpp"
#include "objects/collective_config.hpp"
#include "objects/constants.hpp"
#include "objects/inventory_config.hpp"
#include "objects/protocol.hpp"
#include "objects/wall.hpp"
#include "systems/clipper.hpp"
#include "systems/clipper_config.hpp"
#include "systems/observation_encoder.hpp"
#include "systems/packed_coordinate.hpp"
#include "systems/stats_tracker.hpp"

namespace py = pybind11;

MettaGrid::MettaGrid(const GameConfig& game_config, const py::list map, unsigned int seed)
    : obs_width(game_config.obs_width),
      obs_height(game_config.obs_height),
      max_steps(game_config.max_steps),
      episode_truncates(game_config.episode_truncates),
      resource_names(game_config.resource_names),
      _global_obs_config(game_config.global_obs),
      _game_config(game_config),
      _num_observation_tokens(game_config.num_observation_tokens),
      _inventory_regen_interval(game_config.inventory_regen_interval) {
  _seed = seed;
  _rng = std::mt19937(seed);

  // `map` is a list of lists of strings, which are the map cells.

  unsigned int num_agents = static_cast<unsigned int>(game_config.num_agents);

  current_step = 0;

  bool observation_size_is_packable =
      obs_width <= PackedCoordinate::MAX_PACKABLE_COORD + 1 && obs_height <= PackedCoordinate::MAX_PACKABLE_COORD + 1;
  if (!observation_size_is_packable) {
    throw std::runtime_error("Observation window size (" + std::to_string(obs_width) + "x" +
                             std::to_string(obs_height) + ") exceeds maximum packable size");
  }

  GridCoord height = static_cast<GridCoord>(py::len(map));
  GridCoord width = static_cast<GridCoord>(py::len(map[0]));

  _grid = std::make_unique<Grid>(height, width);
  _obs_encoder = std::make_unique<ObservationEncoder>(
      game_config.protocol_details_obs, resource_names, game_config.feature_ids, game_config.token_value_base);

  // Initialize ObservationFeature namespace with feature IDs
  ObservationFeature::Initialize(game_config.feature_ids);

  // Initialize feature_id_to_name map from GameConfig
  for (const auto& [name, id] : game_config.feature_ids) {
    feature_id_to_name[id] = name;
  }

  _stats = std::make_unique<StatsTracker>(&resource_names);

  _action_success.resize(num_agents);

  init_action_handlers();

  _init_grid(game_config, map);

  // Initialize collectives from config
  for (const auto& [name, collective_cfg] : game_config.collectives) {
    auto collective = std::make_unique<Collective>(*collective_cfg, &resource_names);
    _collectives_by_name[name] = collective.get();
    _collectives.push_back(std::move(collective));
  }

  // Associate alignable objects with their collective based on tags
  // Tags of the form "collective:name" indicate membership
  // Only objects that implement Alignable can belong to a collective
  const std::string collective_tag_prefix = "collective:";
  for (unsigned int obj_id = 1; obj_id < _grid->objects.size(); obj_id++) {
    auto obj = _grid->object(obj_id);
    if (!obj) continue;

    // Try to cast to Alignable - only alignable objects can have a collective
    Alignable* alignable = dynamic_cast<Alignable*>(obj);
    if (!alignable) continue;

    // Check for collective tags
    for (int tag_id : obj->tag_ids) {
      auto tag_it = game_config.tag_id_map.find(tag_id);
      if (tag_it != game_config.tag_id_map.end()) {
        const std::string& tag_name = tag_it->second;
        if (tag_name.rfind(collective_tag_prefix, 0) == 0) {
          // Extract collective name from tag
          std::string collective_name = tag_name.substr(collective_tag_prefix.length());
          auto collective_it = _collectives_by_name.find(collective_name);
          if (collective_it != _collectives_by_name.end()) {
            alignable->setCollective(collective_it->second);
          }
        }
      }
    }
  }

  // Pre-compute goal_obs tokens for each agent
  if (_global_obs_config.goal_obs) {
    _agent_goal_obs_tokens.resize(_agents.size());
    for (size_t i = 0; i < _agents.size(); i++) {
      _compute_agent_goal_obs_tokens(i);
    }
  }

  // Create buffers
  _make_buffers(num_agents);

  // Initialize global systems
  if (_game_config.clipper) {
    auto& clipper_cfg = *_game_config.clipper;
    if (clipper_cfg.unclipping_protocols.empty()) {
      throw std::runtime_error("Clipper config provided but unclipping_protocols is empty");
    }
    _clipper = std::make_unique<Clipper>(*_grid,
                                         clipper_cfg.unclipping_protocols,
                                         clipper_cfg.length_scale,
                                         clipper_cfg.scaled_cutoff_distance,
                                         clipper_cfg.clip_period,
                                         _rng);
  }
}

MettaGrid::~MettaGrid() = default;

void MettaGrid::_init_grid(const GameConfig& game_config, const py::list& map) {
  GridCoord height = static_cast<GridCoord>(py::len(map));
  GridCoord width = static_cast<GridCoord>(py::len(map[0]));

  object_type_names.resize(game_config.objects.size());

  for (const auto& [key, object_cfg] : game_config.objects) {
    TypeId type_id = object_cfg->type_id;

    if (type_id >= object_type_names.size()) {
      // Sometimes the type_ids are not contiguous, so we need to resize the vector.
      object_type_names.resize(type_id + 1);
    }

    if (object_type_names[type_id] != "" && object_type_names[type_id] != object_cfg->type_name) {
      throw std::runtime_error("Object type_id " + std::to_string(type_id) + " already exists with type_name " +
                               object_type_names[type_id] + ". Trying to add " + object_cfg->type_name + ".");
    }
    object_type_names[type_id] = object_cfg->type_name;
  }

  // Initialize objects from map
  for (GridCoord r = 0; r < height; r++) {
    for (GridCoord c = 0; c < width; c++) {
      auto py_cell = map[r].cast<py::list>()[c].cast<py::str>();
      auto cell = py_cell.cast<std::string>();

      // #HardCodedConfig
      if (cell == "empty" || cell == "." || cell == " ") {
        continue;
      }

      if (!game_config.objects.contains(cell)) {
        throw std::runtime_error("Unknown object type: " + cell);
      }

      const GridObjectConfig* object_cfg = game_config.objects.at(cell).get();

      // Create object from config using the factory
      GridObject* created_object = mettagrid::create_object_from_config(
          r, c, object_cfg, _stats.get(), &resource_names, _grid.get(), _obs_encoder.get(), &current_step);

      // Add to grid and track stats
      _grid->add_object(created_object);
      _stats->incr("objects." + cell);

      // Handle agent-specific setup (agent_id and registration)
      if (Agent* agent = dynamic_cast<Agent*>(created_object)) {
        if (_agents.size() > std::numeric_limits<decltype(agent->agent_id)>::max()) {
          throw std::runtime_error("Too many agents for agent_id type");
        }
        agent->agent_id = static_cast<decltype(agent->agent_id)>(_agents.size());
        add_agent(agent);
      }
    }
  }
}

void MettaGrid::_make_buffers(unsigned int num_agents) {
  // Create and set buffers
  std::vector<ssize_t> shape;
  shape = {static_cast<ssize_t>(num_agents), static_cast<ssize_t>(_num_observation_tokens), static_cast<ssize_t>(3)};
  auto observations = py::array_t<ObservationType, py::array::c_style>(shape);
  auto terminals =
      py::array_t<TerminalType, py::array::c_style>({static_cast<ssize_t>(num_agents)}, {sizeof(TerminalType)});
  auto truncations =
      py::array_t<TruncationType, py::array::c_style>({static_cast<ssize_t>(num_agents)}, {sizeof(TruncationType)});
  auto rewards = py::array_t<RewardType, py::array::c_style>({static_cast<ssize_t>(num_agents)}, {sizeof(RewardType)});
  auto actions = py::array_t<ActionType, py::array::c_style>(std::vector<ssize_t>{static_cast<ssize_t>(num_agents)});
  this->_episode_rewards =
      py::array_t<float, py::array::c_style>({static_cast<ssize_t>(num_agents)}, {sizeof(RewardType)});

  set_buffers(observations, terminals, truncations, rewards, actions);
}

void MettaGrid::_init_buffers(unsigned int num_agents) {
  assert(current_step == 0 && "current_step should be initialized to 0 at the start of _init_buffers");

  // Clear all buffers
  std::fill(static_cast<bool*>(_terminals.request().ptr),
            static_cast<bool*>(_terminals.request().ptr) + _terminals.size(),
            0);
  std::fill(static_cast<bool*>(_truncations.request().ptr),
            static_cast<bool*>(_truncations.request().ptr) + _truncations.size(),
            0);
  std::fill(static_cast<float*>(_episode_rewards.request().ptr),
            static_cast<float*>(_episode_rewards.request().ptr) + _episode_rewards.size(),
            0.0f);
  std::fill(
      static_cast<float*>(_rewards.request().ptr), static_cast<float*>(_rewards.request().ptr) + _rewards.size(), 0.0f);

  // Clear observations
  auto obs_ptr = static_cast<uint8_t*>(_observations.request().ptr);
  auto obs_size = _observations.size();
  std::fill(obs_ptr, obs_ptr + obs_size, EmptyTokenByte);

  // Compute initial observations. Every agent starts with a noop.
  std::vector<ActionType> executed_actions(_agents.size());
  std::fill(executed_actions.begin(), executed_actions.end(), ActionType(0));
  _compute_observations(executed_actions);
}

void MettaGrid::init_action_handlers() {
  auto result = create_action_handlers(_game_config, _grid.get(), &_rng);
  _max_action_priority = result.max_priority;
  _action_handlers = std::move(result.actions);
  _action_handler_impl = std::move(result.handlers);
}

void MettaGrid::add_agent(Agent* agent) {
  agent->init(&_rewards.mutable_unchecked<1>()(_agents.size()));
  _agents.push_back(agent);
  if (_global_obs_config.goal_obs) {
    _agent_goal_obs_tokens.resize(_agents.size());
    _compute_agent_goal_obs_tokens(_agents.size() - 1);
  }
}

void MettaGrid::_compute_agent_goal_obs_tokens(size_t agent_idx) {
  auto& agent = _agents[agent_idx];
  std::vector<PartialObservationToken> goal_tokens;

  // Track which resources we've already added goal tokens for
  std::unordered_set<std::string> added_resources;

  // Iterate through stat_rewards to find rewarding resources
  for (const auto& [stat_name, reward_value] : agent->stat_rewards) {
    // Extract resource name from stat name (e.g., "carbon.amount" -> "carbon", "carbon.gained" -> "carbon")
    size_t dot_pos = stat_name.find('.');
    if (dot_pos != std::string::npos) {
      std::string resource_name = stat_name.substr(0, dot_pos);
      // Only add one goal token per resource
      if (added_resources.find(resource_name) == added_resources.end()) {
        // Find the resource index in resource_names
        for (size_t i = 0; i < resource_names.size(); i++) {
          if (resource_names[i] == resource_name) {
            // Get the inventory feature ID for this resource
            ObservationType inventory_feature_id =
                _obs_encoder->get_inventory_feature_id(static_cast<InventoryItem>(i));
            // Add a goal token with the resource's inventory feature ID as the value
            goal_tokens.push_back({ObservationFeature::Goal, inventory_feature_id});
            added_resources.insert(resource_name);
            break;
          }
        }
      }
    }
  }

  _agent_goal_obs_tokens[agent_idx] = std::move(goal_tokens);
}

void MettaGrid::_compute_observation(GridCoord observer_row,
                                     GridCoord observer_col,
                                     ObservationCoord observable_width,
                                     ObservationCoord observable_height,
                                     size_t agent_idx,
                                     ActionType action) {
  // Calculate observation boundaries
  ObservationCoord obs_width_radius = observable_width >> 1;
  ObservationCoord obs_height_radius = observable_height >> 1;

  int r_start = std::max(static_cast<int>(observer_row) - static_cast<int>(obs_height_radius), 0);
  int c_start = std::max(static_cast<int>(observer_col) - static_cast<int>(obs_width_radius), 0);

  int r_end = std::min(static_cast<int>(observer_row) + static_cast<int>(obs_height_radius) + 1,
                       static_cast<int>(_grid->height));
  int c_end =
      std::min(static_cast<int>(observer_col) + static_cast<int>(obs_width_radius) + 1, static_cast<int>(_grid->width));

  const int map_center_r = static_cast<int>(_grid->height) / 2;
  const int map_center_c = static_cast<int>(_grid->width) / 2;

  // Fill in visible objects. Observations should have been cleared in _step, so
  // we don't need to do that here.
  size_t attempted_tokens_written = 0;
  size_t tokens_written = 0;
  auto observation_view = _observations.mutable_unchecked<3>();
  auto rewards_view = _rewards.unchecked<1>();

  // Global tokens
  ObservationToken* agent_obs_ptr = reinterpret_cast<ObservationToken*>(observation_view.mutable_data(agent_idx, 0, 0));
  ObservationTokens agent_obs_tokens(
      agent_obs_ptr, static_cast<size_t>(observation_view.shape(1)) - static_cast<size_t>(tokens_written));

  // Build global tokens based on configuration
  std::vector<PartialObservationToken> global_tokens;

  if (_global_obs_config.episode_completion_pct) {
    ObservationType episode_completion_pct = 0;
    if (max_steps > 0) {
      if (current_step >= max_steps) {
        // The episode should be over, so this observation shouldn't matter. But let's max our for
        // better continuity.
        episode_completion_pct = std::numeric_limits<ObservationType>::max();
      } else {
        episode_completion_pct = static_cast<ObservationType>(
          (static_cast<uint32_t>(std::numeric_limits<ObservationType>::max()) + 1) * current_step / max_steps
        );
      }
    }
    global_tokens.push_back({ObservationFeature::EpisodeCompletionPct, episode_completion_pct});
  }

  if (_global_obs_config.last_action) {
    global_tokens.push_back({ObservationFeature::LastAction, static_cast<ObservationType>(action)});
  }

  if (_global_obs_config.last_reward) {
    ObservationType reward_int = static_cast<ObservationType>(std::round(rewards_view(agent_idx) * 100.0f));
    global_tokens.push_back({ObservationFeature::LastReward, reward_int});
  }

  // Add pre-computed goal tokens for rewarding resources when enabled
  if (_global_obs_config.goal_obs) {
    global_tokens.insert(
        global_tokens.end(), _agent_goal_obs_tokens[agent_idx].begin(), _agent_goal_obs_tokens[agent_idx].end());
  }

  // Global tokens are always at the center of the observation.
  uint8_t global_location =
      PackedCoordinate::pack(static_cast<uint8_t>(obs_height_radius), static_cast<uint8_t>(obs_width_radius));

  attempted_tokens_written +=
      _obs_encoder->append_tokens_if_room_available(agent_obs_tokens, global_tokens, global_location);
  tokens_written = std::min(attempted_tokens_written, static_cast<size_t>(observation_view.shape(1)));

  /*
   * COMPASS TOKEN EMISSION
   * ----------------------
   * Some missions opt in to a lightweight "compass" hint by enabling the global_obs.compass flag.
   * Rather than mutate the world, we inject a synthetic observation token that occupies one of the
   * eight neighbor slots around the agent inside its egocentric window. The location byte alone
   * communicates the direction: it is offset one step toward the assembler hub (which always sits
   * in the map center for CvC missions). The token value is a simple sentinel (currently 1).
   * When the agent is already at the hub there is no direction to emit, and although the offset
   * should always land inside the observation window, we keep the bounds check as a defensive guard.
   */
  if (_global_obs_config.compass) {
    const int delta_r = map_center_r - static_cast<int>(observer_row);
    const int delta_c = map_center_c - static_cast<int>(observer_col);

    int step_r = 0;
    int step_c = 0;
    if (delta_r != 0) {
      step_r = (delta_r > 0) ? 1 : -1;
    }
    if (delta_c != 0) {
      step_c = (delta_c > 0) ? 1 : -1;
    }

    if (step_r != 0 || step_c != 0) {
      int obs_r = static_cast<int>(obs_height_radius) + step_r;
      int obs_c = static_cast<int>(obs_width_radius) + step_c;

      if (obs_r >= 0 && obs_r < static_cast<int>(observable_height) && obs_c >= 0 &&
          obs_c < static_cast<int>(observable_width)) {
        uint8_t compass_location = PackedCoordinate::pack(static_cast<uint8_t>(obs_r), static_cast<uint8_t>(obs_c));

        ObservationToken* compass_ptr =
            reinterpret_cast<ObservationToken*>(observation_view.mutable_data(agent_idx, tokens_written, 0));
        ObservationTokens compass_tokens(
            compass_ptr, static_cast<size_t>(observation_view.shape(1)) - static_cast<size_t>(tokens_written));

        const std::vector<PartialObservationToken> compass_token = {
            {ObservationFeature::Compass, static_cast<ObservationType>(1)}};

        attempted_tokens_written +=
            _obs_encoder->append_tokens_if_room_available(compass_tokens, compass_token, compass_location);
        tokens_written = std::min(attempted_tokens_written, static_cast<size_t>(observation_view.shape(1)));
      }
    }
  }

  // Process locations in increasing manhattan distance order
  for (const auto& [r_offset, c_offset] : PackedCoordinate::ObservationPattern{observable_height, observable_width}) {
    int r = static_cast<int>(observer_row) + r_offset;
    int c = static_cast<int>(observer_col) + c_offset;

    // Skip if outside map bounds
    if (r < r_start || r >= r_end || c < c_start || c >= c_end) {
      continue;
    }

    //  process a single grid location
    GridLocation object_loc(static_cast<GridCoord>(r), static_cast<GridCoord>(c));
    auto obj = _grid->object_at(object_loc);
    if (!obj) {
      continue;
    }

    // Prepare observation buffer for this object
    ObservationToken* obs_ptr =
        reinterpret_cast<ObservationToken*>(observation_view.mutable_data(agent_idx, tokens_written, 0));
    ObservationTokens obs_tokens(
        obs_ptr, static_cast<size_t>(observation_view.shape(1)) - static_cast<size_t>(tokens_written));

    // Calculate position within the observation window (agent is at the center)
    int obs_r = r - static_cast<int>(observer_row) + static_cast<int>(obs_height_radius);
    int obs_c = c - static_cast<int>(observer_col) + static_cast<int>(obs_width_radius);

    // Encode location and add tokens
    uint8_t location = PackedCoordinate::pack(static_cast<uint8_t>(obs_r), static_cast<uint8_t>(obs_c));
    attempted_tokens_written += _obs_encoder->encode_tokens(obj, obs_tokens, location);
    tokens_written = std::min(attempted_tokens_written, static_cast<size_t>(observation_view.shape(1)));
  }

  _stats->add("tokens_written", tokens_written);
  _stats->add("tokens_dropped", attempted_tokens_written - tokens_written);
  _stats->add("tokens_free_space", static_cast<size_t>(observation_view.shape(1)) - tokens_written);
}

void MettaGrid::_compute_observations(const std::vector<ActionType>& executed_actions) {
  for (size_t idx = 0; idx < _agents.size(); idx++) {
    auto& agent = _agents[idx];
    ActionType action_idx = executed_actions[idx];
    _compute_observation(agent->location.r, agent->location.c, obs_width, obs_height, idx, action_idx);
  }
}

void MettaGrid::_handle_invalid_action(size_t agent_idx, const std::string& stat, ActionType type) {
  auto& agent = _agents[agent_idx];
  agent->stats.incr(stat);
  agent->stats.incr(stat + "." + std::to_string(type));
  _action_success[agent_idx] = false;
}

void MettaGrid::_step() {
  auto actions_view = _actions.unchecked<1>();

  // Reset rewards and observations
  auto rewards_view = _rewards.mutable_unchecked<1>();

  std::fill(
      static_cast<float*>(_rewards.request().ptr), static_cast<float*>(_rewards.request().ptr) + _rewards.size(), 0);

  auto obs_ptr = static_cast<ObservationType*>(_observations.request().ptr);
  auto obs_size = _observations.size();
  std::fill(obs_ptr, obs_ptr + obs_size, EmptyTokenByte);

  std::fill(_action_success.begin(), _action_success.end(), false);

  // Increment timestep and process events
  current_step++;

  // Create and shuffle agent indices for randomized action order
  std::vector<size_t> agent_indices(_agents.size());
  std::iota(agent_indices.begin(), agent_indices.end(), 0);
  std::shuffle(agent_indices.begin(), agent_indices.end(), _rng);

  std::vector<ActionType> executed_actions(_agents.size());
  // Fill with noop. Replace this with the actual action if it's successful.
  std::fill(executed_actions.begin(), executed_actions.end(), ActionType(0));
  // Process actions by priority levels (highest to lowest)
  for (unsigned char offset = 0; offset <= _max_action_priority; offset++) {
    unsigned char current_priority = _max_action_priority - offset;

    for (const auto& agent_idx : agent_indices) {
      ActionType action_idx = actions_view(agent_idx);

      if (action_idx < 0 || static_cast<size_t>(action_idx) >= _action_handlers.size()) {
        _handle_invalid_action(agent_idx, "action.invalid_index", action_idx);
        continue;
      }

      Action& action = _action_handlers[static_cast<size_t>(action_idx)];
      if (action.handler()->priority != current_priority) {
        continue;
      }

      auto* agent = _agents[agent_idx];
      bool success = action.handle(*agent);
      _action_success[agent_idx] = success;
      if (success) {
        executed_actions[agent_idx] = action_idx;
      }
    }
  }

  // Handle per-agent inventory regeneration (global interval check, vibe-dependent amounts)
  if (_inventory_regen_interval > 0 && current_step % _inventory_regen_interval == 0) {
    for (auto* agent : _agents) {
      if (!agent->inventory_regen_amounts.empty()) {
        // Look up regen amounts for agent's current vibe, fall back to "default" (vibe ID 0)
        auto vibe_it = agent->inventory_regen_amounts.find(agent->vibe);
        if (vibe_it == agent->inventory_regen_amounts.end()) {
          vibe_it = agent->inventory_regen_amounts.find(0);  // "default" is vibe ID 0
        }
        if (vibe_it != agent->inventory_regen_amounts.end()) {
          for (const auto& [item, amount] : vibe_it->second) {
            agent->inventory.update(item, amount);
          }
        }
      }
    }
  }

  // Check and apply damage for all agents
  for (auto* agent : _agents) {
    agent->check_and_apply_damage(_rng);
  }

  // Apply global systems
  if (_clipper) {
    _clipper->maybe_clip_new_assembler();
  }

  // Compute observations for next step
  _compute_observations(executed_actions);

  // Compute stat-based rewards for all agents
  for (auto& agent : _agents) {
    agent->compute_stat_rewards(_stats.get());
  }

  // Update episode rewards
  auto episode_rewards_view = _episode_rewards.mutable_unchecked<1>();
  for (py::ssize_t i = 0; i < rewards_view.shape(0); i++) {
    episode_rewards_view(i) += rewards_view(i);
  }

  // Check for truncation
  if (max_steps > 0 && current_step >= max_steps) {
    if (episode_truncates) {
      std::fill(static_cast<bool*>(_truncations.request().ptr),
                static_cast<bool*>(_truncations.request().ptr) + _truncations.size(),
                1);
    } else {
      std::fill(static_cast<bool*>(_terminals.request().ptr),
                static_cast<bool*>(_terminals.request().ptr) + _terminals.size(),
                1);
    }
  }
}

void MettaGrid::validate_buffers() {
  // We should validate once buffers and agents are set.
  // data types and contiguity are handled by pybind11. We still need to check
  // shape.
  auto num_agents = _agents.size();
  auto observation_info = _observations.request();
  auto observation_shape = observation_info.shape;
  if (observation_info.ndim != 3) {
    std::stringstream ss;
    ss << "observations has " << observation_info.ndim << " dimensions but expected 3";
    throw std::runtime_error(ss.str());
  }
  if (observation_shape[0] != static_cast<ssize_t>(num_agents) || observation_shape[2] != 3) {
    std::stringstream ss;
    ss << "observations has shape [" << observation_shape[0] << ", " << observation_shape[1] << ", "
       << observation_shape[2] << "] but expected [" << num_agents << ", [something], 3]";
    throw std::runtime_error(ss.str());
  }
  {
    auto terminals_info = _terminals.request();
    auto terminals_shape = terminals_info.shape;
    if (terminals_info.ndim != 1 || terminals_shape[0] != static_cast<ssize_t>(num_agents)) {
      throw std::runtime_error("terminals has the wrong shape");
    }
  }
  {
    auto truncations_info = _truncations.request();
    auto truncations_shape = truncations_info.shape;
    if (truncations_info.ndim != 1 || truncations_shape[0] != static_cast<ssize_t>(num_agents)) {
      throw std::runtime_error("truncations has the wrong shape");
    }
  }
  {
    auto rewards_info = _rewards.request();
    auto rewards_shape = rewards_info.shape;
    if (rewards_info.ndim != 1 || rewards_shape[0] != static_cast<ssize_t>(num_agents)) {
      throw std::runtime_error("rewards has the wrong shape");
    }
  }
}

void MettaGrid::set_buffers(const py::array_t<uint8_t, py::array::c_style>& observations,
                            const py::array_t<bool, py::array::c_style>& terminals,
                            const py::array_t<bool, py::array::c_style>& truncations,
                            const py::array_t<float, py::array::c_style>& rewards,
                            const py::array_t<ActionType, py::array::c_style>& actions) {
  // These are initialized in reset()
  _observations = observations;
  _terminals = terminals;
  _truncations = truncations;
  _rewards = rewards;
  _actions = actions;
  for (size_t i = 0; i < _agents.size(); i++) {
    _agents[i]->init(&_rewards.mutable_unchecked<1>()(i));
  }

  validate_buffers();
  _init_buffers(_agents.size());
}

void MettaGrid::step() {
  auto info = _actions.request();

  // Validate that actions array has correct shape
  if (info.ndim != 1) {
    throw std::runtime_error("actions must be 1D array");
  }
  if (info.shape[0] != static_cast<ssize_t>(_agents.size())) {
    throw std::runtime_error("actions has the wrong shape");
  }

  _step();
}
