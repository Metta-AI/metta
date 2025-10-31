#include "bindings/mettagrid_c.hpp"

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <random>
#include <unordered_set>
#include <vector>

#include "actions/action_handler.hpp"
#include "actions/attack.hpp"
#include "actions/change_vibe.hpp"
#include "actions/move.hpp"
#include "actions/move_config.hpp"
#include "actions/noop.hpp"
#include "actions/resource_mod.hpp"
#include "core/event.hpp"
#include "core/grid.hpp"
#include "core/hash.hpp"
#include "core/types.hpp"
#include "objects/agent.hpp"
#include "objects/assembler.hpp"
#include "objects/assembler_config.hpp"
#include "objects/chest.hpp"
#include "objects/constants.hpp"
#include "objects/inventory_config.hpp"
#include "objects/protocol.hpp"
#include "objects/wall.hpp"
#include "systems/clipper.hpp"
#include "systems/clipper_config.hpp"
#include "systems/observation_encoder.hpp"
#include "config/observation_features.hpp"
#include "systems/packed_coordinate.hpp"
#include "systems/stats_tracker.hpp"
#include "supervisors/supervisor_bindings.hpp"

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
      _track_movement_metrics(game_config.track_movement_metrics),
      _resource_loss_prob(game_config.resource_loss_prob),
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
      resource_names.size(),
      game_config.protocol_details_obs,
      &resource_names,
      &game_config.feature_ids);

  // Initialize ObservationFeature namespace with feature IDs
  ObservationFeature::Initialize(game_config.feature_ids);

  // Initialize feature_id_to_name map from GameConfig
  for (const auto& [name, id] : game_config.feature_ids) {
    feature_id_to_name[id] = name;
  }

  _event_manager = std::make_unique<EventManager>();
  _stats = std::make_unique<StatsTracker>(&resource_names);

  _event_manager->init(_grid.get());

  _action_success.resize(num_agents);

  init_action_handlers(game_config);

  _init_grid(game_config, map);

  // Initialize buffers and compute initial observations
  _init_buffers(num_agents);

  // Initialize global systems
  if (_game_config.clipper) {
    auto& clipper_cfg = *_game_config.clipper;
    if (clipper_cfg.unclipping_protocols.empty()) {
      throw std::runtime_error("Clipper config provided but unclipping_protocols is empty");
    }
    _clipper = std::make_unique<Clipper>(*_grid,
                                         clipper_cfg.unclipping_protocols,
                                         clipper_cfg.length_scale,
                                         clipper_cfg.cutoff_distance,
                                         clipper_cfg.clip_rate,
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

    const AgentConfig* agent_config = dynamic_cast<const AgentConfig*>(object_cfg.get());
    if (agent_config) {
      unsigned int id = agent_config->group_id;
      _group_sizes[id] = 0;
      _group_reward_pct[id] = agent_config->group_reward_pct;
    }
  }

  // Initialize objects from map
  std::string grid_hash_data;                                        // String to accumulate grid data for hashing
  grid_hash_data.reserve(static_cast<size_t>(height * width * 20));  // Pre-allocate for efficiency

  for (GridCoord r = 0; r < height; r++) {
    for (GridCoord c = 0; c < width; c++) {
      auto py_cell = map[r].cast<py::list>()[c].cast<py::str>();
      auto cell = py_cell.cast<std::string>();

      // Add cell position and type to hash data
      grid_hash_data += std::to_string(r) + "," + std::to_string(c) + ":" + cell + ";";

      // #HardCodedConfig
      if (cell == "empty" || cell == "." || cell == " ") {
        continue;
      }

      if (!game_config.objects.contains(cell)) {
        throw std::runtime_error("Unknown object type: " + cell);
      }

      const GridObjectConfig* object_cfg = game_config.objects.at(cell).get();

      // TODO: replace the dynamic casts with virtual dispatch

      const WallConfig* wall_config = dynamic_cast<const WallConfig*>(object_cfg);
      if (wall_config) {
        Wall* wall = new Wall(r, c, *wall_config);
        _grid->add_object(wall);
        _stats->incr("objects." + cell);
        continue;
      }

      const AgentConfig* agent_config = dynamic_cast<const AgentConfig*>(object_cfg);
      if (agent_config) {
        Agent* agent = new Agent(r, c, *agent_config, &resource_names, &game_config.feature_ids);
        _grid->add_object(agent);
        if (_agents.size() > std::numeric_limits<decltype(agent->agent_id)>::max()) {
          throw std::runtime_error("Too many agents for agent_id type");
        }
        agent->agent_id = static_cast<decltype(agent->agent_id)>(_agents.size());
        // Only initialize visitation grid if visitation counts are enabled
        if (_global_obs_config.visitation_counts) {
          agent->init_visitation_grid(height, width);
        }
        add_agent(agent);
        _group_sizes[agent->group] += 1;
        continue;
      }

      const AssemblerConfig* assembler_config = dynamic_cast<const AssemblerConfig*>(object_cfg);
      if (assembler_config) {
        Assembler* assembler = new Assembler(r, c, *assembler_config);
        _grid->add_object(assembler);
        _stats->incr("objects." + cell);
        assembler->set_grid(_grid.get());
        assembler->set_current_timestep_ptr(&current_step);
        continue;
      }

      const ChestConfig* chest_config = dynamic_cast<const ChestConfig*>(object_cfg);
      if (chest_config) {
        Chest* chest = new Chest(r, c, *chest_config, _stats.get());
        _grid->add_object(chest);
        _stats->incr("objects." + cell);
        chest->set_grid(_grid.get());
        continue;
      }

      throw std::runtime_error("Unable to create object of type " + cell + " at (" + std::to_string(r) + ", " +
                               std::to_string(c) + ")");
    }

    _group_rewards.resize(_group_sizes.size());
  }

  // Use wyhash for deterministic, high-performance grid fingerprinting across platforms
  initial_grid_hash = wyhash::hash_string(grid_hash_data);
}

void MettaGrid::_init_buffers(unsigned int num_agents) {
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
  set_buffers(observations, terminals, truncations, rewards, actions);

  // Reset visitation counts for all agents (only if enabled)
  if (_global_obs_config.visitation_counts) {
    for (auto& agent : _agents) {
      agent->reset_visitation_counts();
    }
  }

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

  // Compute initial observations
  auto zero_actions = py::array_t<ActionType, py::array::c_style>(std::vector<ssize_t>{static_cast<ssize_t>(_agents.size())});
  std::fill(static_cast<ActionType*>(zero_actions.request().ptr),
            static_cast<ActionType*>(zero_actions.request().ptr) + zero_actions.size(),
            static_cast<ActionType>(0));

  _compute_observations(zero_actions);
}

void MettaGrid::init_action_handlers(const GameConfig& game_config) {
  _max_action_priority = 0;

  // Noop
  auto noop = std::make_unique<Noop>(*game_config.actions.at("noop"));
  noop->init(_grid.get(), &_rng);
  if (noop->priority > _max_action_priority) _max_action_priority = noop->priority;
  for (const auto& action : noop->actions()) {
    _action_handlers.push_back(action);
  }
  _action_handler_impl.push_back(std::move(noop));

  // Move
  auto move_config = std::static_pointer_cast<const MoveActionConfig>(game_config.actions.at("move"));
  auto move = std::make_unique<Move>(*move_config, &game_config);
  move->init(_grid.get(), &_rng);
  if (move->priority > _max_action_priority) _max_action_priority = move->priority;
  for (const auto& action : move->actions()) {
    _action_handlers.push_back(action);
  }
  _action_handler_impl.push_back(std::move(move));

  // Attack
  auto attack_config = std::static_pointer_cast<const AttackActionConfig>(game_config.actions.at("attack"));
  auto attack = std::make_unique<Attack>(*attack_config, &game_config);
  attack->init(_grid.get(), &_rng);
  if (attack->priority > _max_action_priority) _max_action_priority = attack->priority;
  for (const auto& action : attack->actions()) {
    _action_handlers.push_back(action);
  }
  _action_handler_impl.push_back(std::move(attack));

  // ChangeVibe
  auto change_vibe_config = std::static_pointer_cast<const ChangeVibeActionConfig>(game_config.actions.at("change_vibe"));
  auto change_vibe = std::make_unique<ChangeVibe>(*change_vibe_config, &game_config);
  change_vibe->init(_grid.get(), &_rng);
  if (change_vibe->priority > _max_action_priority) _max_action_priority = change_vibe->priority;
  for (const auto& action : change_vibe->actions()) {
    _action_handlers.push_back(action);
  }
  _action_handler_impl.push_back(std::move(change_vibe));

  // ResourceMod
  auto resource_mod_config = std::static_pointer_cast<const ResourceModConfig>(game_config.actions.at("resource_mod"));
  auto resource_mod = std::make_unique<ResourceMod>(*resource_mod_config);
  resource_mod->init(_grid.get(), &_rng);
  if (resource_mod->priority > _max_action_priority) _max_action_priority = resource_mod->priority;
  for (const auto& action : resource_mod->actions()) {
    _action_handlers.push_back(action);
  }
  _action_handler_impl.push_back(std::move(resource_mod));
}

void MettaGrid::add_agent(Agent* agent) {
  agent->init(&_rewards.mutable_unchecked<1>()(_agents.size()));
  _agents.push_back(agent);
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
      float fraction = (static_cast<float>(current_step) / static_cast<float>(max_steps));
      episode_completion_pct =
          static_cast<ObservationType>(std::round(fraction * std::numeric_limits<ObservationType>::max()));
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

  // Add visitation counts for this agent
  if (_global_obs_config.visitation_counts) {
    auto& agent = _agents[agent_idx];
    auto visitation_counts = agent->get_visitation_counts();
    for (size_t i = 0; i < 5; i++) {
      global_tokens.push_back(
          {ObservationFeature::VisitationCounts, static_cast<ObservationType>(visitation_counts[i])});
    }
  }

  // Global tokens are always at the center of the observation.
  uint8_t global_location =
      PackedCoordinate::pack(static_cast<uint8_t>(obs_height_radius), static_cast<uint8_t>(obs_width_radius));

  attempted_tokens_written +=
      _obs_encoder->append_tokens_if_room_available(agent_obs_tokens, global_tokens, global_location);
  tokens_written = std::min(attempted_tokens_written, static_cast<size_t>(observation_view.shape(1)));

  // Process locations in increasing manhattan distance order
  for (const auto& [r_offset, c_offset] : PackedCoordinate::ObservationPattern{observable_height, observable_width}) {
    int r = static_cast<int>(observer_row) + r_offset;
    int c = static_cast<int>(observer_col) + c_offset;

    // Skip if outside map bounds
    if (r < r_start || r >= r_end || c < c_start || c >= c_end) {
      continue;
    }

    //  process a single grid location
    for (Layer layer = 0; layer < GridLayer::GridLayerCount; layer++) {
      GridLocation object_loc(static_cast<GridCoord>(r), static_cast<GridCoord>(c), layer);
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
  }

  _stats->add("tokens_written", tokens_written);
  _stats->add("tokens_dropped", attempted_tokens_written - tokens_written);
  _stats->add("tokens_free_space", static_cast<size_t>(observation_view.shape(1)) - tokens_written);
}

void MettaGrid::_compute_observations(const py::array_t<ActionType, py::array::c_style> actions) {
  auto info = actions.request();

  // Actions must be 1D now
  if (info.ndim != 1) {
    throw std::runtime_error("actions must be 1D array");
  }

  auto actions_view = actions.unchecked<1>();
  for (size_t idx = 0; idx < _agents.size(); idx++) {
    auto& agent = _agents[idx];
    ActionType action_idx = actions_view(idx);
    _compute_observation(
        agent->location.r, agent->location.c, obs_width, obs_height, idx, action_idx);
  }
}

void MettaGrid::_handle_invalid_action(size_t agent_idx, const std::string& stat, ActionType type) {
  auto& agent = _agents[agent_idx];
  agent->stats.incr(stat);
  agent->stats.incr(stat + "." + std::to_string(type));
  _action_success[agent_idx] = false;
  *agent->reward -= agent->action_failure_penalty;
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
  _event_manager->process_events(current_step);

  // Create and shuffle agent indices for randomized action order
  std::vector<size_t> agent_indices(_agents.size());
  std::iota(agent_indices.begin(), agent_indices.end(), 0);
  std::shuffle(agent_indices.begin(), agent_indices.end(), _rng);

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
      _action_success[agent_idx] = action.handle(*agent);
    }
  }

  // Handle resource loss
  for (auto* agent : _agents) {
    if (_resource_loss_prob > 0.0f) {
      // For every resource in an agent's inventory, it should disappear with probability _resource_loss_prob
      // Make a real copy of the agent's inventory map to avoid iterator invalidation
      const auto inventory_copy = agent->inventory.get();
      for (const auto& [item, qty] : inventory_copy) {
        if (qty > 0) {
          float loss = _resource_loss_prob * qty;
          InventoryDelta lost = static_cast<InventoryDelta>(std::floor(loss));
          // With probability equal to the fractional part, lose one more
          if (std::generate_canonical<float, 10>(_rng) < loss - lost) {
            lost += 1;
          }

          if (lost > 0) {
            agent->update_inventory(item, -lost);
          }
        }
      }
    }
  }

  // Handle per-agent inventory regeneration (global interval check, per-agent amounts)
  if (_inventory_regen_interval > 0 && current_step % _inventory_regen_interval == 0) {
    for (auto* agent : _agents) {
      if (!agent->inventory_regen_amounts.empty()) {
        for (const auto& [item, amount] : agent->inventory_regen_amounts) {
          agent->update_inventory(item, amount);
        }
      }
    }
  }

  // Apply global systems
  if (_clipper) {
    _clipper->maybe_clip_new_assembler();
  }

  // Compute observations for next step
  _compute_observations(_actions);

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
                            const py::array_t<ActionType, py::array::c_style>& actions
                          ) {
  // These are initialized in reset()
  _observations = observations;
  _terminals = terminals;
  _truncations = truncations;
  _rewards = rewards;
  _actions = actions;
  _episode_rewards = py::array_t<float, py::array::c_style>({static_cast<ssize_t>(_rewards.shape(0))}, {sizeof(float)});
  for (size_t i = 0; i < _agents.size(); i++) {
    _agents[i]->init(&_rewards.mutable_unchecked<1>()(i));
  }

  validate_buffers();
}

py::tuple MettaGrid::step() {
  auto info = _actions.request();

  // Validate that actions array has correct shape
  if (info.ndim != 1) {
    throw std::runtime_error("actions must be 1D array");
  }
  if (info.shape[0] != static_cast<ssize_t>(_agents.size())) {
    throw std::runtime_error("actions has the wrong shape");
  }

  _step();

  auto rewards_view = _rewards.mutable_unchecked<1>();

  // Clear group rewards from previous step
  std::fill(_group_rewards.begin(), _group_rewards.end(), 0.0f);

  bool share_rewards = false;

  for (size_t agent_idx = 0; agent_idx < _agents.size(); agent_idx++) {
    if (rewards_view(agent_idx) != 0.0f) {
      share_rewards = true;
      auto& agent = _agents[agent_idx];
      auto group_id = agent->group;

      RewardType agent_reward = rewards_view(agent_idx);
      RewardType group_reward = agent_reward * _group_reward_pct[group_id];
      rewards_view(agent_idx) = agent_reward - group_reward;

      _group_rewards[group_id] += group_reward / static_cast<RewardType>(_group_sizes[group_id]);
    }
  }

  if (share_rewards) {
    for (size_t agent_idx = 0; agent_idx < _agents.size(); agent_idx++) {
      auto& agent = _agents[agent_idx];
      size_t group_id = static_cast<size_t>(agent->group);
      rewards_view(agent_idx) += _group_rewards[group_id];
    }
  }

  return py::make_tuple(_observations, _rewards, _terminals, _truncations, py::dict());
}

py::dict MettaGrid::grid_objects(int min_row, int max_row, int min_col, int max_col, const py::list& ignore_types) {
  py::dict objects;

  // Determine if bounding box filtering is enabled
  bool use_bounds = (min_row >= 0 && max_row >= 0 && min_col >= 0 && max_col >= 0);

  // Convert ignore_types list (type names) to type IDs for O(1) integer comparison
  std::unordered_set<TypeId> ignore_type_ids;
  for (const auto& item : ignore_types) {
    std::string type_name = item.cast<std::string>();
    // Find the type_id for this type_name
    for (size_t type_id = 0; type_id < object_type_names.size(); ++type_id) {
      if (object_type_names[type_id] == type_name) {
        ignore_type_ids.insert(static_cast<TypeId>(type_id));
        break;
      }
    }
  }
  bool use_type_filter = !ignore_type_ids.empty();

  for (unsigned int obj_id = 1; obj_id < _grid->objects.size(); obj_id++) {
    auto obj = _grid->object(obj_id);
    if (!obj) continue;

    // Filter by type_id if specified (fast integer comparison)
    if (use_type_filter) {
      if (ignore_type_ids.find(obj->type_id) != ignore_type_ids.end()) {
        continue;
      }
    }

    // Filter by bounding box if specified
    if (use_bounds) {
      if (obj->location.r < min_row || obj->location.r >= max_row || obj->location.c < min_col ||
          obj->location.c >= max_col) {
        continue;
      }
    }

    py::dict obj_dict;
    obj_dict["id"] = obj_id;
    obj_dict["type_name"] = object_type_names[obj->type_id];
    // Location here is defined as XYZ coordinates specifically to be used by MettaScope.
    // We define that for location: x is column, y is row, and z is layer.
    // Note: it might be different for matrix computations.
    obj_dict["location"] = py::make_tuple(obj->location.c, obj->location.r, obj->location.layer);
    obj_dict["is_swappable"] = obj->swappable();

    obj_dict["r"] = obj->location.r;          // To remove
    obj_dict["c"] = obj->location.c;          // To remove
    obj_dict["layer"] = obj->location.layer;  // To remove

    // Inject observation features
    auto features = obj->obs_features();
    for (const auto& feature : features) {
      auto feature_name_it = feature_id_to_name.find(feature.feature_id);
      if (feature_name_it != feature_id_to_name.end()) {
        obj_dict[py::str(feature_name_it->second)] = feature.value;
      }
    }

    if (auto* has_inventory = dynamic_cast<HasInventory*>(obj)) {
      py::dict inventory_dict;
      for (const auto& [resource, quantity] : has_inventory->inventory.get()) {
        inventory_dict[py::int_(resource)] = quantity;
      }
      obj_dict["inventory"] = inventory_dict;
    }

    // Inject agent-specific info
    if (auto* agent = dynamic_cast<Agent*>(obj)) {
      obj_dict["group_id"] = agent->group;
      obj_dict["group_name"] = agent->group_name;
      obj_dict["is_frozen"] = !!agent->frozen;
      obj_dict["freeze_remaining"] = agent->frozen;
      obj_dict["freeze_duration"] = agent->freeze_duration;
      obj_dict["vibe"] = agent->vibe;
      obj_dict["agent_id"] = agent->agent_id;
      obj_dict["action_failure_penalty"] = agent->action_failure_penalty;
      obj_dict["current_stat_reward"] = agent->current_stat_reward;
      obj_dict["prev_action_name"] = agent->prev_action_name;
      obj_dict["steps_without_motion"] = agent->steps_without_motion;

      // We made resource limits more complicated than this, and need to review how to expose them.
      // py::dict resource_limits_dict;
      // for (const auto& [resource, quantity] : agent->inventory.limits) {
      //   resource_limits_dict[py::int_(resource)] = quantity;
      // }
      // obj_dict["resource_limits"] = resource_limits_dict;
    }

    // Add assembler-specific info
    if (auto* assembler = dynamic_cast<Assembler*>(obj)) {
      obj_dict["cooldown_remaining"] = assembler->cooldown_remaining();
      obj_dict["cooldown_duration"] = assembler->cooldown_duration;
      obj_dict["cooldown_progress"] = assembler->cooldown_progress();
      obj_dict["is_clipped"] = assembler->is_clipped;
      obj_dict["is_clip_immune"] = assembler->clip_immune;
      obj_dict["uses_count"] = assembler->uses_count;
      obj_dict["max_uses"] = assembler->max_uses;
      obj_dict["allow_partial_usage"] = assembler->allow_partial_usage;
      obj_dict["exhaustion"] = assembler->exhaustion;
      obj_dict["cooldown_multiplier"] = assembler->cooldown_multiplier;

      // Add current protocol ID (pattern byte)
      obj_dict["current_protocol_id"] = static_cast<int>(assembler->get_local_vibe());

      // Add current protocol information
      const Protocol* current_protocol = assembler->get_current_protocol();
      if (current_protocol) {
        py::dict input_resources_dict;
        for (const auto& [resource, quantity] : current_protocol->input_resources) {
          input_resources_dict[py::int_(resource)] = quantity;
        }
        obj_dict["current_protocol_inputs"] = input_resources_dict;

        py::dict output_resources_dict;
        for (const auto& [resource, quantity] : current_protocol->output_resources) {
          output_resources_dict[py::int_(resource)] = quantity;
        }
        obj_dict["current_protocol_outputs"] = output_resources_dict;
        obj_dict["current_protocol_cooldown"] = current_protocol->cooldown;
      }

      // Add all protocols information
      const std::unordered_map<uint64_t, std::shared_ptr<Protocol>>& active_protocols =
          assembler->is_clipped ? assembler->unclip_protocols : assembler->protocols;
      py::list protocols_list;

      for (const auto& [vibe, protocol] : active_protocols) {
        py::dict protocol_dict;

        py::dict input_resources_dict;
        for (const auto& [resource, quantity] : protocol->input_resources) {
          input_resources_dict[py::int_(resource)] = quantity;
        }
        protocol_dict["inputs"] = input_resources_dict;

        py::dict output_resources_dict;
        for (const auto& [resource, quantity] : protocol->output_resources) {
          output_resources_dict[py::int_(resource)] = quantity;
        }
        protocol_dict["outputs"] = output_resources_dict;
        protocol_dict["cooldown"] = protocol->cooldown;

        protocols_list.append(protocol_dict);
      }
      obj_dict["protocols"] = protocols_list;
    }

    // Add chest-specific info
    if (auto* chest = dynamic_cast<Chest*>(obj)) {
      obj_dict["resource_type"] = static_cast<int>(chest->resource_type);
      obj_dict["max_inventory"] = chest->max_inventory;

      // Convert position_deltas map to dict
      py::dict position_deltas_dict;
      for (const auto& [pos, delta] : chest->position_deltas) {
        position_deltas_dict[py::int_(pos)] = delta;
      }
      obj_dict["position_deltas"] = position_deltas_dict;
    }

    objects[py::int_(obj_id)] = obj_dict;
  }

  return objects;
}

GridCoord MettaGrid::map_width() {
  return _grid->width;
}

GridCoord MettaGrid::map_height() {
  return _grid->height;
}

py::array_t<float> MettaGrid::get_episode_rewards() {
  return _episode_rewards;
}


py::array_t<ActionType> MettaGrid::actions() {
  return _actions;
}

py::dict MettaGrid::get_episode_stats() {
  // Returns a dictionary with the following structure:
  // {
  //   "game": dict[str, float],  // Global game statistics
  //   "agent": list[dict[str, float]],  // Per-agent statistics
  // }

  py::dict stats;
  stats["game"] = py::cast(_stats->to_dict());

  py::list agent_stats;
  for (const auto& agent : _agents) {
    agent_stats.append(py::cast(agent->stats.to_dict()));
  }
  stats["agent"] = agent_stats;

  return stats;
}

py::list MettaGrid::action_success_py() {
  return py::cast(_action_success);
}


py::none MettaGrid::set_inventory(GridObjectId agent_id,
                                  const std::unordered_map<InventoryItem, InventoryQuantity>& inventory) {
  if (agent_id < _agents.size()) {
    this->_agents[agent_id]->set_inventory(inventory);
  }
  return py::none();
}

py::array_t<ObservationType> MettaGrid::observations() {
  return _observations;
}

py::array_t<TerminalType> MettaGrid::terminals() {
  return _terminals;
}


py::array_t<TruncationType> MettaGrid::truncations() {
  return _truncations;
}

py::array_t<RewardType> MettaGrid::rewards() {
  return _rewards;
}

py::array_t<MaskType> MettaGrid::masks() {
  // Return action masks - currently not computed, return empty array
  // TODO: Implement proper action masking if needed
  auto result = py::array_t<MaskType>({static_cast<py::ssize_t>(_agents.size()), static_cast<py::ssize_t>(_action_handlers.size())});
  auto r = result.template mutable_unchecked<2>();
  for (py::ssize_t i = 0; i < r.shape(0); i++) {
    for (py::ssize_t j = 0; j < r.shape(1); j++) {
      r(i, j) = 1;  // All actions available by default
    }
  }
  return result;
}

// Pybind11 module definition
PYBIND11_MODULE(mettagrid_c, m) {
  m.doc() = "MettaGrid environment";  // optional module docstring

  PackedCoordinate::bind_packed_coordinate(m);

  // Bind Protocol near its definition
  bind_protocol(m);

  // MettaGrid class bindings
  py::class_<MettaGrid>(m, "MettaGrid")
      .def(py::init<const GameConfig&, const py::list&, unsigned int>())
      .def("step", &MettaGrid::step)
      .def("set_buffers",
           &MettaGrid::set_buffers,
           py::arg("observations").noconvert(),
           py::arg("terminals").noconvert(),
           py::arg("truncations").noconvert(),
           py::arg("rewards").noconvert(),
           py::arg("actions").noconvert())
      .def("grid_objects",
           &MettaGrid::grid_objects,
           py::arg("min_row") = -1,
           py::arg("max_row") = -1,
           py::arg("min_col") = -1,
           py::arg("max_col") = -1,
           py::arg("ignore_types") = py::list())
      .def("observations", &MettaGrid::observations)
      .def("terminals", &MettaGrid::terminals)
      .def("truncations", &MettaGrid::truncations)
      .def("rewards", &MettaGrid::rewards)
      .def("masks", &MettaGrid::masks)
      .def("actions", &MettaGrid::actions)
      .def_property_readonly("map_width", &MettaGrid::map_width)
      .def_property_readonly("map_height", &MettaGrid::map_height)
      .def("get_episode_rewards", &MettaGrid::get_episode_rewards)
      .def("get_episode_stats", &MettaGrid::get_episode_stats)
      .def("action_success", &MettaGrid::action_success_py)
      .def_readonly("obs_width", &MettaGrid::obs_width)
      .def_readonly("obs_height", &MettaGrid::obs_height)
      .def_readonly("max_steps", &MettaGrid::max_steps)
      .def_readonly("current_step", &MettaGrid::current_step)
      .def_readonly("object_type_names", &MettaGrid::object_type_names)
      .def_readonly("resource_names", &MettaGrid::resource_names)
      .def_readonly("initial_grid_hash", &MettaGrid::initial_grid_hash)
      .def("set_inventory", &MettaGrid::set_inventory, py::arg("agent_id"), py::arg("inventory"));

  // Expose this so we can cast python WallConfig / AgentConfig to a common GridConfig cpp object.
  py::class_<GridObjectConfig, std::shared_ptr<GridObjectConfig>>(m, "GridObjectConfig");

  bind_wall_config(m);

  // ##MettaGridConfig
  // We expose these as much as we can to Python. Defining the initializer (and the object's constructor) means
  // we can create these in Python as AgentConfig(**agent_config_dict). And then we expose the fields individually.
  // This is verbose! But it seems like it's the best way to do it.
  //
  // We use shared_ptr because we expect to effectively have multiple python objects wrapping the same C++ object.
  // This comes from us creating (e.g.) various config objects, and then storing them in GameConfig's maps.
  // We're, like 80% sure on this reasoning.

  bind_inventory_config(m);
  bind_supervisor_configs(m);  // Must be before bind_agent_config since AgentConfig uses AgentSupervisorConfig
  bind_agent_config(m);
  bind_assembler_config(m);
  bind_chest_config(m);
  bind_action_config(m);
  bind_attack_action_config(m);
  bind_change_vibe_action_config(m);
  bind_move_action_config(m);
  bind_resource_mod_config(m);
  bind_global_obs_config(m);
  bind_clipper_config(m);
  bind_game_config(m);

  // Export data types from types.hpp
  m.attr("dtype_observations") = dtype_observations();
  m.attr("dtype_terminals") = dtype_terminals();
  m.attr("dtype_truncations") = dtype_truncations();
  m.attr("dtype_rewards") = dtype_rewards();
  m.attr("dtype_actions") = dtype_actions();
  m.attr("dtype_masks") = dtype_masks();
  m.attr("dtype_success") = dtype_success();

#ifdef METTA_WITH_RAYLIB
  py::class_<HermesPy>(m, "Hermes")
      .def(py::init<>())
      .def("update", &HermesPy::update, py::arg("env"))
      .def("render", &HermesPy::render);
#endif
}
