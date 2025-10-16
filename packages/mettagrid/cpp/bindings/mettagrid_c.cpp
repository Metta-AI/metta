#include "bindings/mettagrid_c.hpp"

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <random>
#include <unordered_set>

#include "actions/action_handler.hpp"
#include "actions/attack.hpp"
#include "actions/change_glyph.hpp"
#include "actions/get_output.hpp"
#include "actions/move.hpp"
#include "actions/noop.hpp"
#include "actions/put_recipe_items.hpp"
#include "actions/resource_mod.hpp"
#include "actions/rotate.hpp"
#include "actions/swap.hpp"
#include "core/event.hpp"
#include "core/grid.hpp"
#include "core/hash.hpp"
#include "core/types.hpp"
#include "objects/agent.hpp"
#include "objects/assembler.hpp"
#include "objects/assembler_config.hpp"
#include "objects/chest.hpp"
#include "objects/constants.hpp"
#include "objects/converter.hpp"
#include "objects/converter_config.hpp"
#include "objects/inventory_config.hpp"
#include "objects/production_handler.hpp"
#include "objects/recipe.hpp"
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
  _obs_encoder = std::make_unique<ObservationEncoder>(resource_names, game_config.recipe_details_obs);

  _event_manager = std::make_unique<EventManager>();
  _stats = std::make_unique<StatsTracker>(&resource_names);

  _event_manager->init(_grid.get());
  _event_manager->event_handlers.insert(
      {EventType::FinishConverting, std::make_unique<ProductionHandler>(_event_manager.get())});
  _event_manager->event_handlers.insert({EventType::CoolDown, std::make_unique<CoolDownHandler>(_event_manager.get())});

  _action_success.resize(num_agents);

  for (const auto& [action_name, action_config] : game_config.actions) {
    if (action_name == "put_items") {
      _action_handlers.push_back(std::make_unique<PutRecipeItems>(*action_config));
    } else if (action_name == "get_items") {
      _action_handlers.push_back(std::make_unique<GetOutput>(*action_config));
    } else if (action_name == "noop") {
      _action_handlers.push_back(std::make_unique<Noop>(*action_config));
    } else if (action_name == "move") {
      _action_handlers.push_back(std::make_unique<Move>(*action_config, &_game_config));
    } else if (action_name == "rotate") {
      _action_handlers.push_back(std::make_unique<Rotate>(*action_config, &_game_config));
    } else if (action_name == "attack") {
      auto attack_config = std::static_pointer_cast<const AttackActionConfig>(action_config);
      _action_handlers.push_back(std::make_unique<Attack>(*attack_config, &_game_config));
    } else if (action_name == "change_glyph") {
      auto change_glyph_config = std::static_pointer_cast<const ChangeGlyphActionConfig>(action_config);
      _action_handlers.push_back(std::make_unique<ChangeGlyph>(*change_glyph_config));
    } else if (action_name == "swap") {
      _action_handlers.push_back(std::make_unique<Swap>(*action_config));
    } else if (action_name == "resource_mod") {
      auto modify_config = std::static_pointer_cast<const ResourceModConfig>(action_config);
      _action_handlers.push_back(std::make_unique<ResourceMod>(*modify_config, action_name));
    } else {
      throw std::runtime_error("Unknown action: " + action_name);
    }
  }
  init_action_handlers();

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

      const ConverterConfig* converter_config = dynamic_cast<const ConverterConfig*>(object_cfg);
      if (converter_config) {
        // Create a new ConverterConfig with the recipe offsets from the observation encoder
        ConverterConfig config_with_offsets(*converter_config);
        config_with_offsets.input_recipe_offset = _obs_encoder->get_input_recipe_offset();
        config_with_offsets.output_recipe_offset = _obs_encoder->get_output_recipe_offset();

        Converter* converter = new Converter(r, c, config_with_offsets);
        _grid->add_object(converter);
        _stats->incr("objects." + cell);
        converter->set_event_manager(_event_manager.get());
        continue;
      }

      const AgentConfig* agent_config = dynamic_cast<const AgentConfig*>(object_cfg);
      if (agent_config) {
        Agent* agent = new Agent(r, c, *agent_config, &resource_names);
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
        // Create a new AssemblerConfig with the recipe offsets from the observation encoder
        AssemblerConfig config_with_offsets(*assembler_config);
        config_with_offsets.input_recipe_offset = _obs_encoder->get_input_recipe_offset();
        config_with_offsets.output_recipe_offset = _obs_encoder->get_output_recipe_offset();
        config_with_offsets.recipe_details_obs = _obs_encoder->recipe_details_obs;

        Assembler* assembler = new Assembler(r, c, config_with_offsets);
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

  // Initialize buffers. The buffers are likely to be re-set by the user anyways,
  // so nothing above should depend on them before this point.
  std::vector<ssize_t> shape;
  shape = {static_cast<ssize_t>(num_agents), static_cast<ssize_t>(_num_observation_tokens), static_cast<ssize_t>(3)};
  auto observations = py::array_t<ObservationType, py::array::c_style>(shape);
  auto terminals =
      py::array_t<TerminalType, py::array::c_style>({static_cast<ssize_t>(num_agents)}, {sizeof(TerminalType)});
  auto truncations =
      py::array_t<TruncationType, py::array::c_style>({static_cast<ssize_t>(num_agents)}, {sizeof(TruncationType)});
  auto rewards = py::array_t<RewardType, py::array::c_style>({static_cast<ssize_t>(num_agents)}, {sizeof(RewardType)});

  set_buffers(observations, terminals, truncations, rewards);

  // Initialize global systems
  if (_game_config.clipper) {
    auto& clipper_cfg = *_game_config.clipper;
    if (clipper_cfg.unclipping_recipes.empty()) {
      throw std::runtime_error("Clipper config provided but unclipping_recipes is empty");
    }
    _clipper = std::make_unique<Clipper>(*_grid,
                                         clipper_cfg.unclipping_recipes,
                                         clipper_cfg.length_scale,
                                         clipper_cfg.cutoff_distance,
                                         clipper_cfg.clip_rate,
                                         _rng);
  }
}

MettaGrid::~MettaGrid() = default;

void MettaGrid::init_action_handlers() {
  _num_action_handlers = _action_handlers.size();
  _max_action_priority = 0;
  _max_action_arg = 0;
  _max_action_args.resize(_action_handlers.size());

  for (size_t i = 0; i < _action_handlers.size(); i++) {
    auto& handler = _action_handlers[i];
    handler->init(_grid.get(), &_rng);
    if (handler->priority > _max_action_priority) {
      _max_action_priority = handler->priority;
    }
    _max_action_args[i] = handler->max_arg();
    if (_max_action_args[i] > _max_action_arg) {
      _max_action_arg = _max_action_args[i];
    }
  }

  build_flat_action_catalog();
}

void MettaGrid::build_flat_action_catalog() {
  _flat_action_map.clear();
  _flat_action_names.clear();
  _action_arg_to_flat.clear();

  size_t total_variants = 0;
  for (unsigned char max_arg : _max_action_args) {
    total_variants += static_cast<size_t>(max_arg) + 1;
  }

  _flat_action_map.reserve(total_variants);
  _flat_action_names.reserve(total_variants);
  _action_arg_to_flat.resize(_action_handlers.size());

  std::unordered_set<std::string> seen_names;
  seen_names.reserve(total_variants);

  for (size_t handler_index = 0; handler_index < _action_handlers.size(); ++handler_index) {
    auto& handler = _action_handlers[handler_index];
    unsigned char max_arg = _max_action_args[handler_index];
    auto& arg_map = _action_arg_to_flat[handler_index];
    arg_map.assign(static_cast<size_t>(max_arg) + 1, -1);

    const unsigned int max_arg_uint = static_cast<unsigned int>(max_arg);
    for (unsigned int raw_arg = 0; raw_arg <= max_arg_uint; ++raw_arg) {
      const ActionArg arg = static_cast<ActionArg>(raw_arg);

      std::string base_name = handler->variant_name(arg);
      if (base_name.empty()) {
        base_name = handler->action_name();
      }

      std::string variant = base_name;
      int suffix = 1;
      while (!seen_names.insert(variant).second) {
        variant = base_name + "_" + std::to_string(suffix++);
      }

      const auto flat_index = static_cast<int>(_flat_action_map.size());
      _flat_action_map.emplace_back(static_cast<ActionType>(handler_index), arg);
      _flat_action_names.emplace_back(std::move(variant));
      arg_map[static_cast<size_t>(arg)] = flat_index;
    }
  }
}

int MettaGrid::flat_action_index(ActionType action, ActionArg arg) const {
  if (action < 0) {
    return -1;
  }
  size_t action_idx = static_cast<size_t>(action);
  if (action_idx >= _action_arg_to_flat.size()) {
    return -1;
  }
  size_t arg_idx = static_cast<size_t>(arg);
  const auto& mapping = _action_arg_to_flat[action_idx];
  if (arg_idx >= mapping.size()) {
    return -1;
  }
  return mapping[arg_idx];
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
                                     ActionType action,
                                     ActionArg action_arg) {
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
  int flat_action = flat_action_index(action, action_arg);

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
    ObservationType action_value = static_cast<ObservationType>(std::max(0, flat_action));
    global_tokens.push_back({ObservationFeature::LastAction, action_value});
    global_tokens.push_back({ObservationFeature::LastActionArg, static_cast<ObservationType>(action_arg)});
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
  auto actions_view = actions.unchecked<2>();
  // auto observation_view = _observations.mutable_unchecked<3>();

  for (size_t idx = 0; idx < _agents.size(); idx++) {
    auto& agent = _agents[idx];
    _compute_observation(
        agent->location.r, agent->location.c, obs_width, obs_height, idx, actions_view(idx, 0), actions_view(idx, 1));
  }
}

void MettaGrid::_handle_invalid_action(size_t agent_idx, const std::string& stat, ActionType type, ActionArg arg) {
  auto& agent = _agents[agent_idx];
  agent->stats.incr(stat);
  agent->stats.incr(stat + "." + std::to_string(type) + "." + std::to_string(arg));
  _action_success[agent_idx] = false;
  *agent->reward -= agent->action_failure_penalty;
}

void MettaGrid::_step(Actions actions) {
  _actions = actions;
  auto actions_view = actions.unchecked<2>();

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
      ActionType action = actions_view(agent_idx, 0);
      ActionArg arg = actions_view(agent_idx, 1);

      if (action < 0 || static_cast<size_t>(action) >= _num_action_handlers) {
        _handle_invalid_action(agent_idx, "action.invalid_type", action, arg);
        continue;
      }
      size_t action_idx = static_cast<size_t>(action);

      auto& handler = _action_handlers[action_idx];
      if (handler->priority != current_priority) {
        continue;
      }

      // Tolerate invalid action arguments
      if (arg < 0 || arg > _max_action_args[action_idx]) {
        _handle_invalid_action(agent_idx, "action.invalid_arg", action, arg);
        continue;
      }

      auto* agent = _agents[agent_idx];
      // handle_action expects a GridObjectId, rather than an agent_id, because of where it does its lookup
      // note that handle_action will assign a penalty for attempting invalid actions as a side effect
      _action_success[agent_idx] = handler->handle_action(*agent, arg);
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
  _compute_observations(actions);

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

py::tuple MettaGrid::reset() {
  if (current_step > 0) {
    throw std::runtime_error("Cannot reset after stepping");
  }

  // Reset visitation counts for all agents (only if enabled)
  if (_global_obs_config.visitation_counts) {
    for (auto& agent : _agents) {
      agent->reset_visitation_counts();
    }
  }

  // Reset all buffers
  // Views are created only for validating types; actual clearing is done via
  // direct memory operations for speed.

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
  std::vector<ssize_t> shape = {static_cast<ssize_t>(_agents.size()), static_cast<ssize_t>(2)};
  auto zero_actions = py::array_t<ActionType, py::array::c_style>(shape);
  std::fill(static_cast<ActionType*>(zero_actions.request().ptr),
            static_cast<ActionType*>(zero_actions.request().ptr) + zero_actions.size(),
            static_cast<ActionType>(0));

  _compute_observations(zero_actions);

  return py::make_tuple(_observations, py::dict());
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
                            const py::array_t<float, py::array::c_style>& rewards) {
  // These are initialized in reset()
  _observations = observations;
  _terminals = terminals;
  _truncations = truncations;
  _rewards = rewards;
  _episode_rewards = py::array_t<float, py::array::c_style>({static_cast<ssize_t>(_rewards.shape(0))}, {sizeof(float)});
  for (size_t i = 0; i < _agents.size(); i++) {
    _agents[i]->init(&_rewards.mutable_unchecked<1>()(i));
  }

  validate_buffers();
}

py::tuple MettaGrid::step(const py::array_t<ActionType, py::array::c_style> actions) {
  auto info = actions.request();
  py::array_t<ActionType, py::array::c_style> converted;

  auto assign_flat = [&](auto& view, size_t agent_idx, ActionType flat_index) {
    if (flat_index < 0 || static_cast<size_t>(flat_index) >= _flat_action_map.size()) {
      view(agent_idx, 0) = -1;
      view(agent_idx, 1) = 0;
      return;
    }
    const auto& mapping = _flat_action_map[static_cast<size_t>(flat_index)];
    view(agent_idx, 0) = mapping.first;
    view(agent_idx, 1) = mapping.second;
  };

  if (info.ndim == 1) {
    if (info.shape[0] != static_cast<ssize_t>(_agents.size())) {
      throw std::runtime_error("actions has the wrong shape");
    }
    auto view = actions.unchecked<1>();
    converted = py::array_t<ActionType, py::array::c_style>(
        py::array::ShapeContainer{static_cast<ssize_t>(_agents.size()), static_cast<ssize_t>(2)});
    auto converted_view = converted.mutable_unchecked<2>();
    for (size_t agent_idx = 0; agent_idx < _agents.size(); ++agent_idx) {
      auto flat_index = view(agent_idx);
      assign_flat(converted_view, agent_idx, flat_index);
    }
    _step(converted);
  } else if (info.ndim == 2) {
    if (info.shape[0] != static_cast<ssize_t>(_agents.size())) {
      throw std::runtime_error("actions has the wrong shape");
    }

    if (info.shape[1] == 1) {
      auto view = actions.unchecked<2>();
      converted = py::array_t<ActionType, py::array::c_style>(
          py::array::ShapeContainer{static_cast<ssize_t>(_agents.size()), static_cast<ssize_t>(2)});
      auto converted_view = converted.mutable_unchecked<2>();
      for (size_t agent_idx = 0; agent_idx < _agents.size(); ++agent_idx) {
        auto flat_index = view(agent_idx, 0);
        assign_flat(converted_view, agent_idx, flat_index);
      }
      _step(converted);
    } else if (info.shape[1] == 2) {
      _step(actions);
    } else {
      throw std::runtime_error("actions has the wrong shape");
    }
  } else {
    throw std::runtime_error("actions has the wrong shape");
  }

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
    obj_dict["type"] = obj->type_id;
    obj_dict["type_id"] = obj->type_id;
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
      obj_dict[py::str(_obs_encoder->feature_names().at(feature.feature_id))] = feature.value;
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
      obj_dict["orientation"] = static_cast<int>(agent->orientation);
      obj_dict["group_id"] = agent->group;
      obj_dict["group_name"] = agent->group_name;
      obj_dict["is_frozen"] = !!agent->frozen;
      obj_dict["freeze_remaining"] = agent->frozen;
      obj_dict["freeze_duration"] = agent->freeze_duration;
      obj_dict["glyph"] = agent->glyph;
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

    if (auto* converter = dynamic_cast<Converter*>(obj)) {
      obj_dict["is_converting"] = converter->converting;
      obj_dict["is_cooling_down"] = converter->cooling_down;
      obj_dict["conversion_duration"] = converter->conversion_ticks;
      obj_dict["cooldown_duration"] = converter->next_cooldown_time();
      obj_dict["output_limit"] = converter->max_output;
      py::dict input_resources_dict;
      for (const auto& [resource, quantity] : converter->input_resources) {
        input_resources_dict[py::int_(resource)] = quantity;
      }
      obj_dict["input_resources"] = input_resources_dict;
      py::dict output_resources_dict;
      for (const auto& [resource, quantity] : converter->output_resources) {
        output_resources_dict[py::int_(resource)] = quantity;
      }
      obj_dict["output_resources"] = output_resources_dict;
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

      // Add current recipe ID (pattern byte)
      obj_dict["current_recipe_id"] = static_cast<int>(assembler->get_agent_pattern_byte());

      // Add current recipe information
      const Recipe* current_recipe = assembler->get_current_recipe();
      if (current_recipe) {
        py::dict input_resources_dict;
        for (const auto& [resource, quantity] : current_recipe->input_resources) {
          input_resources_dict[py::int_(resource)] = quantity;
        }
        obj_dict["current_recipe_inputs"] = input_resources_dict;

        py::dict output_resources_dict;
        for (const auto& [resource, quantity] : current_recipe->output_resources) {
          output_resources_dict[py::int_(resource)] = quantity;
        }
        obj_dict["current_recipe_outputs"] = output_resources_dict;
        obj_dict["current_recipe_cooldown"] = current_recipe->cooldown;
      }

      // Add all recipes information (only non-null recipes)
      const std::vector<std::shared_ptr<Recipe>>& active_recipes =
          assembler->is_clipped ? assembler->unclip_recipes : assembler->recipes;
      py::list recipes_list;

      for (size_t i = 0; i < active_recipes.size(); ++i) {
        if (active_recipes[i]) {
          py::dict recipe_dict;

          py::dict input_resources_dict;
          for (const auto& [resource, quantity] : active_recipes[i]->input_resources) {
            input_resources_dict[py::int_(resource)] = quantity;
          }
          recipe_dict["inputs"] = input_resources_dict;

          py::dict output_resources_dict;
          for (const auto& [resource, quantity] : active_recipes[i]->output_resources) {
            output_resources_dict[py::int_(resource)] = quantity;
          }
          recipe_dict["outputs"] = output_resources_dict;
          recipe_dict["cooldown"] = active_recipes[i]->cooldown;

          recipes_list.append(recipe_dict);
        }
      }
      obj_dict["recipes"] = recipes_list;
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

py::list MettaGrid::action_names() {
  py::list names;
  for (const auto& name : _flat_action_names) {
    names.append(py::str(name));
  }
  return names;
}

GridCoord MettaGrid::map_width() {
  return _grid->width;
}

GridCoord MettaGrid::map_height() {
  return _grid->height;
}

// These should correspond to the features we emit in the observations -- either
// the channel or the feature_id.
py::dict MettaGrid::feature_spec() {
  py::dict feature_spec;
  const auto& names = _obs_encoder->feature_names();
  const auto& normalizations = _obs_encoder->feature_normalizations();

  for (const auto& [feature_id, feature_name] : names) {
    py::dict spec;
    spec["normalization"] = py::float_(normalizations.at(feature_id));
    spec["id"] = py::int_(feature_id);

    // Add tag mapping for the tag feature
    if (feature_name == "tag") {
      py::dict tag_map;
      for (const auto& [tag_id, tag_name] : _game_config.tag_id_map) {
        tag_map[py::int_(tag_id)] = py::str(tag_name);
      }
      spec["values"] = tag_map;
    }

    feature_spec[py::str(feature_name)] = spec;
  }
  return feature_spec;
}

size_t MettaGrid::num_agents() const {
  return _agents.size();
}

py::array_t<float> MettaGrid::get_episode_rewards() {
  return _episode_rewards;
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

py::object MettaGrid::action_space() {
  auto gym = py::module_::import("gymnasium");
  auto spaces = gym.attr("spaces");

  return spaces.attr("Discrete")(py::int_(_flat_action_map.size()));
}

py::object MettaGrid::observation_space() {
  auto gym = py::module_::import("gymnasium");
  auto spaces = gym.attr("spaces");

  auto observation_info = _observations.request();
  auto shape = observation_info.shape;
  auto space_shape = py::tuple(observation_info.ndim - 1);

  for (ssize_t i = 0; i < observation_info.ndim - 1; i++) {
    space_shape[static_cast<size_t>(i)] = shape[static_cast<size_t>(i + 1)];
  }

  ObservationType min_value = std::numeric_limits<ObservationType>::min();  // 0
  ObservationType max_value = std::numeric_limits<ObservationType>::max();  // 255

  // TODO: consider spaces other than "Box". They're more correctly descriptive, but I don't know if
  // that matters to us.
  return spaces.attr("Box")(min_value, max_value, space_shape, py::arg("dtype") = dtype_observations());
}

py::list MettaGrid::action_success_py() {
  return py::cast(_action_success);
}

py::list MettaGrid::max_action_args() {
  return py::cast(_max_action_args);
}

py::list MettaGrid::action_catalog() {
  py::list catalog;
  for (size_t idx = 0; idx < _flat_action_map.size(); ++idx) {
    const auto& mapping = _flat_action_map[idx];
    py::dict entry;
    entry["flat_index"] = py::int_(idx);
    entry["action_id"] = py::int_(mapping.first);
    entry["param"] = py::int_(mapping.second);
    entry["base_name"] = py::str(_action_handlers[static_cast<size_t>(mapping.first)]->action_name());
    entry["variant_name"] = py::str(_flat_action_names[idx]);
    catalog.append(std::move(entry));
  }
  return catalog;
}

py::list MettaGrid::object_type_names_py() {
  return py::cast(object_type_names);
}

py::list MettaGrid::resource_names_py() {
  return py::cast(resource_names);
}

py::none MettaGrid::set_inventory(GridObjectId agent_id,
                                  const std::unordered_map<InventoryItem, InventoryQuantity>& inventory) {
  if (agent_id < num_agents()) {
    this->_agents[agent_id]->set_inventory(inventory);
  }
  return py::none();
}

// Pybind11 module definition
PYBIND11_MODULE(mettagrid_c, m) {
  m.doc() = "MettaGrid environment";  // optional module docstring

  PackedCoordinate::bind_packed_coordinate(m);

  // Bind Recipe near its definition
  bind_recipe(m);

  // MettaGrid class bindings
  py::class_<MettaGrid>(m, "MettaGrid")
      .def(py::init<const GameConfig&, const py::list&, unsigned int>())
      .def("reset", &MettaGrid::reset)
      .def("step", &MettaGrid::step, py::arg("actions").noconvert())
      .def("set_buffers",
           &MettaGrid::set_buffers,
           py::arg("observations").noconvert(),
           py::arg("terminals").noconvert(),
           py::arg("truncations").noconvert(),
           py::arg("rewards").noconvert())
      .def("grid_objects",
           &MettaGrid::grid_objects,
           py::arg("min_row") = -1,
           py::arg("max_row") = -1,
           py::arg("min_col") = -1,
           py::arg("max_col") = -1,
           py::arg("ignore_types") = py::list())
      .def("action_names", &MettaGrid::action_names)
      .def_property_readonly("map_width", &MettaGrid::map_width)
      .def_property_readonly("map_height", &MettaGrid::map_height)
      .def_property_readonly("num_agents", &MettaGrid::num_agents)
      .def("get_episode_rewards", &MettaGrid::get_episode_rewards)
      .def("get_episode_stats", &MettaGrid::get_episode_stats)
      .def_property_readonly("action_space", &MettaGrid::action_space)
      .def_property_readonly("observation_space", &MettaGrid::observation_space)
      .def("action_success", &MettaGrid::action_success_py)
      .def("max_action_args", &MettaGrid::max_action_args)
      .def("action_catalog", &MettaGrid::action_catalog)
      .def("object_type_names", &MettaGrid::object_type_names_py)
      .def("feature_spec", &MettaGrid::feature_spec)
      .def_readonly("obs_width", &MettaGrid::obs_width)
      .def_readonly("obs_height", &MettaGrid::obs_height)
      .def_readonly("max_steps", &MettaGrid::max_steps)
      .def_readonly("current_step", &MettaGrid::current_step)
      .def("resource_names", &MettaGrid::resource_names_py)
      .def_readonly("initial_grid_hash", &MettaGrid::initial_grid_hash)
      .def("set_inventory", &MettaGrid::set_inventory, py::arg("agent_id"), py::arg("inventory"));

  // Expose this so we can cast python WallConfig / AgentConfig / ConverterConfig to a common GridConfig cpp object.
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
  bind_agent_config(m);
  bind_converter_config(m);
  bind_assembler_config(m);
  bind_chest_config(m);
  bind_action_config(m);
  bind_attack_action_config(m);
  bind_change_glyph_action_config(m);
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
