#include "env/mettagrid_engine.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
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
#include "config/mettagrid_config.hpp"
#include "core/event.hpp"
#include "core/grid.hpp"
#include "core/hash.hpp"
#include "core/types.hpp"
#include "env/buffer_views.hpp"
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

namespace mettagrid::env {

namespace {

constexpr ObservationType kEmptyToken = EmptyTokenByte;

}  // namespace

MettaGridEngine::MettaGridEngine(const GameConfig& game_config,
                                 const std::vector<std::vector<std::string>>& map,
                                 unsigned int seed)
    : current_step(0),
      max_steps(game_config.max_steps),
      episode_truncates(game_config.episode_truncates),
      initial_grid_hash(0),
      resource_names(game_config.resource_names),
      _global_obs_config(game_config.global_obs),
      _grid(nullptr),
      _event_manager(nullptr),
      _obs_encoder(nullptr),
      _stats(nullptr),
      _action_handlers(),
      _agents(),
      _num_observation_tokens(game_config.num_observation_tokens),
      _flat_action_map(),
      _flat_action_names(),
      _action_arg_to_flat(),
      _max_action_args(),
      _max_action_arg(0),
      _max_action_priority(0),
      _group_reward_pct(),
      _group_sizes(),
      _group_rewards(),
      _resource_rewards(),
      _action_success(),
      _resource_loss_prob(game_config.resource_loss_prob),
      _inventory_regen_interval(game_config.inventory_regen_interval),
      _track_movement_metrics(game_config.track_movement_metrics),
      _clipper(nullptr),
      _buffers(),
      _rng(seed),
      _seed(seed) {
  if (map.empty() || map.front().empty()) {
    throw std::runtime_error("Map must contain at least one row and one column");
  }

  obs_width = game_config.obs_width;
  obs_height = game_config.obs_height;

  bool observation_size_is_packable =
      obs_width <= PackedCoordinate::MAX_PACKABLE_COORD + 1 && obs_height <= PackedCoordinate::MAX_PACKABLE_COORD + 1;
  if (!observation_size_is_packable) {
    throw std::runtime_error("Observation window size (" + std::to_string(obs_width) + "x" + std::to_string(obs_height) +
                             ") exceeds maximum packable size");
  }

  GridCoord height = static_cast<GridCoord>(map.size());
  GridCoord width = static_cast<GridCoord>(map.front().size());

  for (const auto& row : map) {
    if (row.size() != width) {
      throw std::runtime_error("All map rows must have equal width");
    }
  }

  _grid = std::make_unique<Grid>(height, width);
  _obs_encoder = std::make_unique<ObservationEncoder>(resource_names, game_config.recipe_details_obs);

  _event_manager = std::make_unique<EventManager>();
  _stats = std::make_unique<StatsTracker>(&resource_names);

  _event_manager->init(_grid.get());
  _event_manager->event_handlers.insert(
      {EventType::FinishConverting, std::make_unique<ProductionHandler>(_event_manager.get())});
  _event_manager->event_handlers.insert({EventType::CoolDown, std::make_unique<CoolDownHandler>(_event_manager.get())});

  const unsigned int num_agents = static_cast<unsigned int>(game_config.num_agents);
  _action_success.assign(num_agents, false);

  // Instantiate action handlers from ordered config list
  for (const auto& [action_name, action_config] : game_config.actions) {
    if (action_name == "put_items") {
      _action_handlers.push_back(std::make_unique<PutRecipeItems>(*action_config));
    } else if (action_name == "get_items") {
      _action_handlers.push_back(std::make_unique<GetOutput>(*action_config));
    } else if (action_name == "noop") {
      _action_handlers.push_back(std::make_unique<Noop>(*action_config));
    } else if (action_name == "move") {
      _action_handlers.push_back(std::make_unique<Move>(*action_config, &game_config));
    } else if (action_name == "rotate") {
      _action_handlers.push_back(std::make_unique<Rotate>(*action_config, &game_config));
    } else if (action_name == "attack") {
      auto attack_config = std::static_pointer_cast<const AttackActionConfig>(action_config);
      _action_handlers.push_back(std::make_unique<Attack>(*attack_config, &game_config));
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
  init_action_handlers(game_config);

  object_type_names.resize(game_config.objects.size());

  // Map initialization
  std::string grid_hash_data;
  grid_hash_data.reserve(static_cast<size_t>(height) * static_cast<size_t>(width) * 20);

  for (GridCoord r = 0; r < height; ++r) {
    for (GridCoord c = 0; c < width; ++c) {
      const std::string& cell = map[r][c];
      grid_hash_data += std::to_string(r) + "," + std::to_string(c) + ":" + cell + ";";

      if (cell == "empty" || cell == "." || cell == " ") {
        continue;
      }

      if (!game_config.objects.contains(cell)) {
        throw std::runtime_error("Unknown object type: " + cell);
      }

      const GridObjectConfig* object_cfg = game_config.objects.at(cell).get();

      const WallConfig* wall_config = dynamic_cast<const WallConfig*>(object_cfg);
      if (wall_config) {
        auto* wall = new Wall(r, c, *wall_config);
        _grid->add_object(wall);
        _stats->incr("objects." + cell);
        continue;
      }

      const ConverterConfig* converter_config = dynamic_cast<const ConverterConfig*>(object_cfg);
      if (converter_config) {
        ConverterConfig config_with_offsets(*converter_config);
        config_with_offsets.input_recipe_offset = _obs_encoder->get_input_recipe_offset();
        config_with_offsets.output_recipe_offset = _obs_encoder->get_output_recipe_offset();

        auto* converter = new Converter(r, c, config_with_offsets);
        _grid->add_object(converter);
        _stats->incr("objects." + cell);
        converter->set_event_manager(_event_manager.get());
        continue;
      }

      const AgentConfig* agent_config = dynamic_cast<const AgentConfig*>(object_cfg);
      if (agent_config) {
        auto* agent = new Agent(r, c, *agent_config, &resource_names);
        if (_global_obs_config.visitation_counts) {
          agent->init_visitation_grid(height, width);
        }
        add_agent(agent);
        _group_sizes[agent->group] += 1;
        _stats->incr("objects." + cell);
        continue;
      }

      const AssemblerConfig* assembler_config = dynamic_cast<const AssemblerConfig*>(object_cfg);
      if (assembler_config) {
        AssemblerConfig config_with_offsets(*assembler_config);
        config_with_offsets.input_recipe_offset = _obs_encoder->get_input_recipe_offset();
        config_with_offsets.output_recipe_offset = _obs_encoder->get_output_recipe_offset();
        config_with_offsets.recipe_details_obs = _obs_encoder->recipe_details_obs;

        auto* assembler = new Assembler(r, c, config_with_offsets);
        _grid->add_object(assembler);
        _stats->incr("objects." + cell);
        assembler->set_grid(_grid.get());
        assembler->set_current_timestep_ptr(&current_step);
        continue;
      }

      const ChestConfig* chest_config = dynamic_cast<const ChestConfig*>(object_cfg);
      if (chest_config) {
        auto* chest = new Chest(r, c, *chest_config, _stats.get());
        _grid->add_object(chest);
        _stats->incr("objects." + cell);
        chest->set_grid(_grid.get());
        continue;
      }

      throw std::runtime_error("Unable to create object of type " + cell + " at (" + std::to_string(r) + ", " +
                               std::to_string(c) + ")");
    }
  }

  _group_rewards.resize(_group_sizes.size());
  _action_success.resize(_agents.size());
  initial_grid_hash = wyhash::hash_string(grid_hash_data);

  if (game_config.clipper) {
    auto& clipper_cfg = *game_config.clipper;
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

MettaGridEngine::~MettaGridEngine() = default;

void MettaGridEngine::set_buffers(const BufferSet& buffers) {
  if (_agents.empty()) {
    throw std::runtime_error("Cannot set buffers before agents are initialized");
  }

  _buffers = buffers;
  validate_buffers();

  for (size_t agent_idx = 0; agent_idx < _agents.size(); ++agent_idx) {
    Agent* agent = _agents[agent_idx];
    if (_buffers.rewards.data == nullptr) {
      throw std::runtime_error("Reward buffer is null");
    }
    agent->init(&_buffers.rewards.data[agent_idx]);
  }
}

void MettaGridEngine::validate_buffers() const {
  const size_t num_agents = _agents.size();
  if (num_agents == 0) {
    throw std::runtime_error("No agents configured");
  }

  if (_buffers.observations.data == nullptr) {
    throw std::runtime_error("Observations buffer must be set");
  }
  if (_buffers.observations.num_agents != num_agents) {
    throw std::runtime_error("Observations buffer agent dimension mismatch");
  }
  if (_buffers.observations.tokens_per_agent != _num_observation_tokens) {
    throw std::runtime_error("Observations buffer token dimension mismatch");
  }
  if (_buffers.observations.components_per_token != 3) {
    throw std::runtime_error("Observations buffer component dimension mismatch");
  }

  auto validate_array = [&](const ArrayView<bool>& view, const char* name) {
    if (view.data == nullptr || view.size != num_agents) {
      throw std::runtime_error(std::string(name) + " buffer has invalid shape");
    }
  };

  auto validate_rewards = [&](const ArrayView<RewardType>& view, const char* name) {
    if (view.data == nullptr || view.size != num_agents) {
      throw std::runtime_error(std::string(name) + " buffer has invalid shape");
    }
  };

  validate_array(_buffers.terminals, "terminals");
  validate_array(_buffers.truncations, "truncations");
  validate_rewards(_buffers.rewards, "rewards");
  validate_rewards(_buffers.episode_rewards, "episode_rewards");
}

void MettaGridEngine::reset() {
  if (current_step > 0) {
    throw std::runtime_error("Cannot reset after stepping");
  }

  validate_buffers();

  if (_global_obs_config.visitation_counts) {
    for (auto* agent : _agents) {
      agent->reset_visitation_counts();
    }
  }

  _buffers.terminals.fill(false);
  _buffers.truncations.fill(false);
  _buffers.episode_rewards.fill(0.0f);
  _buffers.rewards.fill(0.0f);
  std::fill_n(_buffers.observations.data, _buffers.observations.total_elements(), kEmptyToken);
  std::fill(_action_success.begin(), _action_success.end(), false);

  std::vector<ActionType> zero_actions(_agents.size() * 2, 0);
  ActionMatrixView view{zero_actions.data(), _agents.size()};
  compute_observations(view);
}

void MettaGridEngine::step(ActionMatrixView actions) {
  if (actions.data == nullptr || actions.num_agents != _agents.size()) {
    throw std::runtime_error("Actions buffer shape mismatch");
  }
  run_step(actions);
}

void MettaGridEngine::init_action_handlers(const GameConfig& game_config) {
  _num_action_handlers = _action_handlers.size();
  _max_action_priority = 0;
  _max_action_arg = 0;
  _max_action_args.resize(_action_handlers.size());

  for (size_t i = 0; i < _action_handlers.size(); ++i) {
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

  for (const auto& [key, object_cfg] : game_config.objects) {
    TypeId type_id = object_cfg->type_id;
    if (type_id >= object_type_names.size()) {
      object_type_names.resize(type_id + 1);
    }
    if (!object_type_names[type_id].empty() && object_type_names[type_id] != object_cfg->type_name) {
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

  build_flat_action_catalog();
}

void MettaGridEngine::add_agent(Agent* agent) {
  if (agent == nullptr) {
    throw std::runtime_error("Cannot add null agent");
  }

  if (_agents.size() > std::numeric_limits<decltype(agent->agent_id)>::max()) {
    throw std::runtime_error("Too many agents for agent_id type");
  }

  agent->agent_id = static_cast<decltype(agent->agent_id)>(_agents.size());
  _agents.push_back(agent);
  _grid->add_object(agent);
}

void MettaGridEngine::compute_observation(GridCoord observer_row,
                                          GridCoord observer_col,
                                          ObservationCoord observable_width,
                                          ObservationCoord observable_height,
                                          size_t agent_idx,
                                          ActionType action,
                                          ActionArg action_arg) {
  ObservationCoord obs_width_radius = observable_width >> 1;
  ObservationCoord obs_height_radius = observable_height >> 1;

  int r_start = std::max(static_cast<int>(observer_row) - static_cast<int>(obs_height_radius), 0);
  int c_start = std::max(static_cast<int>(observer_col) - static_cast<int>(obs_width_radius), 0);

  int r_end = std::min(static_cast<int>(observer_row) + static_cast<int>(obs_height_radius) + 1,
                       static_cast<int>(_grid->height));
  int c_end =
      std::min(static_cast<int>(observer_col) + static_cast<int>(obs_width_radius) + 1, static_cast<int>(_grid->width));

  size_t attempted_tokens_written = 0;
  size_t tokens_written = 0;

  const size_t capacity = _buffers.observations.tokens_per_agent;
  ObservationToken* agent_tokens_base = _buffers.observations.tokens(agent_idx);
  ObservationTokens agent_tokens(agent_tokens_base, capacity);

  const RewardType last_reward = _buffers.rewards.data[agent_idx];

  std::vector<PartialObservationToken> global_tokens;
  int flat_action = -1;
  if (action >= 0) {
    size_t action_idx = static_cast<size_t>(action);
    if (action_idx < _action_arg_to_flat.size()) {
      const auto& mapping = _action_arg_to_flat[action_idx];
      if (static_cast<size_t>(action_arg) < mapping.size()) {
        flat_action = mapping[static_cast<size_t>(action_arg)];
      }
    }
  }

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
    ObservationType reward_int = static_cast<ObservationType>(std::round(last_reward * 100.0f));
    global_tokens.push_back({ObservationFeature::LastReward, reward_int});
  }

  if (_global_obs_config.visitation_counts) {
    auto& agent = _agents[agent_idx];
    auto visitation_counts = agent->get_visitation_counts();
    for (size_t i = 0; i < 5; i++) {
      global_tokens.push_back(
          {ObservationFeature::VisitationCounts, static_cast<ObservationType>(visitation_counts[i])});
    }
  }

  uint8_t global_location =
      PackedCoordinate::pack(static_cast<uint8_t>(obs_height_radius), static_cast<uint8_t>(obs_width_radius));

  attempted_tokens_written +=
      _obs_encoder->append_tokens_if_room_available(agent_tokens, global_tokens, global_location);
  tokens_written = std::min(attempted_tokens_written, capacity);

  for (const auto& [r_offset, c_offset] : PackedCoordinate::ObservationPattern{observable_height, observable_width}) {
    int r = static_cast<int>(observer_row) + r_offset;
    int c = static_cast<int>(observer_col) + c_offset;

    if (r < r_start || r >= r_end || c < c_start || c >= c_end) {
      continue;
    }

    for (Layer layer = 0; layer < GridLayer::GridLayerCount; layer++) {
      GridLocation object_loc(static_cast<GridCoord>(r), static_cast<GridCoord>(c), layer);
      auto obj = _grid->object_at(object_loc);
      if (!obj) {
        continue;
      }

      ObservationToken* obs_ptr = agent_tokens_base + tokens_written;
      size_t remaining_capacity = (tokens_written < capacity) ? (capacity - tokens_written) : 0;
      if (remaining_capacity == 0) {
        continue;
      }

      ObservationTokens obs_tokens(obs_ptr, remaining_capacity);

      int obs_r = r - static_cast<int>(observer_row) + static_cast<int>(obs_height_radius);
      int obs_c = c - static_cast<int>(observer_col) + static_cast<int>(obs_width_radius);
      uint8_t location = PackedCoordinate::pack(static_cast<uint8_t>(obs_r), static_cast<uint8_t>(obs_c));

      attempted_tokens_written += _obs_encoder->encode_tokens(obj, obs_tokens, location);
      tokens_written = std::min(attempted_tokens_written, capacity);
    }
  }

  _stats->add("tokens_written", tokens_written);
  _stats->add("tokens_dropped", attempted_tokens_written - tokens_written);
  _stats->add("tokens_free_space", capacity - tokens_written);
}

void MettaGridEngine::compute_observations(ActionMatrixView actions) {
  for (size_t idx = 0; idx < _agents.size(); ++idx) {
    auto* agent = _agents[idx];
    const ActionType* action_row = actions.row(idx);
    compute_observation(
        agent->location.r, agent->location.c, obs_width, obs_height, idx, action_row[0], action_row[1]);
  }
}

void MettaGridEngine::run_step(ActionMatrixView actions) {
  validate_buffers();

  std::fill(_buffers.rewards.begin(), _buffers.rewards.end(), 0.0f);
  std::fill_n(_buffers.observations.data, _buffers.observations.total_elements(), kEmptyToken);
  std::fill(_action_success.begin(), _action_success.end(), false);

  current_step++;
  _event_manager->process_events(current_step);

  std::vector<size_t> agent_indices(_agents.size());
  std::iota(agent_indices.begin(), agent_indices.end(), 0);
  std::shuffle(agent_indices.begin(), agent_indices.end(), _rng);

  for (unsigned char offset = 0; offset <= _max_action_priority; offset++) {
    unsigned char current_priority = _max_action_priority - offset;

    for (const auto agent_idx : agent_indices) {
      const ActionType* action_row = actions.row(agent_idx);
      ActionType action = action_row[0];
      ActionArg arg = action_row[1];

      if (action < 0 || static_cast<size_t>(action) >= _num_action_handlers) {
        handle_invalid_action(agent_idx, "action.invalid_type", action, arg);
        continue;
      }
      size_t action_index = static_cast<size_t>(action);

      auto& handler = _action_handlers[action_index];
      if (handler->priority != current_priority) {
        continue;
      }

      if (arg < 0 || arg > _max_action_args[action_index]) {
        handle_invalid_action(agent_idx, "action.invalid_arg", action, arg);
        continue;
      }

      auto* agent = _agents[agent_idx];
      _action_success[agent_idx] = handler->handle_action(*agent, arg);
    }
  }

  for (auto* agent : _agents) {
    if (_resource_loss_prob <= 0.0f) {
      continue;
    }
    const auto inventory_copy = agent->inventory.get();
    for (const auto& [item, qty] : inventory_copy) {
      if (qty <= 0) {
        continue;
      }
      float loss = _resource_loss_prob * qty;
      InventoryDelta lost = static_cast<InventoryDelta>(std::floor(loss));
      if (std::generate_canonical<float, 10>(_rng) < loss - lost) {
        lost += 1;
      }
      if (lost > 0) {
        agent->update_inventory(item, -lost);
      }
    }
  }

  if (_inventory_regen_interval > 0 && current_step % _inventory_regen_interval == 0) {
    for (auto* agent : _agents) {
      if (agent->inventory_regen_amounts.empty()) {
        continue;
      }
      for (const auto& [item, amount] : agent->inventory_regen_amounts) {
        agent->update_inventory(item, amount);
      }
    }
  }

  if (_clipper) {
    _clipper->maybe_clip_new_assembler();
  }

  compute_observations(actions);

  for (auto* agent : _agents) {
    agent->compute_stat_rewards(_stats.get());
  }

  for (size_t i = 0; i < _agents.size(); ++i) {
    _buffers.episode_rewards.data[i] += _buffers.rewards.data[i];
  }

  if (max_steps > 0 && current_step >= max_steps) {
    if (episode_truncates) {
      _buffers.truncations.fill(true);
    } else {
      _buffers.terminals.fill(true);
    }
  }

  std::fill(_group_rewards.begin(), _group_rewards.end(), 0.0f);
  bool share_rewards = false;

  for (size_t agent_idx = 0; agent_idx < _agents.size(); agent_idx++) {
    RewardType& agent_reward = _buffers.rewards.data[agent_idx];
    if (agent_reward == 0.0f) {
      continue;
    }
    share_rewards = true;
    auto& agent = _agents[agent_idx];
    auto group_id = agent->group;

    RewardType group_reward = agent_reward * _group_reward_pct[group_id];
    agent_reward -= group_reward;
    _group_rewards[group_id] += (group_reward / static_cast<RewardType>(_group_sizes[group_id]));
  }

  if (share_rewards) {
    for (size_t agent_idx = 0; agent_idx < _agents.size(); agent_idx++) {
      auto& agent = _agents[agent_idx];
      size_t group_id = static_cast<size_t>(agent->group);
      _buffers.rewards.data[agent_idx] += _group_rewards[group_id];
    }
  }
}

void MettaGridEngine::handle_invalid_action(size_t agent_idx,
                                            const std::string& stat,
                                            ActionType type,
                                            ActionArg arg) {
  auto& agent = _agents[agent_idx];
  agent->stats.incr(stat);
  agent->stats.incr(stat + "." + std::to_string(type) + "." + std::to_string(arg));
  _action_success[agent_idx] = false;
  *agent->reward -= agent->action_failure_penalty;
}

void MettaGridEngine::build_flat_action_catalog() {
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

const Grid& MettaGridEngine::grid() const {
  return *_grid;
}

Grid& MettaGridEngine::mutable_grid() {
  return *_grid;
}

StatsTracker& MettaGridEngine::stats() {
  return *_stats;
}

const StatsTracker& MettaGridEngine::stats() const {
  return *_stats;
}

ObservationCoord MettaGridEngine::map_width() const {
  return _grid ? _grid->width : 0;
}

ObservationCoord MettaGridEngine::map_height() const {
  return _grid ? _grid->height : 0;
}

}  // namespace mettagrid::env
