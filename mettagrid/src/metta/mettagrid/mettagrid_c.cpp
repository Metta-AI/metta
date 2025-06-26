#include "mettagrid_c.hpp"

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>

#include "action_handler.hpp"
#include "actions/attack.hpp"
#include "actions/attack_nearest.hpp"
#include "actions/change_color.hpp"
#include "actions/get_output.hpp"
#include "actions/move.hpp"
#include "actions/noop.hpp"
#include "actions/put_recipe_items.hpp"
#include "actions/rotate.hpp"
#include "actions/swap.hpp"
#include "event.hpp"
#include "grid.hpp"
#include "hash.hpp"
#include "objects/agent.hpp"
#include "objects/constants.hpp"
#include "objects/converter.hpp"
#include "objects/production_handler.hpp"
#include "objects/wall.hpp"
#include "observation_encoder.hpp"
#include "stats_tracker.hpp"
#include "types.hpp"

namespace py = pybind11;

MettaGrid::MettaGrid(py::dict cfg, py::list map) {
  // cfg is a dict-form of the OmegaConf config.
  // `map` is a list of lists of strings, which are the map cells.

  int num_agents = cfg["num_agents"].cast<int>();
  max_steps = cfg["max_steps"].cast<unsigned int>();
  obs_width = cfg["obs_width"].cast<unsigned short>();
  obs_height = cfg["obs_height"].cast<unsigned short>();
  inventory_item_names = cfg["inventory_item_names"].cast<std::vector<std::string>>();

  _num_observation_tokens =
      cfg.contains("num_observation_tokens") ? cfg["num_observation_tokens"].cast<unsigned int>() : 0;

  current_step = 0;

  std::vector<Layer> layer_for_type_id;
  for (const auto& layer : ObjectLayers) {
    layer_for_type_id.push_back(layer.second);
  }
  int height = map.size();
  int width = map[0].cast<py::list>().size();

  _grid = std::make_unique<Grid>(width, height, layer_for_type_id);
  _obs_encoder = std::make_unique<ObservationEncoder>(inventory_item_names);
  _feature_normalizations = _obs_encoder->feature_normalizations();

  _event_manager = std::make_unique<EventManager>();
  _stats = std::make_unique<StatsTracker>(inventory_item_names);
  _stats->set_environment(this);

  _event_manager->init(_grid.get());
  _event_manager->event_handlers.insert(
      {EventType::FinishConverting, std::make_unique<ProductionHandler>(_event_manager.get())});
  _event_manager->event_handlers.insert({EventType::CoolDown, std::make_unique<CoolDownHandler>(_event_manager.get())});

  _action_success.resize(num_agents);

  // TODO: These conversions to ActionConfig are copying. I don't want to pass python objects down further,
  // but maybe it would be better to just do the conversion once?
  if (cfg["actions"]["put_items"]["enabled"].cast<bool>()) {
    _action_handlers.push_back(std::make_unique<PutRecipeItems>(cfg["actions"]["put_items"].cast<ActionConfig>()));
  }
  if (cfg["actions"]["get_items"]["enabled"].cast<bool>()) {
    _action_handlers.push_back(std::make_unique<GetOutput>(cfg["actions"]["get_items"].cast<ActionConfig>()));
  }
  if (cfg["actions"]["noop"]["enabled"].cast<bool>()) {
    _action_handlers.push_back(std::make_unique<Noop>(cfg["actions"]["noop"].cast<ActionConfig>()));
  }
  if (cfg["actions"]["move"]["enabled"].cast<bool>()) {
    _action_handlers.push_back(std::make_unique<Move>(cfg["actions"]["move"].cast<ActionConfig>()));
  }
  if (cfg["actions"]["rotate"]["enabled"].cast<bool>()) {
    _action_handlers.push_back(std::make_unique<Rotate>(cfg["actions"]["rotate"].cast<ActionConfig>()));
  }
  if (cfg["actions"]["attack"]["enabled"].cast<bool>()) {
    _action_handlers.push_back(std::make_unique<Attack>(cfg["actions"]["attack"].cast<ActionConfig>(),
                                                        cfg["actions"]["laser_item_id"].cast<InventoryItem>(),
                                                        cfg["actions"]["armor_item_id"].cast<InventoryItem>()));
    _action_handlers.push_back(std::make_unique<AttackNearest>(cfg["actions"]["attack"].cast<ActionConfig>(),
                                                               cfg["actions"]["laser_item_id"].cast<InventoryItem>(),
                                                               cfg["actions"]["armor_item_id"].cast<InventoryItem>()));
  }
  if (cfg["actions"]["swap"]["enabled"].cast<bool>()) {
    _action_handlers.push_back(std::make_unique<Swap>(cfg["actions"]["swap"].cast<ActionConfig>()));
  }
  if (cfg["actions"]["change_color"]["enabled"].cast<bool>()) {
    _action_handlers.push_back(
        std::make_unique<ChangeColorAction>(cfg["actions"]["change_color"].cast<ActionConfig>()));
  }
  init_action_handlers();

  auto agent_groups = cfg["agent_groups"].cast<py::dict>();

  for (const auto& [key, value] : agent_groups) {
    auto agent_group = value.cast<py::dict>();
    unsigned int id = agent_group["group_id"].cast<unsigned int>();
    _group_sizes[id] = 0;
    _group_reward_pct[id] = agent_group["group_reward_pct"].cast<float>();
  }

  // Initialize objects from map
  std::string grid_hash_data;                   // String to accumulate grid data for hashing
  grid_hash_data.reserve(height * width * 20);  // Pre-allocate for efficiency

  for (int r = 0; r < height; r++) {
    for (int c = 0; c < width; c++) {
      std::string cell = map[r].cast<py::list>()[c].cast<std::string>();

      // Add cell position and type to hash data
      grid_hash_data += std::to_string(r) + "," + std::to_string(c) + ":" + cell + ";";

      Converter* converter = nullptr;
      if (cell == "wall") {
        Wall* wall = new Wall(r, c, cfg["objects"]["wall"].cast<ObjectConfig>());
        _grid->add_object(wall);
        _stats->incr("objects.wall");
      } else if (cell == "block") {
        Wall* block = new Wall(r, c, cfg["objects"]["block"].cast<ObjectConfig>());
        _grid->add_object(block);
        _stats->incr("objects.block");
      } else if (cell == "mine_red") {
        auto converter_cfg = _create_converter_config(cfg["objects"]["mine_red"]);
        converter = new Converter(r, c, converter_cfg, ObjectType::MineRedT);
      } else if (cell == "mine_blue") {
        auto converter_cfg = _create_converter_config(cfg["objects"]["mine_blue"]);
        converter = new Converter(r, c, converter_cfg, ObjectType::MineBlueT);
      } else if (cell == "mine_green") {
        auto converter_cfg = _create_converter_config(cfg["objects"]["mine_green"]);
        converter = new Converter(r, c, converter_cfg, ObjectType::MineGreenT);
      } else if (cell == "generator_red") {
        auto converter_cfg = _create_converter_config(cfg["objects"]["generator_red"]);
        converter = new Converter(r, c, converter_cfg, ObjectType::GeneratorRedT);
      } else if (cell == "generator_blue") {
        auto converter_cfg = _create_converter_config(cfg["objects"]["generator_blue"]);
        converter = new Converter(r, c, converter_cfg, ObjectType::GeneratorBlueT);
      } else if (cell == "generator_green") {
        auto converter_cfg = _create_converter_config(cfg["objects"]["generator_green"]);
        converter = new Converter(r, c, converter_cfg, ObjectType::GeneratorGreenT);
      } else if (cell == "altar") {
        auto converter_cfg = _create_converter_config(cfg["objects"]["altar"]);
        converter = new Converter(r, c, converter_cfg, ObjectType::AltarT);
      } else if (cell == "armory") {
        auto converter_cfg = _create_converter_config(cfg["objects"]["armory"]);
        converter = new Converter(r, c, converter_cfg, ObjectType::ArmoryT);
      } else if (cell == "lasery") {
        auto converter_cfg = _create_converter_config(cfg["objects"]["lasery"]);
        converter = new Converter(r, c, converter_cfg, ObjectType::LaseryT);
      } else if (cell == "lab") {
        auto converter_cfg = _create_converter_config(cfg["objects"]["lab"]);
        converter = new Converter(r, c, converter_cfg, ObjectType::LabT);
      } else if (cell == "factory") {
        auto converter_cfg = _create_converter_config(cfg["objects"]["factory"]);
        converter = new Converter(r, c, converter_cfg, ObjectType::FactoryT);
      } else if (cell == "temple") {
        auto converter_cfg = _create_converter_config(cfg["objects"]["temple"]);
        converter = new Converter(r, c, converter_cfg, ObjectType::TempleT);
      } else if (cell.starts_with("agent.")) {
        auto agent_group_cfg_py = agent_groups[py::str(cell)].cast<py::dict>();

        Agent* agent = MettaGrid::create_agent(r, c, agent_group_cfg_py);
        _grid->add_object(agent);
        agent->agent_id = _agents.size();
        agent->stats.set_environment(this);
        add_agent(agent);
        _group_sizes[agent->group] += 1;
      }
      if (converter != nullptr) {
        _stats->incr("objects." + cell);
        _grid->add_object(converter);
        converter->set_event_manager(_event_manager.get());
        converter->stats.set_environment(this);
        converter = nullptr;
      }
    }
  }

  // Use wyhash for deterministic, high-performance grid fingerprinting across platforms
  initial_grid_hash = wyhash::hash_string(grid_hash_data);

  // Initialize buffers. The buffers are likely to be re-set by the user anyways,
  // so nothing above should depend on them before this point.
  std::vector<ssize_t> shape;
  shape = {static_cast<ssize_t>(num_agents), static_cast<ssize_t>(_num_observation_tokens), static_cast<ssize_t>(3)};
  auto observations = py::array_t<uint8_t, py::array::c_style>(shape);
  auto terminals = py::array_t<bool, py::array::c_style>({static_cast<ssize_t>(num_agents)}, {sizeof(bool)});
  auto truncations = py::array_t<bool, py::array::c_style>({static_cast<ssize_t>(num_agents)}, {sizeof(bool)});
  auto rewards = py::array_t<float, py::array::c_style>({static_cast<ssize_t>(num_agents)}, {sizeof(float)});

  set_buffers(observations, terminals, truncations, rewards);
}

MettaGrid::~MettaGrid() = default;

void MettaGrid::init_action_handlers() {
  _num_action_handlers = _action_handlers.size();
  _max_action_priority = 0;
  _max_action_arg = 0;
  _max_action_args.resize(_action_handlers.size());

  for (size_t i = 0; i < _action_handlers.size(); i++) {
    auto& handler = _action_handlers[i];
    handler->init(_grid.get());
    if (handler->priority > _max_action_priority) {
      _max_action_priority = handler->priority;
    }
    _max_action_args[i] = handler->max_arg();
    if (_max_action_args[i] > _max_action_arg) {
      _max_action_arg = _max_action_args[i];
    }
  }
}

void MettaGrid::add_agent(Agent* agent) {
  agent->init(&_rewards.mutable_unchecked<1>()(_agents.size()));
  _agents.push_back(agent);
}

void MettaGrid::_compute_observation(unsigned int observer_row,
                                     unsigned int observer_col,
                                     unsigned short obs_width,
                                     unsigned short obs_height,
                                     size_t agent_idx,
                                     ActionType action,
                                     ActionArg action_arg) {
  // Calculate observation boundaries
  unsigned int obs_width_radius = obs_width >> 1;
  unsigned int obs_height_radius = obs_height >> 1;

  unsigned int r_start = observer_row >= obs_height_radius ? observer_row - obs_height_radius : 0;
  unsigned int c_start = observer_col >= obs_width_radius ? observer_col - obs_width_radius : 0;

  unsigned int r_end = observer_row + obs_height_radius + 1;
  if (r_end > _grid->height) {
    r_end = _grid->height;
  }
  unsigned int c_end = observer_col + obs_width_radius + 1;
  if (c_end > _grid->width) {
    c_end = _grid->width;
  }

  // Fill in visible objects. Observations should have been cleared in _step, so
  // we don't need to do that here.
  size_t attempted_tokens_written = 0;
  size_t tokens_written = 0;
  auto observation_view = _observations.mutable_unchecked<3>();
  auto rewards_view = _rewards.unchecked<1>();

  // Global tokens
  ObservationToken* agent_obs_ptr = reinterpret_cast<ObservationToken*>(observation_view.mutable_data(agent_idx, 0, 0));
  ObservationTokens agent_obs_tokens(agent_obs_ptr, observation_view.shape(1) - tokens_written);
  unsigned int episode_completion_pct = 0;
  if (max_steps > 0) {
    episode_completion_pct =
        static_cast<unsigned int>(std::round((static_cast<double>(current_step) / max_steps) * 255.0));
  }
  int reward_int = static_cast<int>(std::round(rewards_view(agent_idx) * 100.0f));
  reward_int = std::clamp(reward_int, 0, 255);
  std::vector<PartialObservationToken> global_tokens = {
      {ObservationFeature::EpisodeCompletionPct, static_cast<uint8_t>(episode_completion_pct)},
      {ObservationFeature::LastAction, static_cast<uint8_t>(action)},
      {ObservationFeature::LastActionArg, static_cast<uint8_t>(action_arg)},
      {ObservationFeature::LastReward, static_cast<uint8_t>(reward_int)}};
  // Global tokens are always at the center of the observation.
  uint8_t global_location = obs_height_radius << 4 | obs_width_radius;
  attempted_tokens_written +=
      _obs_encoder->append_tokens_if_room_available(agent_obs_tokens, global_tokens, global_location);
  tokens_written = std::min(attempted_tokens_written, static_cast<size_t>(observation_view.shape(1)));

  // Order the tokens by distance from the agent, so if we need to drop tokens, we drop the farthest ones first.
  for (unsigned int distance = 0; distance <= obs_width_radius + obs_height_radius; distance++) {
    for (unsigned int r = r_start; r < r_end; r++) {
      // In this row, there should be one or two columns that have the correct [L1] distance.
      unsigned int r_dist = std::abs(static_cast<int>(r) - static_cast<int>(observer_row));
      if (r_dist > distance) continue;
      int c_dist = distance - r_dist;
      // This is a bit ugly. We want to run over {c_dist, -c_dist}, but only do it once if c_dist == 0.
      // Here's how we're trying to do that, and to be performant (e.g., not re-allocating a set).
      for (int i = 0; i < 2; i++) {
        if (c_dist == 0 && i == 1) continue;
        int c_offset = i == 0 ? c_dist : -c_dist;
        int c = observer_col + c_offset;
        // c could still be outside of our bounds.
        if (c < c_start || c >= c_end) continue;

        for (unsigned int layer = 0; layer < _grid->num_layers; layer++) {
          GridLocation object_loc(r, c, layer);
          auto obj = _grid->object_at(object_loc);
          if (!obj) continue;

          uint8_t* obs_data = observation_view.mutable_data(agent_idx, tokens_written, 0);
          ObservationToken* agent_obs_ptr = reinterpret_cast<ObservationToken*>(obs_data);
          ObservationTokens agent_obs_tokens(agent_obs_ptr, observation_view.shape(1) - tokens_written);

          int obs_r = object_loc.r + obs_height_radius - observer_row;
          int obs_c = object_loc.c + obs_width_radius - observer_col;
          uint8_t location = obs_r << 4 | obs_c;

          attempted_tokens_written += _obs_encoder->encode_tokens(obj, agent_obs_tokens, location);
          tokens_written = std::min(attempted_tokens_written, static_cast<size_t>(observation_view.shape(1)));
        }
      }
    }
  }
  _stats->add("tokens_written", static_cast<float>(tokens_written));
  _stats->add("tokens_dropped", static_cast<float>(attempted_tokens_written - tokens_written));
  _stats->add("tokens_free_space", static_cast<float>(observation_view.shape(1) - tokens_written));
}

void MettaGrid::_compute_observations(py::array_t<ActionType, py::array::c_style> actions) {
  auto actions_view = actions.unchecked<2>();
  auto observation_view = _observations.mutable_unchecked<3>();
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

void MettaGrid::_step(py::array_t<ActionType, py::array::c_style> actions) {
  auto actions_view = actions.unchecked<2>();

  // Reset rewards and observations
  auto rewards_view = _rewards.mutable_unchecked<1>();

  std::fill(
      static_cast<float*>(_rewards.request().ptr), static_cast<float*>(_rewards.request().ptr) + _rewards.size(), 0);

  auto obs_ptr = static_cast<uint8_t*>(_observations.request().ptr);
  auto obs_size = _observations.size();
  std::fill(obs_ptr, obs_ptr + obs_size, EmptyTokenByte);

  std::fill(_action_success.begin(), _action_success.end(), false);

  // Increment timestep and process events
  current_step++;
  _event_manager->process_events(current_step);

  // Process actions by priority levels (highest to lowest)
  for (unsigned char offset = 0; offset <= _max_action_priority; offset++) {
    unsigned char current_priority = _max_action_priority - offset;

    for (size_t agent_idx = 0; agent_idx < _agents.size(); agent_idx++) {
      ActionType action = actions_view(agent_idx, 0);
      ActionArg arg = actions_view(agent_idx, 1);

      // Tolerate invalid action types
      if (action < 0 || action >= _num_action_handlers) {
        _handle_invalid_action(agent_idx, "action.invalid_type", action, arg);
        continue;
      }

      auto& handler = _action_handlers[action];
      if (handler->priority != current_priority) {
        continue;
      }

      // Tolerate invalid action arguments
      if (arg > _max_action_args[action]) {
        _handle_invalid_action(agent_idx, "action.invalid_arg", action, arg);
        continue;
      }

      auto& agent = _agents[agent_idx];
      // handle_action expects a GridObjectId, rather than an agent_id, because of where it does its lookup
      // note that handle_action will assign a penalty for attempting invalid actions as a side effect
      _action_success[agent_idx] = handler->handle_action(agent->id, arg);
    }
  }

  // Compute observations for next step
  _compute_observations(actions);

  // Update episode rewards
  auto episode_rewards_view = _episode_rewards.mutable_unchecked<1>();
  for (py::ssize_t i = 0; i < rewards_view.shape(0); i++) {
    episode_rewards_view(i) += rewards_view(i);
  }

  // Check for truncation
  if (max_steps > 0 && current_step >= max_steps) {
    std::fill(static_cast<bool*>(_truncations.request().ptr),
              static_cast<bool*>(_truncations.request().ptr) + _truncations.size(),
              1);
  }
}

py::tuple MettaGrid::reset() {
  if (current_step > 0) {
    throw std::runtime_error("Cannot reset after stepping");
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
  auto zero_actions = py::array_t<int>(shape);
  _compute_observations(zero_actions);

  return py::make_tuple(_observations, py::dict());
}

void MettaGrid::validate_buffers() {
  // We should validate once buffers and agents are set.
  // data types and contiguity are handled by pybind11. We still need to check
  // shape.
  unsigned int num_agents = _agents.size();
  auto observation_info = _observations.request();
  auto shape = observation_info.shape;
  if (observation_info.ndim != 3) {
    std::stringstream ss;
    ss << "observations has " << observation_info.ndim << " dimensions but expected 3";
    throw std::runtime_error(ss.str());
  }
  if (shape[0] != num_agents || shape[2] != 3) {
    std::stringstream ss;
    ss << "observations has shape [" << shape[0] << ", " << shape[1] << ", " << shape[2] << "] but expected ["
       << num_agents << ", [something], 3]";
    throw std::runtime_error(ss.str());
  }
  {
    auto terminals_info = _terminals.request();
    auto shape = terminals_info.shape;
    if (terminals_info.ndim != 1 || shape[0] != num_agents) {
      throw std::runtime_error("terminals has the wrong shape");
    }
  }
  {
    auto truncations_info = _truncations.request();
    auto shape = truncations_info.shape;
    if (truncations_info.ndim != 1 || shape[0] != num_agents) {
      throw std::runtime_error("truncations has the wrong shape");
    }
  }
  {
    auto rewards_info = _rewards.request();
    auto shape = rewards_info.shape;
    if (rewards_info.ndim != 1 || shape[0] != num_agents) {
      throw std::runtime_error("rewards has the wrong shape");
    }
  }
}

void MettaGrid::set_buffers(const py::array_t<uint8_t, py::array::c_style>& observations,
                            const py::array_t<bool, py::array::c_style>& terminals,
                            const py::array_t<bool, py::array::c_style>& truncations,
                            const py::array_t<float, py::array::c_style>& rewards) {
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

py::tuple MettaGrid::step(py::array_t<ActionType, py::array::c_style> actions) {
  _step(actions);

  auto rewards_view = _rewards.mutable_unchecked<1>();
  // Clear group rewards

  // Handle group rewards
  bool share_rewards = false;
  // TODO: We're creating this vector every time we step, even though reward
  // should be sparse, and so we're unlikely to use it. We could decide to only
  // create it if we need it, but that would increase complexity.
  std::vector<double> group_rewards(_group_sizes.size());
  for (size_t agent_idx = 0; agent_idx < _agents.size(); agent_idx++) {
    if (rewards_view(agent_idx) != 0) {
      share_rewards = true;
      auto& agent = _agents[agent_idx];
      unsigned int group_id = agent->group;
      float group_reward = rewards_view(agent_idx) * _group_reward_pct[group_id];
      rewards_view(agent_idx) -= group_reward;
      group_rewards[group_id] += group_reward / _group_sizes[group_id];
    }
  }

  if (share_rewards) {
    for (size_t agent_idx = 0; agent_idx < _agents.size(); agent_idx++) {
      auto& agent = _agents[agent_idx];
      unsigned int group_id = agent->group;
      float group_reward = group_rewards[group_id];
      rewards_view(agent_idx) += group_reward;
    }
  }

  return py::make_tuple(_observations, _rewards, _terminals, _truncations, py::dict());
}

py::dict MettaGrid::grid_objects() {
  py::dict objects;

  for (unsigned int obj_id = 1; obj_id < _grid->objects.size(); obj_id++) {
    auto obj = _grid->object(obj_id);
    if (!obj) continue;

    py::dict obj_dict;
    obj_dict["id"] = obj_id;
    obj_dict["type"] = obj->_type_id;
    obj_dict["r"] = obj->location.r;
    obj_dict["c"] = obj->location.c;
    obj_dict["layer"] = obj->location.layer;

    auto features = obj->obs_features();
    for (const auto& feature : features) {
      obj_dict[py::str(_obs_encoder->feature_names().at(feature.feature_id))] = feature.value;
    }

    objects[py::int_(obj_id)] = obj_dict;
  }

  // Add agent IDs
  for (size_t agent_idx = 0; agent_idx < _agents.size(); agent_idx++) {
    auto agent_object = objects[py::int_(_agents[agent_idx]->id)];
    agent_object["agent_id"] = agent_idx;
  }

  return objects;
}

py::list MettaGrid::action_names() {
  py::list names;
  for (const auto& handler : _action_handlers) {
    names.append(handler->action_name());
  }
  return names;
}

unsigned int MettaGrid::map_width() {
  return _grid->width;
}

unsigned int MettaGrid::map_height() {
  return _grid->height;
}

// These should correspond to the features we emit in the observations -- either
// the channel or the feature_id.
py::dict MettaGrid::feature_normalizations() {
  return py::cast(_feature_normalizations);
}

unsigned int MettaGrid::num_agents() {
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
  //   "converter": list[dict[str, float]]  // Per-converter statistics
  // }
  // All stat values are guaranteed to be floats from StatsTracker::to_dict()

  py::dict stats;
  stats["game"] = py::cast(_stats->to_dict());

  py::list agent_stats;
  for (const auto& agent : _agents) {
    agent_stats.append(py::cast(agent->stats.to_dict()));
  }
  stats["agent"] = agent_stats;

  // Collect converter stats
  py::list converter_stats;
  for (unsigned int obj_id = 1; obj_id < _grid->objects.size(); obj_id++) {
    auto obj = _grid->object(obj_id);
    if (!obj) continue;

    // Check if this is a converter
    Converter* converter = dynamic_cast<Converter*>(obj);
    if (converter) {
      // Add metadata to the converter's stats tracker BEFORE converting to dict
      converter->stats.set("type_id", static_cast<int>(converter->_type_id));
      converter->stats.set("location.r", static_cast<int>(converter->location.r));
      converter->stats.set("location.c", static_cast<int>(converter->location.c));

      // Now convert to dict - all values will be floats
      py::dict converter_stat = py::cast(converter->stats.to_dict());
      converter_stats.append(converter_stat);
    }
  }
  stats["converter"] = converter_stats;
  return stats;
}

py::object MettaGrid::action_space() {
  auto gym = py::module_::import("gymnasium");
  auto spaces = gym.attr("spaces");

  size_t number_of_actions = py::len(action_names());
  size_t number_of_action_args = _max_action_arg + 1;
  return spaces.attr("MultiDiscrete")(py::make_tuple(number_of_actions, number_of_action_args),
                                      py::arg("dtype") = dtype_actions());
}

py::object MettaGrid::observation_space() {
  auto gym = py::module_::import("gymnasium");
  auto spaces = gym.attr("spaces");

  auto observation_info = _observations.request();
  auto shape = observation_info.shape;
  auto space_shape = py::tuple(observation_info.ndim - 1);
  for (size_t i = 0; i < observation_info.ndim - 1; i++) {
    space_shape[i] = shape[i + 1];
  }

  ObservationType min_value = std::numeric_limits<ObservationType>::min();  // 0
  ObservationType max_value = std::numeric_limits<ObservationType>::max();  // 255

  // TODO: consider spaces other than "Box". They're more correctly descriptive, but I don't know if
  // that matters to us.
  return spaces.attr("Box")(min_value, max_value, space_shape, py::arg("dtype") = dtype_observations());
}

py::list MettaGrid::action_success() {
  return py::cast(_action_success);
}

py::list MettaGrid::max_action_args() {
  return py::cast(_max_action_args);
}

py::list MettaGrid::object_type_names() {
  return py::cast(ObjectTypeNames);
}

py::list MettaGrid::inventory_item_names_py() {
  return py::cast(inventory_item_names);
}

Agent* MettaGrid::create_agent(int r, int c, const py::dict& agent_group_cfg_py) {
  unsigned char freeze_duration = agent_group_cfg_py["freeze_duration"].cast<unsigned char>();
  float action_failure_penalty = agent_group_cfg_py["action_failure_penalty"].cast<float>();
  std::map<InventoryItem, uint8_t> max_items_per_type =
      agent_group_cfg_py["max_items_per_type"].cast<std::map<InventoryItem, uint8_t>>();
  std::map<InventoryItem, float> resource_rewards =
      agent_group_cfg_py["resource_rewards"].cast<std::map<InventoryItem, float>>();
  std::map<InventoryItem, float> resource_reward_max =
      agent_group_cfg_py["resource_reward_max"].cast<std::map<InventoryItem, float>>();
  std::string group_name = agent_group_cfg_py["group_name"].cast<std::string>();
  unsigned int group_id = agent_group_cfg_py["group_id"].cast<unsigned int>();

  return new Agent(r,
                   c,
                   freeze_duration,
                   action_failure_penalty,
                   max_items_per_type,
                   resource_rewards,
                   resource_reward_max,
                   group_name,
                   group_id,
                   inventory_item_names);
}

py::array_t<unsigned int> MettaGrid::get_agent_groups() const {
  py::array_t<unsigned int> groups(_agents.size());
  auto groups_view = groups.mutable_unchecked<1>();
  for (size_t i = 0; i < _agents.size(); i++) {
    groups_view(i) = _agents[i]->group;
  }
  return groups;
}

// StatsTracker implementation that needs complete MettaGrid definition
unsigned int StatsTracker::get_current_step() const {
  if (!_env) return 0;
  return static_cast<MettaGrid*>(_env)->current_step;
}

ConverterConfig MettaGrid::_create_converter_config(const py::dict& converter_cfg_py) {
  std::map<InventoryItem, uint8_t> recipe_input =
      converter_cfg_py["recipe_input"].cast<std::map<InventoryItem, uint8_t>>();
  std::map<InventoryItem, uint8_t> recipe_output =
      converter_cfg_py["recipe_output"].cast<std::map<InventoryItem, uint8_t>>();
  short max_output = converter_cfg_py["max_output"].cast<short>();
  unsigned short conversion_ticks = converter_cfg_py["conversion_ticks"].cast<unsigned short>();
  unsigned short cooldown = converter_cfg_py["cooldown"].cast<unsigned short>();
  unsigned char initial_items = converter_cfg_py["initial_items"].cast<unsigned char>();
  ObsType color = converter_cfg_py["color"].cast<ObsType>();
  return ConverterConfig{
      recipe_input, recipe_output, max_output, conversion_ticks, cooldown, initial_items, color, inventory_item_names};
}

// Pybind11 module definition
PYBIND11_MODULE(mettagrid_c, m) {
  m.doc() = "MettaGrid environment";  // optional module docstring

  py::class_<MettaGrid>(m, "MettaGrid")
      .def(py::init<py::dict, py::list>())
      .def("reset", &MettaGrid::reset)
      .def("step", &MettaGrid::step, py::arg("actions").noconvert())
      .def("set_buffers",
           &MettaGrid::set_buffers,
           py::arg("observations").noconvert(),
           py::arg("terminals").noconvert(),
           py::arg("truncations").noconvert(),
           py::arg("rewards").noconvert())
      .def("grid_objects", &MettaGrid::grid_objects)
      .def("action_names", &MettaGrid::action_names)
      .def_property_readonly("map_width", &MettaGrid::map_width)
      .def_property_readonly("map_height", &MettaGrid::map_height)
      .def("feature_normalizations", &MettaGrid::feature_normalizations)
      .def_property_readonly("num_agents", &MettaGrid::num_agents)
      .def("get_episode_rewards", &MettaGrid::get_episode_rewards)
      .def("get_episode_stats", &MettaGrid::get_episode_stats)
      .def_property_readonly("action_space", &MettaGrid::action_space)
      .def_property_readonly("observation_space", &MettaGrid::observation_space)
      .def("action_success", &MettaGrid::action_success)
      .def("max_action_args", &MettaGrid::max_action_args)
      .def("object_type_names", &MettaGrid::object_type_names)
      .def_readonly("obs_width", &MettaGrid::obs_width)
      .def_readonly("obs_height", &MettaGrid::obs_height)
      .def_readonly("max_steps", &MettaGrid::max_steps)
      .def_readonly("current_step", &MettaGrid::current_step)
      .def("inventory_item_names", &MettaGrid::inventory_item_names_py)
      .def("get_agent_groups", &MettaGrid::get_agent_groups)
      .def_readonly("initial_grid_hash", &MettaGrid::initial_grid_hash);
}
