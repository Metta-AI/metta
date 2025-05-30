#include "mettagrid_c.hpp"

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

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
#include "objects/agent.hpp"
#include "objects/constants.hpp"
#include "objects/converter.hpp"
#include "objects/production_handler.hpp"
#include "objects/wall.hpp"
#include "observation_encoder.hpp"
#include "stats_tracker.hpp"
#include "types.hpp"

namespace py = pybind11;

MettaGrid::MettaGrid(py::dict env_cfg, py::list map) {
  // env_cfg is a dict-form of the OmegaCong config.
  // `map` is a list of lists of strings, which are the map cells.
  auto cfg = env_cfg["game"].cast<py::dict>();
  _cfg = cfg;

  int num_agents = cfg["num_agents"].cast<int>();
  max_steps = cfg["max_steps"].cast<unsigned int>();
  obs_width = cfg["obs_width"].cast<unsigned short>();
  obs_height = cfg["obs_height"].cast<unsigned short>();

  _use_observation_tokens = cfg.contains("use_observation_tokens") && cfg["use_observation_tokens"].cast<bool>();
  _num_observation_tokens =
      cfg.contains("num_observation_tokens") ? cfg["num_observation_tokens"].cast<unsigned int>() : 0;

  current_step = 0;

  // Initialize mode flags
  _gym_mode = false;
  _set_buffers_called = false;

  _observations = nullptr;
  _terminals = nullptr;
  _truncations = nullptr;
  _rewards = nullptr;

  _observations_size = 0;
  _terminals_size = 0;
  _truncations_size = 0;
  _rewards_size = 0;

  std::vector<Layer> layer_for_type_id;
  for (const auto& layer : ObjectLayers) {
    layer_for_type_id.push_back(layer.second);
  }
  int height = map.size();
  int width = map[0].cast<py::list>().size();

  _grid = std::make_unique<Grid>(width, height, layer_for_type_id);
  _obs_encoder = std::make_unique<ObservationEncoder>();
  _grid_features = _obs_encoder->feature_names();

  _event_manager = std::make_unique<EventManager>();
  _stats = std::make_unique<StatsTracker>();

  _event_manager->init(_grid.get(), _stats.get());
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
    _action_handlers.push_back(std::make_unique<Attack>(cfg["actions"]["attack"].cast<ActionConfig>()));
    _action_handlers.push_back(std::make_unique<AttackNearest>(cfg["actions"]["attack"].cast<ActionConfig>()));
  }
  if (cfg["actions"]["swap"]["enabled"].cast<bool>()) {
    _action_handlers.push_back(std::make_unique<Swap>(cfg["actions"]["swap"].cast<ActionConfig>()));
  }
  if (cfg["actions"]["change_color"]["enabled"].cast<bool>()) {
    _action_handlers.push_back(
        std::make_unique<ChangeColorAction>(cfg["actions"]["change_color"].cast<ActionConfig>()));
  }
  init_action_handlers();

  auto groups = cfg["groups"].cast<py::dict>();

  for (const auto& [key, value] : groups) {
    auto group = value.cast<py::dict>();
    unsigned int id = group["id"].cast<unsigned int>();
    _group_sizes[id] = 0;
    _group_reward_pct[id] = group.contains("group_reward_pct") ? group["group_reward_pct"].cast<float>() : 0.0f;
  }

  // Initialize objects from map
  for (int r = 0; r < height; r++) {
    for (int c = 0; c < width; c++) {
      std::string cell = map[r].cast<py::list>()[c].cast<std::string>();
      Converter* converter = nullptr;
      if (cell == "wall") {
        Wall* wall = new Wall(r, c, cfg["objects"]["wall"].cast<ObjectConfig>());
        _grid->add_object(wall);
        _stats->incr("objects.wall");
      } else if (cell == "block") {
        Wall* block = new Wall(r, c, cfg["objects"]["block"].cast<ObjectConfig>());
        _grid->add_object(block);
        _stats->incr("objects.block");
      } else if (cell.starts_with("mine")) {
        std::string m = cell;
        if (m.find('.') == std::string::npos) {
          m = "mine.red";
        }
        converter = new Converter(r, c, cfg["objects"][py::str(m)].cast<ObjectConfig>(), ObjectType::MineT);
      } else if (cell.starts_with("generator")) {
        std::string m = cell;
        if (m.find('.') == std::string::npos) {
          m = "generator.red";
        }
        converter = new Converter(r, c, cfg["objects"][py::str(m)].cast<ObjectConfig>(), ObjectType::GeneratorT);
      } else if (cell == "altar") {
        converter = new Converter(r, c, cfg["objects"]["altar"].cast<ObjectConfig>(), ObjectType::AltarT);
      } else if (cell == "armory") {
        converter = new Converter(r, c, cfg["objects"]["armory"].cast<ObjectConfig>(), ObjectType::ArmoryT);
      } else if (cell == "lasery") {
        converter = new Converter(r, c, cfg["objects"]["lasery"].cast<ObjectConfig>(), ObjectType::LaseryT);
      } else if (cell == "lab") {
        converter = new Converter(r, c, cfg["objects"]["lab"].cast<ObjectConfig>(), ObjectType::LabT);
      } else if (cell == "factory") {
        converter = new Converter(r, c, cfg["objects"]["factory"].cast<ObjectConfig>(), ObjectType::FactoryT);
      } else if (cell == "temple") {
        converter = new Converter(r, c, cfg["objects"]["temple"].cast<ObjectConfig>(), ObjectType::TempleT);
      } else if (cell.starts_with("agent.")) {
        std::string group_name = cell.substr(6);
        auto group_cfg_py = groups[py::str(group_name)]["props"].cast<py::dict>();
        auto agent_cfg_py = cfg["agent"].cast<py::dict>();
        unsigned int group_id = groups[py::str(group_name)]["id"].cast<unsigned int>();
        Agent* agent = MettaGrid::create_agent(r, c, group_name, group_id, group_cfg_py, agent_cfg_py);
        _grid->add_object(agent);
        agent->agent_id = _agents.size();
        add_agent(agent);
        _group_sizes[group_id] += 1;
      }
      if (converter != nullptr) {
        _stats->incr("objects." + cell);
        _grid->add_object(converter);
        converter->set_event_manager(_event_manager.get());
        converter = nullptr;
      }
    }
  }
}

MettaGrid::~MettaGrid() {
  _free_internal_buffers();
}

void MettaGrid::_setup_gym_mode() {
  if (_gym_mode) {
    return;  // Already allocated
  }

  size_t num_agents = _agents.size();
  if (_use_observation_tokens) {
    // Flattened: [num_agents, num_tokens * 3]
    _observations_size = num_agents * _num_observation_tokens * 3;
  } else {
    // Original: [num_agents, height, width, features]
    _observations_size = num_agents * obs_height * obs_width * _grid_features.size();
  }
  _terminals_size = num_agents;
  _truncations_size = num_agents;
  _rewards_size = num_agents;

  _internal_observations = std::make_unique<c_observations_type[]>(_observations_size);
  _internal_terminals = std::make_unique<c_terminals_type[]>(_terminals_size);
  _internal_truncations = std::make_unique<c_truncations_type[]>(_truncations_size);
  _internal_rewards = std::make_unique<c_rewards_type[]>(_rewards_size);

  _observations = _internal_observations.get();
  _terminals = _internal_terminals.get();
  _truncations = _internal_truncations.get();
  _rewards = _internal_rewards.get();

  // resize internal buffers
  _episode_rewards.resize(_agents.size(), 0.0f);

  // assign reward slots to agents
  for (size_t i = 0; i < _agents.size(); i++) {
    _agents[i]->init(&_rewards[i]);
  }

  _gym_mode = true;
}

void MettaGrid::_free_internal_buffers() {
  if (_gym_mode) {
    _internal_observations.reset();
    _internal_terminals.reset();
    _internal_truncations.reset();
    _internal_rewards.reset();

    _observations = nullptr;
    _terminals = nullptr;
    _truncations = nullptr;
    _rewards = nullptr;
  }
}

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
  _agents.push_back(agent);
  // Don't call agent->init() until after the reward buffer is available!
}

void MettaGrid::_compute_observation(unsigned int observer_row,
                                     unsigned int observer_col,
                                     unsigned short obs_width,
                                     unsigned short obs_height,
                                     size_t agent_idx) {
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

  if (_use_observation_tokens) {
    size_t tokens_written = 0;
    const size_t max_tokens = _num_observation_tokens;
    const size_t agent_offset = agent_idx * max_tokens * 3;

    // Clear agent's observation buffer
    std::memset(_observations + agent_offset, 0, max_tokens * 3 * sizeof(c_observations_type));

    // TODO: Order tokens by distance for better truncation behavior
    for (unsigned int r = r_start; r < r_end && tokens_written < max_tokens; r++) {
      for (unsigned int c = c_start; c < c_end && tokens_written < max_tokens; c++) {
        for (unsigned int layer = 0; layer < _grid->num_layers && tokens_written < max_tokens; layer++) {
          GridLocation object_loc(r, c, layer);
          auto obj = _grid->object_at(object_loc);
          if (!obj) continue;

          int obs_r = object_loc.r + obs_height_radius - observer_row;
          int obs_c = object_loc.c + obs_width_radius - observer_col;
          uint8_t location = (obs_r << 4) | obs_c;  // Pack location into single byte

          // Get object features
          vector<PartialObservationToken> features = obj->obs_features();

          // Write tokens directly to flattened buffer
          for (const auto& feature : features) {
            if (tokens_written >= max_tokens) break;

            size_t token_base = agent_offset + tokens_written * 3;
            _observations[token_base + 0] = location;
            _observations[token_base + 1] = feature.feature_id;
            _observations[token_base + 2] = feature.value;

            tokens_written++;
          }
        }
      }
    }
  } else {
    // Original grid-based observations
    size_t num_features = _grid_features.size();
    size_t agent_stride = obs_height * obs_width * num_features;
    size_t row_stride = obs_width * num_features;
    size_t col_stride = num_features;

    for (unsigned int r = r_start; r < r_end; r++) {
      for (unsigned int c = c_start; c < c_end; c++) {
        for (unsigned int layer = 0; layer < _grid->num_layers; layer++) {
          GridLocation object_loc(r, c, layer);
          auto obj = _grid->object_at(object_loc);
          if (!obj) continue;

          int obs_r = object_loc.r + obs_height_radius - observer_row;
          int obs_c = object_loc.c + obs_width_radius - observer_col;

          auto agent_obs = _observations + agent_idx * agent_stride + obs_r * row_stride + obs_c * col_stride;
          _obs_encoder->encode(obj, agent_obs);
        }
      }
    }
  }
}

void MettaGrid::_compute_observations(py::array_t<int> actions) {
  for (size_t idx = 0; idx < _agents.size(); idx++) {
    auto& agent = _agents[idx];
    _compute_observation(agent->location.r, agent->location.c, obs_width, obs_height, idx);
  }
}

void MettaGrid::_step(py::array_t<int> actions) {
  auto actions_view = actions.unchecked<2>();

  // Reset state for new step
  std::memset(_rewards, 0, _rewards_size * sizeof(c_rewards_type));
  std::memset(_observations, 0, _observations_size * sizeof(c_observations_type));
  std::fill(_action_success.begin(), _action_success.end(), false);

  // Increment timestep and process events
  current_step++;
  _event_manager->process_events(current_step);

  // Collect unique priority levels from action handlers
  std::set<unsigned char> priority_levels;
  for (const auto& handler : _action_handlers) {
    priority_levels.insert(handler->priority);
  }

  // Process actions by priority levels (highest to lowest)
  for (auto it = priority_levels.rbegin(); it != priority_levels.rend(); ++it) {
    unsigned char current_priority = *it;

    for (size_t agent_idx = 0; agent_idx < _agents.size(); agent_idx++) {
      // Skip agents who already successfully performed an action this step
      if (_action_success[agent_idx]) {
        continue;
      }

      // Extract action data
      int action_id = actions_view(agent_idx, 0);
      ActionArg action_arg = static_cast<ActionArg>(actions_view(agent_idx, 1));
      Agent* agent = _agents[agent_idx];

      // Validate action ID
      if (action_id < 0 || action_id >= _num_action_handlers) {
        throw std::runtime_error("Invalid action ID " + std::to_string(action_id) + " for agent " +
                                 std::to_string(agent_idx) + ". Valid range: 0 to " +
                                 std::to_string(_num_action_handlers - 1));
      }

      auto& handler = _action_handlers[action_id];

      // Skip if this handler doesn't match current priority level
      if (handler->priority != current_priority) {
        continue;
      }

      // Validate action argument
      if (action_arg > _max_action_args[action_id]) {
        throw std::runtime_error("Action argument " + std::to_string(action_arg) + " exceeds maximum " +
                                 std::to_string(_max_action_args[action_id]) + " for action " +
                                 std::to_string(action_id) + " (" + handler->action_name() + ")" + " on agent " +
                                 std::to_string(agent_idx));
      }

      // Validate agent
      if (agent == nullptr) {
        throw std::runtime_error("Agent is null at index " + std::to_string(agent_idx));
      }

      // Validate agent ID
      if (agent->id <= 0) {
        throw std::runtime_error("Agent ID must be positive. Agent " + std::to_string(agent_idx) + " has ID " +
                                 std::to_string(agent->id));
      }

      if (agent->id >= _grid->objects.size()) {
        throw std::runtime_error("Agent ID " + std::to_string(agent->id) + " exceeds grid object count " +
                                 std::to_string(_grid->objects.size()) + " for agent " + std::to_string(agent_idx));
      }

      // Execute the action
      bool success = handler->handle_action(agent->id, action_arg, current_step);
      _action_success[agent_idx] = success;
    }
  }

  // Update observations and episode state
  _compute_observations(actions);

  for (size_t i = 0; i < _agents.size(); i++) {
    _episode_rewards[i] += _rewards[i];
  }

  if (max_steps > 0 && current_step >= max_steps) {
    std::fill_n(_truncations, _truncations_size, 1);
  }
}

py::tuple MettaGrid::reset() {
  // If reset is called before set_buffers, enter gym mode
  if (!_set_buffers_called) {
    if (_observations != nullptr) {
      throw std::runtime_error("reset called before set_buffers implies gym mode but buffers are not null?");
    }
    _setup_gym_mode();
  }

  if (current_step > 0) {
    throw std::runtime_error("Cannot reset after stepping");
  }

  // reset external buffers using direct memory access
  std::memset(_terminals, 0, _terminals_size * sizeof(c_terminals_type));
  std::memset(_truncations, 0, _truncations_size * sizeof(c_truncations_type));
  std::memset(_rewards, 0, _rewards_size * sizeof(c_rewards_type));
  std::memset(_observations, 0, _observations_size * sizeof(c_observations_type));

  // reset internal buffers
  std::fill(_episode_rewards.begin(), _episode_rewards.end(), 0.0f);

  // Compute initial observations
  std::vector<ssize_t> shape = {static_cast<ssize_t>(_agents.size()), static_cast<ssize_t>(2)};
  auto zero_actions = py::array_t<int>(shape);
  _compute_observations(zero_actions);

  if (_use_observation_tokens) {
    std::vector<ssize_t> obs_shape = {
        static_cast<ssize_t>(_agents.size()), static_cast<ssize_t>(_num_observation_tokens), static_cast<ssize_t>(3)};
    py::array_t<c_observations_type> obs_view(obs_shape, _observations);
    return py::make_tuple(obs_view, py::dict());
  } else {
    std::vector<ssize_t> obs_shape = {static_cast<ssize_t>(_agents.size()),
                                      static_cast<ssize_t>(obs_height),
                                      static_cast<ssize_t>(obs_width),
                                      static_cast<ssize_t>(_grid_features.size())};
    py::array_t<c_observations_type> obs_view(obs_shape, _observations);
    return py::make_tuple(obs_view, py::dict());
  }
}

void MettaGrid::validate_buffers() {
  if (_observations == nullptr || _terminals == nullptr || _truncations == nullptr || _rewards == nullptr) {
    throw std::runtime_error("Buffers not set - call set_buffers first");
  }

  unsigned int num_agents = _agents.size();

  if (_use_observation_tokens) {
    size_t expected_obs_size = num_agents * _num_observation_tokens * 3;
    if (_observations_size != expected_obs_size) {
      std::stringstream ss;
      ss << "observations buffer size is " << _observations_size << " but expected " << expected_obs_size
         << " (agents=" << num_agents << ", tokens=" << _num_observation_tokens << ", token_size=3)";
      throw std::runtime_error(ss.str());
    }
  } else {
    size_t expected_obs_size = num_agents * obs_height * obs_width * _grid_features.size();
    if (_observations_size != expected_obs_size) {
      std::stringstream ss;
      ss << "observations buffer size is " << _observations_size << " but expected " << expected_obs_size
         << " (agents=" << num_agents << ", height=" << obs_height << ", width=" << obs_width
         << ", features=" << _grid_features.size() << ")";
      throw std::runtime_error(ss.str());
    }
  }

  if (_terminals_size != num_agents) {
    std::stringstream ss;
    ss << "terminals buffer size is " << _terminals_size << " but expected " << num_agents;
    throw std::runtime_error(ss.str());
  }

  if (_truncations_size != num_agents) {
    std::stringstream ss;
    ss << "truncations buffer size is " << _truncations_size << " but expected " << num_agents;
    throw std::runtime_error(ss.str());
  }

  if (_rewards_size != num_agents) {
    std::stringstream ss;
    ss << "rewards buffer size is " << _rewards_size << " but expected " << num_agents;
    throw std::runtime_error(ss.str());
  }
}

void MettaGrid::set_buffers(py::array_t<c_observations_type, py::array::c_style>& observations,
                            py::array_t<c_terminals_type, py::array::c_style>& terminals,
                            py::array_t<c_truncations_type, py::array::c_style>& truncations,
                            py::array_t<c_rewards_type, py::array::c_style>& rewards) {
  // If we're in gym mode, throw error
  if (_gym_mode) {
    throw std::runtime_error("Cannot call set_buffers when environment is in gym mode.");
  }

  _set_buffers_called = true;

  // Store raw pointers to Python-managed memory
  _observations = static_cast<c_observations_type*>(observations.mutable_data());
  _terminals = static_cast<c_terminals_type*>(terminals.mutable_data());
  _truncations = static_cast<c_truncations_type*>(truncations.mutable_data());
  _rewards = static_cast<c_rewards_type*>(rewards.mutable_data());

  // Store buffer sizes
  _observations_size = observations.size();
  _terminals_size = terminals.size();
  _truncations_size = truncations.size();
  _rewards_size = rewards.size();

  // resize internal buffers
  _episode_rewards.resize(_agents.size(), 0.0f);

  // Validate and initialize
  validate_buffers();

  // assign reward slots to agents
  for (size_t i = 0; i < _agents.size(); i++) {
    _agents[i]->init(&_rewards[i]);
  }
}

py::tuple MettaGrid::step(py::array_t<int> actions) {
  _step(actions);

  bool share_rewards = false;
  std::vector<double> group_rewards(_group_sizes.size(), 0.0);

  for (size_t agent_idx = 0; agent_idx < _agents.size(); agent_idx++) {
    if (_rewards[agent_idx] != 0) {
      share_rewards = true;
      auto& agent = _agents[agent_idx];
      unsigned int group_id = agent->group;
      float group_reward = _rewards[agent_idx] * _group_reward_pct[group_id];
      _rewards[agent_idx] -= group_reward;
      group_rewards[group_id] += group_reward / _group_sizes[group_id];
    }
  }

  if (share_rewards) {
    for (size_t agent_idx = 0; agent_idx < _agents.size(); agent_idx++) {
      auto& agent = _agents[agent_idx];
      unsigned int group_id = agent->group;
      _rewards[agent_idx] += group_rewards[group_id];
    }
  }

  if (_use_observation_tokens) {
    std::vector<ssize_t> obs_shape = {
        static_cast<ssize_t>(_agents.size()), static_cast<ssize_t>(_num_observation_tokens), static_cast<ssize_t>(3)};
    py::array_t<c_observations_type> obs_view(obs_shape, _observations);

    std::vector<ssize_t> rewards_shape = {static_cast<ssize_t>(_rewards_size)};
    std::vector<ssize_t> terminals_shape = {static_cast<ssize_t>(_terminals_size)};
    std::vector<ssize_t> truncations_shape = {static_cast<ssize_t>(_truncations_size)};

    py::array_t<c_rewards_type> rewards_view(rewards_shape, _rewards);
    py::array_t<c_terminals_type> terminals_view(terminals_shape, _terminals);
    py::array_t<c_truncations_type> truncations_view(truncations_shape, _truncations);

    return py::make_tuple(obs_view, rewards_view, terminals_view, truncations_view, py::dict());
  } else {
    // Original shape logic
    std::vector<ssize_t> obs_shape = {static_cast<ssize_t>(_agents.size()),
                                      static_cast<ssize_t>(obs_height),
                                      static_cast<ssize_t>(obs_width),
                                      static_cast<ssize_t>(_grid_features.size())};

    std::vector<ssize_t> rewards_shape = {static_cast<ssize_t>(_rewards_size)};
    std::vector<ssize_t> terminals_shape = {static_cast<ssize_t>(_terminals_size)};
    std::vector<ssize_t> truncations_shape = {static_cast<ssize_t>(_truncations_size)};

    py::array_t<c_observations_type> obs_view(obs_shape, _observations);
    py::array_t<c_rewards_type> rewards_view(rewards_shape, _rewards);
    py::array_t<c_terminals_type> terminals_view(terminals_shape, _terminals);
    py::array_t<c_truncations_type> truncations_view(truncations_shape, _truncations);

    return py::make_tuple(obs_view, rewards_view, terminals_view, truncations_view, py::dict());
  }
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

    // Get feature offsets for this object type
    auto type_features = _obs_encoder->type_feature_names()[obj->_type_id];
    std::vector<uint8_t> offsets(type_features.size());
    // We shouldn't have more than 256 features, since we're storing the feature_ids
    // as uint_8ts.
    assert(offsets.size() < 256);
    for (uint8_t i = 0; i < offsets.size(); i++) {
      offsets[i] = i;
    }
    std::vector<unsigned char> obj_data(type_features.size());

    // Encode object features
    _obs_encoder->encode(obj, obj_data.data(), offsets);

    // Add features to object dict
    for (size_t i = 0; i < type_features.size(); i++) {
      obj_dict[py::str(type_features[i])] = obj_data[i];
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

unsigned int MettaGrid::map_width() const {
  return _grid->width;
}

unsigned int MettaGrid::map_height() const {
  return _grid->height;
}

py::list MettaGrid::grid_features() {
  return py::cast(_grid_features);
}

unsigned int MettaGrid::num_agents() const {
  return _agents.size();
}

py::array_t<float> MettaGrid::get_episode_rewards() {
  std::vector<ssize_t> episode_rewards_shape = {static_cast<ssize_t>(_rewards_size)};
  return py::array_t<c_rewards_type>(episode_rewards_shape, _episode_rewards.data());
}

py::dict MettaGrid::get_episode_stats() {
  py::dict stats;
  stats["game"] = _stats->stats();

  py::list agent_stats;
  for (const auto& agent : _agents) {
    agent_stats.append(agent->stats.stats());
  }
  stats["agent"] = agent_stats;

  return stats;
}

py::object MettaGrid::action_space() {
  auto gym = py::module_::import("gymnasium");
  auto spaces = gym.attr("spaces");
  return spaces.attr("MultiDiscrete")(py::make_tuple(py::len(action_names()), _max_action_arg + 1),
                                      py::arg("dtype") = np_actions_dtype());
}

py::object MettaGrid::observation_space() {
  auto gym = py::module_::import("gymnasium");
  auto spaces = gym.attr("spaces");
  if (_use_observation_tokens) {
    // TODO: consider spaces other than "Box". They're more correctly descriptive, but I don't know if
    // that matters to us.
    return spaces.attr("Box")(
        0, 255, py::make_tuple(_agents.size(), _num_observation_tokens, 3), py::arg("dtype") = np_observations_dtype());
  } else {
    return spaces.attr("Box")(0,
                              255,
                              py::make_tuple(obs_height, obs_width, _grid_features.size()),
                              py::arg("dtype") = np_observations_dtype());
  }
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

py::list MettaGrid::inventory_item_names() {
  return py::cast(InventoryItemNames);
}

Agent* MettaGrid::create_agent(int r,
                               int c,
                               const std::string& group_name,
                               unsigned int group_id,
                               const py::dict& group_cfg_py,
                               const py::dict& agent_cfg_py) {
  // Rewards default to 0 for inventory unless overridden.
  // But we should be rewarding these all the time, e.g., for hearts.
  std::map<std::string, float> rewards;
  for (const auto& inv_item : InventoryItemNames) {
    // TODO: We shouldn't need to populate this with 0, since that's
    // the default anyways. Confirm that we don't care about the keys
    // and simplify.
    auto it = rewards.find(inv_item);
    if (it == rewards.end()) {
      rewards.insert(std::make_pair(inv_item, 0));
    }
    it = rewards.find(inv_item + "_max");
    if (it == rewards.end()) {
      rewards.insert(std::make_pair(inv_item + "_max", 1000));
    }
  }
  if (agent_cfg_py.contains("rewards")) {
    py::dict rewards_py = agent_cfg_py["rewards"];
    for (const auto& [key, value] : rewards_py) {
      rewards[key.cast<std::string>()] = value.cast<float>();
    }
  }
  if (group_cfg_py.contains("rewards")) {
    py::dict rewards_py = group_cfg_py["rewards"];
    for (const auto& [key, value] : rewards_py) {
      rewards[key.cast<std::string>()] = value.cast<float>();
    }
  }

  ObjectConfig agent_cfg;
  for (const auto& [key, value] : agent_cfg_py) {
    if (key.cast<std::string>() == "rewards") {
      continue;
    }
    agent_cfg[key.cast<std::string>()] = value.cast<int>();
  }
  for (const auto& [key, value] : group_cfg_py) {
    if (key.cast<std::string>() == "rewards") {
      continue;
    }
    agent_cfg[key.cast<std::string>()] = value.cast<int>();
  }

  return new Agent(r, c, group_name, group_id, agent_cfg, rewards);
}

py::array_t<unsigned int> MettaGrid::get_agent_groups() const {
  py::array_t<unsigned int> groups(_agents.size());
  auto groups_view = groups.mutable_unchecked<1>();
  for (size_t i = 0; i < _agents.size(); i++) {
    groups_view(i) = _agents[i]->group;
  }
  return groups;
}

// Pybind11 module definition
PYBIND11_MODULE(mettagrid_c, m) {
  m.doc() = "MettaGrid environment";  // optional module docstring

  py::class_<MettaGrid>(m, "MettaGrid")
      .def(py::init<py::dict, py::list>())
      .def("reset", &MettaGrid::reset)
      .def("step", &MettaGrid::step)
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
      .def("grid_features", &MettaGrid::grid_features)
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
      .def("is_gym_mode", &MettaGrid::is_gym_mode);
  .def("inventory_item_names", &MettaGrid::inventory_item_names).def("get_agent_groups", &MettaGrid::get_agent_groups);
}
